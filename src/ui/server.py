"""FastAPI REST Server & Differential Delta Server-Sent Events (SSE) Protocol.

Serves the institutional n8n-style Node Cockpit Web UI on Port 7860 with zero
thread blocking. Decoupled from heavy CPU evolution via SharedMemory IPC and Windows
Job Object worker isolation.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.evolution.archive_store import ArchiveStore
from src.execution.risk_sentry import RiskSentry
from src.execution.shoonya_gateway import ShoonyaGateway
from src.manifold.lake_dataset_loader import LakeDatasetLoader
from src.pipeline.evolution_worker import spawn_evolution_worker
from src.pipeline.shared_telemetry import SharedMemoryBus
from src.pipeline.windows_job import WindowsJobManager


class IngestRequest(BaseModel):
    instrument: str = "NIFTYBEES"
    lookback_years: float = 1.0


class EvolveRequest(BaseModel):
    instrument: str = "NIFTYBEES"
    generations: int = 10
    population_size: int = 50
    lookback_years: float = 1.0


class WorkerSupervisor:
    """Oversees child evolution worker lifecycle, heartbeats, and IPC telemetry."""

    def __init__(self, db_path: str = "audit/alpha_elites.db"):
        self.db_path = db_path
        self.job_manager = WindowsJobManager("BioQuantCockpitWorkerJob")
        self.shm_bus = SharedMemoryBus(create=True)
        self.archive_store = ArchiveStore(self.db_path)
        self.worker_process = None
        self.instrument: str = "NIFTYBEES"
        self.lookback_years: float = 1.0
        self.generations: int = 10
        self.population_size: int = 50

    def start_worker(
        self,
        instrument: str,
        generations: int = 10,
        population_size: int = 50,
        lookback_years: float = 1.0
    ) -> bool:
        """Launches a new child worker process under Windows Job Object guard."""
        if self.is_worker_running():
            logger.warning("[Supervisor] Worker is already running. Cannot start another.")
            return False

        self.instrument = instrument
        self.generations = generations
        self.population_size = population_size
        self.lookback_years = lookback_years

        self.worker_process = spawn_evolution_worker(
            instrument=instrument,
            lookback_years=lookback_years,
            generations=generations,
            population_size=population_size,
            archive_db_path=self.db_path,
            shm_name=self.shm_bus.shm_name
        )
        self.job_manager.assign_process(self.worker_process.pid)
        logger.info(f"[Supervisor] Spawned Worker PID {self.worker_process.pid} bound to Job Object.")
        return True

    def abort_worker(self) -> bool:
        """Sets atomic abort flag in shared memory for sub-millisecond worker termination."""
        if self.shm_bus:
            self.shm_bus.request_abort()
            logger.warning("[Supervisor] Atomic abort requested via SharedMemoryBus.")
            return True
        return False

    def is_worker_running(self) -> bool:
        """Returns True if worker is alive and actively emitting microsecond heartbeats."""
        if self.worker_process is None or not self.worker_process.is_alive():
            return False

        if self.shm_bus:
            last_hb = self.shm_bus.header.heartbeat_ns
            if last_hb > 0:
                age_s = (time.time_ns() - last_hb) / 1_000_000_000.0
                if age_s > 15.0:
                    logger.error(f"[Supervisor] Worker PID {self.worker_process.pid} hung (stale {age_s:.1f}s)!")
                    return False
        return True

    def get_telemetry(self) -> Dict[str, Any]:
        """Reads non-blocking telemetry snapshot directly from shared memory."""
        if not self.shm_bus:
            return {}
        hdr = self.shm_bus.header
        return {
            "current_gen": int(hdr.current_gen),
            "total_gens": int(hdr.total_gens),
            "resident_count": int(hdr.resident_count),
            "survivor_count": int(hdr.survivor_count),
            "quarantined_count": int(hdr.quarantined_count),
            "occupancy_pct": float(hdr.occupancy_pct),
            "top_sharpe": float(hdr.top_sharpe),
            "best_net_return": float(hdr.best_net_return),
            "worker_status": int(hdr.worker_status),
            "abort_requested": int(hdr.abort_requested),
            "heartbeat_ns": int(hdr.heartbeat_ns),
        }

    def cleanup(self) -> None:
        """Releases shared memory and closes Job Object."""
        if self.shm_bus:
            try:
                self.shm_bus.close()
            except Exception:
                pass
        self.job_manager.close()


# Application Initialization
server_start_time = time.time()
supervisor = WorkerSupervisor()
gateway = ShoonyaGateway(paper_trading=True)
sentry = RiskSentry(gateway=gateway, shm_bus=supervisor.shm_bus)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Server] Bio-Quant Institutional Server starting on Port 7860...")
    yield
    logger.info("[Server] Shutting down server and cleaning up supervisor...")
    supervisor.cleanup()


app = FastAPI(
    title="Bio-Quant Institutional Alpha Cockpit",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Lightweight server health probe."""
    return {
        "status": "ok",
        "uptime_s": round(time.time() - server_start_time, 2),
        "is_frozen": sentry.is_frozen,
        "worker_running": supervisor.is_worker_running()
    }


@app.get("/api/status")
def status() -> Dict[str, Any]:
    """System-wide telemetry and worker status query."""
    return {
        "status": "ok",
        "worker_running": supervisor.is_worker_running(),
        "telemetry": supervisor.get_telemetry(),
        "resident_elites_count": supervisor.archive_store.count_champions(),
        "sentry_frozen": sentry.is_frozen
    }


@app.post("/api/pipeline/ingest")
def ingest(payload: IngestRequest) -> Dict[str, Any]:
    """Discrete pipeline ingestion node."""
    loader = LakeDatasetLoader()
    try:
        df = loader.load_dataset(ticker=payload.instrument, lookback_years=payload.lookback_years)
        return {
            "status": "INGESTED",
            "instrument": payload.instrument,
            "lookback_years": payload.lookback_years,
            "bars_count": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pipeline/evolve")
def evolve(payload: EvolveRequest) -> Dict[str, Any]:
    """Discrete pipeline evolution node triggering background worker."""
    started = supervisor.start_worker(
        instrument=payload.instrument,
        generations=payload.generations,
        population_size=payload.population_size,
        lookback_years=payload.lookback_years
    )
    if not started:
        raise HTTPException(status_code=409, detail="Evolution worker is already active.")
    return {
        "status": "STARTED",
        "instrument": payload.instrument,
        "generations": payload.generations,
        "population_size": payload.population_size
    }


@app.post("/api/pipeline/abort")
def abort() -> Dict[str, Any]:
    """Discrete pipeline abort node."""
    supervisor.abort_worker()
    return {"status": "ABORT_REQUESTED", "message": "Abort flag set in shared memory."}


@app.get("/api/elites/leaderboard")
def elites_leaderboard(
    limit: int = Query(50, ge=1, le=500),
    metric: str = Query("sharpe"),
    ticker: Optional[str] = Query(None),
    lookback: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Returns sortable leaderboard of alpha champions from SQLite WAL."""
    elites = supervisor.archive_store.get_top_elites(
        limit=limit,
        metric=metric,
        ticker=ticker,
        lookback=lookback
    )
    return {"status": "ok", "count": len(elites), "elites": elites}


@app.post("/api/emergency/freeze")
def emergency_freeze() -> Dict[str, Any]:
    """Stage 1: Out-of-band in-process atomic freeze (<0.5ms)."""
    return sentry.freeze_stage1()


@app.post("/api/emergency/panic")
async def emergency_panic() -> Dict[str, Any]:
    """Stage 2: Parallel async socket dispatch to 100% cash."""
    return await sentry.execute_stage2_liquidate()


async def sse_event_generator(
    request: Request,
    max_ticks: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """Generates differential Server-Sent Events with 10Hz tick and 1Hz ping."""
    yield f"event: connected\ndata: {json.dumps({'time': time.time(), 'status': 'CONNECTED'})}\n\n"
    last_ping = time.time()
    seen_elites_count = supervisor.archive_store.count_champions()
    tick_count = 0

    while True:
        if await request.is_disconnected():
            break

        if max_ticks is not None and tick_count >= max_ticks:
            break

        now = time.time()
        # Ping keepalive every 1 second (satisfies 5s dead-man switch)
        if now - last_ping >= 1.0:
            last_ping = now
            yield f"event: ping\ndata: {json.dumps({'time': now, 'healthy': True})}\n\n"

        # Telemetry tick
        hdr = supervisor.shm_bus.header if supervisor.shm_bus else None
        if hdr:
            current_gen = int(hdr.current_gen)
            total_gens = int(hdr.total_gens)
            yield f"event: tick\ndata: {json.dumps(supervisor.get_telemetry())}\n\n"
            tick_count += 1

            # Check if new elites were saved
            current_elites_count = supervisor.archive_store.count_champions()
            if current_elites_count > seen_elites_count:
                delta = current_elites_count - seen_elites_count
                seen_elites_count = current_elites_count
                recent_elites = supervisor.archive_store.get_top_elites(limit=delta)
                for elite in recent_elites:
                    yield f"event: elite_added\ndata: {json.dumps(elite)}\n\n"

            # Check for completion or abort
            if (total_gens > 0 and current_gen >= total_gens) or hdr.abort_requested == 1:
                if not supervisor.is_worker_running():
                    yield f"event: complete\ndata: {json.dumps({'status': 'COMPLETE', 'gen': current_gen})}\n\n"
                    break

        await asyncio.sleep(0.05)  # 20Hz streaming tick rate


@app.get("/api/stream/evolution")
async def stream_evolution(
    request: Request,
    max_ticks: Optional[int] = Query(None)
):
    """Server-Sent Events endpoint for real-time differential telemetry."""
    return StreamingResponse(
        sse_event_generator(request, max_ticks=max_ticks),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Static Assets Mounting
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
