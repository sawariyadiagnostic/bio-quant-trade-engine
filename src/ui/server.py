"""FastAPI REST Server & Differential Delta Server-Sent Events (SSE) Protocol.

Serves the institutional n8n-style Node Cockpit Web UI on Port 7860 with zero
thread blocking. Decoupled from heavy CPU evolution via SharedMemory IPC and Windows
Job Object worker isolation.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import math
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
    seed: Optional[int] = None


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

        # Seed shared memory header from archive store on startup
        try:
            initial_count = self.archive_store.count_champions(ticker=self.instrument)
            top_elites = self.archive_store.get_top_elites(limit=1, ticker=self.instrument)
            top_fitness = top_elites[0]["fitness"] if top_elites else 0.0
            best_return = top_elites[0]["net_return_pct"] if top_elites else 0.0
            occ = min(100.0, (initial_count / 400.0 * 100.0)) if initial_count else 0.0
            self.shm_bus.update_header(
                current_gen=0,
                total_gens=self.generations,
                resident_count=initial_count,
                occupancy_pct=occ,
                top_sharpe=top_fitness,
                best_net_return=best_return,
                worker_status=0
            )
        except Exception:
            pass

    def start_worker(
        self,
        instrument: str,
        generations: int = 10,
        population_size: int = 50,
        lookback_years: float = 1.0,
        seed: Optional[int] = None
    ) -> bool:
        """Launches a new child worker process under Windows Job Object guard."""
        if self.is_worker_running():
            logger.warning("[Supervisor] Worker is already running. Cannot start another.")
            return False

        self.instrument = instrument
        self.generations = generations
        self.population_size = population_size
        self.lookback_years = lookback_years

        if self.shm_bus:
            self.shm_bus.header.abort_requested = 0
            self.shm_bus.update_header(
                current_gen=0,
                total_gens=generations,
                worker_status=1
            )

        self.worker_process = spawn_evolution_worker(
            instrument=instrument,
            lookback_years=lookback_years,
            generations=generations,
            population_size=population_size,
            archive_db_path=self.db_path,
            shm_name=self.shm_bus.shm_name,
            seed=seed
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

        def safe_float(v: Any, default: float = 0.0) -> float:
            try:
                f = float(v)
                if math.isinf(f):
                    return 999999.0 if f > 0 else -999999.0
                if math.isnan(f):
                    return default
                return f
            except Exception:
                return default

        return {
            "current_gen": int(hdr.current_gen),
            "total_gens": int(hdr.total_gens),
            "resident_count": int(hdr.resident_count),
            "survivor_count": int(hdr.survivor_count),
            "quarantined_count": int(hdr.quarantined_count),
            "occupancy_pct": safe_float(hdr.occupancy_pct),
            "top_sharpe": safe_float(hdr.top_sharpe),
            "best_net_return": safe_float(hdr.best_net_return),
            "worker_status": int(hdr.worker_status),
            "abort_requested": int(hdr.abort_requested),
            "heartbeat_ns": int(hdr.heartbeat_ns),
        }

    def cleanup(self) -> None:
        """Releases shared memory and closes Job Object."""
        if self.shm_bus:
            try:
                self.shm_bus.close()
                self.shm_bus.unlink()
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


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Enforces no-cache headers across all static and API responses to eliminate browser cache skew."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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
        lookback_years=payload.lookback_years,
        seed=payload.seed
    )
    if not started:
        raise HTTPException(status_code=409, detail="Evolution worker is already active.")
    return {
        "status": "STARTED",
        "instrument": payload.instrument,
        "generations": payload.generations,
        "population_size": payload.population_size,
        "seed": payload.seed
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


@app.get("/api/elites/grid")
def elites_grid(
    ticker: str = Query("NIFTYBEES"),
    lookback: Optional[str] = Query(None),
    metric: str = Query("fitness")
) -> Dict[str, Any]:
    """Returns dense 20x20 MAP-Elites grid with metrics, occupancy, and hover metadata."""
    return supervisor.archive_store.get_grid_matrix(ticker=ticker, lookback=lookback, metric=metric)


@app.get("/api/universe/assets")
def universe_assets() -> Dict[str, Any]:
    """Discovers all partitioned assets from data lake and SQLite archive."""
    lake_dir = os.path.join(os.getcwd(), "data", "lake")
    categories: Dict[str, List[str]] = {}
    all_assets: List[str] = []

    if os.path.exists(lake_dir):
        for entry in os.scandir(lake_dir):
            if entry.is_dir():
                cat_name = entry.name.replace("_", " ")
                tickers = []
                for f in os.scandir(entry.path):
                    if f.name.endswith(".parquet"):
                        t = f.name[:-8]
                        tickers.append(t)
                        all_assets.append(t)
                if tickers:
                    categories[cat_name] = sorted(tickers)

    resident_tickers = supervisor.archive_store.get_resident_tickers()
    clean_resident = sorted(list(set([t.replace(".NS", "") for t in resident_tickers])))

    return {
        "status": "ok",
        "total_count": len(all_assets),
        "categories": categories,
        "all_assets": sorted(list(set(all_assets))),
        "resident_tickers": clean_resident
    }


@app.get("/api/elites/manifold")
def elites_manifold(
    ticker: str = Query("NIFTYBEES"),
    lookback: Optional[str] = Query(None),
    limit: int = Query(50, ge=4, le=200),
    method: str = Query("auto")
) -> Dict[str, Any]:
    """Generates 3D strategy manifold projection and 1024D pairwise cosine similarity."""
    from src.domain.formula_parser import FormulaParser
    from src.clustering.phenotype_extractor import PhenotypeExtractor
    from src.clustering.alpha_manifold import AlphaManifoldPruner
    import numpy as np

    elites = supervisor.archive_store.get_top_elites(
        limit=limit,
        metric="fitness",
        ticker=ticker,
        lookback=lookback
    )
    if not elites or len(elites) < 4:
        elites = supervisor.archive_store.get_top_elites(limit=limit, metric="fitness")

    if not elites or len(elites) < 2:
        return {
            "status": "empty",
            "count": 0,
            "compression_ratio": 0.0,
            "clusters_count": 0,
            "points": [],
            "cosine_matrix": [],
            "labels": []
        }

    parsed_nodes = []
    valid_elites = []
    for e in elites:
        f_str = str(e.get("formula_str", ""))
        try:
            if f_str.startswith("ASTNode("):
                from src.domain.formula_ast import ASTNode
                node = eval(f_str, {"ASTNode": ASTNode})
            else:
                node = FormulaParser.parse(f_str)
            parsed_nodes.append(node)
            valid_elites.append(e)
        except Exception:
            try:
                from src.domain.formula_ast import ASTNode
                node = ASTNode(op="FEATURE", val="Close_Adj")
                parsed_nodes.append(node)
                valid_elites.append(e)
            except Exception:
                continue

    if len(parsed_nodes) < 2:
        return {
            "status": "empty",
            "count": 0,
            "compression_ratio": 0.0,
            "clusters_count": 0,
            "points": [],
            "cosine_matrix": [],
            "labels": []
        }

    # Load real multi-asset lakes from data/lake, fallback to mock if unpopulated
    try:
        from src.manifold.lake_dataset_loader import LakeDatasetLoader
        from src.data.vector_lake import VectorDataLake
        loader = LakeDatasetLoader()
        core_tickers = [ticker]
        for candidate in ["NIFTYBEES", "BANKBEES", "GOLDBEES"]:
            if candidate != ticker and len(core_tickers) < 3:
                core_tickers.append(candidate)
        lakes = {}
        for t in core_tickers:
            try:
                df_t = loader.load_dataset(t, lookback_years=1.0)
                if len(df_t) >= 55:
                    lakes[t] = VectorDataLake.from_dataframe(df_t, ticker=t)
            except Exception:
                pass
        extractor = PhenotypeExtractor(asset_lakes=lakes) if lakes else PhenotypeExtractor.create_mock(n_assets=3, n_bars=100)
    except Exception:
        extractor = PhenotypeExtractor.create_mock(n_assets=3, n_bars=100)
    embeddings = extractor.extract_dense_batch(parsed_nodes, target_dim=1024)
    cosine_matrix = np.clip(embeddings @ embeddings.T, 0.0, 1.0)

    ast_lens = [len(e["formula_str"]) for e in valid_elites]
    sortinos = [float(e.get("sortino", e.get("fitness", 1.0))) for e in valid_elites]

    pruner = AlphaManifoldPruner(min_cluster_size=max(2, min(4, len(valid_elites) // 3)))
    use_umap = (method == "umap") or (method == "auto" and len(valid_elites) <= 30)
    if method in {"fast", "svd"}:
        use_umap = False
    prune_res = pruner.cluster_and_prune(embeddings, ast_lens, sortinos, use_umap=use_umap)

    coords = prune_res["embedding_3d"]
    raw_labels = prune_res["cluster_labels"]
    labels = raw_labels.tolist() if hasattr(raw_labels, "tolist") else list(raw_labels)
    surv_set = set(prune_res["surviving_indices"])
    orth_set = set(prune_res["orthogonal_indices"])

    points = []
    for i, e in enumerate(valid_elites):
        points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "cluster": int(labels[i]),
            "is_survivor": i in surv_set,
            "is_orthogonal": i in orth_set,
            "ticker": e.get("ticker", ticker),
            "fitness": round(float(e.get("fitness", 0.0)), 2),
            "cagr_pct": round(float(e.get("cagr_pct", 0.0)), 1),
            "max_drawdown": round(float(e.get("max_drawdown", 0.0)), 1),
            "formula": e.get("formula_str", "")
        })

    return {
        "status": "ok",
        "ticker": ticker,
        "count": len(points),
        "compression_ratio": round(float(prune_res["compression_ratio"]), 4),
        "clusters_count": int(prune_res["clusters_count"]),
        "points": points,
        "cosine_matrix": [[round(float(v), 3) for v in row] for row in cosine_matrix],
        "labels": [f"#{i+1} {e['formula_str'][:16]}.." for i, e in enumerate(valid_elites)]
    }


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
    last_seen_ts = supervisor.archive_store.get_max_updated_at()
    seen_elites_count = supervisor.archive_store.count_champions()
    tick_count = 0

    hdr_init = supervisor.shm_bus.header if supervisor.shm_bus else None
    complete_signaled = bool(hdr_init and hdr_init.worker_status == 3)
    abort_signaled = bool(hdr_init and (hdr_init.worker_status == 2 or hdr_init.abort_requested == 1))

    while True:
        if await request.is_disconnected():
            break

        if max_ticks is not None and tick_count >= max_ticks:
            break

        now = time.time()
        # Ping keepalive every 1 second (satisfies 5s dead-man switch)
        if now - last_ping >= 1.0:
            last_ping = now
            yield f"event: ping\ndata: {json.dumps({'time': now, 'healthy': True}, default=str)}\n\n"

        # Telemetry tick
        hdr = supervisor.shm_bus.header if supervisor.shm_bus else None
        if hdr:
            current_gen = int(hdr.current_gen)
            total_gens = int(hdr.total_gens)
            yield f"event: tick\ndata: {json.dumps(supervisor.get_telemetry(), default=str)}\n\n"
            tick_count += 1

            # Check if new or replaced elites were saved (dual-trigger: timestamp and count)
            current_max_ts = supervisor.archive_store.get_max_updated_at()
            current_count = supervisor.archive_store.count_champions()
            updated_elites = []
            if current_max_ts > last_seen_ts:
                updated_elites = supervisor.archive_store.get_elites_updated_since(last_seen_ts, limit=50)
                last_seen_ts = current_max_ts
                seen_elites_count = current_count
            elif current_count > seen_elites_count:
                delta = current_count - seen_elites_count
                seen_elites_count = current_count
                updated_elites = supervisor.archive_store.get_recent_elites(limit=delta)

            for elite in updated_elites:
                try:
                    yield f"event: elite_added\ndata: {json.dumps(elite, default=str)}\n\n"
                except Exception as err:
                    logger.error(f"[SSE] Failed to serialize elite: {err}")

            # Check for completion or abort (signal once without severing connection)
            is_running = supervisor.is_worker_running()
            if is_running:
                complete_signaled = False
                abort_signaled = False
            else:
                if (total_gens > 0 and current_gen >= total_gens) or hdr.worker_status == 3:
                    if not complete_signaled:
                        complete_signaled = True
                        yield f"event: complete\ndata: {json.dumps({'status': 'COMPLETE', 'gen': current_gen}, default=str)}\n\n"
                elif hdr.abort_requested == 1 or hdr.worker_status == 2:
                    if not abort_signaled:
                        abort_signaled = True
                        yield f"event: abort\ndata: {json.dumps({'status': 'ABORTED', 'gen': current_gen}, default=str)}\n\n"

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
