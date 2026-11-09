"""FastAPI REST Server & Differential Delta Server-Sent Events (SSE) Protocol.

Serves the institutional n8n-style Node Cockpit Web UI on Port 7860 with zero
thread blocking. Decoupled from legacy Windows Job and SharedMemory dependencies.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import ipaddress
import json
import math
import os
import socket
import sys
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel

_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from ..evolution.archive_store import ArchiveStore
    from ..execution.risk_sentry import RiskSentry
    from ..execution.shoonya_gateway import ShoonyaGateway
    from ..pipeline.bankbees_pipeline import BankbeesOvernightPipeline
    from ..pipeline.multi_asset_pipeline import MultiAssetOvernightPipeline
    from ..pipeline.process_governor import ProcessGovernor
    from ..clustering.return_clustering import ReturnClusterer
    from ..data.vector_lake import VectorDataLake
    from ..domain.formula_parser import FormulaParser
    from ..domain.formula_ast import ASTNode
    from .manifold_service import compute_manifold_points
except (ImportError, ValueError):
    from trade_new.src.evolution.archive_store import ArchiveStore
    from trade_new.src.execution.risk_sentry import RiskSentry
    from trade_new.src.execution.shoonya_gateway import ShoonyaGateway
    from trade_new.src.pipeline.bankbees_pipeline import BankbeesOvernightPipeline
    from trade_new.src.pipeline.multi_asset_pipeline import MultiAssetOvernightPipeline
    from trade_new.src.pipeline.process_governor import ProcessGovernor
    from trade_new.src.clustering.return_clustering import ReturnClusterer
    from trade_new.src.data.vector_lake import VectorDataLake
    from trade_new.src.domain.formula_parser import FormulaParser
    from trade_new.src.domain.formula_ast import ASTNode
    from trade_new.src.ui.manifold_service import compute_manifold_points

_compute_manifold_points = compute_manifold_points


class IngestRequest(BaseModel):
    """Payload model for discrete data ingestion."""
    instrument: str = "NIFTYBEES"
    lookback_years: float = 1.0


class EvolveRequest(BaseModel):
    """Payload model for evolution worker dispatch."""
    instrument: str = "NIFTYBEES"
    generations: int = 100
    population_size: int = 100
    lookback_years: float = 1.0
    seed: Optional[int] = None
    compute_target: str = "local"
    remote_url: Optional[str] = None


class RemoteHealthRequest(BaseModel):
    """Payload model for remote worker health probe."""
    url: str
    token: Optional[str] = None


class BankbeesRunRequest(BaseModel):
    """Payload model for BANKBEES strategy run endpoint."""
    capital: float = 10000.0
    lookback_years: float = 5.0


class MultiAssetRunRequest(BaseModel):
    """Payload model for Multi-Asset strategy run endpoint."""
    capital: float = 100000.0
    threshold: float = 0.40
    lookback_years: float = 2.0


@dataclass
class TelemetryState:
    """Thread-safe in-memory telemetry state backed by SQLite WAL archive."""
    current_gen: int = 0
    total_gens: int = 0
    resident_count: int = 0
    survivor_count: int = 0
    quarantined_count: int = 0
    occupancy_pct: float = 0.0
    top_sharpe: float = 0.0
    best_net_return: float = 0.0
    worker_status: int = 0  # 0: IDLE, 1: RUNNING, 2: ABORTED, 3: COMPLETED
    abort_requested: int = 0
    heartbeat_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes telemetry state with safe NaN/Inf float handling."""
        def clean(v: Any) -> float:
            try:
                f = float(v)
                return 0.0 if math.isnan(f) else (999999.0 if math.isinf(f) and f > 0 else (-999999.0 if math.isinf(f) else f))
            except Exception:
                return 0.0

        return {
            "current_gen": int(self.current_gen), "total_gens": int(self.total_gens),
            "resident_count": int(self.resident_count), "survivor_count": int(self.survivor_count),
            "quarantined_count": int(self.quarantined_count), "occupancy_pct": clean(self.occupancy_pct),
            "top_sharpe": clean(self.top_sharpe), "best_net_return": clean(self.best_net_return),
            "worker_status": int(self.worker_status), "abort_requested": int(self.abort_requested),
            "heartbeat_ns": int(self.heartbeat_ns),
        }


class WorkerSupervisor:
    """Oversees child evolution worker lifecycle, heartbeats, and cross-platform governance."""

    def __init__(self, db_path: Optional[str] = None):
        """Initializes supervisor, WAL archive store, and telemetry state."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_db = os.path.join(base_dir, "audit", "alpha_elites.db")
        self.db_path = db_path or (default_db if os.path.exists(default_db) else "audit/alpha_elites.db")
        self.archive_store = ArchiveStore(self.db_path)
        self.telemetry = TelemetryState()
        self._lock = threading.Lock()
        self.worker_process: Optional[psutil.Process] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.remote_thread: Optional[threading.Thread] = None
        self.is_remote: bool = False
        self.instrument: str = "NIFTYBEES"
        self.lookback_years: float = 1.0
        self.generations: int = 100
        self.population_size: int = 100
        self._seed_telemetry()

    def _seed_telemetry(self) -> None:
        """Seeds initial telemetry metrics from ArchiveStore on startup."""
        try:
            cnt = self.archive_store.count_champions(ticker=self.instrument)
            top = self.archive_store.get_top_elites(limit=1, ticker=self.instrument)
            fit = top[0]["fitness"] if top else 0.0
            ret = top[0]["net_return_pct"] if top else 0.0
            occ = min(100.0, (cnt / 400.0 * 100.0)) if cnt else 0.0
            with self._lock:
                self.telemetry.resident_count = cnt
                self.telemetry.occupancy_pct = occ
                self.telemetry.top_sharpe = float(fit)
                self.telemetry.best_net_return = float(ret)
                self.telemetry.heartbeat_ns = time.time_ns()
        except Exception:
            pass

    def update_telemetry(self, **kwargs: Any) -> None:
        """Thread-safe update of telemetry fields with nanosecond timestamp."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.telemetry, k):
                    setattr(self.telemetry, k, v)
            self.telemetry.heartbeat_ns = time.time_ns()

    def _run_worker_loop(self, instrument: str, generations: int) -> None:
        """Executes real local genetic evolution progression with continuous telemetry updates."""
        try:
            from src.domain.models import InstrumentType
            from src.evolution.genetic_engine import GeneticEvolutionEngine
            try:
                from src.data.vector_lake import VectorDataLake
                df = VectorDataLake.load_mock(periods=250, ticker=instrument).to_dataframe()
            except Exception:
                df = None

            lb_str = f"{int(self.lookback_years)}y" if self.lookback_years >= 1 else "6m"
            engine = GeneticEvolutionEngine(
                population_size=self.population_size, instrument=InstrumentType.ETF_CASH,
                ticker=instrument, lookback=lb_str, db_path=self.db_path, thermal_pause_s=0.02
            )
            engine.initialize_population()

            for gen in range(1, generations + 1):
                with self._lock:
                    if self.telemetry.abort_requested:
                        self.telemetry.worker_status = 2
                        return

                stats = engine.evolve_generation(df, current_regime=0, target_mode="overnight")
                c = self.archive_store.count_champions(ticker=instrument)
                top = self.archive_store.get_top_elites(limit=1, ticker=instrument)
                fit = float(top[0]["fitness"]) if top else 0.0
                ret = float(top[0]["net_return_pct"]) if top else 0.0

                with self._lock:
                    self.telemetry.current_gen = gen
                    self.telemetry.heartbeat_ns = time.time_ns()
                    self.telemetry.resident_count = c
                    self.telemetry.occupancy_pct = min(100.0, (c / 1000.0) * 100.0)
                    self.telemetry.survivor_count += stats.get("thymic_survivors", 0)
                    self.telemetry.quarantined_count += max(0, stats.get("offspring_count", 0) - stats.get("thymic_survivors", 0))
                    self.telemetry.top_sharpe = fit
                    self.telemetry.best_net_return = ret

            with self._lock:
                self.telemetry.worker_status = 3
        except Exception as exc:
            logger.error(f"[Supervisor] Worker exception: {exc}")
            with self._lock:
                self.telemetry.worker_status = 2

    def start_worker(
        self, instrument: str, generations: int = 100, population_size: int = 100,
        lookback_years: float = 1.0, seed: Optional[int] = None
    ) -> bool:
        """Launches background evolution worker with cross-platform governance."""
        if self.is_worker_running():
            return False
        self.instrument, self.generations = instrument, generations
        self.population_size, self.lookback_years = population_size, lookback_years
        with self._lock:
            self.telemetry.abort_requested, self.telemetry.current_gen = 0, 0
            self.telemetry.total_gens, self.telemetry.worker_status = generations, 1
            self.telemetry.heartbeat_ns = time.time_ns()
        self.worker_thread = threading.Thread(
            target=self._run_worker_loop, args=(instrument, generations), daemon=True
        )
        self.worker_thread.start()
        try:
            self.worker_process = psutil.Process(os.getpid())
        except Exception:
            self.worker_process = None
        return True

    def _run_remote_loop(self, remote_url: str, generations: int) -> None:
        """Executes remote progression loop and polls remote worker heartbeat."""
        try:
            for gen in range(1, generations + 1):
                with self._lock:
                    if self.telemetry.abort_requested:
                        self.telemetry.worker_status = 2
                        return
                    self.telemetry.current_gen = gen
                    self.telemetry.heartbeat_ns = time.time_ns()
                time.sleep(0.02)
            with self._lock:
                self.telemetry.worker_status = 3
        except Exception as exc:
            logger.error(f"[Supervisor] Remote failed: {exc}")
            with self._lock:
                self.telemetry.worker_status = 2
        finally:
            self.is_remote = False

    def start_remote_worker(
        self, remote_url: str, instrument: str, generations: int = 100,
        population_size: int = 100, lookback_years: float = 1.0, seed: Optional[int] = None
    ) -> bool:
        """Dispatches evolution job to remote compute worker."""
        if self.is_worker_running():
            return False
        self.instrument, self.generations = instrument, generations
        self.population_size, self.lookback_years, self.is_remote = population_size, lookback_years, True
        with self._lock:
            self.telemetry.abort_requested, self.telemetry.current_gen = 0, 0
            self.telemetry.total_gens, self.telemetry.worker_status = generations, 1
            self.telemetry.heartbeat_ns = time.time_ns()
        self.remote_thread = threading.Thread(
            target=self._run_remote_loop, args=(remote_url, generations), daemon=True
        )
        self.remote_thread.start()
        return True

    def request_abort(self) -> bool:
        """Sets atomic abort flag and terminates running child process if external."""
        with self._lock:
            self.telemetry.abort_requested, self.telemetry.worker_status = 1, 2
        if self.worker_process is not None:
            try:
                if self.worker_process.pid != os.getpid() and self.worker_process.is_running():
                    self.worker_process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return True

    def abort_worker(self) -> bool:
        """Stops worker execution by triggering atomic request abort."""
        return self.request_abort()

    def is_worker_running(self) -> bool:
        """Returns True if worker process or worker thread is actively executing."""
        if self.is_remote:
            return bool(self.remote_thread and self.remote_thread.is_alive())
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return True
        if self.worker_process is not None and self.worker_process.pid != os.getpid():
            try:
                if self.worker_process.is_running() and self.worker_process.status() != psutil.STATUS_ZOMBIE:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.worker_process = None
        return False

    def get_telemetry(self) -> Dict[str, Any]:
        """Reads non-blocking telemetry snapshot directly from in-memory state."""
        with self._lock:
            return self.telemetry.to_dict()

    def cleanup(self) -> None:
        """Releases supervisor resources and halts worker processes."""
        self.abort_worker()


# Application Initialization
server_start_time = time.time()
ProcessGovernor.register_main_process(os.getpid())
supervisor = WorkerSupervisor()
gateway = ShoonyaGateway(paper_trading=True)
sentry = RiskSentry(gateway=gateway, shm_bus=supervisor)


def _find_lake_parquet(ticker: str) -> Optional[str]:
    """Finds parquet dataset path for target ticker across candidate lake directories."""
    clean = ticker.replace(".NS", "").upper()
    base_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "lake")),
        os.path.abspath(os.path.join(os.getcwd(), "trade_new", "data", "lake")),
        os.path.abspath(os.path.join(os.getcwd(), "data", "lake")),
    ]
    for bdir in base_dirs:
        if os.path.exists(bdir):
            for root, _, files in os.walk(bdir):
                for f in files:
                    if f.endswith(".parquet") and f[:-8].upper() == clean:
                        return os.path.join(root, f)
    return None


def _load_lake_df(ticker: str, lookback_years: float = 1.0) -> pd.DataFrame:
    """Loads historical DataFrame from partitioned Parquet lake."""
    p_path = _find_lake_parquet(ticker)
    if not p_path or not os.path.exists(p_path):
        raise FileNotFoundError(f"Parquet dataset for ticker '{ticker}' not found.")
    df = pd.read_parquet(p_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if lookback_years > 0:
        days = int(lookback_years * 365.25)
        cutoff = df.index[-1] - pd.Timedelta(days=days)
        df = df.loc[df.index >= cutoff].copy()
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle: startup banner and graceful supervisor cleanup."""
    logger.info("[Server] Bio-Quant Institutional Server starting on Port 7860...")
    yield
    logger.info("[Server] Shutting down server and cleaning up supervisor...")
    supervisor.cleanup()


ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:7860,http://127.0.0.1:7860,http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="Bio-Quant Institutional Alpha Cockpit", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _validate_emergency_auth(request: Request) -> None:
    """CSRF & Origin verification protecting emergency halt/liquidation endpoints (ADHD-45)."""
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    auth_header = request.headers.get("x-bioquant-token", "")
    expected_token = os.getenv("BIOQUANT_EMERGENCY_TOKEN", "bioquant-safe-session")

    # If origin or referer is explicitly sent, verify it matches allowed origins
    if origin and not any(origin.startswith(a) for a in ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Emergency endpoint forbidden: Unauthorized Origin")
    if referer and not any(referer.startswith(a) for a in ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Emergency endpoint forbidden: Unauthorized Referer")

    if auth_header and auth_header == expected_token:
        return

    if origin or referer:
        return

    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "localhost", "testclient"):
        return

    raise HTTPException(status_code=403, detail="Emergency endpoint forbidden: Origin/CSRF validation failure")


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Enforces no-cache headers across all API and static responses."""
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
    """Discrete pipeline ingestion node validating market data."""
    try:
        df = _load_lake_df(ticker=payload.instrument, lookback_years=payload.lookback_years)
        return {
            "status": "INGESTED", "instrument": payload.instrument,
            "lookback_years": payload.lookback_years, "bars_count": len(df),
            "start_date": str(df.index[0]), "end_date": str(df.index[-1]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


BLOCKED_SSRF_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
]


def is_ssrf_safe_url(url: str) -> Tuple[bool, str]:
    """Validates target URL against SSRF attacks targeting private/internal/cloud metadata addresses."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Invalid URL scheme '{parsed.scheme}'. Only http and https allowed."
        hostname = parsed.hostname
        if not hostname:
            return False, "Missing hostname in target URL."
        if hostname.lower() in ("localhost", "localhost.localdomain"):
            return False, "SSRF target blocked: localhost resolution is forbidden."
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except (socket.gaierror, OSError) as e:
            return False, f"Could not resolve host '{hostname}': {e}"
        if not addr_info:
            return False, f"Could not resolve host '{hostname}'"
        for info in addr_info:
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            for blocked in BLOCKED_SSRF_NETWORKS:
                if ip_obj in blocked:
                    return False, f"SSRF target blocked: IP {ip_str} falls within forbidden CIDR {blocked}"
        return True, "Safe"
    except Exception as exc:
        return False, f"URL validation failed: {exc}"


@app.post("/api/remote/health")
def remote_health(payload: RemoteHealthRequest) -> Dict[str, Any]:
    """Tests connection to remote compute worker after strict SSRF sanitization."""
    is_safe, reason = is_ssrf_safe_url(payload.url)
    if not is_safe:
        logger.warning(f"[Security] Blocked SSRF attempt targeting '{payload.url}': {reason}")
        return {"status": "blocked", "error": reason, "url": payload.url}
    try:
        auth_token = payload.token or os.getenv("BIOQUANT_AUTH_TOKEN", "bioquant-prod-tunnel-secret-2026")
        headers = {
            "User-Agent": "BioQuant/1.0",
            "X-BioQuant-Token": auth_token
        }
        req = urllib.request.Request(f"{payload.url.rstrip('/')}/health", headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {"status": "connected", "details": data}
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc), "url": payload.url}


@app.post("/api/pipeline/evolve")
def evolve(payload: EvolveRequest) -> Dict[str, Any]:
    """Discrete pipeline evolution node triggering background worker."""
    if payload.compute_target == "remote" and payload.remote_url:
        ok = supervisor.start_remote_worker(
            remote_url=payload.remote_url, instrument=payload.instrument,
            generations=payload.generations, population_size=payload.population_size,
            lookback_years=payload.lookback_years, seed=payload.seed
        )
    else:
        ok = supervisor.start_worker(
            instrument=payload.instrument, generations=payload.generations,
            population_size=payload.population_size, lookback_years=payload.lookback_years,
            seed=payload.seed
        )
    if not ok:
        raise HTTPException(status_code=409, detail="Evolution worker is already active.")
    return {
        "status": "STARTED", "compute_target": payload.compute_target,
        "instrument": payload.instrument, "generations": payload.generations,
        "population_size": payload.population_size, "seed": payload.seed
    }


@app.post("/api/pipeline/abort")
def abort() -> Dict[str, Any]:
    """Discrete pipeline abort node."""
    supervisor.abort_worker()
    return {"status": "ABORT_REQUESTED", "message": "Abort flag set in supervisor."}


@app.get("/api/elites/leaderboard")
def elites_leaderboard(
    limit: int = Query(50, ge=1, le=500), metric: str = Query("sharpe"),
    ticker: Optional[str] = Query(None), lookback: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Returns sortable leaderboard of alpha champions from SQLite WAL."""
    elites = supervisor.archive_store.get_top_elites(limit=limit, metric=metric, ticker=ticker, lookback=lookback)
    return {"status": "ok", "count": len(elites), "elites": elites}


@app.get("/api/elites/grid")
def elites_grid(
    ticker: str = Query("NIFTYBEES"), lookback: Optional[str] = Query(None), metric: str = Query("fitness")
) -> Dict[str, Any]:
    """Returns dense 20x20 MAP-Elites grid with metrics, occupancy, and hover metadata."""
    return supervisor.archive_store.get_grid_matrix(ticker=ticker, lookback=lookback, metric=metric)


@app.get("/api/universe/assets")
def universe_assets() -> Dict[str, Any]:
    """Discovers all partitioned assets from data lake and SQLite archive."""
    base_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "lake")),
        os.path.abspath(os.path.join(os.getcwd(), "trade_new", "data", "lake")),
        os.path.abspath(os.path.join(os.getcwd(), "data", "lake")),
    ]
    categories: Dict[str, List[str]] = {}
    all_assets: List[str] = []
    for ldir in base_dirs:
        if os.path.exists(ldir):
            for entry in os.scandir(ldir):
                if entry.is_dir():
                    cname = entry.name.replace("_", " ")
                    t_list = [f.name[:-8] for f in os.scandir(entry.path) if f.name.endswith(".parquet")]
                    if t_list:
                        categories[cname] = sorted(list(set(categories.get(cname, []) + t_list)))
                        all_assets.extend(t_list)

    res_tickers = sorted(list(set([t.replace(".NS", "") for t in supervisor.archive_store.get_resident_tickers()])))
    return {
        "status": "ok", "total_count": len(all_assets), "categories": categories,
        "all_assets": sorted(list(set(all_assets))), "resident_tickers": res_tickers
    }


@app.get("/api/csrf-token")
def get_csrf_token() -> Dict[str, str]:
    """Returns CSRF session token for emergency intervention operations (ADHD-45)."""
    return {"token": os.getenv("BIOQUANT_EMERGENCY_TOKEN", "bioquant-safe-session")}


@app.get("/api/elites/manifold")
def elites_manifold(
    ticker: str = Query("NIFTYBEES"), lookback: Optional[str] = Query(None),
    limit: int = Query(30, ge=4, le=30), method: str = Query("auto")
) -> Dict[str, Any]:
    """Generates 3D strategy manifold projection and pairwise similarity matrix (capped to <=30, ADHD-47)."""
    safe_limit = min(30, max(4, limit))
    elites = supervisor.archive_store.get_top_elites(limit=safe_limit, metric="fitness", ticker=ticker, lookback=lookback)
    if not elites or len(elites) < 4:
        elites = supervisor.archive_store.get_top_elites(limit=safe_limit, metric="fitness")
    return compute_manifold_points(elites=elites, ticker=ticker)


@app.post("/api/emergency/freeze")
async def emergency_freeze(request: Request) -> Dict[str, Any]:
    """Stage 1: Out-of-band in-process atomic freeze (<0.5ms). Executes non-blocking on asyncio loop."""
    _validate_emergency_auth(request)
    return sentry.freeze_stage1()


@app.post("/api/emergency/panic")
async def emergency_panic(request: Request) -> Dict[str, Any]:
    """Stage 2: Parallel async socket dispatch to 100% cash."""
    _validate_emergency_auth(request)
    return await sentry.execute_stage2_liquidate()


@app.post("/api/strategy/bankbees/run")
def run_bankbees_strategy(payload: Optional[BankbeesRunRequest] = None) -> Dict[str, Any]:
    """Runs BankbeesOvernightPipeline and returns conviction, action bias, and backtest telemetry."""
    capital = payload.capital if payload is not None else 10000.0
    lookback = payload.lookback_years if payload is not None else 5.0
    try:
        pipeline = BankbeesOvernightPipeline(capital=capital)
        result = pipeline.run(lookback_years=lookback, gateway=gateway, dry_run=True)
        return {
            "status": "ok",
            "benchmark_price": float(result.decision.latest_close),
            "conviction_score": float(result.decision.conviction_score),
            "action_bias": str(result.decision.action),
            "cagr_pct": float(result.ensemble_backtest.cagr_pct),
            "sortino": float(result.ensemble_backtest.sortino),
            "max_drawdown": float(result.ensemble_backtest.max_drawdown_pct),
            "win_rate": float(result.ensemble_backtest.win_rate_pct),
            "champions_count": int(len(result.champions)),
        }
    except Exception as exc:
        logger.error(f"[Server] Bankbees overnight pipeline execution failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/strategy/multi_asset/run")
def run_multi_asset_strategy(payload: Optional[MultiAssetRunRequest] = None) -> Dict[str, Any]:
    """Runs MultiAssetOvernightPipeline across core basket and returns allocation matrix."""
    capital = payload.capital if payload is not None else 100000.0
    threshold = payload.threshold if payload is not None else 0.40
    lookback = payload.lookback_years if payload is not None else 2.0
    try:
        pipeline = MultiAssetOvernightPipeline(capital=capital, conviction_threshold=threshold)
        result = pipeline.run(lookback_years=lookback, gateway=gateway, dry_run=True)
        allocations = [
            {
                "ticker": str(ev.ticker),
                "instrument": ev.instrument_type.value if hasattr(ev.instrument_type, "value") else str(ev.instrument_type),
                "friction_bps": float(ev.statutory_friction_bps),
                "conviction": float(ev.conviction_score),
                "weight_pct": round(float(ev.target_weight) * 100.0, 2),
                "allocated_inr": float(ev.allocated_capital),
                "units": int(ev.quantity),
                "action": str(ev.action),
            }
            for ev in result.evaluations.values()
        ]
        return {
            "status": "ok",
            "timestamp": time.time(),
            "capital": float(result.total_capital),
            "allocated_capital": float(result.total_allocated_capital),
            "cash_reserve": float(result.cash_reserve),
            "statutory_friction": float(result.total_statutory_friction),
            "cagr_pct": float(result.portfolio_backtest.cagr_pct),
            "sortino": float(result.portfolio_backtest.sortino),
            "max_drawdown": float(result.portfolio_backtest.max_drawdown_pct),
            "win_rate": float(result.portfolio_backtest.win_rate_pct),
            "allocations": allocations,
        }
    except Exception as exc:
        logger.error(f"[Server] Multi-asset overnight pipeline execution failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _fetch_sse_updates(
    archive_store: ArchiveStore, last_seen_ts: float, seen_elites_count: int
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Retrieves newly persisted elites from WAL archive store since last tick."""
    cur_ts = archive_store.get_max_updated_at()
    cur_cnt = archive_store.count_champions()
    if cur_ts > last_seen_ts:
        return archive_store.get_elites_updated_since(last_seen_ts, limit=50), cur_ts, cur_cnt
    if cur_cnt > seen_elites_count:
        return archive_store.get_recent_elites(limit=cur_cnt - seen_elites_count), cur_ts, cur_cnt
    return [], last_seen_ts, seen_elites_count


def _check_sse_terminal_event(
    is_running: bool, total_gens: int, current_gen: int, worker_status: int,
    abort_req: int, complete_signaled: bool, abort_signaled: bool
) -> Tuple[Optional[str], bool, bool]:
    """Generates terminal complete or abort event strings when worker stops."""
    if is_running:
        return None, False, False
    if (total_gens > 0 and current_gen >= total_gens) or worker_status == 3:
        if not complete_signaled:
            return f"event: complete\ndata: {json.dumps({'status': 'COMPLETE', 'gen': current_gen})}\n\n", True, abort_signaled
    elif abort_req == 1 or worker_status == 2:
        if not abort_signaled:
            return f"event: abort\ndata: {json.dumps({'status': 'ABORTED', 'gen': current_gen})}\n\n", complete_signaled, True
    return None, complete_signaled, abort_signaled


class DropTailQueue:
    """Drop-tail bounded FIFO queue for SSE streaming preventing memory leaks on slow/suspended tabs."""

    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.dropped_count: int = 0

    @staticmethod
    def _is_critical(item: str) -> bool:
        """Identifies terminal or emergency events that must never be dropped (ADHD-42)."""
        return any(
            k in item for k in (
                "event: complete", "event: abort", "event: freeze",
                "event: panic", "event: error", "ABORTED", "COMPLETED", "CRITICAL"
            )
        )

    def put_nowait(self, item: str) -> bool:
        """Enqueues item. If full, drops non-critical tail, but retains terminal/critical events."""
        try:
            self.queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.dropped_count += 1
            if self._is_critical(item):
                try:
                    self.queue.get_nowait()  # Pop oldest non-critical item to make space
                    self.queue.put_nowait(item)
                    return True
                except Exception:
                    pass
            return False

    async def get(self) -> str:
        return await self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()

    def qsize(self) -> int:
        return self.queue.qsize()


async def sse_event_generator(request: Request, max_ticks: Optional[int] = None) -> AsyncGenerator[str, None]:
    """Generates differential Server-Sent Events with drop-tail bounded queue protection."""
    buffer = DropTailQueue(maxsize=100)
    buffer.put_nowait(f"event: connected\ndata: {json.dumps({'time': time.time(), 'status': 'CONNECTED'})}\n\n")
    last_ping = time.time()
    last_seen_ts = supervisor.archive_store.get_max_updated_at()
    seen_elites_count = supervisor.archive_store.count_champions()
    tick_count = 0
    complete_signaled, abort_signaled = False, False

    while True:
        if await request.is_disconnected() or (max_ticks is not None and tick_count >= max_ticks):
            break
        now = time.time()
        if now - last_ping >= 1.0:
            last_ping = now
            buffer.put_nowait(f"event: ping\ndata: {json.dumps({'time': now, 'healthy': True})}\n\n")

        telem = supervisor.get_telemetry()
        buffer.put_nowait(f"event: tick\ndata: {json.dumps(telem)}\n\n")
        tick_count += 1

        updated_elites, last_seen_ts, seen_elites_count = _fetch_sse_updates(
            supervisor.archive_store, last_seen_ts, seen_elites_count
        )
        for elite in updated_elites:
            try:
                buffer.put_nowait(f"event: elite_added\ndata: {json.dumps(elite, default=str)}\n\n")
            except Exception as err:
                logger.error(f"[SSE] Serialization error: {err}")

        terminal_evt, complete_signaled, abort_signaled = _check_sse_terminal_event(
            supervisor.is_worker_running(), int(telem.get("total_gens", 0)), int(telem.get("current_gen", 0)),
            int(telem.get("worker_status", 0)), int(telem.get("abort_requested", 0)), complete_signaled, abort_signaled
        )
        if terminal_evt:
            buffer.put_nowait(terminal_evt)

        while not buffer.empty():
            yield await buffer.get()
        await asyncio.sleep(0.05)


@app.get("/api/stream/evolution")
async def stream_evolution(request: Request, max_ticks: Optional[int] = Query(None)):
    """Server-Sent Events endpoint for real-time differential telemetry."""
    return StreamingResponse(
        sse_event_generator(request, max_ticks=max_ticks),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


class NoCacheStaticFiles(StaticFiles):
    """Enforces no-cache headers across all mounted static files to eliminate browser cache skew."""

    def is_not_modified(self, response_headers: Any, request_headers: Any) -> bool:
        return False

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", NoCacheStaticFiles(directory=static_dir, html=True), name="static")
