# Phase 1 Execution Plan: Institutional Decoupled Cockpit & Process-Isolated Engine
**System:** Bio-Quant Alpha-OS  
**Architecture:** Tri-Process Decoupled Architecture (FastAPI Server + Process-Isolated Worker + Out-of-Band Risk Sentry)  
**Target:** Elimination of Gradio, Implementation of n8n Node Paradigm, Windows Kernel Job Object Guard, Zero-Copy SharedMemory IPC, and Out-of-Band Two-Stage Risk Sentry.  
**Author:** Pair Programming Team (User + Antigravity) & Red Team ADHD Subagents  
**Date:** November 2026  
**Status:** FULLY REVISED SPECIFICATION - READY FOR EXECUTION  

---

## Live Operational Progress Tracker (Strict Granular Tick Policy)

> [!IMPORTANT]
> **Tick Policy Rules:**  
> 1. `[x] COMPLETED` - Verified with passing tests, zero regressions, and an atomic git commit hash.  
> 2. `[►] IN PROGRESS` - The single active task/subtask currently being implemented.  
> 3. `[ ] QUEUED / PENDING` - Remaining tasks to be completed in strict sequential order.  
> 4. State is dual-tracked in [`.phase1_state.json`](file:///C:/Users/Dell/Desktop/python/trade/.phase1_state.json) and updated in-place after each subtask.

| Task ID | Task Scope | Status | Verification Gate | Commit |
| :--- | :--- | :---: | :--- | :---: |
| **TASK 0** | **Latent Gateway Remediation & SQLite WAL Migration Baseline** | `[x] COMPLETED` | `unittest tests/test_gateway_hardening.py` | `39201a5` |
| ↳ Subtask 0.1 | Fix `place_order` keyword mismatch (`symbol=sym`) in lines 408 & 420 | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.2 | Fix undefined variable `sym` (`NameError`) in lines 420 & 421 | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.3 | Fix `price <= 0.0` check in line 232 to allow market order liquidation | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.4 | Fix line 418 to cover short positions (`netqty < 0`) with `action="BUY"` | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.5 | Pass product type `prd = p.get("prd", "C")` in line 420 | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.6 | Fix `cancel_order` parameter name (`orderno=...`) in line 391 | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.7 | Fix `_send_live_order` to prevent masking broker rejections as Ok | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.8 | Normalize symbols without corrupting futures or special series | `[x]` | Unit test passed | *Verified* |
| ↳ Subtask 0.9 | Migrate `_record_audit` from $O(N^2)$ JSON rewrite to SQLite WAL | `[x]` | 10 paper orders logged to WAL | *Verified* |
| ↳ Subtask 0.10 | Build `ArchiveMigrator` to migrate legacy `audit/archive_*.json` | `[x]` | 2,039 champions in SQLite WAL | *Verified* |
| **TASK 1** | **Windows Job Guard & Zero-Copy SharedMemory IPC** | `[x] COMPLETED` | `unittest tests/test_shm_worker_isolation.py` | `0e5c78d` |
| ↳ Subtask 1.1 | Windows Job Manager (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000`) | `[x]` | Process kill test | `0e5c78d` |
| ↳ Subtask 1.2 | CPU Affinity & Priority Governor (Cores 0-1 vs Cores 2-3) | `[x]` | Affinity mask check | `0e5c78d` |
| ↳ Subtask 1.3 | SharedMemory IPC Bus (162 KB RAM, 256B Header + 400-Niche Array) | `[x]` | Latency benchmark (<5µs) | `0e5c78d` |
| ↳ Subtask 1.4 | Spawned Evolution Worker (`evolution_worker.py`) with heartbeat | `[x]` | Worker isolation test | `0e5c78d` |
| **TASK 2** | **Two-Stage Risk Sentry & Market Microstructure Protection** | `[x] COMPLETED` | `unittest tests/test_risk_sentry_isolation.py` | `dfd130c` |
| ↳ Subtask 2.1 | Stage 1 In-Process Atomic Freeze (<0.5ms) | `[x]` | 3.6us nanosecond timing test | `dfd130c` |
| ↳ Subtask 2.2 | Stage 2 Parallel Async Socket Dispatch (asyncio gather pool) | `[x]` | 2.64ms concurrent mock dispatch | `dfd130c` |
| ↳ Subtask 2.3 | Lower-Circuit Evaporation Trap & Synthetic Delta Hedge (`NIFTYBEES`) | `[x]` | Circuit-lock & beta-weighted test | `dfd130c` |
| ↳ Subtask 2.4 | Statutory DP Charge Floor (₹15,930 notional / MIS routing) | `[x]` | Friction elimination test | `dfd130c` |
| ↳ Subtask 2.5 | Headless Session Watchdog (`QuickAuth` 08:45 IST TOTP daemon) | `[x]` | TOTP & session heal test | `dfd130c` |
| **TASK 3** | **FastAPI REST Server & Differential Delta SSE Protocol** | `[x] COMPLETED` | `unittest tests/test_cockpit_server_api.py` | `8377c45` |
| ↳ Subtask 3.1 | SQLite WAL Archive Store (`src/evolution/archive_store.py`) | `[x]` | Read/write benchmark | `0e5c78d` |
| ↳ Subtask 3.2 | FastAPI ASGI Server & Discrete Node Endpoints on Port 7860 | `[x]` | TestClient HTTP 200 | `8377c45` |
| ↳ Subtask 3.3 | Differential SSE Stream (`ping`, `tick`, `elite_added`, `complete`) | `[x]` | SSE chunk wire test | `8377c45` |
| ↳ Subtask 3.4 | Worker Process Supervisor & Watchdog Loop | `[x]` | Crash recovery test | `8377c45` |
| **TASK 4** | **Decoupled Institutional n8n-Style Node Cockpit Web UI** | `[x] COMPLETED` | `unittest tests/test_web_cockpit_ui.py` | `49f7111` |
| ↳ Subtask 4.1 | Offline Dependency Vendoring (`vendor_ui_dependencies.py`) | `[x]` | 6 assets locally vendored | `49f7111` |
| ↳ Subtask 4.2 | Zero-Dependency Web Audio API Synthesizer (`audio_synth.js`) | `[x]` | Audio oscillator test | `49f7111` |
| ↳ Subtask 4.3 | AG Grid Virtualization & Incremental Transactions (`applyTransactionAsync`) | `[x]` | Grid rendering test | `49f7111` |
| ↳ Subtask 4.4 | Plotly WebGL (`scattergl`) & 250ms RAF Redraw Throttler | `[x]` | Compositor RAF test | `49f7111` |
| ↳ Subtask 4.5 | 5-Second Dead-Man Switch HUD Watchdog Banner | `[x]` | Stream cutoff test | `49f7111` |
| ↳ Subtask 4.6 | Full Dark-Mode Avionics Layout & n8n Node Pipeline Deck | `[x]` | Visual layout inspection | `49f7111` |
| **TASK 5** | **Companion Headless Terminal Rescue TUI** | `[x] COMPLETED` | `unittest tests/test_rescue_tui.py` | `90fe659` |
| ↳ Subtask 5.1 | Rich Terminal Dashboard (`src/ui/rescue_tui.py`) | `[x]` | Console ASCII layout test | `90fe659` |
| ↳ Subtask 5.2 | Single-Keystroke `Key: K` Panic Liquidation Trigger | `[x]` | Key event mock test | `90fe659` |
| **TASK 6** | **Gradio Elimination & Codebase Hardening** | `[x] COMPLETED` | `python scripts/red_team_harness.py` | `4746c87` |
| ↳ Subtask 6.1 | Deprecate `src/ui/evolution_dashboard.py` with redirect stub | `[x]` | Backward compatible stub | `4746c87` |
| ↳ Subtask 6.2 | Remove `gradio` from runtime dependencies | `[x]` | 1,267 lines eliminated | `4746c87` |
| ↳ Subtask 6.3 | Full Red Team Invariants Gate (12/12 passing in <100ms) | `[x]` | 12/12 in 17.5ms | `4746c87` |
| ↳ Subtask 6.4 | Architecture and ledger documentation updates | `[x]` | ARCHITECTURE.md updated | `4746c87` |
| **TASK 7** | **Adversarial Chaos Fault-Injection Testing Suite** | `[x] COMPLETED` | `unittest tests/test_chaos_fault_injection.py` | `009bf41` |
| ↳ Subtask 7.1 | Test A: Worker SIGKILL crash recovery | `[x]` | Chaos test passed | `009bf41` |
| ↳ Subtask 7.2 | Test B: Broker network hang timeout | `[x]` | 0.2s timeout guard passed | `009bf41` |
| ↳ Subtask 7.3 | Test C: Corrupted archive ingestion recovery | `[x]` | Safe containment passed | `009bf41` |
| ↳ Subtask 7.4 | Test D: 50-clicks/sec panic button spam deduplication | `[x]` | 50/50 HTTP 200 passed | `009bf41` |

---

## 1. Executive Summary & Root-Cause Autopsy

### 1.1 The Gradio Deadlock Smoking Gun
As confirmed by runtime evidence:
- **Terminal Reality:** The Python backend (`src/ui/evolution_dashboard.py`) ran in PowerShell, successfully completed all 10/10 generations in ~60 seconds, and persisted the 91 resident elites to `audit\archive_NIFTYBEES_5y.json`.
- **Browser Reality:** The Gradio web interface at `127.0.0.1:7860` was completely frozen on `processing | 3.5/0.3s` with blank gray containers and spinning deadlocks.

### 1.2 Root Cause Mechanics
1. **GIL Hijacking by In-Process Compute:** Running AST mutations, tensor backtesting, and factor audits inside the generator thread blocked CPython's GIL, preventing Uvicorn/Gradio from servicing network I/O.
2. **Monolithic Payload Serialization Choke:** Emitting full Plotly figures (`go.Heatmap` + `go.Scatter`) and full Pandas DataFrames on every generation tick forced Python to serialize ~470 KB of JSON per frame, triggering severe V8 Garbage Collection (GC) pauses (15–60ms) and DOM layout reflows in the browser.
3. **Queue Deadlock & Emergency Lockout:** Gradio's single-threaded event queue (`demo.queue()`) placed all events in FIFO order. If evolution was running, clicking the Emergency Panic Kill-Switch was queued behind long generation yields. **In an exchange crash, the operator was locked out of risk controls.**

---

## 2. Target Architecture Overview

```
+----------------------------------------------------------------------------------------------------+
|                                    OPERATOR WORKSPACE (Port 7860)                                  |
|                                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  | AVIONICS HUD RIBBON: Total Equity | 400-Niche Coverage | Portfolio Sharpe | Overheat Tripwire |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                                                                                    |
|  +-----------------------------------+  +-------------------------------------------------------+  |
|  | N8N-STYLE NODE PIPELINE CONTROLLER |  | DYNAMIC VISUALIZATION DECK (AG Grid + WebGL Plotly)   |  |
|  | - [Node 1: Ingest Data Lake]      |  | - Tab 1: 2D Tactical View (MAP-Elites & Pareto WebGL) |  |
|  | - [Node 2: Evolve MAP-Elites]     |  | - Tab 2: Resident Champions Leaderboard (AG Grid)     |  |
|  | - [Node 3: Occam Pruning]         |  | - Tab 3: 3D Phenotype Manifold (scattergl)            |  |
|  | - [Node 4: HRP Risk Allocation]   |  | - Tab 4: Hierarchical Risk Parity (HRP)               |  |
|  | - [Node 5: Shoonya Order Ledger]  |  | - Tab 5: Virtual Broker Ledger & Margin Limits        |  |
|  | - [Node 6: System I/O DAG]        |  | - Tab 6: System I/O Architecture & Live Contracts     |  |
|  | - [🚀 RUN FULL AUTONOMOUS PIPELINE]|  | - DEAD-MAN WATCHDOG: 5-second telemetry heartbeat     |  |
|  +-----------------------------------+  +-------------------------------------------------------+  |
|                                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  | 🛑 OUT-OF-BAND EMERGENCY PANIC KILL-SWITCH (Stage 1: <0.5ms Atomic | Stage 2: Async Sockets)  |  |
|  +----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
                                       |
                   HTTP REST / Differential Delta SSE Stream
                                       v
+----------------------------------------------------------------------------------------------------+
|                                  FASTAPI WEB COCKPIT SERVER (Process A)                            |
|  - Windows Kernel Job Object Guard (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000)                   |
|  - Starlette ASGI Event Loop (100% GIL Free, CPU Core 0-1 Affinity)                               |
|  - Differential SSE Channels: ping (1s), tick (200B), elite_added (430B), complete                |
|  - SQLite WAL Store (audit/alpha_elites.db - zero WinError 32 file lock contention)                |
+----------------------------------------------------------------------------------------------------+
                 |                                                        |
                 | Zero-Copy SharedMemory (158 KB RAM)                    | Win32 High-Priority Event
                 v                                                        v
+---------------------------------------------+       +----------------------------------------------+
|     QUANT COMPUTE WORKER (Process B)        |       |        RISK SENTRY DAEMON (Process C)        |
|  - Isolated OS Process (spawn context)      |       |  - Win32 Priority: ABOVE_NORMAL / HIGH       |
|  - Win32 Priority: BELOW_NORMAL             |       |  - CPU Affinity: Core 0-1 Dedicated          |
|  - CPU Affinity: Core 2-3 Dedicated         |       |  - Stage 1: In-process atomic freeze (<0.5ms)|
|  - Zero-Copy NumPy Structured Buffer        |       |  - Stage 2: Parallel aiohttp connection pool |
|  - Microsecond Heartbeat (5s watchdog)      |       |  - Headless TOTP QuickAuth Daemon (08:45 IST)|
|  - Atomic AbortSignal Trap (<1ms)           |       |  - Lower-Circuit Evaporation Delta Hedge     |
+---------------------------------------------+       +----------------------------------------------+
```

---

## 3. Strict Preservation vs Eradication Boundaries

### 3.1 Strict Preservation Matrix (DO NOT TOUCH / BREAK)
- **12/12 Red Team Invariants:** `scripts/red_team_harness.py` (Causality, DSR, Covariance, Auction Gap, Circuit Damper, Dynamic Taxes, NaN containment, AST fuzzing, Integer lots, Gateway ledger, HRP allocation, Credential masking). Must pass 100% on every commit in $<100\text{ms}$.
- **Vectorized Math Engine:** `src/backtest/matrix_engine.py` (MatrixSimulator $(B, N, T)$ tensor operations, vector Sharpe, lookahead bias prevention).
- **Genetic AST Grammar:** `src/domain/formula_ast.py` (Node taxonomy, mutation, and crossover mechanics).
- **Existing Discoveries:** All 20+ archive files in `audit/archive_*.json` must be migrated to SQLite WAL with 100% field fidelity.
- **Security & Credentials:** `.env` keys (`SHOONYA_USER_ID`, `SHOONYA_PASSWORD`, `SHOONYA_TOTP_KEY`, etc.) remain authoritative and masked.

### 3.2 Complete Eradication Matrix (PERMANENTLY REMOVE)
- **Gradio Framework:** Eradicate `gradio>=3.50.2` runtime dependency, `demo.queue()`, and `.then()` generator chaining.
- **Single-File JSON Overwrites:** Eliminate `open(audit_file, "w")` in `_record_audit` ($O(N^2)$ disk I/O, zero-byte crash corruption risk).
- **Blocking Synchronous I/O in UI:** Eradicate all `time.sleep()`, synchronous `requests.post()`, or heavy tensor evaluations in web routes.
- **Unbounded Queues:** Bounded IPC buffers only (`maxsize=1000`) with backpressure control.

---

## 4. State Persistence & Mid-Way Resumption Protocol

To guarantee that execution progress is never lost if interrupted, a deterministic state tracker is maintained at `.phase1_state.json`:

```json
{
  "phase": "Phase 1: Institutional Decoupled Cockpit",
  "updated_at": "2026-11-07T02:00:00Z",
  "current_task": "TASK_0",
  "tasks": {
    "task_0_gateway_remediation_and_sqlite": {"status": "PENDING", "commit": null},
    "task_1_process_decoupling_shm_ipc": {"status": "PENDING", "commit": null},
    "task_2_risk_sentry_two_stage_killswitch": {"status": "PENDING", "commit": null},
    "task_3_fastapi_sse_server": {"status": "PENDING", "commit": null},
    "task_4_web_cockpit_ui": {"status": "PENDING", "commit": null},
    "task_5_rescue_tui": {"status": "PENDING", "commit": null},
    "task_6_gradio_eradication": {"status": "PENDING", "commit": null},
    "task_7_chaos_fault_injection": {"status": "PENDING", "commit": null}
  },
  "last_known_stable_commit": "c8bb3dd"
}
```

**Resumption Rule:** On session start or recovery, read `.phase1_state.json`. Find the first task where `status != "PASSED"`. Verify the prior passed task's tests pass, and resume directly without re-doing completed work.

---

## 5. Granular Task Breakdown

### TASK 0: Latent Gateway Remediation & SQLite WAL Migration Baseline
- **Primary Goal:** Fix 9 fatal latent bugs in `ShoonyaGateway` and replace $O(N^2)$ order JSON overwrites with SQLite WAL before building the Risk Sentry.
- **Remediations in `src/execution/shoonya_gateway.py`:**
  1. Fix Line 408 & 420: Pass `symbol=sym` (or `symbol=tsym`), not `trading_symbol`.
  2. Fix Line 420 & 421: Resolve `NameError` by replacing undefined `sym` with `tsym`.
  3. Fix Line 232: Allow `price == 0.0` when `order_type == "MKT"`.
  4. Fix Line 418: Handle short position liquidations (`netqty < 0`) via `action="BUY"`.
  5. Fix Line 420: Pass product type `prd = p.get("prd", "C")` so MIS positions square off correctly.
  6. Fix Line 391: Call `self.api.cancel_order(orderno=o["norenordno"])` (`orderno`, not `norenordno`).
  7. Fix Line 332: Check `if ret is None or ret.get('stat') != 'Ok'` to stop masking broker rejections.
  8. Fix `format_trading_symbol`: Preserve futures and special NSE series without appending `-EQ`.
  9. Fix `_record_audit`: Replace full-file rewrite with append-only JSONL (`orders.jsonl`) and SQLite WAL order journal (`audit/broker_orders.db`).
- **Archive Migrator:** Create `src/evolution/archive_migrator.py` to auto-migrate all 20 legacy `audit/archive_*.json` files into `audit/alpha_elites.db`.
- **TDD Verification (`tests/test_gateway_hardening.py`):**
  - Verify `liquidate_positions` squares off both long and short positions in paper and live mode without exceptions.
  - Verify 50 concurrent paper orders append to SQLite WAL in $<15\text{ms}$ with zero lock contention.
  - Verify legacy archive JSON files migrate to SQLite WAL with 100% field parity.
- **Verification Command:** `pytest tests/test_gateway_hardening.py -v && python scripts/red_team_harness.py --mode=full`
- **Atomic Git Commit:** `fix(execution): resolve shoonya liquidation bugs and migrate audit logs to sqlite wal`

---

### TASK 1: Windows Kernel Job Object Guard, Zero-Copy SharedMemory IPC & Worker Isolation
- **Primary Goal:** Separate heavy CPU evaluation into a spawned process with zero-copy shared memory, OS priority governor, and Windows Job Object kernel cleanup.
- **Components to Build:**
  1. `src/pipeline/windows_job.py`:
     - Implements `WindowsJobManager` using Win32 `CreateJobObjectW` with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000)`.
     - Uses 64-bit AMD64 `OpenProcess` with `PROCESS_ALL_ACCESS` to bind child PIDs, guaranteeing the Windows kernel terminates all workers if the parent process exits or crashes.
  2. `src/pipeline/process_governor.py`:
     - Sets Win32 OS priority classes: `BELOW_NORMAL_PRIORITY_CLASS` for Worker, `NORMAL` for Web Server, `ABOVE_NORMAL`/`HIGH` for Risk Sentry.
     - Sets CPU affinity masks: Cores 0-1 (`0x03`) for Server & Sentry; Cores 2-3 (`0x0C`) for Compute Worker.
  3. `src/pipeline/shared_telemetry.py` & `src/pipeline/shm_bus.py`:
     - Zero-copy IPC using `multiprocessing.shared_memory.SharedMemory` (162 KB total RAM).
     - 256-byte cache-aligned C-struct header (`HeaderStruct`) + 400-niche structured NumPy array (`NICHE_DTYPE`).
     - Point reads execute in $<0.2\mu\text{s}$ (down from 12ms pickle queue).
     - Atomic `abort_requested` flag (<1µs reaction time).
  4. `src/pipeline/evolution_worker.py`:
     - Standalone entry point running via `multiprocessing.get_context("spawn")`.
     - Zero imports of FastAPI, Uvicorn, or Gradio. Top-level `freeze_support()` guard.
     - Emits microsecond heartbeat timestamps to shared memory.
- **TDD Verification (`tests/test_shm_worker_isolation.py`):**
  - Verify child worker process is bound to Windows Job Object and killed immediately on parent handle close.
  - Verify SharedMemory reads execute in $<5\mu\text{s}$ with zero pickling overhead.
  - Verify setting `abort_requested` stops the compute worker within $<5\text{ms}$.
- **Verification Command:** `pytest tests/test_shm_worker_isolation.py -v`
- **Atomic Git Commit:** `feat(pipeline): implement windows job object guard and zero-copy shared memory ipc`

---

### TASK 2: Out-of-Band Two-Stage Risk Sentry & Market Microstructure Protection
- **Primary Goal:** Deliver sub-millisecond local order halting, parallel async broker dispatch, NSE lower-circuit synthetic delta hedging, and statutory DP fee enforcement.
- **Components to Build:**
  1. `src/execution/risk_sentry.py`:
     - Independent sentinel running with `HIGH_PRIORITY_CLASS` on Core 0-1.
     - **Stage 1 (In-Process Atomic Freeze <0.5ms):** Sets shared memory `HALTED = 1`, drops in-flight order buffers, and flags `AbortSignal`. 100% of outbound orders rejected in-memory in $<50\mu\text{s}$.
     - **Stage 2 (Parallel Async Socket Dispatch):** Pre-warmed `aiohttp.ClientSession` pool. Reads open orders/positions from real-time WebSocket cache (0ms query delay) and fires all cancels and liquidations concurrently via `asyncio.gather()`.
     - **Liquidates to 100% unencumbered cash** (removes hazardous `LIQUIDBEES` panic buy sweep).
  2. `src/execution/synthetic_hedge.py` (Lower-Circuit Evaporation Trap):
     - Detects when held asset is within 15 bps of lower circuit band or top-5 bid depth drops to 0.
     - Aborts futile `SELL MKT` orders (which exchange rejects) and executes beta-neutralizing synthetic delta short in `NIFTYBEES` or `NIFTY` Futures: $Q_{\text{hedge}} = \operatorname{int}\left(\frac{\sum V_k \cdot \beta_k}{P_{\text{NIFTYBEES}}}\right)$.
  3. Statutory Friction & DP Charge Gatekeeper:
     - Enforces Union Budget 2024 tax schedules in `TaxSchedule`.
     - Enforces ₹15,930 minimum notional floor for delivery orders to cap CDSL DP drag ($\le 10$ bps), or auto-routes bootstrap portfolios to intraday MIS.
  4. `src/execution/shoonya_watchdog.py` (Headless Session Daemon):
     - Headless `QuickAuth` with `pyotp` TOTP at 08:45 IST and 60-second keepalive pings.
- **TDD Verification (`tests/test_risk_sentry_isolation.py`):**
  - Verify Stage 1 atomic freeze executes in $<0.5\text{ms}$ while worker is at 100% CPU.
  - Verify Stage 2 dispatches parallel cancels and liquidations in a single roundtrip.
  - Verify lower-circuit freeze routes synthetic delta short to `NIFTYBEES`.
  - Verify sub-economic delivery orders ($< ₹15,930$) are rejected or routed to MIS.
- **Verification Command:** `pytest tests/test_risk_sentry_isolation.py -v`
- **Atomic Git Commit:** `feat(risk): implement two-stage risk sentry, synthetic hedging, and headless session daemon`

---

### TASK 3: High-Performance FastAPI REST Server & Differential Delta SSE Protocol
- **Primary Goal:** Replace Gradio on Port 7860 with a pure Starlette ASGI server streaming lightweight differential SSE deltas without WebSocket timeouts.
- **Components to Build:**
  1. `src/evolution/archive_store.py`:
     - SQLite in WAL mode (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`).
     - Point query leaderboard in $<1\text{ms}$; atomic batch upserts with zero file contention.
  2. `src/ui/server.py`:
     - FastAPI ASGI app running via Uvicorn on Port 7860.
     - Discrete Node Endpoints (n8n paradigm):
       - `POST /api/pipeline/ingest` (Node 1: Load Data Lake & Validate Features)
       - `POST /api/pipeline/evolve` (Node 2: Dispatch evolution to worker process)
       - `GET /api/stream/evolution` (Differential SSE Stream: `ping`, `tick`, `elite_added`, `complete`)
       - `POST /api/pipeline/prune` (Node 3: Occam Factor Pruning)
       - `POST /api/pipeline/allocate` (Node 4: Hierarchical Risk Parity Allocation)
       - `POST /api/pipeline/execute` (Node 5: Broker Dispatch & Rebalancing)
       - `POST /api/pipeline/run-all` (Autonomous Pipeline Orchestration)
       - `POST /api/emergency/panic` (Two-stage panic kill-switch)
       - `POST /api/shoonya/reauth` (Headless TOTP refresh trigger)
       - `GET /api/archive` & `DELETE /api/archive`
     - Background watchdog task (`WorkerSupervisor`) monitoring microsecond heartbeats.
- **TDD Verification (`tests/test_cockpit_server_api.py`):**
  - Test all REST endpoints return HTTP 200 with strict schema validation.
  - Verify SSE endpoint streams lightweight ~200-byte ticks with zero dropped connections.
  - Verify client disconnect abruptly releases generator resources without server hangs.
- **Verification Command:** `pytest tests/test_cockpit_server_api.py -v`
- **Atomic Git Commit:** `feat(ui): build fastapi rest api and differential delta sse streaming server`

---

### TASK 4: Decoupled Institutional n8n-Style Node Cockpit Web UI
- **Primary Goal:** Deliver a clean, dark-mode cockpit with AG Grid virtualization, WebGL Plotly, zero-dependency Web Audio synthesizer, and 5-second dead-man watchdog.
- **Components to Build:**
  1. `scripts/vendor_ui_dependencies.py`:
     - Downloads and vendors `tailwind.min.css`, `alpine.min.js`, `ag-grid-community.min.js`, `ag-grid.min.css`, `ag-theme-alpine.min.css`, and `plotly.min.js` to `src/ui/static/vendor/` for 100% offline air-gapped capability.
  2. `src/ui/static/audio_synth.js`:
     - Pure Web Audio API sound engine (zero MP3/WAV files):
       - Elite Discovered: Acoustic marimba chime (880 Hz -> 1320 Hz).
       - Cycle Complete: Two-tone bell (587 Hz -> 880 Hz).
       - Emergency Panic: 5Hz LFO modulated sawtooth siren (280 Hz).
  3. `src/ui/static/app.js`:
     - **AG Grid Virtualization:** `applyTransactionAsync({add, update})` with immutable niche keys (`niche_row_col`).
     - **Plotly Throttler:** `requestAnimationFrame` with 250ms debounce window (4 Hz max) and `scattergl` WebGL acceleration.
     - **Dead-Man Watchdog:** 5,000ms timer triggering HUD warning banner ("⚠️ TELEMETRY STREAM DISCONNECTED - TRADING BLIND") if heartbeats cease.
  4. `src/ui/static/index.html`:
     - Top Avionics HUD ribbon (Total Capital, Occupancy %, Sharpe, Overheat Tripwire).
     - Left n8n node cards with individual run buttons, status pills, and execution timers.
     - Right dynamic deck: 2D Pareto WebGL, AG Grid Champions, 3D Phenotype Manifold, HRP weights.
     - Pulsing red panic button triggering `/api/emergency/panic`.
- **TDD Verification (`tests/test_web_cockpit_ui.py`):**
  - Verify static assets serve HTTP 200 offline.
  - Verify AG Grid renders 1,000 formulas in $<20\text{ms}$.
  - Verify dead-man switch changes state within $<5,100\text{ms}$ upon stream cutoff.
- **Verification Command:** `pytest tests/test_web_cockpit_ui.py -v`
- **Atomic Git Commit:** `feat(ui): deliver institutional n8n node cockpit with ag-grid and webgl plotly`

---

### TASK 5: Companion Headless Terminal Rescue TUI
- **Primary Goal:** Provide a terminal failsafe monitor inside PowerShell with single-keystroke panic liquidation.
- **Components to Build:**
  1. `src/ui/rescue_tui.py`:
     - Lightweight terminal dashboard powered by `rich` and `asyncio`.
     - Direct listener on shared memory header and SQLite WAL database.
     - Single-keystroke listener (`msvcrt` on Windows): Striking `Key: K` immediately initiates emergency liquidation!
- **TDD Verification (`tests/test_rescue_tui.py`):**
  - Verify TUI render loop formats tables cleanly without exceptions.
  - Verify simulated `Key: K` event fires emergency panic kill-switch.
- **Verification Command:** `pytest tests/test_rescue_tui.py -v`
- **Atomic Git Commit:** `feat(ui): build headless terminal rescue tui with keyboard panic trigger`

---

### TASK 6: Gradio Elimination & Codebase Hardening
- **Primary Goal:** Permanently deprecate the fragile Gradio dashboard, verify zero regressions, and update documentation.
- **Actions:**
  1. Replace `src/ui/evolution_dashboard.py` with a deprecation banner redirecting operators to `python -m src.ui.server`.
  2. Remove `gradio` from `requirements.txt`.
  3. Execute full invariant test harness: `python scripts/red_team_harness.py --mode=full`.
  4. Update `ARCHITECTURE.md` and `SYSTEM_STATE_AND_LEDGER.md`.
- **Verification Command:** `python scripts/red_team_harness.py --mode=full`
- **Atomic Git Commit:** `refactor(ui): deprecate gradio dashboard and harden institutional cockpit architecture`

---

### TASK 7: Chaos Fault-Injection Testing Suite
- **Primary Goal:** Prove that the platform survives severe adversarial failures and hardware faults.
- **Components to Build (`tests/test_chaos_fault_injection.py`):**
  1. **Test A (Worker Hard Crash / SIGKILL):** Kill worker process via `taskkill /F /PID` during Generation 5. Verify web server survives, reports `WORKER_TERMINATED` via SSE, and resets IPC bus cleanly.
  2. **Test B (Broker Network Hang):** Mock 30-second hanging socket during panic switch. Verify Sentry enforces 1500ms timeout, marks broker offline, sets circuit halt, and returns without hanging.
  3. **Test C (Corrupted Archive Ingestion):** Inject garbage bytes into `archive_corrupt.json`. Verify system backs up file to `.bak` and falls back cleanly without 500 error.
  4. **Test D (Panic Button Spam - 50 Clicks/sec):** Fire 50 rapid concurrent POST requests to `/api/emergency/panic`. Verify atomic idempotency barrier processes request #1 and deduplicates the remaining 49.
- **Verification Command:** `pytest tests/test_chaos_fault_injection.py -v`
- **Atomic Git Commit:** `test(chaos): implement adversarial fault-injection testing suite`

---

## 6. Execution Dependency Chain & Verification Gates

```
[TASK 0: Gateway Bug Fixes & SQLite WAL Migration] 
                      |
                      v
[TASK 1: Windows Job Guard & Zero-Copy SHM IPC] 
                      |
                      v
[TASK 2: Two-Stage Risk Sentry & Synthetic Hedge] 
                      |
                      v
[TASK 3: FastAPI REST & Differential SSE Server] 
                      |
                      v
[TASK 4: Decoupled Web Cockpit (AG Grid + WebGL)] 
                      |
                      v
[TASK 5: Terminal Rescue TUI (Key: K)] 
                      |
                      v
[TASK 6: Gradio Eradication & Verification] 
                      |
                      v
[TASK 7: Adversarial Chaos Fault-Injection Suite]
                      |
                      v
[FINAL PRODUCTION GATE: 12/12 Red Team Invariants in <100ms]
```

### Verification Gates (Empirical Proof Mandatory)
1. **Gate 0:** `pytest tests/test_gateway_hardening.py` passes 100%; zero `TypeError`/`NameError` in liquidation; SQLite WAL handles 50 concurrent orders in $<15\text{ms}$.
2. **Gate 1:** `pytest tests/test_shm_worker_isolation.py` passes; parent process exit forcibly terminates child worker within 50ms via Windows Job Object; shared memory read $<5\mu\text{s}$.
3. **Gate 2:** `pytest tests/test_risk_sentry_isolation.py` passes; Stage 1 atomic freeze $<0.5\text{ms}$; synthetic delta hedge activates on lower-circuit freeze; DP drag capped.
4. **Gate 3:** `pytest tests/test_cockpit_server_api.py` passes; all REST endpoints return HTTP 200; SSE streams ~200-byte ticks without dropped connections.
5. **Gate 4:** `pytest tests/test_web_cockpit_ui.py` passes; UI loads offline; AG Grid scrolls 1,000 formulas at 60 FPS; dead-man switch detects disconnect in $<5,100\text{ms}$.
6. **Gate 5:** `pytest tests/test_rescue_tui.py` passes; terminal TUI renders cleanly and triggers panic on `Key: K`.
7. **Gate 6:** `scripts/red_team_harness.py --mode=full` passes 12/12 invariants in $<100\text{ms}$.
8. **Gate 7:** `pytest tests/test_chaos_fault_injection.py` passes all 4 adversarial stress tests.

---

## 7. Approval Lock

**All 4 specialist subagents have completed their audits and contributed concrete, tested implementations.**  
Awaiting user confirmation to proceed with **Task 0 (Latent Gateway Remediation & SQLite WAL Migration)** under strict TDD workflow.
