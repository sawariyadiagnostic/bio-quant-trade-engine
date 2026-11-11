# Bio-Quant Autonomous Trading System: Institutional Remediation & Upgrade Plan

> **Target:** `trade_new`  
> **Source Ledger:** `trade_new/RED_TEAM_MASTER_FINDINGS_LEDGER.md` (2,476 lines, 20 Vectors Audited)  
> **Standard:** Global Quantitative Hedge Fund & Institutional Production Standards (Zero Hollow Code, Strict Empirical Validation)

---

## 1. Master Architectural Upgrade Overview

```
+---------------------------------------------------------------------------------------------------+
|                         INSTITUTIONAL BIO-QUANT TARGET ARCHITECTURE                                |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  +-------------------------------------+             +-----------------------------------------+  |
|  |     Population Streamer (N=1000)    |             |       Multi-Dimensional CVT Archive     |  |
|  |  - Epigenetic Subtree Crossover     |             |  - K=1000 Voronoi Centroids (K-Means)   |  |
|  |  - Active Methylation Masking       |             |  - 4D Behavioral Manifold:              |  |
|  |  - Hardware Pacing & Thermal Guard  |             |    1. Annualized Volatility [0.05-0.60] |  |
|  +------------------+------------------+             |    2. Annualized Turnover [2.0-150.0]   |  |
|                     |                                |    3. Downside Beta [-0.5 - 1.5]        |  |
|                     v                                |    4. Autocorrelation lag-1 [-0.5, 0.5] |  |
|  +-------------------------------------+             +--------------------+--------------------+  |
|  |      PyTorch / CUDA 3D Tensor Pool  |                                  ^                       |
|  |  - Shape: [N_assets, T_bars, F_feat]|                                  |                       |
|  |  - Zero-Copy Win32 Shared Memory    |                                  |                       |
|  |  - >10,000 AST Evaluations/sec      |                                  |                       |
|  +------------------+------------------+                                  |                       |
|                     |                                                     |                       |
|                     v                                                     |                       |
|  +-------------------------------------+                                  |                       |
|  |   Purged Matrix Backtester & DSR    |                                  |                       |
|  |  - Bipartite Statutory DP Fee Model |                                  |                       |
|  |  - Deflated Sharpe Ratio (DSR)      |                                  |                       |
|  |  - Permutation Equivariant CS_RANK  |                                  |                       |
|  +------------------+------------------+                                  |                       |
|                     |                                                     |                       |
|                     +-----------------------------------------------------+                       |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Execution Waves & Subagent Orchestration

### Wave 1: Quant Mathematical Core, Statutory Engine & Microstructure (Vectors V01, V02, V03, V04, V12, V13)
- **Subagent A (Quant Math & Tensors):**
  - Patch `src/domain/tensor_evaluator.py`:
    - `ts_ema`: Safe initialization with forward-fill so initial/intermediate NaNs do not permanently collapse output to zero.
    - `cs_rank_tensor`: Replace non-deterministic `argsort(argsort)` with fractional tie-breaking; mask NaNs to prevent corrupt data promotion.
    - `cs_demean_tensor`: Use `nanmean` with finite masking to isolate missing assets.
    - `ts_delta`: Add `torch.nan_to_num(nan=0.0)`.
    - `div_safe_tensor` and unary operations: Finite bounds clamping ($\pm 10^4$).
  - Patch `src/domain/formula_ast.py` & `formula_parser.py`:
    - Fix unary negative parsing (`-TS_RANK_20(...)`).
    - Incorporate `active_regimes` into `canonical_hash()`.
    - Linear $O(N)$ parenthesis stripping.
- **Subagent B (Statutory Frictions, Lake & HRP Allocator):**
  - Patch `src/data/vector_lake.py`:
    - Corporate actions split adjustment: scale $o, h, l$ and inverse scale $v$ by $k_t = \text{Close\_Adj}_t / \text{Close}_t$.
    - Rolling volume z-score: clip rolling standard deviation $\ge 1.0$.
  - Patch `src/execution/overnight_engine.py`, `models.py`, `risk_sentry.py`:
    - Implement bipartite fee: $\text{Friction} = \text{Trade Value} \times \text{rate} + \text{₹}15.93 \text{ DP Charge}$.
    - Enforce 5-paise tick quantization for securities $\ge$ ₹250.
    - Eliminate CNC-to-MIS auto-reroute; enforce economic minimum notional floor (₹1,60,000) or route to Futures.
  - Patch `src/backtest/matrix_engine.py`:
    - Include exit bar closing transaction costs in net returns.
    - Correct Sortino annualization for sparse trading.
    - Implement Sharpe and Deflated Sharpe Ratio (DSR).
  - Patch `src/execution/hrp_allocator.py`:
    - Enforce volatility floor $\sigma_{\min} = 10^{-4}$ to prevent zero-variance capital hijacking.
    - Implement dendrogram tree traversal for recursive bisection with box bounds ($w \in [0.02, 0.30]$).
  - Patch `src/clustering/return_clustering.py`:
    - Isolate zero-variance strategies into an inert cluster and prune them immediately.
    - Add Ledoit-Wolf shrinkage for $T < N$.

---

### Wave 2: Evolutionary Engine, MAP-Elites & 3D Manifold (Vectors V06, V15, V16, V17)
- **Subagent C (Genetic Engine & MAP-Elites):**
  - Re-integrate and implement `src/evolution/genetic_engine.py` and `map_elites.py`:
    - 1000-generation streaming execution with $O(1)$ memory (250-individual micro-batches, thermal pacing).
    - Centroidal Voronoi Tessellation (CVT) multi-dimensional archive: 1000 Voronoi cells in 4D behavioral space with `cKDTree` fast lookup.
  - Patch `src/evolution/thymic_chamber.py`:
    - Extend synthetic COVID profile to 60+ bars to support rolling lookbacks.
    - Pre-compute all standard features on empirical slices so formulas don't evaluate to 0 and pass falsely.
    - Support continuous long/short/hedging drawdowns.
  - Patch `src/evolution/nursery.py` & `symbiosis.py`:
    - Scale speciation distance threshold relative to dimension $D$.
    - Align `SENSOR_FEATURES` and `ACTUATOR_FEATURES` with actual vector lake features.
  - Patch `src/ui/server.py`:
    - Connect `/api/pipeline/evolve` to the real `GeneticEngine`.
    - Replace `np.random.normal` in `_compute_manifold_points` with real AST formula evaluation on price data and 3D projection.

---

### Wave 3: SRE Daemon, Shoonya Headless QuickAuth & Concurrency (Vectors V07, V08, V10, V11, V14, V18)
- **Subagent D (SRE & Shoonya Daemon):**
  - Patch `scripts/overnight_daemon.py`:
    - Implement strict date latch (`entry_executed_date`, `exit_executed_date`) to prevent 20x duplicate buy blitzes.
    - Pass `gateway=gateway` to `pipeline.run()` in live mode.
    - Add NSE official holiday calendar check and enforce `ZoneInfo("Asia/Kolkata")`.
    - Add OS signal handlers (`SIGTERM`, `SIGINT`) and crash recovery loops.
  - Patch `src/execution/shoonya_gateway.py` & `shoonya_watchdog.py`:
    - Implement headless Noren REST `POST /QuickAuth` (SHA-256 user+api_key, password, `pyotp` TOTP) for automated 04:00 AM IST re-authentication.
    - Implement 20 req/s token bucket rate limiter.
    - Add persistent SQLite client order ID ledger.
    - Poll order status until confirmed fill (`COMPLETE`) rather than treating `/PlaceOrder` ack as fill.
  - Patch `src/ui/server.py` & `process_governor.py`:
    - Isolate worker processes using `multiprocessing.Process` / Win32 Job Objects so the main FastAPI web server is never throttled.
    - Change `emergency_freeze` to `async def` for non-blocking $<50\mu\text{s}$ execution.
    - Implement bounded SSE broadcast bus with drop-tail backpressure.
    - Sanitize `/api/remote/health` to block SSRF.

---

### Wave 4: International Standards, Abstract Broker Architecture & Clean Refactoring (Vectors V08, V19, V20)
- **Subagent E (Clean Code & International Expansion):**
  - Create `src/execution/base_gateway.py` defining `AbstractBrokerGateway`.
  - Implement multi-currency monetary ledger (`Currency.INR`, `Currency.USD`, `Currency.EUR`).
  - Patch cloud GPU worker notebooks (`kaggle_gpu_worker.ipynb`, `google_colab_gpu_worker.ipynb`):
    - Add API token authentication header.
    - Enable Dual-T4 multi-GPU distribution (`cuda:0` and `cuda:1`).
    - FP16 autocast and per-batch memory clearing.
  - Code refactoring:
    - Decompose functions $>50$ lines using guard clauses and sub-methods.
    - Decompose `server.py` into modular route files (`src/ui/routes/`).
    - Add strict type annotations.
    - Execute end-to-end integration and unit test suite.

---

## 3. Empirical Verification Gates
Every wave must pass strict automated validation before proceeding:
1. `pytest tests/` passing with 100% success rate.
2. Zero duplicate orders on 30-second loop simulation.
3. 100% DP fee accuracy tested against CDSL schedule.
4. Zero NaN leaks across 10,000 random AST formula evaluations.
5. Zero capital allocated to dead assets in HRP allocator.
