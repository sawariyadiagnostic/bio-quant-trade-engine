# Bio-Quant Autonomous Trading Architecture & System Directory
**NSE India Quantitative Trading Engine**

---

## 1. System Philosophy & Design Principles
The system operates as an institutional multi-tiered trading and research engine specialized for Indian Equities and Index ETFs (NSE):
1. **Separation of Manifold Regime vs Micro Strategy**:
   - **Phase A (Manifold)**: Continuous 18D embedding tensor projected via 3D UMAP into HDBSCAN density clustering. Unstructured Brownian chaos is isolated as Noise (`cluster_id = -1`), mandating **100% Cash / Liquid BeES capital preservation**.
   - **Phase B (Dynamic Alpha)**: AST formula evolution using 20x20 MAP-Elites behavioral grids to select non-correlated strategies tailored for distinct active regimes.
2. **Statutory Friction & Tax Arbitrage**:
   - Single-Stock Futures incur 0.02% STT (~3.0 bps friction) compared to 25.0 bps roundtrip for Cash Delivery.
   - Index ETFs (`NIFTYBEES`, `BANKBEES`) operate at ~1.2 bps statutory cost.
   - All engines leverage the unified `TaxSchedule` to dynamically configure or stress-test arbitrary friction/slippage levels.

---

## 2. Component Directory: Who Does What

| Directory / Module | Component Name | Exact Responsibility |
| :--- | :--- | :--- |
| [`src/domain/models.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/models.py) | **Domain Core** | Immutable entities: `Ticker`, `RegimeState`, `CircuitBands`, `InstrumentType`, `TaxSchedule`, and execution records. Pure Python with no heavy dependencies. |
| [`src/data/vector_lake.py`](file:///C:/Users/Dell/Desktop/python/trade/src/data/vector_lake.py) | **Vector Lake** | 5-Year Parquet data ingestion, multi-timeframe candle resampling, and calculation of the 18D continuous market embedding tensor. |
| [`src/manifold/`](file:///C:/Users/Dell/Desktop/python/trade/src/manifold) | **Topological Manifold** | 3D UMAP dimension reduction, anchored out-of-sample HDBSCAN regime classification, centroid drift stabilization, and Ward hierarchical phylogenetic market clustering. |
| [`src/evolution/map_elites.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/map_elites.py) | **MAP-Elites Archive** | 20x20 behavioral grid indexing strategies across Volatility vs Turnover. Applies statutory tax hurdles, Composite Institutional Alpha Score (CIAS), and temporal decay eviction. |
| [`src/evolution/genetic_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/genetic_engine.py) | **AST Genetic Engine** | Formula generator and mutator with speciation nurseries and memory-safe streaming micro-batch evaluation. |
| [`src/backtest/matrix_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/backtest/matrix_engine.py) | **MatrixSimulator** | Vectorized matrix backtester computing CAGR, Drawdown, Win Rate, and Sortino with semi-deviation floors across batch dimensions `[B, T]`. |
| [`src/backtest/factor_analytics.py`](file:///C:/Users/Dell/Desktop/python/trade/src/backtest/factor_analytics.py) | **Factor Analytics** | Information Coefficient (IC), rank IC, factor autocorrelation, and turnover decay metrics. |
| [`src/execution/risk_controller.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/risk_controller.py) | **Risk & Circuit Damper** | Statutory NSE price band detection (`CircuitBands`), ATR volatility sizing, and `MicroCapitalController` crash-regime lockout. |
| [`src/execution/hrp_allocator.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/hrp_allocator.py) | **HRP Allocator** | Hierarchical Risk Parity matrix allocation using correlation dendrograms and inverse-variance weighting. |
| [`src/execution/overnight_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/overnight_engine.py) | **Overnight Engine** | Tier 1 (ETF daily compound) & Tier 2 (ADR-gated single equity) execution with trailing micro-stops and auction gap resolution. |
| [`src/execution/portfolio_paper_trader.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/portfolio_paper_trader.py) | **Portfolio Paper Broker**| Multi-asset virtual broker handling order lots, balance tracking, two-pass rebalancing (sells before buys), and fee deductions. |
| [`src/execution/shoonya_gateway.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/shoonya_gateway.py) | **Shoonya Gateway** | Dual-mode broker interface: Local zero-risk simulation and Live Noren REST API execution with TOTP 2FA. |
| [`src/webui/studio_v2.py`](file:///C:/Users/Dell/Desktop/python/trade/src/webui/studio_v2.py) | **Control Tower UI** | Unified Gradio dashboard for interactive backtesting, MAP-Elites inspection, and live paper monitoring. |

---

## 3. Dynamic Statutory Friction Matrix

The system dynamically applies friction through [`TaxSchedule`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/models.py):

| Instrument Type | Identifier | Default Statutory Rate | Regulatory Cost Basis |
| :--- | :--- | :--- | :--- |
| **ETF Cash Delivery** | `ETF_CASH` | **1.2 bps (0.012%)** | 0.001% STT delivery + exchange charges |
| **Single-Stock Futures**| `FUTURES` | **3.0 bps (0.030%)** | 0.02% sell STT + turnover + GST |
| **Intraday Cash** | `INTRADAY_EQUITY` | **3.5 bps (0.035%)** | 0.025% sell STT + turnover + GST |
| **Cash Equity Delivery**| `EQUITY_CASH` | **25.0 bps (0.250%)** | 0.10% buy + 0.10% sell STT + statutory levies |

Custom slippage can be added directly via `TaxSchedule(custom_slippage_bps=X)`.

---

## 4. Institutional Decoupled Tri-Process Architecture

To eradicate GIL contention, browser freezes, and UI lockups, Phase 1 establishes an institutional Tri-Process Architecture:

```
+-----------------------------------------------------------------------------------------+
|                                    WINDOWS OS HOST                                      |
|                                                                                         |
|  [ PROCESS A: Web Cockpit ]          [ PROCESS B: Compute Worker ]                      |
|  * FastAPI ASGI (Port 7860)         * Spawned Python Process (multiprocessing spawn)     |
|  * Cores 0-1 (NORMAL Priority)      * Cores 2-3 (BELOW_NORMAL Priority)                 |
|  * Non-blocking Async Event Loop    * Bound to Windows Kernel Job Object                |
|  * Differential Delta SSE Stream     * Auto-terminates on parent close (0x2000)          |
|  * Offline Vendored UI (Alpine/AG)   * Emits microsecond heartbeats                      |
|              |                                     |                                    |
|              +============== SHARED MEMORY ========+                                    |
|              |      (162 KB Cache-Aligned C-Struct Header + 400-Niche Array)            |
|              |      Zero-copy point reads in <0.2us; atomic abort in <1us               |
|              |                                                                          |
|  [ PROCESS C: Risk Sentry ]                                                             |
|  * High-Priority Sentinel (Cores 0-1)                                                   |
|  * Stage 1: In-Process Atomic Freeze (<0.5ms) halting outbounds and signaling SHM abort  |
|  * Stage 2: Parallel Async Socket Dispatch via asyncio.gather to 100% unencumbered cash  |
|  * NSE Lower-Circuit Evaporation Trap: Beta-neutralizing synthetic short on NIFTYBEES    |
|  * Statutory Friction Gatekeeper: Enforces Rs 15,930 notional floor (auto-routes to MIS)|
|  * Headless QuickAuth Watchdog: Auto-refreshes Shoonya broker tokens via TOTP pyotp     |
|                                                                                         |
|  [ PERSISTENCE LAYER: SQLite WAL Engine ]                                               |
|  * Database: audit/alpha_elites.db                                                      |
|  * Mode: PRAGMA journal_mode=WAL, synchronous=NORMAL                                    |
|  * ACID single-transaction batch commits (<1ms) with zero WinError 32 file contention   |
+-----------------------------------------------------------------------------------------+
```

| Component | Responsibility | Tech Stack |
| :--- | :--- | :--- |
| **Process A: Web Cockpit** | Discrete n8n-style REST endpoints, 10Hz differential SSE, static asset serving | FastAPI, Uvicorn, AG Grid, Plotly WebGL, Alpine.js, Tailwind |
| **Process B: Compute Worker** | Heavy vectorized AST genetic mutation, backtesting, and niche tournament evaluation | Isolated spawned process, Win32 Job Object, SharedMemory IPC |
| **Process C: Risk Sentry** | Sub-0.5ms atomic freezing, parallel async socket liquidation, synthetic hedging | Asyncio, aiohttp/httpx, pyotp, ProcessGovernor |
| **Rescue TUI** | Headless companion text cockpit with single-keystroke `[K]` panic liquidation | Python `rich`, `msvcrt` non-blocking keyboard polling |
| **Archive Store** | Thread-safe, process-safe resident alpha champion leaderboard | SQLite WAL mode, single-batch ACID transactions |

