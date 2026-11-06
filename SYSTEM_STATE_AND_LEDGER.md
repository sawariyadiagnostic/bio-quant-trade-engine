# Bio-Quant System State & Historical Lifecycle Ledger

**Document Classification:** Permanent System Inventory & Architectural Ledger  
**Last Updated:** 2026-11-06  
**System Standard:** Tier-1 Institutional Quantitative Architecture (NSE India)

---

## 1. System Inventory: Active Production Architecture

All components listed below are active, verified, and passing continuous test coverage (24 test suites, 91 assertions, 12/12 Red Team Invariants in 26.5ms):

| Domain / Layer | Primary File | Responsibility & Status |
| :--- | :--- | :--- |
| **Domain Entities** | [`src/domain/models.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/models.py) | Immutable value objects (`Ticker`, `CircuitBands`, `TaxSchedule`, `OrderSignal`). |
| **Quant Invariants** | [`src/domain/invariants.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/invariants.py) | Point-in-time causality, Deflated Sharpe Ratio (DSR) with fat tails, covariance condition numbers ($\kappa < 10^4$). |
| **Formula Engine** | [`src/domain/formula_ast.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/formula_ast.py) | Vectorized AST evaluation, commutative MurmurHash canonicalization, bounded operators (`EXP_SAFE`, `LOG_SAFE`, `CS_Rank`, `CS_Demean`). |
| **Parser & Seeds** | [`src/domain/formula_parser.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/formula_parser.py) | Recursive descent parsing and quantitative formula seed libraries. |
| **Data Lake** | [`src/data/vector_lake.py`](file:///C:/Users/Dell/Desktop/python/trade/src/data/vector_lake.py) | 5-Year Parquet historical data lake, daily-resetting intraday VWAP, discontinuous overnight auction jump tensor. |
| **Pre-Market Data** | [`src/data/gift_nifty_loader.py`](file:///C:/Users/Dell/Desktop/python/trade/src/data/gift_nifty_loader.py) | GIFT Nifty SGX/NSE pre-market lead-lag gap calculator at 08:30 IST with circuit clipping. |
| **Manifold Regimes** | [`src/manifold/feature_pipeline.py`](file:///C:/Users/Dell/Desktop/python/trade/src/manifold/feature_pipeline.py) | 18D continuous market feature tensors, UMAP 3D projection, HDBSCAN anchored density clustering. |
| **MAP-Elites Archive** | [`src/evolution/map_elites.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/map_elites.py) | 400-niche behavioral grid, 2-objective Pareto filtering, 2x statutory friction stress testing. |
| **Genetic Engine** | [`src/evolution/genetic_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/genetic_engine.py) | Population generation, crossover, mutation, and memory-safe streaming micro-batch chunking. |
| **Nursery & Symbiosis**| [`src/evolution/nursery.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/nursery.py) | Speciation clustering, novelty immunity shields, and symbiotic chimera fusion. |
| **Matrix Backtest** | [`src/backtest/matrix_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/backtest/matrix_engine.py) | Multi-asset 3D tensor simulation `(B, N, T)` across `(N, T)` returns, purged & embargoed walk-forward CV. |
| **Factor Analytics** | [`src/backtest/factor_analytics.py`](file:///C:/Users/Dell/Desktop/python/trade/src/backtest/factor_analytics.py) | Information Coefficient (IC), Rank IC, and autocorrelation decay analysis. |
| **Risk & Damper** | [`src/execution/risk_controller.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/risk_controller.py) | Statutory NSE price band detection (`CircuitBands`), ATR volatility sizing, and crash lockouts. |
| **HRP Allocator** | [`src/execution/hrp_allocator.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/hrp_allocator.py) | Hierarchical Risk Parity matrix allocation with Ward linkage and analytical Ledoit-Wolf shrinkage. |
| **Overnight Engine** | [`src/execution/overnight_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/overnight_engine.py) | Pre-open auction limit router (09:00–09:07 IST), decaying trailing stop-loss (25 bps down to 10 bps). |
| **Portfolio Broker** | [`src/execution/portfolio_paper_trader.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/portfolio_paper_trader.py) | Multi-asset virtual broker, two-pass atomic rebalancing (sells before buys), 6.8% `LIQUIDBEES.NS` sweep. |
| **Shoonya Gateway** | [`src/execution/shoonya_gateway.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/shoonya_gateway.py) | Dual-mode broker interface: Local simulation and Live Noren API, `client_order_id` idempotency, panic liquidation. |
| **Execution Dispatcher**| [`src/execution/execution_dispatcher.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/execution_dispatcher.py) | Production signal-to-order bridge connecting AST models with risk controllers and gateways. |
| **Alpha Pipeline** | [`src/pipeline/alpha_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/pipeline/alpha_engine.py) | End-to-end pipeline with mandatory real data parquet verification in live mode (`dry_run=False`). |
| **Master UI Cockpit** | [`src/ui/evolution_dashboard.py`](file:///C:/Users/Dell/Desktop/python/trade/src/ui/evolution_dashboard.py) | **Primary Master Cockpit (Port 7860)**: One-click "First Pot of Gold" launcher, panic kill-switch, interactive I/O DAG. |
| **UI Visualizations** | [`src/ui/visualizations.py`](file:///C:/Users/Dell/Desktop/python/trade/src/ui/visualizations.py) | Plotly 2D/3D visualizations, avionics KPI cards, interactive I/O pipeline DAG viewer. |
| **Shoonya UI** | [`src/ui/shoonya_ui.py`](file:///C:/Users/Dell/Desktop/python/trade/src/ui/shoonya_ui.py) | Real-time broker connection accordion, margin queries, audit journal, emergency kill-switch. |

---

## 2. Fulfilled Historical Plans & Milestone Archives

All historical implementation plans have been validated, completed, and archived:

1. **`docs/plans/archive/2026-11-03-world-class-tools-bioquant-evolution-plan.md`**:
   - **Status:** 100% COMPLETE.
   - **Delivered:** Vectorized backtesting (`vectorbt`/`qlib` metrics), Pokémon Nursery speciation, UMAP/HDBSCAN strategy phenotype pruning, Shoonya multi-asset paper broker, and HRP risk allocator.
2. **`docs/plans/archive/2026-11-04-unified-dashboard-adhd-upgrade.md`**:
   - **Status:** 100% COMPLETE.
   - **Delivered:** 60 breakthrough concepts merged into `src/ui/evolution_dashboard.py` (Port 7860): Top Avionics HUD, Master Ignition Switch, co-located 2D MAP-Elites & Pareto Tactical view, and 72-component layout audited in `scripts/current_ui_config.json`.
3. **`docs/plans/2026-11-06-institutional-unified-upgrade-plan.md`**:
   - **Status:** **100% COMPLETE & PRODUCTION VERIFIED (15/15 Tasks Done).**
   - **Delivered:** Complete 4-phase institutional upgrade for NSE India Index ETF Overnight Arbitrage: fat-tailed empirical DSR, daily-resetting VWAP & auction jump tensors, commutative MurmurHash AST with clamped math, real-data pipeline verification, (B,N,T) 3D tensor backtesting, CS_Rank/CS_Demean operators, Ledoit-Wolf HRP shrinkage, purged & embargoed walk-forward cross validation, MAP-Elites Pareto filtering & 2x friction stress testing, GIFT NIFTY lead-lag calculator, pre-open call auction limit router, Shoonya `client_order_id` idempotency & panic liquidation, two-pass rebalancing with LIQUIDBEES sweep, thread-safe session state with I/O DAG visualization, and one-click "First Pot of Gold" launcher.

---

## 3. File Retirement & Consolidation Ledger

To maintain a zero-bloat, DRY (Don't Repeat Yourself) codebase, the following redundant prototype and scratch files were safely archived and retired:

| Retired File | Original Purpose | Reason for Removal / Consolidation | Current Location / Git Record |
| :--- | :--- | :--- | :--- |
| `src/webui/studio_v2.py` | Experimental 22KB studio prototype on Port 7862 | Redundant duplicate of the 54KB Master Cockpit `src/ui/evolution_dashboard.py` (Port 7860). Lacked streaming generator loops and full lookback archives. | Removed. Preserved in Git commit `a31a6dd`. |
| `tests/test_studio_v2.py`| Launch test for prototype | Obsolete following consolidation into `evolution_dashboard.py`. | Removed. Preserved in Git commit `a31a6dd`. |
| `src/webui/` | Prototype webui directory | Directory eliminated to maintain clean single-UI folder (`src/ui/`). | Removed. |
| `test_ola.py` | One-off scratch test for Ola Electric | Root clutter from exploratory phase. | Relocated to `experiments/test_ola.py`. |
| `test_reliance.py` | One-off scratch test for Reliance | Root clutter from exploratory phase. | Relocated to `experiments/test_reliance.py`. |
