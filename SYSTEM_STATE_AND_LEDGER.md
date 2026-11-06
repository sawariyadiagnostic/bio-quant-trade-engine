# Bio-Quant System State & Historical Lifecycle Ledger

**Document Classification:** Permanent System Inventory & Architectural Ledger  
**Last Updated:** 2026-11-06  
**System Standard:** Tier-1 Institutional Quantitative Architecture (NSE India)

---

## 1. System Inventory: Active Production Architecture

All components listed below are active, verified, and passing continuous test coverage (23 test suites, 74 assertions):

| Domain / Layer | Primary File | Responsibility & Status |
| :--- | :--- | :--- |
| **Domain Entities** | [`src/domain/models.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/models.py) | Immutable value objects (`Ticker`, `CircuitBands`, `TaxSchedule`, `OrderSignal`). |
| **Quant Invariants** | [`src/domain/invariants.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/invariants.py) | Point-in-time causality, Deflated Sharpe Ratio (DSR), covariance condition numbers ($\kappa < 10^4$). |
| **Formula Engine** | [`src/domain/formula_ast.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/formula_ast.py) | Vectorized AST evaluation, safe operators (`DIV_SAFE`, `EXP_SAFE`), rolling indicator windows. |
| **Parser & Seeds** | [`src/domain/formula_parser.py`](file:///C:/Users/Dell/Desktop/python/trade/src/domain/formula_parser.py) | Recursive descent parsing and quantitative formula seed libraries. |
| **Data Lake** | [`src/data/vector_lake.py`](file:///C:/Users/Dell/Desktop/python/trade/src/data/vector_lake.py) | 5-Year Parquet historical data lake, multi-timeframe candle resampler, 18D embedding tensors. |
| **Manifold Regimes** | [`src/manifold/feature_pipeline.py`](file:///C:/Users/Dell/Desktop/python/trade/src/manifold/feature_pipeline.py) | 18D continuous market feature tensors, UMAP 3D projection, HDBSCAN anchored density clustering. |
| **MAP-Elites Archive** | [`src/evolution/map_elites.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/map_elites.py) | 400-niche behavioral grid (Volatility vs Turnover), CIAS institutional alpha scoring, statutory tax hurdles. |
| **Genetic Engine** | [`src/evolution/genetic_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/genetic_engine.py) | Population generation, crossover, mutation, and memory-safe streaming micro-batch chunking. |
| **Nursery & Symbiosis**| [`src/evolution/nursery.py`](file:///C:/Users/Dell/Desktop/python/trade/src/evolution/nursery.py) | Speciation clustering, novelty immunity shields, and symbiotic chimera fusion. |
| **Matrix Backtest** | [`src/backtest/matrix_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/backtest/matrix_engine.py) | Vectorized matrix simulator computing Sortino (20 bps floor), CAGR, Drawdown across batches. |
| **Factor Analytics** | [`src/backtest/factor_analytics.py`](file:///C:/Users/Dell/Desktop/python/trade/src/backtest/factor_analytics.py) | Information Coefficient (IC), Rank IC, and autocorrelation decay analysis. |
| **Risk & Damper** | [`src/execution/risk_controller.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/risk_controller.py) | Statutory NSE price band detection (`CircuitBands`), ATR volatility sizing, and crash lockouts. |
| **HRP Allocator** | [`src/execution/hrp_allocator.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/hrp_allocator.py) | Hierarchical Risk Parity matrix allocation using correlation dendrograms and inverse variance. |
| **Overnight Engine** | [`src/execution/overnight_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/overnight_engine.py) | Tier 1 (ETF compounding) & Tier 2 (ADR-gated single equity) with trailing micro-stops. |
| **Portfolio Broker** | [`src/execution/portfolio_paper_trader.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/portfolio_paper_trader.py) | Multi-asset virtual broker handling lot sizes, balance tracking, two-pass rebalancing. |
| **Shoonya Gateway** | [`src/execution/shoonya_gateway.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/shoonya_gateway.py) | Dual-mode broker interface: Local zero-risk simulation and Live Noren REST API with TOTP 2FA. |
| **Execution Dispatcher**| [`src/execution/execution_dispatcher.py`](file:///C:/Users/Dell/Desktop/python/trade/src/execution/execution_dispatcher.py) | Production signal-to-order bridge connecting AST models with risk controllers and gateways. |
| **Alpha Pipeline** | [`src/pipeline/alpha_engine.py`](file:///C:/Users/Dell/Desktop/python/trade/src/pipeline/alpha_engine.py) | Unified end-to-end orchestrator connecting data, evolution, manifold pruning, and execution. |
| **Master UI Cockpit** | [`src/ui/evolution_dashboard.py`](file:///C:/Users/Dell/Desktop/python/trade/src/ui/evolution_dashboard.py) | **Primary Master Cockpit (Port 7860)**: 72 components, real-time generator streaming, full leaderboards. |
| **UI Visualizations** | [`src/ui/visualizations.py`](file:///C:/Users/Dell/Desktop/python/trade/src/ui/visualizations.py) | High-contrast Plotly 2D/3D visualizations, avionics KPI cards, and combat telemetry widgets. |
| **Shoonya UI** | [`src/ui/shoonya_ui.py`](file:///C:/Users/Dell/Desktop/python/trade/src/ui/shoonya_ui.py) | Real-time broker connection accordion, margin queries, and order audit dataframes. |

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
   - **Status:** **ACTIVE SUPER PLAN.**
   - **Scope:** 4-Phase Master Dependency DAG implementing the 68 Full Adoption and 43 Surgical Upgrades for overnight index ETF gap arbitrage on NSE.

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
