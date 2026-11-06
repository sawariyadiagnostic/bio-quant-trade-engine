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
