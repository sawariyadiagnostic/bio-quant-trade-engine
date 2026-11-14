# RAVANA Sovereign Engine: Vulnerabilities & Audit Ledger

This ledger persists all identified mathematical, microstructure, statistical, and execution vulnerabilities discovered during architecture reviews. It serves as an active tracking registry to verify whether each item is addressed in current or future implementation cycles.

---

## Active Vulnerability Register

### 1. Small-Sample Population Stability Index (PSI) Instability
* **Impacted Module:** Muscle 20 (`ravana/monitor/drift_dashboard.py`)
* **Category:** Statistical / Observability Trap
* **Severity:** **HIGH**
* **Description:** At the start of a trading day (e.g. 09:20 IST on 5-minute bars), the live feature DataFrame contains only 1–2 observations. Running `compute_psi` with 10 quantile bins on $<30$ bars creates severe zero-frequency artifacts where `(cur_pct - ref_pct) * log(cur_pct / ref_pct)` explodes towards infinity. This reports a false $\text{PSI} \ge 0.50$, tripping the emergency Kill Switch and halting the system within the first 10 minutes of market open.
* **Remediation Specification:**
  - Enforce `min_observations = 30` (or 60) before evaluating PSI.
  - Return `is_drifted = False` and `psi = 0.0` while in the morning warmup window.
* **Status:** **FLAGGED FOR MONITORING** (Guarded in current implementation).

---

### 2. Potential Naked Shorting in Delta Rebalancing
* **Impacted Module:** Muscle 19 (`ravana/execution/router.py`)
* **Category:** Indian Regulatory / Microstructure Violation
* **Severity:** **HIGH**
* **Description:** In `route_target_allocation`, target shares are diffed against current holdings: `delta = targ_qty - curr_qty`. In Indian cash equity delivery (`NSE`), overnight short sales are legally prohibited and trigger forced exchange auctions with penalties up to 20%. If an accounting discrepancy reports `curr_qty = 0` or if `targ_qty < 0`, an unauthorized naked `SELL` order could be routed to the exchange.
* **Remediation Specification:**
  - When `allow_shorting = False` (default for Indian cash delivery), strictly clamp sell orders:
    ```python
    if not self.allow_shorting:
        sell_shares = min(curr_qty, abs(delta))
    ```
  - Reject order if `sell_shares <= 0`.
* **Status:** **HARDENED IN IMPLEMENTATION**.

---

### 3. Dataclass Mutable Default Syntax in `BacktestResult`
* **Impacted Module:** Muscle 11 (`ravana/backtest/engine.py`)
* **Category:** Code Quality / Static Typing
* **Severity:** **LOW**
* **Description:** The proposed draft assigned `closed_round_trips: List[ object ] = ()`. In Python dataclasses, assigning an empty tuple as the default value to a field typed as `List` violates typing contracts and causes type checker warnings.
* **Remediation Specification:**
  - Use `closed_round_trips: List[RoundTripTrade] = field(default_factory=list)`.
* **Status:** **HARDENED IN IMPLEMENTATION**.

---

### 4. Multi-Day `day_start_equity` Drift in Circuit Breakers
* **Impacted Module:** Muscle 18 (`ravana/risk/circuit_breaker.py`)
* **Category:** Risk Management / State Tracking
* **Severity:** **MEDIUM**
* **Description:** `CircuitBreakerKillSwitch.evaluate` computes single-day drawdown:
  $$\text{daily\_ret} = \frac{\text{current\_equity} - \text{day\_start\_equity}}{\text{day\_start\_equity}}$$
  If a live runtime session spans multiple calendar days without resetting `day_start_equity`, `daily_ret` becomes a multi-day cumulative return rather than single-day loss, causing false halts on ordinary multi-day volatility.
* **Remediation Specification:**
  - Provide an explicit `start_new_day(opening_equity: float)` method.
  - Check calendar date transitions dynamically against `pd.Timestamp.now().date()`.
* **Status:** **HARDENED IN IMPLEMENTATION**.

---

### 5. Lot-Size & Tick-Size Granularity in Sizing
* **Impacted Module:** Muscle 17 (`ravana/risk/sizing.py`)
* **Category:** Microstructure Execution
* **Severity:** **MEDIUM**
* **Description:** Raw share allocation `int(notional // px)` works for equities trading in 1-share increments, but fails for derivative contracts, index futures, or special ETF units with lot size constraints (e.g., lot size = 25 or 50).
* **Remediation Specification:**
  - Support `lot_size` parameter from `AssetMetadata`:
    $$\text{shares} = \left\lfloor \frac{\text{notional} // \text{price}}{\text{lot\_size}} \right\rfloor \times \text{lot\_size}$$
* **Status:** **HARDENED IN IMPLEMENTATION**.

---

### 6. Missing Rejection Audit Rationale
* **Impacted Module:** Muscle 19 (`ravana/execution/router.py`)
* **Category:** Operational Observability
* **Severity:** **LOW**
* **Description:** When an order is rejected by the pre-trade risk filter (kill switch, margin limit, or zero reference price), the order was flagged `REJECTED` but dropped without a human-readable failure reason, making incident diagnosis difficult.
* **Remediation Specification:**
  - Capture and propagate rejection rationale in `RoutingReceipt` and `Order.status_reason`.
* **Status:** **HARDENED IN IMPLEMENTATION**.

---

### 7. US Macro 10.5-Hour Lookahead Leak & Naive Index Constant-Collapse
* **Impacted Module:** Muscle 4 (`ravana/data/cleaner/cleaner.py`)
* **Category:** Temporal Microstructure / Lookahead Bias
* **Severity:** **CRITICAL**
* **Description:** Daily US macro series (e.g., `^GSPC`, `^IXIC`, `^VIX`) stamped at 16:00 naive close were directly localized into `Asia/Kolkata` as 16:00 IST (5:30 AM US Eastern), making the US market close appear 10.5 hours before Wall Street actually closed. Furthermore, when `target_dates` were naive, backward asof matching found no prior bar and filled with `1.0`, corrupting index values (e.g. S&P 500 at 5000 became 1.0).
* **Remediation Specification:**
  - Localize US macro to `America/New_York` at 16:00, then causally convert to `Asia/Kolkata` (02:30 AM IST next morning) so the prior day's US close is causally available at NSE 09:15 open.
  - Fallback fill uses first observed price in `m_copy["Close"]`, preventing constant 1.0 collapse.
* **Status:** **RESOLVED & VERIFIED**.

---

### 8. Execution Timing Contract Mismatch Between M8 (Targets) & M11 (Backtester)
* **Impacted Module:** Muscle 8 (`ravana/targets/builder.py`) & Muscle 11 (`ravana/backtest/engine.py`)
* **Category:** Execution Timing / Overnight Gap Loss
* **Severity:** **CRITICAL**
* **Description:** Muscle 8 calculates overnight returns from Day 1 `15:25 Close` to Day 2 `09:15 Open`. When an alpha signal fired at 15:25, Muscle 11 queued the order for next bar Open (Day 2 `09:15 Open`), filled at the morning peak, and immediately sold at 09:20 Open, completely missing the overnight jump and suffering double friction.
* **Remediation Specification:**
  - In `EventDrivenBacktester`, if `regime == OVERNIGHT` and a buy order is placed on the final bar of the trading session (15:25), execute as Market-on-Close (MOC) fill at bar Close.
  - Position carries overnight and auto-exits at Day 2 09:15 Open, capturing the exact overnight return.
* **Status:** **RESOLVED & VERIFIED**.

---

### 9. Walk-Forward Step/Purge Destruction on Intraday Data
* **Impacted Module:** Muscle 13 (`ravana/validation/walk_forward.py`)
* **Category:** Statistical Cross-Validation / Sample Integrity
* **Severity:** **HIGH**
* **Description:** When dividing raw 5-minute bars into $k+1$ partitions with a 75-bar purge, test fold length shrank to `step - 75` bars (as few as 5 bars), containing zero 15:20 entry bars. This resulted in zero out-of-sample trades evaluated across all folds on real intraday data.
* **Remediation Specification:**
  - Partition walk-forward splits on distinct trading session dates rather than raw 5m bars.
  - Calculate holding embargos in session units (`max_hold - 1`), preserving complete intraday sessions in test folds.
* **Status:** **RESOLVED & VERIFIED**.

---

### 10. Monte Carlo Annualization Collapse on 5m Backtest Equity Curves
* **Impacted Module:** Muscle 16 (`ravana/validation/monte_carlo.py`)
* **Category:** Statistical Annualization / Time-Scale Distortion
* **Severity:** **HIGH**
* **Description:** Passing a 5-minute `BacktestResult.equity_curve` (18,900 bars/yr) to `run_from_result` caused `periods_per_year = 252` to treat 18,900 bars as 75 years of trading, collapsing a 20% annual CAGR down to 0.24% and fracturing session overnight jumps via arbitrary 5m blocks.
* **Remediation Specification:**
  - Resample `BacktestResult.equity_curve` to daily session closes before computing percentage returns and running circular block bootstrap.
* **Status:** **RESOLVED & VERIFIED**.

---

### 11. In-Sample Training Fold Cold-Start Artifacts in Rolling Walk-Forward
* **Impacted Module:** Muscle 13 (`ravana/validation/walk_forward.py`)
* **Category:** Feature Generation / Signal Fidelity
* **Severity:** **MEDIUM**
* **Description:** While test folds had a 60-bar causal warmup, training folds in non-anchored rolling walk-forward began cold at bar `i * step`, causing rolling operators (`EMA_10`, `TS_STD_20`) to produce cold-start artifacts during the initial 20–30 bars of each in-sample fold.
* **Remediation Specification:**
  - Added 60-bar causal warmup window before `train_sl.start` and sliced off the warmup offset.
* **Status:** **RESOLVED & VERIFIED**.

---

### 12. Genetic Point Mutation Physical Dimensional Violation
* **Impacted Module:** Muscle 9 (`ravana/domain/genetics.py`)
* **Category:** Symbolic AST Grammar / Dimensional Invariants
* **Severity:** **MEDIUM**
* **Description:** When `allowed_features` was supplied to `GeneticMutator.mutate_point`, it sampled randomly from `allowed_features` without checking `target_dimension`, allowing a `PRICE` node to mutate into a `VOLUME` node.
* **Remediation Specification:**
  - Filter `allowed_features` by `current_dim` matching `FEATURE_DIMENSIONS`.
* **Status:** **RESOLVED & VERIFIED**.

---

### 13. `TS_DELAY` Prefix Edge-Padding Discrepancy on Short Slices
* **Impacted Module:** Muscle 1 (`ravana/domain/formula_ast.py`)
* **Category:** Numerical Parity
* **Severity:** **LOW**
* **Description:** For short series where `len(s) <= period`, `s.shift().bfill()` produced NaNs filled with `0.0`, whereas PyTorch tensor primitives edge-padded with `child[0]`.
* **Remediation Specification:**
  - Standardized `.fillna(float(child[0]))` in `formula_ast.py`.
* **Status:** **RESOLVED & VERIFIED**.

---

### 14. Risk-Free Rate Inconsistency Between Metrics Engine and Backtester
* **Impacted Module:** Muscle 11 (`ravana/backtest/engine.py`)
* **Category:** Performance Metric Standardization
* **Severity:** **LOW**
* **Description:** Metrics engine deducted 6.5% annual risk-free rate for Sharpe/Sortino calculations, while Backtester computed gross Sharpe without subtracting risk-free rate, causing a ~0.5 discrepancy.
* **Remediation Specification:**
  - Added `risk_free_rate: float = 0.065` to `BacktestConfig` and applied excess return scaling in `_compute_performance_metrics`.
* **Status:** **RESOLVED & VERIFIED**.

---

### 15. `ValidationPipeline` Exception Isolation
* **Impacted Module:** Phase 5 (`ravana/validation/pipeline.py`)
* **Category:** Pipeline Reliability
* **Severity:** **LOW**
* **Description:** Lack of try/except isolation in `certify_evolution_result` caused unhandled formula exceptions to crash the entire validation pipeline.
* **Remediation Specification:**
  - Wrapped each validation muscle in isolated try/except blocks with graceful fallbacks.
* **Status:** **RESOLVED & VERIFIED**.

---

### 16. Multi-Horizon Forward Return Session Boundary & Quarantine Leakage
* **Impacted Module:** Muscle 8 (`ravana/targets/builder.py`, `config.py`)
* **Category:** Lookahead & Session Quarantine
* **Severity:** **HIGH**
* **Description:** When evaluating intraday forward horizons ($N$-bars ahead, e.g. 6 bars on 5m data), bars near the end of the session could attempt to compute forward returns crossing the 15:30 IST session close into the next day's 09:15 open, or enter within the broker auto-square-off window (15:15 IST).
* **Remediation Specification:**
  - Implemented `TargetHorizon.INTRADAY_N_BARS` with strict session equality validation: `session_dates[i+1] == session_dates[i+1+k]`.
  - Enforced `intraday_cutoff_time = "14:45"` IST; any bar evaluated after 14:45 has `is_entry_bar = 0`, `sample_weight = 0.0`, and zero forward return.
  - Applied `IntradayFrictionModel` reflecting exact Indian Finance Act statutory rules (0.0% ETF STT, Rs 0.00 CDSL DP fee, total friction < 1.0 bps).
* **Status:** **RESOLVED & VERIFIED**.

---

### 17. Intraday MIS Passive Holding Drag (Blind 15:15 Square-Off)
* **Impacted Module:** Muscle 11 (`ravana/backtest/engine.py`, `config.py`)
* **Category:** Execution Efficiency / Risk Hysteresis
* **Severity:** **MEDIUM**
* **Description:** Intraday MIS positions held until the mandatory 15:15 IST auto-square-off even if the alpha signal violently reversed at 10:30 IST, suffering severe adverse excursion and drawdown.
* **Remediation Specification:**
  - Implemented signal-reversal exit logic for `TradingRegime.INTRADAY_MIS`.
  - Added `min_hold_bars = 2` hysteresis to prevent adjacent-bar whipsaw exits.
  - When `position.bars_held >= min_hold_bars` and signal flips (`signal < -threshold` or `signal < signal_exit_threshold`), `EventDrivenBacktester` causally emits a `SELL` order at the next bar's Open.
* **Status:** **RESOLVED & VERIFIED**.

---

### 18. Cross-Sectional Degeneracy on Single-Asset and 2D Slices
* **Impacted Module:** Muscle 1 & 2 (`ravana/compute/primitives.py`, `tensor_evaluator.py`, `ravana/domain/formula_ast.py`)
* **Category:** Mathematical Robustness
* **Severity:** **MEDIUM**
* **Description:** Applying cross-sectional operators (`CS_RANK`, `CS_ZSCORE`) to single-asset portfolios (1D `[T]`, 2D `[T, K]`, or 3D with $B=1$) previously returned flat constant `0.5` or `0.0`, rendering any generated alpha with cross-sectional operators zero-variance and dead during single-asset symbolic search.
* **Remediation Specification:**
  - In `cs_rank`, gracefully fall back to `ts_rank_20(x)` when $B=1$ or tensor is 1D/2D, preserving valid bounded variance $[0.0, 1.0]$.
  - In `cs_zscore`, gracefully fall back to `ts_zscore_20(x)` when $B=1$ or tensor is 1D/2D, preserving standardized variance in $[-4.0, 4.0]$.
  - For 3D multi-asset tensors $[B, T, K]$ with $B > 1$, rank and standardize strictly across dim 0.
* **Status:** **RESOLVED & VERIFIED**.

---

### 19. Hardcoded `100.0` Parity Ratio Basis Distortion
* **Impacted Module:** Muscle 4 (`ravana/data/cleaner/spread_aligner.py`)
* **Category:** Data Integrity / Microstructure Distortion
* **Severity:** **HIGH**
* **Description:** Hardcoded `KNOWN_ETF_RATIOS = {"BANKBEES.NS": 100.0, "NIFTYBEES.NS": 100.0}` created artificial basis offsets over 20 years of tracking divergence (e.g. `^NSEBANK` ~52,000 vs `BANKBEES` ~590 gives ratio $\approx 88.13$, creating a $+1,346$ bps permanent error at 100.0). For Sector BeES like `ITBEES` (ratio $\approx 954$), tracking was completely unusable.
* **Remediation Specification:**
  - Removed static `KNOWN_ETF_RATIOS`.
  - Implemented strictly causal rolling/expanding median parity ratio:
    `raw_ratios = (idx_close / np.maximum(0.01, etf_close)).shift(1)`
    `rolling_mult = raw_ratios.rolling(parity_window, min_periods=5).median()`
    `expanding_mult = raw_ratios.expanding(min_periods=1).median()`
  - Self-calibrates without lookahead bias across all Sector BeES and index splits.
* **Status:** **RESOLVED & VERIFIED**.

---

### 20. Sector BeES Universe & Benchmark Index Linkage
* **Impacted Module:** Muscle 3 (`ravana/data/ingestion/config.py`)
* **Category:** Asset Universe Expansion
* **Severity:** **LOW / ENHANCEMENT**
* **Description:** Ingestion module previously restricted to `BANKBEES.NS` and `NIFTYBEES.NS`, omitting the full liquid Indian Sector BeES complex and their corresponding benchmark indices.
* **Remediation Specification:**
  - Expanded `CORE_ETF_UNIVERSE` to include 9 liquid ETFs: `BANKBEES.NS`, `NIFTYBEES.NS`, `ITBEES.NS`, `PHARMABEES.NS`, `PSUBNKBEES.NS`, `JUNIORBEES.NS`, `AUTOBEES.NS`, `CPSEETF.NS`, `GOLDBEES.NS`.
  - Expanded `DOMESTIC_MACRO_UNIVERSE` to include benchmark indices: `^NSEBANK`, `^NSEI`, `^CNXIT`, `^CNXPHARMA`, `^CNXPSUBANK`, `^NSMIDCP`, `^CNXAUTO`, `^INDIAVIX`.
  - Configured `ETF_BENCHMARK_MAP` connecting each sector ETF directly to its underlying NSE index.
* **Status:** **RESOLVED & VERIFIED**.

---

### 21. Clock-Blindness in Muscle 7 Feature Matrix
* **Impacted Module:** Muscle 7 (`ravana/features/config.py`, `matrix_engine.py`) & Muscle 1 (`ravana/domain/formula_ast.py`)
* **Category:** Microstructure & Temporal Feature Representation
* **Severity:** **HIGH**
* **Description:** All 21 features in Muscle 7 were purely rolling statistical filters without any session-time awareness or opening-range reference. Symbolic alphas in Muscle 9 could not distinguish 09:20 (Morning Overflow) or 14:05 (Bond Window) from 12:30 (mid-day lull), and could not measure opening range breakouts or avoid the 15:00-15:15 retail MIS forced liquidation dump.
* **Remediation Specification:**
  - Expanded feature set from 21 to 26 stationary features.
  - Added `MACRO_YIELD_RET_1`: 1-bar return in 10Y Bond Yield (`^TNX`).
  - Added `SESSION_CLOCK_NORM`: normalized session progress in $[-1.0, +1.0]$.
  - Added `MORNING_OVERFLOW_WINDOW`: binary flag for `09:15-10:30` IST morning digestion wave.
  - Added `AFTERNOON_BOND_WINDOW`: `+1.0` during `13:55-14:40` IST bond/European window, `-1.0` during `14:55-15:15` retail MIS auto-square-off dump, `0.0` otherwise.
  - Added `OPENING_RANGE_DRIFT`: ATR-normalized breakout drift from `09:15-09:45` opening range midpoint.
  - Added all 5 features to `FEATURE_DIMENSIONS` in `formula_ast.py` as `DIMENSIONLESS`.
* **Status:** **RESOLVED & VERIFIED**.

---

### 22. Broker Mismatch & Exchange Cancellation Status Normalization
* **Impacted Module:** Muscle 19 (`ravana/execution/shoonya_adapter.py`, `ravana/execution/router.py`, `ravana/risk/circuit_breaker.py`)
* **Category:** Live Execution Gateway & Microstructure
* **Severity:** **HIGH**
* **Description:** The system had defaulted to generic Zerodha Kite routing concepts despite having a complete, zero-brokerage Shoonya (Finvasia Noren API) infrastructure in the codebase. Furthermore, terminal order status checks in execution polling strictly looked for `"CANCELED"`, causing orders cancelled by Finvasia (spelled with British double-L `"CANCELLED"`) to busy-wait until polling timeout.
* **Remediation Specification:**
  - Implemented `ShoonyaBrokerGateway` in `ravana/execution/shoonya_adapter.py` subclassing `ravana.execution.router.BrokerGateway`.
  - Added headless RFC 6238 TOTP authentication via `pyotp` with 4-second epoch boundary guard to eliminate race conditions.
  - Added `TokenBucketRateLimiter` pacing outbound requests to $\le 8.0$ req/s (125ms delay) with exponential backoff on HTTP 429.
  - Added thread-safe auto-relogin retry logic upon encountering `Session Expired`.
  - Hardened fill polling to immediately recognize terminal statuses: `("REJECTED", "CANCELED", "CANCELLED")`.
  - Wired Muscle 18 circuit breaker emergency interlocks (`cancel_all_orders` and `liquidate_positions`).
  - Full test suite verified across 17 comprehensive unit/integration tests in `test_shoonya_adapter.py`.
* **Status:** **RESOLVED & VERIFIED**.

---

## Chronological Audit Log

| Date | Phase | Module(s) | Finding | Resolution Status |
| :--- | :--- | :--- | :--- | :--- |
| **2026-11-11** | Phase 1–4 | M9, M10, M11 | 39-node tree bloat, clone frontier, static DSR variance | Resolved with `max_nodes=15`, correlation niching, and dynamic DSR. |
| **2026-11-12** | Phase 5 | M13, M14, M15, M16 | In-sample frozen standardization, circular block bootstrap | Resolved with 7 Red-Team fixes & 140 green tests. |
| **2026-11-14** | M1–12 Audit | M1, M4, M5, M7, M8, M11, M12 | 10 system-wide bugs (overnight timing, slice cache, flat RSI, SVD) | Resolved with `test_audit_remediation.py` & 150 green tests. |
| **2026-11-14** | Phase 6–7 | M17, M18, M19, M20 | Small-sample PSI instability, naked shorting risks, lot sizes | Cataloged in this ledger; hardened in implementation. |
| **2026-11-14** | Forensic Review | M1, M4, M8, M9, M11, M13, M16 | US Macro 10.5h leak, M8/M11 timing, WF session chunking, MC daily | Resolved & verified across all 156 unit tests. |
| **2026-11-15** | Phase A & B | M1, M2, M8, M11 | Multi-horizon targets, intraday friction, signal reversal, cross-sectional ops | Resolved & verified across 201 unit tests. |
| **2026-11-15** | Dynamic Parity & Sector Expansion | M3, M4, M1, M8 | Dynamic rolling parity ratio, Sector BeES mapping, non-degenerate CS fallback | Resolved & verified across 202 unit tests (100% green). |
| **2026-11-15** | Intraday Windows & Clock Features | M7, M1 | 26 stationary features, session clock, morning overflow, afternoon bond, opening range drift | Resolved & verified across 204 unit tests (100% green). |
| **2026-11-15** | Shoonya Noren Broker Gateway | M19, M18, M11 | Zero-brokerage Finvasia adapter, TOTP epoch buffer, token bucket pacing, cancel status normalization | Resolved & verified across 221 unit tests (100% green). |
