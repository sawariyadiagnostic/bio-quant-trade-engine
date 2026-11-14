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

## Chronological Audit Log

| Date | Phase | Module(s) | Finding | Resolution Status |
| :--- | :--- | :--- | :--- | :--- |
| **2026-11-11** | Phase 1–4 | M9, M10, M11 | 39-node tree bloat, clone frontier, static DSR variance | Resolved with `max_nodes=15`, correlation niching, and dynamic DSR. |
| **2026-11-12** | Phase 5 | M13, M14, M15, M16 | In-sample frozen standardization, circular block bootstrap | Resolved with 7 Red-Team fixes & 140 green tests. |
| **2026-11-14** | M1–12 Audit | M1, M4, M5, M7, M8, M11, M12 | 10 system-wide bugs (overnight timing, slice cache, flat RSI, SVD) | Resolved with `test_audit_remediation.py` & 150 green tests. |
| **2026-11-14** | Phase 6–7 | M17, M18, M19, M20 | Small-sample PSI instability, naked shorting risks, lot sizes | Cataloged in this ledger; hardened in implementation. |
