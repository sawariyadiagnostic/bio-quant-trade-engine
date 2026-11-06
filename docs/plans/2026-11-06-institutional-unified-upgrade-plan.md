# Institutional Quant Alpha-OS: Master Granular 3-in-1 Upgrade Plan

> **For Claude / Subagent:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.  
> **Status Conventions:** `[ ] Not Started` | `[/] In Progress` | `[x] Complete`

**Goal:** Transform the Bio-Quant trading system into an institutional-grade, zero-bloat algorithmic trading engine specialized for Indian Equity Index ETFs (`NIFTYBEES`, `BANKBEES`), executing high-expectancy overnight gap arbitrage with a statutory 1.2 bps tax shield to compound the "First Pot of Gold".

**Architecture:** 
1. **P0 (Foundations & Invariants):** Correct mathematical bugs in causality and DSR, reset daily VWAP at 09:15:00 IST, decompose overnight vs intraday jumps, and clamp AST numerical ranges.
2. **P1 (Core Quant & Risk Engine):** Vectorize multi-asset 3D simulation `(B, N, T)`, add cross-sectional operators (`CS_Rank`, `CS_Demean`), upgrade HRP to Ward/Average linkage with Ledoit-Wolf shrinkage, and enforce Purged Walk-Forward validation.
3. **P2 ("First Pot of Gold" Execution):** Operationalize GIFT Nifty pre-market lead-lag, pre-open call auction routing (09:00–09:07 IST), two-pass atomic rebalancing (sells before buys), `clordid` idempotency, and 6.8% `LIQUIDBEES` auto-sweep.
4. **P3 (Master Cockpit UI & Dynamic I/O DAG):** Provide a thread-safe Gradio UI (`gr.State`) in `src/ui/evolution_dashboard.py` (Port 7860) with an interactive I/O data flow DAG and physical emergency panic kill-switch.

**Tech Stack:** Python 3.11, NumPy, SciPy (hierarchical clustering & Ledoit-Wolf), Pandas, PyArrow/Polars (memory-mapped Parquet), Numba/Vectorized NumPy, Finvasia Shoonya Noren API, Gradio 3.50+ / Plotly.

---

## Master Progress Tracker (Always Update In-Place)

| Priority | Task ID | Core Focus | Primary Target Files | Status | Commit / Stamp |
| :---: | :---: | :--- | :--- | :---: | :---: |
| **P0** | **Task 1** | Fix Causality Check & Empirical Fat-Tailed DSR | `src/domain/invariants.py`, `tests/test_invariants.py` | `[x]` Complete | `fix(invariants)` |
| **P0** | **Task 2** | Daily-Resetting VWAP & Auction Jump Decomposition | `src/data/vector_lake.py`, `tests/test_vector_lake.py` | `[ ]` Not Started | - |
| **P0** | **Task 3** | AST Math Clamping (`EXP/LOG/POW_SAFE`) & MurmurHash | `src/domain/formula_ast.py`, `tests/test_mega_evolution_ast.py`| `[ ]` Not Started | - |
| **P0** | **Task 4** | Statutory Tax Schedule & Real Data Enforcing | `src/domain/models.py`, `src/pipeline/alpha_engine.py` | `[ ]` Not Started | - |
| **P1** | **Task 5** | Multi-Asset 3D Tensor Simulation `(B, N, T)` | `src/backtest/matrix_engine.py`, `tests/test_matrix_engine.py` | `[ ]` Not Started | - |
| **P1** | **Task 6** | Cross-Sectional Alpha Operators (`CS_Rank`, `CS_Demean`) | `src/domain/formula_ast.py`, `tests/test_mega_evolution_ast.py`| `[ ]` Not Started | - |
| **P1** | **Task 7** | Ward/Average Linkage HRP & Ledoit-Wolf Shrinkage | `src/execution/hrp_allocator.py`, `tests/test_hrp_allocator.py`| `[ ]` Not Started | - |
| **P1** | **Task 8** | Purged & Embargoed Walk-Forward Validation | `src/backtest/matrix_engine.py`, `tests/test_matrix_engine.py` | `[ ]` Not Started | - |
| **P1** | **Task 9** | MAP-Elites 2-Objective Pareto & 2x Friction Stress Test | `src/evolution/map_elites.py`, `tests/test_lookback_cagr_partitioning.py` | `[ ]` Not Started | - |
| **P2** | **Task 10**| GIFT NIFTY Overnight Lead-Lag Ingestion (06:30–08:30 IST)| `src/data/gift_nifty_loader.py`, `src/execution/overnight_engine.py` | `[ ]` Not Started | - |
| **P2** | **Task 11**| Pre-Open Auction Limit Router & Micro-Trailing Stop | `src/execution/overnight_engine.py`, `tests/test_overnight_engine.py` | `[ ]` Not Started | - |
| **P2** | **Task 12**| Shoonya Gateway Hardening (`clordid`, Retries, Sell Fix)| `src/execution/shoonya_gateway.py`, `src/execution/execution_dispatcher.py` | `[ ]` Not Started | - |
| **P2** | **Task 13**| Two-Pass Atomic Rebalancing & Liquid BeES Auto-Sweep | `src/execution/portfolio_paper_trader.py`, `tests/test_portfolio_paper_trader.py` | `[ ]` Not Started | - |
| **P3** | **Task 14**| Thread-Safe State & Interactive I/O DAG in Master Cockpit| `src/ui/evolution_dashboard.py`, `src/ui/visualizations.py` | `[ ]` Not Started | - |
| **P3** | **Task 15**| One-Click "First Pot of Gold" Launcher & Panic Switch | `src/ui/evolution_dashboard.py`, `src/ui/shoonya_ui.py` | `[ ]` Not Started | - |

---

## Detailed Task Specifications (Bite-Sized Actionable Run)

### PRIORITY 0: Foundational Prerequisites (Zero-Tolerance Quant Invariants)
*Must be completed first. Downstream multi-asset simulations cannot run on flawed mathematical or data leakage baselines.*

---

### Task 1: Fix Causality Check & Empirical Fat-Tailed DSR
**Priority:** `P0 - Blocker`  
**Files:**
- Modify: `src/domain/invariants.py:45-95`
- Test: `tests/test_invariants.py`

- [x] **Step 1: Write the failing test**
  In `tests/test_invariants.py`, add `test_causality_allows_positive_forward_ic_and_rejects_backward_leak` and `test_dsr_with_empirical_kurtosis`. Verify that valid predictive forward correlations ($IC = 0.45$) pass, while contemporaneous/backward leakage is caught.
- [x] **Step 2: Run test to verify it fails**
  Run: `python tests/test_invariants.py`  
  Expected: FAIL (due to static $|r| \le 0.25$ cap).
- [x] **Step 3: Implement minimal code**
  In `src/domain/invariants.py`:
  - Update `verify_point_in_time_causality`: Remove the $|r| \le 0.25$ forward cap; add contemporaneous and negative lag checks.
  - Update `calculate_deflated_sharpe_ratio`: Accept empirical `returns_series` and calculate `scipy.stats.skew` and `scipy.stats.kurtosis` dynamically.
- [x] **Step 4: Run test to verify it passes**
  Run: `python tests/test_invariants.py`  
  Expected: PASS (all assertions green).
- [x] **Step 5: Commit changes**
  Commit: `git commit -m "fix(invariants): patch causality forward IC check and add fat-tailed empirical DSR"`

---

### Task 2: Daily-Resetting VWAP & Auction Jump Decomposition
**Priority:** `P0 - Blocker`  
**Files:**
- Modify: `src/data/vector_lake.py:65-115`
- Test: `tests/test_vector_lake.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_vector_lake.py`, assert that `VectorDataLake` contains `overnight_auction_jump` ($Close_{t-1} \to Open_t$), `intraday_core_return` ($Open_t \to Close_t$), and that VWAP resets at session day boundaries.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_vector_lake.py`  
  Expected: FAIL (`KeyError: 'overnight_auction_jump'`).
- [ ] **Step 3: Implement minimal code**
  In `src/data/vector_lake.py`:
  - Compute `overnight_auction_jump = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)`.
  - Compute `intraday_core_return = (df["Close"] - df["Open"]) / df["Open"]`.
  - Reset cumulative volume and typical price arrays when dates change.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_vector_lake.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(data): implement daily session-resetting VWAP and overnight auction jump decomposition"`

---

### Task 3: AST Math Clamping (`EXP/LOG/POW_SAFE`) & Canonical MurmurHash
**Priority:** `P0 - Blocker`  
**Files:**
- Modify: `src/domain/formula_ast.py:100-140`
- Test: `tests/test_mega_evolution_ast.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_mega_evolution_ast.py`, verify that `ASTNode.canonical_hash()` generates identical hashes for commutative operations (`ADD(A, B) == ADD(B, A)`), and `EXP_SAFE` bounds inputs to $[-20.0, 20.0]$.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_mega_evolution_ast.py`  
  Expected: FAIL (`AttributeError: 'ASTNode' object has no attribute 'canonical_hash'`).
- [ ] **Step 3: Implement minimal code**
  In `src/domain/formula_ast.py`:
  - Implement `canonical_hash()` sorting commutative children.
  - Implement bounded math operators: `EXP_SAFE = np.exp(np.clip(child, -20.0, 20.0))`, `LOG_SAFE = np.log(np.maximum(1e-8, np.abs(child)))`.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_mega_evolution_ast.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(ast): add canonical formula hashing and bounded mathematical operators"`

---

### Task 4: Statutory Tax Schedule & Real Data Enforcing
**Priority:** `P0 - Blocker`  
**Files:**
- Modify: `src/domain/models.py:80-130`
- Modify: `src/pipeline/alpha_engine.py:40-70`
- Test: `tests/test_skills_validation.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_skills_validation.py`, assert that `AlphaEnginePipeline.run_cycle(dry_run=False)` raises `FileNotFoundError` if Parquet files are missing, banning random geometric Brownian motion in live mode.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_skills_validation.py`  
  Expected: FAIL if live mock suppression is not enforced.
- [ ] **Step 3: Implement minimal code**
  In `src/pipeline/alpha_engine.py`:
  - When `live_mode` or `dry_run=False`, throw explicit error if `data/lake/` Parquet files are missing.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_skills_validation.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "fix(pipeline): enforce real data requirements and statutory tax schedules"`

---

### PRIORITY 1: Core Quant Engine & Risk Parity Upgrades
*Executed in parallel once P0 foundations are locked.*

---

### Task 5: Multi-Asset 3D Tensor Simulation (B, N, T)
**Priority:** `P1 - High`  
**Files:**
- Modify: `src/backtest/matrix_engine.py:10-70`
- Test: `tests/test_matrix_engine.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_matrix_engine.py`, add `test_matrix_simulator_3d_multi_asset` testing `positions` of shape `(B, N, T)` and `returns` of shape `(N, T)` across `NIFTYBEES`, `BANKBEES`, and `GOLDBEES`.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_matrix_engine.py`  
  Expected: FAIL (`ValueError: positions must be 2D`).
- [ ] **Step 3: Implement minimal code**
  In `src/backtest/matrix_engine.py`:
  - Support both 2D and 3D inputs. For 3D, compute cross-sectional portfolio net return at bar $t$:
    $$R_{port, b, t} = \sum_{n=1}^N (P_{b, n, t} \times R_{n, t}) - \text{friction} \times \sum_{n=1}^N |\Delta P_{b, n, t}|$$
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_matrix_engine.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(backtest): implement vectorized 3D tensor simulation for multi-asset portfolios"`

---

### Task 6: Cross-Sectional Alpha Operators (`CS_Rank`, `CS_Demean`)
**Priority:** `P1 - High`  
**Files:**
- Modify: `src/domain/formula_ast.py:60-120`
- Test: `tests/test_mega_evolution_ast.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_mega_evolution_ast.py`, assert that `CS_Rank` normalizes across assets to $[0.0, 1.0]$, and `CS_Demean` centers values to zero mean.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_mega_evolution_ast.py`  
  Expected: FAIL (`KeyError: 'CS_Rank'`).
- [ ] **Step 3: Implement minimal code**
  In `src/domain/formula_ast.py`:
  - Implement `CS_Rank` using `scipy.stats.rankdata(axis=0)`.
  - Implement `CS_Demean`: `arr - np.mean(arr, axis=0)`.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_mega_evolution_ast.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(ast): add cross-sectional rank and demean operators"`

---

### Task 7: Ward/Average Linkage HRP & Ledoit-Wolf Shrinkage
**Priority:** `P1 - High`  
**Files:**
- Modify: `src/execution/hrp_allocator.py:25-70`
- Test: `tests/test_hrp_allocator.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_hrp_allocator.py`, test collinear/ill-conditioned covariance matrices and assert linkage is `"ward"` or `"average"`, producing non-chaining balanced risk allocations.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_hrp_allocator.py`  
  Expected: FAIL if single linkage causes degenerate chaining.
- [ ] **Step 3: Implement minimal code**
  In `src/execution/hrp_allocator.py`:
  - When condition number $\kappa > 1000$, apply `sklearn.covariance.ledoit_wolf`.
  - Update `linkage(condensed, method="average")` or `"ward"`.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_hrp_allocator.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "fix(risk): upgrade HRP to Ward/Average linkage with Ledoit-Wolf shrinkage"`

---

### Task 8: Purged & Embargoed Walk-Forward Validation
**Priority:** `P1 - High`  
**Files:**
- Modify: `src/backtest/matrix_engine.py:75-110`
- Test: `tests/test_matrix_engine.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_matrix_engine.py`, assert `MatrixSimulator.purged_walk_forward_split` generates splits with a 5-bar purge and 5-bar embargo window.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_matrix_engine.py`  
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  Implement `purged_walk_forward_split(n_bars, train_ratio=0.7, purge_window=5, embargo_window=5)` in `matrix_engine.py`.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_matrix_engine.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(backtest): implement purged and embargoed walk-forward cross-validation"`

---

### Task 9: MAP-Elites 2-Objective Pareto & 2x Friction Stress Test
**Priority:** `P1 - High`  
**Files:**
- Modify: `src/evolution/map_elites.py:70-130`
- Test: `tests/test_lookback_cagr_partitioning.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_lookback_cagr_partitioning.py`, assert that candidates are stress-tested at 2x friction ($2.4$ bps for ETFs); formulas that lose edge under 2x friction are rejected.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_lookback_cagr_partitioning.py`  
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  In `MapElitesArchive._calculate_metrics`, compute both base and $2\times$ friction net returns. Only admit candidates where $2\times$ friction Sharpe remains $> 0.80$.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_lookback_cagr_partitioning.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(evolution): add 2x friction stress testing and Pareto filtering to MAP-Elites"`

---

### PRIORITY 2: "First Pot of Gold" Overnight Execution Engine
*Builds directly upon P0 and P1.*

---

### Task 10: GIFT NIFTY Overnight Lead-Lag Ingestion (06:30–08:30 IST)
**Priority:** `P2 - Core Edge`  
**Files:**
- Create: `src/data/gift_nifty_loader.py`
- Modify: `src/execution/overnight_engine.py:20-55`
- Test: `tests/test_gift_nifty_telemetry.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_gift_nifty_telemetry.py`, verify that GIFT Nifty pre-market drift (06:30–08:30 IST) generates an `OvernightLeadSignal` with expected open gap direction.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_gift_nifty_telemetry.py`  
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.data.gift_nifty_loader'`).
- [ ] **Step 3: Implement minimal code**
  Create `src/data/gift_nifty_loader.py`:
  - Fetch early morning GIFT Nifty index delta.
  - Compute normalized lead signal: $\Delta_{GIFT} = \frac{P_{08:30} - P_{15:30}}{P_{15:30}}$.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_gift_nifty_telemetry.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(data): implement GIFT Nifty early morning telemetry ingestion"`

---

### Task 11: Pre-Open Auction Limit Router & Micro-Trailing Stop
**Priority:** `P2 - Core Edge`  
**Files:**
- Modify: `src/execution/overnight_engine.py:80-140`
- Test: `tests/test_overnight_engine.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_overnight_engine.py`, assert:
  1. Orders generated between 09:00 and 09:07 IST are tagged as `LIMIT` auction orders at indicative price.
  2. Trailing stop decays dynamically: $0.20\%$ at 09:15:00, widening to $0.45\%$ by 09:30:00.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_overnight_engine.py`  
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  In `TrailingStopSimulator.resolve_exit`, implement time-dependent stop decay based on elapsed minutes from market open.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_overnight_engine.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(execution): implement pre-open call auction routing and decaying trailing stops"`

---

### Task 12: Shoonya Gateway Hardening (`clordid`, Retries, Sell Fix)
**Priority:** `P2 - Core Edge`  
**Files:**
- Modify: `src/execution/shoonya_gateway.py:220-310`
- Modify: `src/execution/execution_dispatcher.py:40-60`
- Test: `tests/test_shoonya_gateway.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_shoonya_gateway.py`, verify that:
  1. Every order sent has a unique `clordid` parameter.
  2. In `ExecutionDispatcher`, an action `"SELL"` actively calls `gateway.place_order(action="SELL")` to liquidate held units.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_shoonya_gateway.py`  
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  In `src/execution/execution_dispatcher.py`:
  - When `action == "SELL"`, query open positions for `ticker`; if held $> 0$, route sell order to `self.gateway.place_order(..., action="SELL")`.
  In `src/execution/shoonya_gateway.py`:
  - Add `clordid` to order payloads and implement exponential backoff on HTTP 429/502.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_shoonya_gateway.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "fix(gateway): add clordid idempotency, retries, and active sell routing"`

---

### Task 13: Two-Pass Atomic Rebalancing & Liquid BeES Auto-Sweep
**Priority:** `P2 - Core Edge`  
**Files:**
- Modify: `src/execution/portfolio_paper_trader.py:35-85`
- Test: `tests/test_portfolio_paper_trader.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_portfolio_paper_trader.py`, verify that:
  1. Rebalancing guarantees all Sells are executed and confirmed before Buys are placed.
  2. Unallocated cash is swept into `LIQUIDBEES` at 15:20 IST, earning $+6.8\%$ annualized interest.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_portfolio_paper_trader.py`
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  In `PortfolioPaperTrader.rebalance`:
  - Enforce strict two-pass loop: Pass 1 (Sells + margin credit update), Pass 2 (Buys with investable cash).
  - Add `sweep_to_liquid_bees(interest_rate=0.068)`.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_portfolio_paper_trader.py`
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(execution): implement two-pass atomic rebalancing and Liquid BeES auto-sweep"`

---

### PRIORITY 3: Master Cockpit UI & Dynamic I/O Telemetry
*Exposes all verified engine capabilities onto the live dashboard on Port 7860.*

---

### Task 14: Thread-Safe State & Interactive I/O DAG in Master Cockpit
**Priority:** `P3 - UI / Avionics`  
**Files:**
- Modify: `src/ui/evolution_dashboard.py:400-500`
- Modify: `src/ui/visualizations.py:200-260`
- Test: `tests/test_visual_spaces.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_visual_spaces.py`, verify that `evolution_dashboard.py` renders an interactive Mermaid/HTML pipeline DAG and encapsulates session state in `gr.State` rather than global module variables.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_visual_spaces.py`  
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  In `src/ui/evolution_dashboard.py`:
  - Wrap session states inside `gr.State()` within `create_ui()`.
  - Add an interactive visual flowchart rendering the exact I/O contracts of each module in the tactical tab.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_visual_spaces.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(ui): add thread-safe session state and interactive I/O pipeline DAG to master cockpit"`

---

### Task 15: One-Click "First Pot of Gold" Launcher & Panic Switch in Master Cockpit
**Priority:** `P3 - UI / Avionics`  
**Files:**
- Modify: `src/ui/evolution_dashboard.py:900-1050`
- Modify: `src/ui/shoonya_ui.py:100-160`
- Test: `tests/test_archive_wipe.py`

- [ ] **Step 1: Write the failing test**
  In `tests/test_archive_wipe.py`, verify that clicking the "Emergency Panic Kill-Switch" in `evolution_dashboard.py` cancels all open broker orders, halts background streams, and sets status to `EMERGENCY_HALTED`.
- [ ] **Step 2: Run test to verify it fails**
  Run: `python tests/test_archive_wipe.py`  
  Expected: FAIL.
- [ ] **Step 3: Implement minimal code**
  In `src/ui/evolution_dashboard.py`:
  - Add the `Emergency Kill-Switch` button wired to `ShoonyaGateway.cancel_all_orders()` and `PortfolioPaperTrader.rebalance(all_zero_weights)`.
  - Add the "First Pot of Gold" one-click launcher pre-configured for `NIFTYBEES` overnight arbitrage.
- [ ] **Step 4: Run test to verify it passes**
  Run: `python tests/test_archive_wipe.py`  
  Expected: PASS.
- [ ] **Step 5: Commit changes**
  Commit: `git commit -m "feat(ui): add one-click overnight arbitrage launcher and emergency kill-switch"`

---

## Global Verification & Completion Gate

Before marking the entire upgrade complete, all three commands must execute cleanly with zero errors:

```powershell
# 1. Execute all unit and regression test suites (23+ files)
python scripts/run_all_tests.py

# 2. Run the continuous Red Team Invariant Harness (All 12 invariants)
python scripts/red_team_harness.py --mode=full

# 3. Verify zero credential leakage
python scripts/check_secrets.py
```
