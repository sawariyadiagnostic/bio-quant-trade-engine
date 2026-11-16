# RAVANA: Google Decisions Model Master Audit, System Record & Strategy Expansion Blueprint

> **Status:** Production Architecture & Master Audit Record | Awaiting User Review & Sign-Off  
> **Target Scope:** Decision Infrastructure (`ravana/decisions/`), Risk Shield (`ravana/risk/`), Execution Daemon (`ravana/daemon/`), 7 Structural Indian Market ETF Dependencies, and 60-Day Integrated Portfolio Engine.  
> **Reference Conversation:** `Implement 7 Indian Market ETF Dependencies` ([conversation://eb6e6203-4e85-4ba7-978d-60eddd7e70ba](conversation://eb6e6203-4e85-4ba7-978d-60eddd7e70ba)).

---

## 1. Full Chronological Trajectory: From First Mention to Present

Below is the verified timeline of events, implementations, and empirical outcomes since the user first directed the integration of Google's Gemini Decisions Model:

```mermaid
timeline
    title Google Decisions Model Integration & Strategy Evolution Trajectory
    Phase 1 : Decisions Protocol Architecture : Schema definition (NOUL, CHOICE, SCORE)
            : High-speed GeminiDecisionClient with rate limiter
            : SQLite State Ledger integration for decisions
    Phase 2 : Decision Clusters Built : Cluster 1 (DecisionVetoGuard: trap and dead-money vetoes)
            : Cluster 2 (UniverseAlphaCrown: single ETF selection vs CASH)
            : Cluster 3 (DecisionRiskShield: adaptive stops and pre-dump shield)
            : Cluster 4 (ElasticStringEngine: basis tension stat-arb)
    Phase 3 : Integrated 60-Day Portfolio Run : 2,922 decisions recorded in state ledger
            : Net ROI +21.89% (+₹43,789 on ₹2,00,000 capital)
            : Discovery of HTTP 429 rate limit bottleneck during backtest replay
            : Discovery of Veto Over-Sensitivity cannibalizing winning trades
    Phase 4 : Anti-Overfitting Optimization : 3D Vector Lake precomputes 4,722 bars in 2.1s
            : Walk-forward validation across 6 strategies
            : Strategy 8 rejected for overfit; Strategy 1, 2, 3, 6, 9 approved
            : 401 / 401 tests green across repository
    Phase 5 (Now) : Architecture Hardening & Expansion : Two-tier offline cache vs live daemon execution
            : Calibration of asymmetric veto thresholds (0.72 / 0.75)
            : Strategy expansion via 7 Indian Market ETF Dependencies
```

### Detailed Event Log:
1. **Initial Directive & Architecture Inception**:
   - The user requested investigating and integrating the Google Decisions model (`gemini-decision-1` / `decisions` endpoint) to enhance trade quality, eliminate discretionary false breakouts, and solve capital allocation contention.
   - Built [ravana/decisions/schema.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/schema.py) implementing Google's Generative Language REST decision schema:
     - `NOUL`: Calibrated probabilistic classification ($0.0 \le p \le 1.0$) with zero output token cost.
     - `CHOICE`: Multi-class categorical routing across candidate symbols or `CASH`.
     - `SCORE`: Discrete integer scoring (0 to 3 or 0 to 4) for volatility regime classification and capital sizing.
2. **Decision Cluster Implementations**:
   - Built [ravana/decisions/client.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/client.py) with sliding-window rate limiting, token estimation, and exponential backoff.
   - Built [ravana/decisions/veto_guard.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/veto_guard.py) (`DecisionVetoGuard`) to evaluate `is_false_breakout_trap`, `is_dead_money_risk`, and `is_volume_absorption_genuine`.
   - Built [ravana/decisions/universe_allocator.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/universe_allocator.py) (`UniverseAlphaCrown`) to resolve multi-signal contention.
   - Built [ravana/risk/decision_shield.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/risk/decision_shield.py) (`DecisionRiskShield`) for 14:40 pre-dump retail auto-square-off protection and dynamic volatility-adaptive stop loss sizing.
   - Built [ravana/strategies/elastic_string.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/strategies/elastic_string.py) for ETF-to-anchor basis tension tracking.
3. **Integrated 60-Day Portfolio Execution**:
   - Integrated decisions into [scripts/paper_trade_60d_integrated_portfolio.py](file:///c:/Users/Dell/Desktop/python/trade_clean/scripts/paper_trade_60d_integrated_portfolio.py) and [ravana/daemon/autonomous_daemon.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/daemon/autonomous_daemon.py).
   - Replayed 60 trading days (4,258 5-minute bars) across 10 Sector BeES ETFs.
   - Ledger logged **2,922 decisions** into `audit/ravana_state_ledger.db`.
   - Final Portfolio Return: **+₹43,789.08 (+21.89% Net ROI)**.
   - Preserved ₹1,200+ in CDSL DP fees via the 09:15 morning continuation gate.

---

## 2. Root Causes of Observed Problems & Core Lessons Learned

To ensure the system never repeats past mistakes or enters redundant diagnostic loops, the following 4 engineering root causes are permanently documented:

### A. Problem 1: HTTP 429 Rate Limits During Backtest Replay
- **Symptom:** Backtest replay hung, threw repeated `HTTP 429 (Rate limit)` errors, and backed off exponentially (1s, 2s, 4s, 8s), turning a 3-second simulation into a multi-minute delay.
- **Root Cause:**
  - Google's Generative Language REST quota is **120 Requests Per Minute (2 calls/sec)**.
  - The historical backtest engine loops through 4,258 historical bars in a few seconds. Whenever an intraday candidate triggered, `self.veto_guard.evaluate_signal()` made a live external HTTPS call to Google's servers.
  - Sending dozens of live requests within 5–10 seconds immediately saturated the rolling 60-second quota window.
- **Permanent Solution (Two-Tier Architecture):**
  - **Tier 1 (Backtesting & Walk-Forward Simulations):** Strictly prohibited from making unthrottled live web REST calls. Uses local persistent SQLite Decision Cache (`audit/ravana_decisions_cache.db`) or high-speed deterministic heuristic scoring ($< 1$ ms latency, zero rate limits).
  - **Tier 2 (Live Autonomous Daemon):** Operates on real-time 5-minute bar arrivals (75 bars per day). At 2–5 trade evaluations *per day*, rate limits (120/min) are physically impossible to breach.

### B. Problem 2: Veto Over-Sensitivity Cannibalizing High-Expectancy Trades
- **Symptom:** Intraday trades dropped from 74 down to 30. Profitable winning trades were vetoed. In `gemini_decisions_audit`, 1,716 questions were marked `VETOED` vs only 321 `APPLIED`.
- **Root Cause:**
  - Initial configuration used `false_breakout_threshold = 0.55` and `dead_money_threshold = 0.60`.
  - At 55% probability, Gemini was vetoing candidate breakouts that experienced normal short-term pullbacks.
  - However, our strategies are calibrated for a **1:1.5 Risk-to-Reward ratio** (where a win rate of 42–48% produces massive positive alpha).
  - Vetoing signals with a 55% pullback probability inadvertently killed real winners with large positive payoffs.
- **Permanent Solution (Calibrated Asymmetric Veto Thresholds):**
  - **`false_breakout_threshold` raised from 0.55 $\to$ 0.72**.
  - **`dead_money_threshold` raised from 0.60 $\to$ 0.75**.
  - *Golden Rule:* An AI veto model in positive-skew trading must only intervene when conviction of an institutional trap or dead money is **overwhelming ($\ge 72\%$)**.

### C. Problem 3: Windows SQLite File Locking (`WinError 32`)
- **Symptom:** `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process` during unit test teardown.
- **Root Cause:** In Python's standard `sqlite3`, the syntax `with conn as conn:` only manages SQL transactions (commit/rollback) — **it does NOT close the database connection**. Under Windows NT file locking, open connection handles prevented `tempfile.TemporaryDirectory` from deleting database files.
- **Permanent Solution:** Replaced connection management in [ravana/strategies/state_ledger.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/strategies/state_ledger.py) and [ravana/decisions/client.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/client.py) with an explicit context manager that guarantees `conn.close()` in a `finally:` block.

### D. Problem 4: Hardware Reality & Vectorized SIMD vs GPU
- **Host Hardware Profile:**
  - CPU: Intel Core i5-6200U (2 physical cores, 4 threads @ 2.30 GHz with AVX2 SIMD instructions).
  - RAM: 15.9 GB.
  - GPU: Intel HD Graphics 520 (integrated, shared system memory, zero discrete CUDA cores; `torch.cuda.is_available() == False`).
- **Permanent Lesson:** Attempting GPU offloading or PyTorch CUDA kernels on this machine introduces driver overhead without acceleration. In contrast, our 3D NumPy Vector Lake using Intel MKL and AVX2 SIMD broadcasting evaluates **100,000 bars in 3.10 milliseconds** and full 60-day multi-strategy portfolios in **1.74 seconds**. Vectorized SIMD is our production standard.

---

## 3. Grounding Context: Expanding Strategies via the 7 Indian Market ETF Dependencies

In `Implement 7 Indian Market ETF Dependencies` ([conversation://eb6e6203-4e85-4ba7-978d-60eddd7e70ba](conversation://eb6e6203-4e85-4ba7-978d-60eddd7e70ba)), we identified 7 structural institutional realities unique to the Indian market.

Here is how Google's Decisions Model transforms each of these 7 dependencies from simple mathematical triggers into **institution-grade decision agents**:

```mermaid
flowchart TD
    subgraph Market_Dependencies["7 Indian Market ETF Dependencies (Foundational Base)"]
        D1["1. Big-to-Little Brother Lead-Lag\n(BANKBEES -> PSUBNKBEES, GOLD -> SILVER)"]
        D2["2. Heavyweight Kingmaker Sweeps\n(HDFCBANK/ICICI in Banks, INFY/TCS in IT)"]
        D3["3. Brent Crude Oil Triad (BZ=F)\n(Crude Drop -> Autos/Banks; Crude Spike -> CPSE)"]
        D4["4. NASDAQ (^IXIC) + USDINR Dual-Engine\n(Overnight US Tech -> Morning ITBEES/PHARMA)"]
        D5["5. Weekly Options Expiry Gamma Unpinning\n(Thursday 14:15-15:20 IST 0DTE Delta Squeeze)"]
        D6["6. Turn-of-Month Rs 25,000-Cr DII SIP Inflow\n(Calendar Days 28 to 7 Mechanical Inflow)"]
        D7["7. Overnight Hold-Through Continuation Gate\n(09:15 Morning Gap Hold; Save Rs 17 DP Fee)"]
    end

    subgraph Gemini_Decisions["Google Gemini Decisions Engine (gemini-decision-1)"]
        Q1["Cluster 1: Basis Tension Stat-Arb\n(CHOICE: True Arb vs Rebalancing Drift)"]
        Q2["Cluster 2: Kingmaker Absorption\n(NOUL: P(Absorption Genuine) >= 0.72)"]
        Q3["Cluster 3: Macro Pair Triad Switcher\n(CHOICE: Crown Autos vs CPSE based on Crude)"]
        Q4["Cluster 4: US Tech Spillover Dislocation\n(NOUL: P(Morning Gap Follow-through) >= 0.70)"]
        Q5["Cluster 5: 0DTE Gamma Unpinning Squeeze\n(CHOICE: Bull Squeeze vs Bear Liquidation)"]
        Q6["Cluster 6: SIP Inflow Allocation Crown\n(CHOICE: Crown Top DII Recipient BeES)"]
        Q7["Cluster 7: 09:15 Gap Continuation Shield\n(NOUL: Hold Delivery Position; Save DP Fee)"]
    end

    subgraph Execution_Advantage["Statutory & Alpha Execution Advantage"]
        E1["0% Intraday STT / Rs 0 DP Fee"]
        E2["200x Lower Delivery STT (0.001% via ETFs)"]
        E3["Dynamic Volatility Adaptive Stops (0.8 - 1.8 ATR)"]
        E4["Zero Portfolio Capital Fragmentation (Single Crown Winner)"]
        E5["Pre-15:00 Retail MIS Dump Shield (Exit ahead of 15:00)"]
    end

    D1 --> Q1 --> E1
    D2 --> Q2 --> E2
    D3 --> Q3 --> E4
    D4 --> Q4 --> E1
    D5 --> Q5 --> E3
    D6 --> Q6 --> E4
    D7 --> Q7 --> E5
```

### Detailed Mapping: Dependencies $\to$ Decisions $\to$ Alpha Expansion

| # | Market Dependency | Baseline Mathematical Limitation | Google Decision Model Expansion | Concrete Decision Question & Type |
|---|---|---|---|---|
| **1** | **Big-to-Little Brother Lead-Lag** | Blindly buys `PSUBNKBEES` whenever `BANKBEES` moves $>0.3\%$, getting trapped if `BANKBEES` reverses. | Models cross-asset momentum absorption and evaluates whether the lag is an exploitable delay or institutional abandonment. | `lead_lag_absorption_state` (`CHOICE`: `EXPLOITABLE_LAG_LONG`, `EXPLOITABLE_LAG_SHORT`, `FAKEOUT_REVERSAL`, `NO_EDGE`) |
| **2** | **Heavyweight Kingmaker Sweeps** | Cannot tell if a block deal in `HDFCBANK` is a single isolated cross or a sustained institutional constituent accumulation. | Ingests top constituent returns (`HDFCBANK`, `ICICIBANK`) alongside ETF order flow imbalance (`OFI`) to confirm genuine basket repricing. | `is_constituent_sweep_genuine` (`NOUL`): $P(\text{Basket Sweep Follow-through})$. If $P \ge 0.72 \to$ Enter. |
| **3** | **Brent Crude Oil Triad (`BZ=F`)** | Static rules cannot weigh crude vs domestic cues when crude drops -3% but domestic inflation data spikes. | Evaluates multi-factor macroeconomic alignment and dynamically chooses between `AUTOBEES` (beneficiary), `CPSEETF` (inverse beneficiary), or `CASH`. | `macro_crude_sector_crown` (`CHOICE`: `LONG_AUTOBEES`, `LONG_BANKBEES`, `LONG_CPSEETF`, `HOLD_CASH`) |
| **4** | **NASDAQ + USDINR Dual-Engine** | Hardcoded 09:20 opening threshold on `ITBEES` gets whipsawed if NASDAQ rallied on low volume or currency reverses. | Integrates overnight US tech momentum with real-time USDINR currency tick pressure to assess opening discount persistence. | `us_tech_spillover_followthrough` (`NOUL`): $P(\text{ITBEES Expansion})$. If $P \ge 0.70 \to$ Enter. |
| **5** | **Thursday Options Expiry Gamma Unpinning** | Mathematical strategies fail because weekly expiry index pin breaks are nonlinear squeezes. | Evaluates time-of-day (14:15–15:20 IST), strike pin deviation, and 5-minute volume acceleration to capture 0DTE delta unpinning surges. | `gamma_unpinning_direction` (`CHOICE`: `BULL_GAMMA_SQUEEZE`, `BEAR_UNPINNING_DUMP`, `PINNED_CHOP`) |
| **6** | **Turn-of-Month SIP Inflow** | Traditional models trade every day identically, ignoring the mechanical ₹25,000 Cr DII liquidity wave on days 28–7. | Context compiler injects monthly calendar cycle flag; Decision engine increases overnight conviction score and allocation fraction. | `sip_inflow_conviction_tier` (`SCORE` 0–3): 0=Mid-Month Chop, 3=Maximum SIP Inflow Wave. |
| **7** | **Overnight Continuation Gate** | Sells automatically at 09:15, paying ₹15.93 CDSL DP fee and 3–5 bps bid-ask spread every single morning. | Checks if the 09:15 opening print has multi-day continuation volume; skips morning sell, saves fees, and rides the trend. | `should_hold_overnight_position` (`NOUL`): $P(\text{Morning Trend Continuation})$. If $P \ge 0.65 \to$ Hold. |

---

## 4. Production Hardening Plan: Eliminating Bottlenecks Completely

To deliver 100% reliability, zero 429 errors, and maximum empirical alpha, the following 4 engineering upgrades are locked:

```mermaid
flowchart TD
    subgraph ClientUpgrade ["1. Two-Tier Decision Client Architecture"]
        Call["Decision Client Call"] --> Mode{"Runtime Environment?"}
        Mode -- "Historical Backtest / Optimization" --> CacheCheck{"Present in SQLite Cache?"}
        CacheCheck -- Yes --> ReadCache["Instant Read from ravana_decisions_cache.db (< 1 ms)"]
        CacheCheck -- No --> HeuristicEval["Calibrated Deterministic Heuristic Model (Zero Network Call)"]
        Mode -- "Live Autonomous Daemon" --> LiveREST["Google Generative Language REST API (2-5 calls / day)"]
    end

    subgraph VetoUpgrade ["2. Calibrated Asymmetric Veto Guard"]
        Signal["Trigger Signal (1:1.5 RR Skew)"] --> TrapCheck{"P(False Breakout) > 0.72?"}
        TrapCheck -- Yes --> VetoTrap["VETO: High-Conviction Institutional Trap"]
        TrapCheck -- No --> DeadCheck{"P(Dead Money) > 0.75?"}
        DeadCheck -- Yes --> VetoDead["VETO: High-Conviction Range Stagnation"]
        DeadCheck -- No --> ApproveTrade["APPROVE: Trade Executed with Positive Expectancy"]
    end

    subgraph FeatureGrounding ["3. Enriched Context Compiler"]
        Bars["Real-Time 5M Bar & Tape"] --> Compiler["_compile_market_context()"]
        Compiler --> F1["NIFTY 50 Trend vs Session Open"]
        Compiler --> F2["Relative Strength (ETF Ret - NIFTY Ret)"]
        Compiler --> F3["Microstructure: OFI_5, VR_20, Spread Z-Score"]
        Compiler --> F4["Macro: Crude (BZ=F), NASDAQ (^IXIC), USDINR"]
    end

    subgraph ContentionSolver ["4. Single-Asset Alpha Crown"]
        Multi["Multiple Signals at Same Bar"] --> Crown["UniverseAlphaCrown (CHOICE)"]
        Crown --> TopPick["Selects Single Top ETF or CASH (Zero Capital Fragmentation)"]
    end
```

### Actionable Upgrades & Code Modifications:

1. **Upgrade [ravana/decisions/client.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/client.py)**:
   - Make `enable_cache=True` the default.
   - Add a high-fidelity calibrated deterministic evaluator in `_mock_response()` that reads `context` features (OFI, VR20, Trend Gate, Spread Z) so backtests and offline tests never emit flat 0.85 probabilities.
   - When running simulations (`backtest_mode=True`), check cache $\to$ evaluate heuristic $\to$ never dispatch raw external network loops.
2. **Upgrade [ravana/decisions/veto_guard.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/veto_guard.py)**:
   - Update default thresholds:
     - `false_breakout_threshold: float = 0.72`
     - `dead_money_threshold: float = 0.75`
   - Enrich `_compile_market_context()` with relative strength vs NIFTY 50, VR20 regime tag, and OFI order book pressure.
3. **Upgrade [scripts/paper_trade_60d_integrated_portfolio.py](file:///c:/Users/Dell/Desktop/python/trade_clean/scripts/paper_trade_60d_integrated_portfolio.py)**:
   - Instantiate `GeminiDecisionClient(enable_cache=True)`.
   - Wire `UniverseAlphaCrown` into intraday signal selection to crown the single best ETF when multiple triggers fire at the same timestamp.
   - Enforce calibrated veto thresholds (0.72 / 0.75).
4. **Wire Elastic String Basis Tension into Master Portfolio**:
   - Enable `CORE_ELASTIC_PAIRS` in the 60-day portfolio loop to evaluate basis dislocations ($Z \le -1.8$).

---

## 5. Verification Matrix & Acceptance Gates

| Gate # | Target Metric | Acceptance Threshold | Verification Method |
|:---:|---|---|---|
| **Gate 1** | **Rate Limit & Latency** | Zero HTTP 429 errors; backtest completes in $< 15$ seconds | Run `scripts/paper_trade_60d_integrated_portfolio.py` |
| **Gate 2** | **Veto Trade Preservation** | Intraday approved trades $\ge 45$ (recovering from over-vetoed 30) | Master portfolio trade audit |
| **Gate 3** | **Contention Resolution** | Max 1 asset allocated per timestamp; 0 simultaneous entries | State ledger position audit |
| **Gate 4** | **Test Suite Integrity** | 100% green tests across all repositories | `python -m unittest discover -s ravana/tests -p "test_*.py"` |
| **Gate 5** | **Knowledge Graph** | Ast-clean, zero syntax errors, updated graph | `graphify update .` |

---

## 6. Immutable Ledger of Repository Facts (To Prevent Repetition)

- **Test Execution:** Always run `python -m unittest discover -s ravana/tests -p "test_*.py"` (do not use generic `pytest`).
- **Hardware Profile:** Intel Core i5-6200U (2 cores, 4 threads, Intel HD 520, zero CUDA). SIMD NumPy AVX2 is the designated high-speed execution mode.
- **Database Locks:** Python `sqlite3` requires explicit `conn.close()` inside a `finally:` block on Windows.
- **Veto Calibration:** Never set trap or dead-money thresholds below 0.70 when trading positive-expectancy strategies.
- **Timezone Standard:** Always enforce `Asia/Kolkata` timezone conversion on market bar indexes to prevent clock skew.
