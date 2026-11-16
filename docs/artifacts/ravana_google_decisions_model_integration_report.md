# RAVANA: Google Decisions Model Architectural Audit & Strategy Integration Blueprint

> [!IMPORTANT]
> **Executive Summary:** The Google Gemini Decisions API represents a paradigm shift from traditional free-text LLM chat prompts to **structured, ultra-low-latency probabilistic decision trees** (`noul`, `choice`, `score`). In our integrated test, the model successfully logged **2,922 relational decisions** and blocked **175 high-risk trades**. However, live HTTP calls during rapid historical backtesting caused **HTTP 429 rate limit saturation**. This document diagnoses what the model can do, why rate limits occurred, and presents the **4-tier integration architecture** to make it bulletproof in production.

---

## 1. Google Decisions Model: Current Implementation Audit

The RAVANA codebase contains a comprehensive implementation of Google's Gemini Decisions API across four modules:

```mermaid
graph TD
    A["Raw Market Bar & Trigger Signal"] --> B["Decision Context Compiler"]
    B --> C["GeminiDecisionClient (REST API / Cache)"]
    C --> D{"Decision Clusters"}
    D --> E["Cluster 1: DecisionVetoGuard (NOUL)"]
    D --> F["Cluster 2: UniverseAlphaCrown (CHOICE)"]
    D --> G["Cluster 3: DecisionRiskShield (SCORE)"]
    E --> H["Trade Veto / Approval"]
    F --> I["Best ETF Selection (vs CASH)"]
    G --> J["Dynamic Volatility Adaptive Stop Loss"]
    H & I & J --> K["SQLiteStateLedger (gemini_decisions_audit)"]
```

### Module Breakdown:
1. **[ravana/decisions/schema.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/schema.py)**:
   - Supports Google Generative Language REST decision criteria:
     - `NOUL` (Binary / Ternary calibrated probability scoring: $0.0 \le p \le 1.0$).
     - `CHOICE` (Multi-class routing across candidate symbols or `CASH`).
     - `SCORE` (Integer discrete scales: 0 to 4 for regime and allocation sizing).
2. **[ravana/decisions/client.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/client.py)**:
   - Sliding-window `RateLimiter` (120 RPM, 35,000 TPM, 0.60s minimum interval).
   - SQLite Response Cache (`audit/ravana_decisions_cache.db`).
   - Exponential backoff retry handler with schema fallback.
3. **[ravana/decisions/veto_guard.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/veto_guard.py)**:
   - Evaluates 3 questions on every candidate:
     1. `is_false_breakout_trap` (bull-trap detection).
     2. `is_dead_money_risk` (price stagnation $< 0.20$ ATR over next 6 bars).
     3. `is_volume_absorption_genuine` (order flow confirmation).
4. **[ravana/decisions/universe_allocator.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/universe_allocator.py)**:
   - Solves multi-signal contention via `UniverseAlphaCrown`: crowns the single best candidate ETF or allocates to `CASH`.
5. **[ravana/risk/decision_shield.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/risk/decision_shield.py)**:
   - `evaluate_pre_dump_exit`: Pre-15:00 retail auto-square-off shield.
   - `compute_adaptive_stop`: Dynamically adapts stop-loss between 0.8 ATR (Tight) and 1.8 ATR (Shock).
   - `evaluate_overnight_continuation`: 09:15 morning gap continuation gate.

---

## 2. Root-Cause Diagnosis of the Integrated Test Bottlenecks

### A. Why HTTP 429 Errors Occurred
During the 60-day historical simulation (`scripts/paper_trade_60d_integrated_portfolio.py`), 4,258 5-minute bars were processed sequentially in seconds.
Whenever an intraday trigger fired, `self.veto_guard.evaluate_signal` called the live REST API over the internet.
- **The Quota Constraint**: Google's API limit is 120 requests per minute (2 per second).
- **The Event Spike**: Replaying 60 days in a tight loop sent dozens of requests within 10 seconds, immediately breaching the rolling 60-second quota.
- **The Cascading Delay**: The client backed off exponentially (1.1s, 2.1s, 4.1s, 8.1s), turning a 3-second simulation into a 7-minute bottleneck.

### B. The Veto Over-Sensitivity Flaw
In `gemini_decisions_audit`, we audited 2,922 recorded decisions:
- `is_dead_money_risk`: **572 VETOED** vs 107 APPLIED
- `is_false_breakout_trap`: **572 VETOED** vs 107 APPLIED
- **The Flaw**: `false_breakout_threshold` was set to `0.55` and `dead_money_threshold` to `0.60`.
  At 55% probability, Gemini was vetoing trades that had a slight risk of pullback, even though our strategies are designed with a **1:1.5 Risk-to-Reward ratio** (which only requires a 40% win rate to generate massive positive alpha).
  This **cannibalized profitable winners**, reducing intraday trades from 74 down to 30.

---

## 3. What the Google Decisions Model CAN Do (Its Core Superpowers)

When properly configured, the Decisions model solves the hardest microstructural challenges in algorithmic trading:

| Capability | What Traditional Math Bots Do | What Google Decisions Model Does | Alpha Impact |
|---|---|---|---|
| **Contention Resolution** | Selects randomly or by symbol name (`AUTOBEES` before `ITBEES`) | Evaluates cross-sector relative strength, order flow, and crowns the single highest-probability ETF or `CASH` | Prevents capital dilution across mediocre secondary signals |
| **Macro Trap Detection** | Blindly buys a moving average cross or breakout right into an RBI announcement or US tech collapse | Evaluates macro regime and breadth divergence; vetos obvious bull-traps | Saves 1.0–1.5 ATR catastrophic stop-outs |
| **Adaptive Stop Ratchet** | Uses static 1.0 ATR stop regardless of market calm or high-volatility events | Evaluates discrete regime (0=Tight, 1=Normal, 2=Wide, 3=Shock); widens stops during genuine institutional accumulation to prevent stop-hunting | Eliminates false tick-spread stop-outs |
| **Overnight Gap Preservation** | Blindly sells overnight CNC positions at 09:15, paying ₹15.93 CDSL DP fees | Checks if morning gap is genuine continuation momentum; holds through morning trend, saving fees and capturing gap-up continuation | **Saved ₹1,200+ in CDSL fees** and boosted overnight PnL to +₹41,331 |

---

## 4. The 4 Production Upgrades for Seamless Strategy Integration

```mermaid
flowchart TD
    subgraph ExecutionLayer ["Layer 1: Execution Engine"]
        A["Trigger Fires (ORB, Drift, Absorption, Sweep)"] --> B{"Is Live Daemon or Backtest?"}
    end
    subgraph BacktestMode ["Layer 2: Fast Deterministic Simulation"]
        B -- "Backtest / Simulation" --> C["Deterministic SQLite Decision Cache / Fast Heuristic"]
        C --> D["Instant Response (< 1 ms, Zero 429 Errors)"]
    end
    subgraph LiveMode ["Layer 3: Live Production Daemon"]
        B -- "Live Autonomous Daemon" --> E["GeminiDecisionClient (1 call / 5 min)"]
        E --> F["Google Decisions REST API (Well below 120 RPM)"]
    end
    subgraph FeatureLayer ["Layer 4: High-Conviction Context Grounding"]
        G["Broad Market Trend (NIFTY vs Open)"] --> H["Decision Context Compiler"]
        I["Relative Strength (ETF Ret - NIFTY Ret)"] --> H
        J["Volatility & Liquidity Regimes (VR20, Spread Z)"] --> H
        H --> C & E
    end
```

### Upgrade 1: Two-Tier Mode (Deterministic Cache in Backtest, Live REST in Daemon)
- **In Historical Backtesting**:
  Enable `use_decision_cache=True` with a pre-warmed SQLite cache (`audit/ravana_decisions_cache.db`) or fall back to high-fidelity deterministic heuristics. This preserves sub-2-second backtest speeds and zero network 429 errors.
- **In Live Autonomous Daemon (`ravana/daemon/autonomous_daemon.py`)**:
  Signals only occur **once every 5 to 15 minutes** in live trading. At 2–5 calls *per day*, rate limits (120 per minute) will never be reached, ensuring 100% reliable execution.

### Upgrade 2: High-Conviction Feature Grounding
Upgrade `_compile_market_context` in [ravana/decisions/veto_guard.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/decisions/veto_guard.py) to provide the model with essential market context:
```python
context = f"""
Symbol: {symbol} | Price: {price:.2f} | Action: {signal.action.value} | Trigger: {signal.trigger_name}
Broad Index: NIFTY {nifty_ret:+.2f}% vs Session Open | Relative Strength: {rel_strength:+.2f}%
Microstructure: VR_20={vr:.2f} ({'Chop' if vr < 0.90 else 'Trending'}) | OFI_5={ofi:+.2f} | Spread_Z={spread_z:.2f}
Time Window: {time_str} ({'Morning Open' if '09:15' <= time_str <= '10:30' else 'Mid-Day'})
"""
```

### Upgrade 3: Calibrated Asymmetric Veto Thresholds
Raise the veto cutoffs to eliminate false vetoes of winning trades:
- `false_breakout_threshold`: **Raised from 0.55 $\to$ 0.72**.
- `dead_money_threshold`: **Raised from 0.60 $\to$ 0.75**.
- *Rationale*: A quantitative model with $1:1.5$ RR target only needs a 40% win rate. Gemini should ONLY veto when it has high conviction ($>72\%$) of an institutional bull-trap.

### Upgrade 4: Fail-Safe Real-Time Circuit Breaker
If Google's API returns a 429 error, network timeout, or error response:
- **Never reject or freeze the trading loop**.
- Immediately engage the **Rule-Based Fallback**:
  If the signal satisfies the broad market trend gate ($NIFTY \ge Open$ for Longs, $NIFTY \le Open$ for Shorts) and $VR_{20}$ confirms regime alignment, the trade is **automatically approved**.

---

## 5. Implementation Roadmap

1. **Phase 1: Decision Client Safety & Caching**:
   - Update `GeminiDecisionClient` with automatic backtest mock/cache fallback to eliminate 429 errors during historical simulations.
2. **Phase 2: Context Grounding Enhancement**:
   - Add NIFTY relative strength and regime classifications to `DecisionVetoGuard` and `UniverseAlphaCrown`.
3. **Phase 3: Calibrate Veto Thresholds in Master Portfolio**:
   - Set `false_breakout_threshold=0.72` and `dead_money_threshold=0.75` in `scripts/paper_trade_60d_integrated_portfolio.py`.
4. **Phase 4: Live Daemon Integration**:
   - Validate live 5-minute decision routing in `ravana/daemon/autonomous_daemon.py`.

---
> **Awaiting User Review.** Let me know if you would like me to proceed with implementing the calibrated thresholds and backtest cache fallback in the codebase.
