# RAVANA v4.3: Master Architectural Plan for Low-Volatility Intraday Expansions & Autonomous Execution

> **Repository:** `c:\Users\Dell\Desktop\python\trade_clean`  
> **Status:** Architecture Blueprint & Implementation Specification | Awaiting Sign-Off  
> **Target Scope:** 1-Minute Resolution Integration, Dynamic Chop Mean-Reversion Mode, Adaptive Volatility Threshold Scaling, High-Frequency iNAV Micro-Spread Capture, Multi-Day Live Paper Trading, and VPS Deployment.

---

## 1. Executive Summary & Root-Cause Diagnosis

During the live autonomous paper trading session on 16 November 2026 (75 candle epochs, 09:15 to 15:30 IST), RAVANA demonstrated **flawless autonomous execution and 100% capital preservation**, closing at **₹200,037.96 (+₹37.96)** with overnight allocations in PSUBNKBEES (35%) and BANKBEES (22.5%).

However, the session yielded **0 intraday trades** due to structural parameter friction on an ultra-narrow 0.66% NIFTY range day:
1. **Variance Ratio Disqualification:** $\text{VR}_{20}$ hovered between 0.49 and 0.70 across all Sector ETFs. The primary momentum triggers (ORB, Sector PEAD Drift) strictly require $\text{VR}_{20} \ge 0.95$ to validate persistent trending momentum, correctly vetoing noise breakouts.
2. **Fixed Threshold Inelasticity:** The momentum hurdle was hardcoded to $|Z_{\text{mom}}| \ge 1.20$ and VWAP dislocation required $\ge 0.85\text{ ATR}$. On low-ATR days (where 5m candle amplitude is 0.08%–0.11%), price never deviates $0.85\text{ ATR}$ from VWAP.
3. **Temporal Granularity Blindness (5-Minute Smoothing):** 5-minute bars hide 8 to 12 intra-candle micro-oscillations of 10–15 bps between ETF market-maker spreads that occur on 1-minute bars.
4. **Absence of Dedicated Mean-Reversion Engine for Chop Regimes:** When $\text{VR}_{20} < 0.75$, the market is statistically confirmed to be mean-reverting. The system had no active strategy dedicated to fading Bollinger Band extremes back to VWAP with micro-targets (0.35 ATR).

```mermaid
flowchart TD
    subgraph Market_Input["1. Multi-Resolution Market Stream"]
        T5["5-Minute Candlestick Tape\n• Regime Filter\n• Macro Trend Gate\n• Session VWAP"]
        T1["1-Minute Candlestick Tape\n• Microstructure Oscillations\n• Fast Order Flow (OFI_1)\n• Bollinger Extremes (+/- 2.0 sigma)"]
    end

    subgraph Dynamic_Regime["2. Dynamic Regime & Volatility Scaler"]
        VR{"Variance Ratio VR_20"}
        ATR{"ATR Ratio = Today ATR / 20D Median ATR"}
        TrendMode["TREND EXPANSION MODE (VR >= 0.85)\n• Two-Way ORB\n• Two-Way Sector Drift (PEAD)\n• Adaptive Z-Hurdle (1.20 * ATR_Ratio)"]
        ChopMode["CHOP MEAN-REVERSION MODE (VR < 0.75)\n• Fade BB Extremes (+/- 1.8 sigma)\n• Target: Session VWAP\n• Micro Stop: 0.40 ATR | Target: 0.50 ATR"]
    end

    subgraph Decision_Shield["3. gemini-decision-1 Defensive Veto Shield"]
        VetoCheck{"Veto Guard P(FalseBreakout) >= 0.72\nor P(DeadMoney) >= 0.75"}
        Block["BLOCK TRADE (Preserve Capital)"]
        Pass["APPROVE EXECUTION"]
    end

    subgraph Execution_Router["4. Turnkey Paper Router & Compounding"]
        Size["Dynamic Position Sizer (0.20% Equity Risk)"]
        Ratchet["5-Stage Trailing Stop & Micro-Breakeven"]
        Overnight["15:15 IST All-Weather Overnight Allocator"]
    end

    T5 & T1 --> VR & ATR
    VR -- "VR >= 0.85" --> TrendMode
    VR -- "VR < 0.75" --> ChopMode
    TrendMode & ChopMode --> Decision_Shield
    VetoCheck -- "Veto" --> Block
    VetoCheck -- "Approve" --> Pass
    Pass --> Size --> Ratchet --> Overnight
```

---

## 2. The 4 Low-Volatility Intraday Expansions

### A. Sub-Minute & 1-Minute Candle Granularity Support
- **Data Engine Enhancement:** `fetch_today_all(interval="1m")` alongside `interval="5m"` in [scripts/fetch_today_live_data.py](file:///c:/Users/Dell/Desktop/python/trade_clean/scripts/fetch_today_live_data.py) and [ravana/daemon/data_feeder.py](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/daemon/data_feeder.py).
- **1-Minute Microstructure Metrics:**
  - Fast Order Flow Imbalance: $\text{OFI}_1 = \text{sign}(\Delta P) \cdot V$ smoothed over 3 bars.
  - Micro Bollinger Bands: 20-bar 1-minute SMA $\pm 2.0\sigma$.
  - Distance to Session VWAP (computed cumulatively across all 1m ticks from 09:15).

### B. Dynamic Range-Adaptive Chop Mean-Reversion Mode (`TwoWayChopMeanReversionTrigger`)
- **Regime Trigger Condition:** Active exclusively when $\text{VR}_{20} < 0.75$ and market session is between **09:35 and 11:15 IST** or **13:00 and 14:30 IST**.
- **Long Mean-Reversion Entry (Fade Lower Band):**
  - Price touches or breaches Lower Bollinger Band ($BB_{\%} \le 0.15$).
  - 1-minute candle closes green ($\text{Close}_1 > \text{Open}_1$) with bullish order flow ($\text{OFI}_1 > 0.05$).
  - Broad index NIFTY is not in a catastrophic breakdown ($\text{NIFTY} \ge \text{Open} - 0.25\%$).
  - Target: Session VWAP or Entry $+ 0.50\text{ ATR}$.
  - Stop Loss: Entry $- 0.40\text{ ATR}$ (strict micro-loss control).
- **Short Mean-Reversion Entry (Fade Upper Band):**
  - Price touches or breaches Upper Bollinger Band ($BB_{\%} \ge 0.85$).
  - 1-minute candle closes red ($\text{Close}_1 < \text{Open}_1$) with bearish order flow ($\text{OFI}_1 < -0.05$).
  - Broad index NIFTY is not in a parabolic breakout ($\text{NIFTY} \le \text{Open} + 0.25\%$).
  - Target: Session VWAP or Entry $- 0.50\text{ ATR}$.
  - Stop Loss: Entry $+ 0.40\text{ ATR}$.

### C. Dynamic Volatility Threshold Scaler (`AdaptiveThresholdScaler`)
- Calculate the **Opening Volatility Ratio:**
  $$\text{ATR}_{\text{ratio}} = \frac{\text{ATR}_{14}(\text{Today Open})}{\text{Median}(\text{ATR}_{14}(\text{Past 20 Days}))}$$
- Dynamically scale strategy entry hurdles and targets:
  - If $\text{ATR}_{\text{ratio}} < 0.80$ (Low Volatility Regime):
    - Scale Momentum Z-Score hurdle: $Z_{\text{hurdle}} = \max(0.85, 1.20 \times \text{ATR}_{\text{ratio}})$.
    - Scale Profit Target: $\text{Target}_{\text{mult}} = 0.75\text{ ATR}$ (down from $1.50\text{ ATR}$).
    - Initial Stop: $0.45\text{ ATR}$ (down from $0.75\text{ ATR}$), maintaining $\ge 1.67:1$ R:R.
  - If $\text{ATR}_{\text{ratio}} \ge 1.20$ (High Volatility Expansion Regime):
    - Expand hurdle to $1.35 Z$ and target to $2.0\text{ ATR}$.

### D. High-Frequency iNAV Arbitrage Quoting (`AdaptiveINAVTrigger`)
- On low-volatility days, Authorized Participant (AP) basket creations and redemptions compress the tracking spread from 15–20 bps down to 8–12 bps.
- Adaptive threshold: On days when $\text{ATR}_{\text{ratio}} < 0.80$, reduce iNAV deviation threshold from $15\text{ bps}$ to $10\text{ bps}$ and $Z_{\text{spread}}$ from $1.50$ to $1.15$ on `NIFTYBEES` and `BANKBEES`.

---

## 2. Implementation Plan & File Modifications

| Component | Target File | Nature of Changes |
| :--- | :--- | :--- |
| **1. Dynamic Chop Mean-Reversion Trigger** | `ravana/strategies/two_way_chop_reversion.py` | New module implementing `TwoWayChopMeanReversionTrigger` with $BB_{\%}$, $\text{VR}_{20} < 0.75$, 1m micro-confirmations, and micro R:R targets. |
| **2. Dynamic Volatility Scaler** | `ravana/strategies/volatility_scaler.py` | New module implementing `AdaptiveThresholdScaler` to scale momentum hurdles and targets dynamically by daily opening ATR ratio. |
| **3. Multi-Timeframe Ingestion** | `scripts/fetch_today_live_data.py` | Add `fetch_today_1m_all()` and integrate 1m resolution caching alongside 5m bars. |
| **4. Live Paper Daemon Integration** | `scripts/run_live_autonomous_paper_daemon.py` | Integrate `TwoWayChopMeanReversionTrigger`, `AdaptiveThresholdScaler`, and 1m micro-checks into `_process_bar_epoch()`. |
| **5. Unit Test Suite (TDD)** | `ravana/tests/test_two_way_chop_reversion.py` & `test_volatility_scaler.py` | Full test coverage for new triggers, boundary conditions, and parameter scaling. Target 100% green across all 401+ tests. |

---

## 4. Multi-Day Paper Trading & VPS Cloud Deployment

1. **Local Multi-Day Verification:**
   - Execute historical replay across varying market regimes (high trend day, moderate day, and low-ATR chop day like 16 Nov 2026).
   - Validate that Chop Mean-Reversion captures 10–25 bps gains on low-ATR days without compromising capital on trending days.
2. **Cloud Deployment on Ubuntu VPS:**
   - Verify `scripts/deploy_vps.sh` and `systemd/ravana.service`.
   - Setup logrotate, automatic restarts (`Restart=always`, `RestartSec=10`), and headless systemd watchdog.
   - Configure secrets via `.env`.

---

## 5. Verification Gate (Empirical Proof Checklist)

1. [ ] **Unit Tests:** Run all 401 existing tests + new unit tests via Python 3.11 runner; ensure 0 failures and 0 errors.
2. [ ] **Chop Simulation:** Run replay on 16 Nov 2026 data to confirm Chop Mean-Reversion triggers generate positive intraday alpha on low-ATR conditions.
3. [ ] **Veto Guard Compatibility:** Confirm `DecisionVetoGuard` correctly evaluates and scores the new mean-reversion setups.
4. [ ] **Git Commit & Audit:** Clean atomic commits for all additions.
