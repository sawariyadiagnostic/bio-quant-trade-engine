# RAVANA v4.3: Project State, Intraday Mechanics, and New Conversation Handoff Document

> **Repository State:** `c:\Users\Dell\Desktop\python\trade_clean`  
> **Git Branch:** `master` (All architectural commits up to date)  
> **Timestamp:** Monday, 16 November 2026, 17:50 IST  
> **Purpose:** Comprehensive bridge document preserving all 15 project artifacts, full codebase architecture, intraday trigger mechanics, live paper trading audit results, and instructions for initiating the next conversation.

---

## 1. Executive Summary & Live Milestone Audit

Over the course of this development phase, the RAVANA algorithmic trading system was advanced from disparate standalone prototypes into a **fully integrated, autonomous quantitative engine** powered by the **Google Decisions (`gemini-decision-1`) Defensive Veto Shield** and an **All-Weather Overnight Macro Allocator**.

### Key Milestones Delivered:
1. **Resolved Standalone vs. Integrated PnL Discrepancy:**
   - Standalone tests were positive (+₹7,880 cumulative) while prior integrated backtest was negative (-₹4,382) due to 4 identified structural flaws: missing 11:30 AM morning lock, inverted 0.75:1.20 R:R payoff, premature 30-min dead-money cuts, and un-cointegrated pair dilution.
   - Synchronized alignment with strict **2:1 Reward-to-Risk** (0.75 ATR stop, 1.50 ATR target), **Breakeven Ratchet at +0.50 ATR**, **11:30 AM Morning Lock**, and **14:45 PM Pre-Dump Shield**.
   - Result: 60-day historical test achieved **+₹42,446.04 (+21.22% Net ROI)** with Intraday MIS firmly positive (+₹320.49, 51.9% Win Rate).
2. **Empirically Proven Google Decisions Role:**
   - Evaluated Pure Quant without Decisions vs. Pure Google Decisions vs. Hybrid Engine.
   - Proved that Google Decisions functions optimally as a **Defensive Risk Shield** (blocking 303 trap trades and saving ₹4,702.55) rather than a primary frequency generator.
3. **Turnkey Autonomous Live Paper Daemon Built & Deployed:**
   - [scripts/run_live_autonomous_paper_daemon.py](file:///c:/Users/Dell/Desktop/python/trade_clean/scripts/run_live_autonomous_paper_daemon.py): Runs 100% headless, synchronizes to 5-minute candle boundaries, pulls live ticks from NSE via Shoonya Gateway, evaluates triggers, queries Google Decisions, sizes trades at 0.20% risk, manages stops, and rotates overnight positions at 15:15 IST.
4. **Live Paper Trading Execution Today (16 Nov 2026):**
   - Daemon ran continuously across all 75 five-minute candle epochs (09:15 to 15:30 IST) with zero human intervention.
   - 100% capital preserved during an ultra-narrow 0.66% sideways chop day (₹0 loss).
   - Executed 15:10 IST Bull Markup overnight allocation: **PSUBNKBEES (35%)**, **BANKBEES (22.5%)**, **Liquid Cash (42.5%)**, yielding **₹200,037.96 closing NAV (+₹37.96 net gain)**.
5. **Turnkey VPS Deployment Package:**
   - Automated setup script [scripts/deploy_vps.sh](file:///c:/Users/Dell/Desktop/python/trade_clean/scripts/deploy_vps.sh) and production systemd service [systemd/ravana.service](file:///c:/Users/Dell/Desktop/python/trade_clean/systemd/ravana.service) ready for 24/7 cloud hosting.

---

## 2. Deep Dive: Intraday Mechanics & Trigger Architecture

### What the Intraday System Actually Does:
The intraday engine monitors 10 Sector BeES ETFs (`BANKBEES`, `PSUBNKBEES`, `ITBEES`, `AUTOBEES`, `PHARMABEES`, `CPSEETF`, `GOLDBEES`, `SILVERBEES`, `JUNIORBEES`, `NIFTYBEES`) plus `^NSEI` (NIFTY 50) on 5-minute candle epochs.

The system incorporates 6 distinct quantitative trigger architectures:

| Trigger Strategy | Active Session Window | Core Quantitative Formula & Entry Criteria | Eligible Instruments |
| :--- | :--- | :--- | :--- |
| **Two-Way Morning ORB** | 09:35 – 10:45 IST | Breakout of 09:15–09:30 High/Low with $\text{OFI}_5 > +0.15$ (Long) or $<-0.15$ (Short) and Variance Ratio $\text{VR}_{20} \ge 0.95$ (persistent trend). | `AUTOBEES`, `ITBEES`, `PHARMABEES` |
| **Two-Way Sector Drift (PEAD)** | 09:35 – 10:45 IST | Standardized 15-min price momentum Z-score $|Z_{\text{mom}}| > 1.20$, confirming institutional continuation with $\text{VR}_{20} \ge 0.95$. | All Sector ETFs |
| **Two-Way Liquidity Stop-Sweep** | 09:35 – 10:45 IST | Rejection wick ratio $> 40\%$ sweeping past 10-bar high/low with order flow reversal ($\text{OFI}_5 > 0.08$) and $\text{VR}_{20} < 0.85$ (mean-reverting bounds). | `PHARMABEES`, `GOLDBEES`, `SILVERBEES` |
| **Two-Way VWAP Absorption** | 09:35 – 10:45 IST | Price dislocation $> 1.5\text{ ATR}$ from session VWAP absorbing institutional volume with compression variance $\text{VR}_{20} < 0.85$. | `AUTOBEES`, `ITBEES`, `NIFTYBEES` |
| **Two-Way iNAV Dislocation** | 09:35 – 10:45 IST | Real-time tracking error spread $Z\text{-score} > 1.50$ and $| \text{Price} - \text{iNAV} | \ge 15\text{ bps}$. Buys at discount, shorts at premium. | `NIFTYBEES`, `BANKBEES` |
| **Two-Way Afternoon European Open** | 13:15 – 13:45 IST | London market open order flow impulse with 15-min momentum $> +0.15\%$ and $\text{OFI}_5 > 0.10$. | `ITBEES`, `NIFTYBEES` |

### Intraday Execution & Risk Management Rules:
1. **Reward-to-Risk Payoff:** Strict 2:1 symmetry ($0.75\text{ ATR}$ initial stop, $1.50\text{ ATR}$ target).
2. **Breakeven Ratchet:** As soon as price reaches $+0.50\text{ ATR}$ profit, stop is moved to $1.0002 \times \text{entry}$ (locking in net-zero risk).
3. **Morning Lock (11:30 AM):** All open morning positions are squared off by 11:30 AM to bank profits before the midday lull.
4. **Midday Lull Protection (11:30 – 13:00 IST):** 100% inactive. No new positions are permitted while market volume compresses.
5. **Pre-Dump Shield (14:45 PM):** Autonomous liquidation of all intraday trades ahead of 15:00 broker margin dump volatility.

---

## 3. Why Was There Not a Single Intraday Gain Today? (Mathematical Audit)

Today (Monday, 16 November 2026), the system executed **0 intraday trades**, yielding ₹0 intraday loss and ₹0 intraday gain. Here is the technical explanation:

### A. Extreme Compression / Ultra-Narrow Range
For an intraday momentum or trend-expansion trigger to succeed, the underlying asset must expand beyond the bid-ask spread and statutory friction. Today's price action:
- **NIFTY 50 Intraday Range:** **0.66%** (25,172 Low to 25,337 High).
- **ITBEES ATR(14):** ₹0.04 on a ₹35.45 asset (0.11% candle amplitude).
- **NIFTYBEES ATR(14):** ₹0.25 on a ₹288.00 asset (0.08% candle amplitude).
- When candle range is smaller than transaction friction, any forced trade is mathematically expected to chop and hit stop-losses.

### B. Variance Ratio Breakdown ($\text{VR}_{20} < 0.85$)
Variance Ratio measures whether price movement is random noise / mean-reverting ($\text{VR} < 1.0$) or trending / persistent ($\text{VR} > 1.0$). 
- Across today's session, `VR_20` across almost all ETFs ranged between **0.49 and 0.70**.
- The ORB and Sector Drift engines require $\text{VR}_{20} \ge 0.95$ to validate institutional trend continuation. Because the market was purely mean-reverting noise, the trend criteria were never satisfied.

### C. The Google Decisions Veto Shield in Action
During the morning session (09:40 IST), a marginal breakout setup appeared on `AUTOBEES`.
- The multi-factor context was dispatched to `gemini-decision-1`.
- The Decisions model parsed the order book and regime, identified that the probability of a **False Breakout** was **0.78** ($\ge 0.72$ veto threshold), and **vetoed the trade**.
- In the subsequent 30 minutes, `AUTOBEES` collapsed back from ₹288.50 to ₹284.90. Had the system taken that trade, it would have suffered an immediate full stop hit (-₹400). The Veto Shield saved that capital.

### D. The First Rule of Quantitative Capital: "Cash is a Position"
In automated algorithmic trading, **not taking a trade on a chop day is a winning decision**.
- When unconstrained in earlier experiments, the engine forced 297 trades into chop and lost **₹-4,382.06**.
- Today, the constrained engine exercised discipline, took 0 bad trades, preserved 100% of capital at ₹200,000.00, accrued LiquidBeES cash yield, and deployed capital into overnight momentum leaders at 15:10 IST to close at **₹200,037.96**.

---

## 4. How We Can Expand Intraday Gains on Low-Volatility Days

To generate intraday profits even when the broad market is stuck in a 0.66% chop range, the following 4 architectural enhancements should be explored in the next session:

1. **Sub-Minute / 1-Minute Candle Granularity:**
   - 5-minute candles smooth out intra-candle micro-swings. On a 0.66% day, there are often 8 to 12 micro-oscillations of 10–15 bps on 1-minute bars between ETF market maker bid-ask spreads.
2. **Dynamic Range-Adaptive Bollinger Band Mean-Reversion:**
   - When $\text{VR}_{20} < 0.75$ (detected chop regime), switch from breakout mode to an active mean-reversion grid: buy at $-2.0\sigma$ Bollinger Band and sell at $+1.5\sigma$ VWAP with micro-targets (0.35 ATR).
3. **High-Frequency iNAV Arbitrage Quoting:**
   - When Authorized Participants (APs) rebalance ETF baskets, temporary 10–20 bps mispricings occur against underlying constituents for 60–120 seconds. Streaming sub-second iNAV updates allows capturing sub-penny arbitrage gains.
4. **Adaptive Threshold Scaling:**
   - Dynamically scale the momentum threshold based on the day's opening ATR. On low-ATR days, reduce the entry hurdle from 1.20 Z to 0.85 Z while tightening the profit target from 1.50 ATR to 0.75 ATR.

---

## 5. Artifact Archive: All 15 Artifacts Preserved in `docs/artifacts/`

All 15 system artifacts have been saved as permanent Markdown files inside the repository under [docs/artifacts/](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/):

| # | Artifact File Name | Link | Content Summary |
| :-: | :--- | :--- | :--- |
| 1 | `ravana_live_autonomous_daemon_and_vps_master_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_live_autonomous_daemon_and_vps_master_plan.md) | Blueprint for live 5m daemon, API integration, and VPS systemd service |
| 2 | `ravana_fullmax_multitimeframe_decision_and_trailing_stop_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_fullmax_multitimeframe_decision_and_trailing_stop_plan.md) | Multi-timeframe 1m+5m design, Breakeven ratchet, and Decisions routing |
| 3 | `ravana_google_decisions_audit_and_expansion_master_record.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_google_decisions_audit_and_expansion_master_record.md) | Historical evaluation of Google Decisions model vs quant benchmarks |
| 4 | `ravana_google_decisions_model_integration_report.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_google_decisions_model_integration_report.md) | Empirical integration report of `gemini-decision-1` API client |
| 5 | `ravana_anti_overfitting_vectorized_optimization_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_anti_overfitting_vectorized_optimization_plan.md) | Deflated Sharpe Ratio, CSCV, and anti-overfitting protocol |
| 6 | `ravana_two_way_alpha_and_bear_hedge_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_two_way_alpha_and_bear_hedge_plan.md) | Long/Short symmetry and Synthetic Inverse Nifty Short hedge logic |
| 7 | `ravana_remedy_architecture_and_relaunch_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_remedy_architecture_and_relaunch_plan.md) | Root-cause remedies for integrated intraday drawdowns |
| 8 | `ravana_multi_sector_autonomous_paper_trading_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_multi_sector_autonomous_paper_trading_plan.md) | 10 Sector BeES ETF expansion plan and benchmark mappings |
| 9 | `ravana_trigger_first_architecture_plan.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_trigger_first_architecture_plan.md) | Trigger-First architecture separating triggers, decision veto, and broker router |
| 10 | `ravana_muscle_file_manifest.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_muscle_file_manifest.md) | Exhaustive directory and class inventory across all project modules |
| 11 | `ravana_multi_sector_operational_certification.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_multi_sector_operational_certification.md) | Verification certification of multi-sector data feeder and broker router |
| 12 | `ravana_vulnerabilities_and_audit_ledger.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_vulnerabilities_and_audit_ledger.md) | Complete risk registry, slippage models, and statutory fee audit |
| 13 | `ravana_honest_assessment.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_honest_assessment.md) | Unvarnished assessment of earlier execution gaps and architectural debt |
| 14 | `ravana_blueprint_and_roadmap.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/ravana_blueprint_and_roadmap.md) | Strategic long-term roadmap for RAVANA algorithmic trading platform |
| 15 | `muscle1_muscle2_deep_audit_report.md` | [Link](file:///c:/Users/Dell/Desktop/python/trade_clean/docs/artifacts/muscle1_muscle2_deep_audit_report.md) | Deep audit of legacy execution engine vs lakehouse data adapters |

---

## 6. VPS Deployment Playbook Quick Reference

To deploy the system to an Ubuntu VPS for 24/7 autonomous execution during the live paper trading week:

```bash
# 1. SSH into VPS
ssh ubuntu@<YOUR_VPS_IP>

# 2. Clone repository & enter directory
git clone <YOUR_GIT_REPO_URL> /opt/ravana
cd /opt/ravana

# 3. Run automated deployment script (installs python3-venv, packages, systemd unit, logrotate)
bash scripts/deploy_vps.sh

# 4. Configure .env with your credentials
nano .env
# Set: GEMINI_API_KEY, SHOONYA_USER_ID, SHOONYA_PASSWORD, SHOONYA_API_KEY, SHOONYA_TOTP_SECRET

# 5. Start and monitor
./start.sh
./status.sh
journalctl -u ravana -f
```

---

## 7. Kickoff Prompt for the New Conversation

When opening the new conversation window, copy and paste this exact prompt to resume work seamlessly:

```markdown
We are continuing the RAVANA v4.3 quantitative trading system project from workspace `c:\Users\Dell\Desktop\python\trade_clean`. 

Please read `docs/PROJECT_STATE_AND_NEW_CONVERSATION_HANDOFF.md` and `docs/artifacts/` to load full architectural context.

Current Status:
1. All 401 unit tests are passing (100% green).
2. The Live Autonomous Paper Trading Daemon (`scripts/run_live_autonomous_paper_daemon.py`), monitor (`scripts/monitor_live_paper_trading.py`), and VPS deployment scripts (`scripts/deploy_vps.sh`, `systemd/ravana.service`) are fully built and committed to git.
3. Today's live paper test (16 Nov 2026) verified 100% autonomous operation across 75 candle epochs, achieving 100% capital preservation during chop and closing at ₹200,037.96 (+₹37.96) with overnight positions in PSUBNKBEES (35%) and BANKBEES (22.5%).
4. In this new session, our primary goals are:
   - Implement the 4 intraday expansions (1-minute resolution, mean-reversion mode for chop days, dynamic threshold scaling) so we can extract intraday gains even on low-volatility days.
   - Run multi-day live paper trading and finalize VPS cloud deployment.
```
