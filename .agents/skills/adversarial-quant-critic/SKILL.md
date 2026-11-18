---
name: adversarial-quant-critic
description: Institutional red-teaming, stress-testing, and anti-overfitting protocol for quantitative trading systems. Use when evaluating alpha signals, backtest realism, execution slippage, or systemic failure states.
---

# Adversarial Quantitative Critic Protocol (DARPA / Red-Team Standard)

This skill provides the systematic methodology for the `@adversarial-critic` to stress-test trading logic, alpha formulations, and execution routers before any capital is committed.

## 1. Anti-Overfitting & Curve-Fitting Verification

When evaluating an alpha formula or backtested strategy:
1. **Deflated Sharpe Ratio (DSR) Check:**
   - Do not trust raw Sharpe ratios $>3.0$ without checking trial count.
   - If 100 hypotheses were tested and the best has Sharpe 3.2, compute the expected maximum Sharpe under null hypothesis.
2. **Out-of-Sample Regime Validation:**
   - Strategy MUST be tested on both high-volatility regime (INDIAVIX $>18$) and narrow sideways chop (VR20 $<0.75$).
   - If performance is positive only during trending days, reject or downgrade to conditional sub-trigger.
3. **Parameter Sensitivity Shakeout:**
   - Shift moving averages / ATR multiples by $\pm 20\%$. If returns collapse, the model is curve-fit and rejected.

## 2. Friction & Microstructure Reality Gate

Never permit a backtest or paper simulation to assume frictionless fills:
1. **Exchange Delivery Charges:** Every equity delivery sell order MUST subtract ₹13.50 + 18% GST (₹15.93) to ₹16.58 DP charges.
2. **Execution Slippage Hurdle:** All limit orders executed in paper mode must deduct at least 1.6 bps for high-liquid BeES ETFs and 3.5 bps for mid-cap sector ETFs.
3. **STT & Turnover Taxes:** Ensure round-trip statutory costs are integrated into net NAV.

## 3. Black Swan & Failure-State Injection

The critic must verify what happens when external dependencies fail:
1. **Shoonya Socket Dropout:** Does the watchdog reconnect within 300ms without dropping orders?
2. **TOTP Epoch Boundary:** Does the router pause order submission if TOTP expiration is $<3$ seconds away?
3. **Gap-Down Open (-3% to -7%):** Does the trailing stop execute at market price or freeze on limit prices?

## 4. Evaluation Checklist

```markdown
- [ ] Overfitting Hurdle: Passed across $\ge 2$ independent market regimes
- [ ] Friction Reality: Statutory DP charges, STT, and slippage deducted
- [ ] Risk Asymmetry: Reward-to-Risk ratio $\ge 2.0:1$
- [ ] Failure Tolerance: Graceful handling of network dropouts and stale data
```
