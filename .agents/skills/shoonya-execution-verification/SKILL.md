---
name: shoonya-execution-verification
description: Microstructure execution verification, TOTP synchronization, order idempotency, and session watchdog protocol for Finvasia Shoonya broker gateway. Use when implementing or testing broker order routing.
---

# Shoonya Microstructure & Execution Verification Protocol

This skill dictates execution integrity rules for trading on the National Stock Exchange (NSE) via Finvasia's Shoonya API.

## 1. TOTP Epoch Synchronization

1. **The 30-Second TOTP Window:**
   - TOTP codes rotate every 30 seconds. Submitting authentication requests at the 28–29s edge frequently causes invalid credential rejections.
   - **Protocol:** If time remaining in the current 30s epoch is $<3.0$ seconds, execution threads MUST sleep $3.15$ seconds and re-generate on the fresh epoch edge.

## 2. Idempotent Order Dispatch & Tokenization

1. **Replay Shielding:**
   - Every generated order must carry an idempotent tag (`ORD_{TICKER}_{EPOCH_HASH}`).
   - If an order is re-transmitted due to network latency, the gateway must verify whether the order number exists in `audit/shoonya_paper_orders.json` before sending to the broker.
2. **Order Status Lifecycle Reconciliation:**
   - An order is NOT filled upon HTTP 200 response.
   - Status polling must verify transition: `PENDING` $\to$ `OPEN` $\to$ `COMPLETE` (or `REJECTED` / `CANCELLED`).
   - Timeout on fill polling must not exceed 5.0 seconds before escalating to watchdog.

## 3. Statutory Friction & DP Charge Protection

1. **Overnight Continuation vs Intraday Flip:**
   - Exiting a delivery position and re-buying the same asset on the next epoch incurs an unnecessary ₹16.58 DP fee plus round-trip STT and broker friction.
   - **Continuation Rule:** If morning momentum signal $\ge 0.50$ on an overnight holding, trigger `Continuation Hold` and skip the morning exit to preserve DP charges.
