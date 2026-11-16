# RAVANA Multi-Sector Autonomous Paper Trading Architecture Plan (v4.1)

> **Classification:** Production Institutional Autonomous Paper Trading Architecture & Multi-Sector Execution Blueprint  
> **Target Universe:** 10 Core Sector & Index BEES ETFs + Paired Underlying Benchmark Indices  
> **Capital Budget:** ₹2,00,000 (₹2.00 Lakh INR)  
> **Operational Horizon:** Fully Autonomous Zero-Intervention Daemon (Paper & Live Exchange Ready)  
> **Date:** November 15, 2026  

---

## Executive Summary & Design Invariants

This plan establishes a **100% autonomous, multi-sector algorithmic execution daemon** capable of running without human supervision. It ingests and maintains live and historical data across the entire National Stock Exchange (NSE) Sector BEES universe, evaluates 10 structural market microstructure triggers across 3 regime buckets, manages capital with a strict ₹2,00,000 allocation budget, and protects capital via an intrabar 5-layer stop-loss shield and the 11:32 AM crash reboot reconciliation protocol.

```
+---------------------------------------------------------------------------------------------------+
|                                 RAVANA MULTI-SECTOR SOVEREIGN STACK                                |
+---------------------------------------------------------------------------------------------------+
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   [Tier 1: Multi-Sector Lakehouse]                               [Tier 2: Microstructure Feature Engine]
   10 Sector BEES + Benchmarks                                    36 Stationary Features + Variance Ratio
   (BANK, NIFTY, IT, AUTO, PHARMA,                                Order Flow Imbalance, Tracking Spreads,
    PSUBANK, JUNIOR, CPSE, GOLD, SILVER)                           Heavyweight Stock Synchrony Confirmations
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  ▼
                               [Tier 3: 10 Structural Triggers & Regimes]
                                  Bucket 1: Trend & Impulse (VR > 1.05)
                                  Bucket 2: Mean Reversion & Absorption (VR <= 1.05)
                                  Bucket 3: Macro TWAP & All-Weather Overnight
                                                  │
                                                  ▼
                               [Tier 4: StrategyRiskBridge (₹2 Lakh Capital)]
                                  * 0.25% Equity Risk-at-Stop (₹500 Max Loss / Trade)
                                  * 2.0x Max MIS Leverage | 0.98x CNC Cash Cap
                                  * HRP Sector Correlation Ceiling (Max 40% in Correlated Cluster)
                                  * Portfolio Contention Gating (Max 2 Concurrent Position Slots)
                                                  │
                                                  ▼
                               [Tier 5: 5-Layer Zero-Giveback Stop Shield]
                                  Layer 1: Thesis Micro-Stop (Index 3-bar low)
                                  Layer 2: +0.45 ATR Breakeven Ratchet (+2 bps friction lock)
                                  Layer 3: Two-Tranche Scale-Out (50% exit @ +0.75 ATR, ₹0 fee)
                                  Layer 4: Chandelier Lock (Peak - 0.35 ATR)
                                  Layer 5: 4-Bar Dead-Money Liquidation Gate
                                                  │
                                                  ▼
                               [Tier 6: Headless Autonomous Daemon & Recovery]
                                  * ShoonyaBrokerGateway (Paper Simulation Mode)
                                  * SQLite WAL State Ledger (audit/ravana_state_ledger.db)
                                  * 11:32 AM Reboot Protocol (Instant Crash Reconciliation)
                                  * Headless TOTP 2FA Watchdog (Auto-refresh every 2 hrs)
                                  * Autonomous Heartbeat Telemetry & Audit Journal
+---------------------------------------------------------------------------------------------------+
```

---

## 1. Multi-Sector Universe & Lakehouse Ingestion Pipeline

### 1.1 The 10 Core Sector BEES Universe
To capture every intraday and overnight structural edge on the NSE, the daemon monitors 10 institutional ETF pairs spanning all primary economic sectors:

| Sector / Theme | ETF Ticker | Underlying Benchmark Index | Heavyweight Synchrony Anchors |
| :--- | :--- | :--- | :--- |
| **Banking & Financials** | `BANKBEES.NS` | `^NSEBANK` | `HDFCBANK.NS`, `ICICIBANK.NS` |
| **Broad Market Bluechips** | `NIFTYBEES.NS` | `^NSEI` | `RELIANCE.NS`, `TCS.NS` |
| **Information Technology** | `ITBEES.NS` | `^CNXIT` | `INFY.NS`, `TCS.NS` |
| **Automotive** | `AUTOBEES.NS` | `^CNXAUTO` | `MARUTI.NS`, `M&M.NS` |
| **Pharma & Healthcare** | `PHARMABEES.NS` | `^CNXPHARMA` | `SUNPHARMA.NS`, `CIPLA.NS` |
| **Public Sector Banks** | `PSUBNKBEES.NS`| `^CNXPSUBANK` | `SBIN.NS`, `BANKBARODA.NS` |
| **Next 50 Midcap Titans**| `JUNIORBEES.NS` | `^NSMIDCP` | `BEL.NS`, `TRENT.NS` |
| **Energy & CPSE PSUs** | `CPSEETF.NS` | `^NSEI` | `NTPC.NS`, `ONGC.NS` |
| **Precious Metals (Gold)**| `GOLDBEES.NS` | `GC=F` (Gold Futures) | `USDINR` FX Rate |
| **Precious Metals (Silver)**|`SILVERBEES.NS` | `SI=F` (Silver Futures) | `USDINR` FX Rate |

### 1.2 Data Ingestion & Incremental Lakehouse Protocol
1. **Historical Partitioning:**
   - Ingest 60 days of 5-minute OHLCV bars for all 10 ETFs and their paired benchmark indices into `ravana/data/lake/m3_ingested/intraday_5m/{SYMBOL}.parquet`.
2. **Lossless Daily Accumulation:**
   - Utilize `IncrementalUpsertEngine` ([`ravana/data/storage/upsert.py`](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/data/storage/upsert.py)):
     - Concatenates incoming daily bars onto existing historical partitions.
     - Deduplicates timestamps preserving the latest verified price ticks.
     - Preserves rolling `.snapshots/` backups for ACID fault tolerance.
     - Over time, your storage organically accumulates 60d $\to$ 180d $\to$ 500d+ historical depth for **₹0 data vendor cost**.
3. **Session Gatekeeping:**
   - Standardize column contracts (`Open`, `High`, `Low`, `Close`, `Volume`).
   - Filter all out-of-session ticks, ensuring data strictly reflects 09:15:00 to 15:30:00 IST.

---

## 2. Cross-Sector Microstructure Feature Pipeline

Every 5-minute bar, the autonomous daemon computes the 36 stationary features for all 10 sector assets:

1. **Variance Ratio ($VR_{20}$ — Lo-MacKinlay):**
   - Determines whether each sector is in an **Impulse Trend** ($VR > 1.05$) or a **Mean-Reverting Range** ($VR \le 1.05$).
2. **Order Flow Imbalance ($OFI_5$):**
   - Measures multi-bar volume-weighted aggressive buyer vs seller pressure to confirm breakouts and prevent retail bull traps.
3. **Tracking Spread ($SPREAD\_BPS$ & $SPREAD\_ZSCORE_{20}$):**
   - Measures ETF price deviation from its underlying theoretical index iNAV:
     $$\text{Spread (bps)} = \left(\frac{P_{\text{ETF}} - \text{iNAV}}{\text{iNAV}}\right) \times 10,000$$
4. **Heavyweight Synchrony Gate:**
   - Before executing a sector breakout (e.g., `BANKBEES`), verify that its top 2 heavyweight stocks (`HDFCBANK` and `ICICIBANK`) are also printing positive directional returns ($R_5 > 0$). If heavyweights disagree, the ETF move is an illiquid head-fake and is rejected.

---

## 3. Capital Management & Cross-Sector Risk Safeguards (₹2,00,000 Budget)

With a ₹2,00,000 portfolio, capital must be protected against simultaneous drawdowns across correlated sectors. The daemon enforces a 4-tier risk architecture via `StrategyRiskBridge`:

```
                                 ₹2,00,000 TOTAL EQUITY
                                           │
            ┌──────────────────────────────┴──────────────────────────────┐
            ▼                                                             ▼
 [Max 2 Concurrent Position Slots]                             [Risk-at-Stop 0.25% Cap]
 Maximum 2 open positions at any time                          Max loss capped at ₹500
 (₹1,00,000 capital allocation per slot)                       (|P_entry - Stop| * Shares <= ₹500)
            │                                                             │
            └──────────────────────────────┬──────────────────────────────┘
                                           ▼
                       [HRP Sector Correlation Cluster Ceiling]
                       Pairwise correlated assets (r > 0.80) capped
                       at maximum 40% total portfolio notional (₹80,000)
                       Prevents double exposure to BANKBEES + PSUBNKBEES
```

### 3.1 Strict Institutional Rules:
1. **0.25% Equity Risk-at-Stop:**
   - Worst-case loss if a trade immediately hits its initial stop-loss is capped at **₹500** ($0.0025 \times ₹2,00,000$).
   - Calculated as:
     $$\text{Shares} = \min\left(\frac{₹500}{|P_{\text{entry}} - \text{Stop}| + \text{Slippage}}, \frac{2.0 \times ₹2,00,000}{P_{\text{entry}}}\right)$$
2. **2.0x MIS Leverage / 0.98x CNC Cash Cap:**
   - Intraday MIS trades utilize up to 2.0x leverage (up to ₹4,00,000 gross exposure).
   - Overnight CNC trades are strictly capped at 0.98x cash (max ₹1,96,000 notional) to eliminate overnight margin calls.
3. **HRP Sector Correlation Cluster Ceiling (40% Cap):**
   - If `BANKBEES` is already open, `PSUBNKBEES` cannot exceed the 40% cluster ceiling (max ₹80,000 notional), preventing sector concentration risk.
4. **Portfolio Contention Resolution:**
   - Exactly **2 concurrent position slots** are available.
   - If 4 sectors trigger buy signals simultaneously, the daemon ranks them by **Modulated Conviction $\times$ Bayesian Bandit Realized Win-Rate Multiplier** and accepts only the top 2.

---

## 4. Multi-Sector Opportunity Execution Engine (10 Triggers, 3 Buckets)

The daemon scans for all structural edges across the trading day:

### Bucket 1: Trend & Impulse ($VR > 1.05$)
1. **09:35 ORB Trigger (09:31 Bull-Trap Shield):**
   - Enforces a 20-minute opening baseline. Enters breakouts strictly at or after **09:35 IST** with $OFI_5 > 0.25$, completely avoiding the 09:31 HFT retail stop-hunt trap.
2. **Thursday Expiry Gamma Trigger:**
   - Detects gamma sweeps on weekly expiry days (holiday-aware: automatically shifts Thursday to Wednesday when NSE holidays occur).
3. **Cross-Sector Lead-Lag Drift Trigger:**
   - Exploits lead-lag discrepancies between index heavyweights and sector ETFs using closed-bar limit orders placed at the bid.
4. **Sector PEAD Drift Trigger:**
   - Rides institutional volume drift following quarterly earnings releases in sector heavyweight constituents ($VR > 1.15$).

### Bucket 2: Mean-Reversion & Absorption ($VR \le 1.05$)
5. **iNAV Spread Mean Reversion Trigger:**
   - Buys sector ETFs trading at a statistically significant discount to underlying index basket ($Z < -2.0$). Exits at mean reversion or on thesis invalidation if the underlying index breaks its 3-bar low.
6. **Closing VWAP Snapback Trigger (15:00 Retail MIS Dump Buy):**
   - Between **15:02 and 15:12 IST**, buys the temporary artificial dip caused by retail broker automated square-offs, capturing the 15:12–15:25 institutional VWAP recovery.
7. **DII SIP VWAP Absorption Trigger (Days 28 to 7):**
   - Moochs off the ₹25,000+ Crore monthly domestic mutual fund SIP inflows by buying pullbacks to session VWAP / Lower Bollinger Band during turn-of-month windows.
8. **ICT Unicorn Stop-Sweep Trigger:**
   - Identifies 10-bar low liquidity sweeps that immediately reclaim the level with a Bullish Fair Value Gap ($FVG > 0$), targeting a 3.3:1 asymmetric reward-to-risk ratio.

### Bucket 3: Macro & Overnight
9. **Afternoon Bond Europe TWAP Trigger:**
   - Enters European opening momentum between 14:00 and 14:30 IST, and **strictly exits or locks stops at 14:45 IST**—front-running the 15:00 retail MIS square-off dump by 15 minutes.
10. **Overnight Continuation Trigger (Tier-1 Whitelist):**
    - Identifies strong institutional closing momentum at 15:20–15:25 IST on whitelisted liquid ETFs (`BANKBEES`, `NIFTYBEES`, `ITBEES`, `GOLDBEES`).
    - Uses the **Continuation Gate**: if overnight momentum remains strong ($S \ge 0.50$), it holds through the 09:15 open, capturing multi-percent trends and saving the ₹15.93 CDSL DP fee.

---

## 5. Five-Layer "Zero-Giveback" Stop-Loss & Two-Tranche Scale-Out Shield

Every trade executed by the daemon is protected by the intrabar 5-layer shield:

```
[Entry Price: P_entry]
        │
        ├── Level 1: Thesis Micro-Stop ───────> Exit immediately if Underlying Index breaks 3-bar low (-10 to -35 bps)
        │
        ├── Level 2: Breakeven Ratchet ────────> When P >= P_entry + 0.45 ATR, Stop snaps to P_entry + 2 bps (Friction-Free)
        │
        ├── Level 3: Two-Tranche Scale-Out ────> At +0.75 ATR: SELL 50% shares (Bank guaranteed cash profit!)
        │                                        Snap remaining 50% Runner Stop to +0.25 ATR
        │
        ├── Level 4: Chandelier Lock ──────────> At +1.00 ATR: Trailing Stop locks at Peak Price - 0.35 ATR
        │
        └── Level 5: Dead-Money Liquidation ───> If trade stays stagnant for 4 bars (< +0.20 ATR), liquidate immediately
```

---

## 6. Zero-Intervention Autonomous Runtime & Self-Healing Architecture

How the daemon operates 100% autonomously without human intervention:

### 6.1 Headless TOTP 2FA Watchdog
- `ShoonyaSessionWatchdog` maintains an active session in the background.
- Employs a **4-second TOTP epoch buffer guard**: if an order or token refresh falls within 4 seconds of a 30-second TOTP cycle edge, the engine pauses until the new epoch begins, preventing invalid token rejections.
- Proactively re-authenticates every 2 hours without prompting the user.

### 6.2 11:32 AM Crash Reboot Recovery Protocol
- All trades, stops, and tranche state are maintained in SQLite WAL database [`audit/ravana_state_ledger.db`](file:///c:/Users/Dell/Desktop/python/trade_clean/audit/ravana_state_ledger.db).
- If the machine reboots, crashes, or loses network connection:
  1. Daemon restarts automatically as a system task.
  2. Queries `active_positions` from the ACID SQLite ledger.
  3. Queries live holdings from Shoonya Broker API.
  4. Automatically reconciles quantities:
     - If positions match $\to$ marked `IN_SYNC`.
     - If broker holds shares unrecorded in SQLite $\to$ automatically adopts them and attaches an emergency stop-loss at $P - 2.0\text{ ATR}$.
     - If SQLite has records closed by broker $\to$ automatically updates ledger to `CLOSED`.

### 6.3 Real-Time Autonomous Audit Trail
- Every signal, sizing decision, routed order, fill, and ratchet update is logged to:
  - Text journal: `audit/ravana_autonomous_daemon.log`
  - JSON order stream: [`audit/shoonya_paper_orders.json`](file:///c:/Users/Dell/Desktop/python/trade_clean/audit/shoonya_paper_orders.json)
  - SQLite database: [`audit/ravana_state_ledger.db`](file:///c:/Users/Dell/Desktop/python/trade_clean/audit/ravana_state_ledger.db)

---

## 7. Operational Modes: Today (Sunday Replay) vs. Live Market (Monday 09:15)

To guarantee complete reliability, the autonomous daemon is structured with a dual-mode entrypoint:

```
                                  [ravana_autonomous_daemon.py]
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
           [Mode 1: SESSION REPLAY]                              [Mode 2: LIVE WEBSOCKET FEED]
          (Active Today / Weekends)                             (Active Monday 09:15-15:30 IST)
   * Streams historical 5m bars sequentially              * Subscribes to live Shoonya WebSocket ticks
   * Evaluates triggers on bar close                      * Updates 5m OHLCV bars in memory
   * Routes paper orders to Shoonya Gateway               * Routes paper/live orders to Shoonya Gateway
   * Tests multi-sector portfolio allocation              * Manages real-time stops, scale-outs, & WAL state
   * Proves 100% autonomous operation                     * Zero manual intervention required
```

---

## 8. Implementation Steps & Verification Checklist

1. **Step 1: Multi-Sector Ingestion Script (`scripts/ingest_all_sectors.py`)**
   - Download/prepare 60 days of 5m bars for all 10 Sector BEES ETFs and their paired benchmark indices into `ravana/data/lake/m3_ingested/intraday_5m/`.
2. **Step 2: Multi-Sector Feature Generation**
   - Run `FeatureMatrixEngine` across all 10 sector assets to generate stationary features, $VR_{20}$, and tracking spreads.
3. **Step 3: Autonomous Execution Daemon (`ravana/daemon/autonomous_daemon.py`)**
   - Implement the zero-intervention daemon with ₹2 Lakh capital controller, HRP sector cluster guards, 2-slot portfolio contention manager, and SQLite WAL ledger.
4. **Step 4: End-to-End Autonomous Replay & Log Audit**
   - Execute the daemon over multi-sector market sessions.
   - Inspect log outputs to verify autonomous trigger execution, 50% scale-out fills, stop updates, and crash reconciliation.
5. **Step 5: Standby for Monday 09:15 AM Live Market Connection**
   - Certify the daemon is primed to transition seamlessly to live market tick streaming.
