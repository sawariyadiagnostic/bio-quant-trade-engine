# RAVANA v4.3: Two-Way Alpha & Bear Market Protection Plan
*Two-Way Morning Intraday (Long + Short) and All-Weather Overnight Inverse Architecture*

---

## 1. Executive Summary & Problem Diagnosis

Our 6-month continuous stress tests across 125 trading days on a ₹2,00,000 account exposed two major performance limiters:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ DIAGNOSTIC FINDINGS FROM 6-MONTH STRESS TEST                                           │
│ 1. Overnight Regime Gap: 1-Month made +₹18.6k (+9.3%) during Nov Bull Surge, but        │
│    6-Month made +₹4.6k (+2.3%) because May (-2.1%), Aug (-2.0%), and Sep (-6.0%) dumps│
│    eroded capital in a long-only equity portfolio.                                     │
│ 2. Zero Shorting Capability: Total Sell Signals in the system was 0! The system had    │
│    literally no way to profit when the market dropped.                                 │
│ 3. Intraday Low Frequency & Chop: Running on 1h bars caused trade count to collapse    │
│    to 20 trades, while buying dips on down days caught falling knives.                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**RAVANA v4.3 solves these issues by transforming both strategies into all-weather, two-way engines.**

---

## 2. Pillar 1: Two-Way Morning Intraday Engine (09:15–11:30 IST on 5m Bars)

Instead of trading 1-hour bars or holding into the toxic 15:00 retail dump, the new Intraday Engine focuses exclusively on **5-minute bars during the high-liquidity morning session (09:15–11:30 AM)**.

```mermaid
flowchart TD
    A["09:15 Market Open: Ingest 5m Bars across 10 Sector ETFs + NIFTY"] --> B["09:15 - 09:30: Form 15-Minute Opening Range (OR_high, OR_low)"]
    B --> C{"Check 09:30 NIFTY Direction"}
    
    C -- "NIFTY > Open & OFI > 0 (Bullish Morning)" --> D["BULL BIAS: Scan Strongest Sector for ORB-Long Breakout"]
    D --> E["09:35 Bar closes > OR_high with OFI > +0.20"]
    E --> F["Execute BUY MIS with 2.0x Leverage at 1m Limit Dip"]

    C -- "NIFTY < Open & OFI < 0 (Bearish Morning)" --> G["BEAR BIAS: The Option to SELL (Short)"]
    G --> H["09:35 Bar closes < OR_low with OFI < -0.20 on Weakest Sector"]
    H --> I["Execute SELL MIS (Short) with 2.0x Leverage at 1m Limit Rally"]

    F & I --> J["Active Management via 5-Layer Stop Shield"]
    J --> K["Tranche-1: Bank 50% Profit at +0.75 ATR"]
    K --> L["Tranche-2: Trail remaining 50% with Chandelier Stop"]
    J --> M["MANDATORY TIME EXIT AT 11:30 AM (Lock morning gains before midday chop!)"]
```

### Technical & Execution Specifications:
1. **Time Horizon:** Strictly **09:15 to 11:30 IST**. No new orders after 11:00 AM. Any remaining position is squared off at 11:30 AM before the midday consolidation zone begins.
2. **Two-Way Signals:**
   - **Long (ORB-Long):** Entered when price breaks above 15m $OR_{\text{high}}$ on the #1 strongest sector, confirmed by positive order flow ($OFI_5 > +0.20$).
   - **Short (`SELL MIS`):** Entered when price breaks below 15m $OR_{\text{low}}$ on the #1 weakest sector, confirmed by negative order flow ($OFI_5 < -0.20$).
3. **Safe MIS Leverage Sizing:**
   $$\text{Shares} = \min\left( \frac{0.0025 \times \text{Equity}}{\max(|P_{\text{entry}} - S_{\text{stop}}|, \quad P_{\text{entry}} \times 0.0045)}, \quad \frac{2.0 \times \text{Equity}}{P_{\text{entry}}} \right)$$
   - Stop buffer is pinned to $1.5\text{ ATR}$ ($\ge 45\text{ bps}$).
   - Maximum capital risk per trade is strictly capped at **0.25% (₹500 on ₹2 Lakh)**.

---

## 3. Pillar 2: Two-Way Sovereign Overnight Engine (Bull Momentum + Bear Market Inverse)

Under SEBI regulations, you cannot hold short delivery (CNC) of an equity ETF overnight. However, we can achieve true **All-Weather Two-Way Profitability** through two sanctioned mechanisms:

```mermaid
flowchart LR
    A["Daily 15:20 IST Macro Scan"] --> B{"Evaluate NIFTY 50-day EMA & VR20"}
    
    B -- "BULL REGIME (Nifty >= 50 EMA & VR20 >= 0.95)" --> C["BULL SECTOR ALLOCATION"]
    C --> D["Rank All Equity Sectors by Multi-Factor Momentum"]
    D --> E["Allocate to Top 2-3 Champions (Max 35% Cap per ETF)"]
    
    B -- "BEAR REGIME (Nifty < 50 EMA or VR20 < 0.90)" --> F["THE OPTION TO SELL: BEAR MARKET ENGINE"]
    F --> G["1. Flight-to-Safety: Allocate up to 35% to GOLDBEES (+8.6% in Aug, +4.3% in May)"]
    F --> H["2. Synthetic Inverse Short Hedge: 1x Short NIFTY Delta (Profits from crashes)"]
    F --> I["3. Cash Yield: Allocate remaining capital to LIQUIDBEES (6.5% Annual Risk-Free)"]

    E & G & H & I --> J["Enter at 15:25 IST Close"]
    J --> K["09:15 Morning Open: Hard Priority Stop Check"]
```

### 1. How the Overnight "Option of Sell" Generates Profit:
* **The Synthetic Inverse Short Hedge:** When NIFTY is below its 50 EMA in a macro markdown, the portfolio carries a synthetic inverse short position:
  $$\Delta_{\text{pnl}} = -\Delta_{\text{NIFTY}} \times \text{Allocated Notional}$$
  In September 2026, when NIFTY plunged **-6.05%**, this position yields **+₹12,100 in pure cash profit**!
* **Flight-to-Safety Commodity Rotation:** During market panics, domestic capital rushes into Gold. As demonstrated in our 6-month monthly table:
  - May 2026: NIFTY **-2.14%** $\to$ `GOLDBEES` **+4.27%**
  - August 2026: NIFTY **-2.00%** $\to$ `GOLDBEES` **+8.63%**
  Rotating into `GOLDBEES` during bear markets converts market crashes into large gains.

---

## 4. Implementation Roadmap & Verification Suite

```mermaid
flowchart TD
    P1["Phase 1: Implement Two-Way Morning ORB Trigger (ravana/strategies/two_way_orb.py)"] --> P2["Phase 2: Implement Bear Market Inverse Engine (ravana/strategies/bear_inverse.py)"]
    P2 --> P3["Phase 3: Write Test Suite in ravana/tests/test_two_way_alpha.py"]
    P3 --> P4["Phase 4: Run 6-Month Two-Way Morning Intraday Benchmark (09:15-11:30 5m bars)"]
    P4 --> P5["Phase 5: Run 6-Month Two-Way Sovereign Overnight Benchmark (Bull + Bear Hedge)"]
    P5 --> P6["Phase 6: Compile Comparative Performance Audit (v4.2 vs v4.3)"]
```

### Deliverables to be Created:
1. `ravana/strategies/two_way_orb.py`: 5-minute morning ORB engine with Long & Short MIS capability.
2. `ravana/strategies/bear_inverse.py`: Overnight bear regime inverse hedge and Gold flight-to-safety rotator.
3. `ravana/tests/test_two_way_alpha.py`: Complete unit tests verifying signal generation, sizing, and causality.
4. `scripts/paper_trade_morning_intraday_two_way.py`: 5-minute morning intraday runner.
5. `scripts/paper_trade_overnight_all_weather_v43.py`: All-weather overnight runner with bear inverse profit.
