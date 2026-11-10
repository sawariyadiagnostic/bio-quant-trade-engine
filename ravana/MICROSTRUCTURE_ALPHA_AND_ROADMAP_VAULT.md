# RAVANA Sovereign Institutional Microstructure & Alpha Vault
## Persistent Quant Knowledge Base, Market Mechanics & Implementation Roadmap

> **Directive:** This vault serves as the permanent, single-source-of-truth for all quantitative research insights, market microstructure realities, institutional flow patterns, and architectural improvements discovered since the inception of the RAVANA trading engine. Context windows may compact, but this knowledge base persists permanently within the codebase.

---

## 1. Indian Equity ETF Microstructure Realities (`BANKBEES.NS` & `NIFTYBEES.NS`)

### 1.1 The 14:00 – 15:15 IST Mutual Fund / DII Inflow Mechanism
* **The Regulatory Mechanism:**  
  Under SEBI mutual fund guidelines, retail and corporate cut-off timing for same-day NAV on equity funds is **15:00:00 IST**.  
  Asset Management Companies (AMCs) and mutual fund treasury desks receive pooled bank subscription funds throughout the morning and deploy cash into constituent baskets or index ETFs between **14:00:00 and 15:15:00 IST**.
* **The Structural Price Push:**  
  This institutional deployment generates systematic upward buying pressure ("Afternoon Liquidity Ramp") leading into market close.

---

### 1.2 The 09:15 – 10:00/10:30 IST Morning Opening Momentum & Gap-and-Go Drive
* **The Market Opening Catalyst:**  
  At 09:15:00 IST, the market opens following the pre-market auction (09:00-09:08). Overnight news, Wall Street closing trends, and Asian market moves trigger an aggressive wave of order execution:
  * **Overnight Short-Covering / Long-Liquidation Cascades:** Trapped overnight positions trigger forced market order executions in the first 15–30 minutes.
  * **Institutional Algo Initialization:** Institutional TWAP and VWAP execution algorithms begin slicing their daily parent orders starting at 09:15 IST.
  * **Opening Range Breakout (ORB) Drive:** High-volume momentum often produces strong directional continuation ("Gap-and-Go") between 09:15 and 09:45 IST before institutional volume drops off into the European open / midday lull (~10:30 to 13:00 IST).
* **The Five Discrete Quantitative Strategy Profiles to Evolve & Backtest:**
  1. **Profile A: Pure Afternoon Momentum (Intraday Cash):**  
     * *Entry:* 14:00:00 or 14:30:00 IST on volume/trend confirmation.  
     * *Exit:* 15:20:00 – 15:25:00 IST before continuous market close.  
     * *Advantage:* Zero overnight gap risk; completely eliminates exposure to Wall Street / geopolitical overnight shocks.
  2. **Profile B: Overnight Structural Gap (Cash Delivery):**  
     * *Entry:* 15:20:00 – 15:25:00 IST (at the end of the DII inflow ramp).  
     * *Exit:* 09:15:00 – 09:20:00 IST next morning at open auction / bar 1.  
     * *Advantage:* Captures global overnight macro beta and morning opening gap at ultra-low delivery friction (~1.5 bps).
  3. **Profile C: Hybrid Afternoon-to-Overnight Continuation:**  
     * *Entry:* 14:30:00 IST if domestic momentum triggers.  
     * *Decision Gate at 15:24 IST:* If position is profitable AND US futures/global sentiment is positive, carry overnight into 09:16 IST exit. If domestic momentum falters, close intraday at 15:25 IST.
  4. **Profile D: Morning Opening Momentum / Gap-and-Go (Intraday Cash):**  
     * *Entry:* 09:15:00 – 09:20:00 IST on opening range breakout / gap continuation.  
     * *Exit:* 09:45:00 – 10:15:00 IST before the midday liquidity exhaustion.  
     * *Advantage:* Captures early morning price discovery momentum and retail stop cascades with high volatility.
  5. **Profile E: Overnight + Morning Momentum Extension:**  
     * *Entry:* 15:20:00 – 15:25:00 IST overnight entry.  
     * *Dynamic Exit:* Instead of blind exit at 09:16 IST, if morning volume is expanding with green continuation candles, trail the stop and ride the morning momentum until 09:45–10:00 IST!


---

### 1.2 ETF iNAV & Tracking Spread Arbitrage
* **What is iNAV?:**  
  An ETF is not an equity stock; it is an open-ended mutual fund unit whose fair value is the **indicative Net Asset Value (iNAV)** based on constituent bank share prices (`^NSEBANK`).
* **The Premium/Discount Hazard ($\text{Spread}_{\text{iNAV}}$):**  
  $$\text{Spread}_{\text{iNAV}}(t) = \frac{P_{\text{ETF}}(t) - \text{iNAV}(t)}{\text{iNAV}(t)}$$
  * When retail buying surges between 15:20 and 15:25 IST, `BANKBEES` frequently trades at a **premium** (+0.30% to +0.80% above fair value).
  * Overnight, Authorized Participants (APs) like Nippon Life, Mirae, and Edelweiss balance creation/redemption units.
  * At 09:15 IST next morning, the market open auction compresses this premium back to 0.00%.
  * **Quant Rule:** An overnight buy must **never enter** if $\text{Spread}_{\text{iNAV}} > +0.50\%$. Even if the index rises +0.40% overnight, a 0.50% premium collapse results in a net losing trade!
* **Architectural Integration:** Muscle 3 natively pairs `BANKBEES.NS` with `^NSEBANK` inside [`IngestedDataset`](file:///c:/Users/Dell/Desktop/python/trade_clean/ravana/data/ingestion/dataset.py) so Muscle 7 (Features) can compute this spread.

---

### 1.3 Concessional Indian Statutory Friction Schedule for ETFs
* **Equity Shares vs Equity ETFs (The Statutory Edge):**  
  * Individual equity shares carry a delivery Securities Transaction Tax (STT) of **0.1% on buy AND 0.1% on sell** (total 20 bps round-trip).
  * Under the Indian Finance Act, **Equity ETFs carry a concessional STT rate:**
    * **Buy STT on Delivery:** **0.000% (Zero STT on Purchase)**!
    * **Sell STT on Delivery:** **0.001% (0.1 basis point on Sale)**!
  * **Total Statutory Round-Trip Friction Breakdown for Cash Delivery ETF:**
    * Brokerage: Zero (Discount brokers like Zerodha, Shoonya, Angel One offer ₹0 delivery brokerage).
    * STT: 0.001% (on sell).
    * Exchange Turnover Charges (NSE): 0.00345% (~0.35 bps).
    * SEBI Turnover Charges: 0.0001% (~0.01 bps).
    * Stamp Duty: 0.015% (1.5 bps on buy).
    * GST (18% on exchange charges): ~0.06 bps.
    * **Net Total Round-Trip Cash Delivery Friction:** **~1.5 basis points (~0.015%)**!
  * This concessional tax regime is the exact economic reason why high-turnover overnight cash delivery ETF trading is exceptionally viable in India.

---

### 1.4 Exchange Hours & Microstructure Conventions
* **Left-Closed Intraday Grid $[t_{\text{start}}, t_{\text{end}})$:**  
  * Bar `09:15:00` = `09:15:00.000` to `09:19:59.999`.
  * Continuous trading concludes at `15:30:00.000` IST.
  * The 75th and final continuous trading bar is stamped **`15:25:00`** (`15:25:00.000` to `15:29:59.999`).
* **The 15:30:00 Synthetic Closing Bar Trap:**  
  * The `15:30:00` tick emitted by exchanges/vendors is the **30-minute Volume-Weighted Average Price (VWAP) closing auction price**, not a tradeable continuous execution.
  * Ingesting `15:30:00` creates a phantom 76th bar with severe lookahead bias. Muscle 3 strictly purges this bar.
* **Special Trading Session Calendar:**  
  * **Union Budget Live Saturdays:** e.g., `2025-02-01` (895,670 BANKBEES shares traded) and `2020-02-01`.
  * **Diwali Muhurat Evening Sessions:** e.g., `2023-11-12` (`18:15–19:15`), `2024-11-01` (`18:00–19:00`).
  * **Disaster Recovery (DR) Split Sessions:** e.g., `2024-03-02` and `2024-05-18` (`09:15–10:00` and `11:30–12:30` with a 90-min trading void).

---

## 2. Global Inter-Market Macro Transmission Channels (15:30 IST to 09:15 IST)

What drives the overnight price gap of Indian ETFs while the NSE is closed?

| Asset | Ticker | Trading Hours (IST) | Transmission Mechanism to Indian Markets |
| :--- | :--- | :--- | :--- |
| **S&P 500 Index** | `^GSPC` / `SPY` | 19:00 – 01:30 IST | Primary global equity beta driver. Strong Wall Street rally drives foreign institutional (FII) overnight buying. |
| **Nasdaq 100 Index** | `^IXIC` / `QQQ` | 19:00 – 01:30 IST | Global technology and risk-appetite proxy; dictates early morning equity sentiment. |
| **CBOE Volatility Index** | `^VIX` | 19:00 – 01:30 IST | Global panic index. Spikes above +10% in US VIX reliably trigger gap-down openings in Indian banking stocks. |
| **USD / INR Currency** | `INR=X` | 24 Hours (UTC) | Currency depreciation ($INR \downarrow$) increases imported inflation and triggers emergency FII equity outflows. |
| **Brent Crude Oil** | `BZ=F` | 24 Hours (UTC) | India imports >80% of its crude oil. Spikes in crude directly pressure Indian bank balance sheets and current account deficits. |
| **US 10-Year Treasury** | `^TNX` | 19:00 – 01:30 IST | Global risk-free rate. Rising yields cause emerging market capital repatriation back to the US. |
| **GIFT NIFTY (SGX)** | `GIFT` | 06:30 – 09:00 IST | Early morning pre-market price discovery reflecting Asian market sentiment before the 09:15 NSE bell. |

### Causal Zero-Lookahead Rule
* At **15:24 IST entry** on day $t$: The US market is closed; only day $t-1$'s US close is observable.
* At **09:16 IST exit** on day $t+1$: The US session completed overnight; night $t$'s US close and returns are realized and observable.

---

## 3. Master Chronological Improvement & Implementation Log

### Phase 0: Foundations & Domain AST (Muscle 1) — 100% Complete
- [x] **Safe Bounded Math:** Zero-crash protected operators (`SAFE_DIV`, `SAFE_LOG`, `TANH`, `CLAMPED_EXP`, `SIGNED_POWER`).
- [x] **Algebraic AST Simplification:** Constant folding (`2+3 -> 5`), identity reductions (`X - X -> 0.0`, `X / X -> 1.0`, `X * 0 -> 0.0`), involution pruning (`ABS(ABS(X)) -> ABS(X)`), and constant rolling folding (`DELTA_5(5.0) -> 0.0`).
- [x] **Institutional Alpha Primitives:** Added `TS_STD_20`, `TS_MEDIAN_10`, `TS_ARGMAX_10`, `TS_ARGMIN_10`, and `SIGNED_POWER`.
- [x] **Dimensional Typing Guard:** Crossover strictly enforces dimension compatibility (`PRICE <-> PRICE`, `VOLUME <-> VOLUME`).

### Phase 0: High-Performance GPU Tensor Engine (Muscle 2) — 100% Complete & Cloud-Certified
- [x] **Vectorized PyTorch Evaluator:** FIR convolution weights cache, multi-channel rolling operators via `unfold`.
- [x] **Chunked Batching:** Memory-capped chunking (`chunk_size=512`) to prevent GPU OOM on large formula populations.
- [x] **Auto-Dispatcher:** Routes CPU DataFrames to C-compiled NumPy and GPU/3D tensors to CUDA PyTorch.
- [x] **Kaggle Tesla T4 Cloud Verification:** Certified at **1,337 formulas/second (6,012,980 points/second)** in 0.748s with peak VRAM of only 37.24 MB.

### Phase 1: Market Data Ingestion & Session Gatekeeper (Muscle 3) — 100% Complete & Battle-Hardened
- [x] **Dual-Source Ingestion:** Remote `yfinance` download with automatic local Parquet/CSV fallback.
- [x] **NSE Regulatory Gatekeeper:** Enforces 09:15 to 15:30 IST market hours with strict left-closed boundaries (`[09:15, 15:30)` admitting exactly 75 bars).
- [x] **Special Session Registry:** Native preservation of Union Budget Saturdays (`2025-02-01`), Diwali Muhurat evening sessions (`18:15–19:15`), and Disaster Recovery live trading drills.
- [x] **Multi-Asset Exchange Profiles:** Auto-deduction of `NSE` (`Asia/Kolkata`), `US` (`America/New_York`), and `GLOBAL_24H` (`UTC`).
- [x] **Physical Candlestick Invariants:** Sanitizer enforcing $P > 0$, $High \ge \max(O,C)$, $Low \le \min(O,C)$, and $Volume \ge 0$.
- [x] **Corporate Action Proportional Scaler:** $O^*, H^*, L^*$ scaled by $\frac{Close\_Adj}{Close}$ preventing geometric inversion and 6800% volatility spikes across stock splits.
- [x] **Sealed Handshake Contract:** `IngestedDataset` container bundling primary ETF, paired index (`^NSEBANK`), macro drivers, and health telemetry.

---

## 4. Pending Improvements for Future Muscles (To Implement Next)

### Muscle 4: Data Quality & Cleaning Engine (`ravana/data/cleaner/`)
- [ ] **Intraday Time-Grid Reindexing:** Bridge the 1–2 missing bar gaps found in real 5m ETF data (e.g., `2026-09-21` and `2026-10-15`) by forward-filling prices and zero-filling volume, guaranteeing an equidistant temporal grid for Muscle 1 & 2 rolling operations.
- [ ] **Flash Spike & Bad Tick Filter:** Flag and clamp erroneous multi-sigma price jumps unconfirmed by volume or neighboring bars.
- [ ] **ETF Premium/Discount Spread Generator:** Synchronize `BANKBEES.NS` with `^NSEBANK` and generate the tracking spread series $\text{Spread}_{\text{iNAV}}$.

### Muscle 5: Causal Multi-Timeframe Resampler (`ravana/data/resampler/`)
- [ ] **Mother Bar Resampling Engine:** Causally resample 1D Mother Bars into derived 1W and 1M bars with strict right-closed intervals (`closed='right', label='right'`).
- [ ] **Intraday Resampling:** Resample clean 1m Mother Bars into 5m, 15m, and 1h intervals with zero future lookahead.

### Muscle 6: Sovereign Columnar Parquet Data Lake (`ravana/data/lake/`)
- [ ] **Full 18-Year Historical Daily Lake:** Ingest and store 4,400+ daily bars for `BANKBEES.NS` and `NIFTYBEES.NS` (2009–present).
- [ ] **Incremental Upsert Engine:** Merge new 5m intraday pulls with existing parquet partitions without duplicating timestamps, continuously expanding local historical archives.

### Muscle 7: Market Vocabulary Feature Matrix Engine (`ravana/features/`)
- [ ] **21 Standard Features:** Compute stationary z-scored technical indicators (`VWAP`, `ATR_14`, `VOLATILITY_20`, `RET_1`, `RET_5`, `VOLUME_ZSCORE`, `HIGH_LOW_SPREAD`).
- [ ] **Macro Inter-Market Features:** Overnight US return ($R_{\text{SP500}}$), VIX delta ($\Delta \text{VIX}$), Brent Crude momentum, and ETF tracking spread ($\text{Spread}_{\text{iNAV}}$).

### Muscle 8: Target Builder & Execution Strategy Variations (`ravana/targets/`)
- [ ] **Overnight Target Return:** $t \text{ [15:24 IST]} \to t+1 \text{ [09:16 IST]}$ minus 1.5 bps statutory cash delivery friction.
- [ ] **Afternoon Intraday Target Return:** $t \text{ [14:00/14:30 IST]} \to t \text{ [15:24 IST]}$ (capturing mutual fund closing inflows without overnight risk).
- [ ] **Hybrid Continuation Target:** Combined afternoon-to-morning holding window.

---

*Vault maintained by RAVANA Core Engineering. Updated synchronously with every muscle completion.*
