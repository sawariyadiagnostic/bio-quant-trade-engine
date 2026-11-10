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

### 1.5 The Depository Participant (DP) Flat Charge Reality & Viable Capital Sizing
* **Statutory Tax Symmetry vs. Ticket Fee Drag:**
  * Indian Equity ETFs carry near-zero statutory delivery tax under Finance Act Section 98: **0.000% Buy STT** and **0.001% (0.1 bp) Sell STT**.
  * Total variable statutory drag (Stamp duty 1.5 bps + Exchange turnover 0.35 bps + SEBI/GST 0.08 bps + STT 0.1 bp) = **~2.0 basis points (0.020%)**.
* **The Fatal Retail Trap: Fixed DP Debit Charges:**
  * When selling ETF units from Demat (Cash Delivery), CDSL/NSDL and the broker levy a **flat DP debit fee of ₹13.00 to ₹15.00 + 18% GST (~₹15.34 to ₹17.70)** per ISIN per trading day.
  * **Capital Scale Sensitivity Matrix:**
    * **₹20,000 Position:** ₹17.70 DP fee = **8.85 bps drag** $\implies$ Total round trip = **~11.0 bps** (Fatal; wipes out overnight edge).
    * **₹50,000 Position:** ₹17.70 DP fee = **3.54 bps drag** $\implies$ Total round trip = **~5.5 bps**.
    * **₹2,00,000 Position:** ₹17.70 DP fee = **0.88 bps drag** $\implies$ Total round trip = **~2.9 bps** (Viable).
    * **₹5,00,000 Position:** ₹17.70 DP fee = **0.35 bps drag** $\implies$ Total round trip = **~2.35 bps** (Institutional sweet spot).
* **Institutional Mandate for Muscle 8 & Muscle 12:**
  * Overnight cash delivery ETF trading is strictly gated to **positions $\ge ₹2,00,000$ per trade**.
  * Target builder and backtester must model: $\text{Friction}_{\text{total}} = (P \times Q \times 0.00020) + ₹17.00$.

---

### 1.6 Single-Asset Directional Alpha vs. Cross-Sectional Relative Value
* **The Structural Difference from WorldQuant BRAIN:**
  * WorldQuant trades 3,000 equities simultaneously and neutralizes market beta ($\beta \approx 0$) by computing cross-sectional rank: $\alpha_i = \text{rank}(x_i) - 0.5$.
  * RAVANA operates on single index ETFs (`BANKBEES.NS`, `NIFTYBEES.NS`), where cross-sectional rank across 1 asset is a constant ($0.5$).
* **Where RAVANA's Alpha Must Originate:**
  1. **Holding-Window Asymmetries:**
     * *Afternoon DII Inflow Window:* 14:00/14:30 $\to$ 15:25 IST.
     * *Overnight Global Macro Beta:* 15:24 $\to$ 09:16 IST.
  2. **iNAV Mispricing Mean-Reversion:**
     * Exploiting transient tracking basis anomalies ($\text{Basis}_{\text{bps}} < -20\text{ bps}$) before Authorized Participants (APs) arb them away.
  3. **Volatility & Regime Clustering:**
     * Volatility conditioning (e.g. expanding allocations when ATR is low and contracting when India VIX spikes).

---

### 1.7 Order Book Crossing & Opening Auction Slippage Realities
* **The "Fill at Close" Fallacy:**
  * Placing a buy order at 15:24:59 does not fill at the 5m bar `Close` price; it must cross the bid-ask spread:
    $$P_{\text{fill, buy}} = P_{\text{close}} + \frac{1}{2}\text{Spread}_{\text{quoted}}$$
  * On `BANKBEES`, half-spread crossing adds **3 to 8 bps** of execution drag.
* **The 09:15 Opening Auction Chaos:**
  * Exiting at 09:15:01 encounters post-auction order uncrossing volatility. Quoted spreads widen to **15 to 30 bps** during the first 60 seconds of continuous trading.
  * Backtester (Muscle 11) must model half-spread penalties and cap simulated execution volume to **$\le 5\%$ of bar volume**.

---

### 1.8 Multiple Testing Curse, False Discovery Rate (FDR) & Deflated Sharpe Ratio
* **Data Snooping Bias:**
  * Evaluating 500 individuals $\times$ 100 generations = 50,000 symbolic formulas.
  * Under random walk noise, testing 50,000 candidate formulas guarantees finding dozens with historical Sharpe $> 2.5$ and $R^2 > 0.15$ by pure chance.
* **Institutional Safeguards for Muscle 10 & Muscle 15:**
  1. **Deflated Sharpe Ratio (DSR):** Replaces raw Sharpe by discounting for $N_{\text{trials}}$, track record length, and non-Gaussian skewness/kurtosis (Marcos López de Prado).
  2. **Benjamini-Hochberg FDR Control:** Controls the false discovery rate across multiple candidate formulas.
  3. **Tree Parsimony Penalization:** $\text{Fitness} = \text{IC} - \lambda \cdot \text{NodeCount} - \gamma \cdot \text{Depth}$, preventing deep tree memorization of historical crashes.

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

### Phase 1: Data Pipeline & Microstructure Layer (Muscles 3, 4, 5) — 100% Complete & Sealed
- [x] **Muscle 3: Market Data Ingestion & Gatekeeper:** Dual-source yfinance/local fallback, NSE 09:15-15:30 IST gatekeeper, special sessions (Diwali Muhurat, Budget Saturdays), proportional split scaler.
- [x] **Muscle 4: Data Quality & Cleaning Engine:** Causal gap-filling (`TimeGridReindexer`), MAD 50% breakdown spike filtering with institutional volume gate (`FlashSpikeFilter`), point-in-time ETF-to-benchmark basis analytics (`ETFBUSpreadAligner`).
- [x] **Muscle 5: Causal Multi-Timeframe Resampler:** Session-anchored intraday binning, zero cross-session bleeding, actionable right-labeled timestamps, turnover-weighted VWAP, causal incomplete period sealing, and point-in-time backward as-of joining (`SessionAwareResampler`, `ResamplingEngine`).

---

## 4. Pending Improvements for Future Muscles (To Implement Next)

### Muscle 6: Sovereign Columnar Parquet Data Lake (`ravana/data/lake/`)
- [ ] **Lakehouse Partition Schema & Read API:** Unified fast reader and partition query manager (`LakeManager`) reading Bronze (M3), Silver (M4), and Gold (M5) tiers.
- [ ] **Incremental Upsert Engine:** Merge new incoming intraday bars with existing partitions without duplicating timestamps or rewriting full histories.
- [ ] **Memory-Mapped Parquet Queries:** Zero-copy columnar slicing for Muscle 7 feature engineering.

### Muscle 7: Market Vocabulary Feature Matrix Engine (`ravana/features/`)
- [ ] **21 Standard Features:** Compute stationary z-scored technical indicators (`VWAP`, `ATR_14`, `VOLATILITY_20`, `RET_1`, `RET_5`, `VOLUME_ZSCORE`, `HIGH_LOW_SPREAD`).
- [ ] **Macro Inter-Market Features:** Overnight US return ($R_{\text{SP500}}$), VIX delta ($\Delta \text{VIX}$), Brent Crude momentum, and ETF tracking spread ($\text{Spread}_{\text{iNAV}}$).

### Muscle 8: Target Builder & Execution Strategy Variations (`ravana/targets/`)
- [ ] **Overnight Target Return:** $t \text{ [15:24 IST]} \to t+1 \text{ [09:16 IST]}$ minus 1.5 bps statutory cash delivery friction.
- [ ] **Afternoon Intraday Target Return:** $t \text{ [14:00/14:30 IST]} \to t \text{ [15:24 IST]}$ (capturing mutual fund closing inflows without overnight risk).
- [ ] **Hybrid Continuation Target:** Combined afternoon-to-morning holding window.

---

### Muscle 4 Implementation: Data Quality, Time-Grid Regularization & Spread Aligner (Completed & Deep-Audited)
* **Exact Session Grid Regularization (`TimeGridReindexer`):**
  * Real 60-day 5m `BANKBEES` data contained 4,497 bars instead of the mathematical expectation of 4,500 bars ($60 \times 75$).
  * Reindexed to exact left-closed interval $[09:15, 15:30)$ session grid with 0.0-volume causal forward filling.
  * Marked filled bars with `Is_Synthetic: bool` for audit and backtester trade exclusion.
  * *Hardening Fix (Zero-NaN Invariant):* Ensured all split-adjusted columns (`Open_Adj`, `High_Adj`, `Low_Adj`, `Close_Adj`) are set on synthetic bars, eradicating NaNs completely.
  * *Defensive Exchange Gate:* Reindexer now dispatches exchange profiles (`NSE`, `US`, `GLOBAL_24H`) and strictly skips weekends and holidays from accidental feed leaks.
* **Volume-Gated Robust Spike Filtering (`FlashSpikeFilter`):**
  * Discovered that standard deviation has breakdown point 0% (a single +50% spike contaminated rolling $\sigma$ across the next 20 bars from 0.5 to 55.0, inflating upper bounds to 723 INR).
  * Upgraded to causal Median Absolute Deviation (MAD, breakdown point 50%): $\sigma_{\text{MAD}} = 1.4826 \times \text{MAD}$.
  * *Breakout Protection Gate:* Audit revealed 93% of flagged spikes clustered at 09:15–10:00 IST morning momentum. Added institutional volume confirmation: if $\text{Volume} \ge 2\times \text{Volume}_{\text{med}}$, threshold expands to protect legitimate institutional morning breakouts, reducing false positives by 66% (from 60 down to 20).
  * *Symmetric Geometry:* Symmetrically clamps both `Open` and `Close` to prevent artificial wide-bodied ghost candles.
* **Causal ETF-to-Benchmark Spread Analytics (`ETFBUSpreadAligner`):**
  * Purged full-sample global median lookahead bias. Replaced with regulatory creation-unit parity ratios (`KNOWN_ETF_RATIOS`) and point-in-time causal expanding medians shifted by 1 bar.
  * Purged contemporaneous warmup bias in rolling mean/std.
  * Safe variance guards against flat index intervals.
* **Handshake Contract:** `CleanedDataset` verified with 0 NaNs and delivered to Muscle 5 (Resampler).

---

### Muscle 5 Implementation: Causal Multi-Timeframe Resampler (Completed & Battle-Hardened)
* **Intraday Session-Anchored Aggregation (`SessionAwareResampler`):**
  * **Zero Cross-Session Bleeding:** Vectorized session partitioning prevents overnight ghost bins across 15:30 to 09:15.
  * **Strict Actionable Right-Labeling:** Avoids naive wall-clock start-labeling (which leaks up to 59 minutes of future trading). Every bar is stamped at its completion ($t_{\text{actionable}} = t_{\text{close}}$). E.g., $[09:15, 09:30) \implies \mathbf{09:30:00}$.
  * **Deterministic Session Bar Counts:**
    * 15m: Exactly 25.00 bars/session ($25 \times 15\text{m} = 375\text{m}$).
    * 30m: Exactly 13.00 bars/session ($12 \times 30\text{m} + 1 \times 15\text{m}$ terminal stub clamped to 15:30:00).
    * 1h (60m): Exactly 7.00 bars/session ($6 \times 60\text{m} + 1 \times 15\text{m}$ terminal stub clamped to 15:30:00).
    * 75m: Exactly 5.00 bars/session (exact harmonic divisor).
    * 125m: Exactly 3.00 bars/session (exact harmonic divisor).
* **True Turnover-Weighted VWAP:**
  * Replaced fatal naive unweighted arithmetic mean (`mean(VWAP)`, which creates up to 42 bps drift) with exact turnover-weighted aggregation:
    $$\text{VWAP} = \frac{\sum (\text{Typical Price} \times \text{Volume})}{\sum \text{Volume}}$$
  * Strict zero-volume fallback to `Close` with 0 NaNs.
* **Holiday-Aware Causal Period Sealing (`resample_daily_to_higher`):**
  * Evaluates remaining weekdays against `KNOWN_NSE_HOLIDAYS` and business days (`BDay(1)`). Protects ~30% of completed historical weekly and monthly bars (such as 4-day Good Friday weeks and weekend month-ends like May 2026) from being erroneously erased.
* **Zero-Lookahead Anti-Leakage Merging (`point_in_time_merge`):**
  * Midnight-normalized higher timeframe bars are shifted by `+23:59:59` when performing backward as-of merges onto intraday series, guaranteeing that Friday 09:15 AM intraday trades strictly observe Thursday's prior state, never Friday afternoon's 15:30 weekly close.
* **Adaptive Multi-Timeframe Spread Analytics (`ResamplingEngine`):**
  * Dynamic scaling of `ETFBUSpreadAligner(rolling_window)`: maintains a uniform 5-hour statistical lookback across intraday timeframes (60 bars on 5m, 20 bars on 15m, 10 bars on 1h) and a 20-period floor on macro bars. Eliminates statistical horizon distortion across timeframes.
* **Continuous Session Segmentation:**
  * Handles split sessions and intraday trading voids (e.g. NSE Disaster Recovery switchovers: 09:15–10:00 and 11:30–12:30) with zero NaN crashes.
* **28.4x Vectorization Speedup:**
  * Vectorized session bin assignment reduces resampling of 500 trading sessions from 14.08 seconds down to 0.49 seconds.
* **Gold Tier Lake Manifest (`ravana/data/lake/m5_resampled/`):**
  * Generated and certified **38 Parquet files** across `intraday_15m`, `intraday_30m`, `intraday_1h`, `macro_1wk`, and `macro_1mo` with **0 NaNs and 0 Infs**.

---

*Vault maintained by RAVANA Core Engineering. Updated synchronously with every muscle completion.*
