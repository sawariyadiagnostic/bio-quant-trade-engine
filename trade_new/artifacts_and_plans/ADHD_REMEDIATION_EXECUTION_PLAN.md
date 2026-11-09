# ADHD Red Team Remediation: Institutional Master Execution Plan

> **Target Workspace:** `c:\Users\Dell\Desktop\python\trade\trade_new`  
> **Master Findings Ledger:** [`RED_TEAM_MASTER_FINDINGS_LEDGER.md`](file:///c:/Users/Dell/Desktop/python/trade/trade_new/RED_TEAM_MASTER_FINDINGS_LEDGER.md)  
> **Standard:** Complete step-by-step remediation, empirical proof, and regression testing after every step.

---

## Progress Overview

| Batch | Domain / Focus | Findings Covered | Status | Tests |
| :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | Micro-Quant, AST & Tensor Math Singularities | `ADHD-01` to `ADHD-09` (9 items) | [x] **COMPLETE** | 9/9 PASS (118/118 overall) |
| **Batch 2** | Broker Protocols, Microstructure & Order Routing | `ADHD-10` to `ADHD-20` (11 items) | [x] **COMPLETE** | 11/11 PASS (129/129 overall) |
| **Batch 3** | SRE Daemon Endurance, Concurrency & OS Interop | `ADHD-21` to `ADHD-28` (8 items) | [x] **COMPLETE** | 6/6 PASS (135/135 overall) |
| **Batch 4** | Evolutionary Archive, Manifold & Speciation | `ADHD-30` to `ADHD-39` (10 items) | [x] **COMPLETE** | 10/10 PASS (145/145 overall) |
| **Batch 5** | Cockpit Avionics, API Security & UI Sensory | `ADHD-40` to `ADHD-47` (8 items) | [x] **COMPLETE** | 7/7 PASS (152/152 overall) |

---

## Detailed Step-by-Step Task Checklist

### Batch 1: Micro-Quant, AST & Tensor Numerical Singularities
- [x] **1.1 (`ADHD-01`)**: Replace $O(B^2 \cdot T)$ 3D broadcast matrix ranking in `tensor_evaluator.py:122-126` and `formula_ast.py:22-26` with $O(B \log B \cdot T)$ argsort ranking (`torch.argsort(torch.argsort(x, dim=0), dim=0)`).
- [x] **1.2 (`ADHD-02`)**: Fix extrema inversion in `ts_max` (use `-inf` identity) and `ts_min` (use `+inf` identity) under `nan_to_num`.
- [x] **1.3 (`ADHD-03`)**: Fix sliding-window percentile deflation under leading NaNs in `ts_rank_20` by dividing by valid non-NaN count.
- [x] **1.4 (`ADHD-04`)**: Fix canonical hash collision on numeric constants: include `self.val` in `ASTNode.canonical_hash()`, fix `to_formula_str()`, and handle constants in `TensorASTEvaluator`.
- [x] **1.5 (`ADHD-05`)**: Bound multi-asset gross leverage in `MatrixSimulator.simulate_batch` using long-only / gross exposure normalization $\sum |w_i| \le 1.0$.
- [x] **1.6 (`ADHD-06`)**: Fix asymmetric distance denominators and NaN crash in `CircuitDamper` (`quantize_nse_tick`).
- [x] **1.7 (`ADHD-07`)**: Fix cumulative dividend volume inflation in `VectorDataLake` by preserving unadjusted raw volume for liquidity metrics.
- [x] **1.8 (`ADHD-08`)**: Fix DSR double penalty on sparse trading alphas in factor analytics.
- [x] **1.9 (`ADHD-09`)**: Harmonize `DIV_SAFE` across AST and Tensor evaluators with unified signed-epsilon threshold and remove $50\times$ step discontinuity.
- [x] **Batch 1 Verification**: Execute `test_quant_math_remediated.py`, `test_tensor_evaluator.py`, `test_matrix_engine.py`, and full test runner.

---

### Batch 2: Broker Protocols, Microstructure & Execution Protocols
- [x] **2.1 (`ADHD-10`)**: Fix `TokenBucketRateLimiter.acquire()` thread deadlock when requested tokens exceed bucket capacity (`tokens = min(tokens, self.capacity)`).
- [x] **2.2 (`ADHD-11`)**: Fix silent order loss and cascading TOTP lockout during token expiry in `ShoonyaGateway`.
- [x] **2.3 (`ADHD-12`)**: Fix partial fill infinite polling hang in `poll_order_confirmation`.
- [x] **2.4 (`ADHD-13`)**: Handle Cloudflare/Nginx 502/503 HTML reverse proxy responses in broker REST clients to prevent crash and double execution.
- [x] **2.5 (`ADHD-14`)**: Fix asynchronous cancel race condition masking matching engine fills.
- [x] **2.6 (`ADHD-15`)**: Fix panic liquidation derivative ticker mangling (prevent appending `-EQ` to options tickers).
- [x] **2.7 (`ADHD-16`)**: Fix Alpaca zero-price market order paper exploit and add `"extended_hours": True`.
- [x] **2.8 (`ADHD-17`)**: Fix IBKR warning confirmation prompt deadlock and conid 0 rejection.
- [x] **2.9 (`ADHD-18`)**: Implement margin check and debiting for short positions in `CCXTGateway`.
- [x] **2.10 (`ADHD-19`)**: Standardize pre-open auction TIF codes and fix trailing stop lookahead bias.
- [x] **2.11 (`ADHD-20`)**: Fix `ClientOrderIdLedger` JSON serialization crash on `Decimal` objects.
- [x] **Batch 2 Verification**: Execute broker gateway tests, paper trader tests, and full test runner.

---

### Batch 3: SRE Daemon Endurance, Concurrency & OS Interop
- [x] **3.1 (`ADHD-21`)**: Fix phantom morning exit bug (`hasattr(gateway, 'paper_positions')` check when paper trading).
- [x] **3.2 (`ADHD-22`)**: Implement ACPI S3 sleep blackout detection and trade window gate in `overnight_daemon.py`.
- [x] **3.3 (`ADHD-23`)**: Fix infinite session death loop on daily broker token expiry.
- [x] **3.4 (`ADHD-24`)**: Make alert webhooks non-blocking with strict timeouts and escape Markdown characters in Telegram exception traces.
- [x] **3.5 (`ADHD-25`)**: Fix Risk Sentry paper liquidation sequence (execute sell orders before clearing paper inventory).
- [x] **3.6 (`ADHD-26`)**: Fix Windows SQLite `WinError 32` file lock contention in `ArchiveStore` with explicit connection closes and WAL checkpoints.
- [x] **3.7 (`ADHD-27`)**: Register OS `SIGTERM` / console close signal handlers and configure Loguru rotation for 30s heartbeats.
- [x] **3.8 (`ADHD-28`)**: Add NTP clock skew detection and offset compensation for broker TOTP generation.
- [x] **Batch 3 Verification**: Execute daemon lifecycle tests, SRE resilience tests, and full test runner.

---

### Batch 4: Evolutionary Archive, Manifold & Speciation
- [x] **4.1 (`ADHD-30`)**: Fix 4D Voronoi hypercube boundary clamping in `cvt_archive.py` with adaptive / dynamic percentile bounds.
- [x] **4.2 (`ADHD-31`)**: Harmonize 1000-cell archive grid to 20x20 UI heatmap projection in `map_elites.py` / `archive_store.py`.
- [x] **4.3 (`ADHD-32`)**: Implement randomized tournament selection across all occupied cells and Pareto immigrant injection in `evolve_generation`.
- [x] **4.4 (`ADHD-33`)**: Wire up `EvolutionaryNursery` (immunity shields) and `SymbioticBreeder` (caste fusion) into `GeneticEvolutionEngine`.
- [x] **4.5 (`ADHD-34`)**: Fix intraday downside beta 1-day temporal phase shift in `map_elites.py`.
- [x] **4.6 (`ADHD-35`)**: Introduce dynamic multi-regime stress profiling and reject zero-signal passes in `ThymicChamber`.
- [x] **4.7 (`ADHD-36`)**: Add parsimony pressure (complexity penalty / minimum improvement delta) in `map_elites.py`.
- [x] **4.8 (`ADHD-37`)**: Eliminate quadratic write amplification in `_stream_to_sqlite()` by tracking dirty cells.
- [x] **4.9 (`ADHD-38`)**: Fix chimeric crossover arity mismatch and constant-zero division collapse in `formula_ast.py` / `symbiosis.py`.
- [x] **4.10 (`ADHD-39`)**: Fix canonical hash collision on constants and string type poisoning on SQLite warm-start in `archive_store.py`.
- [x] **Batch 4 Verification**: Execute evolution, speciation, and nursery test suites and full test runner.

---

### Batch 5: Cockpit Avionics, API Security & UI Sensory
- [x] **5.1 (`ADHD-40`)**: Add user gesture audio arming gate in `audio_synth.js` with HUD state indicator.
- [x] **5.2 (`ADHD-41`)**: Implement Web Audio master dynamics compressor / limiter and oscillator node cleanup to prevent DAC clipping.
- [x] **5.3 (`ADHD-42`)**: Upgrade `DropTailQueue` to prioritize emergency terminal events (`complete`/`abort`) over telemetry ticks.
- [x] **5.4 (`ADHD-43`)**: Fix AG Grid transaction update in `app.js` to upsert (`add` if not present in grid).
- [x] **5.5 (`ADHD-44`)**: Separate Panic button from mute button and implement two-step confirmation latch modal.
- [x] **5.6 (`ADHD-45`)**: Restrict CORS origins and implement CSRF token validation on emergency REST endpoints in `server.py`.
- [x] **5.7 (`ADHD-46`)**: Fix responsive CSS on high-DPI / small screens to ensure emergency controls are never truncated.
- [x] **5.8 (`ADHD-47`)**: Cap similarity matrix dimensions to $N \le 30$ in `manifold_service.py` to prevent browser main-thread freeze.
- [x] **Batch 5 Verification**: Execute UI API tests, security checks, and full test runner.
