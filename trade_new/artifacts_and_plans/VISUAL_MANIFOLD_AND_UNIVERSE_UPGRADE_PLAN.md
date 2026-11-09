# Plan: Restoring 20x20 MAP-Elites Heatmap, Dynamic Universe, Seed Diversity, and 3D Phenotype Manifold

## Objective
Upgrade the Decoupled Bio-Quant Alpha Cockpit on Port 7860 to fully restore and modernize the visual, analytical, and execution capabilities from the architecture:
1. **20×20 MAP-Elites WebGL Heatmap Grid:** Replace the 5-point dummy scatter with the genuine 400-niche behavioral heatmap (Volatility Regime Sigma vs Turnover Factor Tau) with fog-of-war and formula hovercards.
2. **Dynamic 100+ Asset Universe Selector:** Provide dynamic universe discovery from the Parquet lake and SQLite database with instant live switching across Index ETFs, Equities, Defense, and Global ADRs.
3. **Random Seed Governor:** Add deterministic and randomized seed controls to evolve unique alpha lineages across runs.
4. **3D Phenotype Manifold (UMAP & HDBSCAN) & 1024D Cosine Correlation Heatmap:** Expose on-demand manifold projection tabs powered by `AlphaManifoldPruner` and `PhenotypeExtractor`.

---

## Task Breakdown

### Task 1: 20×20 MAP-Elites Matrix Backend & Interactive WebGL Heatmap
- **Backend (`src/evolution/archive_store.py`):**
  - Implement `get_grid_matrix(ticker: str, lookback: str) -> Dict[str, Any]` returning a 20×20 dense matrix ($20 \times 20 = 400$ cells) containing `fitness`, `cagr_pct`, `formula_str`, `max_drawdown`, `trades_count`, and `occupied` status.
- **REST API (`src/ui/server.py`):**
  - Expose `GET /api/elites/grid` with query parameters `ticker`, `lookback`, and `metric` (`fitness`, `cagr_pct`, `sortino`, `win_rate`).
- **Frontend (`src/ui/static/app.js` & `index.html`):**
  - Implement `renderMapElitesHeatmap()` using Plotly WebGL heatmap.
  - Define custom institutional colorscale (`Viridis` on `#0f172a` slate background with empty niches rendered as fog-of-war).
  - Hover tooltip displays: `Formula`, `Sharpe`, `CAGR %`, `Max DD %`, `Trades`, and `Niche (Row, Col)`.

### Task 2: Dynamic Multi-Asset Universe Ingestion & Selection
- **REST API (`src/ui/server.py`):**
  - Expose `GET /api/universe/assets` returning categorized lists of available assets from both `data/*.parquet` and `resident_elites` in `audit/alpha_elites.db` (Categories: `Index ETFs`, `Defense & PSU`, `Equities`, `Global & ADRs`).
- **Frontend (`src/ui/static/app.js` & `index.html`):**
  - Replace hardcoded single option in `index.html` with dynamic Alpine `x-for` populated via `/api/universe/assets` on load.
  - On asset change, trigger instant re-fetch of `/api/elites/leaderboard` and `/api/elites/grid` for that asset.

### Task 3: Evolutionary Random Seed Governor
- **Worker IPC (`src/pipeline/evolution_worker.py`):**
  - Accept `seed: Optional[int] = None` in `run_evolution_process` and `spawn_evolution_worker`.
  - Apply `np.random.seed(seed)` and `random.seed(seed)` to guarantee reproducibility or exploratory speciation.
- **REST API (`src/ui/server.py`):**
  - Add `seed: Optional[int] = None` to `EvolveRequest` and propagate to `start_worker`.
- **Frontend (`src/ui/static/index.html` & `app.js`):**
  - Add Seed input with `🎲 Randomize` button in the Node 2 (MAP-Elites Engine) sidebar panel.

### Task 4: 3D Phenotype Manifold (UMAP & HDBSCAN) & 1024D Cosine Heatmap
- **REST API (`src/ui/server.py`):**
  - Expose `GET /api/elites/manifold?ticker=...&lookback=...`.
  - Utilize `PhenotypeExtractor` and `AlphaManifoldPruner` to project resident champion behaviors into 3D UMAP coordinates $[x, y, z]$, assign HDBSCAN cluster labels, and compute the pairwise 1024D cosine similarity correlation matrix.
- **Frontend (`src/ui/static/index.html` & `app.js`):**
  - Add navigation tabs:
    - `🗺 MAP-Elites Grid (20×20 Heatmap)`
    - `🌌 3D Phenotype Manifold (UMAP & HDBSCAN)`
    - `🧬 1024D Latent Space (Cosine Heatmap)`
    - `🏆 Resident Champions (AG Grid)`
    - `📟 Streaming Audit Log`
  - Implement `renderManifoldPlot()` (Plotly `scatter3d`) and `renderCosineHeatmap()` (Plotly `heatmap`).

### Task 5: Testing, Validation & Ledger
- Add unit and integration tests in `tests/test_cockpit_server_api.py`.
- Run full 31+ test suite across Phase 1.
- Run Red Team Invariant harness (`12/12 Invariants Satisfied`).
- Update state ledger and commit atomically.
