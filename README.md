# Bio-Quant Institutional Alpha Cockpit (Clean Rebuild)

## Architecture Overview
High-performance, decoupled quantitative research and execution engine for Indian Equities, Index ETFs, and Derivatives.

- **Process A (Cockpit UI):** FastAPI ASGI server (`src/ui/server.py`) on Port 7860 with Differential SSE telemetry.
- **Process B (MAP-Elites Engine):** Multi-core worker (`src/pipeline/evolution_worker.py`) isolated via Win32 Job Objects on Cores 2-3.
- **Process C (Risk Sentry):** Ultra-low latency guardian (`src/execution/risk_sentry.py`) on Cores 0-1 with sub-0.5ms Stage 1 freeze.
- **Storage Engine:** SQLite WAL database (`audit/alpha_elites.db`) storing 400-niche MAP-Elites champions.
- **Compute Bridge:** Native Kaggle/Colab PyTorch CUDA GPU acceleration over Cloudflare Tunnels.

## Quick Start
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test verification suite
python -m unittest discover -s tests

# 3. Launch Bio-Quant Cockpit
python -m uvicorn src.ui.server:app --host 127.0.0.1 --port 7860
```

## Directory Structure
- `src/domain/`: AST Node grammar, recursive descent parser, and invariants.
- `src/data/`: Vector lake Parquet readers and market loaders.
- `src/evolution/`: MAP-Elites grid, genetic engine, thymic chamber, and SQLite store.
- `src/backtest/`: Vectorized SIMD matrix backtester.
- `src/clustering/`: 3D UMAP manifold projector & Occam clone pruner.
- `src/pipeline/`: Multi-process IPC, Windows Job Objects, and SharedMemory.
- `src/execution/`: Out-of-band Risk Sentry, synthetic hedging, and Shoonya broker gateway.
- `src/ui/`: Decoupled FastAPI server and air-gapped avionics UI deck.
- `notebooks/`: Production Kaggle/Colab GPU worker notebooks.
- `legacy_technical_debt/`: Isolated legacy archives and deprecated Gradio code.
- `artifacts_and_plans/`: Architectural blueprints, state ledgers, and roadmap specs.
