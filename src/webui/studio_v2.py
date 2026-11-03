"""
Bio-Quant Alpha-OS v2.0 Control Studio.
Unified Interactive WebUI Dashboard featuring:
- 🌌 3D Phenotypic Vector Space Manifold (UMAP + HDBSCAN)
- 🧬 Evolutionary Nursery & CVT-MAP-Elites Archive Monitoring
- ⚖️ Hierarchical Risk Parity (HRP) Optimal Allocation
- 💼 Real-time Multi-Asset Paper Trading Execution
"""
import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.alpha_engine import AlphaEnginePipeline
from src.manifold.lake_dataset_loader import LakeDatasetLoader


def get_default_tickers() -> List[str]:
    """Retrieve available ETF lake tickers or safe defaults."""
    try:
        loader = LakeDatasetLoader()
        available = loader.list_available_tickers()
        valid = [t for t in available if "BEES" in t or "NIFTY" in t]
        return valid[:4] if valid else ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"]
    except Exception:
        return ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"]


def render_pipeline_summary(report: Dict[str, Any]) -> str:
    """Format pipeline execution metrics into structured, human-readable markdown."""
    if not report.get("success", False):
        error_msg = report.get("error", "Unknown error occurred.")
        return f"### ❌ Pipeline Execution Failed\n> {error_msg}"

    equity = report.get("portfolio_equity", 0.0)
    ratio = report.get("compression_ratio", 0.0) * 100
    noise_count = report.get("orthogonal_count", 0)
    cov = report.get("archive_coverage", 0.0) * 100
    candidates = report.get("discovered_candidates", 0)
    champions = report.get("pruned_champions", [])
    weights = report.get("allocated_weights", {})

    lines = [
        "## 🧬 Alpha-OS Cycle Complete",
        f"- **Total Portfolio Equity**: `₹{equity:,.2f}`",
        f"- **Clone Compression Ratio**: `{ratio:.1f}%` ({candidates} candidates pruned to {len(champions)} champions)",
        f"- **Orthogonal Alphas Discovered**: `{noise_count}` pure uncorrelated signals",
        f"- **CVT Archive Cell Coverage**: `{cov:.1f}%`",
        "\n### ⚖️ HRP Target Allocations:",
    ]
    for ticker, w in weights.items():
        lines.append(f"- **{ticker}**: `{w * 100:.1f}%`")

    lines.append("\n### 🏆 Top De-cloned Champion Formula ASTs:")
    for i, c in enumerate(champions[:5], 1):
        lines.append(f"{i}. `{c}`")
    return "\n".join(lines)


def build_3d_manifold_plot(coords: Optional[np.ndarray] = None, labels: Optional[List[int]] = None, texts: Optional[List[str]] = None) -> go.Figure:
    """Render interactive 3D UMAP manifold scatter plot."""
    fig = go.Figure()
    if coords is not None and len(coords) > 0 and coords.shape[1] >= 3:
        colors = labels if labels is not None else [0] * len(coords)
        hover = texts if texts is not None else [f"Alpha #{i}" for i in range(len(coords))]
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=6, color=colors, colorscale="Viridis", opacity=0.85, showscale=True),
            text=hover,
            hoverinfo="text"
        ))
        fig.update_layout(
            title="3D UMAP Phenotypic Manifold (HDBSCAN Clustered)",
            scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
            margin=dict(l=0, r=0, b=0, t=30),
            template="plotly_dark"
        )
    else:
        fig.add_annotation(text="No Manifold Data Yet — Launch Cycle Below", showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(template="plotly_dark", height=450)
    return fig


def build_hrp_weights_plot(weights: Dict[str, float]) -> go.Figure:
    """Generate bar chart of Hierarchical Risk Parity capital distribution."""
    fig = go.Figure()
    if weights:
        tickers = list(weights.keys())
        allocations = [weights[t] * 100 for t in tickers]
        fig.add_trace(go.Bar(
            x=tickers,
            y=allocations,
            marker=dict(color=["#00CC96", "#636EFA", "#EF553B", "#AB63FA"][:len(tickers)])
        ))
        fig.update_layout(
            title="HRP Optimal Capital Allocation (%)",
            yaxis_title="Allocation %",
            yaxis_range=[0, 100],
            template="plotly_dark",
            margin=dict(l=20, r=20, t=35, b=20),
            height=320
        )
    else:
        fig.update_layout(template="plotly_dark", height=320)
    return fig


def run_pipeline_callback(tickers: List[str], generations: int, population: int, initial_cash: float):
    """Execute complete alpha engine pipeline and return updated UI components."""
    if not tickers:
        tickers = ["NIFTYBEES.NS", "BANKBEES.NS"]
    pipeline = AlphaEnginePipeline(initial_cash=float(initial_cash))
    report = pipeline.run_pipeline_cycle(
        target_tickers=tickers,
        generations=int(generations),
        population_size=int(population)
    )
    summary_md = render_pipeline_summary(report)
    weights_fig = build_hrp_weights_plot(report.get("allocated_weights", {}))
    
    # Virtual paper portfolio table
    trader = pipeline.paper_trader
    holdings = []
    for t in tickers:
        qty = trader.positions.get(t, 0)
        px = trader.latest_prices.get(t, 0.0)
        holdings.append({"Ticker": t, "Lots": qty, "Price (₹)": f"₹{px:.2f}", "Value (₹)": f"₹{qty * px:,.2f}"})
    df_holdings = pd.DataFrame(holdings)

    # 3D Manifold visualization from pipeline results
    pruner_res = report.get("pruning_results", {})
    coords = pruner_res.get("embeddings")
    labels = pruner_res.get("labels")
    champions = report.get("pruned_champions", [])
    manifold_fig = build_3d_manifold_plot(coords=coords, labels=labels, texts=champions)

    return summary_md, manifold_fig, weights_fig, df_holdings


def create_dashboard_app() -> gr.Blocks:
    """Construct Gradio UI Blocks for Bio-Quant Alpha-OS v2.0."""
    default_tickers = get_default_tickers()
    with gr.Blocks(title="Bio-Quant Alpha-OS v2.0 Control Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧬 Bio-Quant Alpha-OS v2.0 Control Studio")
        gr.Markdown("*High-Dimensional Phenotypic Vector Space • Mega Evolution • UMAP 3D • HRP Allocation*")

        with gr.Row():
            ticker_sel = gr.CheckboxGroup(
                choices=default_tickers,
                value=default_tickers[:2],
                label="Target Liquid ETF Lake Assets"
            )
            gen_slider = gr.Slider(minimum=2, maximum=100, value=5, step=1, label="Evolution Generations")
            pop_slider = gr.Slider(minimum=10, maximum=500, value=20, step=5, label="Gene Pool Population")
            cash_input = gr.Number(value=50000.0, label="Initial Paper Capital (₹)")

        btn_run = gr.Button("🚀 Launch Autonomous Alpha Cycle", variant="primary")
        summary_box = gr.Markdown("### ⏳ System Ready. Click Launch to begin evolutionary cycle.")

        with gr.Tabs():
            with gr.Tab("🌌 3D Phenotype Manifold (UMAP & HDBSCAN)"):
                manifold_plot = gr.Plot(value=build_3d_manifold_plot())
            with gr.Tab("⚖️ Hierarchical Risk Parity (HRP) Allocation"):
                hrp_plot = gr.Plot(value=build_hrp_weights_plot({}))
            with gr.Tab("💼 Multi-Asset Virtual Portfolio"):
                portfolio_table = gr.Dataframe(headers=["Ticker", "Lots", "Price (₹)", "Value (₹)"])

        btn_run.click(
            fn=run_pipeline_callback,
            inputs=[ticker_sel, gen_slider, pop_slider, cash_input],
            outputs=[summary_box, manifold_plot, hrp_plot, portfolio_table]
        )
    return demo


if __name__ == "__main__":
    app = create_dashboard_app()
    app.launch(server_name="127.0.0.1", server_port=7862)
