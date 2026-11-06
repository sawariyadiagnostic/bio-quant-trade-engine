"""
Interactive Visual Space Engine for Bio-Quant Evolution Studio.
Provides:
1. 2D MAP-Elites 400-Niche Phenotypic Heatmap (Plotly).
2. Risk-Reward Pareto Optimal Frontier Scatter Plot (Plotly).
3. Resident Champions Spotlight Metric Cards (Markdown).
4. Multi-Metric Sortable Leaderboard (Pandas DataFrame).
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

VOL_BINS = np.linspace(0.05, 0.60, 20)
TURNOVER_BINS = np.linspace(0.5, 25.0, 20)

EMPTY_LEADERBOARD = pd.DataFrame([{
    "Rank": "-",
    "Net Return %": "Pending...",
    "Annual CAGR %": "-",
    "Max Drawdown %": "-",
    "CIAS Score": "-",
    "Win Rate %": "-",
    "Sortino": "-",
    "Profit Factor": "-",
    "Turnover": "-",
    "Trades": "-",
    "Formula AST Expression": "Launch evolution to discover institutional alpha genomes"
}])

def build_map_elites_heatmap(records: List[Dict[str, Any]], color_metric: str = "net_return_pct") -> go.Figure:
    """
    Renders an interactive 20x20 MAP-Elites phenotypic illumination heatmap.
    Rows = Annualized Volatility (5% to 60%).
    Cols = Annualized Turnover (0.5x to 25.0x).
    Uses pure Python types (lists/None/floats) to guarantee JSON serializability in Gradio.
    """
    z_matrix: List[List[Optional[float]]] = [[None for _ in range(20)] for _ in range(20)]
    hover_matrix: List[List[str]] = [["" for _ in range(20)] for _ in range(20)]

    for r in range(20):
        for c in range(20):
            hover_matrix[r][c] = (
                f"<b>Niche [{r}, {c}]</b>: Empty (Unilluminated)<br>"
                f"<i>Vol {float(VOL_BINS[r])*100:.0f}% · Turn {float(TURNOVER_BINS[c]):.1f}x</i>"
            )

    valid_vals = []
    for item in records:
        r = int(np.clip(item.get("row", 0), 0, 19))
        c = int(np.clip(item.get("col", 0), 0, 19))
        val = float(item.get("net_return_pct" if color_metric == "net_return_pct" else "fitness", 0.0))
        z_matrix[r][c] = val
        valid_vals.append(val)

        f_str = str(item.get("formula_str", ""))
        if len(f_str) > 60:
            f_str = f_str[:57] + "..."

        hover_matrix[r][c] = (
            f"<b>🧬 Niche [{r}, {c}]</b>: Vol {float(VOL_BINS[r])*100:.0f}% | Turn {float(TURNOVER_BINS[c]):.1f}x<br>"
            f"<b>Net Return:</b> +{float(item.get('net_return_pct', 0.0)):,.1f}%<br>"
            f"<b>Max Drawdown:</b> {float(item.get('max_drawdown', 0.0)):.1f}%<br>"
            f"<b>Institutional CIAS:</b> {float(item.get('fitness', 0.0)):.2f}<br>"
            f"<b>Win Rate:</b> {float(item.get('win_rate', 0.0)):.1f}% | <b>Trades:</b> {int(item.get('trades_count', 0))}<br>"
            f"<b>Formula:</b> <code>{f_str}</code>"
        )

    occupied_count = len(records)
    occupancy_pct = float(occupied_count / 400.0) * 100.0

    x_labels = [f"{float(t):.1f}x" for t in TURNOVER_BINS]
    y_labels = [f"{float(v)*100:.0f}%" for v in VOL_BINS]

    metric_title = "Net Return %" if color_metric == "net_return_pct" else "CIAS Score"
    colorscale = "Viridis" if color_metric == "net_return_pct" else "Plasma"

    zmin = float(min(valid_vals)) if valid_vals else 0.0
    zmax = float(max(valid_vals)) if valid_vals else 100.0
    if zmin == zmax:
        zmax = zmin + 1.0

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        text=hover_matrix,
        hovertemplate="%{text}<extra></extra>",
        colorscale=colorscale,
        colorbar=dict(title=dict(text=metric_title, font=dict(color="#E5E7EB"))),
        showscale=True,
        xgap=2,
        ygap=2,
        zmin=zmin,
        zmax=zmax
    ))

    fig.update_layout(
        title=dict(
            text=f"🗺️ <b>MAP-Elites 400-Niche Behavioral Space</b> ({occupied_count}/400 Niches Illuminated · {occupancy_pct:.1f}% Occupancy)",
            font=dict(color="#F9FAFB", size=14)
        ),
        xaxis=dict(
            title=dict(text="Annualized Turnover Niche (Trading Churn ➡️)", font=dict(color="#9CA3AF", size=11)),
            tickfont=dict(color="#9CA3AF", size=9),
            gridcolor="rgba(255, 255, 255, 0.05)"
        ),
        yaxis=dict(
            title=dict(text="Annualized Volatility Niche (Market Risk ⬆️)", font=dict(color="#9CA3AF", size=11)),
            tickfont=dict(color="#9CA3AF", size=9),
            gridcolor="rgba(255, 255, 255, 0.05)"
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        margin=dict(l=55, r=30, t=50, b=45),
        height=380
    )

    if occupied_count == 0:
        fig.add_annotation(
            text="Archive Unpopulated. Click 'Launch Live Evolution' to illuminate niches.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#9CA3AF")
        )

    return fig

def build_pareto_scatter(records: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders an interactive Risk-Reward Pareto Optimal Frontier scatter plot.
    X-axis = Maximum Drawdown % (Risk - Lower is better).
    Y-axis = Net Return % (Reward - Higher is better).
    Bubble Color = CIAS Institutional Score.
    Bubble Size = Win Rate %.
    Highlights non-dominated Pareto Champions with star markers and step-line.
    """
    fig = go.Figure()

    if not records:
        fig.update_layout(
            title=dict(text="⚡ <b>Risk-Reward Pareto Optimal Frontier</b> (Pending Evolution...)", font=dict(color="#F9FAFB", size=14)),
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#1F2937",
            margin=dict(l=55, r=30, t=50, b=45),
            height=380
        )
        fig.add_annotation(
            text="No resident champions yet. Launch evolution to map the Pareto frontier.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#9CA3AF")
        )
        return fig

    x_vals = [r.get("max_drawdown", 0.0) for r in records]
    y_vals = [r.get("net_return_pct", 0.0) for r in records]
    fitness_vals = [r.get("fitness", 0.0) for r in records]
    win_rates = [r.get("win_rate", 50.0) for r in records]
    sizes = [max(10, min(24, int(w / 3.5))) for w in win_rates]

    hover_texts = []
    for r in records:
        f_str = r.get("formula_str", "")
        if len(f_str) > 55:
            f_str = f_str[:52] + "..."
        hover_texts.append(
            f"<b>Formula:</b> <code>{f_str}</code><br>"
            f"<b>Net Return:</b> +{r.get('net_return_pct', 0.0):,.1f}%<br>"
            f"<b>Max Drawdown:</b> {r.get('max_drawdown', 0.0):.1f}%<br>"
            f"<b>CIAS Score:</b> {r.get('fitness', 0.0):.2f}<br>"
            f"<b>Win Rate:</b> {r.get('win_rate', 0.0):.1f}% | <b>Trades:</b> {r.get('trades_count', 0)}<br>"
            f"<b>Turnover:</b> {r.get('turnover', 0.0):.1f}x"
        )

    # General resident bubble trace
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(
            size=sizes,
            color=fitness_vals,
            colorscale="Plasma",
            colorbar=dict(title=dict(text="CIAS Score", font=dict(color="#E5E7EB"))),
            showscale=True,
            line=dict(width=1, color="rgba(255, 255, 255, 0.5)")
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Resident Alpha Genomes"
    ))

    # Calculate Pareto Optimal Frontier (Non-dominated in Drawdown vs Net Return)
    pareto_candidates = []
    for i, p1 in enumerate(records):
        dominated = False
        dd1, ret1 = p1.get("max_drawdown", 999.0), p1.get("net_return_pct", -999.0)
        for j, p2 in enumerate(records):
            if i == j: continue
            dd2, ret2 = p2.get("max_drawdown", 999.0), p2.get("net_return_pct", -999.0)
            if dd2 <= dd1 and ret2 >= ret1 and (dd2 < dd1 or ret2 > ret1):
                dominated = True
                break
        if not dominated:
            pareto_candidates.append(p1)

    # Sort Pareto front by Drawdown ascending
    pareto_sorted = sorted(pareto_candidates, key=lambda x: x.get("max_drawdown", 0.0))
    if len(pareto_sorted) > 1:
        px = [p.get("max_drawdown", 0.0) for p in pareto_sorted]
        py = [p.get("net_return_pct", 0.0) for p in pareto_sorted]
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="lines+markers",
            line=dict(color="#00FFA3", width=2.5, dash="dash"),
            marker=dict(size=14, symbol="star", color="#00FFA3", line=dict(width=1, color="#FFFFFF")),
            name="🏆 Pareto Frontier",
            hoverinfo="skip"
        ))

    fig.update_layout(
        title=dict(
            text=f"⚡ <b>Risk-Reward Pareto Frontier</b> ({len(records)} Genomes · {len(pareto_sorted)} Pareto Champions)",
            font=dict(color="#F9FAFB", size=14)
        ),
        xaxis=dict(
            title=dict(text="Maximum Drawdown % (Risk - Lower is Better ⬅️)", font=dict(color="#9CA3AF", size=11)),
            tickfont=dict(color="#9CA3AF", size=10),
            gridcolor="rgba(255, 255, 255, 0.08)"
        ),
        yaxis=dict(
            title=dict(text="Net Cumulative Return % (Reward - Higher is Better ⬆️)", font=dict(color="#9CA3AF", size=11)),
            tickfont=dict(color="#9CA3AF", size=10),
            gridcolor="rgba(255, 255, 255, 0.08)"
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        margin=dict(l=55, r=30, t=50, b=45),
        height=380,
        legend=dict(
            yanchor="bottom", y=0.02, xanchor="right", x=0.98,
            bgcolor="rgba(17, 24, 39, 0.7)", font=dict(color="#F3F4F6", size=10)
        )
    )

    return fig

def build_spotlight_markdown(records: List[Dict[str, Any]]) -> str:
    """Generates high-contrast spotlight cards highlighting top performers across dimensions."""
    if not records:
        return (
            "### 🌟 Resident Champions Spotlight\n"
            "*Awaiting initial generation... Run evolution to crown champions.*"
        )

    top_earner = max(records, key=lambda x: x.get("net_return_pct", 0.0))
    top_institutional = max(records, key=lambda x: x.get("fitness", 0.0))
    top_accuracy = max(records, key=lambda x: x.get("win_rate", 0.0))

    def fmt_formula(f_str: str) -> str:
        return f"`{f_str[:65]}...`" if len(f_str) > 65 else f"`{f_str}`"

    return (
        "### 🌟 Resident Champions Spotlight (Across All Niches)\n"
        "| 🚀 **Top Net Earner (Max Profit)** | 🛡️ **Top Institutional (Best CIAS Score)** | 🎯 **Top Accuracy (Highest Win Rate)** |\n"
        "| :--- | :--- | :--- |\n"
        f"| **+{top_earner.get('net_return_pct', 0.0):,.1f}%** Cumulative Return | **{top_institutional.get('fitness', 0.0):.2f}** Institutional Score | **{top_accuracy.get('win_rate', 0.0):.1f}%** Win Rate |\n"
        f"| Max DD: `{top_earner.get('max_drawdown', 0.0):.1f}%` · Win: `{top_earner.get('win_rate', 0.0):.1f}%` | Return: `+{top_institutional.get('net_return_pct', 0.0):,.1f}%` · Max DD: `{top_institutional.get('max_drawdown', 0.0):.1f}%` | Return: `+{top_accuracy.get('net_return_pct', 0.0):,.1f}%` · Trades: `{top_accuracy.get('trades_count', 0)}` |\n"
        f"| **AST:** {fmt_formula(top_earner.get('formula_str', ''))} | **AST:** {fmt_formula(top_institutional.get('formula_str', ''))} | **AST:** {fmt_formula(top_accuracy.get('formula_str', ''))} |\n"
    )

def format_leaderboard_table(
    records: List[Dict[str, Any]],
    sort_by: str = "net_return_pct",
    max_rows: int = 50
) -> pd.DataFrame:
    """
    Constructs a formatted, multi-metric sortable DataFrame for the UI.
    """
    if not records:
        return EMPTY_LEADERBOARD

    if sort_by == "cagr_pct":
        sorted_records = sorted(records, key=lambda x: x.get("cagr_pct", x.get("net_return_pct", 0.0)), reverse=True)
    elif sort_by == "net_return_pct":
        sorted_records = sorted(records, key=lambda x: x.get("net_return_pct", 0.0), reverse=True)
    elif sort_by == "fitness":
        sorted_records = sorted(records, key=lambda x: x.get("fitness", 0.0), reverse=True)
    elif sort_by == "max_drawdown":
        sorted_records = sorted(records, key=lambda x: x.get("max_drawdown", 999.0), reverse=False)
    elif sort_by == "win_rate":
        sorted_records = sorted(records, key=lambda x: x.get("win_rate", 0.0), reverse=True)
    elif sort_by == "trades_count":
        sorted_records = sorted(records, key=lambda x: x.get("trades_count", 0), reverse=True)
    else:
        sorted_records = sorted(records, key=lambda x: x.get("cagr_pct", x.get("net_return_pct", 0.0)), reverse=True)

    rows = []
    for rank, r in enumerate(sorted_records[:max_rows], 1):
        cagr_val = r.get("cagr_pct", r.get("net_return_pct", 0.0))
        cagr_str = f"+{cagr_val:,.1f}%" if cagr_val >= 0 else f"{cagr_val:,.1f}%"
        rows.append({
            "Rank": f"#{rank}",
            "Net Return %": f"+{r.get('net_return_pct', 0.0):,.1f}%",
            "Annual CAGR %": cagr_str,
            "Max Drawdown %": f"{r.get('max_drawdown', 0.0):.1f}%",
            "CIAS Score": round(r.get("fitness", 0.0), 2),
            "Win Rate %": f"{r.get('win_rate', 0.0):.1f}%",
            "Sortino": round(r.get("sortino", 0.0), 2),
            "Profit Factor": round(r.get("profit_factor", 0.0), 2),
            "Turnover": f"{r.get('turnover', 0.0):.1f}x",
            "Trades": r.get("trades_count", 0),
            "Formula AST Expression": r.get("formula_str", "")
        })

    return pd.DataFrame(rows) if rows else EMPTY_LEADERBOARD


EMPTY_PORTFOLIO = pd.DataFrame([{
    "Ticker": "-", "Lots": 0, "Price (₹)": "₹0.00", "Value (₹)": "₹0.00", "Weight": "0.0%"
}])


def build_3d_manifold_plot(
    coords: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
    texts: Optional[List[str]] = None
) -> go.Figure:
    """Render interactive 3D UMAP manifold scatter plot with ambient Fog-of-War grid."""
    fig = go.Figure()
    if coords is not None and len(coords) > 0 and coords.shape[1] >= 3:
        colors = labels if labels is not None else [0] * len(coords)
        hover = texts if texts is not None else [f"Alpha #{i}" for i in range(len(coords))]
        # Ambient Fog-of-War bounding shell (unexplored latent space)
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        fig.add_trace(go.Scatter3d(
            x=[x_min, x_max, x_max, x_min, x_min],
            y=[y_min, y_min, y_max, y_max, y_min],
            z=[z_min, z_min, z_min, z_min, z_min],
            mode="lines",
            line=dict(color="rgba(100, 150, 255, 0.25)", width=2),
            name="Explored Perimeter",
            hoverinfo="skip"
        ))
        # Active illuminated strategy clusters
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=7, color=colors, colorscale="Viridis", opacity=0.9, showscale=True),
            text=hover, hoverinfo="text",
            name="Alpha Clusters"
        ))
        fig.update_layout(
            title="🌌 3D Phenotype Strategy Manifold (Fog-of-War & HDBSCAN Clustered)",
            scene=dict(
                xaxis_title="UMAP 1 (Factor Harmonic)",
                yaxis_title="UMAP 2 (Downside Vol)",
                zaxis_title="UMAP 3 (Return Trajectory)",
                bgcolor="rgba(10, 15, 25, 0.95)"
            ),
            margin=dict(l=0, r=0, b=0, t=30), template="plotly_dark", height=440
        )
    else:
        fig.add_annotation(text="🌌 3D Fog-of-War: Launch Engine to Illuminate Uncharted Alpha Sectors", showarrow=False, font=dict(size=13, color="gray"))
        fig.update_layout(template="plotly_dark", height=440)
    return fig


def build_hrp_weights_plot(weights: Dict[str, float]) -> go.Figure:
    """Generate bar chart of Hierarchical Risk Parity capital distribution."""
    fig = go.Figure()
    if weights and any(v > 0 for v in weights.values()):
        tickers = list(weights.keys())
        allocations = [weights[t] * 100 for t in tickers]
        palette = ["#00CC96", "#636EFA", "#EF553B", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
        colors = [palette[i % len(palette)] for i in range(len(tickers))]
        fig.add_trace(go.Bar(
            x=tickers, y=allocations,
            marker=dict(color=colors)
        ))
        fig.update_layout(
            title="HRP Optimal Capital Allocation (%)",
            yaxis_title="Allocation %", yaxis_range=[0, 100],
            template="plotly_dark", margin=dict(l=20, r=20, t=35, b=20), height=320
        )
    else:
        fig.update_layout(template="plotly_dark", height=320)
    return fig


def format_portfolio_table(
    positions: Dict[str, Any],
    prices: Dict[str, float],
    cash: float = 50000.0
) -> pd.DataFrame:
    """Format active positions and ledger into presentation dataframe."""
    rows = []
    total_val = cash
    for sym, pos in positions.items():
        qty = pos.get("quantity", 0) if isinstance(pos, dict) else int(pos)
        px = prices.get(sym, pos.get("avg_price", 0.0) if isinstance(pos, dict) else 0.0)
        val = qty * px
        total_val += val
        rows.append({
            "Ticker": sym, "Lots": qty, "Price (₹)": f"₹{px:,.2f}",
            "Value (₹)": f"₹{val:,.2f}", "Weight": "Pending"
        })
    for r in rows:
        val_num = float(r["Value (₹)"].replace("₹", "").replace(",", ""))
        r["Weight"] = f"{val_num / total_val * 100:.1f}%" if total_val > 0 else "0.0%"
    rows.append({
        "Ticker": "💵 CASH_RESERVE", "Lots": 1, "Price (₹)": f"₹{cash:,.2f}",
        "Value (₹)": f"₹{cash:,.2f}", "Weight": f"{cash / total_val * 100:.1f}%" if total_val > 0 else "100.0%"
    })
    return pd.DataFrame(rows) if rows else EMPTY_PORTFOLIO


def build_cosine_similarity_heatmap(records: List[Dict[str, Any]]) -> go.Figure:
    """Renders a pairwise cosine similarity matrix across top resident champions in 1024D embedding space."""
    fig = go.Figure()
    if records and len(records) >= 2:
        from ..domain.formula_parser import FormulaParser
        from ..clustering.phenotype_extractor import PhenotypeExtractor

        n = min(len(records), 10)
        parsed_nodes = []
        valid_indices = []
        for i in range(n):
            f_str = records[i].get("formula_str", "")
            try:
                node = FormulaParser.parse(f_str)
                parsed_nodes.append(node)
                valid_indices.append(i)
            except Exception:
                pass

        if len(parsed_nodes) >= 2:
            extractor = PhenotypeExtractor.create_mock(n_assets=3, n_bars=100)
            embeddings = extractor.extract_dense_batch(parsed_nodes, target_dim=1024)
            # Dot product of unit-norm vectors = exact cosine similarity
            matrix = np.clip(embeddings @ embeddings.T, 0.0, 1.0)
            names = [f"#{valid_indices[i]+1} {records[valid_indices[i]].get('formula_str', '')[:16]}.." for i in range(len(valid_indices))]
        else:
            matrix = np.eye(n)
            names = [f"#{i+1} {records[i].get('formula_str', '')[:16]}.." for i in range(n)]

        fig.add_trace(go.Heatmap(
            z=matrix, x=names, y=names, colorscale="Viridis", zmin=0.0, zmax=1.0,
            colorbar=dict(title="Cosine Sim")
        ))
        fig.update_layout(
            title="1024D Latent Space Pairwise Cosine Similarity (Protein Fold Analogy)",
            template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40), height=380
        )
    else:
        fig.add_annotation(text="Launch Evolution or Load Archive to Compute 1024D Similarity Matrix", showarrow=False, font=dict(color="gray"))
        fig.update_layout(template="plotly_dark", height=380)
    return fig


def build_avionics_hud_markdown(
    equity: float = 100000.0,
    elites_count: int = 0,
    compression_ratio: float = 0.0,
    sharpe: float = 0.0,
    overheat_level: str = "LOW (Optimal)",
    status_msg: str = "SYSTEM ARMED & READY"
) -> str:
    """Renders a top fixed-aspect avionics annunciator HUD ribbon with live KPI cards."""
    overheat_color = "#00CC96" if "LOW" in overheat_level else ("#FFA500" if "MED" in overheat_level else "#FF3366")
    comp_pct = compression_ratio * 100 if compression_ratio <= 1.0 else compression_ratio
    return f"""
<div style="background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #21262d; padding-bottom: 8px; margin-bottom: 10px;">
    <div style="font-weight: 700; font-size: 1.1rem; color: #58a6ff; letter-spacing: 0.5px;">
      🧬 BIO-QUANT ALPHA-OS <span style="font-size: 0.85rem; color: #8b949e;">v2.0 MASTER COCKPIT</span>
    </div>
    <div style="display: flex; gap: 12px; font-size: 0.8rem; font-family: monospace;">
      <span style="background: #238636; color: #fff; padding: 2px 8px; border-radius: 4px;">🟢 {status_msg}</span>
      <span style="background: #1f6feb; color: #fff; padding: 2px 8px; border-radius: 4px;">⚡ SIMD/CUDA ACTIVE</span>
      <span style="background: #8957e5; color: #fff; padding: 2px 8px; border-radius: 4px;">🛡️ SHOONYA GATEWAY</span>
    </div>
  </div>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: center;">
    <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 6px;">
      <div style="font-size: 0.75rem; color: #8b949e;">TOTAL CAPITAL (₹)</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: #39d353;">₹{equity:,.2f}</div>
    </div>
    <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 6px;">
      <div style="font-size: 0.75rem; color: #8b949e;">ARCHIVE CHAMPIONS</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: #58a6ff;">{elites_count} Elites</div>
    </div>
    <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 6px;">
      <div style="font-size: 0.75rem; color: #8b949e;">CLONE PRUNED %</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: #d2a8ff;">{comp_pct:.1f}%</div>
    </div>
    <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 6px;">
      <div style="font-size: 0.75rem; color: #8b949e;">PORTFOLIO SHARPE</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: #f0883e;">{sharpe:.2f}</div>
    </div>
    <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 6px;">
      <div style="font-size: 0.75rem; color: #8b949e;">OVERHEAT TRIPWIRE</div>
      <div style="font-size: 1.15rem; font-weight: 700; color: {overheat_color};">{overheat_level}</div>
    </div>
  </div>
</div>
"""


def build_combat_telemetry_markdown(title: str, lines: List[str]) -> str:
    """Formats live execution ticks and generational throughput into an RPG combat log style."""
    rendered_lines = "<br>".join([f"› {line}" for line in lines]) if lines else "› Ready to engage master engine."
    return f"""
<div style="background: #090d13; border: 1px solid #30363d; border-left: 4px solid #58a6ff; border-radius: 4px; padding: 8px 12px; font-family: monospace; font-size: 0.85rem; color: #c9d1d9;">
  <div style="font-weight: 600; color: #79c0ff; margin-bottom: 4px;">📜 TELEMETRY COMBAT STRIP: {title}</div>
  <div style="line-height: 1.4;">{rendered_lines}</div>
</div>
"""


def build_pipeline_dag_html() -> str:
    """Renders high-contrast interactive HTML DAG of system architecture and I/O contracts."""
    return """
<div style="background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #c9d1d9;">
  <div style="font-weight: 700; font-size: 1.1rem; color: #58a6ff; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
    <span>⚡</span> SYSTEM PIPELINE WORKFLOW & DYNAMIC I/O CONTRACT DAG
  </div>
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
    <div style="background: #161b22; border: 1px solid #388bfd33; border-top: 3px solid #58a6ff; border-radius: 6px; padding: 10px;">
      <div style="font-size: 0.8rem; font-weight: 700; color: #58a6ff;">1. DATA LAYER</div>
      <div style="font-size: 0.75rem; color: #8b949e; margin: 4px 0;"><b>VectorDataLake & GiftNiftyLoader</b></div>
      <div style="font-size: 0.75rem; color: #7ee787;">IN: NSE Parquet / GIFT Quote</div>
      <div style="font-size: 0.75rem; color: #d2a8ff;">PROC: Session VWAP + Circuit Limits</div>
      <div style="font-size: 0.75rem; color: #f0883e;">OUT: 20+ Normalized Tensors</div>
    </div>
    <div style="background: #161b22; border: 1px solid #388bfd33; border-top: 3px solid #3fb950; border-radius: 6px; padding: 10px;">
      <div style="font-size: 0.8rem; font-weight: 700; color: #3fb950;">2. QUANT ALPHA CORE</div>
      <div style="font-size: 0.75rem; color: #8b949e; margin: 4px 0;"><b>ASTEvaluator & MatrixSimulator</b></div>
      <div style="font-size: 0.75rem; color: #7ee787;">IN: Tensors + Epigenetic Trees</div>
      <div style="font-size: 0.75rem; color: #d2a8ff;">PROC: 3D Tensor (B,N,T) + Purged CV</div>
      <div style="font-size: 0.75rem; color: #f0883e;">OUT: Sortino Floor + DSR Stats</div>
    </div>
    <div style="background: #161b22; border: 1px solid #388bfd33; border-top: 3px solid #bc8cff; border-radius: 6px; padding: 10px;">
      <div style="font-size: 0.8rem; font-weight: 700; color: #bc8cff;">3. RISK & MAP-ELITES</div>
      <div style="font-size: 0.75rem; color: #8b949e; margin: 4px 0;"><b>HRPAllocator & CircuitDamper</b></div>
      <div style="font-size: 0.75rem; color: #7ee787;">IN: Strategy Returns & Covariance</div>
      <div style="font-size: 0.75rem; color: #d2a8ff;">PROC: Ledoit-Wolf + Ward Linkage</div>
      <div style="font-size: 0.75rem; color: #f0883e;">OUT: Risk-Budgeted Weights</div>
    </div>
    <div style="background: #161b22; border: 1px solid #388bfd33; border-top: 3px solid #f0883e; border-radius: 6px; padding: 10px;">
      <div style="font-size: 0.8rem; font-weight: 700; color: #f0883e;">4. EXECUTION TOWER</div>
      <div style="font-size: 0.75rem; color: #8b949e; margin: 4px 0;"><b>ShoonyaGateway & PortfolioPaperTrader</b></div>
      <div style="font-size: 0.75rem; color: #7ee787;">IN: Weights + Pre-Open Auction</div>
      <div style="font-size: 0.75rem; color: #d2a8ff;">PROC: Pass 1 Sells -> Pass 2 Buys</div>
      <div style="font-size: 0.75rem; color: #f0883e;">OUT: Fills + LIQUIDBEES (6.8%) Sweep</div>
    </div>
  </div>
</div>
"""



