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
