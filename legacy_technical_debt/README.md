# Legacy Technical Debt Archive

This directory preserves all legacy, deprecated, and superseded modules from the earlier single-threaded Gradio architecture.

### Contents:
- `legacy_gradio_dashboard.py`: Monolithic 1,259-line Gradio WebUI.
- `legacy_matplotlib_visualizations.py`: Backend Matplotlib/Seaborn image generators.
- `evolution_dashboard_redirect_stub.py`: Migration launcher stub.
- `shoonya_ui_gradio_tab.py`: Deprecated Gradio broker UI tab.
- `rescue_tui.py`: Legacy terminal curses UI.
- `plotly_partial_basic.min.js`: 2D-only Plotly bundle without 3D/Heatmap.
- `legacy_json_archives/`: Historical JSON experiment dumps (now unified into `audit/alpha_elites.db` SQLite WAL).

**Note:** None of the files in this directory are imported or executed by the clean production engine.
