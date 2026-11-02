"""
Shoonya (Finvasia) UI Component for Gradio Dashboard.
Displays gateway status, live margin limits, and order execution journal.
"""
import json
import os
import sys
from typing import Tuple
import gradio as gr
import pandas as pd
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def get_shoonya_status_md(mode: str) -> str:
    """Generates markdown status summary for Shoonya paper or live gateway."""
    paper = "Paper" in mode
    creds = ShoonyaCredentials.from_env()
    gw = ShoonyaGateway(credentials=creds, paper_trading=paper)
    
    if paper:
        summary = gw.get_status_summary()
        return (
            f"### 🛡️ Shoonya Paper Gateway: ONLINE\n"
            f"- **Execution Mode:** Paper Trading Simulation (0% Capital Risk)\n"
            f"- **Virtual Capital:** ₹{summary['current_balance']:,.2f}\n"
            f"- **Active Positions:** {summary['active_positions_count']}\n"
            f"- **Friction Deductions:** 1.2 bps ETF, 25 bps Equities\n"
            f"- **Audit Journal:** `{os.path.basename(gw.audit_file)}`"
        )
    
    if not creds.is_valid():
        return (
            "### ⚠️ Shoonya Live Gateway: Not Configured\n"
            "Shoonya credentials are not configured in `.env`.\n\n"
            "**To connect your Shoonya account:**\n"
            "Run in terminal: `python scripts/connect_shoonya.py --setup`"
        )
    
    ok = gw.authenticate()
    if not ok:
        totp_code = gw.generate_totp() if creds.totp_key else "N/A"
        oauth_url = gw.get_oauth_url()
        return (
            "### ⚠️ Shoonya Live Gateway: OAuth 2.0 Authorization Required\n"
            f"- **User ID:** `{creds.user_id}` | **Client ID:** `{creds.client_id}`\n"
            f"- **Current Live Dynamic TOTP:** **`{totp_code}`** (auto-generated)\n\n"
            f"**Quick Authorization Steps:**\n"
            f"1. [👉 Click here to open Shoonya OAuth Login]({oauth_url})\n"
            f"2. Log in with your password and dynamic TOTP (`{totp_code}`).\n"
            f"3. Copy the redirected URL (or the `code=` parameter) and paste it below, then click **Connect Live Session**.\n\n"
            f"*(Or in terminal: `python scripts/connect_shoonya.py`)*"
        )
    
    limits = gw.get_limits()
    cash = limits.get("cash", "0.0")
    margin = limits.get("marginused", "0.0")
    masked_uid = f"{creds.user_id[:2]}***{creds.user_id[-1:]}" if len(creds.user_id) >= 3 else "***"
    return (
        f"### 🟢 Shoonya Live Gateway: AUTHENTICATED\n"
        f"- **Client ID:** `{masked_uid}` | **Vendor Code:** `{creds.vendor_code}`\n"
        f"- **Session Protocol:** Noren OAuth 2.0 (Bearer Token Active)\n"
        f"- **Available Cash:** ₹{float(cash):,.2f}\n"
        f"- **Margin Utilized:** ₹{float(margin):,.2f}\n"
        f"- **Exchange Routing:** NSE Live"
    )

def get_shoonya_orders_table() -> pd.DataFrame:
    """Loads recent paper and live orders from persistent audit journal."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "audit"))
    audit_file = os.path.join(base_dir, "shoonya_paper_orders.json")
    cols = ["Order ID", "Symbol", "Action", "Qty", "Price", "Friction", "Balance", "Timestamp"]
    if not os.path.exists(audit_file):
        return pd.DataFrame(columns=cols)
    try:
        with open(audit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame(columns=cols)
        rows = [{
            "Order ID": d.get("norenordno", ""),
            "Symbol": d.get("tsym", ""),
            "Action": d.get("action", ""),
            "Qty": d.get("qty", 0),
            "Price": f"₹{float(d.get('price', 0)):.2f}",
            "Friction": f"₹{float(d.get('friction', 0)):.2f}",
            "Balance": f"₹{float(d.get('balance_after', 0)):.2f}",
            "Timestamp": d.get("timestamp", "")[:19].replace("T", " ")
        } for d in reversed(data[-30:])]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=cols)

def refresh_shoonya(mode: str) -> Tuple[str, pd.DataFrame]:
    """Callback returning updated status markdown and recent order history table."""
    return get_shoonya_status_md(mode), get_shoonya_orders_table()

def submit_auth_code(raw_code: str, mode: str) -> Tuple[str, pd.DataFrame]:
    """Exchanges auth code from UI and refreshes live gateway status."""
    if not raw_code or not raw_code.strip():
        return get_shoonya_status_md(mode), get_shoonya_orders_table()
    creds = ShoonyaCredentials.from_env()
    gw = ShoonyaGateway(credentials=creds, paper_trading=False)
    clean_code = raw_code.strip()
    if "code=" in clean_code:
        import urllib.parse
        parsed = urllib.parse.urlparse(clean_code)
        qs = urllib.parse.parse_qs(parsed.query)
        if "code" in qs and qs["code"]:
            clean_code = qs["code"][0].strip()
    gw.exchange_auth_code(clean_code)
    return get_shoonya_status_md("Live Exchange (Shoonya Noren)"), get_shoonya_orders_table()

def render_shoonya_accordion() -> None:
    """Renders the Shoonya gateway controls and status accordion inside Gradio UI."""
    with gr.Accordion("🏦 Shoonya (Finvasia) Gateway & Order Execution", open=False):
        with gr.Row():
            shoonya_mode = gr.Radio(
                choices=["Paper Trading (0% Risk)", "Live Exchange (Shoonya Noren)"],
                value="Paper Trading (0% Risk)",
                label="Broker Mode",
                scale=2
            )
            check_shoonya_btn = gr.Button("🔄 Test Connection & Refresh Balance", variant="secondary", scale=1)
        
        shoonya_status_display = gr.Markdown(value=get_shoonya_status_md("Paper Trading (0% Risk)"))
        
        with gr.Row():
            auth_code_input = gr.Textbox(
                label="OAuth Redirect URL or Auth Code",
                placeholder="Paste the full redirect URL or the 'code=' query parameter here...",
                scale=3
            )
            submit_code_btn = gr.Button("⚡ Connect Live Session", variant="primary", scale=1)

        gr.Markdown("#### 📜 Recent Order Audit Journal")
        orders_df = gr.DataFrame(value=get_shoonya_orders_table(), label="Executed Orders", wrap=True)
        
        check_shoonya_btn.click(
            fn=refresh_shoonya,
            inputs=[shoonya_mode],
            outputs=[shoonya_status_display, orders_df]
        )
        shoonya_mode.change(
            fn=refresh_shoonya,
            inputs=[shoonya_mode],
            outputs=[shoonya_status_display, orders_df]
        )
        submit_code_btn.click(
            fn=submit_auth_code,
            inputs=[auth_code_input, shoonya_mode],
            outputs=[shoonya_status_display, orders_df]
        )
