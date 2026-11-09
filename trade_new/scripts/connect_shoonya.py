"""
Shoonya (Finvasia) Production Connection & Diagnostic Utility.
Enables real-time 2FA TOTP authentication verification, account margin checking,
and safe configuration of environment credentials without leaking secrets.
"""
import argparse
import getpass
import os
import sys
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv, set_key

# Ensure trade root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout/stderr for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials

ENV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
ENV_EXAMPLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env.example"))

def check_env_status() -> Dict[str, Any]:
    """Inspects .env existence and checks for missing or placeholder credentials."""
    load_dotenv(ENV_FILE, override=True)
    creds = ShoonyaCredentials.from_env()
    
    has_file = os.path.exists(ENV_FILE)
    placeholders = [
        "YourShoonyaPasswordHere", "JBSWY3DPEHPK3PXP",
        "FA12345", "FA12345_U", "your_generated_api_key_here"
    ]
    
    is_placeholder = any(
        getattr(creds, field) in placeholders
        for field in ("user_id", "password", "totp_key", "vendor_code", "api_key")
    )
    
    return {
        "exists": has_file,
        "valid": creds.is_valid() and not is_placeholder,
        "creds": creds,
        "is_placeholder": is_placeholder
    }

def interactive_setup() -> None:
    """Guides the developer through setting up .env securely."""
    print("\n" + "=" * 65)
    print("🔐 SHOONYA (FINVASIA) CREDENTIALS SETUP")
    print("=" * 65)
    print("Your credentials will be stored securely in .env (never committed to git).\n")

    user_id = input("1. Shoonya Client ID (e.g., FA12345): ").strip().upper()
    password = getpass.getpass("2. Shoonya Login Password: ").strip()
    totp_key = input("3. Shoonya Base32 TOTP Secret Key: ").strip().replace(" ", "").upper()
    default_vc = f"{user_id}_U"
    vendor_code = input(f"4. Vendor Code [Default {default_vc}]: ").strip() or default_vc
    api_key = input("5. Shoonya API Key (from Prism portal): ").strip()

    if not os.path.exists(ENV_FILE) and os.path.exists(ENV_EXAMPLE):
        import shutil
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
    elif not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write("# Shoonya Credentials\n")

    set_key(ENV_FILE, "SHOONYA_USER_ID", user_id)
    set_key(ENV_FILE, "SHOONYA_PASSWORD", password)
    set_key(ENV_FILE, "SHOONYA_TOTP_KEY", totp_key)
    set_key(ENV_FILE, "SHOONYA_VENDOR_CODE", vendor_code)
    set_key(ENV_FILE, "SHOONYA_API_KEY", api_key)

    print("\n✅ Successfully updated .env with your Shoonya credentials!\n")

def test_totp_display(creds: ShoonyaCredentials) -> str:
    """Calculates and displays current live 6-digit TOTP with remaining valid seconds."""
    import pyotp
    clean_key = creds.totp_key.replace(" ", "").upper()
    totp = pyotp.TOTP(clean_key)
    now = time.time()
    remaining = 30 - int(now % 30)
    current_code = totp.now()
    print(f"🔑 Live Dynamic TOTP: {current_code}  (Valid for {remaining}s)")
    return current_code

def stream_totp(creds: ShoonyaCredentials) -> None:
    """Live terminal ticker updating dynamic TOTP code and countdown every second."""
    import pyotp
    clean_key = creds.totp_key.replace(" ", "").upper()
    totp = pyotp.TOTP(clean_key)
    print("\n" + "=" * 65)
    print("⏱️ LIVE DYNAMIC 2FA TOTP STREAM (Press Ctrl+C to stop)")
    print("=" * 65)
    try:
        while True:
            now = time.time()
            rem = 30 - int(now % 30)
            code = totp.now()
            bar = "█" * (rem // 2) + "░" * (15 - rem // 2)
            sys.stdout.write(f"\r🔑 TOTP: \033[1;32m{code}\033[0m | [{bar}] {rem:2d}s remaining ")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopped TOTP monitor.\n")

def extract_code(raw_input: str) -> str:
    """Extracts authorization code from raw string or redirect URL."""
    val = raw_input.strip()
    if "code=" in val:
        import urllib.parse
        parsed = urllib.parse.urlparse(val)
        qs = urllib.parse.parse_qs(parsed.query)
        if "code" in qs and qs["code"]:
            return qs["code"][0].strip()
    return val

def handle_oauth_exchange(gw: ShoonyaGateway, code: str) -> bool:
    """Exchanges code for access token and prints account limits."""
    clean_code = extract_code(code)
    print(f"\nExchanging authorization code with Shoonya GenAcsTok...")
    res = gw.exchange_auth_code(clean_code)
    if res.get("stat") != "Ok":
        print(f"❌ Token exchange failed: {res.get('emsg', 'Unknown error')}")
        return False

    print("🟢 Access token acquired and saved to .env!")
    return print_live_limits(gw)

def print_live_limits(gw: ShoonyaGateway) -> bool:
    """Fetches and displays live account margin and cash limits."""
    limits = gw.get_limits()
    if limits.get("stat") == "Ok":
        cash = limits.get("cash", "0.0")
        margin_used = limits.get("marginused", "0.0")
        payin = limits.get("payin", "0.0")
        print("\n" + "=" * 65)
        print("📊 LIVE SHOONYA ACCOUNT STATUS & LIMITS:")
        print("=" * 65)
        print(f"   Account / User ID: {gw.creds.user_id}")
        print(f"   Available Cash:    ₹{float(cash):,.2f}")
        print(f"   Margin Utilized:   ₹{float(margin_used):,.2f}")
        print(f"   Today's Pay-in:    ₹{float(payin):,.2f}")
        print("=" * 65 + "\n")
        return True

    print(f"⚠️ Limits query warning: {limits.get('emsg', 'Unknown')}")
    return False

def test_live_connection(creds: ShoonyaCredentials, direct_code: Optional[str] = None) -> bool:
    """Executes live OAuth authentication or displays login flow."""
    print("\n" + "=" * 65)
    print("⚡ SHOONYA LIVE AUTHENTICATION HANDSHAKE")
    print("=" * 65)
    
    masked_uid = f"{creds.user_id[:2]}***{creds.user_id[-1:]}" if len(creds.user_id) >= 3 else "***"
    print(f"Target User ID: {masked_uid} ({creds.user_id})")
    print(f"Vendor Code:    {creds.vendor_code}")
    print(f"Client ID:      {creds.client_id}")
    live_totp = test_totp_display(creds)
    
    gw = ShoonyaGateway(credentials=creds, paper_trading=False)

    if direct_code:
        return handle_oauth_exchange(gw, direct_code)

    # If we already have an active access token in .env, verify it
    if creds.access_token:
        print("\nVerifying existing session token from .env...")
        if print_live_limits(gw):
            return True
        print("Existing session expired or invalid. Re-authenticating via OAuth...")

    # Display OAuth 2.0 instructions
    oauth_url = gw.get_oauth_url()
    print("\n" + "-" * 65)
    print("🌐 STEP 1: OPEN SHOONYA OAUTH URL IN YOUR BROWSER:")
    print(f"\033[1;34m{oauth_url}\033[0m")
    print("-" * 65)
    print("🔑 STEP 2: LOG IN WITH YOUR SHOONYA CREDENTIALS:")
    print(f"   • User ID:  {creds.user_id}")
    print(f"   • Password: (Saved in .env)")
    print(f"   • TOTP:     Use the dynamic code above: {live_totp}")
    print("-" * 65)
    print("📋 STEP 3: CAPTURE REDIRECT:")
    print("   After logging in, your browser redirects to your registered URL.")
    print("   Copy the full redirected URL (or the 'code=' parameter) and paste it below.\n")

    try:
        user_input = input("Paste Redirect URL or Auth Code (or Ctrl+C to exit): ").strip()
        if user_input:
            return handle_oauth_exchange(gw, user_input)
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
    return False

def test_paper_connection() -> None:
    """Runs zero-risk simulation verification."""
    print("\n" + "=" * 65)
    print("🛡️ RUNNING PAPER TRADING GATEWAY VERIFICATION...")
    print("=" * 65)
    gw = ShoonyaGateway(paper_trading=True, initial_balance=10000.0)
    summary = gw.get_status_summary()
    print("Gateway Mode:         ", summary["mode"])
    print("Virtual Capital:      ", f"₹{summary['current_balance']:,.2f}")
    print("Paper Journal Audit:  ", gw.audit_file)
    
    # Simulate test order
    res = gw.place_order(symbol="NIFTYBEES", quantity=10, price=270.0, action="BUY", is_etf=True)
    if res.get("stat") == "Ok":
        print(f"✅ Paper Test Order Executed: {res['action']} 10 NIFTYBEES @ ₹270.00")
        print(f"   Simulated Order ID:  {res['norenordno']}")
        print(f"   Friction Deducted:   ₹{res['friction']:.2f} (1.2 bps ETF statutory)")
        print(f"   Remaining Capital:   ₹{res['balance_after']:,.2f}")
    print("=" * 65 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Shoonya (Finvasia) Connection Utility")
    parser.add_argument("--setup", action="store_true", help="Interactive prompt to enter credentials into .env")
    parser.add_argument("--paper", action="store_true", help="Verify Paper Trading simulation mode")
    parser.add_argument("--live", action="store_true", help="Force live remote exchange authentication")
    parser.add_argument("--totp", action="store_true", help="Stream live dynamic TOTP with second-by-second countdown")
    parser.add_argument("--code", type=str, default=None, help="Directly exchange an authorization code for session token")
    parser.add_argument("--token", type=str, default=None, help="Directly configure an active access token into .env")
    args = parser.parse_args()

    if args.setup:
        interactive_setup()
        return

    if args.paper:
        test_paper_connection()
        return

    status = check_env_status()
    creds = status["creds"]

    if args.token:
        set_key(ENV_FILE, "SHOONYA_ACCESS_TOKEN", args.token.strip())
        print(f"✅ Saved SHOONYA_ACCESS_TOKEN into {ENV_FILE}")
        creds.access_token = args.token.strip()
        gw = ShoonyaGateway(credentials=creds, paper_trading=False)
        print_live_limits(gw)
        return

    if args.totp:
        if not creds.totp_key:
            print("❌ SHOONYA_TOTP_KEY is not configured in .env")
            return
        stream_totp(creds)
        return

    if not status["exists"] or not status["valid"]:
        print("\n" + "=" * 65)
        print("📋 SHOONYA CREDENTIALS CONFIGURATION STATUS")
        print("=" * 65)
        def mark(val: str) -> str:
            return "✅ Configured" if val and val not in ("YourShoonyaPasswordHere", "JBSWY3DPEHPK3PXP", "FA12345", "FA12345_U", "your_generated_api_key_here") else "❌ Missing"

        print(f"  1. Client ID (User ID):  {creds.user_id or 'Missing'}  [{mark(creds.user_id)}]")
        print(f"  2. Password / MPIN:      {'*' * len(creds.password) if creds.password else 'Missing'}  [{mark(creds.password)}]")
        print(f"  3. Vendor Code:          {creds.vendor_code or 'Missing'}  [{mark(creds.vendor_code)}]")
        print(f"  4. Client ID (OAuth):    {creds.client_id or 'Missing'}  [{mark(creds.client_id)}]")
        print(f"  5. Secret Code (OAuth):  {creds.secret_code[:6] + '...' if len(creds.secret_code) > 6 else 'Missing'}  [{mark(creds.secret_code)}]")
        print(f"  6. TOTP Key (Google 2FA):{creds.totp_key[:4] + '...' if len(creds.totp_key) > 4 else 'Missing'}  [{mark(creds.totp_key)}]")
        
        print("\nNext steps to complete connection:")
        print("  • Tap 'Get API Key' at the bottom right of your Prism screen to copy your API Key.")
        print("  • Run:  python scripts/connect_shoonya.py --setup   (or edit .env)")
        print("  • Or run:  python scripts/connect_shoonya.py --paper (for zero-risk paper trading)")
        print("=" * 65 + "\n")
        return

    test_live_connection(status["creds"], direct_code=args.code)

if __name__ == "__main__":
    main()
