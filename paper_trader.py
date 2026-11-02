"""
FREE INDIAN EQUITY PAPER TRADING SIMULATOR
No API keys needed. No real money used.
Scans India's top companies and simulates trade signals.
"""
import yfinance as yf
import pandas as pd
from tabulate import tabulate

# Top 5 resilient Indian companies (NSE)
STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

VIRTUAL_CAPITAL = 50000.0  # Starting with ₹50,000 virtual cash
MAX_RISK_PER_TRADE = 500.0 # Maximum virtual loss allowed per trade (1%)

print("=" * 65)
print("   INDIAN BLUE-CHIP ALGO SCANNER (SIMULATION MODE)")
print(f"   Virtual Capital: ₹{VIRTUAL_CAPITAL:,.2f}")
print("=" * 65)

results = []

for name, ticker in STOCKS.items():
    try:
        # Fetch last 30 days of data for the stock
        data = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=False)
        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        # Calculate 20-day Average Price
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        latest_close = float(data["Close"].iloc[-1])
        ema_20 = float(data["EMA_20"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2])

        # Check if the stock is in an uptrend (Price above 20 EMA)
        if latest_close > ema_20 and prev_close <= ema_20:
            status = "BUY SIGNAL"
            stop_loss = round(latest_close * 0.985, 2)  # 1.5% below entry
            target = round(latest_close * 1.03, 2)      # 3.0% above entry
            
            # Position sizing: how many shares to buy
            risk_per_share = latest_close - stop_loss
            qty = int(MAX_RISK_PER_TRADE // risk_per_share) if risk_per_share > 0 else 0
        elif latest_close > ema_20:
            status = "HOLD / IN TREND"
            stop_loss = round(latest_close * 0.985, 2)
            target = round(latest_close * 1.03, 2)
            qty = 0
        else:
            status = "NO TRADE (Downtrend)"
            stop_loss = "-"
            target = "-"
            qty = 0

        results.append([
            name, 
            f"₹{latest_close:.2f}", 
            f"₹{ema_20:.2f}", 
            status, 
            qty, 
            f"₹{stop_loss}" if stop_loss != "-" else "-", 
            f"₹{target}" if target != "-" else "-"
        ])

    except Exception as e:
        print(f"Could not fetch {name}: {e}")

# Display results in a clean table
headers = ["Stock", "Price", "20-Day Avg", "Action", "Qty to Buy", "Stop Loss", "Target"]
print(tabulate(results, headers=headers, tablefmt="grid"))
print("\n[INFO] This is a simulation. No real money or orders were placed.")