"""
OLA ELECTRIC MOBILITY (OLAELEC.NS) ALGO TEST
Tests trend-following logic and risk management on Ola Electric.
"""
import yfinance as yf
import pandas as pd

SYMBOL = "OLAELEC.NS"

print("=" * 60)
print("   ANALYZING OLA ELECTRIC MOBILITY (OLAELEC.NS)")
print("=" * 60)

# 1. Fetch historical data from NSE
df = yf.download(SYMBOL, period="6mo", interval="1d", progress=False, auto_adjust=False)

if df.empty:
    print(f"[ERROR] Could not fetch data for {SYMBOL}. Check your internet connection.")
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# 2. Calculate Indicators
# 20-Day Exponential Moving Average (Trend Filter)
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

# RSI (Momentum Strength)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# 3. Read the Latest Candle
latest = df.iloc[-1]
prev = df.iloc[-2]

latest_close = float(latest["Close"])
ema_20 = float(latest["EMA_20"])
rsi = float(latest["RSI"])

print(f"Current Market Price: ₹{latest_close:.2f}")
print(f"20-Day Average (EMA): ₹{ema_20:.2f}")
print(f"RSI Momentum (14):    {rsi:.1f}")
print("-" * 60)

# 4. Algo Decision Rules
# RULE: Only buy if Price is ABOVE 20-day EMA and RSI is healthy (> 50)
if latest_close > ema_20 and rsi > 50:
    print(">>> ALGO SIGNAL: [BUY / UPTREND] <<<")
    stop_loss = round(latest_close * 0.985, 2)  # 1.5% Stop Loss
    target = round(latest_close * 1.03, 2)      # 3.0% Target
    print(f"Suggested Entry:     ₹{latest_close:.2f}")
    print(f"Hard Stop-Loss:      ₹{stop_loss}  (Exit if price falls here)")
    print(f"Profit Target:       ₹{target}  (Exit to lock profit)")
else:
    print(">>> ALGO SIGNAL: [NO TRADE / DOWNTREND] <<<")
    print("Reason: Price is below its 20-day average or momentum is weak.")
    print("The algo will NOT buy right now. It waits patiently for a confirmed trend.")

print("=" * 60)