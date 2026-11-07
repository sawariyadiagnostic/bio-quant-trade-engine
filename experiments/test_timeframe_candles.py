"""
Empirical Evaluation of Candle Timeframes for NSE India:
Comparing Daily (1D) vs 1-Hour (1H) vs Multi-Hour (Half-Day / Session) candles.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate

def analyze_timeframes(ticker: str = "RELIANCE.NS"):
    t = yf.Ticker(ticker)
    
    # 1. Fetch 1D (2 years)
    df_1d = t.history(period="2y", interval="1d")
    
    # 2. Fetch 1H (1 year - max reliable Yahoo limit)
    df_1h = t.history(period="1y", interval="1h")
    
    # 3. Resample to 3H/Half-Day session bars
    # Using 3h resample to split the 6h15m Indian trading day into 2 main sessions
    df_session = df_1h.resample("3h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    df_session = df_session[df_session["Volume"] > 0]

    # Calculate metrics
    results = []
    for name, df in [("Daily (1D)", df_1d), ("Session (~3H)", df_session), ("Hourly (1H)", df_1h)]:
        rets = df["Close"].pct_change().dropna()
        ann_factor = 252 if "1D" in name else (252 * 2 if "Session" in name else 252 * 6.25)
        vol = float(rets.std() * np.sqrt(ann_factor))
        
        # Microstructure noise: first-order serial autocorrelation of returns
        autocorr_1 = float(rets.autocorr(lag=1))
        
        # Average bar range
        avg_range_pct = float(((df["High"] - df["Low"]) / df["Close"]).mean() * 100)
        
        results.append([
            name,
            len(df),
            f"{vol:.1%}",
            f"{autocorr_1:+.3f}",
            f"{avg_range_pct:.2f}%",
            "Macro trend & holding" if "1D" in name else ("Intraday swing" if "Session" in name else "Execution timing")
        ])
        
    headers = ["Timeframe", "Bar Count (1Y/2Y)", "Ann Volatility", "Lag-1 Autocorr", "Avg Bar Range %", "Optimal Role"]
    print("\n" + "=" * 90)
    print(f"   NSE CANDLESTICK TIMEFRAME COMPARISON ({ticker})")
    print("=" * 90)
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    analyze_timeframes()
