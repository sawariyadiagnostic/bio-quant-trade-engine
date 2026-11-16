"""Fetch today's live 5m market data for all 10 Sector BeES ETFs and NIFTY 50."""
import os, sys
import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = {
    "NIFTYBEES": "NIFTYBEES.NS",
    "BANKBEES": "BANKBEES.NS",
    "ITBEES": "ITBEES.NS",
    "AUTOBEES": "AUTOBEES.NS",
    "PHARMABEES": "PHARMABEES.NS",
    "PSUBNKBEES": "PSUBNKBEES.NS",
    "CPSEETF": "CPSEETF.NS",
    "GOLDBEES": "GOLDBEES.NS",
    "SILVERBEES": "SILVERBEES.NS",
    "JUNIORBEES": "JUNIORBEES.NS",
    "NSEI": "^NSEI",
}

def fetch_today_all() -> dict:
    today_dfs = {}
    print("Downloading today's live 5m candles from NSE...")
    for sym, yf_sym in TICKERS.items():
        try:
            df = yf.download(yf_sym, period="1d", interval="5m", progress=False)
            if df.empty:
                print(f"Warning: No data for {sym}")
                continue
            # Flatten multiindex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            # Ensure proper columns
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in cols if c in df.columns]].copy()
            
            # Compute technicals needed by triggers: ATR_14, VWAP, VR_20, OFI_5
            tr = np.maximum(
                df["High"] - df["Low"],
                np.maximum(
                    np.abs(df["High"] - df["Close"].shift(1)),
                    np.abs(df["Low"] - df["Close"].shift(1))
                )
            )
            df["ATR_14"] = tr.rolling(window=14, min_periods=1).mean()
            
            # Session VWAP
            typical_px = (df["High"] + df["Low"] + df["Close"]) / 3.0
            cum_pv = (typical_px * df["Volume"]).cumsum()
            cum_vol = df["Volume"].cumsum()
            df["VWAP"] = cum_pv / (cum_vol + 1e-6)
            
            # Variance Ratio 20
            log_ret_1 = np.log(df["Close"] / df["Close"].shift(1).replace(0.0, np.nan)).fillna(0.0)
            log_ret_k = np.log(df["Close"] / df["Close"].shift(5).replace(0.0, np.nan)).fillna(0.0)
            var_1 = log_ret_1.rolling(20, min_periods=5).var()
            var_k = log_ret_k.rolling(20, min_periods=5).var()
            df["VR_20"] = (var_k / (5.0 * var_1 + 1e-8)).fillna(1.0)
            
            # OFI proxy
            dp = df["Close"].diff().fillna(0.0)
            vol = df["Volume"]
            df["OFI_5"] = (np.sign(dp) * vol).rolling(5, min_periods=1).mean() / (vol.rolling(5, min_periods=1).mean() + 1e-6)
            
            today_dfs[sym] = df
            print(f"  [OK] {sym:<12}: {len(df)} bars (Latest: {df.index[-1].strftime('%H:%M')} | Close: {df['Close'].iloc[-1]:.2f})")
        except Exception as exc:
            print(f"  [FAIL] {sym}: {exc}")
    return today_dfs


def fetch_daily_all(period: str = "3mo") -> dict:
    """Fetch daily bars for all 10 Sector ETFs + NIFTY 50 with EMA_50 and VR_20."""
    daily_dfs = {}
    for sym, yf_sym in TICKERS.items():
        try:
            df = yf.download(yf_sym, period=period, interval="1d", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in cols if c in df.columns]].copy()
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            log_ret_1 = np.log(df["Close"] / df["Close"].shift(1).replace(0.0, np.nan)).fillna(0.0)
            log_ret_k = np.log(df["Close"] / df["Close"].shift(5).replace(0.0, np.nan)).fillna(0.0)
            var_1 = log_ret_1.rolling(20, min_periods=5).var()
            var_k = log_ret_k.rolling(20, min_periods=5).var()
            df["VR_20"] = (var_k / (5.0 * var_1 + 1e-8)).fillna(1.0)
            daily_dfs[sym] = df
        except Exception:
            pass
    return daily_dfs


if __name__ == "__main__":
    dfs = fetch_today_all()
    print(f"Successfully fetched {len(dfs)} instruments for today.")
