"""Fetch today's live 1m and 5m market data for all 10 Sector BeES ETFs and NIFTY 50."""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS: Dict[str, str] = {
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


def _enrich_intraday_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich intraday DataFrame with ATR_14, VWAP, VR_20, OFI_5, and BB_PCT."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    res = df[[c for c in cols if c in df.columns]].copy()

    # True Range & ATR 14
    prev_close = res["Close"].shift(1)
    tr = np.maximum(
        res["High"] - res["Low"],
        np.maximum(
            np.abs(res["High"] - prev_close),
            np.abs(res["Low"] - prev_close),
        ),
    )
    res["ATR_14"] = tr.rolling(window=14, min_periods=1).mean()

    # Session VWAP
    typical_px = (res["High"] + res["Low"] + res["Close"]) / 3.0
    cum_pv = (typical_px * res["Volume"]).cumsum()
    cum_vol = res["Volume"].cumsum()
    res["VWAP"] = cum_pv / (cum_vol + 1e-6)

    # Variance Ratio 20
    log_ret_1 = np.log(res["Close"] / res["Close"].shift(1).replace(0.0, np.nan)).fillna(0.0)
    log_ret_k = np.log(res["Close"] / res["Close"].shift(5).replace(0.0, np.nan)).fillna(0.0)
    var_1 = log_ret_1.rolling(20, min_periods=5).var()
    var_k = log_ret_k.rolling(20, min_periods=5).var()
    res["VR_20"] = (var_k / (5.0 * var_1 + 1e-8)).fillna(1.0)

    # OFI proxy
    dp = res["Close"].diff().fillna(0.0)
    vol = res["Volume"]
    res["OFI_5"] = (np.sign(dp) * vol).rolling(5, min_periods=1).mean() / (
        vol.rolling(5, min_periods=1).mean() + 1e-6
    )

    # Bollinger Bands % (20, 2)
    bb_mean = res["Close"].rolling(20, min_periods=5).mean()
    bb_std = res["Close"].rolling(20, min_periods=5).std()
    bb_upper = bb_mean + (2.0 * bb_std)
    bb_lower = bb_mean - (2.0 * bb_std)
    bw = (bb_upper - bb_lower).replace(0.0, np.nan)
    res["BB_PCT"] = ((res["Close"] - bb_lower) / bw).fillna(0.5).clip(0.0, 1.0)

    return res


def fetch_today_all() -> Dict[str, pd.DataFrame]:
    """Download today's live 5m candles from NSE and enrich technicals."""
    today_dfs: Dict[str, pd.DataFrame] = {}
    print("Downloading today's live 5m candles from NSE...")
    for sym, yf_sym in TICKERS.items():
        try:
            df = yf.download(yf_sym, period="1d", interval="5m", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            enriched = _enrich_intraday_technicals(df)
            today_dfs[sym] = enriched
            print(f"  [OK 5m] {sym:<12}: {len(enriched)} bars (Latest: {enriched.index[-1].strftime('%H:%M')})")
        except Exception as exc:
            print(f"  [FAIL 5m] {sym}: {exc}")
    return today_dfs


def fetch_today_1m_all() -> Dict[str, pd.DataFrame]:
    """Download today's live 1m microstructure candles from NSE and enrich technicals."""
    today_1m_dfs: Dict[str, pd.DataFrame] = {}
    print("Downloading today's live 1m microstructure candles from NSE...")
    for sym, yf_sym in TICKERS.items():
        try:
            df = yf.download(yf_sym, period="1d", interval="1m", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            enriched = _enrich_intraday_technicals(df)
            today_1m_dfs[sym] = enriched
            print(f"  [OK 1m] {sym:<12}: {len(enriched)} bars (Latest: {enriched.index[-1].strftime('%H:%M')})")
        except Exception as exc:
            print(f"  [FAIL 1m] {sym}: {exc}")
    return today_1m_dfs


def fetch_daily_all(period: str = "3mo") -> Dict[str, pd.DataFrame]:
    """Fetch daily bars for all 10 Sector ETFs + NIFTY 50 with EMA_50 and VR_20."""
    daily_dfs: Dict[str, pd.DataFrame] = {}
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
    dfs_5m = fetch_today_all()
    print(f"Successfully fetched {len(dfs_5m)} 5m instruments for today.")
    dfs_1m = fetch_today_1m_all()
    print(f"Successfully fetched {len(dfs_1m)} 1m instruments for today.")
