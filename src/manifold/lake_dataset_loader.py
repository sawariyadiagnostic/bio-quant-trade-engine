"""
Universal Data Lake Asset Loader & Manifold Feature Enricher (Phase B).
Loads any financial asset from data/lake, aligns with global leads (S&P 500, US ADRs),
and calculates continuous 18D manifold features for evolutionary alpha mining.
"""
import os
import glob
from typing import Optional
import numpy as np
import pandas as pd

class LakeDatasetLoader:
    def __init__(self, lake_dir: str = "data/lake"):
        self.lake_dir = os.path.abspath(lake_dir)
        self._ticker_map = self._scan_lake()

    def _scan_lake(self) -> dict[str, str]:
        """Scans data lake and maps clean ticker names to file paths."""
        ticker_map = {}
        for p in glob.glob(os.path.join(self.lake_dir, "**", "*.parquet"), recursive=True):
            fname = os.path.basename(p)
            ticker = fname.replace(".parquet", "").replace(".NS", "")
            ticker_map[ticker] = p
        return ticker_map

    def list_available_tickers(self) -> list[str]:
        """Returns sorted list of all available assets across all sectors."""
        return sorted(list(self._ticker_map.keys()))

    def load_dataset(
        self,
        ticker: str = "NIFTYBEES",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_years: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Loads asset data, joins with global lead feeds, and builds all 20 structural features.
        Supports custom lookback horizons (e.g. 1, 2, 5, 10, or 20 years).
        """
        clean_ticker = ticker.replace(".parquet", "").replace(".NS", "")
        if clean_ticker not in self._ticker_map:
            raise FileNotFoundError(f"Ticker '{ticker}' not found in data lake. Available: {len(self._ticker_map)} assets.")

        file_path = self._ticker_map[clean_ticker]
        df_raw = pd.read_parquet(file_path).sort_index()
        df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None).normalize()

        # If user requests extended lookback > 5 years (e.g. 10 years), attempt to fetch full historical depth
        if lookback_years and lookback_years > 5.0 and len(df_raw) < int(lookback_years * 240):
            try:
                import yfinance as yf
                sym = f"{clean_ticker}.NS" if not clean_ticker.startswith("^") else clean_ticker
                period = f"{int(lookback_years)}y" if lookback_years <= 10 else "max"
                r = yf.download(sym, period=period, progress=False, auto_adjust=False)
                if not r.empty and len(r) > len(df_raw):
                    if isinstance(r.columns, pd.MultiIndex):
                        r.columns = [col[0] for col in r.columns]
                    if "Adj Close" in r.columns and "Close_Adj" not in r.columns:
                        r["Close_Adj"] = r["Adj Close"]
                    df_raw = r.sort_index()
                    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None).normalize()
            except Exception:
                pass

        if start_date:
            df_raw = df_raw.loc[df_raw.index >= pd.to_datetime(start_date)]
        if end_date:
            df_raw = df_raw.loc[df_raw.index <= pd.to_datetime(end_date)]

        if lookback_years:
            target_bars = int(lookback_years * 252)
            if len(df_raw) > target_bars:
                df_raw = df_raw.iloc[-target_bars:]

        if len(df_raw) < 55:
            raise ValueError(f"Insufficient history for {ticker} ({len(df_raw)} bars). Need at least 55.")

        c = df_raw.get("Close_Adj", df_raw.get("Close", df_raw.get("Close_Raw"))).astype(float)
        h = df_raw["High"].astype(float)
        l = df_raw["Low"].astype(float)
        o = df_raw["Open"].astype(float)
        v = df_raw["Volume"].astype(float).replace(0, 1.0)

        df = pd.DataFrame(index=df_raw.index)
        df["Close_Adj"] = c
        df["Close_Raw"] = c
        df["Open"] = o
        df["High"] = h
        df["Low"] = l
        df["Volume"] = v

        # 1. Structural Spreads & Ranges
        df["Spread"] = (h - l) / (c + 1e-8)
        c_prev = c.shift(1).fillna(c)
        tr_raw = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
        df["True_Range"] = tr_raw / (c + 1e-8)
        tr_mean = df["True_Range"].rolling(20, min_periods=1).mean()
        df["TR_Expansion"] = df["True_Range"] / tr_mean.replace(0, 1.0)

        # 2. Volume & Flow Dynamics
        vol_mean = v.rolling(20, min_periods=1).mean()
        vol_std = v.rolling(20, min_periods=1).std().replace(0, 1.0).fillna(1.0)
        df["Vol_ZScore"] = (v - vol_mean) / vol_std
        df["Vol_Price_Flow"] = c.pct_change().fillna(0.0) * (v / vol_mean.replace(0, 1.0)).clip(0, 5)

        # 3. Moving Averages & Trend Slopes
        ema_10 = c.ewm(span=10, adjust=False).mean()
        ema_20 = c.ewm(span=20, adjust=False).mean()
        df["EMA_Slope"] = (ema_10 - ema_20) / (ema_20 + 1e-8)
        df["Dist_EMA20"] = (c - ema_20) / (ema_20 + 1e-8)

        # 4. VWAP Equilibrium
        pv = (c * v).rolling(20, min_periods=1).sum()
        cv = v.rolling(20, min_periods=1).sum().replace(0, 1.0)
        vwap = pv / cv
        df["Dist_VWAP"] = (c - vwap) / (vwap + 1e-8)

        # 5. Momentum Oscillator (RSI-14)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean().replace(0, 1e-6)
        rs = gain / loss
        df["RSI_14"] = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        # 6. Asymmetric Downside Risk
        ret_neg = c.pct_change().clip(upper=0).rolling(20, min_periods=1).std().replace(0, 1e-6)
        ret_pos = c.pct_change().clip(lower=0).rolling(20, min_periods=1).std().replace(0, 1e-6)
        df["Downside_Vol_Ratio"] = ret_neg / (ret_neg + ret_pos)

        # 7. Volatility Estimators (Parkinson & Garman-Klass)
        hl_ratio = np.log(h / l.replace(0, 1e-6))
        df["Parkinson_Vol"] = np.sqrt(hl_ratio**2 / (4 * np.log(2))).rolling(20, min_periods=1).mean()
        co_ratio = np.log(c / o.replace(0, 1e-6))
        df["Garman_Klass_Vol"] = (0.5 * hl_ratio**2 - (2 * np.log(2) - 1) * co_ratio**2).rolling(20, min_periods=1).mean()

        # 8. Align with Global Lead Feeds (Strict 1-day lag to eliminate lookahead bias)
        sp_path = os.path.join(self.lake_dir, "Global", "SP500.parquet")
        if os.path.exists(sp_path):
            df_sp = pd.read_parquet(sp_path).sort_index()
            df_sp.index = pd.to_datetime(df_sp.index).tz_localize(None).normalize()
            sp_ret = df_sp.get("Close_Adj", df_sp.get("Close")).pct_change().shift(1).fillna(0.0)
            sp_map = dict(zip(df_sp.index, sp_ret))
            df["SP500_Ret"] = [sp_map.get(d, 0.0) for d in df.index]
        else:
            df["SP500_Ret"] = 0.0

        # ADR alignment if applicable
        adr_path = os.path.join(self.lake_dir, "Global", f"{clean_ticker}_ADR.parquet")
        if not os.path.exists(adr_path) and clean_ticker in ("INFY", "HDFCBANK", "ICICIBANK"):
            alt_name = "INFY_ADR" if clean_ticker == "INFY" else ("HDFC_ADR" if "HDFC" in clean_ticker else "ICICI_ADR")
            adr_path = os.path.join(self.lake_dir, "Global", f"{alt_name}.parquet")

        if os.path.exists(adr_path):
            df_adr = pd.read_parquet(adr_path).sort_index()
            df_adr.index = pd.to_datetime(df_adr.index).tz_localize(None).normalize()
            adr_ret = df_adr.get("Close_Adj", df_adr.get("Close")).pct_change().shift(1).fillna(0.0)
            adr_map = dict(zip(df_adr.index, adr_ret))
            df["ADR_Ret"] = [adr_map.get(d, 0.0) for d in df.index]
        else:
            df["ADR_Ret"] = df["SP500_Ret"] * 1.05

        df["Global_Confluence"] = np.where(df["SP500_Ret"] > 0, 1.0, -1.0)
        return df.dropna()
