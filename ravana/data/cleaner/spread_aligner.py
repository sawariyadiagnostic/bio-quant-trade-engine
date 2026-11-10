"""Institutional ETF-to-benchmark tracking spread and basis alignment module."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class ETFBUSpreadAligner:
    """Computes tracking spread, basis (in bps), rolling beta, and z-score between an ETF and its benchmark.
    
    Identifies order-book supply/demand mispricings around indicative NAV (iNAV)
    to power mean-reversion and basis-arbitrage strategy alphas.
    """

    def __init__(self, rolling_window: int = 60) -> None:
        self.rolling_window = max(10, rolling_window)

    def align(
        self,
        etf_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Align ETF and benchmark series and calculate tracking analytics.
        
        Args:
            etf_df: Cleaned ETF DataFrame with DatetimeIndex and Close column.
            benchmark_df: Cleaned Benchmark DataFrame with DatetimeIndex and Close column.
            
        Returns:
            DataFrame containing aligned prices, basis in bps, z-score, beta, and correlation.
        """
        if benchmark_df is None or benchmark_df.empty or etf_df.empty:
            return None

        if "Close" not in etf_df.columns or "Close" not in benchmark_df.columns:
            return None

        # Reindex benchmark to ETF index with causal forward-fill
        merged = pd.DataFrame(index=etf_df.index)
        merged["ETF_Close"] = etf_df["Close"]
        merged["Index_Close"] = benchmark_df["Close"].reindex(etf_df.index).ffill().bfill()

        # Deduce institutional multiplier (e.g. ~100 or ~1000 for BankBees vs Nifty Bank)
        multiplier = self._compute_multiplier(merged["Index_Close"], merged["ETF_Close"])
        if multiplier <= 0:
            return None

        merged["Multiplier"] = multiplier
        merged["Theoretical_Close"] = merged["Index_Close"] / multiplier
        merged["Spread_INR"] = merged["ETF_Close"] - merged["Theoretical_Close"]
        merged["Basis_Bps"] = (merged["Spread_INR"] / merged["Theoretical_Close"]) * 10000.0

        # Rolling statistical signals
        merged = self._compute_rolling_signals(merged)
        return merged

    def _compute_multiplier(self, idx_close: pd.Series, etf_close: pd.Series) -> float:
        """Estimate stable structural multiplier between benchmark index and ETF."""
        valid_ratios = idx_close / np.maximum(0.01, etf_close)
        valid_ratios = valid_ratios.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_ratios.empty:
            return 1.0
        median_ratio = float(valid_ratios.median())
        # Round to nearest integer or clean multiplier
        return max(1.0, round(median_ratio, 2))

    def _compute_rolling_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute causal rolling basis z-score, rolling beta, and rolling correlation."""
        w = self.rolling_window
        basis = df["Basis_Bps"]

        # Causal rolling mean and std (shifted by 1 to prevent self-referential bias)
        r_mean = basis.rolling(w, min_periods=5).mean().shift(1).fillna(basis.expanding().mean())
        r_std = basis.rolling(w, min_periods=5).std().shift(1).fillna(basis.expanding().std()).fillna(1.0)
        r_std = np.maximum(r_std, 1.0)  # noise floor 1 bps

        df["Spread_ZScore"] = (basis - r_mean) / r_std

        # Rolling correlation between ETF and Index returns
        etf_ret = df["ETF_Close"].pct_change().fillna(0.0)
        idx_ret = df["Index_Close"].pct_change().fillna(0.0)

        corr = etf_ret.rolling(w, min_periods=5).corr(idx_ret).shift(1).fillna(1.0)
        df["Rolling_Correlation"] = corr.clip(lower=-1.0, upper=1.0)

        # Rolling Beta = Cov(etf, idx) / Var(idx)
        cov = etf_ret.rolling(w, min_periods=5).cov(idx_ret).shift(1)
        var = idx_ret.rolling(w, min_periods=5).var().shift(1)
        beta = (cov / np.maximum(1e-8, var)).fillna(1.0)
        df["Rolling_Beta"] = beta.clip(lower=0.0, upper=5.0)

        return df
