"""Institutional ETF-to-benchmark tracking spread and basis alignment module."""
from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd

# Standard creation-unit benchmark parity ratios
KNOWN_ETF_RATIOS = {
    "BANKBEES.NS": 100.0,
    "NIFTYBEES.NS": 100.0,
}


class ETFBUSpreadAligner:
    """Computes tracking spread, basis (in bps), rolling beta, and z-score between an ETF and its benchmark.
    
    Identifies order-book supply/demand mispricings around indicative NAV (iNAV)
    to power mean-reversion and basis-arbitrage strategy alphas with zero lookahead bias.
    """

    def __init__(
        self,
        rolling_window: int = 60,
        contract_multiplier: Optional[float] = None,
    ) -> None:
        self.rolling_window = max(10, rolling_window)
        self.contract_multiplier = contract_multiplier

    def align(
        self,
        etf_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
        ticker: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Align ETF and benchmark series and calculate tracking analytics with strict causality.
        
        Args:
            etf_df: Cleaned ETF DataFrame with DatetimeIndex and Close column.
            benchmark_df: Cleaned Benchmark DataFrame with DatetimeIndex and Close column.
            ticker: Optional ticker symbol to look up official creation-unit ratio.
            
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

        aligned_idx = benchmark_df["Close"].reindex(etf_df.index).ffill()
        first_valid = aligned_idx.first_valid_index()
        if first_valid is None:
            return None
        if aligned_idx.isna().any():
            merged = merged.loc[first_valid:].copy()
            aligned_idx = aligned_idx.loc[first_valid:]
        merged["Index_Close"] = aligned_idx

        # Compute strictly causal point-in-time multiplier
        mult_series = self._resolve_multiplier(merged["Index_Close"], merged["ETF_Close"], ticker=ticker)
        merged["Multiplier"] = mult_series
        merged["Theoretical_Close"] = merged["Index_Close"] / np.maximum(1e-4, mult_series)
        merged["Spread_INR"] = merged["ETF_Close"] - merged["Theoretical_Close"]
        merged["Basis_Bps"] = (merged["Spread_INR"] / merged["Theoretical_Close"]) * 10000.0

        # Rolling statistical signals with zero lookahead
        merged = self._compute_rolling_signals(merged)
        return merged

    def _resolve_multiplier(
        self, idx_close: pd.Series, etf_close: pd.Series, ticker: Optional[str] = None
    ) -> pd.Series:
        """Resolve multiplier using contract specification or point-in-time expanding median."""
        if self.contract_multiplier is not None and self.contract_multiplier > 0:
            return pd.Series(self.contract_multiplier, index=etf_close.index)

        t_norm = ticker.upper() if ticker else ""
        if t_norm in KNOWN_ETF_RATIOS:
            return pd.Series(KNOWN_ETF_RATIOS[t_norm], index=etf_close.index)

        # Strictly causal expanding median lagged by 1 bar to prevent future information leakage
        raw_ratios = (idx_close / np.maximum(0.01, etf_close)).shift(1)
        expanding_mult = raw_ratios.expanding(min_periods=5).median()
        first_valid = expanding_mult.dropna().iloc[0] if not expanding_mult.dropna().empty else 1.0
        return expanding_mult.fillna(first_valid).round(2)

    def _compute_rolling_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute causal rolling basis z-score, rolling beta, and rolling correlation."""
        w = self.rolling_window
        basis = df["Basis_Bps"]

        # Causal rolling mean and std: strictly shifted by 1 bar
        r_mean = basis.rolling(w, min_periods=5).mean().shift(1).fillna(
            basis.expanding(min_periods=1).mean().shift(1)
        ).fillna(basis)

        r_std = basis.rolling(w, min_periods=5).std().shift(1).fillna(
            basis.expanding(min_periods=1).std().shift(1)
        ).fillna(1.0)
        r_std = np.maximum(r_std, 1.0)  # noise floor 1 bps

        df["Spread_ZScore"] = (basis - r_mean) / r_std

        # Rolling correlation between ETF and Index returns
        etf_ret = df["ETF_Close"].pct_change().fillna(0.0)
        idx_ret = df["Index_Close"].pct_change().fillna(0.0)

        corr = etf_ret.rolling(w, min_periods=5).corr(idx_ret).shift(1).fillna(1.0)
        df["Rolling_Correlation"] = corr.clip(lower=-1.0, upper=1.0)

        # Safe Rolling Beta = Cov(etf, idx) / Var(idx)
        cov = etf_ret.rolling(w, min_periods=5).cov(idx_ret).shift(1)
        var = idx_ret.rolling(w, min_periods=5).var().shift(1)
        var_clean = np.where(np.isnan(var.values) | (var.values <= 1e-12), 1e-8, var.values)
        beta = cov.values / var_clean
        df["Rolling_Beta"] = pd.Series(beta, index=df.index).fillna(1.0).clip(lower=0.0, upper=5.0)

        return df
