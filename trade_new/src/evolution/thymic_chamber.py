"""Thymic Negative Selection Stress Chamber (Ponytail MVC)."""
import os
from typing import List, Optional
import numpy as np
import pandas as pd
from ..domain.formula_ast import ASTNode, ASTEvaluator


def _build_covid_synthetic_profile(n_bars: int = 70) -> pd.DataFrame:
    """Generates synthetic COVID-19 crash profile with 70+ bars and pre-computed features."""
    dates = pd.date_range("2020-02-15", periods=n_bars, freq="B")
    rng = np.random.RandomState(42)
    # Severe crash sequence followed by volatile recovery
    base_rets = rng.normal(-0.008, 0.025, n_bars)
    # Inject acute panic crash phase in bars 15 to 35
    base_rets[15:25] = np.array([-0.035, -0.060, -0.042, -0.075, -0.025, -0.055, -0.080, -0.030, -0.045, -0.065])
    base_rets[25:35] = np.array([0.045, -0.050, 0.060, -0.035, 0.025, -0.040, 0.055, -0.020, 0.035, -0.015])
    prices = 1000.0 * np.cumprod(1.0 + base_rets)

    c = pd.Series(prices, index=dates)
    o = pd.Series(np.roll(prices, 1), index=dates)
    o.iloc[0] = 1000.0
    h = np.maximum(c, o) * (1.0 + rng.uniform(0.005, 0.02, n_bars))
    l = np.minimum(c, o) * (1.0 - rng.uniform(0.005, 0.02, n_bars))
    v = rng.uniform(2e6, 8e6, n_bars)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema_slope = (ema20 - ema20.shift(1)).fillna(0.0) / (ema20 + 1e-8)
    v_mean = pd.Series(v, index=dates).rolling(20, min_periods=1).mean()
    v_std = pd.Series(v, index=dates).rolling(20, min_periods=1).std().fillna(1.0)
    vol_z = (v - v_mean) / (v_std + 1e-8)

    tr = np.maximum(h - l, np.maximum(np.abs(h - o), np.abs(l - o)))
    dist_vwap = (c - ((h + l + c) / 3.0)) / (c + 1e-8)
    log_hl = np.log(np.maximum(h / np.maximum(l, 1e-8), 1e-8)) ** 2
    parkinson = np.sqrt(pd.Series(log_hl, index=dates).rolling(20, min_periods=1).mean() / (4.0 * np.log(2.0))) * np.sqrt(252)

    return pd.DataFrame({
        "Close_Adj": c, "Close_Raw": c, "Close": c, "Open": o, "High": h, "Low": l, "Volume": v,
        "Spread": 0.05, "Vol_ZScore": vol_z, "Vol_Price_Flow": base_rets * 3.0,
        "True_Range": tr, "TR_Expansion": 2.5, "EMA_Slope": ema_slope,
        "Dist_EMA20": (c - ema20) / (ema20 + 1e-8), "Dist_VWAP": dist_vwap,
        "Downside_Vol_Ratio": 0.85, "RSI_14": 0.25, "Parkinson_Vol": parkinson.fillna(0.35),
        "Garman_Klass_Vol": 0.38, "SP500_Ret": -0.02, "ADR_Ret": -0.03, "Global_Confluence": -1.0,
        "gross_overnight_rets": (o - c.shift(1)).fillna(0.0) / (c.shift(1) + 1e-8),
        "gross_intraday_rets": (c - o) / (o + 1e-8), "gross_close_to_close_rets": base_rets,
        "c2c": base_rets
    }, index=dates)


def _build_election_synthetic_profile(n_bars: int = 65) -> pd.DataFrame:
    """Generates synthetic 2024 Election crash profile with June 4 crash bar and pre-computed features."""
    dates = pd.date_range("2024-05-01", periods=n_bars, freq="B")
    rng = np.random.RandomState(2024)
    rets = rng.normal(0.001, 0.010, n_bars)
    # Inject June 4, 2024 election crash (-6%) and subsequent sharp recovery
    rets[22] = -0.062
    rets[23] = 0.034
    rets[24] = 0.028
    prices = 2200.0 * np.cumprod(1.0 + rets)

    c = pd.Series(prices, index=dates)
    o = pd.Series(np.roll(prices, 1), index=dates)
    o.iloc[0] = 2200.0
    h = np.maximum(c, o) * (1.0 + rng.uniform(0.004, 0.015, n_bars))
    l = np.minimum(c, o) * (1.0 - rng.uniform(0.004, 0.015, n_bars))
    v = rng.uniform(1e6, 5e6, n_bars)
    v[22] = 9e6

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema_slope = (ema20 - ema20.shift(1)).fillna(0.0) / (ema20 + 1e-8)
    v_mean = pd.Series(v, index=dates).rolling(20, min_periods=1).mean()
    v_std = pd.Series(v, index=dates).rolling(20, min_periods=1).std().fillna(1.0)
    vol_z = (v - v_mean) / (v_std + 1e-8)
    dist_vwap = (c - ((h + l + c) / 3.0)) / (c + 1e-8)
    log_hl = np.log(np.maximum(h / np.maximum(l, 1e-8), 1e-8)) ** 2
    parkinson = np.sqrt(pd.Series(log_hl, index=dates).rolling(20, min_periods=1).mean() / (4.0 * np.log(2.0))) * np.sqrt(252)

    return pd.DataFrame({
        "Close_Adj": c, "Close_Raw": c, "Close": c, "Open": o, "High": h, "Low": l, "Volume": v,
        "Spread": 0.04, "Vol_ZScore": vol_z, "Vol_Price_Flow": rets * 2.0,
        "True_Range": (h - l), "TR_Expansion": 2.0, "EMA_Slope": ema_slope,
        "Dist_EMA20": (c - ema20) / (ema20 + 1e-8), "Dist_VWAP": dist_vwap,
        "Downside_Vol_Ratio": 0.65, "RSI_14": 0.45, "Parkinson_Vol": parkinson.fillna(0.25),
        "Garman_Klass_Vol": 0.28, "SP500_Ret": 0.002, "ADR_Ret": 0.001, "Global_Confluence": 0.5,
        "gross_overnight_rets": (o - c.shift(1)).fillna(0.0) / (c.shift(1) + 1e-8),
        "gross_intraday_rets": (c - o) / (o + 1e-8), "gross_close_to_close_rets": rets,
        "c2c": rets
    }, index=dates)


class ThymicStressChamber:
    """Stress tests candidate formulas against acute crash regimes (COVID-19 and Election 2024)."""

    def __init__(self, data_lake_dir: str = "data/lake") -> None:
        self.data_lake_dir = data_lake_dir
        if not os.path.exists(self.data_lake_dir):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.join(base_dir, "data", "lake")
            if os.path.exists(candidate):
                self.data_lake_dir = candidate
        self.stress_profiles: List[pd.DataFrame] = []
        self._build_stress_profiles()

    def _load_empirical_election_profile(self) -> Optional[pd.DataFrame]:
        """Attempts to load empirical 2024 Election crash slice with pre-computed features."""
        try:
            from ..manifold.lake_dataset_loader import LakeDatasetLoader
            loader = LakeDatasetLoader(lake_dir=self.data_lake_dir)
            df = loader.load_dataset(ticker="NIFTYBEES", lookback_years=2.0)
            if df is not None and len(df) >= 60:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                slice_df = df.loc[(df.index >= "2024-04-15") & (df.index <= "2024-07-15")].copy()
                if len(slice_df) >= 40:
                    return slice_df
        except Exception:
            pass
        return None

    def _build_stress_profiles(self) -> None:
        """Constructs synthetic COVID-19 profile (70 bars) and empirical/synthetic 2024 Election crash."""
        self.stress_profiles.append(_build_covid_synthetic_profile(n_bars=70))

        empirical = self._load_empirical_election_profile()
        if empirical is not None:
            self.stress_profiles.append(empirical)
        else:
            self.stress_profiles.append(_build_election_synthetic_profile(n_bars=65))

    def measure_stress_drawdown(self, formula: ASTNode, active_regime: int = 0) -> float:
        """
        Calculates maximum peak-to-trough drawdown across all historical stress profiles.
        Supports continuous and short/hedging signals (e.g. range [-1.0, 1.0]).
        Rejects dead/silenced formulas producing zero active positions across profiles.
        """
        max_dd = 0.0
        total_active_bars = 0
        for profile in self.stress_profiles:
            try:
                sig = ASTEvaluator.eval_node(formula, profile, current_regime=active_regime)
                if not np.all(np.isfinite(sig)):
                    return 1.0
                # Support continuous long/short positions clipped to [-1.0, 1.0]
                pos = np.clip(sig, -1.0, 1.0)
                active_bars = int(np.sum(np.abs(pos) > 1e-4))
                total_active_bars += active_bars
                if active_bars == 0:
                    continue
                c = profile.get("Close_Adj", profile.get("Close", profile.get("Close_Raw")))
                strat_rets = pos[:-1] * c.pct_change().fillna(0.0).to_numpy()[1:]
                if len(strat_rets) > 0:
                    cum_eq = np.cumprod(1.0 + strat_rets)
                    peaks = np.maximum.accumulate(cum_eq)
                    drawdowns = (peaks - cum_eq) / (peaks + 1e-8)
                    dd_val = float(np.max(drawdowns))
                    if dd_val > max_dd:
                        max_dd = dd_val
            except Exception:
                return 1.0

        # Reject zero-signal passes: dead or silenced formulas fail quarantine
        if total_active_bars == 0:
            return 1.0
        return max_dd

    def test_formula_safety(self, formula: ASTNode, active_regime: int = 0, max_dd_tolerance: float = 0.06) -> bool:
        """Quarantines formulas suffering drawdowns exceeding max_dd_tolerance (default 6%)."""
        observed_dd = self.measure_stress_drawdown(formula, active_regime=active_regime)
        return observed_dd <= max_dd_tolerance


ThymicChamber = ThymicStressChamber
