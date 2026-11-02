"""Thymic Negative Selection Stress Chamber (Ponytail MVC)."""
import os
import numpy as np
import pandas as pd
from ..domain.formula_ast import ASTNode, ASTEvaluator


class ThymicStressChamber:
    def __init__(self, data_lake_dir: str = "data/lake"):
        self.data_lake_dir = data_lake_dir
        self.stress_profiles: list[pd.DataFrame] = []
        self._build_stress_profiles()

    def _build_stress_profiles(self):
        dates = pd.date_range("2020-03-01", periods=15)
        rets = np.array([-0.02, -0.04, -0.06, -0.03, -0.05, -0.01, -0.07, -0.04, -0.02, 0.03, -0.05, -0.03, -0.02, 0.04, -0.03])
        prices = 1000.0 * np.cumprod(1.0 + rets)
        synth = pd.DataFrame({
            "Close_Adj": prices, "Close_Raw": prices, "Open": np.roll(prices, 1), "High": prices * 1.01, "Low": prices * 0.96,
            "Volume": np.linspace(2e6, 8e6, 15), "Spread": 0.05, "Vol_ZScore": np.linspace(1.5, 4.0, 15),
            "Vol_Price_Flow": rets * 3.0, "True_Range": 0.05, "TR_Expansion": 2.5, "EMA_Slope": -0.04,
            "Dist_EMA20": np.linspace(-0.02, -0.20, 15), "Dist_VWAP": np.linspace(-0.01, -0.15, 15),
            "Downside_Vol_Ratio": 0.85, "RSI_14": np.linspace(0.40, 0.12, 15), "Parkinson_Vol": 0.35,
            "Garman_Klass_Vol": 0.38, "SP500_Ret": -0.02, "ADR_Ret": -0.03, "Global_Confluence": -1.0
        }, index=dates)
        synth.loc[dates[0], "Open"] = 1000.0
        self.stress_profiles.append(synth)

        nifty_p = os.path.join(self.data_lake_dir, "Indices", "NIFTYBEES.parquet")
        if os.path.exists(nifty_p):
            df = pd.read_parquet(nifty_p)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            slice_df = df.loc[(df.index >= "2024-05-28") & (df.index <= "2024-06-15")].copy()
            if len(slice_df) >= 5:
                if "Close_Adj" not in slice_df.columns:
                    slice_df["Close_Adj"] = slice_df.get("Close", slice_df.get("Close_Raw", 0.0))
                self.stress_profiles.append(slice_df)

    def test_formula_safety(self, formula: ASTNode, active_regime: int = 0, max_dd_tolerance: float = 0.06) -> bool:
        for profile in self.stress_profiles:
            try:
                sig = ASTEvaluator.eval_node(formula, profile, current_regime=active_regime)
                if not np.all(np.isfinite(sig)):
                    return False
                pos = np.where(sig > 0, 1.0, 0.0)
                if np.sum(pos) == 0:
                    continue
                c = profile.get("Close_Adj", profile.get("Close", profile.get("Close_Raw")))
                strat_rets = pos[:-1] * c.pct_change().fillna(0.0).to_numpy()[1:]
                if len(strat_rets) > 0 and float(np.min(np.cumsum(strat_rets))) < -max_dd_tolerance:
                    return False
            except Exception:
                return False
        return True
