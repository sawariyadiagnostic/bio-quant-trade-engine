"""
18-Dimensional Normalized Market Manifold Feature Pipeline.
Extracts continuous structural tensors from dual-stream market series:
Returns, Volatility (Parkinson, Garman-Klass, True Range), Institutional Flow,
Moving Average Slopes, Downside Semi-Variance, Rejection Shadows, and Oscillators.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class MarketEmbeddingPipeline:
    def __init__(self, window_size: int = 20, use_full_18d: bool = True):
        self.window_size = window_size
        self.use_full_18d = use_full_18d
        self.scaler = StandardScaler()

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        """
        Transforms raw dual-stream OHLCV DataFrame into a normalized (N x 18) tensor.
        Uses Close_Adj for returns and momentum calculations, Close_Raw/Open/High/Low for ranges.
        """
        if len(df) < 55:
            raise ValueError(f"Insufficient historical data: {len(df)} rows. Need at least 55.")

        c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close_Raw"]
        h = df["High"]
        l = df["Low"]
        o = df["Open"] if "Open" in df.columns else c.shift(1).fillna(c)
        v = df["Volume"]

        tensor = pd.DataFrame(index=df.index)

        # 1. Multi-scale momentum (1d, 5d, 20d)
        tensor["Ret_1d"] = c.pct_change(1)
        tensor["Ret_5d"] = c.pct_change(5)
        tensor["Ret_20d"] = c.pct_change(20)

        # 2. Structural volatility & range expansion
        tensor["True_Range"] = (h - l) / c
        tr_rolling = tensor["True_Range"].rolling(self.window_size).mean()
        tensor["TR_Expansion"] = tensor["True_Range"] / tr_rolling.replace(0, np.nan)

        # 3. Volume dynamics & price-flow intensity
        vol_mean = v.rolling(self.window_size).mean()
        vol_std = v.rolling(self.window_size).std().replace(0, np.nan)
        tensor["Vol_ZScore"] = ((v - vol_mean) / vol_std).fillna(0.0)
        tensor["Vol_Price_Flow"] = tensor["Ret_1d"] * (v / vol_mean.replace(0, np.nan)).clip(0, 5)

        # 4. Equilibrium distances & trend curvature
        ema_10 = c.ewm(span=10, adjust=False).mean()
        ema_20 = c.ewm(span=20, adjust=False).mean()
        ema_50 = c.ewm(span=50, adjust=False).mean()
        tensor["Dist_EMA20"] = (c - ema_20) / ema_20
        tensor["EMA_Convergence"] = (ema_20 - ema_50) / ema_50
        tensor["EMA_Slope"] = (ema_10 - ema_20) / ema_20

        # 5. Downside semi-variance ratio (asymmetric risk)
        neg_rets = tensor["Ret_1d"].clip(upper=0)
        pos_rets = tensor["Ret_1d"].clip(lower=0)
        down_vol = neg_rets.rolling(self.window_size).std().replace(0, np.nan)
        up_vol = pos_rets.rolling(self.window_size).std().replace(0, np.nan)
        tensor["Downside_Vol_Ratio"] = (down_vol / (down_vol + up_vol)).fillna(0.5)

        # 6. Momentum acceleration / 2nd derivative
        tensor["Mom_Curvature"] = tensor["Ret_5d"] - tensor["Ret_20d"] * 0.25

        if self.use_full_18d:
            # 7. Advanced Volatility Estimators (Parkinson & Garman-Klass)
            hl_ratio = np.log(h / l.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            tensor["Parkinson_Vol"] = (hl_ratio**2 / (4 * np.log(2))).rolling(self.window_size).mean().apply(np.sqrt)
            
            co_ratio = np.log(c / o.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            tensor["Garman_Klass_Vol"] = (0.5 * hl_ratio**2 - (2 * np.log(2) - 1) * co_ratio**2).rolling(self.window_size).mean()

            # 8. Rejection shadows (Supply wick vs Demand wick)
            bar_range = (h - l).replace(0, 1e-6)
            tensor["Upper_Shadow"] = (h - np.maximum(o, c)) / bar_range
            tensor["Lower_Shadow"] = (np.minimum(o, c) - l) / bar_range

            # 9. Volume-Weighted Average Price (VWAP) Distance
            pv = (c * v).rolling(self.window_size).sum()
            cum_v = v.rolling(self.window_size).sum().replace(0, np.nan)
            vwap = pv / cum_v
            tensor["Dist_VWAP"] = (c - vwap) / vwap

            # 10. Normalized RSI Momentum Oscillator [0, 1]
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
            rs = gain / loss
            tensor["RSI_14"] = (100 - (100 / (1 + rs))) / 100.0

        # Drop warm-up rows
        clean_tensor = tensor.dropna()
        if clean_tensor.empty:
            raise ValueError("All rows dropped during feature calculation warm-up.")

        # Normalize features
        scaled_matrix = self.scaler.fit_transform(clean_tensor)
        return scaled_matrix, clean_tensor.index
