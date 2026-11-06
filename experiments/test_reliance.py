"""
BIOINFORMATICS-INSPIRED QUANT REGIME DISCOVERY (BIO-QUANT v2.0)
Translating Conformational Dynamics & Thermodynamic Basins into Financial Regime Detection.
Target Asset: Reliance Industries (RELIANCE.NS)
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN


@dataclass(frozen=True)
class EngineConfig:
    ticker: str = "RELIANCE.NS"
    period: str = "2y"
    interval: str = "1h"
    bars_per_day: int = 6          # ~6.25 trading hours on NSE (9:15 - 15:30)
    fwd_horizon_bars: int = 6      # 1-day forward horizon for evaluation
    min_cluster_size: int = 25     # Minimum samples for a persistent regime
    min_samples: int = 15          # Core neighborhood density
    latent_dims: int = 3           # Low-dimensional manifold components


class BioQuantFeatureEngine:
    """Extracts internal invariant order parameters (analogous to biomolecular reaction coordinates)."""

    @staticmethod
    def compute_features(df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        # 1. Multi-scale Log Momentum (Reaction Kinetics)
        log_close = np.log(df["Close"])
        features["Log_Ret_1d"] = log_close.diff(bars_per_day)
        features["Log_Ret_5d"] = log_close.diff(bars_per_day * 5)
        features["Log_Ret_20d"] = log_close.diff(bars_per_day * 20)

        # 2. Garman-Klass Realized Volatility (Thermodynamic Thermal Fluctuations)
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2
        gk_instant = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        features["GK_Vol_5d"] = np.sqrt(gk_instant.rolling(bars_per_day * 5).mean())

        # 3. Microstructural Pressure & Equilibrium Distance
        features["Spread_Efficiency"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-8)
        vol_ma = df["Volume"].rolling(bars_per_day * 5).mean()
        vol_std = df["Volume"].rolling(bars_per_day * 5).std() + 1e-8
        features["Volume_ZScore"] = (df["Volume"] - vol_ma) / vol_std

        # 4. Thermodynamic Equilibrium Delta (Distance from fast & slow mean-paths)
        ema_fast = df["Close"].ewm(span=bars_per_day * 5, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=bars_per_day * 20, adjust=False).mean()
        features["Dist_Equilibrium"] = (df["Close"] - ema_fast) / ema_fast
        features["Curvature_Delta"] = (ema_fast - ema_slow) / ema_slow

        return features.dropna()


class ThermodynamicRegimeClassifier:
    """Projects high-dimensional coordinates to slow collective variables and discovers dense basins."""

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.scaler = RobustScaler()
        self.manifold = PCA(n_components=config.latent_dims, random_state=42)
        self.clusterer = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            metric="euclidean",
            cluster_selection_method="eom"
        )

    def fit_transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_scaled = self.scaler.fit_transform(X)
        X_latent = self.manifold.fit_transform(X_scaled)
        labels = self.clusterer.fit_predict(X_latent)
        # Probability of point belonging to its assigned cluster (thermodynamic stability)
        probabilities = self.clusterer.probabilities_
        return X_latent, labels, probabilities


class RegimeAnalytics:
    """Computes unbiased forward metrics and state transition dynamics."""

    @staticmethod
    def calculate_forward_metrics(
        df: pd.DataFrame, 
        features: pd.DataFrame, 
        labels: np.ndarray, 
        fwd_bars: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = df.loc[features.index].copy()
        dataset["Regime"] = labels
        
        # True forward continuous returns (NO lookahead bias, continuous time)
        dataset["Fwd_Return"] = dataset["Close"].pct_change(fwd_bars).shift(-fwd_bars)
        
        summary_records = []
        for reg in sorted(dataset["Regime"].unique()):
            subset = dataset[dataset["Regime"] == reg]
            fwd_ret = subset["Fwd_Return"].dropna()
            
            mean_ret = fwd_ret.mean()
            vol = fwd_ret.std()
            sharpe = (mean_ret / vol * np.sqrt(252 * 6 / fwd_bars)) if vol > 0 else 0.0
            win_rate = (fwd_ret > 0).mean() if len(fwd_ret) > 0 else 0.0

            summary_records.append({
                "Regime": reg,
                "Sample_Count": len(subset),
                "Allocation_Pct": len(subset) / len(dataset) * 100,
                "Fwd_Mean_Return": mean_ret,
                "Fwd_Volatility": vol,
                "Annualized_Sharpe": sharpe,
                "Win_Rate": win_rate
            })

        metrics_df = pd.DataFrame(summary_records).set_index("Regime")
        return dataset, metrics_df

    @staticmethod
    def compute_transition_matrix(labels: np.ndarray) -> pd.DataFrame:
        """Computes empirical Markov state transition probabilities P(S_{t+1} | S_t)."""
        series = pd.Series(labels)
        current_state = series
        next_state = series.shift(-1)
        matrix = pd.crosstab(current_state, next_state, normalize="index")
        matrix.index.name = "From_Regime"
        matrix.columns.name = "To_Regime"
        return matrix


def run_pipeline() -> None:
    config = EngineConfig()
    print("=" * 70)
    print(f"  BIO-QUANT REGIME ENGINE v2.0: {config.ticker}")
    print("=" * 70)

    # 1. Fetch Market Data
    raw_df = yf.download(config.ticker, period=config.period, interval=config.interval, progress=False, auto_adjust=False)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = [c[0] for c in raw_df.columns]

    if raw_df.empty:
        raise ValueError("No price data retrieved. Check internet connection or ticker.")

    # 2. Feature Extraction
    feature_df = BioQuantFeatureEngine.compute_features(raw_df, config.bars_per_day)

    # 3. Latent Manifold & Basin Clustering
    classifier = ThermodynamicRegimeClassifier(config)
    _, labels, probabilities = classifier.fit_transform(feature_df)

    # 4. Metrics & Statistical Evaluation
    annotated_df, metrics = RegimeAnalytics.calculate_forward_metrics(
        raw_df, feature_df, labels, config.fwd_horizon_bars
    )
    transitions = RegimeAnalytics.compute_transition_matrix(labels)

    # 5. Output Reporting
    print(f"\nAnalyzed {len(feature_df)} hourly bars (~{len(feature_df)//config.bars_per_day} sessions).")
    print(f"Manifold Variance Explained: {classifier.manifold.explained_variance_ratio_.sum():.2%}\n")
    print("REGIME METRICS TABLE (1-Day Forward Horizon):")
    print("-" * 70)
    for idx, row in metrics.iterrows():
        name = "TRANSITION NOISE (Entropy Valley)" if idx == -1 else f"Dense Energy Basin {idx}"
        print(
            f"Regime {idx:2d} | {name:<32} | Count: {int(row['Sample_Count']):4d} ({row['Allocation_Pct']:4.1f}%) | "
            f"Fwd Ret: {row['Fwd_Mean_Return']:+6.2%} | Sharpe: {row['Annualized_Sharpe']:+5.2f} | WinRate: {row['Win_Rate']:5.1%}"
        )

    # 6. Live Decision Engine for Today
    current_regime = labels[-1]
    current_prob = probabilities[-1]
    last_price = float(raw_df["Close"].iloc[-1])
    
    print("\n" + "=" * 70)
    print(f"LIVE REGIME STATUS: {config.ticker} @ ₹{last_price:.2f}")
    print("=" * 70)
    
    if current_regime == -1:
        print(f"STATE: HIGH THERMODYNAMIC ENTROPY (Cluster -1)")
        print(f"Confidence: {current_prob:.1%}")
        print("ACTION: STAND DOWN (100% Cash). Market is traversing an unstable non-equilibrium zone.")
    else:
        regime_row = metrics.loc[current_regime]
        persistence = transitions.loc[current_regime, current_regime] if current_regime in transitions.columns else 0.0
        bias = "BULLISH BIAS" if regime_row["Fwd_Mean_Return"] > 0 else "BEARISH / HEDGE BIAS"
        
        print(f"STATE: METASTABLE BASIN {current_regime} ({bias})")
        print(f"Confidence (Density Membership): {current_prob:.1%}")
        print(f"Regime Persistence P(S_t+1 = S_t): {persistence:.1%}")
        print(f"Expected 1-Day Return: {regime_row['Fwd_Mean_Return']:+.2%} | Sharpe: {regime_row['Annualized_Sharpe']:.2f}")
        
        if regime_row["Annualized_Sharpe"] > 0.5 and persistence > 0.60:
            print("ACTION: ACTIVE LONG. Favorable drift, high persistence, stable free energy basin.")
        elif regime_row["Annualized_Sharpe"] < -0.5:
            print("ACTION: ACTIVE DEFENSE / SHORT. High downward drift.")
        else:
            print("ACTION: NEUTRAL / RANGE-BOUND. Low directional edge; trade mean-reversion.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()