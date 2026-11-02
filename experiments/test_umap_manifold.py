"""
Empirical Comparison: Raw 7D HDBSCAN vs. Dissertation UMAP+HDBSCAN Architecture.
Applies parameters from Rupesh's dissertation:
  - UMAP: n_neighbors=15, min_dist=0.1, n_components=3
  - HDBSCAN: min_cluster_size=6, min_samples=3, metric='euclidean'
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
import umap
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSEDataLoader

def build_expanded_12d_features(df: pd.DataFrame) -> np.ndarray:
    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close_Raw"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    tensor = pd.DataFrame(index=df.index)

    # Core returns
    tensor["Ret_1d"] = c.pct_change(1)
    tensor["Ret_5d"] = c.pct_change(5)
    tensor["Ret_20d"] = c.pct_change(20)

    # Volatility and Range
    tensor["True_Range"] = (h - l) / c
    tr_rolling_20 = tensor["True_Range"].rolling(20).mean()
    tensor["TR_Expansion"] = tensor["True_Range"] / tr_rolling_20.replace(0, np.nan)

    # Volume dynamics
    vol_mean = v.rolling(20).mean()
    vol_std = v.rolling(20).std().replace(0, np.nan)
    tensor["Vol_ZScore"] = ((v - vol_mean) / vol_std).fillna(0.0)
    tensor["Vol_Price_Flow"] = tensor["Ret_1d"] * (v / vol_mean.replace(0, np.nan)).clip(0, 5)

    # Trend moving averages
    ema_10 = c.ewm(span=10, adjust=False).mean()
    ema_20 = c.ewm(span=20, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()
    tensor["Dist_EMA20"] = (c - ema_20) / ema_20
    tensor["EMA_Convergence"] = (ema_20 - ema_50) / ema_50
    tensor["EMA_Slope"] = (ema_10 - ema_20) / ema_20

    # Downside semi-variance ratio
    neg_rets = tensor["Ret_1d"].clip(upper=0)
    pos_rets = tensor["Ret_1d"].clip(lower=0)
    down_vol = neg_rets.rolling(20).std().replace(0, np.nan)
    up_vol = pos_rets.rolling(20).std().replace(0, np.nan)
    tensor["Downside_Vol_Ratio"] = (down_vol / (down_vol + up_vol)).fillna(0.5)

    # Acceleration / 2nd derivative of momentum
    tensor["Mom_Curvature"] = tensor["Ret_5d"] - tensor["Ret_20d"] * 0.25

    clean = tensor.dropna()
    scaler = StandardScaler()
    return scaler.fit_transform(clean)

def evaluate_stock(ticker: str, sector: str):
    lake_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lake", sector))
    parquet_path = os.path.join(lake_dir, f"{ticker}.parquet")
    if not os.path.exists(parquet_path):
        return None
    df = pd.read_parquet(parquet_path)

    # Method 1: Baseline 7D Direct HDBSCAN (min_cluster_size=15, min_samples=5)
    from src.manifold.feature_pipeline import MarketEmbeddingPipeline
    base_pipeline = MarketEmbeddingPipeline(window_size=20)
    X_base, _ = base_pipeline.transform(df)
    base_hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
    labels_base = base_hdb.fit_predict(X_base)
    n_clust_base = len(set(labels_base)) - (1 if -1 in labels_base else 0)
    noise_pct_base = float(np.mean(labels_base == -1) * 100)

    # Method 2: Dissertation UMAP (3D) + HDBSCAN (min_cluster_size=6, min_samples=3)
    X_exp = build_expanded_12d_features(df)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3,
        metric="euclidean",
        random_state=42
    )
    X_umap = reducer.fit_transform(X_exp)
    umap_hdb = HDBSCAN(min_cluster_size=6, min_samples=3)
    labels_umap = umap_hdb.fit_predict(X_umap)
    n_clust_umap = len(set(labels_umap)) - (1 if -1 in labels_umap else 0)
    noise_pct_umap = float(np.mean(labels_umap == -1) * 100)

    return {
        "ticker": ticker,
        "sector": sector,
        "base_clusters": n_clust_base,
        "base_noise": noise_pct_base,
        "umap_clusters": n_clust_umap,
        "umap_noise": noise_pct_umap
    }

if __name__ == "__main__":
    test_tickers = [
        ("HAL", "Capital_Goods_Defense"),
        ("BEL", "Capital_Goods_Defense"),
        ("RELIANCE", "Energy_Oil_Gas"),
        ("TCS", "Information_Technology"),
        ("HDFCBANK", "Banking_Financials"),
        ("MARUTI", "Automotive"),
        ("SUNPHARMA", "Pharma_Healthcare"),
        ("ITC", "FMCG_Consumption")
    ]

    rows = []
    for ticker, sector in test_tickers:
        res = evaluate_stock(ticker, sector)
        if res:
            rows.append([
                res["ticker"],
                res["sector"],
                f"{res['base_clusters']} clusters",
                f"{res['base_noise']:.1f}%",
                f"{res['umap_clusters']} clusters",
                f"{res['umap_noise']:.1f}%",
                f"{res['base_noise'] - res['umap_noise']:+.1f}% drop"
            ])

    headers = ["Ticker", "Sector", "Baseline (7D)", "Base Noise %", "Dissertation UMAP+HDBSCAN", "UMAP Noise %", "Noise Delta"]
    print("\n" + "=" * 95)
    print("   ARCHITECTURAL COMPARISON: BASELINE 7D vs DISSERTATION UMAP (12D -> 3D)")
    print("=" * 95)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
