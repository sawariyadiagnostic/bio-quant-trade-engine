"""
EXPERIMENT 1: 12-Sector Multi-Cycle Manifold Regime Discovery Audit.
Evaluates structural noise ratio (-1) across all key sectors of the Indian economy.
"""
import os
import sys
import json
from tabulate import tabulate
from loguru import logger

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSEDataLoader, NSE_SECTORS
from src.manifold.feature_pipeline import MarketEmbeddingPipeline
from src.manifold.hdbscan_engine import AnchoredManifoldClassifier

def run_regime_audit():
    logger.info("=== Starting Experiment 1: 12-Sector Manifold Regime Audit ===")
    loader = NSEDataLoader()
    pipeline = MarketEmbeddingPipeline()
    classifier = AnchoredManifoldClassifier(min_cluster_size=15, min_samples=5)

    results = []
    audit_data = {}

    # Sample top 1-2 leaders per sector to keep CPU execution light and fast
    for sector_name, tickers in NSE_SECTORS.items():
        leader = tickers[0]
        df = loader.fetch_ticker(leader, sector=sector_name, period="2y")
        if df.empty or len(df) < 60:
            continue

        try:
            X, idx = pipeline.transform(df)
            fwd_5d = df["Close_Adj"].pct_change(5).shift(-5).loc[idx].to_numpy()
            labels = classifier.fit(X, forward_returns=fwd_5d)

            total_days = len(labels)
            noise_days = int(np.sum(labels == -1)) if "np" in globals() else int((labels == -1).sum())
            noise_pct = (noise_days / total_days) * 100
            active_clusters = len(set(labels) - {-1})

            today_state = classifier.predict_single(X[-1])
            today_str = "NOISE (-1)" if today_state.is_noise else f"REGIME {today_state.cluster_id}"

            results.append([
                sector_name,
                leader.replace(".NS", ""),
                total_days,
                f"{noise_pct:.1f}%",
                active_clusters,
                today_str
            ])

            audit_data[sector_name] = {
                "leader": leader,
                "noise_pct": float(noise_pct),
                "active_clusters": int(active_clusters),
                "today_regime": int(today_state.cluster_id)
            }

        except Exception as e:
            logger.warning(f"Audit failed for {leader}: {e}")

    # Output formatted table
    headers = ["Sector", "Titan", "Days", "Noise Filtered %", "Active Regimes", "Today's State"]
    table_str = tabulate(results, headers=headers, tablefmt="grid")
    print("\n" + table_str)

    def _np_serial(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    # Persist JSON audit
    audit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit", "regime_audits.json"))
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=_np_serial)
    logger.success(f"Audit report persisted to: {audit_path}")

if __name__ == "__main__":
    import numpy as np
    run_regime_audit()
