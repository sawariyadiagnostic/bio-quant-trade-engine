"""
Full Universe Ingestion & Multi-Sector Manifold Analysis Engine (Phase A Completion).
Upgraded to 18-Dimensional Market Tensor + Dissertation UMAP (3D) + HDBSCAN Density Manifold.
Evaluates all 81 companies across all 12 NSE sectors over 5 years.
"""
import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
from tabulate import tabulate
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSEDataLoader, NSE_SECTORS
from src.manifold.feature_pipeline import MarketEmbeddingPipeline
from src.manifold.hdbscan_engine import AnchoredManifoldClassifier

def ingest_and_analyze_universe():
    logger.info("=== Starting Upgraded Phase A Full-Universe Lake Ingestion & Structural Analysis (18D UMAP) ===")
    loader = NSEDataLoader()
    pipeline = MarketEmbeddingPipeline(window_size=20, use_full_18d=True)

    sector_summaries = []
    company_details = []
    full_audit_dict = {}

    total_stocks = sum(len(tickers) for tickers in NSE_SECTORS.values())
    processed_count = 0

    print(f"\nIngesting and Analyzing {total_stocks} companies across {len(NSE_SECTORS)} sectors...")

    for sector_name, tickers in NSE_SECTORS.items():
        logger.info(f"--- Processing Sector: {sector_name} ({len(tickers)} companies) ---")
        sector_returns = []
        sector_vols = []
        sector_drawdowns = []
        sector_noise_ratios = []
        sector_calmars = []
        sector_cluster_counts = []
        sector_active_counts = {"Noise": 0, "Active": 0}

        full_audit_dict[sector_name] = {}

        for ticker in tickers:
            processed_count += 1
            ticker_clean = ticker.replace(".NS", "")
            
            # Fetch 5-year data into local Parquet lake
            df = loader.fetch_ticker(ticker, sector=sector_name, period="5y")
            time.sleep(0.03) # Pacing pause for hardware cooling

            if df.empty or len(df) < 60:
                logger.warning(f"Skipping {ticker}: insufficient data ({len(df)} rows)")
                continue

            # Core Quantitative Metrics
            c_adj = df["Close_Adj"]
            daily_rets = c_adj.pct_change().dropna().to_numpy()
            
            cum_ret = float(np.prod(1.0 + daily_rets) - 1.0)
            ann_vol = float(np.std(daily_rets) * np.sqrt(252))
            
            # Maximum Drawdown
            cum_curve = np.cumprod(1.0 + daily_rets)
            peaks = np.maximum.accumulate(cum_curve)
            drawdowns = (cum_curve - peaks) / peaks
            mdd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

            # Calmar & Sortino Ratios
            calmar = float(cum_ret / abs(mdd)) if abs(mdd) > 0.01 else 0.0
            downside_rets = daily_rets[daily_rets < 0]
            downside_vol = float(np.std(downside_rets) * np.sqrt(252)) if len(downside_rets) > 0 else ann_vol
            sortino = float((np.mean(daily_rets) * 252) / (downside_vol + 1e-6))

            # 18D Market Embedding & UMAP-HDBSCAN Regime Discovery
            classifier = AnchoredManifoldClassifier(
                min_cluster_size=6,
                min_samples=3,
                use_umap=True,
                umap_components=3,
                umap_neighbors=15,
                umap_min_dist=0.1
            )
            try:
                X, idx = pipeline.transform(df)
                labels = classifier.fit(X)
                
                num_clusters = len(classifier.centroids)
                noise_pct = float(np.mean(labels == -1) * 100)
                today_state = classifier.predict_single(X[-1])
                today_regime = int(today_state.cluster_id)
                today_is_noise = bool(today_state.is_noise)
            except Exception as e:
                logger.warning(f"Error classifying {ticker}: {e}")
                num_clusters = 0
                noise_pct = 100.0
                today_regime = -1
                today_is_noise = True

            sector_returns.append(cum_ret)
            sector_vols.append(ann_vol)
            sector_drawdowns.append(mdd)
            sector_noise_ratios.append(noise_pct)
            sector_calmars.append(calmar)
            sector_cluster_counts.append(num_clusters)

            if today_is_noise:
                sector_active_counts["Noise"] += 1
                state_str = "NOISE (-1)"
            else:
                sector_active_counts["Active"] += 1
                state_str = f"REGIME {today_regime}"

            latest_price = float(df["Close_Raw"].iloc[-1])
            company_details.append([
                sector_name,
                ticker_clean,
                f"INR {latest_price:,.2f}",
                f"{cum_ret:+.1%}",
                f"{ann_vol:.1%}",
                f"{mdd:.1%}",
                f"{calmar:.2f}",
                f"{sortino:.2f}",
                f"{num_clusters}",
                f"{noise_pct:.1f}%",
                state_str
            ])

            full_audit_dict[sector_name][ticker_clean] = {
                "latest_price": float(latest_price),
                "5y_cum_return": float(cum_ret),
                "ann_volatility": float(ann_vol),
                "max_drawdown": float(mdd),
                "calmar_ratio": float(calmar),
                "sortino_ratio": float(sortino),
                "clusters_count": int(num_clusters),
                "noise_ratio_pct": float(noise_pct),
                "today_regime": int(today_regime),
                "is_noise": bool(today_is_noise)
            }

        # Sector Level Aggregations
        if sector_returns:
            avg_ret = float(np.mean(sector_returns))
            avg_vol = float(np.mean(sector_vols))
            avg_mdd = float(np.mean(sector_drawdowns))
            avg_calmar = float(np.mean(sector_calmars))
            avg_noise = float(np.mean(sector_noise_ratios))
            avg_clusters = float(np.mean(sector_cluster_counts))
            active_pct = (sector_active_counts["Active"] / len(sector_returns)) * 100

            sector_summaries.append([
                sector_name,
                len(sector_returns),
                f"{avg_ret:+.1%}",
                f"{avg_vol:.1%}",
                f"{avg_mdd:.1%}",
                f"{avg_calmar:.2f}",
                f"{avg_clusters:.1f}",
                f"{avg_noise:.1f}%",
                f"{sector_active_counts['Noise']}",
                f"{sector_active_counts['Active']}",
                f"{active_pct:.0f}%"
            ])

        gc.collect()

    # 1. Print Sector Summaries Table
    print("\n" + "=" * 110)
    print("   12-SECTOR MACRO STRUCTURAL COMPARISON (18D UMAP-HDBSCAN MANIFOLD CALIBRATION)")
    print("=" * 110)
    sec_headers = ["Sector", "Stocks", "Avg 5Y Ret", "Avg Vol", "Avg MDD", "Calmar", "Avg Clusters", "Avg Noise %", "In Cash", "Active", "Active %"]
    print(tabulate(sector_summaries, headers=sec_headers, tablefmt="grid"))

    # 2. Print Top 20 Best Risk-Adjusted Companies across entire Universe
    print("\n" + "=" * 110)
    print("   TOP 20 HIGH-ALPHA STRUCTURAL EQUITIES ACROSS NSE UNIVERSE (BY CALMAR & RETURN)")
    print("=" * 110)
    # Sort companies by 5Y Return descending
    sorted_companies = sorted(company_details, key=lambda x: float(x[3].replace("%", "").replace("+", "")), reverse=True)
    comp_headers = ["Sector", "Ticker", "Price", "5Y Return", "Volatility", "Max DD", "Calmar", "Sortino", "Clusters", "Noise %", "State Today"]
    print(tabulate(sorted_companies[:20], headers=comp_headers, tablefmt="grid"))

    def _np_serial(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    # 3. Save full audit to JSON
    audit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit", "full_universe_analysis.json"))
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(full_audit_dict, f, indent=2, default=_np_serial)
    logger.success(f"Full universe audit persisted to: {audit_path}")

if __name__ == "__main__":
    ingest_and_analyze_universe()
