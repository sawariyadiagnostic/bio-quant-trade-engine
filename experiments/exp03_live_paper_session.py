"""
EXPERIMENT 3: Complete Pre-Close 3:10 PM IST Autonomous Trading Session.
Orchestrates: Ingestion -> Anchored Manifold Gating -> Epigenetic Execution -> Circuit Damper -> Paper Gateway.
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from loguru import logger
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifold.data_loader import NSEDataLoader, NSE_SECTORS
from src.manifold.feature_pipeline import MarketEmbeddingPipeline
from src.manifold.hdbscan_engine import AnchoredManifoldClassifier
from src.evolution.map_elites import MapElitesArchive
from src.execution.risk_controller import CircuitDamper, PositionSizer
from src.execution.paper_gateway import PaperTradingGateway
from src.domain.models import OrderSignal
from experiments.exp02_map_elites_archive import build_candidate_library

def run_live_paper_session():
    logger.info("=== Starting Pre-Close 3:10 PM IST Bio-Quant Autonomous Session ===")
    
    loader = NSEDataLoader()
    pipeline = MarketEmbeddingPipeline()
    classifier = AnchoredManifoldClassifier(min_cluster_size=15, min_samples=5)
    sizer = PositionSizer(total_capital=50000.0, max_risk_pct=1.0)
    gateway = PaperTradingGateway(initial_capital=50000.0)

    # Pre-train manifold & MAP-Elites archive on calibration anchor
    ref_df = loader.fetch_ticker("RELIANCE.NS", sector="Energy_Oil_Gas", period="2y")
    X_ref, idx_ref = pipeline.transform(ref_df)
    classifier.fit(X_ref)

    archive = MapElitesArchive()
    for cand in build_candidate_library():
        archive.evaluate_and_insert(cand, ref_df)

    elites = archive.get_resident_elites()
    active_formula = elites[0].formula if elites else None

    # Scan universe sample (1 leader per sector)
    print("\n" + "=" * 70)
    print("   PRE-CLOSE 3:10 PM BIO-QUANT AUTONOMOUS SCANNER")
    print("=" * 70)

    executed_count = 0

    for sector_name, tickers in list(NSE_SECTORS.items())[:6]:
        ticker = tickers[0]
        df = loader.fetch_ticker(ticker, sector=sector_name, period="1y")
        if df.empty or len(df) < 60:
            continue

        try:
            X, idx = pipeline.transform(df)
            today_vector = X[-1]
            regime = classifier.predict_single(today_vector)
            
            latest_raw = float(df["Close_Raw"].iloc[-1])
            prev_raw = float(df["Close_Raw"].iloc[-2])

            print(f"\n[{sector_name}] Evaluating {ticker} @ INR {latest_raw:.2f}")

            # 1. Manifold Noise Gating (Upgrade 1)
            if regime.is_noise or not regime.allows_trading:
                print(f"  >>> STATE: NOISE (-1) -> Standing down in 100% Cash / Liquid BeES.")
                continue

            print(f"  >>> STATE: ACTIVE REGIME {regime.cluster_id} (Density: {regime.density_score:.2f})")

            # 2. Statutory Circuit Damper Check (Upgrade 9)
            bands = CircuitDamper.calculate_bands(latest_raw, prev_raw, band_pct=0.10)
            if bands.is_locked:
                print("  >>> CIRCUIT WARNING: Price is within 1% of daily circuit band. Order locked out.")
                continue

            damper_factor = bands.allocation_damper_factor()

            # 3. Position Sizing with ATR
            atr = sizer.calculate_atr(df)
            qty, sl, tp = sizer.size_order(latest_raw, atr, circuit_factor=damper_factor)

            if qty <= 0:
                print("  >>> Risk check: Sized quantity is 0. No position taken.")
                continue

            # 4. Execute Simulated Paper Order (Upgrade 10)
            signal = OrderSignal(
                ticker=ticker,
                action="BUY",
                raw_price=latest_raw,
                stop_loss=sl,
                target=tp,
                quantity=qty,
                regime_id=regime.cluster_id
            )

            success = gateway.execute_order(signal, is_delivery=True)
            if success:
                executed_count += 1
                print(f"  *** [ORDER FILLED] BUY {qty} {ticker} @ INR {latest_raw:.2f} | SL: INR {sl} | TP: INR {tp} ***")

        except Exception as e:
            logger.warning(f"Session error on {ticker}: {e}")

    print("\n" + "=" * 70)
    print(f"Session Complete. Executed Orders: {executed_count} | Audit Log: {gateway.csv_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_live_paper_session()
