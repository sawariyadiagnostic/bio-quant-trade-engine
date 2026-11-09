"""Empirical Real-Work Benchmark Test for RAVANA Muscle 1 on 5-Minute Intraday Data.

Tests Muscle 1's AST grammar, bounded math, and genetic engine on 4,497 real 5-minute
candlestick bars of BANKBEES (~60 trading days up to the latest market session).

Measures:
1. Performance on high-frequency intraday bar sequences (4,497 bars).
2. 1,500-Formula Evolutionary Stress Test across 6,745,500 intraday data points.
3. Intraday execution throughput (Formulas/Sec and Points/Sec).
4. Numerical stability under rapid 5-minute price tick fluctuations.
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf

from ravana.domain.formula_ast import (
    ASTNode,
    ASTEvaluator,
    TreeGenerator,
    GeneticMutator,
    GeneticCrossover,
    FeatureDimension,
)
from ravana.domain.formula_parser import FormulaParser


def fetch_and_prepare_5m_data() -> pd.DataFrame:
    """Downloads real 5-minute BANKBEES bars and extracts technical features."""
    print("Downloading real 5-minute historical data for BANKBEES.NS via yfinance...")
    raw = yf.download("BANKBEES.NS", interval="5m", period="60d", progress=False)
    
    # Handle multi-index columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    c = raw["Close"].to_numpy(dtype=np.float64)
    h = raw["High"].to_numpy(dtype=np.float64)
    l = raw["Low"].to_numpy(dtype=np.float64)
    o = raw["Open"].to_numpy(dtype=np.float64)
    v = raw["Volume"].to_numpy(dtype=np.float64)

    df["Close"] = c
    df["Close_Adj"] = c
    df["Open"] = o
    df["High"] = h
    df["Low"] = l
    df["Volume"] = v
    df["Spread"] = np.nan_to_num((h - l) / (c + 1e-8), nan=0.0)

    v_series = pd.Series(v)
    v_mean = v_series.rolling(20, min_periods=1).mean()
    v_std = v_series.rolling(20, min_periods=1).std(ddof=1).replace(0.0, 1.0)
    df["Vol_ZScore"] = np.nan_to_num(((v_series - v_mean) / v_std).fillna(0.0).to_numpy(), nan=0.0)
    df["Vol_Price_Flow"] = np.nan_to_num((c - o) * v / (c + 1e-8), nan=0.0)

    c_prev = np.roll(c, 1)
    c_prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    df["True_Range"] = tr
    tr_ma = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    df["TR_Expansion"] = np.nan_to_num(tr / (tr_ma + 1e-8), nan=1.0)

    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    df["Dist_EMA20"] = np.nan_to_num((c - ema20) / (c + 1e-8), nan=0.0)
    df["EMA_Slope"] = np.nan_to_num(pd.Series(ema20).diff(5).fillna(0.0).to_numpy() / (c + 1e-8), nan=0.0)
    df["Dist_VWAP"] = df["Dist_EMA20"]

    log_hl = np.log(np.maximum(h / np.maximum(l, 1e-6), 1.0))
    df["Parkinson_Vol"] = np.sqrt((log_hl ** 2) / (4.0 * np.log(2.0)))
    df["Garman_Klass_Vol"] = df["Parkinson_Vol"]

    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df["RSI_14"] = np.nan_to_num(100.0 - (100.0 / (1.0 + rs)), nan=50.0)

    df["SP500_Ret"] = pd.Series(c).pct_change(12).fillna(0.0).to_numpy()
    df["ADR_Ret"] = df["SP500_Ret"]
    df["Global_Confluence"] = np.sign(df["SP500_Ret"])

    return df


def run_intraday_benchmark():
    print("=" * 70)
    print("RAVANA MUSCLE 1: 5-MINUTE INTRADAY EMPIRICAL BENCHMARK")
    print("=" * 70)

    df = fetch_and_prepare_5m_data()
    n_bars = len(df)
    print(f"[OK] Prepared {n_bars} real 5-minute bars of BANKBEES.")
    print(f"     Date span: {df.index[0]} to {df.index[-1]}")
    print(f"     Features ready: {len(df.columns)} columns")
    print("-" * 70)

    # 1. Test Institutional 5-Minute Alphas
    print("1. Evaluating 4 Intraday Institutional Quantitative Alphas on 5-Minute Bars:")
    alpha_formulas = [
        ("5m Price-Volume Accumulation", "TS_CORR_20(Close, Volume)"),
        ("5m Volatility Compression", "TS_ZSCORE_20((High - Low) / Close) * SQRT_ABS(Volume)"),
        ("5m Smooth Regime Blend", "BLEND(Vol_ZScore, TS_RANK_20(Close), (Close - TS_DELAY_1(Close)) / Close)"),
        ("5m Lagged Momentum Decay", "TS_DECAY_LINEAR_10(Close - TS_DELAY_5(Close))")
    ]

    for name, f_str in alpha_formulas:
        ast = FormulaParser.parse(f_str)
        t0 = time.perf_counter()
        res = ASTEvaluator.eval_node(ast, df)
        dt = (time.perf_counter() - t0) * 1000
        is_finite = np.all(np.isfinite(res))
        assert is_finite, f"Formula {name} produced non-finite values!"
        print(f"   [PASS] {name:30s} | Time: {dt:6.2f} ms | Range: [{res.min():8.2f}, {res.max():8.2f}] | Mean: {res.mean():8.2f}")

    print("-" * 70)

    # 2. 1,500-Formula High-Volume Genetic Stress Test
    print("2. Running 1,500-Formula Evolutionary Stress Test on 4,497 5-Minute Bars:")
    n_formulas = 1000
    t_start = time.perf_counter()

    population = []
    for _ in range(n_formulas):
        tree = TreeGenerator.generate(max_depth=4, target_dimension=FeatureDimension.DIMENSIONLESS)
        population.append(tree)

    for i in range(0, n_formulas // 2, 2):
        population[i] = GeneticMutator.mutate_subtree(population[i], max_depth=3)
        population[i + 1] = GeneticMutator.mutate_point(population[i + 1])
        c1, c2 = GeneticCrossover.crossover(population[i], population[i + 1])
        population.append(c1)
        population.append(c2)

    total_trees = len(population)
    total_data_points = total_trees * n_bars
    print(f"   [OK] Generated {total_trees} diverse candidate formulas.")
    print(f"   Evaluating {total_trees} formulas across {n_bars} 5-min bars ({total_data_points:,} data points)...")

    t_eval_start = time.perf_counter()
    nan_count = 0
    inf_count = 0
    hashes = set()

    for tree in population:
        res = ASTEvaluator.eval_node(tree, df)
        if np.any(np.isnan(res)):
            nan_count += 1
        if np.any(np.isinf(res)):
            inf_count += 1
        hashes.add(tree.canonical_hash())

    t_eval = time.perf_counter() - t_eval_start
    throughput = total_trees / t_eval
    point_throughput = total_data_points / t_eval

    print("-" * 70)
    print("3. 5-MINUTE INTRADAY EMPIRICAL RESULTS:")
    print(f"   * Total 5-Minute Bars Scanned : {n_bars:,}")
    print(f"   * Total Formulas Evaluated    : {total_trees:,}")
    print(f"   * Total Data Points Processed : {total_data_points:,}")
    print(f"   * Total Evaluation Time       : {t_eval:.3f} seconds")
    print(f"   * Formula Throughput          : {throughput:,.1f} formulas / second")
    print(f"   * Point Throughput            : {point_throughput:,.0f} points / second")
    print(f"   * NaN Crashes Detected        : {nan_count} (Must be 0)")
    print(f"   * Inf Overflows Detected      : {inf_count} (Must be 0)")
    print(f"   * Unique Formula Diversity    : {len(hashes)} / {total_trees} ({len(hashes)/total_trees*100:.1f}%)")
    print("=" * 70)

    assert nan_count == 0, f"Found {nan_count} NaNs on 5m data!"
    assert inf_count == 0, f"Found {inf_count} Infs on 5m data!"
    print("BENCHMARK VERDICT: 100% PASS - MUSCLE 1 HANDLES 5-MINUTE INTRADAY DATA WITH ZERO ERRORS!")


if __name__ == "__main__":
    run_intraday_benchmark()
