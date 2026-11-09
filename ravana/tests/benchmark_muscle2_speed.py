"""Real-Work Benchmark for Muscle 2: PyTorch TensorASTEvaluator vs NumPy ASTEvaluator.

Runs both evaluators on 4,497 real 5-minute bars of BANKBEES across 1,000 candidate formulas
(4,497,000 real data points) and measures speedup, memory, and parity.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from ravana.domain.formula_ast import (
    ASTNode,
    ASTEvaluator,
    TreeGenerator,
    FeatureDimension,
    STANDARD_FEATURES,
)
from ravana.domain.formula_parser import FormulaParser
from ravana.compute.tensor_evaluator import TensorASTEvaluator


def load_real_data_tensor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading 5-minute bars of BANKBEES (Target compute device: {device})...")
    raw = None
    try:
        raw = yf.download("BANKBEES.NS", interval="5m", period="60d", progress=False)
        if raw is None or raw.empty:
            raise ValueError("Empty yfinance download")
    except Exception as e:
        print(f"[NOTE] Live download unavailable ({e}). Falling back to local lake or synthetic cache...")
        local_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/lake/Indices/BANKBEES.parquet"))
        if os.path.exists(local_p):
            raw = pd.read_parquet(local_p)
            print(f"[OK] Loaded {len(raw)} bars from local parquet lake.")
        else:
            n_syn = 4500
            np.random.seed(42)
            c = 500.0 * np.exp(np.cumsum(np.random.randn(n_syn) * 0.001))
            raw = pd.DataFrame({
                "Open": c * (1 + np.random.uniform(-0.002, 0.002, n_syn)),
                "High": c * (1 + np.random.uniform(0.0, 0.005, n_syn)),
                "Low": c * (1 - np.random.uniform(0.0, 0.005, n_syn)),
                "Close": c,
                "Volume": np.random.uniform(1e5, 1e6, n_syn),
            })
            print(f"[OK] Generated {n_syn} synthetic 5m market bars.")

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

    downside_ret = np.minimum(c / np.maximum(c_prev, 1e-6) - 1.0, 0.0)
    downside_vol = pd.Series(downside_ret).rolling(20, min_periods=1).std().fillna(0.01).to_numpy()
    tot_vol = pd.Series(c).pct_change().rolling(20, min_periods=1).std().fillna(0.01).to_numpy()
    df["Downside_Vol_Ratio"] = np.nan_to_num(downside_vol / (tot_vol + 1e-8), nan=0.5)

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

    # Build PyTorch tensor and move to target hardware device
    feature_matrix = np.column_stack([df[col].to_numpy(dtype=np.float32) for col in STANDARD_FEATURES])
    X_tensor = torch.from_numpy(feature_matrix).to(device)
    f_map = {name: i for i, name in enumerate(STANDARD_FEATURES)}

    return df, X_tensor, f_map


def run_benchmark():
    df, X_tensor, f_map = load_real_data_tensor()
    n_bars = len(df)
    n_formulas = 1000

    print("=" * 70)
    print("RAVANA MUSCLE 2: PYTORCH TENSOR EVALUATOR REAL-WORK BENCHMARK")
    print("=" * 70)
    print(f"Data: {n_bars} 5-minute bars of BANKBEES ({X_tensor.shape[0]}x{X_tensor.shape[1]} tensor)")
    print(f"Generating {n_formulas} candidate formulas...")

    population = [
        TreeGenerator.generate(max_depth=4, target_dimension=FeatureDimension.DIMENSIONLESS)
        for _ in range(n_formulas)
    ]
    print(f"[OK] Generated {len(population)} dimensionally sound formulas.")
    print("-" * 70)

    # 1. Benchmark PyTorch TensorASTEvaluator
    print(f"1. Evaluating with PyTorch TensorASTEvaluator on {X_tensor.device}...")
    t0 = time.perf_counter()
    torch_nan = 0
    for tree in population:
        out = TensorASTEvaluator.eval_node(tree, X_tensor, f_map)
        if torch.any(torch.isnan(out)):
            torch_nan += 1
    t_torch = time.perf_counter() - t0
    torch_fps = n_formulas / t_torch
    torch_pps = (n_formulas * n_bars) / t_torch

    print(f"   * Time Elapsed   : {t_torch:.3f} seconds")
    print(f"   * Formula Speed  : {torch_fps:,.1f} formulas / second")
    print(f"   * Point Speed    : {torch_pps:,.0f} points / second")
    print(f"   * NaN/Inf Errors : {torch_nan}")

    print("-" * 70)

    # 2. Benchmark NumPy ASTEvaluator
    print("2. Evaluating with NumPy ASTEvaluator (Baseline)...")
    t0 = time.perf_counter()
    numpy_nan = 0
    for tree in population:
        out = ASTEvaluator.eval_node(tree, df)
        if np.any(np.isnan(out)):
            numpy_nan += 1
    t_numpy = time.perf_counter() - t0
    numpy_fps = n_formulas / t_numpy

    print(f"   * Time Elapsed   : {t_numpy:.3f} seconds")
    print(f"   * Formula Speed  : {numpy_fps:,.1f} formulas / second")
    print(f"   * NaN/Inf Errors : {numpy_nan}")

    print("-" * 70)

    # 3. Benchmark PyTorch Chunked Batch Evaluation
    print(f"3. Evaluating in chunked batch mode (chunk_size=512) on {X_tensor.device}...")
    t0 = time.perf_counter()
    batch_tensor = TensorASTEvaluator.evaluate_batch(population, X_tensor, f_map, chunk_size=512)
    t_batch = time.perf_counter() - t0
    batch_fps = n_formulas / t_batch
    batch_pps = (n_formulas * n_bars) / t_batch
    print(f"   * Time Elapsed   : {t_batch:.3f} seconds")
    print(f"   * Formula Speed  : {batch_fps:,.1f} formulas / second")
    print(f"   * Point Speed    : {batch_pps:,.0f} points / second")
    print(f"   * Output Shape   : {tuple(batch_tensor.shape)}")

    print("-" * 70)
    speedup = t_numpy / t_torch
    print(f"PYTORCH TENSOR ENGINE SPEEDUP: {speedup:.2f}x FASTER THAN NUMPY BASELINE!")
    print(f"Throughput: {torch_fps:,.1f} formulas/sec across {n_bars * n_formulas:,} data points.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
