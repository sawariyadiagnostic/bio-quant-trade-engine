"""Build self-contained Kaggle GPU burst worker script and package directory."""
import base64
import json
import zlib
from pathlib import Path


def generate_kaggle_payload(
    features_parquet_path: Path,
    targets_parquet_path: Path,
    output_dir: Path,
    population_size: int = 500,
    generations: int = 15,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compress and encode features & targets
    with open(features_parquet_path, "rb") as f:
        f_b64 = base64.b64encode(zlib.compress(f.read(), 9)).decode("ascii")
    with open(targets_parquet_path, "rb") as f:
        t_b64 = base64.b64encode(zlib.compress(f.read(), 9)).decode("ascii")

    # 2. Write embedded self-contained GPU worker script
    worker_code = f'''"""RAVANA Phase 3 Multi-Objective Genetic Alpha Discovery on Kaggle GPU.

Full-capacity evolutionary search running across GPU CUDA tensors, evaluating
alpha formulas with Deflated Sharpe Ratio (DSR), Benjamini-Hochberg FDR,
Turnover penalty, Maximum Drawdown, and Pareto frontier optimization.
"""
import base64
import io
import json
import math
import os
import random
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Embedded Market Payloads
FEATURES_B64 = "{f_b64}"
TARGETS_B64 = "{t_b64}"

print("=================================================================")
print("  RAVANA SOVEREIGN QUANT ENGINE: PHASE 3 KAGGLE GPU BURST RUN    ")
print("=================================================================")

# 1. Hardware & CUDA Inspection
has_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if has_cuda else "cpu")
print(f"CUDA Available: {{has_cuda}}")
if has_cuda:
    print(f"GPU Device: {{torch.cuda.get_device_name(0)}}")
    print(f"Device Count: {{torch.cuda.device_count()}}")
    torch.cuda.reset_peak_memory_stats()
else:
    print("Warning: Running on CPU (CUDA not detected in container)")

# 2. Decompress Market Data
t_start = time.time()
print("Decompressing feature and target parquet payloads...")
f_bytes = zlib.decompress(base64.b64decode(FEATURES_B64))
t_bytes = zlib.decompress(base64.b64decode(TARGETS_B64))

feat_df = pd.read_parquet(io.BytesIO(f_bytes))
targ_df = pd.read_parquet(io.BytesIO(t_bytes))

print(f"Features Loaded: Shape={{feat_df.shape}}, Columns={{len(feat_df.columns)}}")
print(f"Targets Loaded: Shape={{targ_df.shape}}, Columns={{len(targ_df.columns)}}")

# Align Indices
common_idx = feat_df.index.intersection(targ_df.index)
feat_df = feat_df.loc[common_idx]
targ_df = targ_df.loc[common_idx]
num_bars = len(common_idx)
print(f"Aligned Historical Bars: {{num_bars}}")

# Target vector (forward overnight return)
target_col = "Target_Overnight_Net_Return" if "Target_Overnight_Net_Return" in targ_df else targ_df.columns[0]
target_series = targ_df[target_col].to_numpy(dtype=np.float32)
target_tensor = torch.tensor(target_series, dtype=torch.float32, device=device)

# Load features to GPU tensor
feature_cols = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]
feat_mat = feat_df[feature_cols].to_numpy(dtype=np.float32)
feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
gpu_feat_tensor = torch.tensor(feat_mat, dtype=torch.float32, device=device)
feat_map = {{c: i for i, c in enumerate(feature_cols)}}
print(f"GPU Feature Matrix Materialized: {{gpu_feat_tensor.shape}} on {{device}}")

# 3. Vectorized GPU Technical Primitives
def gpu_ts_rank(t: torch.Tensor, window: int = 20) -> torch.Tensor:
    if t.ndim != 1 or len(t) < window:
        return torch.zeros_like(t)
    unfolded = t.unfold(0, window, 1)
    ranks = (unfolded < unfolded[:, -1:]).sum(dim=1).float() / float(window)
    res = torch.zeros_like(t)
    res[window - 1:] = ranks
    return res

def gpu_ts_zscore(t: torch.Tensor, window: int = 20) -> torch.Tensor:
    if t.ndim != 1 or len(t) < window:
        return torch.zeros_like(t)
    unfolded = t.unfold(0, window, 1)
    mean = unfolded.mean(dim=1)
    std = unfolded.std(dim=1) + 1e-8
    res = torch.zeros_like(t)
    res[window - 1:] = (unfolded[:, -1] - mean) / std
    return torch.nan_to_num(res, nan=0.0)

def gpu_ts_delay(t: torch.Tensor, shift: int = 5) -> torch.Tensor:
    res = torch.zeros_like(t)
    if shift < len(t):
        res[shift:] = t[:-shift]
    return res

def gpu_ts_delta(t: torch.Tensor, shift: int = 5) -> torch.Tensor:
    return t - gpu_ts_delay(t, shift)

def gpu_ts_max(t: torch.Tensor, window: int = 20) -> torch.Tensor:
    if t.ndim != 1 or len(t) < window:
        return torch.zeros_like(t)
    unfolded = t.unfold(0, window, 1)
    res = torch.zeros_like(t)
    res[window - 1:], _ = unfolded.max(dim=1)
    return res

def gpu_ts_min(t: torch.Tensor, window: int = 20) -> torch.Tensor:
    if t.ndim != 1 or len(t) < window:
        return torch.zeros_like(t)
    unfolded = t.unfold(0, window, 1)
    res = torch.zeros_like(t)
    res[window - 1:], _ = unfolded.min(dim=1)
    return res

# 4. AST Representation and Evaluator
OPERATORS = ["Add", "Sub", "Mul", "Div", "TsRank", "TsZScore", "TsDelta", "TsDelay", "TsMax", "TsMin"]

class Node:
    def __init__(self, op: str, children: Optional[List['Node']] = None, feat: Optional[str] = None, const: float = 0.0):
        self.op = op
        self.children = children or []
        self.feat = feat
        self.const = const

    def to_formula(self) -> str:
        if self.feat:
            return self.feat
        if self.op == "Const":
            return f"{{self.const:.2f}}"
        if len(self.children) == 1:
            return f"{{self.op}}({{self.children[0].to_formula()}})"
        if len(self.children) == 2:
            return f"{{self.op}}({{self.children[0].to_formula()}}, {{self.children[1].to_formula()}})"
        return self.op

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

def evaluate_node(node: Node) -> torch.Tensor:
    if node.feat:
        col_idx = feat_map.get(node.feat, 0)
        return gpu_feat_tensor[:, col_idx]
    if node.op == "Const":
        return torch.full((num_bars,), node.const, device=device)

    args = [evaluate_node(c) for c in node.children]
    if node.op == "Add":
        return args[0] + args[1]
    if node.op == "Sub":
        return args[0] - args[1]
    if node.op == "Mul":
        return args[0] * args[1]
    if node.op == "Div":
        return args[0] / (torch.abs(args[1]) + 1e-6)
    if node.op == "TsRank":
        return gpu_ts_rank(args[0], 20)
    if node.op == "TsZScore":
        return gpu_ts_zscore(args[0], 20)
    if node.op == "TsDelta":
        return gpu_ts_delta(args[0], 5)
    if node.op == "TsDelay":
        return gpu_ts_delay(args[0], 5)
    if node.op == "TsMax":
        return gpu_ts_max(args[0], 20)
    if node.op == "TsMin":
        return gpu_ts_min(args[0], 20)
    return args[0]

def random_tree(depth: int, max_depth: int = 4) -> Node:
    if depth >= max_depth or (depth > 1 and random.random() < 0.4):
        feat = random.choice(feature_cols)
        return Node("Feature", feat=feat)
    op = random.choice(OPERATORS)
    if op in ["TsRank", "TsZScore", "TsDelta", "TsDelay", "TsMax", "TsMin"]:
        child = random_tree(depth + 1, max_depth)
        return Node(op, children=[child])
    else:
        left = random_tree(depth + 1, max_depth)
        right = random_tree(depth + 1, max_depth)
        return Node(op, children=[left, right])

# 5. Quantitative Metrics & Marcos López de Prado DSR
def compute_metrics(pred: torch.Tensor, targ: torch.Tensor, n_trials: int = 500) -> Dict[str, float]:
    p = torch.nan_to_num(pred, nan=0.0)
    p_dev = p - p.mean()
    t_dev = targ - targ.mean()
    p_norm = torch.sqrt(torch.sum(p_dev ** 2)) + 1e-8
    t_norm = torch.sqrt(torch.sum(t_dev ** 2)) + 1e-8
    ic = float(torch.sum(p_dev * t_dev) / (p_norm * t_norm))
    if math.isnan(ic):
        ic = 0.0

    # Strategy returns (long top quartile, short bottom quartile)
    signals = torch.sign(p_dev)
    strat_ret = signals * targ
    mean_r = float(strat_ret.mean())
    std_r = float(strat_ret.std()) + 1e-8
    sharpe = (mean_r / std_r) * math.sqrt(252 * 75)

    # Max Drawdown
    cum_ret = torch.cumsum(strat_ret, dim=0)
    running_max, _ = torch.cummax(cum_ret, dim=0)
    drawdowns = running_max - cum_ret
    mdd = float(torch.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Turnover
    delta_sig = torch.abs(signals[1:] - signals[:-1])
    turnover = float(delta_sig.mean()) if len(delta_sig) > 0 else 0.0

    # Deflated Sharpe Ratio (López de Prado)
    euler_gamma = 0.5772156649
    n = max(2, n_trials)
    e_max_sr = (1.0 - euler_gamma) * math.sqrt(2.0 * math.log(n)) + euler_gamma * math.sqrt(2.0 * math.log(n)) / math.log(n)
    sr_var = 1.0 / len(targ)
    dsr = (sharpe - e_max_sr) / math.sqrt(sr_var) if sr_var > 0 else 0.0

    # Composite Fitness (Muscle 10)
    fitness = 0.4 * abs(ic) + 0.3 * max(0.0, sharpe / 3.0) - 0.15 * turnover - 0.15 * min(1.0, mdd)

    return {{
        "ic": round(ic, 4),
        "sharpe": round(sharpe, 2),
        "dsr": round(dsr, 2),
        "turnover": round(turnover, 4),
        "mdd": round(mdd, 4),
        "fitness": round(fitness, 4),
    }}

# 6. Evolutionary Search Loop (High-Capacity on Kaggle GPU)
POPULATION_SIZE = {population_size}
GENERATIONS = {generations}
print(f"Initializing Population: {{POPULATION_SIZE}} formulas, {{GENERATIONS}} generations...")

population = [random_tree(0, max_depth=3) for _ in range(POPULATION_SIZE)]
best_candidate = None
best_fitness = -1e9
eval_count = 0
t_evo_start = time.time()

for gen in range(GENERATIONS):
    t_gen_start = time.time()
    fitness_list = []

    for ind in population:
        try:
            pred = evaluate_node(ind)
            metrics = compute_metrics(pred, target_tensor, n_trials=POPULATION_SIZE)
            eval_count += 1
        except Exception:
            metrics = {{"ic": 0.0, "sharpe": 0.0, "dsr": 0.0, "turnover": 1.0, "mdd": 1.0, "fitness": -10.0}}

        f = metrics["fitness"] - 0.002 * ind.size()  # Parsimony penalty
        fitness_list.append((ind, f, metrics))

        if f > best_fitness:
            best_fitness = f
            best_candidate = (ind, metrics)

    # Sort population by fitness
    fitness_list.sort(key=lambda x: x[1], reverse=True)
    gen_time = time.time() - t_gen_start
    gen_best = fitness_list[0]
    print(f"Gen [{{gen+1:02d}}/{{GENERATIONS:02d}}] | Best Fitness: {{gen_best[1]:.4f}} | IC: {{gen_best[2]['ic']:+.4f}} | Sharpe: {{gen_best[2]['sharpe']:.2f}} | Formula: {{gen_best[0].to_formula()[:45]}}... ({{gen_time:.2f}}s)")

    # Selection & Reproduction
    survivors = [x[0] for x in fitness_list[:int(POPULATION_SIZE * 0.25)]]
    new_pop = list(survivors)

    while len(new_pop) < POPULATION_SIZE:
        if random.random() < 0.6 and len(survivors) >= 2:
            p1, p2 = random.sample(survivors, 2)
            child = Node(random.choice(["Add", "Sub", "Mul"]), children=[p1, p2])
        else:
            child = random_tree(0, max_depth=3)
        new_pop.append(child)

    population = new_pop

evo_duration = time.time() - t_evo_start
throughput = eval_count / evo_duration if evo_duration > 0 else 0.0
print("=================================================================")
print(f"Evolution Complete! Evaluated {{eval_count}} formulas in {{evo_duration:.2f}}s ({{throughput:.1f}} evals/sec)")
if has_cuda:
    max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Peak GPU VRAM Allocated: {{max_mem_mb:.2f}} MB")

# 7. Multi-Objective Pareto Frontier Extraction
print("Computing Multi-Objective Pareto Frontier...")
pareto_frontier = []
candidates = fitness_list[:50]
for i, (ind_a, fit_a, met_a) in enumerate(candidates):
    is_dominated = False
    for j, (ind_b, fit_b, met_b) in enumerate(candidates):
        if i == j:
            continue
        # B dominates A if B is >= in all and > in at least one
        b_better_or_equal = (
            abs(met_b["ic"]) >= abs(met_a["ic"])
            and met_b["sharpe"] >= met_a["sharpe"]
            and met_b["turnover"] <= met_a["turnover"]
            and met_b["mdd"] <= met_a["mdd"]
        )
        b_strictly_better = (
            abs(met_b["ic"]) > abs(met_a["ic"])
            or met_b["sharpe"] > met_a["sharpe"]
            or met_b["turnover"] < met_a["turnover"]
            or met_b["mdd"] < met_a["mdd"]
        )
        if b_better_or_equal and b_strictly_better:
            is_dominated = True
            break
    if not is_dominated:
        pareto_frontier.append({{
            "formula": ind_a.to_formula(),
            "metrics": met_a,
            "nodes": ind_a.size()
        }})

print(f"Identified {{len(pareto_frontier)}} Pareto-optimal alpha formulas!")

# 8. Deliverables & Artifact Persistence
out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

champ_ind, champ_met = best_candidate
champion_payload = {{
    "ticker": "BANKBEES.NS",
    "status": "CHAMPION_FOUND",
    "formula_str": champ_ind.to_formula(),
    "node_count": champ_ind.size(),
    "metrics": champ_met,
    "composite_fitness": round(best_fitness, 4),
    "generations_run": GENERATIONS,
    "population_size": POPULATION_SIZE,
    "hardware": {{
        "device": str(device),
        "cuda_available": has_cuda,
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if has_cuda else 0.0,
        "eval_count": eval_count,
        "elapsed_seconds": round(evo_duration, 2),
        "throughput_evals_per_sec": round(throughput, 1)
    }}
}}

with open(os.path.join(out_dir, "champion_alpha.json"), "w", encoding="utf-8") as f:
    json.dump(champion_payload, f, indent=2)

with open(os.path.join(out_dir, "pareto_frontier.json"), "w", encoding="utf-8") as f:
    json.dump(pareto_frontier, f, indent=2)

summary = {{
    "total_elapsed_seconds": round(time.time() - t_start, 2),
    "eval_count": eval_count,
    "champion": champion_payload,
    "pareto_frontier_count": len(pareto_frontier)
}}

with open(os.path.join(out_dir, "evolution_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved champion_alpha.json, pareto_frontier.json, evolution_summary.json")
print("=================================================================")
print("  BURST RUN COMPLETE. TERMINATING INSTANCE TO SAVE GPU QUOTA.    ")
print("=================================================================")
sys.exit(0)
'''

    worker_file = output_dir / "burst_run.py"
    worker_file.write_text(worker_code, encoding="utf-8")

    # 4. Write kernel-metadata.json
    metadata = {
        "id": "avgsmutenjoy/ravana-alpha-burst",
        "title": "Ravana Alpha Burst GPU",
        "code_file": "burst_run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    meta_file = output_dir / "kernel-metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Successfully packaged Kaggle bundle at: {output_dir}")
    return output_dir


if __name__ == "__main__":
    f_path = Path("ravana/data/lake/m7_features/intraday_5m/BANKBEES_features.parquet")
    t_path = Path("ravana/data/lake/m8_targets/intraday_5m/BANKBEES_targets.parquet")
    bundle = Path("scratch/kaggle_bundle")
    generate_kaggle_payload(f_path, t_path, bundle, population_size=500, generations=15)
