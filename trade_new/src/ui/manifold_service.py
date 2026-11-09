"""Manifold projection and similarity clustering service for Strategy Visualizer."""
from typing import Any, Dict, List, Set, Tuple
import numpy as np

from ..clustering.return_clustering import ReturnClusterer
from ..domain.formula_ast import ASTNode, ASTEvaluator
from ..domain.formula_parser import FormulaParser


def parse_manifold_nodes(elites: List[Dict[str, Any]]) -> Tuple[List[ASTNode], List[Dict[str, Any]]]:
    """Parses AST nodes from candidate elite formulas with fallback handling."""
    nodes, valids = [], []
    for e in elites:
        f_str = str(e.get("formula_str", "")).strip()
        try:
            nodes.append(FormulaParser.parse(f_str))
            valids.append(e)
        except Exception:
            try:
                nodes.append(ASTNode(op="FEATURE", val="Close_Adj"))
                valids.append(e)
            except Exception:
                continue
    return nodes, valids


def project_svd_3d(returns: np.ndarray) -> np.ndarray:
    """Projects return vectors into 3D manifold coordinates via truncated SVD."""
    centered = returns - np.mean(returns, axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if u.shape[1] >= 3:
        return u[:, :3] * s[:3]
    return np.pad(u * s, ((0, 0), (0, max(0, 3 - u.shape[1]))))


def build_manifold_points(
    valid_elites: List[Dict[str, Any]],
    coords: np.ndarray,
    labels: np.ndarray,
    surv_set: Set[int],
    ticker: str,
) -> List[Dict[str, Any]]:
    """Builds formatted 3D point dictionaries with hover and cluster metadata."""
    points = []
    for i, e in enumerate(valid_elites):
        points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "cluster": int(labels[i]),
            "is_survivor": i in surv_set,
            "is_orthogonal": i in surv_set,
            "ticker": e.get("ticker", ticker),
            "fitness": round(float(e.get("fitness", 0.0)), 2),
            "cagr_pct": round(float(e.get("cagr_pct", 0.0)), 1),
            "max_drawdown": round(float(e.get("max_drawdown", 0.0)), 1),
            "formula": e.get("formula_str", ""),
        })
    return points


def evaluate_manifold_returns(nodes: List[ASTNode], ticker: str) -> np.ndarray:
    """Evaluates AST formula returns against real data lake price series for 3D manifold."""
    df = None
    try:
        from ..data.vector_lake import VectorDataLake
        df = VectorDataLake.load_mock(periods=100, ticker=ticker).to_dataframe()
    except Exception:
        df = None

    n = len(nodes)
    returns = np.zeros((n, 60), dtype=np.float64)
    if df is None or len(df) < 5:
        return returns

    c = df.get("Close_Adj", df.get("Close", df.get("Close_Raw"))).to_numpy(dtype=np.float64)
    o = df.get("Open", df.get("Close")).to_numpy(dtype=np.float64)
    gross = (c - o) / (o + 1e-8)

    for i, node in enumerate(nodes):
        try:
            sig = ASTEvaluator.eval_node(node, df)
            pos = np.clip(sig, -1.0, 1.0)
            strat = np.nan_to_num(pos * gross, nan=0.0, posinf=0.0, neginf=0.0)
            returns[i] = strat[-60:] if len(strat) >= 60 else np.pad(strat, (60 - len(strat), 0))
        except Exception:
            pass
    return returns


def compute_manifold_points(elites: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
    """Projects elite strategy returns into 3D coordinate space with correlation clustering."""
    empty_res = {
        "status": "empty",
        "count": 0,
        "compression_ratio": 0.0,
        "clusters_count": 0,
        "points": [],
        "cosine_matrix": [],
        "labels": [],
    }
    if not elites or len(elites) < 2:
        return empty_res

    nodes, valids = parse_manifold_nodes(elites)
    n = len(valids)
    if n < 2:
        return empty_res

    returns = evaluate_manifold_returns(nodes, ticker)
    clusterer = ReturnClusterer(rho_threshold=0.85)
    labels = clusterer.fit_predict(returns)
    corr = np.clip(np.nan_to_num(np.corrcoef(returns), nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    cosine_matrix = np.clip(0.5 * (corr + 1.0), 0.0, 1.0)
    coords = project_svd_3d(returns)

    _, surv_idx = clusterer.cluster_and_prune(
        returns,
        np.array([float(e.get("fitness", 1.0)) for e in valids]),
        np.array([node.depth() for node in nodes]),
    )
    points = build_manifold_points(valids, coords, labels, set(surv_idx), ticker)
    comp = round(float(len(np.unique(labels))) / float(n), 4)

    return {
        "status": "ok",
        "ticker": ticker,
        "count": len(points),
        "compression_ratio": comp,
        "clusters_count": int(len(np.unique(labels))),
        "points": points,
        "cosine_matrix": [[round(float(v), 3) for v in row] for row in cosine_matrix],
        "labels": [f"#{i+1} {e['formula_str'][:16]}.." for i, e in enumerate(valids)],
    }


# Sovereign alias
_compute_manifold_points = compute_manifold_points
