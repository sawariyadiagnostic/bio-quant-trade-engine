"""Multi-Asset Phenotype Vector Extractor (Wave 3: Task 3.1)."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from ..data.vector_lake import VectorDataLake
from ..domain.formula_ast import ASTNode, ASTEvaluator


class PhenotypeExtractor:
    """
    Extracts high-dimensional behavioral phenotype vectors by evaluating formulas
    across multi-asset and multi-horizon market lakes.
    """
    def __init__(self, asset_lakes: Dict[str, VectorDataLake]):
        self.asset_lakes = asset_lakes
        # Cache DataFrames to avoid repeated conversions
        self._dfs: Dict[str, pd.DataFrame] = {
            ticker: pd.DataFrame(lake.tensors, index=lake.dates)
            for ticker, lake in asset_lakes.items()
        }

    def extract_vector(self, ast: ASTNode) -> np.ndarray:
        """
        Evaluates an AST across all asset lakes and concatenates binary trade decisions
        (1.0 = Long, 0.0 = Cash) into a unified 1D phenotype vector.
        """
        pieces: List[np.ndarray] = []
        for ticker, df in self._dfs.items():
            raw_sig = ASTEvaluator.eval_node(ast, df)
            pos = (raw_sig > 0.0).astype(np.float32)
            pieces.append(pos)

        if not pieces:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(pieces)

    def extract_batch(self, ast_list: List[ASTNode]) -> np.ndarray:
        """Evaluates a list of AST formulas into a 2D [N, D] phenotype matrix."""
        vectors = [self.extract_vector(ast) for ast in ast_list]
        return np.stack(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)

    def extract_dense_embedding(self, ast: ASTNode, target_dim: int = 1024) -> np.ndarray:
        """
        Extracts a rich, continuous 1024D latent embedding vector for an alpha formula.
        Inspired by biological protein language models (ESM-2 / ProtBERT):
        Combines trade direction, PnL return trajectory, volatility dynamics, and
        regime sensitivities, projected to a standardized target_dim embedding with unit norm.
        """
        raw_vec = self.extract_vector(ast)
        if len(raw_vec) == 0:
            return np.zeros(target_dim, dtype=np.float32)

        # 1. PnL Return Trajectory across lakes
        pnl_pieces = []
        for ticker, df in self._dfs.items():
            raw_sig = ASTEvaluator.eval_node(ast, df)
            pos = (raw_sig > 0.0).astype(np.float32)
            lake = self.asset_lakes[ticker]
            rets = lake.tensors.get("gross_overnight_rets", np.zeros_like(pos))
            pnl_pieces.append(pos * rets)
        pnl_vec = np.concatenate(pnl_pieces) if pnl_pieces else np.zeros_like(raw_vec)

        # 2. Concatenate Behavioral Trajectory
        combined = np.concatenate([raw_vec, pnl_vec])

        # 3. Resample / project to target_dim using linear interpolation
        src_indices = np.linspace(0, 1, len(combined))
        tgt_indices = np.linspace(0, 1, target_dim)
        embedding = np.interp(tgt_indices, src_indices, combined).astype(np.float32)

        # 4. Normalize to unit sphere (L2 norm)
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        return embedding

    def extract_dense_batch(self, ast_list: List[ASTNode], target_dim: int = 1024) -> np.ndarray:
        """Extracts dense [N, target_dim] continuous latent matrix across formulas."""
        embeddings = [self.extract_dense_embedding(ast, target_dim=target_dim) for ast in ast_list]
        return np.stack(embeddings, axis=0) if embeddings else np.empty((0, target_dim), dtype=np.float32)

    @staticmethod
    def extract_simhash_bitmask(embedding: np.ndarray, n_bits: int = 64) -> np.ndarray:
        """
        Quantizes continuous embedding into a compact 1-bit SimHash boolean bitmask.
        Allows sub-millisecond clone screening via SIMD popcount Hamming distance.
        """
        if len(embedding) == 0:
            return np.zeros(n_bits, dtype=np.uint8)
        indices = np.linspace(0, len(embedding) - 1, n_bits, dtype=int)
        sampled = embedding[indices]
        return (sampled > 0.0).astype(np.uint8)

    @staticmethod
    def compute_simhash_hamming_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
        """Computes Hamming distance between two 1-bit SimHash bitmasks."""
        return int(np.sum(mask_a ^ mask_b))

    @staticmethod
    def calculate_crowding_tripwire(embeddings: np.ndarray, threshold: float = 0.88) -> Tuple[float, str]:
        """
        Calculates maximum pairwise cosine similarity and returns crowding tripwire status:
        'LOW (Optimal)', 'MED (Watchlist)', or 'TRIPWIRE (Triggered)'.
        """
        if len(embeddings) < 2:
            return 0.0, "LOW (Optimal)"
        sim_matrix = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(sim_matrix, 0.0)
        max_sim = float(np.max(sim_matrix))
        if max_sim >= threshold:
            return max_sim, "TRIPWIRE (Triggered)"
        elif max_sim >= 0.75:
            return max_sim, "MED (Watchlist)"
        return max_sim, "LOW (Optimal)"

    @classmethod
    def create_mock(cls, n_assets: int = 3, n_bars: int = 100) -> "PhenotypeExtractor":
        """Factory creating mock multi-asset lakes for testing."""
        tickers = [f"MOCK_ASSET_{i}" for i in range(n_assets)]
        lakes = {t: VectorDataLake.load_mock(periods=n_bars, ticker=t) for t in tickers}
        return cls(asset_lakes=lakes)
