"""Vectorized Matrix Backtesting Engine (Wave 1: Task 1.2)."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class MatrixSimulator:
    """Ultra-fast vectorized backtester processing thousands of formulas simultaneously."""
    friction_bps: float = 1.2  # 0.00012 for ETF Delivery

    def simulate_batch(
        self,
        positions: np.ndarray,
        returns: np.ndarray,
        lookback_years: float = 5.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulates an entire batch of formulas simultaneously without Python loops.

        Args:
            positions: 2D array of binary trade decisions (shape: [B, T]).
            returns: 1D array of gross market returns (shape: [T]).
            lookback_years: Historical horizon duration for CAGR normalization.

        Returns:
            Dictionary of performance metric arrays (each of shape: [B]).
        """
        if positions.ndim != 2:
            raise ValueError(f"positions must be 2D [B, T], got shape {positions.shape}")
        B, T = positions.shape
        if len(returns) != T:
            raise ValueError(f"returns length ({len(returns)}) must match positions columns ({T})")

        friction = self.friction_bps / 10000.0

        # 1. Vectorized Position Flips & Friction
        zero_col = np.zeros((B, 1), dtype=positions.dtype)
        flips = np.abs(np.diff(positions, axis=1, prepend=zero_col))
        net_rets = (positions * returns[np.newaxis, :]) - (flips * friction)

        # 2. Cumulative Equity & Drawdowns
        cum_eq = np.cumprod(1.0 + net_rets, axis=1)
        tot_ret = cum_eq[:, -1] - 1.0

        peaks = np.maximum.accumulate(cum_eq, axis=1)
        drawdowns = (peaks - cum_eq) / (peaks + 1e-8)
        max_dd = np.max(drawdowns, axis=1) * 100.0

        # 3. Normalized CAGR
        years = max(0.08, float(lookback_years))
        final_eq = np.maximum(1e-4, cum_eq[:, -1])
        cagr = (final_eq ** (1.0 / years) - 1.0) * 100.0

        # 4. Institutional Sortino Semi-Deviation with 20 bps Volatility Floor
        active_mask = positions > 0.0
        n_active = np.maximum(1.0, np.sum(active_mask, axis=1))
        active_rets = np.where(active_mask, net_rets, 0.0)
        mean_ret = np.sum(active_rets, axis=1) / n_active

        downside_sq = np.minimum(0.0, active_rets) ** 2
        # Institutional 20 bps (0.002) daily volatility floor prevents denominator collapse
        downside_dev = np.maximum(0.002, np.sqrt(np.sum(downside_sq, axis=1) / n_active))
        sortino = (mean_ret / downside_dev) * np.sqrt(252.0)

        # 5. Win Rate & Trade Confidences
        wins = np.sum((net_rets > 0.0) & active_mask, axis=1)
        win_rate = (wins / n_active) * 100.0

        calmar = np.maximum(0.0, cagr) / np.maximum(4.0, max_dd)

        return {
            "net_returns": net_rets,
            "cum_equity": cum_eq,
            "total_return_pct": tot_ret * 100.0,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "sortino": sortino,
            "calmar": calmar,
            "win_rate": win_rate,
            "mean_return": mean_ret,
            "n_active_bars": n_active
        }
