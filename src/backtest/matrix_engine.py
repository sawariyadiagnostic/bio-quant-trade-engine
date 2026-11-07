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
            positions: 2D array [B, T] or 3D multi-asset array [B, N, T].
            returns: 1D array [T] or 2D multi-asset array [N, T].
            lookback_years: Historical horizon duration for CAGR normalization.

        Returns:
            Dictionary of performance metric arrays (each of shape: [B]).
        """
        friction = self.friction_bps / 10000.0

        if positions.ndim == 2:
            B, T = positions.shape
            if len(returns) != T:
                raise ValueError(f"returns length ({len(returns)}) must match positions columns ({T})")
            zero_col = np.zeros((B, 1), dtype=positions.dtype)
            flips = np.abs(np.diff(positions, axis=1, prepend=zero_col))
            net_rets = (positions * returns[np.newaxis, :]) - (flips * friction)
            active_mask = positions > 0.0

        elif positions.ndim == 3:
            B, N, T = positions.shape
            rets_arr = np.asarray(returns)
            if rets_arr.ndim == 1:
                if len(rets_arr) != T:
                    raise ValueError(f"returns length ({len(rets_arr)}) must match columns ({T})")
                rets_2d = np.broadcast_to(rets_arr[np.newaxis, :], (N, T))
            elif rets_arr.ndim == 2:
                if rets_arr.shape != (N, T):
                    raise ValueError(f"returns shape {rets_arr.shape} must match (N={N}, T={T})")
                rets_2d = rets_arr
            else:
                raise ValueError(f"returns must be 1D [T] or 2D [N, T], got shape {rets_arr.shape}")

            zero_col = np.zeros((B, N, 1), dtype=positions.dtype)
            flips = np.abs(np.diff(positions, axis=2, prepend=zero_col))
            turnover_loss = np.sum(flips, axis=1) * friction
            gross_port_rets = np.sum(positions * rets_2d[np.newaxis, :, :], axis=1)
            net_rets = gross_port_rets - turnover_loss
            active_mask = np.sum(np.abs(positions) > 0.0, axis=1) > 0.0

        else:
            raise ValueError(f"positions must be 2D [B, T] or 3D [B, N, T], got shape {positions.shape}")

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

    def walk_forward_cv(
        self,
        positions: np.ndarray,
        returns: np.ndarray,
        n_splits: int = 5,
        purge_bars: int = 5,
        embargo_bars: int = 2,
        min_train_bars: int = 60
    ) -> Dict[str, Any]:
        """
        Executes Purged & Embargoed Walk-Forward Cross Validation (López de Prado, 2018).
        Ensures out-of-sample slices have strict temporal isolation:
        test_start >= train_end + purge_bars + embargo_bars.
        """
        T = positions.shape[-1]
        buffer_bars = purge_bars + embargo_bars
        available_t = T - min_train_bars - buffer_bars
        if available_t <= 0 or n_splits < 1:
            raise ValueError(f"Insufficient time steps T={T} for n_splits={n_splits} with buffer={buffer_bars}")

        test_chunk = max(10, available_t // n_splits)
        folds = []
        test_sortinos = []
        test_cagrs = []
        test_dds = []

        for split in range(n_splits):
            test_start = min_train_bars + (split * test_chunk) + buffer_bars
            test_end = min(T, test_start + test_chunk)
            train_end = test_start - buffer_bars

            if test_start >= T or train_end <= 0:
                break

            if positions.ndim == 2:
                pos_train, pos_test = positions[:, :train_end], positions[:, test_start:test_end]
                ret_train, ret_test = returns[:train_end], returns[test_start:test_end]
            else:
                pos_train, pos_test = positions[:, :, :train_end], positions[:, :, test_start:test_end]
                ret_train = returns[:, :train_end] if returns.ndim == 2 else returns[:train_end]
                ret_test = returns[:, test_start:test_end] if returns.ndim == 2 else returns[test_start:test_end]

            train_res = self.simulate_batch(pos_train, ret_train, lookback_years=max(0.1, train_end / 252.0))
            test_res = self.simulate_batch(pos_test, ret_test, lookback_years=max(0.1, (test_end - test_start) / 252.0))

            folds.append({
                "split": split,
                "train_range": (0, train_end),
                "test_range": (test_start, test_end),
                "train_sortino": train_res["sortino"],
                "test_sortino": test_res["sortino"],
                "test_cagr": test_res["cagr"],
                "test_drawdown": test_res["max_drawdown"],
            })
            test_sortinos.append(test_res["sortino"])
            test_cagrs.append(test_res["cagr"])
            test_dds.append(test_res["max_drawdown"])

        mean_sortino = np.mean(test_sortinos, axis=0) if test_sortinos else np.zeros(positions.shape[0])
        mean_cagr = np.mean(test_cagrs, axis=0) if test_cagrs else np.zeros(positions.shape[0])
        mean_dd = np.mean(test_dds, axis=0) if test_dds else np.zeros(positions.shape[0])

        return {
            "folds": folds,
            "mean_test_sortino": mean_sortino,
            "mean_test_cagr": mean_cagr,
            "mean_test_drawdown": mean_dd
        }

    @staticmethod
    def purged_walk_forward_split(
        n_bars: int,
        train_ratio: float = 0.7,
        purge_window: int = 5,
        embargo_window: int = 5
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Generates train and test slice boundaries with strict purge and embargo buffers."""
        train_end = int(n_bars * train_ratio)
        test_start = min(n_bars, train_end + purge_window + embargo_window)
        return (0, train_end), (test_start, n_bars)
