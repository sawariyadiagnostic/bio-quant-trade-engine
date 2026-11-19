"""Muscle 16: Monte Carlo Path Reshuffler & Stress Tester.

Uses Stationary Block Bootstrapping (Politis & Romano 1994) to preserve serial
correlation and volatility clustering across 1,000+ synthetic equity paths,
estimating tail risk (VaR, CVaR, CDaR, 95th/99th percentile Max Drawdown) and
Probability of breaching Muscle 12 / 18 circuit breakers (4% CAUTION, 8% HALT).
"""
from __future__ import annotations

import math
from typing import Optional, Union
import numpy as np
import pandas as pd

from ravana.backtest.engine import BacktestResult
from ravana.fitness.metrics import compute_max_drawdown
from ravana.validation.config import (
    MonteCarloConfig,
    MonteCarloReport,
    WalkForwardResult,
)


class MonteCarloStressTester:
    """Stationary block-bootstrap path simulator quantifying tail drawdown and ruin risk."""

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self.config = config or MonteCarloConfig()

    def run_from_result(
        self, result: Union[WalkForwardResult, BacktestResult, np.ndarray]
    ) -> MonteCarloReport:
        """Chain directly from Muscle 13 WalkForwardResult or Muscle 11 BacktestResult."""
        if isinstance(result, WalkForwardResult):
            return self.run_stress_test(result.stitched_oos_returns)
        if isinstance(result, BacktestResult):
            # Resample intraday equity curve to daily session closes if available
            if isinstance(result.equity_curve.index, pd.DatetimeIndex):
                daily_equity = result.equity_curve.groupby(result.equity_curve.index.date).last()
                rets = daily_equity.pct_change().dropna().to_numpy(dtype=np.float64)
            else:
                rets = result.equity_curve.pct_change().dropna().to_numpy(dtype=np.float64)
            return self.run_stress_test(rets)
        return self.run_stress_test(np.asarray(result, dtype=np.float64))

    def _empty_report(self) -> MonteCarloReport:
        """Return a conservative rejection report for empty or degenerate input."""
        return MonteCarloReport(
            num_simulations=0,
            block_size=1,
            median_cagr=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            median_max_drawdown=1.0,
            dd_95_percentile=1.0,
            dd_99_percentile=1.0,
            probability_of_caution=1.0,
            probability_of_ruin=1.0,
            passed_stress_test=False,
            cdar_95=1.0,
        )

    @staticmethod
    def _simulate_paths(
        valid: np.ndarray,
        n: int,
        n_sims: int,
        p: float,
        ann_exponent: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic return paths via stationary bootstrap."""
        max_drawdowns = np.zeros(n_sims, dtype=np.float64)
        annualized_cagrs = np.zeros(n_sims, dtype=np.float64)
        for sim in range(n_sims):
            indices = np.zeros(n, dtype=np.int64)
            current_idx = rng.integers(0, n)
            for t in range(n):
                if t > 0:
                    current_idx = rng.integers(0, n) if rng.random() < p else (current_idx + 1) % n
                indices[t] = current_idx
            sim_path = valid[indices]
            max_drawdowns[sim] = compute_max_drawdown(sim_path)
            wealth_end = float(np.prod(1.0 + np.clip(sim_path, -0.99, 10.0)))
            annualized_cagrs[sim] = (wealth_end ** ann_exponent) - 1.0 if wealth_end > 0.0 else -1.0
        return max_drawdowns, annualized_cagrs

    def run_stress_test(self, returns: np.ndarray) -> MonteCarloReport:
        """Simulate synthetic return paths via Politis & Romano (1994) Stationary Bootstrap."""
        valid = returns[np.isfinite(returns)]
        n = len(valid)
        if n < 5:
            return self._empty_report()

        rng = np.random.default_rng(self.config.random_seed)
        avg_block = self.config.block_size or max(5, int(round(n ** (1.0 / 3.0))))
        p = 1.0 / float(avg_block)
        n_sims = self.config.num_simulations
        periods_per_yr = getattr(self.config, "periods_per_year", 252.0)
        ann_exponent = periods_per_yr / float(max(1, n))

        max_drawdowns, annualized_cagrs = self._simulate_paths(valid, n, n_sims, p, ann_exponent, rng)

        var_95 = float(np.percentile(valid, 5))
        var_99 = float(np.percentile(valid, 1))
        tail_losses = valid[valid <= var_95]
        cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

        med_dd = float(np.median(max_drawdowns))
        dd_95 = float(np.percentile(max_drawdowns, 95))
        dd_99 = float(np.percentile(max_drawdowns, 99))
        caution_prob = float(np.mean(max_drawdowns >= self.config.caution_drawdown_threshold))
        ruin_prob = float(np.mean(max_drawdowns >= self.config.ruin_drawdown_threshold))
        med_cagr = float(np.median(annualized_cagrs))

        worst_drawdowns = max_drawdowns[max_drawdowns >= dd_95]
        cdar_95 = float(np.mean(worst_drawdowns)) if len(worst_drawdowns) > 0 else dd_95
        max_ruin_p = getattr(self.config, "max_acceptable_ruin_prob", 0.05)
        passed = (ruin_prob <= max_ruin_p and med_cagr > 0.0)

        return MonteCarloReport(
            num_simulations=n_sims,
            block_size=avg_block,
            median_cagr=round(med_cagr, 5),
            var_95=round(var_95, 5),
            var_99=round(var_99, 5),
            cvar_95=round(cvar_95, 5),
            median_max_drawdown=round(med_dd, 5),
            dd_95_percentile=round(dd_95, 5),
            dd_99_percentile=round(dd_99, 5),
            probability_of_caution=round(caution_prob, 4),
            probability_of_ruin=round(ruin_prob, 4),
            passed_stress_test=passed,
            cdar_95=round(cdar_95, 5),
        )
