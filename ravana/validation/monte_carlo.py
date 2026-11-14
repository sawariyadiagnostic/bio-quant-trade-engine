"""Muscle 16: Monte Carlo Path Reshuffler & Stress Tester.

Uses Circular Block Bootstrapping (Politis & Romano) to preserve serial correlation
and volatility clustering across 1,000+ synthetic equity paths, estimating tail risk
(VaR, CVaR, 95th/99th percentile Max Drawdown) and Probability of breaching
Muscle 12's 4% CAUTION and 8% HALT drawdown circuit breakers.
"""
from __future__ import annotations

import math
from typing import Optional, Union
import numpy as np

from ravana.backtest.engine import BacktestResult
from ravana.fitness.metrics import compute_max_drawdown
from ravana.validation.config import (
    MonteCarloConfig,
    MonteCarloReport,
    WalkForwardResult,
)


class MonteCarloStressTester:
    """Block-bootstrap path simulator quantifying tail drawdown and ruin risk."""

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self.config = config or MonteCarloConfig()

    def run_from_result(
        self, result: Union[WalkForwardResult, BacktestResult, np.ndarray]
    ) -> MonteCarloReport:
        """Chain directly from Muscle 13 WalkForwardResult or Muscle 11 BacktestResult."""
        if isinstance(result, WalkForwardResult):
            return self.run_stress_test(result.stitched_oos_returns)
        if isinstance(result, BacktestResult):
            rets = result.equity_curve.pct_change().dropna().to_numpy(dtype=np.float64)
            return self.run_stress_test(rets)
        return self.run_stress_test(np.asarray(result, dtype=np.float64))

    def run_stress_test(self, returns: np.ndarray) -> MonteCarloReport:
        """Simulate synthetic return paths via Circular Block Bootstrap."""
        valid = returns[np.isfinite(returns)]
        n = len(valid)
        if n < 5:
            return MonteCarloReport(
                num_simulations=0, block_size=1, median_cagr=0.0,
                var_95=0.0, var_99=0.0, cvar_95=0.0,
                median_max_drawdown=0.0, dd_95_percentile=0.0,
                dd_99_percentile=0.0, probability_of_caution=1.0,
                probability_of_ruin=1.0, passed_stress_test=False,
            )

        rng = np.random.default_rng(self.config.random_seed)
        block_size = self.config.block_size or max(2, int(round(n ** (1.0 / 3.0))))
        n_blocks = int(math.ceil(float(n) / float(block_size)))

        wrapped = np.concatenate([valid, valid[:block_size]])
        n_sims = self.config.num_simulations

        max_drawdowns = np.zeros(n_sims, dtype=np.float64)
        annualized_cagrs = np.zeros(n_sims, dtype=np.float64)
        ann_exponent = self.config.periods_per_year / float(max(1, n))

        for sim in range(n_sims):
            starts = rng.integers(0, n, size=n_blocks)
            blocks = [wrapped[s : s + block_size] for s in starts]
            sim_path = np.concatenate(blocks)[:n]

            max_drawdowns[sim] = compute_max_drawdown(sim_path)
            wealth_end = float(np.prod(1.0 + np.clip(sim_path, -0.99, 10.0)))
            annualized_cagrs[sim] = (wealth_end ** ann_exponent) - 1.0 if wealth_end > 0.0 else -1.0

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

        passed = (
            ruin_prob <= self.config.max_acceptable_ruin_prob
            and med_cagr > 0.0
        )

        return MonteCarloReport(
            num_simulations=n_sims,
            block_size=block_size,
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
        )
