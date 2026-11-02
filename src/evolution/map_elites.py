"""
20x20 MAP-Elites Behavioral Archive (Phase B).
Illuminates 400 behavioral niches indexed by Volatility and Turnover.
Enforces:
- Statutory Tax Deductions (1.2 bps ETF / 3 bps Futures / 25 bps Equity Cash).
- Calmar / Sortino multi-objective fitness.
- Minimum Tax Hurdle (rejects high churn with sub-friction alpha).
- Bloat Deduplication via behavioral correlation (>0.95 threshold).
"""
import os
import json
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from ..domain.formula_ast import ASTNode, ASTEvaluator
from ..domain.formula_parser import FormulaParser
from ..domain.models import InstrumentType

@dataclass
class EliteIndividual:
    formula: ASTNode
    fitness: float          # Composite Institutional Alpha Score (Sortino x ln(1+PF) x Confidence)
    volatility: float       # Annualized volatility
    turnover: float         # Annualized turnover
    win_rate: float         # Win rate %
    net_return_pct: float   # Net cumulative return %
    behavior_signature: np.ndarray # Trailing signal vector
    sortino: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    max_drawdown: float = 0.0
    cagr_pct: float = 0.0   # Compound Annual Growth Rate % (normalized across lookbacks)
    lookback: str = "5y"

class MapElitesArchive:
    def __init__(self, grid_size: tuple[int, int] = (20, 20)):
        self.rows, self.cols = grid_size
        self.grid = np.empty(grid_size, dtype=object)

        # Behavioral boundary bins
        self.vol_bins = np.linspace(0.05, 0.60, self.rows)
        self.turnover_bins = np.linspace(2.0, 150.0, self.cols)

    def _get_bin_indices(self, vol: float, turnover: float) -> tuple[int, int]:
        r = int(np.clip(np.digitize(vol, self.vol_bins) - 1, 0, self.rows - 1))
        c = int(np.clip(np.digitize(turnover, self.turnover_bins) - 1, 0, self.cols - 1))
        return r, c

    def _get_statutory_friction(self, instrument: InstrumentType) -> float:
        if instrument == InstrumentType.ETF_CASH:
            return 0.00012
        if instrument == InstrumentType.FUTURES:
            return 0.00030
        return 0.00250

    def _calculate_metrics(
        self,
        positions: np.ndarray,
        open_p: np.ndarray,
        close_p: np.ndarray,
        instrument: InstrumentType,
        target_mode: str
    ) -> Optional[dict]:
        """Calculates institutional metrics, Sortino, Profit Factor, and Composite Alpha Score."""
        if target_mode == "overnight":
            gross_returns = (open_p[1:] - close_p[:-1]) / (close_p[:-1] + 1e-8)
            trade_positions = positions[:-1]
        else:
            gross_returns = (close_p - open_p) / (open_p + 1e-8)
            trade_positions = positions

        pos_flips = np.abs(np.diff(trade_positions, prepend=0))
        friction = self._get_statutory_friction(instrument)
        net_returns = (trade_positions * gross_returns) - (pos_flips * friction)

        active_trades = net_returns[trade_positions > 0]
        n_trades = len(active_trades)
        if n_trades < 5:
            return None

        annual_vol = float(np.std(net_returns) * np.sqrt(252))
        turnover = float(np.mean(pos_flips) * 252)

        cum_equity = np.cumprod(1.0 + net_returns)
        total_net_ret = float(cum_equity[-1] - 1.0) if len(cum_equity) > 0 else 0.0
        n_bars = len(net_returns)
        years = max(0.08, n_bars / 252.0)
        cagr = float(((1.0 + total_net_ret) ** (1.0 / years)) - 1.0) * 100.0 if total_net_ret > -0.999 else -100.0
        peak = np.maximum.accumulate(cum_equity)
        drawdowns = (peak - cum_equity) / (peak + 1e-8)
        max_drawdown = float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0
        win_rate = float(np.mean(active_trades > 0) * 100.0) if n_trades > 0 else 0.0

        mean_ret = float(np.mean(active_trades))
        downside_trades = active_trades[active_trades < 0]
        downside_dev = float(np.std(downside_trades)) if len(downside_trades) > 0 else 0.001
        sortino = float((mean_ret / (downside_dev + 1e-6)) * np.sqrt(252))

        gross_gains = float(np.sum(active_trades[active_trades > 0])) if np.any(active_trades > 0) else 0.0
        gross_losses = abs(float(np.sum(active_trades[active_trades < 0]))) if np.any(active_trades < 0) else 1e-6
        profit_factor = float(gross_gains / gross_losses)

        calmar_ratio = max(0.0, cagr) / max(4.0, max_drawdown)
        expected_trades = max(15.0, 5.0 * years)
        trade_conf = min(1.2, max(0.2, float(np.sqrt(n_trades / expected_trades))))
        quality_mult = min(3.0, max(0.5, float(np.sqrt(max(0.25, min(sortino, 16.0))))))

        if turnover < 1.0:
            turnover_penalty = max(0.2, float(turnover / 1.0))
        elif turnover > 180.0:
            turnover_penalty = max(0.3, float(180.0 / turnover))
        else:
            turnover_penalty = 1.0

        institutional_fitness = float(max(0.0, calmar_ratio * quality_mult * trade_conf * turnover_penalty))

        sig = trade_positions[-60:] if len(trade_positions) >= 60 else np.pad(trade_positions, (60 - len(trade_positions), 0))

        return {
            "annual_vol": annual_vol,
            "turnover": turnover,
            "total_net_ret": total_net_ret,
            "cagr_pct": cagr,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "mean_ret": mean_ret,
            "sortino": sortino,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "institutional_fitness": institutional_fitness,
            "sig": sig,
        }

    def evaluate_and_insert(
        self,
        formula: ASTNode,
        df: pd.DataFrame,
        current_regime: int = 0,
        instrument: InstrumentType = InstrumentType.ETF_CASH,
        target_mode: str = "overnight" # "overnight" or "intraday"
    ) -> bool:
        """
        Evaluates candidate formula with exact statutory friction and inserts into archive if elite.
        """
        raw_signals = ASTEvaluator.eval_node(formula, df, current_regime=current_regime)
        if np.all(raw_signals == 0.0):
            return False

        # Long positions generated when signal > 0
        positions = np.where(raw_signals > 0, 1.0, 0.0)
        if np.sum(positions) == 0:
            return False

        c = df.get("Close_Adj", df.get("Close", df.get("Close_Raw"))).to_numpy(dtype=np.float64)
        o = df["Open"].to_numpy(dtype=np.float64)

        metrics = self._calculate_metrics(positions, o, c, instrument, target_mode)
        if metrics is None:
            return False

        institutional_fitness = metrics["institutional_fitness"]
        if institutional_fitness <= 0.0 or metrics["mean_ret"] <= 0.0:
            return False

        # Bloat Deduplication check
        sig = metrics["sig"]
        annual_vol = metrics["annual_vol"]
        turnover = metrics["turnover"]
        if not np.isfinite(annual_vol) or not np.isfinite(turnover):
            return False

        r, c_idx = self._get_bin_indices(annual_vol, turnover)
        incumbent = self.grid[r, c_idx]

        if incumbent is not None:
            std1 = float(np.std(sig))
            std2 = float(np.std(incumbent.behavior_signature))
            if std1 > 1e-6 and std2 > 1e-6:
                corr = np.corrcoef(sig, incumbent.behavior_signature)[0, 1]
                if not np.isnan(corr) and corr > 0.95 and institutional_fitness <= incumbent.fitness:
                    return False
            elif np.array_equal(sig, incumbent.behavior_signature) and institutional_fitness <= incumbent.fitness:
                return False

        # Elite Insertion
        if incumbent is None or institutional_fitness > incumbent.fitness:
            self.grid[r, c_idx] = EliteIndividual(
                formula=formula,
                fitness=round(institutional_fitness, 2),
                volatility=round(annual_vol, 4),
                turnover=round(turnover, 2),
                win_rate=round(metrics["win_rate"], 2),
                net_return_pct=round(metrics["total_net_ret"] * 100.0, 2),
                behavior_signature=sig,
                sortino=round(metrics["sortino"], 2),
                profit_factor=round(min(metrics["profit_factor"], 99.0), 2),
                trades_count=metrics["n_trades"],
                max_drawdown=round(metrics.get("max_drawdown", 0.0), 2),
                cagr_pct=round(metrics.get("cagr_pct", 0.0), 2)
            )
            return True

        return False

    def get_resident_elites(self) -> list[EliteIndividual]:
        return [cell for cell in self.grid.flatten() if cell is not None]

    def get_top_elites(self, n: int = 10) -> list[EliteIndividual]:
        elites = self.get_resident_elites()
        return sorted(elites, key=lambda x: x.fitness, reverse=True)[:n]

    def get_all_elite_records(self) -> list[dict]:
        """Returns structured dictionary records for all resident champions in the archive."""
        records = []
        for r in range(self.rows):
            for c in range(self.cols):
                elite = self.grid[r, c]
                if elite is not None:
                    records.append({
                        "row": r,
                        "col": c,
                        "fitness": elite.fitness,
                        "net_return_pct": elite.net_return_pct,
                        "cagr_pct": getattr(elite, "cagr_pct", elite.net_return_pct),
                        "max_drawdown": elite.max_drawdown,
                        "win_rate": elite.win_rate,
                        "sortino": elite.sortino,
                        "profit_factor": elite.profit_factor,
                        "turnover": elite.turnover,
                        "volatility": elite.volatility,
                        "trades_count": elite.trades_count,
                        "formula_str": elite.formula.to_formula_str(),
                        "niche_name": f"Vol {self.vol_bins[r]*100:.0f}% / Turn {self.turnover_bins[c]:.1f}x"
                    })
        return records

    def occupancy_rate(self) -> float:
        occupied = len(self.get_resident_elites())
        return float(occupied / (self.rows * self.cols))

    def save_to_disk(self, file_path: str):
        """Serializes the entire 20x20 MAP-Elites archive to disk for persistent continual evolution."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        records = []
        for r in range(self.rows):
            for c in range(self.cols):
                elite = self.grid[r, c]
                if elite is not None:
                    records.append({
                        "row": r,
                        "col": c,
                        "fitness": elite.fitness,
                        "volatility": elite.volatility,
                        "turnover": elite.turnover,
                        "win_rate": elite.win_rate,
                        "net_return_pct": elite.net_return_pct,
                        "cagr_pct": getattr(elite, "cagr_pct", elite.net_return_pct),
                        "max_drawdown": elite.max_drawdown,
                        "sortino": elite.sortino,
                        "profit_factor": elite.profit_factor,
                        "trades_count": elite.trades_count,
                        "formula_str": elite.formula.to_formula_str(),
                        "active_regimes": sorted(list(elite.formula.active_regimes)),
                        "behavior_signature": elite.behavior_signature.tolist()
                    })

        with open(file_path, "w") as f:
            json.dump({
                "rows": self.rows,
                "cols": self.cols,
                "resident_count": len(records),
                "occupancy_pct": round(self.occupancy_rate() * 100.0, 2),
                "elites": records
            }, f, indent=2)
        logger.info(f"Archive saved to {file_path}: {len(records)} resident elites persisted.")

    def load_from_disk(self, file_path: str) -> int:
        """Loads persistent archive from disk, restoring resident champions across runs."""
        if not os.path.exists(file_path):
            return 0

        with open(file_path, "r") as f:
            data = json.load(f)

        loaded_count = 0
        for item in data.get("elites", []):
            try:
                r, c = item["row"], item["col"]
                formula_node = FormulaParser.parse(item["formula_str"], active_regimes=set(item.get("active_regimes", range(10))))
                sig = np.array(item.get("behavior_signature", np.zeros(60)))
                self.grid[r, c] = EliteIndividual(
                    formula=formula_node,
                    fitness=item["fitness"],
                    volatility=item["volatility"],
                    turnover=item["turnover"],
                    win_rate=item["win_rate"],
                    net_return_pct=item["net_return_pct"],
                    behavior_signature=sig,
                    sortino=item.get("sortino", 0.0),
                    profit_factor=item.get("profit_factor", 0.0),
                    trades_count=item.get("trades_count", 0),
                    max_drawdown=item.get("max_drawdown", 0.0),
                    cagr_pct=item.get("cagr_pct", round(item.get("net_return_pct", 0.0), 2))
                )
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Skipped loading corrupted elite formula: {e}")

        logger.info(f"Loaded {loaded_count} resident elites from {file_path}. Occupancy: {self.occupancy_rate() * 100.0:.2f}%")
        return loaded_count

    def prune_decayed_elites(
        self,
        df: pd.DataFrame,
        current_regime: int = 0,
        instrument: InstrumentType = InstrumentType.ETF_CASH,
        target_mode: str = "overnight"
    ) -> int:
        """
        Re-evaluates every stored champion on updated market data.
        Evicts outdated or decayed elites that no longer beat statutory friction or Calmar threshold.
        """
        pruned_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                elite = self.grid[r, c]
                if elite is None:
                    continue

                raw_signals = ASTEvaluator.eval_node(elite.formula, df, current_regime=current_regime)
                if np.all(raw_signals == 0.0):
                    self.grid[r, c] = None
                    pruned_count += 1
                    continue

                positions = np.where(raw_signals > 0, 1.0, 0.0)
                if np.sum(positions) == 0:
                    self.grid[r, c] = None
                    pruned_count += 1
                    continue

                close_p = df.get("Close_Adj", df.get("Close", df.get("Close_Raw"))).to_numpy(dtype=np.float64)
                open_p = df["Open"].to_numpy(dtype=np.float64)

                metrics = self._calculate_metrics(positions, open_p, close_p, instrument, target_mode)
                if (
                    metrics is None
                    or metrics["institutional_fitness"] <= 0.0
                    or metrics["total_net_ret"] <= 0.0
                    or metrics["win_rate"] < 45.0
                ):
                    self.grid[r, c] = None # Outdated or decayed: evicted from niche
                    pruned_count += 1
                else:
                    # Update metrics to fresh data
                    elite.fitness = round(metrics["institutional_fitness"], 2)
                    elite.win_rate = round(metrics["win_rate"], 2)
                    elite.net_return_pct = round(metrics["total_net_ret"] * 100.0, 2)
                    elite.cagr_pct = round(metrics.get("cagr_pct", 0.0), 2)
                    elite.sortino = round(metrics["sortino"], 2)
                    elite.profit_factor = round(min(metrics["profit_factor"], 99.0), 2)
                    elite.trades_count = metrics["n_trades"]
                    elite.volatility = round(metrics["annual_vol"], 4)
                    elite.turnover = round(metrics["turnover"], 2)

        if pruned_count > 0:
            logger.info(f"Temporal Decay Filter: Evicted {pruned_count} outdated elites from archive.")
        return pruned_count
