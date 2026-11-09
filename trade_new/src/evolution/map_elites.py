"""
Institutional MAP-Elites Behavioral Archive with 4D CVT Voronoi Tessellation.
Illuminates 1,000 continuous behavioral niches indexed by:
- b1: Annualized Realized Volatility ([0.05, 0.60])
- b2: Annualized Portfolio Turnover ([2.0, 150.0])
- b3: Downside Beta to Market ([-0.5, 1.5])
- b4: Signal Lag-1 Autocorrelation ([-0.5, 0.5])
Enforces statutory frictions, semi-deviation Sortino, and bloat deduplication.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from ..domain.formula_ast import ASTNode, ASTEvaluator
from ..domain.formula_parser import FormulaParser
from ..domain.models import InstrumentType, TaxSchedule
from .cvt_archive import CVTArchiveEngine


@dataclass
class EliteIndividual:
    """Encapsulates an elite trading strategy inhabiting a quality-diversity niche."""
    formula: ASTNode
    fitness: float
    volatility: float
    turnover: float
    downside_beta: float = 0.0
    autocorr_lag1: float = 0.0
    win_rate: float = 0.0
    net_return_pct: float = 0.0
    behavior_signature: np.ndarray = field(default_factory=lambda: np.zeros(60, dtype=np.float32))
    behavior_coords: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    cell_idx: int = 0
    niche_row: int = 0
    niche_col: int = 0
    sortino: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    max_drawdown: float = 0.0
    cagr_pct: float = 0.0
    lookback: str = "5y"

    @property
    def formula_str(self) -> str:
        return FormulaParser.to_string(self.formula)


class MapElitesArchive:
    """
    Quality-Diversity MAP-Elites Archive powered by 4D Centroidal Voronoi Tessellation.
    Supports sub-microsecond O(log K) nearest niche assignment via cKDTree.
    """
    def __init__(
        self,
        cells: int = 1000,
        dims: int = 4,
        tax_schedule: Optional[TaxSchedule] = None,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> None:
        self.cells = 400 if grid_size == (20, 20) else cells
        self.dims = dims
        self.tax_schedule = tax_schedule or TaxSchedule()
        self.cvt = CVTArchiveEngine(cells=self.cells, dims=self.dims)
        self.elites: Dict[int, EliteIndividual] = {}
        self.dirty_cells: set[int] = set()

    def _get_friction(self, instrument: InstrumentType) -> float:
        """Returns statutory friction rate for instrument."""
        if self.tax_schedule:
            return self.tax_schedule.get_friction_rate(instrument)
        if instrument == InstrumentType.ETF_CASH:
            return 0.00012
        if instrument == InstrumentType.FUTURES:
            return 0.00030
        return 0.00250

    def _compute_returns(
        self,
        positions: np.ndarray,
        open_p: np.ndarray,
        close_p: np.ndarray,
        instrument: InstrumentType,
        target_mode: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates gross and friction-adjusted net returns for long/short/continuous positions."""
        if target_mode == "overnight":
            gross = (open_p[1:] - close_p[:-1]) / (close_p[:-1] + 1e-8)
            trade_pos = positions[:-1]
        else:
            gross = (close_p - open_p) / (open_p + 1e-8)
            trade_pos = positions

        gross = np.clip(gross, -0.10, 0.10)
        pos_flips = np.abs(np.diff(trade_pos, prepend=0.0))
        friction = self._get_friction(instrument)
        net = (trade_pos * gross) - (pos_flips * friction)
        return net, trade_pos, pos_flips

    def _compute_downside_beta(
        self,
        net_returns: np.ndarray,
        open_p: np.ndarray,
        close_p: np.ndarray,
        target_mode: str
    ) -> float:
        """Calculates downside beta against underlying benchmark returns matching target mode."""
        if len(close_p) <= 2:
            return 0.0
        if target_mode == "overnight":
            bench_rets = (open_p[1:] - close_p[:-1]) / (close_p[:-1] + 1e-8)
        else:
            bench_rets = (close_p - open_p) / (open_p + 1e-8)
        min_len = min(len(net_returns), len(bench_rets))
        r_strat = net_returns[:min_len]
        r_bench = bench_rets[:min_len]

        down_mask = r_bench < 0.0
        if np.sum(down_mask) < 3:
            return 0.0
        d_strat = r_strat[down_mask]
        d_bench = r_bench[down_mask]
        var_bench = float(np.var(d_bench))
        if var_bench < 1e-8:
            return 0.0
        cov = float(np.cov(d_strat, d_bench)[0, 1])
        return float(np.clip(cov / var_bench, -0.5, 1.5))

    def _compute_lag1_autocorr(self, trade_positions: np.ndarray) -> float:
        """Calculates signal lag-1 autocorrelation for behavioral profiling."""
        if len(trade_positions) < 3 or np.std(trade_positions) < 1e-6:
            return 0.0
        rho1 = float(np.corrcoef(trade_positions[1:], trade_positions[:-1])[0, 1])
        return 0.0 if np.isnan(rho1) else float(np.clip(rho1, -0.5, 0.5))

    def _calculate_metrics(
        self,
        positions: np.ndarray,
        open_p: np.ndarray,
        close_p: np.ndarray,
        instrument: InstrumentType,
        target_mode: str
    ) -> Optional[Dict[str, Any]]:
        """Calculates institutional metrics, Sortino, Downside Beta, and Composite Alpha Score."""
        net, trade_pos, flips = self._compute_returns(positions, open_p, close_p, instrument, target_mode)
        active = net[np.abs(trade_pos) > 0.01]
        if len(active) < 5:
            return None

        ann_vol = float(np.std(net) * np.sqrt(252))
        turnover = float(np.mean(flips) * 252)
        downside_beta = self._compute_downside_beta(net, open_p, close_p, target_mode)
        autocorr_lag1 = self._compute_lag1_autocorr(trade_pos)

        cum = np.cumprod(1.0 + net)
        tot_net = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0
        years = max(0.08, len(net) / 252.0)
        cagr = float(((1.0 + tot_net) ** (1.0 / years)) - 1.0) * 100.0 if tot_net > -0.999 else -100.0
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / (peak + 1e-8)
        max_dd = float(np.max(dd) * 100.0) if len(dd) > 0 else 0.0
        win_rate = float(np.mean(active > 0) * 100.0)

        mean_ret = float(np.mean(active))
        down_dev = max(0.002, float(np.sqrt(np.mean(np.minimum(0.0, active) ** 2))))
        sortino = float((mean_ret / down_dev) * np.sqrt(252))

        gains = float(np.sum(active[active > 0])) if np.any(active > 0) else 0.0
        losses = max(0.001, abs(float(np.sum(active[active < 0]))))
        profit_factor = float(gains / losses)

        calmar = max(0.0, cagr) / max(4.0, max_dd)
        trade_conf = min(1.2, max(0.2, float(np.sqrt(len(active) / max(15.0, 5.0 * years)))))
        qual_mult = min(3.0, max(0.5, float(np.sqrt(max(0.25, min(sortino, 16.0))))))
        fitness = float(max(0.0, calmar * qual_mult * trade_conf))

        sig = trade_pos[-60:] if len(trade_pos) >= 60 else np.pad(trade_pos, (60 - len(trade_pos), 0))
        b_coords = np.array([ann_vol, turnover, downside_beta, autocorr_lag1], dtype=np.float32)

        return {
            "annual_vol": ann_vol, "turnover": turnover, "downside_beta": downside_beta,
            "autocorr_lag1": autocorr_lag1, "total_net_ret": tot_net, "cagr_pct": cagr,
            "max_drawdown": max_dd, "win_rate": win_rate, "mean_ret": mean_ret,
            "sortino": sortino, "profit_factor": profit_factor, "n_trades": len(active),
            "institutional_fitness": fitness, "sig": sig, "behavior_coords": b_coords
        }

    def _is_redundant_bloat(self, sig: np.ndarray, incumbent: EliteIndividual, fitness: float) -> bool:
        """Determines if challenger is duplicate behavioral bloat relative to incumbent champion."""
        std1 = float(np.std(sig))
        std2 = float(np.std(incumbent.behavior_signature))
        if std1 > 1e-6 and std2 > 1e-6:
            corr = np.corrcoef(sig, incumbent.behavior_signature)[0, 1]
            if not np.isnan(corr) and corr > 0.95 and fitness <= incumbent.fitness:
                return True
        elif np.array_equal(sig, incumbent.behavior_signature) and fitness <= incumbent.fitness:
            return True
        return False

    def evaluate_and_insert(
        self,
        formula: ASTNode,
        df: pd.DataFrame,
        current_regime: int = 0,
        instrument: InstrumentType = InstrumentType.ETF_CASH,
        target_mode: str = "overnight"
    ) -> bool:
        """Evaluates formula AST against historical data and inserts into 4D Voronoi niche if elite."""
        raw_signals = ASTEvaluator.eval_node(formula, df, current_regime=current_regime)
        if np.all(raw_signals == 0.0) or not np.all(np.isfinite(raw_signals)):
            return False

        # Support continuous and long/short signals in range [-1.0, 1.0]
        positions = np.clip(raw_signals, -1.0, 1.0)
        if np.sum(np.abs(positions)) == 0:
            return False

        c = df.get("Close_Adj", df.get("Close", df.get("Close_Raw"))).to_numpy(dtype=np.float64)
        o = df.get("Open", df.get("Close")).to_numpy(dtype=np.float64)
        metrics = self._calculate_metrics(positions, o, c, instrument, target_mode)
        if metrics is None or metrics["institutional_fitness"] <= 0.0 or metrics["mean_ret"] <= 0.0:
            return False

        b_coords = metrics["behavior_coords"]
        cell_idx = self.cvt.find_cell(b_coords)
        incumbent = self.elites.get(cell_idx)
        raw_fitness = metrics["institutional_fitness"]

        # ADHD-36: Parsimony pressure penalizing bloated trees
        formula_size = formula.count_nodes() if hasattr(formula, "count_nodes") else formula.size()
        complexity_penalty = 0.005 * max(0, formula_size - 5)
        fitness = max(0.01, raw_fitness - complexity_penalty)

        if incumbent is not None and self._is_redundant_bloat(metrics["sig"], incumbent, fitness):
            return False

        # If incumbent exists, require meaningful improvement (delta >= 0.02) if challenger is more complex
        if incumbent is not None:
            incumbent_size = incumbent.formula.count_nodes() if hasattr(incumbent.formula, "count_nodes") else incumbent.formula.size()
            if formula_size > incumbent_size and fitness < incumbent.fitness + 0.02:
                return False
            if fitness <= incumbent.fitness:
                return False

        row = cell_idx // 40
        col = cell_idx % 40
        elite = EliteIndividual(
            formula=formula, fitness=round(fitness, 2),
            volatility=round(metrics["annual_vol"], 4), turnover=round(metrics["turnover"], 2),
            downside_beta=round(metrics["downside_beta"], 4), autocorr_lag1=round(metrics["autocorr_lag1"], 4),
            win_rate=round(metrics["win_rate"], 2), net_return_pct=round(metrics["total_net_ret"] * 100.0, 2),
            behavior_signature=metrics["sig"].astype(np.float32) if isinstance(metrics["sig"], np.ndarray) else np.zeros(60, dtype=np.float32),
            behavior_coords=b_coords,
            cell_idx=cell_idx, niche_row=row, niche_col=col,
            sortino=round(metrics["sortino"], 2), profit_factor=round(min(metrics["profit_factor"], 99.0), 2),
            trades_count=metrics["n_trades"], max_drawdown=round(metrics["max_drawdown"], 2),
            cagr_pct=round(metrics["cagr_pct"], 2)
        )
        self.elites[cell_idx] = elite
        self.dirty_cells.add(cell_idx)
        self.cvt.insert(FormulaParser.to_string(formula), fitness, b_coords, metadata={"cell_idx": cell_idx})
        return True

    def insert(
        self,
        row: int,
        col: int,
        fitness: float,
        formula: ASTNode,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Inserts an elite directly into the archive (for warm-start or grid compatibility)."""
        m = metrics or {}
        cell_idx = row * 40 + col
        b_coords = np.array([
            float(m.get("volatility", 0.15)), float(m.get("turnover", 10.0)),
            float(m.get("downside_beta", 0.0)), float(m.get("autocorr_lag1", 0.0))
        ], dtype=np.float32)
        elite = EliteIndividual(
            formula=formula, fitness=round(fitness, 2),
            volatility=round(float(m.get("volatility", 0.0)), 4),
            turnover=round(float(m.get("turnover", 0.0)), 2),
            downside_beta=round(float(m.get("downside_beta", 0.0)), 4),
            autocorr_lag1=round(float(m.get("autocorr_lag1", 0.0)), 4),
            win_rate=round(float(m.get("win_rate", 0.0)), 2),
            net_return_pct=round(float(m.get("net_return_pct", 0.0)), 2),
            behavior_signature=np.zeros(60, dtype=np.float32), behavior_coords=b_coords,
            cell_idx=cell_idx, niche_row=row, niche_col=col,
            sortino=round(float(m.get("sortino", 0.0)), 2),
            profit_factor=round(min(float(m.get("profit_factor", 0.0)), 99.0), 2),
            trades_count=int(m.get("trades_count", 0)),
            max_drawdown=round(float(m.get("max_drawdown", 0.0)), 2),
            cagr_pct=round(float(m.get("cagr_pct", 0.0)), 2)
        )
        self.elites[cell_idx] = elite
        self.dirty_cells.add(cell_idx)
        return True

    def get_resident_elites(self) -> List[EliteIndividual]:
        """Returns list of all active resident elites in archive."""
        return list(self.elites.values())

    def get_top_elites(self, n: int = 10, metric: str = "fitness") -> List[Dict[str, Any]]:
        """Returns top elites formatted for UI or pipeline consumption."""
        sorted_elites = sorted(
            self.elites.values(),
            key=lambda e: getattr(e, metric, e.fitness),
            reverse=True
        )[:n]
        return [{
            "formula_str": FormulaParser.to_string(e.formula),
            "fitness": e.fitness, "cagr_pct": e.cagr_pct,
            "max_drawdown": e.max_drawdown, "sortino": e.sortino,
            "turnover": e.turnover, "volatility": e.volatility,
            "win_rate": e.win_rate, "net_return_pct": e.net_return_pct,
            "cell_idx": e.cell_idx, "row": e.niche_row, "col": e.niche_col
        } for e in sorted_elites]

    def occupancy_rate(self) -> float:
        """Returns archive niche occupancy fraction [0.0, 1.0]."""
        return len(self.elites) / max(1, self.cells)

    def to_archive_records(
        self, ticker: str, lookback: str = "1y", dirty_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Converts elites to dictionaries ready for SQLite WAL ArchiveStore batch upsert."""
        records = []
        source_elites = (
            [self.elites[idx] for idx in list(self.dirty_cells) if idx in self.elites]
            if dirty_only else list(self.elites.values())
        )
        for e in source_elites:
            records.append({
                "ticker": ticker, "lookback": lookback,
                "row": e.niche_row, "col": e.niche_col,
                "fitness": e.fitness, "net_return_pct": e.net_return_pct,
                "cagr_pct": e.cagr_pct, "max_drawdown": e.max_drawdown,
                "win_rate": e.win_rate, "turnover": e.turnover,
                "volatility": e.volatility, "sortino": e.sortino,
                "profit_factor": e.profit_factor, "trades_count": e.trades_count,
                "formula_str": FormulaParser.to_string(e.formula),
                "active_regimes": [0], "behavior_signature": e.behavior_signature.tobytes().hex()
            })
        if dirty_only:
            self.dirty_cells.clear()
        return records
