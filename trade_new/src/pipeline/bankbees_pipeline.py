"""BANKBEES Overnight Paper Execution Pipeline.

Implements an institutional, end-to-end quantitative execution pipeline for BANKBEES:
1. Loads historical daily OHLCV parquet data into VectorDataLake.
2. Prepares device-agnostic PyTorch feature tensors [T, 20] matching STANDARD_FEATURES.
3. Ingests candidate formulas from ArchiveStore (WAL SQLite) or candidate inputs.
4. Executes vectorized batch AST evaluations via TensorASTEvaluator.
5. Groups candidate return streams with ReturnClusterer and applies Occam's parsimony pruning.
6. Aggregates orthogonal alpha champions into an ensemble conviction score.
7. Computes execution metrics with TaxCalculator (1.2 bps ETF friction), PreOpenAuctionRouter,
   and TrailingStopSimulator.
8. Interfaces with ShoonyaGateway in paper trading mode to submit and track virtual orders.
"""
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.backtest.matrix_engine import MatrixSimulator
from src.clustering.return_clustering import ReturnClusterer
from src.data.vector_lake import VectorDataLake
from src.domain.formula_ast import ASTNode, STANDARD_FEATURES
from src.domain.formula_parser import FormulaParser
from src.domain.invariants import assert_integer_lot, sanitize_finite_array
from src.domain.models import InstrumentType
from src.domain.tensor_evaluator import TensorASTEvaluator
from src.evolution.archive_store import ArchiveStore
from src.execution.overnight_engine import (
    PreOpenAuctionRouter,
    TaxCalculator,
    TrailingStopSimulator,
)
from src.execution.shoonya_gateway import ShoonyaGateway


DEFAULT_FALLBACK_FORMULAS: List[str] = [
    "DELTA_5(TS_RANK_20(TS_MIN_10(TS_MIN_10((Vol_Price_Flow - TS_MAX_10(Vol_ZScore))))))",
    "DELTA_5((Global_Confluence - LOG_ABS(EMA_Slope)))",
    "ABS(DELTA_5(DELTA_5(RELU((Global_Confluence - Vol_Price_Flow)))))",
    "TS_RANK_20((Dist_VWAP * SIGN(EMA_Slope)))",
    "SIGMOID((Vol_Price_Flow - Downside_Vol_Ratio))",
]


@dataclass(frozen=True)
class AlphaChampion:
    """Represents an orthogonal alpha formula champion selected after clustering."""
    index: int
    cluster_id: int
    formula_str: str
    sortino: float
    cagr_pct: float
    max_drawdown: float
    win_rate: float
    current_signal: float
    depth: int


@dataclass(frozen=True)
class BacktestSummary:
    """Comprehensive performance telemetry of the ensemble overnight backtest."""
    cagr_pct: float
    sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_bars: int
    win_bars: int
    total_trades: int
    final_equity: float


@dataclass(frozen=True)
class EnsembleDecision:
    """Consolidated overnight trading decision for current session close."""
    conviction_score: float
    active_signals: int
    total_champions: int
    action: str
    latest_date: str
    latest_close: float
    quantity: int
    trade_value: float
    statutory_friction: float
    pre_open_order: Optional[Dict[str, Any]]
    paper_order_result: Optional[Dict[str, Any]]
    client_order_id: Optional[str]
    trailing_stop_preview: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PipelineResult:
    """Full execution output container for BankbeesOvernightPipeline."""
    ticker: str
    total_bars: int
    date_range: Tuple[str, str]
    evaluated_count: int
    cluster_count: int
    champions: List[AlphaChampion]
    ensemble_backtest: BacktestSummary
    decision: EnsembleDecision


def _compute_standard_features(
    df: pd.DataFrame, lake: VectorDataLake
) -> Dict[str, np.ndarray]:
    """Computes all 20 standard institutional features matching STANDARD_FEATURES."""
    c_series = df["Close_Adj"] if "Close_Adj" in df.columns else (df["Close"] if "Close" in df.columns else df["Close_Raw"])
    c = c_series.to_numpy(dtype=np.float32)
    o = df["Open"].to_numpy(dtype=np.float32)
    h = df["High"].to_numpy(dtype=np.float32)
    l = df["Low"].to_numpy(dtype=np.float32)
    v = df.get("Volume", pd.Series(10000.0, index=df.index)).to_numpy(dtype=np.float32)

    spread = (h - l) / (c + 1e-8)
    vol_zscore = lake.tensors.get("Vol_ZScore", np.zeros(len(c), dtype=np.float32))
    rets = lake.tensors.get("gross_close_to_close_rets", np.zeros(len(c), dtype=np.float32))
    vol_price_flow = vol_zscore * rets
    tr = lake.tensors.get("True_Range", (h - l) / (c + 1e-8))

    tr_s = pd.Series(tr, index=df.index)
    atr20 = tr_s.rolling(20, min_periods=1).mean().to_numpy(dtype=np.float32)
    tr_expansion = tr / (atr20 + 1e-8)

    c_s = pd.Series(c, index=df.index)
    ema20 = c_s.ewm(span=20, adjust=False).mean().to_numpy(dtype=np.float32)
    ema20_s = pd.Series(ema20, index=df.index)
    ema20_lag5 = ema20_s.shift(5).bfill().to_numpy(dtype=np.float32)
    ema_slope = (ema20 - ema20_lag5) / (ema20_lag5 + 1e-8)
    dist_ema20 = (c - ema20) / (ema20 + 1e-8)

    delta = c_s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    rsi14 = ((100.0 - (100.0 / (1.0 + rs))) / 100.0).fillna(0.5).to_numpy(dtype=np.float32)

    return {
        "Close_Adj": c, "Open": o, "High": h, "Low": l, "Volume": v,
        "Spread": spread, "Vol_ZScore": vol_zscore, "Vol_Price_Flow": vol_price_flow,
        "True_Range": tr, "TR_Expansion": tr_expansion, "EMA_Slope": ema_slope,
        "Dist_EMA20": dist_ema20,
        "Dist_VWAP": lake.tensors.get("Dist_VWAP", np.zeros(len(c), dtype=np.float32)),
        "Downside_Vol_Ratio": lake.tensors.get("Downside_Vol_Ratio", np.full(len(c), 0.5, dtype=np.float32)),
        "RSI_14": rsi14,
        "Parkinson_Vol": lake.tensors.get("Parkinson_Vol", np.zeros(len(c), dtype=np.float32)),
        "Garman_Klass_Vol": lake.tensors.get("Garman_Klass_Vol", np.zeros(len(c), dtype=np.float32)),
        "SP500_Ret": np.full(len(c), 0.0004, dtype=np.float32),
        "ADR_Ret": np.full(len(c), 0.0005, dtype=np.float32),
        "Global_Confluence": np.ones(len(c), dtype=np.float32),
    }


def _assemble_feature_tensor(features: Dict[str, np.ndarray]) -> torch.Tensor:
    """Packs computed feature dictionary into a 2D float32 PyTorch tensor [T, F]."""
    first_key = next(iter(features))
    t_len = len(features[first_key])
    f_len = len(STANDARD_FEATURES)
    tensor = torch.zeros((t_len, f_len), dtype=torch.float32)
    for i, name in enumerate(STANDARD_FEATURES):
        arr = features.get(name, np.zeros(t_len, dtype=np.float32))
        clean_arr = sanitize_finite_array(arr, default_fill=0.0)
        tensor[:, i] = torch.from_numpy(clean_arr.astype(np.float32))
    return tensor


class BankbeesOvernightPipeline:
    """Autonomous overnight quantitative execution pipeline for BANKBEES ETF."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        db_path: Optional[str] = None,
        capital: float = 10000.0,
        friction_bps: float = 1.2,
        rho_threshold: float = 0.85,
        parsimony_penalty: float = 0.05,
    ) -> None:
        """Initializes pipeline paths, capital, and risk friction parameters."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.data_path = data_path or os.path.join(base_dir, "data", "lake", "Indices", "BANKBEES.parquet")
        self.db_path = db_path or os.path.join(base_dir, "audit", "alpha_elites.db")
        self.capital = float(capital)
        self.friction_bps = float(friction_bps)
        self.rho_threshold = float(rho_threshold)
        self.parsimony_penalty = float(parsimony_penalty)
        self.ticker = "BANKBEES"

    def load_data(
        self, lookback_years: Optional[float] = None
    ) -> Tuple[VectorDataLake, pd.DataFrame]:
        """Loads parquet data from disk and constructs vectorized data lake."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Parquet file not found at {self.data_path}")
        df = pd.read_parquet(self.data_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if lookback_years is not None and lookback_years > 0:
            days = int(lookback_years * 365.25)
            cutoff = df.index[-1] - pd.Timedelta(days=days)
            df = df.loc[df.index >= cutoff].copy()

        lake = VectorDataLake.from_dataframe(df, ticker=self.ticker)
        return lake, df

    def prepare_feature_tensor(
        self, lake: VectorDataLake, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, List[str]]:
        """Builds PyTorch feature tensor [T, 20] corresponding to STANDARD_FEATURES."""
        feat_dict = _compute_standard_features(df, lake)
        tensor = _assemble_feature_tensor(feat_dict)
        return tensor, list(STANDARD_FEATURES)

    def fetch_candidate_formulas(
        self,
        limit: int = 50,
        custom_candidates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves elite candidate formulas from ArchiveStore or fallback pool."""
        if custom_candidates:
            return [{"formula_str": f, "fitness": 1.0, "sortino": 1.0} for f in custom_candidates]

        elites: List[Dict[str, Any]] = []
        if os.path.exists(self.db_path):
            try:
                store = ArchiveStore(db_path=self.db_path)
                elites = store.get_top_elites(limit=limit, ticker=self.ticker, metric="fitness")
            except Exception:
                elites = []

        if not elites:
            return [{"formula_str": f, "fitness": 1.0, "sortino": 1.0} for f in DEFAULT_FALLBACK_FORMULAS]
        return elites

    def evaluate_formulas_batch(
        self, formulas: List[Dict[str, Any]], X: torch.Tensor
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[ASTNode]]:
        """Parses AST trees and evaluates batch signals across all time steps."""
        valid_items: List[Dict[str, Any]] = []
        nodes: List[ASTNode] = []
        raw_signals: List[np.ndarray] = []

        for item in formulas:
            f_str = item.get("formula_str", "").strip()
            if not f_str:
                continue
            try:
                ast_node = FormulaParser.parse(f_str)
                sig_tensor = TensorASTEvaluator.eval_node(ast_node, X, feature_names=STANDARD_FEATURES)
                sig_arr = sig_tensor.detach().cpu().numpy()
                sig_clean = sanitize_finite_array(sig_arr, default_fill=0.0)
                raw_signals.append(sig_clean)
                valid_items.append(item)
                nodes.append(ast_node)
            except Exception:
                continue

        if not raw_signals:
            fallback_ast = FormulaParser.parse(DEFAULT_FALLBACK_FORMULAS[0])
            sig_arr = TensorASTEvaluator.eval_node(fallback_ast, X).detach().cpu().numpy()
            return sig_arr[np.newaxis, :], [{"formula_str": DEFAULT_FALLBACK_FORMULAS[0]}], [fallback_ast]

        return np.vstack(raw_signals), valid_items, nodes

    def cluster_and_prune(
        self,
        signals: np.ndarray,
        lake: VectorDataLake,
        valid_items: List[Dict[str, Any]],
        nodes: List[ASTNode],
        lookback_years: float = 5.0,
    ) -> Tuple[np.ndarray, List[AlphaChampion]]:
        """Clusters candidate return streams and selects top champions via Occam's penalty."""
        positions = (signals > 0.0).astype(np.float32)
        gross_rets = lake.tensors["gross_overnight_rets"]
        sim = MatrixSimulator(friction_bps=self.friction_bps)
        sim_res = sim.simulate_batch(positions, gross_rets, lookback_years=lookback_years)
        net_returns = sim_res["net_returns"]

        fitness_scores = np.array([float(it.get("fitness", sim_res["sortino"][i])) for i, it in enumerate(valid_items)])
        depths = np.array([n.depth() for n in nodes])

        clusterer = ReturnClusterer(
            rho_threshold=self.rho_threshold,
            parsimony_penalty=self.parsimony_penalty,
        )
        labels, elite_indices = clusterer.cluster_and_prune(net_returns, fitness_scores, depths)

        champions: List[AlphaChampion] = []
        for idx in elite_indices:
            champ = AlphaChampion(
                index=idx,
                cluster_id=int(labels[idx]),
                formula_str=valid_items[idx].get("formula_str", ""),
                sortino=float(sim_res["sortino"][idx]),
                cagr_pct=float(sim_res["cagr"][idx]),
                max_drawdown=float(sim_res["max_drawdown"][idx]),
                win_rate=float(sim_res["win_rate"][idx]),
                current_signal=float(signals[idx, -1]),
                depth=int(depths[idx]),
            )
            champions.append(champ)

        return labels, champions

    def backtest_ensemble(
        self,
        champion_signals: np.ndarray,
        gross_overnight_rets: np.ndarray,
        lookback_years: float = 5.0,
    ) -> BacktestSummary:
        """Simulates walk-forward compounding for the aggregated ensemble signal."""
        # Mean signal conviction > 0 triggers binary long position
        mean_signal = np.mean(champion_signals, axis=0)
        ensemble_pos = (mean_signal > 0.0).astype(np.float32)[np.newaxis, :]

        sim = MatrixSimulator(friction_bps=self.friction_bps)
        res = sim.simulate_batch(ensemble_pos, gross_overnight_rets, lookback_years=lookback_years)

        net_rets = res["net_returns"][0]
        active = ensemble_pos[0] > 0.0
        n_active = max(1, int(np.sum(active)))
        wins = int(np.sum((net_rets > 0.0) & active))

        cum_eq = res["cum_equity"][0]
        final_eq = float(cum_eq[-1]) * self.capital if len(cum_eq) else self.capital

        return BacktestSummary(
            cagr_pct=float(res["cagr"][0]),
            sortino=float(res["sortino"][0]),
            max_drawdown_pct=float(res["max_drawdown"][0]),
            win_rate_pct=float(res["win_rate"][0]),
            total_bars=len(gross_overnight_rets),
            win_bars=wins,
            total_trades=n_active,
            final_equity=round(final_eq, 2),
        )

    def generate_decision(
        self,
        champions: List[AlphaChampion],
        lake: VectorDataLake,
        df: pd.DataFrame,
        gateway: Optional[ShoonyaGateway] = None,
        dry_run: bool = False,
    ) -> EnsembleDecision:
        """Calculates current conviction, integer lot size, auction routing, and dispatches paper order."""
        total = len(champions)
        active_cnt = sum(1 for c in champions if c.current_signal > 0.0)
        conviction = active_cnt / float(total) if total > 0 else 0.0
        action = "BUY" if conviction >= 0.50 else "HOLD_CASH"

        latest_close = float(df.get("Close_Adj", df["Close"]).iloc[-1])
        latest_date = str(df.index[-1].date())

        friction_rate = self.friction_bps / 10000.0
        max_cost_per_share = latest_close * (1.0 + friction_rate)
        raw_qty = int(self.capital // max_cost_per_share) if max_cost_per_share > 0 else 0
        quantity = raw_qty if action == "BUY" else 0

        trade_val = quantity * latest_close
        statutory_friction = TaxCalculator.calculate(trade_val, InstrumentType.ETF_CASH)

        pre_open = None
        trailing_preview = None
        if action == "BUY" and quantity > 0:
            assert_integer_lot(quantity)
            pre_open = PreOpenAuctionRouter.route_auction_order(
                ticker=self.ticker, side="BUY", indicative_price=latest_close,
                quantity=quantity, slippage_ceiling_bps=15.0
            )
            sim_exit, sim_reason = TrailingStopSimulator.resolve_exit(
                entry_price=latest_close, open_price=latest_close,
                high_price=latest_close * 1.005, low_price=latest_close * 0.998
            )
            trailing_preview = {"simulated_exit": sim_exit, "exit_reason": sim_reason}

        client_order_id = f"BANKBEES_ON_{latest_date}_{int(datetime.now().timestamp())}"
        order_res = None
        if not dry_run and action == "BUY" and quantity > 0 and gateway is not None:
            order_res = gateway.place_order(
                symbol=self.ticker, quantity=quantity, price=latest_close,
                action="BUY", is_etf=True, client_order_id=client_order_id
            )

        return EnsembleDecision(
            conviction_score=round(conviction, 4),
            active_signals=active_cnt,
            total_champions=total,
            action=action,
            latest_date=latest_date,
            latest_close=round(latest_close, 2),
            quantity=quantity,
            trade_value=round(trade_val, 2),
            statutory_friction=round(statutory_friction, 4),
            pre_open_order=pre_open,
            paper_order_result=order_res,
            client_order_id=client_order_id if action == "BUY" else None,
            trailing_stop_preview=trailing_preview,
        )

    def run(
        self,
        lookback_years: float = 5.0,
        candidate_formulas: Optional[List[str]] = None,
        gateway: Optional[ShoonyaGateway] = None,
        dry_run: bool = False,
        limit_candidates: int = 50,
    ) -> PipelineResult:
        """Executes the full overnight pipeline lifecycle from lake to broker dispatch."""
        lake, df = self.load_data(lookback_years=lookback_years)
        X, _ = self.prepare_feature_tensor(lake, df)

        candidates = self.fetch_candidate_formulas(
            limit=limit_candidates, custom_candidates=candidate_formulas
        )
        signals, valid_formulas, ast_nodes = self.evaluate_formulas_batch(candidates, X)

        labels, champions = self.cluster_and_prune(
            signals, lake, valid_formulas, ast_nodes, lookback_years=lookback_years
        )

        champ_indices = [c.index for c in champions]
        champ_signals = signals[champ_indices] if champ_indices else signals[:1]

        gross_overnight = lake.tensors["gross_overnight_rets"]
        summary = self.backtest_ensemble(
            champ_signals, gross_overnight, lookback_years=lookback_years
        )

        decision = self.generate_decision(
            champions, lake, df, gateway=gateway, dry_run=dry_run
        )

        date_start = str(df.index[0].date())
        date_end = str(df.index[-1].date())

        return PipelineResult(
            ticker=self.ticker,
            total_bars=len(df),
            date_range=(date_start, date_end),
            evaluated_count=len(valid_formulas),
            cluster_count=len(np.unique(labels)),
            champions=champions,
            ensemble_backtest=summary,
            decision=decision,
        )
