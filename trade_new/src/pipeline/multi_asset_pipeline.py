"""Multi-Asset Overnight Quantitative Portfolio Engine.

Provides an institutional, end-to-end multi-asset portfolio execution pipeline:
1. Dynamic instrument classification (ETF_CASH at 1.2 bps vs FUTURES at 3.0 bps).
2. Data lake ingestion and standard feature preparation via VectorDataLake.
3. Alpha signal generation via TensorASTEvaluator and return-space clustering.
4. Active asset filtering by conviction score (default threshold: 0.40).
5. Risk-parity capital allocation across active assets via HRPAllocator with Ledoit-Wolf shrinkage.
6. Single-asset exposure cap enforcement (max 30%, sum <= 1.0).
7. Vectorized portfolio backtesting via MatrixSimulator with exact instrument frictions.
8. Integer-lot order dispatch and audit journaling to ShoonyaGateway.
"""
from dataclasses import dataclass
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
from src.execution.hrp_allocator import HRPAllocator
from src.execution.overnight_engine import (
    PreOpenAuctionRouter,
    TaxCalculator,
    TrailingStopSimulator,
)
from src.execution.shoonya_gateway import ShoonyaGateway
from src.pipeline.bankbees_pipeline import (
    AlphaChampion,
    _assemble_feature_tensor,
    _compute_standard_features,
    DEFAULT_FALLBACK_FORMULAS,
)

CORE_MULTI_ASSET_BASKET: List[str] = [
    "BANKBEES",
    "NIFTYBEES",
    "GOLDBEES",
    "JUNIORBEES",
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
]


def classify_instrument(ticker: str) -> Tuple[InstrumentType, float]:
    """
    Dynamically classifies instrument as ETF_CASH or FUTURES with statutory friction.

    Args:
        ticker: Symbol string (e.g., 'BANKBEES', 'RELIANCE').

    Returns:
        Tuple of (InstrumentType, friction_in_bps).
    """
    clean = ticker.upper().replace(".NS", "").replace("-EQ", "")
    if "BEES" in clean:
        return InstrumentType.ETF_CASH, 1.2
    return InstrumentType.FUTURES, 3.0


def apply_exposure_limits(weights: np.ndarray, max_weight: float = 0.30) -> np.ndarray:
    """
    Caps single-asset allocation to max_weight and ensures portfolio sum <= 1.0.

    Args:
        weights: 1D array of unconstrained weights.
        max_weight: Upper bound fraction per asset (default: 0.30).

    Returns:
        1D array of constrained weights.
    """
    if len(weights) == 0:
        return weights
    w = np.clip(weights, 0.0, 1.0)
    total_w = np.sum(w)
    if total_w <= 0:
        return w
    w = w / total_w
    if np.all(w <= max_weight + 1e-6):
        return w
    if len(w) * max_weight <= 1.0 + 1e-6:
        return np.full(len(w), min(max_weight, 1.0 / len(w)), dtype=np.float64)

    capped = np.zeros(len(w), dtype=bool)
    for _ in range(len(w)):
        exceeded = (w > max_weight + 1e-6) & (~capped)
        if not np.any(exceeded):
            break
        capped = capped | exceeded
        w[capped] = max_weight
        uncapped = ~capped
        if not np.any(uncapped):
            break
        rem = 1.0 - np.sum(w[capped])
        if rem <= 0:
            w[uncapped] = 0.0
            break
        uncapped_sum = np.sum(w[uncapped])
        if uncapped_sum > 0:
            w[uncapped] = (w[uncapped] / uncapped_sum) * rem

    w = np.minimum(w, max_weight)
    if np.sum(w) > 1.0:
        w = w / np.sum(w)
    return w


@dataclass(frozen=True)
class AssetEvaluation:
    """Consolidated signal evaluation and execution decision for an individual asset."""
    ticker: str
    instrument_type: InstrumentType
    statutory_friction_bps: float
    conviction_score: float
    active_signals: int
    total_champions: int
    is_bullish: bool
    latest_close: float
    target_weight: float
    allocated_capital: float
    quantity: int
    statutory_friction_inr: float
    action: str
    client_order_id: Optional[str] = None
    pre_open_order: Optional[Dict[str, Any]] = None
    paper_order_result: Optional[Dict[str, Any]] = None
    trailing_stop_preview: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PortfolioBacktestSummary:
    """Historical performance metrics of the combined multi-asset portfolio."""
    cagr_pct: float
    sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_bars: int
    win_bars: int
    total_trades: int
    final_equity: float
    date_range: Tuple[str, str]


@dataclass(frozen=True)
class MultiAssetPipelineResult:
    """End-to-end execution container for MultiAssetOvernightPipeline."""
    evaluated_assets: List[str]
    active_assets: List[str]
    evaluations: Dict[str, AssetEvaluation]
    target_weights: Dict[str, float]
    portfolio_backtest: PortfolioBacktestSummary
    total_capital: float
    total_allocated_capital: float
    total_statutory_friction: float
    cash_reserve: float
    latest_date: str


class MultiAssetOvernightPipeline:
    """Institutional Multi-Asset Overnight Quantitative Portfolio Engine."""

    def __init__(
        self,
        lake_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        capital: float = 100000.0,
        conviction_threshold: float = 0.40,
        max_asset_weight: float = 0.30,
        rho_threshold: float = 0.85,
        parsimony_penalty: float = 0.05,
    ) -> None:
        """Initializes multi-asset pipeline directories, capital, and risk parameters."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.lake_dir = lake_dir or os.path.join(base_dir, "data", "lake")
        self.db_path = db_path or os.path.join(base_dir, "audit", "alpha_elites.db")
        self.capital = float(capital)
        self.conviction_threshold = float(conviction_threshold)
        self.max_asset_weight = float(max_asset_weight)
        self.rho_threshold = float(rho_threshold)
        self.parsimony_penalty = float(parsimony_penalty)
        self.lake_catalog: Dict[str, str] = self._build_lake_catalog()

    def _build_lake_catalog(self) -> Dict[str, str]:
        """Indexes all parquet files in the data lake by clean ticker symbol."""
        catalog: Dict[str, str] = {}
        if not os.path.exists(self.lake_dir):
            return catalog
        for root, _, files in os.walk(self.lake_dir):
            for f in files:
                if f.endswith(".parquet"):
                    key = f[:-8].upper().replace(".NS", "").replace("-EQ", "")
                    catalog[key] = os.path.join(root, f)
        return catalog

    def resolve_parquet_path(self, ticker: str) -> str:
        """Resolves the disk path of an asset's parquet file from the lake catalog."""
        clean = ticker.upper().replace(".NS", "").replace("-EQ", "")
        if clean in self.lake_catalog:
            return self.lake_catalog[clean]
        raise FileNotFoundError(f"Parquet table for ticker '{ticker}' not found in {self.lake_dir}")

    def load_asset_data(
        self, ticker: str, lookback_years: Optional[float] = None
    ) -> Tuple[VectorDataLake, pd.DataFrame]:
        """Loads parquet data from lake and creates a VectorDataLake instance."""
        p_path = self.resolve_parquet_path(ticker)
        df = pd.read_parquet(p_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        if lookback_years is not None and lookback_years > 0:
            days = int(lookback_years * 365.25)
            cutoff = df.index[-1] - pd.Timedelta(days=days)
            df = df.loc[df.index >= cutoff].copy()

        clean_ticker = ticker.upper().replace(".NS", "")
        lake = VectorDataLake.from_dataframe(df, ticker=clean_ticker)
        return lake, df

    def _prepare_feature_tensor(
        self, lake: VectorDataLake, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, List[str]]:
        """Computes and packs standard 20 institutional features into PyTorch tensor [T, F]."""
        feat_dict = _compute_standard_features(df, lake)
        tensor = _assemble_feature_tensor(feat_dict)
        return tensor, list(STANDARD_FEATURES)

    def _fetch_candidate_formulas(
        self,
        ticker: str,
        limit: int = 50,
        custom_candidates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves candidate formulas from SQLite archive or default fallback pool."""
        if custom_candidates:
            return [{"formula_str": f, "fitness": 1.0, "sortino": 1.0} for f in custom_candidates]

        clean_ticker = ticker.upper().replace(".NS", "")
        elites: List[Dict[str, Any]] = []
        if os.path.exists(self.db_path):
            try:
                store = ArchiveStore(db_path=self.db_path)
                elites = store.get_top_elites(limit=limit, ticker=clean_ticker, metric="fitness")
                if not elites:
                    elites = store.get_top_elites(limit=limit, metric="fitness")
            except Exception:
                elites = []

        if not elites:
            return [{"formula_str": f, "fitness": 1.0, "sortino": 1.0} for f in DEFAULT_FALLBACK_FORMULAS]
        return elites

    def _evaluate_formulas_batch(
        self, formulas: List[Dict[str, Any]], X: torch.Tensor
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[ASTNode]]:
        """Evaluates batch AST trees via TensorASTEvaluator across market sessions."""
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
            fallback = FormulaParser.parse(DEFAULT_FALLBACK_FORMULAS[0])
            sig_arr = TensorASTEvaluator.eval_node(fallback, X).detach().cpu().numpy()
            return sig_arr[np.newaxis, :], [{"formula_str": DEFAULT_FALLBACK_FORMULAS[0]}], [fallback]

        return np.vstack(raw_signals), valid_items, nodes

    def _cluster_and_select_champions(
        self,
        signals: np.ndarray,
        lake: VectorDataLake,
        valid_items: List[Dict[str, Any]],
        nodes: List[ASTNode],
        friction_bps: float,
        lookback_years: float = 5.0,
    ) -> Tuple[np.ndarray, List[AlphaChampion]]:
        """Clusters candidate return streams and selects parsimonious champions."""
        positions = (signals > 0.0).astype(np.float32)
        gross_rets = lake.tensors["gross_overnight_rets"]
        sim = MatrixSimulator(friction_bps=friction_bps)
        sim_res = sim.simulate_batch(positions, gross_rets, lookback_years=lookback_years)
        net_returns = sim_res["net_returns"]

        fitness = np.array([float(it.get("fitness", sim_res["sortino"][i])) for i, it in enumerate(valid_items)])
        depths = np.array([n.depth() for n in nodes])

        clusterer = ReturnClusterer(
            rho_threshold=self.rho_threshold,
            parsimony_penalty=self.parsimony_penalty,
        )
        labels, elite_indices = clusterer.cluster_and_prune(net_returns, fitness, depths)

        champions: List[AlphaChampion] = []
        for idx in elite_indices:
            champions.append(
                AlphaChampion(
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
            )
        return labels, champions

    def evaluate_asset(
        self,
        ticker: str,
        lookback_years: float = 5.0,
        candidate_formulas: Optional[List[str]] = None,
        limit_candidates: int = 50,
    ) -> Tuple[VectorDataLake, pd.DataFrame, float, int, int, np.ndarray, List[AlphaChampion]]:
        """Loads data, evaluates AST formulas, and computes conviction score for a single asset."""
        lake, df = self.load_asset_data(ticker, lookback_years=lookback_years)
        X, _ = self._prepare_feature_tensor(lake, df)
        _, friction_bps = classify_instrument(ticker)

        formulas = self._fetch_candidate_formulas(
            ticker, limit=limit_candidates, custom_candidates=candidate_formulas
        )
        signals, valid_items, nodes = self._evaluate_formulas_batch(formulas, X)
        _, champions = self._cluster_and_select_champions(
            signals, lake, valid_items, nodes, friction_bps=friction_bps, lookback_years=lookback_years
        )

        active_cnt = sum(1 for c in champions if c.current_signal > 0.0)
        total_champs = len(champions)
        conviction = active_cnt / float(total_champs) if total_champs > 0 else 0.0

        champ_indices = [c.index for c in champions]
        champ_signals = signals[champ_indices] if champ_indices else signals[:1]
        ensemble_signal = np.mean(champ_signals, axis=0)

        return lake, df, conviction, active_cnt, total_champs, ensemble_signal, champions

    def compute_risk_parity_weights(
        self, returns_matrix: np.ndarray, active_tickers: List[str]
    ) -> Dict[str, float]:
        """Calculates Ledoit-Wolf risk-parity weights across active assets with exposure caps."""
        if len(active_tickers) == 0:
            return {}
        if len(active_tickers) == 1:
            capped_w = apply_exposure_limits(np.array([1.0]), max_weight=self.max_asset_weight)
            return {active_tickers[0]: float(capped_w[0])}

        raw_weights = HRPAllocator.compute_weights(returns_matrix, use_shrinkage=True)
        constrained_weights = apply_exposure_limits(raw_weights, max_weight=self.max_asset_weight)
        return {ticker: float(w) for ticker, w in zip(active_tickers, constrained_weights)}

    def simulate_portfolio(
        self,
        aligned_returns: np.ndarray,
        aligned_positions: np.ndarray,
        friction_bps_list: List[float],
        date_range: Tuple[str, str],
        lookback_years: float = 5.0,
    ) -> PortfolioBacktestSummary:
        """Executes multi-asset vectorized portfolio backtest via MatrixSimulator."""
        if aligned_returns.size == 0 or aligned_positions.size == 0:
            return PortfolioBacktestSummary(
                cagr_pct=0.0, sortino=0.0, max_drawdown_pct=0.0, win_rate_pct=0.0,
                total_bars=0, win_bars=0, total_trades=0, final_equity=self.capital,
                date_range=date_range,
            )

        sim = MatrixSimulator(friction_bps=friction_bps_list)
        res = sim.simulate_batch(aligned_positions, aligned_returns, lookback_years=lookback_years)

        net_rets = res["net_returns"][0]
        active = np.sum(np.abs(aligned_positions[0]), axis=0) > 0.0
        n_active = max(1, int(np.sum(active)))
        wins = int(np.sum((net_rets > 0.0) & active))
        cum_eq = res["cum_equity"][0]
        final_eq = float(cum_eq[-1]) * self.capital if len(cum_eq) else self.capital

        return PortfolioBacktestSummary(
            cagr_pct=float(res["cagr"][0]),
            sortino=float(res["sortino"][0]),
            max_drawdown_pct=float(res["max_drawdown"][0]),
            win_rate_pct=float(res["win_rate"][0]),
            total_bars=len(net_rets),
            win_bars=wins,
            total_trades=n_active,
            final_equity=round(final_eq, 2),
            date_range=date_range,
        )

    def _dispatch_paper_orders(
        self,
        evaluations: Dict[str, AssetEvaluation],
        gateway: Optional[ShoonyaGateway] = None,
        dry_run: bool = False,
    ) -> Dict[str, AssetEvaluation]:
        """Submits integer lot orders to ShoonyaGateway paper journal with client order IDs."""
        updated: Dict[str, AssetEvaluation] = {}
        for ticker, ev in evaluations.items():
            if ev.action != "BUY" or ev.quantity <= 0:
                updated[ticker] = ev
                continue

            order_res = None
            if not dry_run and gateway is not None:
                is_etf = (ev.instrument_type == InstrumentType.ETF_CASH)
                order_res = gateway.place_order(
                    symbol=ticker,
                    quantity=ev.quantity,
                    price=ev.latest_close,
                    action="BUY",
                    is_etf=is_etf,
                    custom_bps=ev.statutory_friction_bps,
                    client_order_id=ev.client_order_id,
                )

            updated[ticker] = AssetEvaluation(
                ticker=ev.ticker,
                instrument_type=ev.instrument_type,
                statutory_friction_bps=ev.statutory_friction_bps,
                conviction_score=ev.conviction_score,
                active_signals=ev.active_signals,
                total_champions=ev.total_champions,
                is_bullish=ev.is_bullish,
                latest_close=ev.latest_close,
                target_weight=ev.target_weight,
                allocated_capital=ev.allocated_capital,
                quantity=ev.quantity,
                statutory_friction_inr=ev.statutory_friction_inr,
                action=ev.action,
                client_order_id=ev.client_order_id,
                pre_open_order=ev.pre_open_order,
                paper_order_result=order_res,
                trailing_stop_preview=ev.trailing_stop_preview,
            )
        return updated

    def _evaluate_all_assets(
        self,
        target_assets: List[str],
        lookback_years: float,
        candidate_formulas: Optional[List[str]],
        limit_candidates: int,
    ) -> Tuple[Dict[str, Tuple[VectorDataLake, pd.DataFrame, np.ndarray]], Dict[str, Tuple[float, int, int]]]:
        """Evaluates conviction and signals across all basket assets."""
        loaded_data: Dict[str, Tuple[VectorDataLake, pd.DataFrame, np.ndarray]] = {}
        convictions: Dict[str, Tuple[float, int, int]] = {}
        for ticker in target_assets:
            lake, df, conv, act_cnt, tot_cnt, ens_sig, _ = self.evaluate_asset(
                ticker,
                lookback_years=lookback_years,
                candidate_formulas=candidate_formulas,
                limit_candidates=limit_candidates,
            )
            loaded_data[ticker] = (lake, df, ens_sig)
            convictions[ticker] = (conv, act_cnt, tot_cnt)
        return loaded_data, convictions

    def _align_active_series(
        self,
        active_tickers: List[str],
        loaded_data: Dict[str, Tuple[VectorDataLake, pd.DataFrame, np.ndarray]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aligns historical returns and signals of active assets on common dates."""
        ret_series: Dict[str, pd.Series] = {}
        sig_series: Dict[str, pd.Series] = {}
        for t in active_tickers:
            lake, df, sig = loaded_data[t]
            idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
            ret_series[t] = pd.Series(lake.tensors["gross_overnight_rets"], index=idx)
            sig_series[t] = pd.Series(sig, index=idx)

        aligned_rets = pd.DataFrame(ret_series).dropna() if ret_series else pd.DataFrame()
        aligned_sigs = pd.DataFrame(sig_series).reindex(aligned_rets.index).fillna(0.0) if ret_series else pd.DataFrame()
        return aligned_rets, aligned_sigs

    def _run_portfolio_backtest(
        self,
        active_tickers: List[str],
        target_weights: Dict[str, float],
        aligned_rets: pd.DataFrame,
        aligned_sigs: pd.DataFrame,
        lookback_years: float,
    ) -> Tuple[PortfolioBacktestSummary, str]:
        """Constructs 3D position tensor and executes MatrixSimulator portfolio backtest."""
        if not active_tickers or aligned_rets.empty:
            d_now = str(datetime.now().date())
            empty_bt = PortfolioBacktestSummary(
                0.0, 0.0, 0.0, 0.0, 0, 0, 0, self.capital, (d_now, d_now)
            )
            return empty_bt, d_now

        k = len(active_tickers)
        t_bars = len(aligned_rets)
        pos_3d = np.zeros((1, k, t_bars), dtype=np.float32)
        for i, t in enumerate(active_tickers):
            w = target_weights.get(t, 0.0)
            pos_3d[0, i, :] = w * (aligned_sigs[t].to_numpy() > 0.0)

        frictions = [classify_instrument(t)[1] for t in active_tickers]
        date_range = (str(aligned_rets.index[0].date()), str(aligned_rets.index[-1].date()))
        rets_arr = aligned_rets.to_numpy(dtype=np.float32).T
        bt = self.simulate_portfolio(rets_arr, pos_3d, frictions, date_range, lookback_years)
        return bt, str(aligned_rets.index[-1].date())

    def _build_evaluations(
        self,
        target_assets: List[str],
        loaded_data: Dict[str, Tuple[VectorDataLake, pd.DataFrame, np.ndarray]],
        asset_convictions: Dict[str, Tuple[float, int, int]],
        target_weights: Dict[str, float],
        latest_date: str,
    ) -> Tuple[Dict[str, AssetEvaluation], float, float]:
        """Calculates integer lots, friction taxes, and order parameters for each asset."""
        evals: Dict[str, AssetEvaluation] = {}
        tot_alloc, tot_tax = 0.0, 0.0
        for ticker in target_assets:
            lake, df, _ = loaded_data[ticker]
            conv, act_cnt, tot_cnt = asset_convictions[ticker]
            inst_type, fric_bps = classify_instrument(ticker)
            is_bull = conv >= self.conviction_threshold
            tgt_w = target_weights.get(ticker, 0.0) if is_bull else 0.0
            alloc_cap = round(self.capital * tgt_w, 2)
            c_s = df["Close_Adj"] if "Close_Adj" in df.columns else (df["Close"] if "Close" in df.columns else df["Close_Raw"])
            c_val = float(c_s.iloc[-1])

            cost_per_share = c_val * (1.0 + fric_bps / 10000.0)
            raw_qty = int(alloc_cap // cost_per_share) if cost_per_share > 0 else 0
            qty = raw_qty if is_bull and raw_qty > 0 else 0
            action = "BUY" if qty > 0 else "HOLD_CASH"
            friction_inr = round(TaxCalculator.calculate(qty * c_val, inst_type), 4) if qty > 0 else 0.0

            pre_open, trail_prev, cl_id = None, None, None
            if qty > 0:
                assert_integer_lot(qty)
                tot_alloc += qty * c_val
                tot_tax += friction_inr
                pre_open = PreOpenAuctionRouter.route_auction_order(ticker, "BUY", c_val, qty)
                s_exit, s_rsn = TrailingStopSimulator.resolve_exit(c_val, c_val, c_val * 1.005, c_val * 0.998)
                trail_prev = {"simulated_exit": s_exit, "exit_reason": s_rsn}
                cl_id = f"{ticker}_MULTI_{latest_date}_{int(datetime.now().timestamp())}"

            evals[ticker] = AssetEvaluation(
                ticker=ticker, instrument_type=inst_type, statutory_friction_bps=fric_bps,
                conviction_score=round(conv, 4), active_signals=act_cnt, total_champions=tot_cnt,
                is_bullish=is_bull, latest_close=round(c_val, 2), target_weight=round(tgt_w, 4),
                allocated_capital=round(qty * c_val, 2), quantity=qty, statutory_friction_inr=friction_inr,
                action=action, client_order_id=cl_id, pre_open_order=pre_open, trailing_stop_preview=trail_prev,
            )
        return evals, tot_alloc, tot_tax

    def run(
        self,
        assets: Optional[List[str]] = None,
        lookback_years: float = 5.0,
        candidate_formulas: Optional[List[str]] = None,
        gateway: Optional[ShoonyaGateway] = None,
        dry_run: bool = False,
        limit_candidates: int = 50,
    ) -> MultiAssetPipelineResult:
        """Executes full multi-asset overnight pipeline lifecycle from lake to broker orders."""
        target_assets = [a.upper().replace(".NS", "") for a in (assets or CORE_MULTI_ASSET_BASKET)]

        loaded_data, convictions = self._evaluate_all_assets(
            target_assets, lookback_years, candidate_formulas, limit_candidates
        )
        active_tickers = [t for t in target_assets if convictions[t][0] >= self.conviction_threshold]

        aligned_rets, aligned_sigs = self._align_active_series(active_tickers, loaded_data)
        rets_mat = aligned_rets.to_numpy(dtype=np.float64) if not aligned_rets.empty else np.empty((0, 0))
        target_weights = self.compute_risk_parity_weights(rets_mat, active_tickers)

        backtest, latest_date = self._run_portfolio_backtest(
            active_tickers, target_weights, aligned_rets, aligned_sigs, lookback_years
        )

        evals, tot_alloc, tot_tax = self._build_evaluations(
            target_assets, loaded_data, convictions, target_weights, latest_date
        )
        final_evals = self._dispatch_paper_orders(evals, gateway=gateway, dry_run=dry_run)

        return MultiAssetPipelineResult(
            evaluated_assets=target_assets,
            active_assets=active_tickers,
            evaluations=final_evals,
            target_weights={t: round(target_weights.get(t, 0.0), 4) for t in target_assets},
            portfolio_backtest=backtest,
            total_capital=self.capital,
            total_allocated_capital=round(tot_alloc, 2),
            total_statutory_friction=round(tot_tax, 4),
            cash_reserve=round(self.capital - tot_alloc, 2),
            latest_date=latest_date,
        )
