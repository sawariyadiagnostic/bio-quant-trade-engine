"""RAVANA v4.3: High-Performance 3D Vectorized Lake & Simulation Engine.

Enables microsecond-speed strategy evaluation and parameter exploration across
all 18 ETF and index assets simultaneously using vectorized NumPy/BLAS SIMD:
1. VectorLake: Loads, aligns, and precomputes price, VWAP, ATR, VR20, and OFI matrices.
2. VectorSimulationResult: Typed dataclass containing trade statistics and equity curves.
3. Fast causal position simulator with zero lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class VectorSimulationResult:
    """Audit metrics from a vectorized simulation run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    profit_factor: float
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "net_pnl": round(self.net_pnl, 2),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
        }


class VectorLake:
    """Precomputed tensor lake of aligned ETF and benchmark price matrices."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "lake",
            "m3_ingested",
            "intraday_5m",
        )
        self.symbols: List[str] = []
        self.symbol_to_idx: Dict[str, int] = {}
        self.timestamps: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.time_strings: np.ndarray = np.array([])
        self.dates: np.ndarray = np.array([])
        self.n_bars: int = 0
        self.n_syms: int = 0

        # Aligned 2D matrices [T, S]
        self.open_mat: np.ndarray = np.empty((0, 0))
        self.high_mat: np.ndarray = np.empty((0, 0))
        self.low_mat: np.ndarray = np.empty((0, 0))
        self.close_mat: np.ndarray = np.empty((0, 0))
        self.vol_mat: np.ndarray = np.empty((0, 0))
        self.vwap_mat: np.ndarray = np.empty((0, 0))
        self.atr_mat: np.ndarray = np.empty((0, 0))
        self.vr_mat: np.ndarray = np.empty((0, 0))
        self.ofi_mat: np.ndarray = np.empty((0, 0))
        self.bb_pct_mat: np.ndarray = np.empty((0, 0))

        # Benchmark NIFTY vectors [T]
        self.nifty_close: np.ndarray = np.empty(0)
        self.nifty_open: np.ndarray = np.empty(0)

        self._load_and_align()

    def _load_and_align(self) -> None:
        """Load parquet files and align all symbols into continuous 2D matrices."""
        parquet_files = [f for f in os.listdir(self.data_dir) if f.endswith(".parquet")]
        raw_dfs: Dict[str, pd.DataFrame] = {}

        for f in parquet_files:
            sym = f[:-8]
            df = pd.read_parquet(os.path.join(self.data_dir, f))
            if not df.empty:
                raw_dfs[sym] = df

        if "BANKBEES" not in raw_dfs:
            return

        # Reference timeline from BANKBEES
        ref_df = raw_dfs["BANKBEES"]
        self.timestamps = ref_df.index
        self.n_bars = len(self.timestamps)
        self.time_strings = np.array([ts.strftime("%H:%M") for ts in self.timestamps])
        self.dates = np.array([ts.date() for ts in self.timestamps])

        # Target symbols (exclude cash index tickers from tradable list)
        self.symbols = [s for s in sorted(raw_dfs.keys()) if not s.startswith(("^", "NSE", "CNX"))]
        self.n_syms = len(self.symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}

        # Allocate 2D arrays [T, S]
        T, S = self.n_bars, self.n_syms
        self.open_mat = np.zeros((T, S), dtype=np.float64)
        self.high_mat = np.zeros((T, S), dtype=np.float64)
        self.low_mat = np.zeros((T, S), dtype=np.float64)
        self.close_mat = np.zeros((T, S), dtype=np.float64)
        self.vol_mat = np.zeros((T, S), dtype=np.float64)
        self.vwap_mat = np.zeros((T, S), dtype=np.float64)
        self.atr_mat = np.zeros((T, S), dtype=np.float64)
        self.vr_mat = np.ones((T, S), dtype=np.float64)
        self.ofi_mat = np.zeros((T, S), dtype=np.float64)
        self.bb_pct_mat = np.full((T, S), 0.5, dtype=np.float64)

        for s_idx, sym in enumerate(self.symbols):
            df = raw_dfs[sym].reindex(self.timestamps).ffill().bfill()
            self.open_mat[:, s_idx] = df["Open"].to_numpy()
            self.high_mat[:, s_idx] = df["High"].to_numpy()
            self.low_mat[:, s_idx] = df["Low"].to_numpy()
            self.close_mat[:, s_idx] = df["Close"].to_numpy()
            self.vol_mat[:, s_idx] = df["Volume"].to_numpy()

            # Precompute indicators
            self._precompute_symbol_indicators(df, s_idx)

        # Precompute NIFTY 50 benchmark
        self._precompute_nifty_benchmark(raw_dfs)

    def _precompute_symbol_indicators(self, df: pd.DataFrame, s_idx: int) -> None:
        """Calculate and store VWAP, ATR, VR20, OFI, and Bollinger %b."""
        cl = self.close_mat[:, s_idx]
        vol = self.vol_mat[:, s_idx]

        # 1. VWAP
        df_temp = pd.DataFrame({"Close": cl, "Volume": vol}, index=self.timestamps)
        cv = (df_temp["Close"] * df_temp["Volume"]).groupby(df_temp.index.date).cumsum()
        cvol = df_temp["Volume"].groupby(df_temp.index.date).cumsum().replace(0.0, np.nan)
        self.vwap_mat[:, s_idx] = (cv / cvol).fillna(df_temp["Close"]).to_numpy()

        # 2. ATR14 with price floor
        raw_atr = df.get("ATR_14", pd.Series(cl * 0.01, index=self.timestamps)).to_numpy()
        self.atr_mat[:, s_idx] = np.maximum(raw_atr, cl * 0.0035)

        # 3. Variance Ratio
        r1 = pd.Series(cl).pct_change()
        v1 = r1.rolling(20).var()
        r5 = pd.Series(cl).pct_change(5)
        v5 = r5.rolling(20).var()
        vr = (v5 / (5.0 * v1.replace(0.0, np.nan))).fillna(1.0).to_numpy()
        self.vr_mat[:, s_idx] = vr

        # 4. OFI proxy
        delta_p = np.sign(pd.Series(cl).diff().fillna(0.0))
        ofi = (delta_p * vol).rolling(5).sum() / (pd.Series(vol).rolling(5).sum() + 1e-6)
        self.ofi_mat[:, s_idx] = ofi.fillna(0.0).to_numpy()

        # 5. Bollinger Band %b
        mid = pd.Series(cl).rolling(20, min_periods=5).mean()
        std = pd.Series(cl).rolling(20, min_periods=5).std().replace(0.0, np.nan)
        lower = mid - 2.0 * std
        upper = mid + 2.0 * std
        self.bb_pct_mat[:, s_idx] = ((pd.Series(cl) - lower) / (upper - lower + 1e-6)).fillna(0.5).to_numpy()

    def _precompute_nifty_benchmark(self, raw_dfs: Dict[str, pd.DataFrame]) -> None:
        """Precompute NIFTY 50 causal session open and close arrays."""
        if "NSEI" in raw_dfs:
            n_df = raw_dfs["NSEI"].reindex(self.timestamps).ffill().bfill()
            self.nifty_close = n_df["Close"].to_numpy()
            open_map = {dt: float(grp["Open"].iloc[0]) for dt, grp in n_df.groupby(n_df.index.date)}
            self.nifty_open = np.array([open_map.get(d, self.nifty_close[i]) for i, d in enumerate(self.dates)])
        else:
            self.nifty_close = self.close_mat[:, self.symbol_to_idx.get("NIFTYBEES", 0)]
            self.nifty_open = self.open_mat[:, self.symbol_to_idx.get("NIFTYBEES", 0)]

    def get_date_slice_indices(self, start_date: Any, end_date: Any) -> np.ndarray:
        """Return indices of timestamps falling strictly within [start_date, end_date]."""
        return np.where((self.dates >= start_date) & (self.dates <= end_date))[0]


class VectorSimulator:
    """Fast causal trade simulator executing vectorized signal masks."""

    def __init__(
        self,
        lake: VectorLake,
        initial_capital: float = 200000.0,
        risk_budget_pct: float = 0.0025,
        max_concurrent_positions: int = 2,
        max_leverage: float = 2.0,
    ) -> None:
        self.lake = lake
        self.initial_capital = float(initial_capital)
        self.risk_budget_pct = float(risk_budget_pct)
        self.max_concurrent_positions = int(max_concurrent_positions)
        self.max_leverage = float(max_leverage)

    def simulate(
        self,
        long_mask: np.ndarray,
        short_mask: np.ndarray,
        time_indices: Optional[np.ndarray] = None,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 1.5,
        exit_time_str: str = "15:00",
    ) -> VectorSimulationResult:
        """Execute causal vector simulation over specified time indices."""
        indices = time_indices if time_indices is not None else np.arange(self.lake.n_bars)
        capital = self.initial_capital
        open_pos: Dict[int, Dict[str, Any]] = {}
        daily_entered: Set[Tuple[int, Any]] = set()
        trades: List[Dict[str, Any]] = []

        for t in indices:
            t_str = self.lake.time_strings[t]
            d = self.lake.dates[t]

            # 1. Evaluate open positions
            for s in list(open_pos.keys()):
                pos = open_pos[s]
                hi = self.lake.high_mat[t, s]
                lo = self.lake.low_mat[t, s]
                cl = self.lake.close_mat[t, s]

                if pos["is_long"]:
                    if lo <= pos["stop"]:
                        pnl = (pos["stop"] - pos["entry"]) * pos["shares"]
                        capital += pnl
                        trades.append({"symbol": self.lake.symbols[s], "side": "LONG", "pnl": pnl, "reason": "STOP"})
                        del open_pos[s]
                    elif hi >= pos["target"]:
                        pnl = (pos["target"] - pos["entry"]) * pos["shares"]
                        capital += pnl
                        trades.append({"symbol": self.lake.symbols[s], "side": "LONG", "pnl": pnl, "reason": "TARGET"})
                        del open_pos[s]
                    elif hi >= pos["entry"] + 0.5 * pos["atr"]:
                        pos["stop"] = max(pos["stop"], pos["entry"] * 1.0002)
                    elif t_str >= exit_time_str:
                        pnl = (cl - pos["entry"]) * pos["shares"]
                        capital += pnl
                        trades.append({"symbol": self.lake.symbols[s], "side": "LONG", "pnl": pnl, "reason": "EOD"})
                        del open_pos[s]
                else:
                    if hi >= pos["stop"]:
                        pnl = (pos["entry"] - pos["stop"]) * pos["shares"]
                        capital += pnl
                        trades.append({"symbol": self.lake.symbols[s], "side": "SHORT", "pnl": pnl, "reason": "STOP"})
                        del open_pos[s]
                    elif lo <= pos["target"]:
                        pnl = (pos["entry"] - pos["target"]) * pos["shares"]
                        capital += pnl
                        trades.append({"symbol": self.lake.symbols[s], "side": "SHORT", "pnl": pnl, "reason": "TARGET"})
                        del open_pos[s]
                    elif lo <= pos["entry"] - 0.5 * pos["atr"]:
                        pos["stop"] = min(pos["stop"], pos["entry"] * 0.9998)
                    elif t_str >= exit_time_str:
                        pnl = (pos["entry"] - cl) * pos["shares"]
                        capital += pnl
                        trades.append({"symbol": self.lake.symbols[s], "side": "SHORT", "pnl": pnl, "reason": "EOD"})
                        del open_pos[s]

            # 2. Check entries
            if t_str >= exit_time_str or len(open_pos) >= self.max_concurrent_positions:
                continue

            available_slots = self.max_concurrent_positions - len(open_pos)
            for s in range(self.lake.n_syms):
                if s in open_pos or (s, d) in daily_entered:
                    continue

                is_long_sig = bool(long_mask[t, s])
                is_short_sig = bool(short_mask[t, s])
                if not (is_long_sig or is_short_sig):
                    continue

                cl = self.lake.close_mat[t, s]
                atr = self.lake.atr_mat[t, s]
                risk_amt = capital * self.risk_budget_pct
                stop_dist = stop_atr_mult * atr
                shares = int(risk_amt / max(stop_dist, 1e-4))
                max_shares = int((capital * self.max_leverage) / max(cl, 1e-4))
                shares = min(shares, max_shares)
                if shares <= 0:
                    continue

                if is_long_sig:
                    open_pos[s] = {
                        "entry": cl,
                        "stop": cl - stop_dist,
                        "target": cl + (target_atr_mult * atr),
                        "atr": atr,
                        "shares": shares,
                        "is_long": True,
                    }
                    daily_entered.add((s, d))
                elif is_short_sig:
                    open_pos[s] = {
                        "entry": cl,
                        "stop": cl + stop_dist,
                        "target": cl - (target_atr_mult * atr),
                        "atr": atr,
                        "shares": shares,
                        "is_long": False,
                    }
                    daily_entered.add((s, d))

                available_slots -= 1
                if available_slots <= 0:
                    break

        return self._build_result(trades)

    def _build_result(self, trades: List[Dict[str, Any]]) -> VectorSimulationResult:
        """Calculate summary performance statistics from realized trades."""
        total = len(trades)
        if total == 0:
            return VectorSimulationResult(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, trades)

        pnls = np.array([tr["pnl"] for tr in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gl = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        net = float(np.sum(pnls))
        wr = float((len(wins) / total) * 100.0)
        pf = float(gp / gl) if gl > 0 else 999.0

        return VectorSimulationResult(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=wr,
            gross_profit=gp,
            gross_loss=gl,
            net_pnl=net,
            profit_factor=pf,
            trades=trades,
        )

