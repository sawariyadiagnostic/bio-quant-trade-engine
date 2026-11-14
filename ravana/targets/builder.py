"""Multi-Horizon Target Builder for Muscle 8.

Extracts forward returns across diverse investment horizons (Overnight,
Intraday N-bars, and Multi-Day Swing), deducts applicable statutory frictions,
and generates continuous targets, binary hurdle classifications, and sample weighting masks.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.targets.config import TargetConfig, TargetHorizon
from ravana.targets.costs import DeliveryFrictionModel, IntradayFrictionModel


class OvernightTargetBuilder:
    """Builds causal forward economic target contracts across multiple horizons."""

    def __init__(self, config: Optional[TargetConfig] = None) -> None:
        self.config = config or TargetConfig()
        self.friction_model = DeliveryFrictionModel(
            variable_bps=self.config.variable_bps,
            slippage_bps=self.config.slippage_bps,
            flat_dp_charge_inr=self.config.dp_fee_inr,
            default_capital_inr=self.config.capital_inr,
            min_capital_inr=self.config.min_capital_inr,
        )
        self.intraday_friction_model = IntradayFrictionModel(
            variable_bps=self.config.intraday_variable_bps,
            slippage_bps=self.config.slippage_bps,
            default_capital_inr=self.config.capital_inr,
            min_capital_inr=self.config.min_capital_inr,
        )

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forward returns and labels for input bar DataFrame."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        n = len(df)
        gross_returns = np.zeros(n, dtype=np.float64)
        net_returns = np.zeros(n, dtype=np.float64)
        binary_labels = np.zeros(n, dtype=np.int32)
        is_entry_bar = np.zeros(n, dtype=np.int32)
        sample_weights = np.zeros(n, dtype=np.float64)
        calendar_days = np.zeros(n, dtype=np.int32)

        if n < 2:
            return self._build_result_df(
                df.index, gross_returns, net_returns, binary_labels, is_entry_bar, sample_weights, calendar_days
            )

        session_dates = df.index.normalize()
        unique_sessions = np.unique(session_dates)
        hurdle_frac = self.config.hurdle_bps / 10000.0

        if self.config.horizon == TargetHorizon.INTRADAY_N_BARS:
            self._build_intraday_targets(
                df, session_dates, gross_returns, net_returns, binary_labels,
                is_entry_bar, sample_weights, calendar_days, hurdle_frac
            )
        elif self.config.horizon == TargetHorizon.MULTI_DAY_SWING:
            self._build_session_pair_targets(
                df, session_dates, unique_sessions, max(1, self.config.holding_days),
                gross_returns, net_returns, binary_labels, is_entry_bar,
                sample_weights, calendar_days, hurdle_frac
            )
        else:  # TargetHorizon.OVERNIGHT default
            self._build_session_pair_targets(
                df, session_dates, unique_sessions, 1,
                gross_returns, net_returns, binary_labels, is_entry_bar,
                sample_weights, calendar_days, hurdle_frac
            )

        return self._build_result_df(
            df.index, gross_returns, net_returns, binary_labels, is_entry_bar, sample_weights, calendar_days
        )

    def _build_intraday_targets(
        self,
        df: pd.DataFrame,
        session_dates: pd.DatetimeIndex,
        gross_returns: np.ndarray,
        net_returns: np.ndarray,
        binary_labels: np.ndarray,
        is_entry_bar: np.ndarray,
        sample_weights: np.ndarray,
        calendar_days: np.ndarray,
        hurdle_frac: float,
    ) -> None:
        """Compute forward intraday N-bar returns and labels within single sessions."""
        k = self.config.forward_bars
        n = len(df)
        cutoff = self.config.intraday_cutoff_time

        for i in range(n):
            if i + 1 + k >= n:
                break

            bar_time = df.index[i].strftime("%H:%M")
            if bar_time > cutoff:
                continue

            if session_dates[i + 1] != session_dates[i + 1 + k]:
                continue

            entry_exec_price = float(df["Open"].iloc[i + 1])
            exit_price = float(df["Open"].iloc[i + 1 + k])

            if (
                not np.isfinite(entry_exec_price)
                or not np.isfinite(exit_price)
                or entry_exec_price <= 0.0
                or exit_price <= 0.0
            ):
                continue

            gross_ret = (exit_price - entry_exec_price) / entry_exec_price
            net_ret = self.intraday_friction_model.apply_friction(gross_ret, self.config.capital_inr)

            gross_returns[i] = gross_ret
            net_returns[i] = net_ret
            binary_labels[i] = 1 if net_ret >= hurdle_frac else 0
            is_entry_bar[i] = 1
            sample_weights[i] = 1.0
            calendar_days[i] = 0

    def _build_session_pair_targets(
        self,
        df: pd.DataFrame,
        session_dates: pd.DatetimeIndex,
        unique_sessions: np.ndarray,
        session_step: int,
        gross_returns: np.ndarray,
        net_returns: np.ndarray,
        binary_labels: np.ndarray,
        is_entry_bar: np.ndarray,
        sample_weights: np.ndarray,
        calendar_days: np.ndarray,
        hurdle_frac: float,
    ) -> None:
        """Compute forward session-pair returns across specified session interval."""
        for s_idx in range(len(unique_sessions) - session_step):
            curr_date = unique_sessions[s_idx]
            next_date = unique_sessions[s_idx + session_step]

            entry_idx, exit_idx = self._get_session_pair_indices(session_dates, curr_date, next_date)
            if entry_idx is None or exit_idx is None:
                continue

            if (
                self.config.entry_bar_offset_from_end > 0
                and entry_idx + 1 < len(df)
                and session_dates[entry_idx + 1] == curr_date
            ):
                entry_exec_price = float(df["Open"].iloc[entry_idx + 1])
            else:
                entry_exec_price = float(df["Close"].iloc[entry_idx])

            exit_price = (
                float(df["Open"].iloc[exit_idx])
                if self.config.use_next_open_as_exit
                else float(df["Close"].iloc[exit_idx])
            )

            if (
                not np.isfinite(entry_exec_price)
                or not np.isfinite(exit_price)
                or entry_exec_price <= 0.0
                or exit_price <= 0.0
            ):
                continue

            gross_ret = (exit_price - entry_exec_price) / entry_exec_price
            net_ret = self.friction_model.apply_friction(gross_ret, self.config.capital_inr)
            days_held = max(1, (pd.Timestamp(next_date) - pd.Timestamp(curr_date)).days)

            gross_returns[entry_idx] = gross_ret
            net_returns[entry_idx] = net_ret
            binary_labels[entry_idx] = 1 if net_ret >= hurdle_frac else 0
            is_entry_bar[entry_idx] = 1
            sample_weights[entry_idx] = 1.0
            calendar_days[entry_idx] = days_held

    def _get_session_pair_indices(
        self, session_dates: pd.DatetimeIndex, curr_date: pd.Timestamp, next_date: pd.Timestamp
    ) -> tuple[Optional[int], Optional[int]]:
        """Find the entry bar index of current session and exit bar index of next session."""
        curr_locs = np.where(session_dates == curr_date)[0]
        next_locs = np.where(session_dates == next_date)[0]

        req_curr = 1 + self.config.entry_bar_offset_from_end
        req_next = 1 + self.config.exit_bar_offset_from_start
        if len(curr_locs) < req_curr or len(next_locs) < req_next:
            return None, None

        entry_idx = int(curr_locs[-req_curr])
        exit_idx = int(next_locs[self.config.exit_bar_offset_from_start])
        return entry_idx, exit_idx

    def _build_result_df(
        self,
        index: pd.DatetimeIndex,
        gross_ret: np.ndarray,
        net_ret: np.ndarray,
        binary_labels: np.ndarray,
        is_entry: np.ndarray,
        weights: np.ndarray,
        calendar_days: np.ndarray,
    ) -> pd.DataFrame:
        """Package aligned outputs into an immutable target contract."""
        return pd.DataFrame(
            {
                "gross_return": gross_ret,
                "net_return": net_ret,
                "binary_label": binary_labels,
                "is_entry_bar": is_entry,
                "sample_weight": weights,
                "calendar_hold_days": calendar_days,
            },
            index=index,
        )


MultiHorizonTargetBuilder = OvernightTargetBuilder
