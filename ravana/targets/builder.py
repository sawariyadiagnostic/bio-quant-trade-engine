"""Overnight Target Builder for Muscle 8.

Extracts forward overnight returns (Session Close -> Next Session Open),
deducts delivery frictions and flat CDSL DP debit fees, and generates
continuous targets, binary hurdle classifications, and sample weighting masks.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ravana.targets.config import TargetConfig
from ravana.targets.costs import DeliveryFrictionModel


class OvernightTargetBuilder:
    """Builds causal forward overnight economic target contracts."""

    def __init__(self, config: Optional[TargetConfig] = None) -> None:
        self.config = config or TargetConfig()
        self.friction_model = DeliveryFrictionModel(
            variable_bps=self.config.variable_bps,
            flat_dp_charge_inr=self.config.dp_fee_inr,
            default_capital_inr=self.config.capital_inr,
            min_capital_inr=self.config.min_capital_inr,
        )

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forward overnight returns and labels for input bar DataFrame."""
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

        # Group by normalized date to identify distinct trading sessions
        session_dates = df.index.normalize()
        unique_sessions = np.unique(session_dates)

        hurdle_frac = self.config.hurdle_bps / 10000.0

        for s_idx in range(len(unique_sessions) - 1):
            curr_date = unique_sessions[s_idx]
            next_date = unique_sessions[s_idx + 1]

            entry_idx, exit_idx = self._get_session_pair_indices(session_dates, curr_date, next_date)
            if entry_idx is None or exit_idx is None:
                continue

            entry_close = float(df["Close"].iloc[entry_idx])
            exit_price = (
                float(df["Open"].iloc[exit_idx])
                if self.config.use_next_open_as_exit
                else float(df["Close"].iloc[exit_idx])
            )

            if not np.isfinite(entry_close) or not np.isfinite(exit_price) or entry_close <= 0.0 or exit_price <= 0.0:
                continue

            gross_ret = (exit_price - entry_close) / entry_close
            net_ret = self.friction_model.apply_friction(gross_ret, self.config.capital_inr)
            days_held = max(1, (pd.Timestamp(next_date) - pd.Timestamp(curr_date)).days)

            gross_returns[entry_idx] = gross_ret
            net_returns[entry_idx] = net_ret
            binary_labels[entry_idx] = 1 if net_ret >= hurdle_frac else 0
            is_entry_bar[entry_idx] = 1
            sample_weights[entry_idx] = 1.0
            calendar_days[entry_idx] = days_held

        return self._build_result_df(
            df.index, gross_returns, net_returns, binary_labels, is_entry_bar, sample_weights, calendar_days
        )

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
