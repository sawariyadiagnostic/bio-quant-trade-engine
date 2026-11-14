"""Configuration for Muscle 8 Multi-Horizon Target & Objective Builder."""

from dataclasses import dataclass
from enum import Enum


class TargetHorizon(str, Enum):
    """Target horizon classification."""

    OVERNIGHT = "OVERNIGHT"
    INTRADAY_N_BARS = "INTRADAY_N_BARS"
    MULTI_DAY_SWING = "MULTI_DAY_SWING"


@dataclass
class TargetConfig:
    """Configuration parameters for multi-horizon target extraction and friction deduction."""

    # Horizon and operational parameters
    horizon: TargetHorizon = TargetHorizon.OVERNIGHT  # default OVERNIGHT for 100% backward compatibility
    forward_bars: int = 6  # for INTRADAY_N_BARS (default 6 bars = 30 min on 5m)
    intraday_cutoff_time: str = "14:45"  # Quarantine window to prevent intraday crossing 15:15
    holding_days: int = 3  # for MULTI_DAY_SWING

    # Account sizing and friction parameters
    capital_inr: float = 200000.0  # Institutional min capital gate: Rs. 2,00,000
    min_capital_inr: float = 200000.0
    variable_bps: float = 2.0  # Brokerage, STT, exchange turnover: 2.0 bps (0.02%)
    slippage_bps: float = 0.0  # Half-spread & execution slippage: 0.0 bps default
    dp_fee_inr: float = 17.0  # CDSL flat DP debit fee: Rs. 17.00 per sell delivery

    # Labeling hurdle
    hurdle_bps: float = 5.0  # Hurdle rate to trigger positive alpha label (5.0 bps)

    # Execution timing conventions
    entry_bar_offset_from_end: int = 0  # 0 = very last bar of session (15:20-15:25)
    exit_bar_offset_from_start: int = 0  # 0 = very first bar of next session (09:15-09:20)
    use_next_open_as_exit: bool = True  # True: exit at 09:15 Open auction fill
