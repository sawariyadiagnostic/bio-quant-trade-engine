"""Configuration for Muscle 8 Overnight Target & Objective Builder."""

from dataclasses import dataclass


@dataclass
class TargetConfig:
    """Configuration parameters for overnight target extraction and friction deduction."""

    # Account sizing and friction parameters
    capital_inr: float = 200000.0  # Institutional min capital gate: Rs. 2,00,000
    min_capital_inr: float = 200000.0
    variable_bps: float = 2.0  # Brokerage, STT, exchange turnover: 2.0 bps (0.02%)
    dp_fee_inr: float = 17.0  # CDSL flat DP debit fee: Rs. 17.00 per sell delivery

    # Labeling hurdle
    hurdle_bps: float = 5.0  # Hurdle rate to trigger positive alpha label (5.0 bps)

    # Execution timing conventions
    entry_bar_offset_from_end: int = 0  # 0 = very last bar of session (15:20-15:25)
    exit_bar_offset_from_start: int = 0  # 0 = very first bar of next session (09:15-09:20)
    use_next_open_as_exit: bool = True  # True: exit at 09:15 Open auction fill
