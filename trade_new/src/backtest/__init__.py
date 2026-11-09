"""Backtest module for high-performance vectorized matrix simulations and Qlib factor metrics."""
from .matrix_engine import MatrixSimulator
from .factor_analytics import calculate_factor_ic, calculate_factor_decay

__all__ = ["MatrixSimulator", "calculate_factor_ic", "calculate_factor_decay"]
