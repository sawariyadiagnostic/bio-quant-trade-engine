"""Microstructural and Technical Feature Primitives for Muscle 7.

Implements pure, causal, scale-invariant feature calculators:
- Log returns (1, 5, 20 bars)
- Realized, Parkinson, and Garman-Klass volatility
- Normalized Average True Range (ATR)
- Lagged volume z-score (zero lookahead)
- Intraday VWAP relative drift
- Relative Strength Index (normalized)
- Bollinger Band %b
- Normalized MACD difference
- Tracking spread in basis points
"""

import numpy as np
import pandas as pd


def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Compute causal rolling log returns: ln(C_t / C_{t-k})."""
    c = close.astype(np.float64)
    c_prev = c.shift(period)
    ret = np.log(np.maximum(1e-8, c) / np.maximum(1e-8, c_prev))
    return ret.fillna(0.0)


def realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute rolling realized volatility of 1-bar log returns."""
    ret = log_returns(close, period=1)
    vol = ret.rolling(period, min_periods=period // 2).std()
    return vol.fillna(0.0)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Compute Parkinson high-low scale-invariant volatility estimator."""
    h = high.astype(np.float64)
    l = low.astype(np.float64)
    log_hl = np.log(np.maximum(1e-8, h) / np.maximum(1e-8, l))
    factor = 1.0 / (4.0 * np.log(2.0))
    squared_terms = factor * (log_hl**2)
    p_vol = np.sqrt(squared_terms.rolling(window, min_periods=window // 2).mean())
    return p_vol.fillna(0.0)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute Garman-Klass OHLC volatility estimator."""
    o = open_.astype(np.float64)
    h = high.astype(np.float64)
    l = low.astype(np.float64)
    c = close.astype(np.float64)

    log_hl = np.log(np.maximum(1e-8, h) / np.maximum(1e-8, l))
    log_co = np.log(np.maximum(1e-8, c) / np.maximum(1e-8, o))

    term1 = 0.5 * (log_hl**2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    gk_variance = term1 - term2
    gk_vol = np.sqrt(np.maximum(0.0, gk_variance.rolling(window, min_periods=window // 2).mean()))
    return gk_vol.fillna(0.0)


def normalized_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Compute Normalized Average True Range (ATR / Close)."""
    h = high.astype(np.float64)
    l = low.astype(np.float64)
    c = close.astype(np.float64)
    c_prev = c.shift(1).fillna(c.iloc[0] if len(c) > 0 else 0.0)

    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window, min_periods=window // 2).mean()
    atr_norm = atr / np.maximum(1e-4, c)
    return atr_norm.fillna(0.0)


def vwap_drift(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """Compute fractional drift of Close from VWAP: (Close - VWAP) / VWAP."""
    c = close.astype(np.float64)
    v = vwap.astype(np.float64)
    drift = (c - v) / np.maximum(1e-4, v)
    return drift.fillna(0.0)


def lagged_volume_zscore(volume: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    """Compute volume z-score strictly lagged by lag bars to prevent lookahead."""
    vol = volume.astype(np.float64)
    vol_lag = vol.shift(lag)
    mean = vol_lag.rolling(window, min_periods=window // 2).mean()
    std = vol_lag.rolling(window, min_periods=window // 2).std().replace(0.0, np.nan)

    z = (vol_lag - mean) / std
    return z.fillna(0.0).clip(-5.0, 5.0)


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index centered at 0 in [-1.0, 1.0]."""
    delta = close.astype(np.float64).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window, min_periods=window // 2).mean()
    avg_loss = loss.rolling(window, min_periods=window // 2).mean()

    rs = avg_gain / np.maximum(1e-8, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Normalize from [0, 100] to [-1.0, 1.0]
    norm_rsi = (rsi - 50.0) / 50.0
    return norm_rsi.fillna(0.0).clip(-1.0, 1.0)


def bollinger_pct(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Compute Bollinger Band %b normalized in [0.0, 1.0]."""
    c = close.astype(np.float64)
    mean = c.rolling(window, min_periods=window // 2).mean()
    std = c.rolling(window, min_periods=window // 2).std()

    upper = mean + num_std * std
    lower = mean - num_std * std
    width = np.maximum(1e-4, upper - lower)

    pct_b = (c - lower) / width
    return pct_b.fillna(0.5).clip(-0.5, 1.5)


def macd_normalized(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """Compute normalized MACD histogram difference: (MACD - Signal) / Close."""
    c = close.astype(np.float64)
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    norm_hist = hist / np.maximum(1e-4, c)
    return norm_hist.fillna(0.0)


def tracking_spread_bps(etf_close: pd.Series, theo_close: pd.Series) -> pd.Series:
    """Compute tracking spread in basis points: (ETF - Theo) / Theo * 10,000."""
    p_etf = etf_close.astype(np.float64)
    p_theo = theo_close.astype(np.float64)
    bps = (p_etf - p_theo) / np.maximum(1e-4, p_theo) * 10000.0
    return bps.fillna(0.0)
