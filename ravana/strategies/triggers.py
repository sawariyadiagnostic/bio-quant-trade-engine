"""Ten Hardened Structural Triggers partitioned into 3 Uncorrelated Regime Buckets.

Bucket 1 (Trend & Impulse - active when VARIANCE_RATIO_20 > 1.05):
1. OpeningRangeBreakoutTrigger
2. ThursdayExpiryGammaTrigger
3. CrossSectorLeadLagTrigger
4. SectorPEADInstitutionalDriftTrigger

Bucket 2 (Mean-Reversion, Stop-Sweep & Absorption - active when VARIANCE_RATIO_20 <= 1.05):
5. INAVSpreadMeanReversionTrigger
6. ClosingVWAPSnapbackTrigger
7. DIISIPVWAPAbsorptionTrigger
8. ICTUnicornStopSweepTrigger

Bucket 3 (Macro, Bond & All-Weather Overnight):
9. AfternoonBondEuropeTWAPTrigger
10. OvernightContinuationTrigger
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ravana.data.ingestion.session_gatekeeper import KNOWN_NSE_HOLIDAYS
from ravana.strategies.base import (
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
)

# ---------------------------------------------------------------------------
# Microstructure Helpers & Resolvers
# ---------------------------------------------------------------------------


def extract_variance_ratio(df: pd.DataFrame, window: int = 20, q: int = 5) -> pd.Series:
    """Calculate Lo-MacKinlay Variance Ratio or return precomputed column."""
    if "VARIANCE_RATIO_20" in df.columns:
        return df["VARIANCE_RATIO_20"].fillna(1.0)

    close = df["Close"]
    log_p = np.log(np.maximum(1e-4, close))
    ret_1 = log_p.diff()
    ret_q = log_p.diff(q)

    var_1 = ret_1.rolling(window, min_periods=max(5, q)).var()
    var_q = ret_q.rolling(window, min_periods=max(5, q)).var()

    vr = var_q / (q * var_1.replace(0.0, np.nan))
    return vr.fillna(1.0).clip(0.1, 5.0)


def extract_ofi_proxy(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Extract Order Flow Imbalance proxy or return precomputed column."""
    for col in ("OFI_5", "OFI", "ofi_5"):
        if col in df.columns:
            return df[col].fillna(0.0).clip(-1.0, 1.0)

    if "Volume" not in df.columns or "Close" not in df.columns:
        return pd.Series(0.0, index=df.index)

    ret = df["Close"].diff()
    signed_vol = np.sign(ret) * df["Volume"]
    vol_sum = df["Volume"].rolling(window, min_periods=1).sum()
    ofi = signed_vol.rolling(window, min_periods=1).sum() / (vol_sum + 1e-6)
    return ofi.fillna(0.0).clip(-1.0, 1.0)


def extract_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Extract or compute intraday session VWAP."""
    if "VWAP" in df.columns:
        return df["VWAP"]

    dates = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else [0] * len(df)
    typical_px = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_pv = (typical_px * df["Volume"]).groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()
    return cum_pv / (cum_vol + 1e-6)


def extract_spread_metrics(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (SPREAD_BPS, SPREAD_ZSCORE_20) from DataFrame."""
    bps = df["SPREAD_BPS"] if "SPREAD_BPS" in df.columns else None
    if bps is None and "Theoretical_Close" in df.columns:
        bps = ((df["Close"] - df["Theoretical_Close"]) / df["Theoretical_Close"]) * 10000.0
    elif bps is None and "Spread_INR" in df.columns:
        bps = (df["Spread_INR"] / df["Close"]) * 10000.0
    elif bps is None:
        bps = pd.Series(0.0, index=df.index)

    if "SPREAD_ZSCORE_20" in df.columns:
        zscore = df["SPREAD_ZSCORE_20"]
    else:
        mean = bps.rolling(20, min_periods=5).mean()
        std = bps.rolling(20, min_periods=5).std().replace(0.0, np.nan)
        zscore = ((bps - mean) / std).fillna(0.0).clip(-5.0, 5.0)

    return bps.fillna(0.0), zscore.fillna(0.0)


def is_weekly_expiry(ts: pd.Timestamp, holidays: Optional[Set[str]] = None) -> bool:
    """Determine if ts falls on the weekly expiry day (Thursday or shifted to Wednesday)."""
    h_set = holidays if holidays is not None else KNOWN_NSE_HOLIDAYS
    dt = ts.normalize()
    # Weekly expiry is normally Thursday (dayofweek == 3)
    thursday = dt + pd.Timedelta(days=(3 - dt.dayofweek))
    expiry_date = thursday

    # Shift back if holiday or weekend
    while expiry_date.strftime("%Y-%m-%d") in h_set or expiry_date.dayofweek >= 5:
        expiry_date = expiry_date - pd.Timedelta(days=1)

    return dt.date() == expiry_date.date()


# ---------------------------------------------------------------------------
# BUCKET 1: Trend & Impulse (VARIANCE_RATIO_20 > 1.05)
# ---------------------------------------------------------------------------


class OpeningRangeBreakoutTrigger(StructuralTrigger):
    """Trigger 1: Opening Range Breakout in 09:35-10:30 IST window.

    Starts at 09:35 after the 09:30-09:34 HFT trap bar closes.
    Breaks 15m range high with OFI_5 > 0.25 and VR >= 0.90. Strictly MIS.
    """

    name: str = "OpeningRangeBreakoutTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        vr = extract_variance_ratio(df)
        ofi = extract_ofi_proxy(df)

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("09:35" <= t_str <= "10:30"):
                continue
            if vr.iloc[i] < 0.90:
                continue
            if ofi.iloc[i] <= 0.25:
                continue

            day_bars = df[df.index.normalize() == ts.normalize()]
            or_bars = day_bars[(day_bars.index.strftime("%H:%M") >= "09:15") & (day_bars.index.strftime("%H:%M") <= "09:25")]
            if len(or_bars) == 0:
                continue
            or_high = or_bars["High"].max()

            if df["Close"].iloc[i] > or_high:
                conviction.iloc[i] = min(1.0, 0.50 + float(ofi.iloc[i]) * 0.50)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            if conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "TREND_IMPULSE",
                            "stop_price": float(df["Low"].iloc[i]),
                        },
                    )
                )
        return signals


class ThursdayExpiryGammaTrigger(StructuralTrigger):
    """Trigger 2: Thursday Expiry Gamma unpinning breakout in 14:15-15:15 IST.

    Shifts to Wednesday if Thursday is an exchange holiday.
    Breaks 6-bar high with volume surge. Strictly MIS.
    """

    name: str = "ThursdayExpiryGammaTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        rolling_high_6 = df["High"].shift(1).rolling(6, min_periods=6).max()
        vol_mean_20 = df["Volume"].rolling(20, min_periods=5).mean()

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("14:15" <= t_str <= "15:15"):
                continue
            if not is_weekly_expiry(ts):
                continue
            if np.isnan(rolling_high_6.iloc[i]):
                continue

            breaks_high = df["Close"].iloc[i] > rolling_high_6.iloc[i]
            vol_surge = df["Volume"].iloc[i] >= 1.5 * vol_mean_20.iloc[i]

            if breaks_high and vol_surge:
                ratio = df["Volume"].iloc[i] / (vol_mean_20.iloc[i] + 1e-6)
                conviction.iloc[i] = min(1.0, 0.60 + min(0.40, (ratio - 1.5) * 0.20))

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            if conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "EXPIRY_GAMMA",
                            "stop_price": float(df["Low"].iloc[i] * 0.998),
                        },
                    )
                )
        return signals


class CrossSectorLeadLagTrigger(StructuralTrigger):
    """Trigger 3: Big-Brother to Little-Brother Lead-Lag in 09:20-14:35 IST.

    Pairs: BANKBEES -> PSUBNKBEES, NIFTYBEES -> JUNIORBEES, GOLDBEES -> SILVERBEES.
    Evaluates leader return strictly on bar t-1 Close (>0.35%) with OFI_5 > 0.30
    while follower has lagged (<0.12%). Follower order placed at bar t Open.
    """

    name: str = "CrossSectorLeadLagTrigger"
    PAIR_MAP = {
        "PSUBNKBEES": "BANKBEES",
        "PSUBNKBEES.NS": "BANKBEES.NS",
        "JUNIORBEES": "NIFTYBEES",
        "JUNIORBEES.NS": "NIFTYBEES.NS",
        "SILVERBEES": "GOLDBEES",
        "SILVERBEES.NS": "GOLDBEES.NS",
    }

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        leader_col = "Leader_Close" if "Leader_Close" in df.columns else None
        if leader_col is None:
            return conviction

        leader_close = df[leader_col]
        leader_ofi = df["Leader_OFI_5"] if "Leader_OFI_5" in df.columns else extract_ofi_proxy(df)
        leader_ret_1 = leader_close.pct_change()
        follower_ret_1 = df["Close"].pct_change()

        for i in range(1, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("09:20" <= t_str <= "14:35"):
                continue

            # Strict t-1 causal signal
            l_ret = leader_ret_1.iloc[i - 1]
            l_ofi = leader_ofi.iloc[i - 1]
            f_ret = follower_ret_1.iloc[i - 1]

            if l_ret > 0.0035 and l_ofi > 0.30 and f_ret < 0.0012:
                conviction.iloc[i] = min(1.0, 0.65 + (l_ret - 0.0035) * 50.0)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            if conv.iloc[i] > 0.0:
                spread = df["Spread_INR"].iloc[i] if "Spread_INR" in df.columns else 0.05
                limit_px = float(df["Open"].iloc[i] + 0.5 * spread)
                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "order_type": "LIMIT",
                            "limit_price": round(limit_px, 2),
                            "regime": "LEAD_LAG",
                        },
                    )
                )
        return signals


class SectorPEADInstitutionalDriftTrigger(StructuralTrigger):
    """Trigger 4: Post-Earnings / Macro Announcement Institutional Drift (09:40-14:30).

    Detects drift via VOLUME_ZSCORE_LAGGED >= 1.8, LOG_RET_20 >= 0.006,
    VARIANCE_RATIO_20 >= 1.15, pulling back toward rising VWAP. Strictly MIS.
    """

    name: str = "SectorPEADInstitutionalDriftTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        vr = extract_variance_ratio(df)
        vwap = extract_session_vwap(df)
        if "LOG_RET_20" in df.columns:
            log_ret_20 = df["LOG_RET_20"]
        else:
            log_ret_20 = np.log(df["Close"] / df["Close"].shift(20).replace(0.0, np.nan)).fillna(0.0)

        if "VOLUME_ZSCORE_LAGGED" in df.columns:
            vol_z = df["VOLUME_ZSCORE_LAGGED"]
        else:
            vol = df["Volume"]
            vol_mean = vol.shift(1).rolling(20, min_periods=5).mean()
            vol_std = vol.shift(1).rolling(20, min_periods=5).std().replace(0.0, np.nan)
            vol_z = ((vol.shift(1) - vol_mean) / vol_std).fillna(0.0)

        for i in range(1, len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("09:40" <= t_str <= "14:30"):
                continue
            if vol_z.iloc[i] < 1.8 or log_ret_20.iloc[i] < 0.006 or vr.iloc[i] < 1.15:
                continue

            vwap_rising = vwap.iloc[i] > vwap.iloc[i - 1]
            pullback_vwap = (df["Close"].iloc[i] >= vwap.iloc[i]) and (
                (df["Close"].iloc[i] - vwap.iloc[i]) / vwap.iloc[i] <= 0.003
                or df["Low"].iloc[i] <= vwap.iloc[i] * 1.002
            )

            if vwap_rising and pullback_vwap:
                conviction.iloc[i] = min(1.0, 0.70 + (log_ret_20.iloc[i] - 0.006) * 30.0)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            if conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "PEAD_INSTITUTIONAL_DRIFT",
                            "stop_price": float(df["Low"].iloc[i] * 0.997),
                        },
                    )
                )
        return signals


# ---------------------------------------------------------------------------
# BUCKET 2: Mean-Reversion, Stop-Sweep & Absorption (VR <= 1.05)
# ---------------------------------------------------------------------------


class INAVSpreadMeanReversionTrigger(StructuralTrigger):
    """Trigger 5: ETF iNAV Tracking Spread Discount Mean-Reversion (09:30-14:35).

    Detects ETF discount (SPREAD_ZSCORE_20 <= -1.8, SPREAD_BPS <= -20).
    Dynamic ATR profit target. Thesis invalidation exit if underlying breaks 3-bar low.
    """

    name: str = "INAVSpreadMeanReversionTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        vr = extract_variance_ratio(df)
        spread_bps, spread_zscore = extract_spread_metrics(df)

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("09:30" <= t_str <= "14:35"):
                continue
            if vr.iloc[i] > 1.05:
                continue

            if spread_zscore.iloc[i] <= -1.8 and spread_bps.iloc[i] <= -20.0:
                excess = abs(spread_zscore.iloc[i] - (-1.8))
                conviction.iloc[i] = min(1.0, 0.60 + excess * 0.15)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []
        theo_col = "Theoretical_Close" if "Theoretical_Close" in df.columns else None
        theo_3_low = df[theo_col].shift(1).rolling(3).min() if theo_col else None

        for i in range(len(df)):
            # Check thesis invalidation exit
            if theo_3_low is not None and not np.isnan(theo_3_low.iloc[i]):
                if df[theo_col].iloc[i] < theo_3_low.iloc[i]:
                    signals.append(
                        TriggerSignal(
                            timestamp=df.index[i],
                            symbol=symbol,
                            action=TriggerAction.EXIT_ALL,
                            conviction=1.0,
                            trigger_name=self.name,
                            metadata={"reason": "underlying_broke_3_bar_low"},
                        )
                    )

            if conv.iloc[i] > 0.0:
                atr = df["ATR_14"].iloc[i] if "ATR_14" in df.columns else df["Close"].iloc[i] * 0.003
                target_px = float(df["Close"].iloc[i] + 1.5 * atr)
                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "INAV_MEAN_REVERSION",
                            "target_price": target_px,
                            "stop_price": float(df["Low"].iloc[i] * 0.997),
                        },
                    )
                )
        return signals


class ClosingVWAPSnapbackTrigger(StructuralTrigger):
    """Trigger 6: Closing VWAP Snapback (15:00-15:12 IST).

    Buys artificial discount from retail broker MIS forced liquidation on trend days
    (LOG_RET_20 > 0.004, SPREAD_ZSCORE_20 <= -1.5). Exits at 15:24 or hands off to Trigger 5.
    """

    name: str = "ClosingVWAPSnapbackTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        _, spread_zscore = extract_spread_metrics(df)
        if "LOG_RET_20" in df.columns:
            log_ret_20 = df["LOG_RET_20"]
        else:
            log_ret_20 = np.log(df["Close"] / df["Close"].shift(20).replace(0.0, np.nan)).fillna(0.0)

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("15:00" <= t_str <= "15:12"):
                continue

            if log_ret_20.iloc[i] > 0.004 and spread_zscore.iloc[i] <= -1.5:
                excess = abs(spread_zscore.iloc[i] - (-1.5))
                conviction.iloc[i] = min(1.0, 0.75 + excess * 0.10)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            # Scheduled terminal exit at 15:24
            if t_str == "15:24":
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.EXIT_ALL,
                        conviction=1.0,
                        trigger_name=self.name,
                        metadata={"reason": "closing_vwap_terminal_exit_1524"},
                    )
                )

            if conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "VWAP_SNAPBACK",
                            "exit_time": "15:24",
                        },
                    )
                )
        return signals


class DIISIPVWAPAbsorptionTrigger(StructuralTrigger):
    """Trigger 7: DII SIP VWAP Absorption (10:00-14:20 IST).

    Active when TURN_OF_MONTH_SIP_WINDOW == 1.0 (days 1..7 and 28..31).
    Intraday pullback to lower Bollinger Band / VWAP (BOLLINGER_BAND_PCT_20 <= 0.22)
    with OFI_5 > 0.20. Strictly MIS.
    """

    name: str = "DIISIPVWAPAbsorptionTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        ofi = extract_ofi_proxy(df)

        # Bollinger Band %b
        if "BOLLINGER_BAND_PCT_20" in df.columns:
            bb_pct = df["BOLLINGER_BAND_PCT_20"]
        else:
            mid = df["Close"].rolling(20, min_periods=5).mean()
            std = df["Close"].rolling(20, min_periods=5).std().replace(0.0, np.nan)
            lower = mid - 2.0 * std
            upper = mid + 2.0 * std
            bb_pct = ((df["Close"] - lower) / (upper - lower + 1e-6)).fillna(0.5)

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("10:00" <= t_str <= "14:20"):
                continue

            is_sip_window = (ts.day <= 7) or (ts.day >= 28)
            if "TURN_OF_MONTH_SIP_WINDOW" in df.columns:
                is_sip_window = bool(df["TURN_OF_MONTH_SIP_WINDOW"].iloc[i] == 1.0)

            if not is_sip_window:
                continue

            if bb_pct.iloc[i] <= 0.22 and ofi.iloc[i] > 0.20:
                conviction.iloc[i] = min(1.0, 0.65 + float(ofi.iloc[i]) * 0.35)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            if conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "DII_SIP_ABSORPTION",
                        },
                    )
                )
        return signals


class ICTUnicornStopSweepTrigger(StructuralTrigger):
    """Trigger 8: ICT Unicorn Stop Sweep Reclaim (09:30-13:30 IST).

    STOP_SWEEP_RECLAIM == 1.0 (pierced prior 10-bar low and closed back above)
    AND (FVG_BULLISH_DRIFT > 0 or SPREAD_ZSCORE_20 <= -1.2) AND OFI_5 > 0.30.
    Tight initial stop just below sweep wick. 3.3:1 reward-to-risk ratio. Strictly MIS.
    """

    name: str = "ICTUnicornStopSweepTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        ofi = extract_ofi_proxy(df)
        _, spread_z = extract_spread_metrics(df)

        prior_10_low = df["Low"].shift(1).rolling(10, min_periods=10).min()
        fvg_drift = (
            df["FVG_BULLISH_DRIFT"]
            if "FVG_BULLISH_DRIFT" in df.columns
            else (df["Low"] - df["High"].shift(2)).clip(lower=0.0)
        )

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("09:30" <= t_str <= "13:30"):
                continue
            if np.isnan(prior_10_low.iloc[i]):
                continue

            sweep_reclaim = (df["Low"].iloc[i] < prior_10_low.iloc[i]) and (
                df["Close"].iloc[i] > prior_10_low.iloc[i]
            )
            has_fvg_or_discount = (fvg_drift.iloc[i] > 0.0) or (spread_z.iloc[i] <= -1.2)
            has_absorption = ofi.iloc[i] > 0.30

            if sweep_reclaim and has_fvg_or_discount and has_absorption:
                conviction.iloc[i] = min(1.0, 0.70 + float(ofi.iloc[i]) * 0.30)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            if conv.iloc[i] > 0.0:
                stop_px = float(df["Low"].iloc[i] - 0.05)
                entry_px = float(df["Close"].iloc[i])
                risk = max(0.05, entry_px - stop_px)
                target_px = float(entry_px + 3.3 * risk)

                signals.append(
                    TriggerSignal(
                        timestamp=df.index[i],
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "ICT_UNICORN_SWEEP",
                            "stop_price": stop_px,
                            "target_price": target_px,
                            "risk_reward_ratio": 3.3,
                        },
                    )
                )
        return signals


# ---------------------------------------------------------------------------
# BUCKET 3: Macro, Bond & All-Weather Overnight
# ---------------------------------------------------------------------------


class AfternoonBondEuropeTWAPTrigger(StructuralTrigger):
    """Trigger 9: Afternoon Bond / Europe TWAP (13:55-14:40 IST).

    Evaluates bond yield easing and EU market open momentum.
    CRITICAL FIX: Exits at 14:45 IST strictly before the 15:00-15:15 retail MIS dump!
    """

    name: str = "AfternoonBondEuropeTWAPTrigger"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        yield_ret = (
            df["MACRO_YIELD_RET_1"]
            if "MACRO_YIELD_RET_1" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        log_ret_5 = np.log(df["Close"] / df["Close"].shift(5).replace(0.0, np.nan)).fillna(0.0)

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")
            if not ("13:55" <= t_str <= "14:40"):
                continue

            bond_easing = yield_ret.iloc[i] < 0.0
            momentum_positive = log_ret_5.iloc[i] > 0.0005

            if bond_easing or momentum_positive:
                conviction.iloc[i] = min(1.0, 0.65 + max(0.0, log_ret_5.iloc[i]) * 50.0)

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")

            # CRITICAL FIX: Terminal exit at 14:45 strictly before retail MIS dump!
            if t_str == "14:45":
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.EXIT_ALL,
                        conviction=1.0,
                        trigger_name=self.name,
                        metadata={"reason": "exit_before_1500_retail_mis_dump"},
                    )
                )

            if conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "MIS",
                            "regime": "AFTERNOON_BOND_EUROPE",
                            "exit_time": "14:45",
                        },
                    )
                )
        return signals


class OvernightContinuationTrigger(StructuralTrigger):
    """Trigger 10: All-Weather Tier-1 Overnight Continuation (15:20-15:25 IST).

    Whitelist: BANKBEES, NIFTYBEES, ITBEES, GOLDBEES (daily turnover >= 10 Cr).
    Regime: CNC (Cash Delivery).
    Continuation Gate at 09:15 skips selling when morning momentum >= 0.50.
    """

    name: str = "OvernightContinuationTrigger"
    TIER1_WHITELIST = {
        "BANKBEES",
        "BANKBEES.NS",
        "NIFTYBEES",
        "NIFTYBEES.NS",
        "ITBEES",
        "ITBEES.NS",
        "GOLDBEES",
        "GOLDBEES.NS",
    }

    def evaluate(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        conviction = pd.Series(0.0, index=df.index)
        if symbol not in self.TIER1_WHITELIST:
            return conviction

        spread_bps, _ = extract_spread_metrics(df)
        vwap = extract_session_vwap(df)

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")

            # Entry window: 15:20 to 15:25 IST
            if "15:20" <= t_str <= "15:25":
                if spread_bps.iloc[i] <= 50.0 and df["Close"].iloc[i] >= vwap.iloc[i] * 0.999:
                    conviction.iloc[i] = 0.85
            # Morning continuation check bar: 09:15
            elif t_str == "09:15":
                day_bars = df[df.index.normalize() == ts.normalize()]
                if len(day_bars) > 0 and df["Close"].iloc[i] >= df["Open"].iloc[i]:
                    conviction.iloc[i] = 0.60  # Continuation bullish momentum >= 0.50
                else:
                    conviction.iloc[i] = 0.20  # Weak morning gap, fail continuation

        return conviction

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[TriggerSignal]:
        if symbol not in self.TIER1_WHITELIST:
            return []

        conv = self.evaluate(df, symbol)
        signals: List[TriggerSignal] = []

        for i in range(len(df)):
            ts = df.index[i]
            t_str = ts.strftime("%H:%M")

            if "15:20" <= t_str <= "15:25" and conv.iloc[i] > 0.0:
                signals.append(
                    TriggerSignal(
                        timestamp=ts,
                        symbol=symbol,
                        action=TriggerAction.BUY,
                        conviction=float(conv.iloc[i]),
                        trigger_name=self.name,
                        metadata={
                            "product_type": "CNC",
                            "regime": "ALL_WEATHER_OVERNIGHT",
                            "continuation_gate": True,
                        },
                    )
                )
            elif t_str == "09:15":
                if conv.iloc[i] >= 0.50:
                    signals.append(
                        TriggerSignal(
                            timestamp=ts,
                            symbol=symbol,
                            action=TriggerAction.HOLD,
                            conviction=float(conv.iloc[i]),
                            trigger_name=self.name,
                            metadata={"reason": "continuation_gate_active_skip_sell"},
                        )
                    )
                else:
                    signals.append(
                        TriggerSignal(
                            timestamp=ts,
                            symbol=symbol,
                            action=TriggerAction.EXIT_ALL,
                            conviction=1.0,
                            trigger_name=self.name,
                            metadata={"reason": "continuation_gate_momentum_failed"},
                        )
                    )
        return signals
