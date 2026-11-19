"""Bounded AST Formula Modulator for RAVANA Structural Strategies.

Evaluates symbolic AST formula expressions to modulate structural trigger conviction.
Enforces:
1. Input Guard: Accepts only stationary features from FeatureName vocabulary.
2. Dynamic Safe Epsilon Normalization:
   z = clip((x - mu) / (sigma + max(1e-4, 0.10 * ATR_14)), -3.0, 3.0)
3. Sigmoid Multiplier:
   multiplier = 0.5 + 1.0 / (1.0 + exp(-z)) in [0.5x, 1.5x]
4. Rolling Variance Sentinel:
   Computes rolling 20-bar variance of raw AST output. If var < 1e-4 for >60 bars,
   sets stagnation_alert = True and defaults multiplier to 1.0x.
5. Zero Ghost Trades:
   When underlying trigger conviction is 0.0, modulated output is strictly 0.0.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ravana.domain.formula_ast import ASTEvaluator, ASTNode
from ravana.features.config import FeatureName
from ravana.strategies.base import TriggerSignal

logger = logging.getLogger(__name__)


class BoundedASTModulator:
    """Modulates structural trigger conviction using bounded symbolic AST formulas."""

    def __init__(
        self,
        variance_window: int = 20,
        zscore_window: int = 20,
        stagnation_threshold: float = 1e-4,
        stagnation_bars_limit: int = 60,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5,
        clip_z_min: float = -3.0,
        clip_z_max: float = 3.0,
    ) -> None:
        """Initialize BoundedASTModulator with quantitative safeguard thresholds."""
        self.variance_window = variance_window
        self.zscore_window = zscore_window
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_bars_limit = stagnation_bars_limit
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.clip_z_min = clip_z_min
        self.clip_z_max = clip_z_max

        # Sentinel state trackers
        self.stagnation_alert: bool = False
        self.stagnant_bar_count: int = 0

        # Build stationary vocabulary guard set
        self.allowed_features: Set[str] = self._build_stationary_feature_vocabulary()

    @staticmethod
    def _build_stationary_feature_vocabulary() -> Set[str]:
        """Construct canonical set of allowed stationary feature identifiers."""
        vocab: Set[str] = set()
        for f in FeatureName:
            vocab.add(f.value)
            vocab.add(f.name)
            vocab.add(f.value.lower())
            vocab.add(f.name.lower())
        return vocab

    def validate_ast(self, node: ASTNode) -> None:
        """Enforce strict Input Guard: verify that all feature nodes are stationary.

        Raises:
            ValueError: If any feature in the AST subtree is non-stationary.
        """
        all_nodes = node.get_all_subnodes()
        for sub in all_nodes:
            if sub.op != "FEATURE":
                continue
            feature_name = sub.val.strip()
            if feature_name not in self.allowed_features:
                raise ValueError(
                    f"Input Guard Violation: Non-stationary feature '{feature_name}' "
                    f"is not permitted in BoundedASTModulator. Permitted vocabulary: "
                    f"{sorted(f.value for f in FeatureName)}"
                )

    def is_valid_ast(self, node: ASTNode) -> bool:
        """Check if an AST formula conforms to the stationary Input Guard."""
        try:
            self.validate_ast(node)
            return True
        except ValueError:
            return False

    def _extract_atr14(self, df: pd.DataFrame) -> pd.Series:
        """Extract or causally estimate 14-bar ATR for safe epsilon scaling."""
        if "ATR_14" in df.columns:
            return df["ATR_14"].fillna(0.0).astype(float)

        if "ATR_14_NORM" in df.columns and "Close" in df.columns:
            return (df["ATR_14_NORM"] * df["Close"]).fillna(0.0).astype(float)

        if all(c in df.columns for c in ("High", "Low", "Close")):
            high, low, close = df["High"], df["Low"], df["Close"]
            prev_close = close.shift(1).fillna(close)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            return tr.rolling(14, min_periods=1).mean().fillna(0.0).astype(float)

        if "Close" in df.columns:
            return (df["Close"] * 0.005).fillna(1.0).astype(float)

        return pd.Series(1.0, index=df.index)

    def evaluate_ast_raw(self, node: ASTNode, df: pd.DataFrame) -> np.ndarray:
        """Validate input guard and evaluate AST expression against DataFrame."""
        self.validate_ast(node)
        raw = ASTEvaluator.eval_node(node, df)
        return np.nan_to_num(raw, nan=0.0, posinf=self.clip_z_max, neginf=self.clip_z_min)

    def _compute_stagnation_mask(self, raw_series: pd.Series) -> Tuple[np.ndarray, bool]:
        """Evaluate rolling variance sentinel and return boolean stagnation mask."""
        rolling_var = (
            raw_series.rolling(self.variance_window, min_periods=1)
            .var()
            .fillna(0.0)
            .to_numpy()
        )
        is_low_var = rolling_var < self.stagnation_threshold

        n = len(raw_series)
        consecutive_counts = np.zeros(n, dtype=np.int32)
        current_streak = 0
        for i in range(n):
            if is_low_var[i]:
                current_streak += 1
            else:
                current_streak = 0
            consecutive_counts[i] = current_streak

        stagnant_mask = consecutive_counts > self.stagnation_bars_limit
        active_alert = bool(stagnant_mask[-1]) if n > 0 else False
        self.stagnation_alert = active_alert
        self.stagnant_bar_count = int(consecutive_counts[-1]) if n > 0 else 0
        return stagnant_mask, active_alert

    def compute_multipliers(
        self,
        node: Optional[ASTNode],
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, bool]:
        """Compute bounded [0.5x, 1.5x] multiplier series with safe epsilon and sentinel."""
        if len(df) == 0:
            return pd.Series(dtype=float), False

        if node is None:
            self.stagnation_alert = False
            self.stagnant_bar_count = 0
            return pd.Series(1.0, index=df.index), False

        raw_arr = self.evaluate_ast_raw(node, df)
        raw_s = pd.Series(raw_arr, index=df.index)

        # 1. Rolling Variance Sentinel Check
        stagnant_mask, alert = self._compute_stagnation_mask(raw_s)

        # 2. Dynamic Safe Epsilon Z-Score Normalization
        mu = raw_s.rolling(self.zscore_window, min_periods=1).mean().fillna(0.0)
        sigma = raw_s.rolling(self.zscore_window, min_periods=1).std().fillna(0.0)
        atr14 = self._extract_atr14(df)
        close_px = df["Close"] if "Close" in df.columns else pd.Series(100.0, index=df.index)
        rel_atr = atr14 / np.maximum(1.0, close_px)

        safe_epsilon = np.maximum(1e-4, 0.10 * rel_atr)
        z = np.clip(
            (raw_s - mu) / (sigma + safe_epsilon),
            self.clip_z_min,
            self.clip_z_max,
        )

        # 3. Sigmoid Multiplier Mapping to [0.5, 1.5]
        multipliers = self.min_multiplier + 1.0 / (1.0 + np.exp(-z))

        # 4. Default to 1.0x on Stagnation Alert
        multipliers[stagnant_mask] = 1.0
        return multipliers, alert

    def compute_bar_multiplier(
        self,
        node: Optional[ASTNode],
        df_window: pd.DataFrame,
    ) -> Tuple[float, bool]:
        """Compute multiplier for the latest bar given historical DataFrame window."""
        multipliers, alert = self.compute_multipliers(node, df_window)
        if len(multipliers) == 0:
            return 1.0, False
        return float(multipliers.iloc[-1]), alert

    def modulate_conviction(self, base_conviction: float, multiplier: float) -> float:
        """Modulate scalar conviction: strictly 0.0 if base is 0.0 (Zero Ghost Trades)."""
        if base_conviction <= 0.0:
            return 0.0
        clamped_mult = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        modulated = base_conviction * clamped_mult
        return float(max(0.0, min(1.0, modulated)))

    def modulate_series(
        self,
        base_convictions: pd.Series,
        node: Optional[ASTNode],
        df: pd.DataFrame,
    ) -> pd.Series:
        """Modulate series of convictions enforcing Zero Ghost Trades."""
        if len(base_convictions) == 0:
            return pd.Series(dtype=float)

        multipliers, _ = self.compute_multipliers(node, df)
        modulated = (base_convictions * multipliers).fillna(0.0)

        # Enforce Zero Ghost Trades strictly
        zero_mask = base_convictions == 0.0
        modulated[zero_mask] = 0.0
        return modulated.clip(0.0, 1.0)

    def modulate_signals(
        self,
        signals: List[TriggerSignal],
        node: Optional[ASTNode],
        df: pd.DataFrame,
    ) -> List[TriggerSignal]:
        """Modulate discrete TriggerSignal convictions and record modulation metadata."""
        if not signals:
            return []

        multipliers, alert = self.compute_multipliers(node, df)
        modulated_signals: List[TriggerSignal] = []

        for sig in signals:
            mult = 1.0
            if sig.timestamp in multipliers.index:
                mult = float(multipliers.loc[sig.timestamp])
            elif len(multipliers) > 0:
                mult = float(multipliers.iloc[-1])

            new_conviction = self.modulate_conviction(sig.conviction, mult)
            meta: Dict[str, Any] = dict(sig.metadata)
            meta["ast_multiplier"] = mult
            meta["stagnation_alert"] = alert

            modulated_signals.append(
                TriggerSignal(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    action=sig.action,
                    conviction=new_conviction,
                    trigger_name=sig.trigger_name,
                    metadata=meta,
                )
            )

        return modulated_signals
