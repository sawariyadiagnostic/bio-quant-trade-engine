"""Muscle 20: Real-Time Observability & Concept Drift Dashboard.

Monitors live Muscle 7 feature matrices against training Lakehouse baselines
using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics,
tracks live vs. Muscle 13 OOS return divergence, and automatically trips
Muscle 18's Kill Switch upon severe concept drift or slippage anomalies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from ravana.execution.router import RoutingReceipt
from ravana.risk.circuit_breaker import CircuitBreakerKillSwitch


@dataclass(frozen=True)
class FeatureDriftMetric:
    """Drift telemetry for a single stationary feature from Muscle 7."""

    feature_name: str
    psi: float
    ks_stat: float
    ks_p_value: float
    is_drifted: bool


@dataclass(frozen=True)
class SystemHealthReport:
    """Comprehensive observability snapshot output by Muscle 20."""

    drifted_features: List[str]
    max_psi: float
    mean_psi: float
    mean_slippage_anomaly_bps: float
    live_vs_oos_sharpe_gap: float
    kill_switch_triggered: bool
    alert_messages: List[str]
    feature_metrics: Dict[str, FeatureDriftMetric]


class ObservabilityDriftMonitor:
    """Detects feature distribution drift and execution degradation in real time."""

    def __init__(
        self,
        psi_threshold: float = 0.25,            # Standard institutional PSI drift threshold
        severe_psi_kill_threshold: float = 0.50,
        max_drifted_feature_ratio: float = 0.35,
        max_slippage_anomaly_bps: float = 15.0,
        min_sample_size: int = 30,              # Guard against early-morning small-sample artifacts
    ) -> None:
        self.psi_threshold = psi_threshold
        self.severe_psi_kill_threshold = severe_psi_kill_threshold
        self.max_drifted_feature_ratio = max_drifted_feature_ratio
        self.max_slippage_anomaly_bps = max_slippage_anomaly_bps
        self.min_sample_size = max(10, int(min_sample_size))

    @staticmethod
    def compute_psi(
        reference: np.ndarray, current: np.ndarray, bins: int = 10, min_samples: int = 30
    ) -> float:
        """Compute Population Stability Index (PSI) with strict sample size protection."""
        ref_clean = reference[np.isfinite(reference)]
        cur_clean = current[np.isfinite(current)]
        if len(ref_clean) < min_samples or len(cur_clean) < min_samples:
            return 0.0

        quantiles = np.linspace(0, 100, bins + 1)
        breakpoints = np.percentile(ref_clean, quantiles)
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0

        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
        cur_counts, _ = np.histogram(cur_clean, bins=breakpoints)

        ref_pct = np.maximum(1e-4, ref_counts / float(len(ref_clean)))
        cur_pct = np.maximum(1e-4, cur_counts / float(len(cur_clean)))

        psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(max(0.0, psi_val))

    def evaluate_system_health(
        self,
        reference_features_df: pd.DataFrame,
        live_features_df: pd.DataFrame,
        routing_receipt: Optional[RoutingReceipt] = None,
        live_sharpe: float = 0.0,
        expected_oos_sharpe: float = 0.0,
        kill_switch: Optional[CircuitBreakerKillSwitch] = None,
    ) -> SystemHealthReport:
        """Audit feature drift, slippage anomalies, and OOS tracking error."""
        common_cols = [c for c in reference_features_df.columns if c in live_features_df.columns]
        feature_metrics: Dict[str, FeatureDriftMetric] = {}
        drifted_cols: List[str] = []
        psi_values: List[float] = []

        is_warmed_up = (
            len(reference_features_df) >= self.min_sample_size
            and len(live_features_df) >= self.min_sample_size
        )

        for col in common_cols:
            ref_arr = reference_features_df[col].to_numpy(dtype=np.float64)
            cur_arr = live_features_df[col].to_numpy(dtype=np.float64)

            if is_warmed_up:
                psi = self.compute_psi(ref_arr, cur_arr, min_samples=self.min_sample_size)
                clean_ref = ref_arr[np.isfinite(ref_arr)]
                clean_cur = cur_arr[np.isfinite(cur_arr)]
                if len(clean_ref) > 0 and len(clean_cur) > 0:
                    ks_res = stats.ks_2samp(clean_ref, clean_cur)
                    ks_stat = float(ks_res.statistic)
                    ks_pval = float(ks_res.pvalue)
                else:
                    ks_stat, ks_pval = 0.0, 1.0
            else:
                psi, ks_stat, ks_pval = 0.0, 0.0, 1.0

            is_drifted = psi >= self.psi_threshold
            if is_drifted:
                drifted_cols.append(col)
            psi_values.append(psi)

            feature_metrics[col] = FeatureDriftMetric(
                feature_name=col,
                psi=round(psi, 4),
                ks_stat=round(ks_stat, 4),
                ks_p_value=round(ks_pval, 6),
                is_drifted=is_drifted,
            )

        max_psi = max(psi_values) if psi_values else 0.0
        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        drift_ratio = len(drifted_cols) / float(max(1, len(common_cols)))

        mean_slip_anom = 0.0
        if routing_receipt and routing_receipt.audit_records:
            mean_slip_anom = float(
                np.mean([a.slippage_anomaly_bps for a in routing_receipt.audit_records])
            )

        sharpe_gap = live_sharpe - expected_oos_sharpe
        alerts: List[str] = []
        should_kill = False

        if is_warmed_up:
            if drift_ratio >= self.max_drifted_feature_ratio or max_psi >= self.severe_psi_kill_threshold:
                msg = f"Severe concept drift detected: {len(drifted_cols)}/{len(common_cols)} features drifted (max PSI={max_psi:.2f})."
                alerts.append(msg)
                should_kill = True

        if mean_slip_anom >= self.max_slippage_anomaly_bps:
            msg = f"Execution slippage anomaly ({mean_slip_anom:.1f} bps) exceeded {self.max_slippage_anomaly_bps:.1f} bps limit."
            alerts.append(msg)
            should_kill = True

        if should_kill and kill_switch is not None:
            kill_switch.trigger_kill_switch("; ".join(alerts))

        return SystemHealthReport(
            drifted_features=drifted_cols,
            max_psi=round(max_psi, 4),
            mean_psi=round(mean_psi, 4),
            mean_slippage_anomaly_bps=round(mean_slip_anom, 2),
            live_vs_oos_sharpe_gap=round(sharpe_gap, 4),
            kill_switch_triggered=should_kill,
            alert_messages=alerts,
            feature_metrics=feature_metrics,
        )
