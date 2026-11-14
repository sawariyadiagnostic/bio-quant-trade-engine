"""Muscle 4: Data Quality & Cleaning Engine orchestrator."""
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd

from ravana.data.cleaner.config import CleanerConfig
from ravana.data.cleaner.dataset import CleanedDataset
from ravana.data.cleaner.grid_reindexer import TimeGridReindexer
from ravana.data.cleaner.spike_filter import FlashSpikeFilter
from ravana.data.cleaner.spread_aligner import ETFBUSpreadAligner
from ravana.data.ingestion.dataset import IngestedDataset


class DataCleaningEngine:
    """Orchestrates time grid regularization, spike sanitization, and ETF tracking spread alignment.
    
    Transforms raw IngestedDataset contracts into hardened, contiguous CleanedDataset
    instances ready for rolling feature calculations and multi-timeframe resampling.
    """

    def __init__(self, config: Optional[CleanerConfig] = None) -> None:
        self.config = config or CleanerConfig.standard()
        self.spike_filter = FlashSpikeFilter(
            window=self.config.spike_window,
            threshold_std=self.config.spike_threshold_std,
            clamp_spikes=self.config.clamp_spikes,
            use_volume_gate=self.config.use_volume_gate,
            volume_surge_multiplier=self.config.volume_surge_multiplier,
            volume_surge_expansion=self.config.volume_surge_expansion,
        )
        self.spread_aligner = ETFBUSpreadAligner(
            rolling_window=self.config.spread_rolling_window,
            contract_multiplier=self.config.contract_multiplier,
        )

    def clean(self, dataset: IngestedDataset) -> CleanedDataset:
        """Execute full cleaning pipeline on an IngestedDataset with strict causality.
        
        Args:
            dataset: Immutable IngestedDataset from Muscle 3.
            
        Returns:
            Hardened, contiguous, and validated CleanedDataset.
        """
        raw_bars = dataset.bar_count
        reindexer = TimeGridReindexer(exchange=dataset.exchange)

        clean_df, synthetic_bars = self._clean_series(dataset.data, dataset.interval, reindexer)
        clean_df, spikes_detected = self._filter_spikes(clean_df)

        clean_benchmark: Optional[pd.DataFrame] = None
        if dataset.benchmark_data is not None and not dataset.benchmark_data.empty:
            bench_in = dataset.benchmark_data.copy()
            if clean_df.index.tz is not None and bench_in.index.tz is not None and bench_in.index.tz != clean_df.index.tz:
                bench_in.index = bench_in.index.tz_convert(clean_df.index.tz)
            clean_benchmark, _ = self._clean_series(bench_in, dataset.interval, reindexer)
            clean_benchmark, _ = self._filter_spikes(clean_benchmark)

        spread_df: Optional[pd.DataFrame] = None
        if self.config.align_spread and clean_benchmark is not None:
            spread_df = self.spread_aligner.align(
                clean_df, clean_benchmark, ticker=dataset.ticker
            )

        clean_macros = self._clean_macro_data(dataset.macro_data, clean_df.index)

        quality_report = self._build_quality_report(
            raw_bars=raw_bars,
            cleaned_bars=len(clean_df),
            synthetic_bars=synthetic_bars,
            spikes=spikes_detected,
            spread_df=spread_df,
        )

        return CleanedDataset(
            ticker=dataset.ticker,
            interval=dataset.interval,
            exchange=dataset.exchange,
            data=clean_df,
            benchmark_ticker=dataset.benchmark_ticker,
            benchmark_data=clean_benchmark,
            spread_data=spread_df,
            macro_data=clean_macros,
            quality_report=quality_report,
            metadata={**dataset.metadata, "cleaning_engine_version": "1.1.0"},
        )

    def _clean_series(
        self, df: pd.DataFrame, interval: str, reindexer: TimeGridReindexer
    ) -> tuple[pd.DataFrame, int]:
        """Apply grid reindexing if enabled."""
        if self.config.fill_gaps:
            return reindexer.reindex(df, interval)
        out = df.copy()
        if "Is_Synthetic" not in out.columns:
            out["Is_Synthetic"] = False
        return out, 0

    def _filter_spikes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Apply spike sanitization if enabled."""
        if self.config.spike_filter_enabled:
            return self.spike_filter.filter(df)
        out = df.copy()
        if "Is_Spike" not in out.columns:
            out["Is_Spike"] = False
        return out, 0

    def _clean_macro_data(
        self,
        macro_dict: Dict[str, pd.DataFrame],
        target_index: pd.DatetimeIndex,
    ) -> Dict[str, pd.DataFrame]:
        """Causally align macro series to the primary session index via backward as-of merge."""
        cleaned_macros: Dict[str, pd.DataFrame] = {}
        target_df = pd.DataFrame(index=target_index.sort_values())

        for ticker, m_df in macro_dict.items():
            if m_df.empty or "Close" not in m_df.columns:
                continue

            m_copy = m_df.copy()
            if (m_copy.index.time == pd.Timestamp("00:00:00").time()).all():
                # Seal daily macro bars at 16:00 local exchange close so US close (16:00 EST/EDT = 01:30/02:30 IST)
                # is causally available before the 09:15 IST NSE market open.
                m_copy.index = m_copy.index + pd.Timedelta(hours=16)

            if m_copy.index.tz != target_index.tz:
                if m_copy.index.tz is None and target_index.tz is not None:
                    m_copy = m_copy.tz_localize(target_index.tz)
                elif m_copy.index.tz is not None and target_index.tz is None:
                    m_copy = m_copy.tz_localize(None)
                else:
                    m_copy = m_copy.tz_convert(target_index.tz)

            aligned_m = pd.merge_asof(
                target_df,
                m_copy[["Close"]].sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
            )
            if aligned_m["Close"].isna().any():
                first_val = aligned_m["Close"].dropna().iloc[0] if not aligned_m["Close"].dropna().empty else 1.0
                aligned_m = aligned_m.bfill().fillna(first_val)
            cleaned_macros[ticker] = aligned_m
        return cleaned_macros

    def _build_quality_report(
        self,
        raw_bars: int,
        cleaned_bars: int,
        synthetic_bars: int,
        spikes: int,
        spread_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Generate quantitative quality audit metrics."""
        report: Dict[str, Any] = {
            "raw_bars": raw_bars,
            "cleaned_bars": cleaned_bars,
            "synthetic_bars_filled": synthetic_bars,
            "synthetic_bar_pct": (synthetic_bars / max(1, cleaned_bars)) * 100.0,
            "spikes_sanitized": spikes,
            "gap_integrity_passed": cleaned_bars >= raw_bars,
        }
        if spread_df is not None and not spread_df.empty:
            report["spread_mean_bps"] = float(spread_df["Basis_Bps"].mean())
            report["spread_std_bps"] = float(spread_df["Basis_Bps"].std())
            report["multiplier"] = float(spread_df["Multiplier"].iloc[-1])
        return report
