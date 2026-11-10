"""Unified Resampling Engine executing multi-timeframe transformation."""
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd

from ravana.data.cleaner.dataset import CleanedDataset
from ravana.data.cleaner.spread_aligner import ETFBUSpreadAligner
from ravana.data.ingestion.config import IngestionConfig
from ravana.data.resampler.config import ResamplerConfig
from ravana.data.resampler.dataset import ResampledDataset
from ravana.data.resampler.session_resampler import SessionAwareResampler


class ResamplingEngine:
    """Orchestrates multi-timeframe resampling for institutional datasets."""

    def __init__(self, config: Optional[ResamplerConfig] = None) -> None:
        self.config = config or ResamplerConfig()
        self.spread_aligner = ETFBUSpreadAligner()

    def resample_dataset(self, cleaned: CleanedDataset) -> ResampledDataset:
        """Transform a CleanedDataset into a complete multi-timeframe ResampledDataset."""
        is_intraday = IngestionConfig.is_interval_intraday(cleaned.interval)
        if is_intraday:
            return self._resample_intraday_dataset(cleaned)
        return self._resample_daily_dataset(cleaned)

    def _resample_intraday_dataset(self, cleaned: CleanedDataset) -> ResampledDataset:
        """Process intraday series across target intraday intervals."""
        timeframes: Dict[str, pd.DataFrame] = {cleaned.interval: cleaned.data}
        bench_timeframes: Dict[str, pd.DataFrame] = {}
        spread_timeframes: Dict[str, pd.DataFrame] = {}

        if cleaned.benchmark_data is not None:
            bench_timeframes[cleaned.interval] = cleaned.benchmark_data
        if cleaned.spread_data is not None:
            spread_timeframes[cleaned.interval] = cleaned.spread_data

        for tf in self.config.intraday_intervals:
            target_m = self.config.parse_interval_to_minutes(tf)
            timeframes[tf] = SessionAwareResampler.resample_intraday(
                cleaned.data, target_m, self.config.base_bar_minutes
            )
            if cleaned.benchmark_data is not None:
                bench_timeframes[tf] = SessionAwareResampler.resample_intraday(
                    cleaned.benchmark_data, target_m, self.config.base_bar_minutes
                )
            if cleaned.benchmark_ticker and tf in bench_timeframes:
                aligned_spread = self.spread_aligner.align(timeframes[tf], bench_timeframes[tf], cleaned.ticker)
                if aligned_spread is not None:
                    spread_timeframes[tf] = aligned_spread
            elif cleaned.spread_data is not None:
                spread_timeframes[tf] = SessionAwareResampler.resample_intraday(
                    cleaned.spread_data, target_m, self.config.base_bar_minutes
                )

        return ResampledDataset(
            ticker=cleaned.ticker,
            base_interval=cleaned.interval,
            exchange=cleaned.exchange,
            timeframes=timeframes,
            benchmark_ticker=cleaned.benchmark_ticker,
            benchmark_timeframes=bench_timeframes,
            spread_timeframes=spread_timeframes,
            quality_report=cleaned.quality_report,
            metadata={"resampler_config": self.config.__dict__},
        )

    def _resample_daily_dataset(self, cleaned: CleanedDataset) -> ResampledDataset:
        """Process daily series across target macro intervals."""
        timeframes: Dict[str, pd.DataFrame] = {cleaned.interval: cleaned.data}
        bench_timeframes: Dict[str, pd.DataFrame] = {}
        spread_timeframes: Dict[str, pd.DataFrame] = {}

        if cleaned.benchmark_data is not None:
            bench_timeframes[cleaned.interval] = cleaned.benchmark_data
        if cleaned.spread_data is not None:
            spread_timeframes[cleaned.interval] = cleaned.spread_data

        for tf in self.config.daily_intervals:
            freq = self.config.parse_daily_to_freq(tf)
            timeframes[tf] = SessionAwareResampler.resample_daily_to_higher(
                cleaned.data, freq, self.config.require_closed_period
            )
            if cleaned.benchmark_data is not None:
                bench_timeframes[tf] = SessionAwareResampler.resample_daily_to_higher(
                    cleaned.benchmark_data, freq, self.config.require_closed_period
                )
            if cleaned.benchmark_ticker and tf in bench_timeframes:
                aligned_spread = self.spread_aligner.align(timeframes[tf], bench_timeframes[tf], cleaned.ticker)
                if aligned_spread is not None:
                    spread_timeframes[tf] = aligned_spread
            elif cleaned.spread_data is not None:
                spread_timeframes[tf] = SessionAwareResampler.resample_daily_to_higher(
                    cleaned.spread_data, freq, self.config.require_closed_period
                )

        return ResampledDataset(
            ticker=cleaned.ticker,
            base_interval=cleaned.interval,
            exchange=cleaned.exchange,
            timeframes=timeframes,
            benchmark_ticker=cleaned.benchmark_ticker,
            benchmark_timeframes=bench_timeframes,
            spread_timeframes=spread_timeframes,
            quality_report=cleaned.quality_report,
            metadata={"resampler_config": self.config.__dict__},
        )
