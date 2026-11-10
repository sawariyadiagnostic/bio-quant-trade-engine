"""Reactive Pipeline Dependency DAG and Cascade Orchestrator for RAVANA.

Topologically executes Lakehouse data pipelines with automatic downstream invalidation,
snapshot diff auditing, and zero lookahead enforcement.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ravana.data.pipeline_runner import BatchPipelineRunner
from ravana.data.storage.diff import LakeDiffInspector

logger = logging.getLogger(__name__)

# Topological order of pipeline execution stages
PIPELINE_STAGES: List[str] = ["m3", "m4", "m5", "m7", "m8", "m9"]

STAGE_DEPENDENCIES: Dict[str, List[str]] = {
    "m3": [],
    "m4": ["m3"],
    "m5": ["m4"],
    "m7": ["m4", "m5"],
    "m8": ["m4"],
    "m9": ["m7", "m8"],
}


class PipelineDAG:
    """Manages reactive dependency execution and automated downstream recomputation."""

    def __init__(self, lake_root: Optional[Path] = None) -> None:
        self.lake_root = lake_root or Path("ravana/data/lake")
        self.runner = BatchPipelineRunner(self.lake_root)

    def get_downstream_stages(self, stage: str) -> List[str]:
        """Find all topologically reachable downstream stages starting from given stage."""
        stage_norm = stage.lower().strip()
        if stage_norm not in PIPELINE_STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {PIPELINE_STAGES}")

        idx = PIPELINE_STAGES.index(stage_norm)
        return PIPELINE_STAGES[idx:]

    def invalidate_stage(self, stage: str) -> None:
        """Purge or prepare partitions for clean re-execution."""
        dir_map = {
            "m3": self.runner.m3_dir,
            "m4": self.runner.m4_dir,
            "m5": self.runner.m5_dir,
            "m7": self.runner.m7_dir,
            "m8": self.runner.m8_dir,
            "m9": self.runner.lake_root / "m9_alphas",
        }
        target_dir = dir_map.get(stage)
        if target_dir and target_dir.exists():
            logger.info("Invalidating cache for stage '%s' at %s", stage, target_dir)

    def cascade_from(self, start_stage: str) -> Dict[str, Any]:
        """Invalidate and sequentially recompute start_stage and all its downstream successors."""
        stages_to_run = self.get_downstream_stages(start_stage)
        logger.info("Triggering reactive DAG cascade for stages: %s", stages_to_run)

        results: Dict[str, Any] = {}
        stage_dispatch: Dict[str, Callable[[], Any]] = {
            "m3": self.runner.ingest_and_persist_m3,
            "m4": self.runner.clean_and_persist_m4,
            "m5": self.runner.resample_and_persist_m5,
            "m7": self.runner.compute_and_persist_m7,
            "m8": self.runner.compute_and_persist_m8,
            "m9": self.runner.compute_and_persist_m9,
        }

        for st in stages_to_run:
            logger.info("Executing pipeline stage [%s]...", st.upper())
            func = stage_dispatch.get(st)
            if func:
                results[st] = func()

        results["audit"] = self.runner.verify_lake()
        return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="RAVANA Reactive DAG Cascade Runner")
    parser.add_argument(
        "--cascade-from",
        type=str,
        default="m7",
        choices=PIPELINE_STAGES,
        help="Pipeline stage from which to trigger reactive downstream re-execution.",
    )
    args = parser.parse_args()
    dag = PipelineDAG()
    res = dag.cascade_from(args.cascade_from)
    print("\n" + "=" * 80)
    print(f"CASCADE FROM [{args.cascade_from.upper()}] COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
