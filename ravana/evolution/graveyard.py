"""Idea Graveyard for Muscle 9: Prevents Circular Rediscovery of Failed Alphas.

Implements Step 5 ('Idea Graveyard') from 'The Self-Improving System' quant framework.
Tracks all rejected or failed formula hypotheses with reasons of rejection,
ensuring the genetic algorithm and multi-agent search never re-tests dead hypotheses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class IdeaGraveyard:
    """Immutable ledger of failed alpha hypotheses and rejected challengers."""

    def __init__(self) -> None:
        self._visited_hashes: Set[str] = set()
        self._records: List[Dict[str, Any]] = []

    def _hash_formula(self, formula: str) -> str:
        """Compute deterministic SHA-256 fingerprint for standardized formula string."""
        normalized = formula.strip().replace(" ", "").upper()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def contains(self, formula: str) -> bool:
        """Check in O(1) time whether this formula has previously failed and been buried."""
        h = self._hash_formula(formula)
        return h in self._visited_hashes

    def record_rejection(
        self,
        formula: str,
        reason: str,
        ic: Optional[float] = None,
        rank_ic: Optional[float] = None,
        fitness: Optional[float] = None,
        generation: int = 0,
    ) -> None:
        """Bury a failed hypothesis into the graveyard with diagnostic attribution."""
        h = self._hash_formula(formula)
        self._visited_hashes.add(h)
        record = {
            "hash": h,
            "formula": formula,
            "reason": reason,
            "ic": ic,
            "rank_ic": rank_ic,
            "fitness": fitness,
            "generation": generation,
        }
        self._records.append(record)

    def get_records(self) -> List[Dict[str, Any]]:
        """Retrieve copy of all graveyard rejection records."""
        return list(self._records)

    def size(self) -> int:
        """Total count of dead formulas buried in the graveyard."""
        return len(self._records)

    def export_graveyard(self, out_path: Path) -> None:
        """Persist graveyard ledger to JSON Lines for cross-session auditability."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
