"""Domain models for Security Master and Instrument Metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AssetClass(str, Enum):
    """Asset class category defining microstructure rules."""
    EQUITY = "EQUITY"
    ETF = "ETF"
    INDEX = "INDEX"
    FUTURE = "FUTURE"
    MACRO = "MACRO"


class ExchangeType(str, Enum):
    """Trading exchange or session calendar."""
    NSE = "NSE"
    BSE = "BSE"
    US = "US"
    GLOBAL_24H = "GLOBAL_24H"


@dataclass(frozen=True)
class Instrument:
    """Canonical Security Master record for a tradable or reference entity."""
    symbol: str                              # e.g. "BANKBEES.NS"
    security_id: str                         # Immutable ISIN or internal ID, e.g. "INE694G01026"
    asset_class: AssetClass                  # ETF, EQUITY, INDEX, etc.
    exchange: ExchangeType = ExchangeType.NSE
    currency: str = "INR"
    underlying_symbol: Optional[str] = None  # e.g. "^NSEBANK" for BANKBEES
    parity_ratio: float = 1.0                # e.g. 100.0 for BANKBEES
    lot_size: int = 1
    tick_size: float = 0.01
    sector: Optional[str] = None             # e.g. "Banking"
    tags: List[str] = field(default_factory=list)
    active_from: Optional[str] = None
    active_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.active_to is None
