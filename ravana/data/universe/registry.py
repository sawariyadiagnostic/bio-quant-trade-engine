"""Asset Registry and Security Master implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from ravana.data.universe.models import AssetClass, ExchangeType, Instrument


class AssetRegistry:
    """Central registry and Single Source of Truth for all tradable and reference assets."""

    def __init__(self, catalog_path: Optional[Path] = None) -> None:
        self.catalog_path = catalog_path or Path(__file__).resolve().parents[2] / "config" / "asset_catalog.yaml"
        self._instruments_by_symbol: Dict[str, Instrument] = {}
        self._instruments_by_id: Dict[str, Instrument] = {}
        self._baskets: Dict[str, List[str]] = {}
        self.load_catalog()

    def load_catalog(self) -> None:
        """Parse asset_catalog.yaml and index instruments."""
        if not self.catalog_path.exists():
            return

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item in data.get("instruments", []):
            inst = Instrument(
                symbol=item["symbol"],
                security_id=item["security_id"],
                asset_class=AssetClass(item["asset_class"]),
                exchange=ExchangeType(item.get("exchange", "NSE")),
                currency=item.get("currency", "INR"),
                underlying_symbol=item.get("underlying_symbol"),
                parity_ratio=float(item.get("parity_ratio", 1.0)),
                lot_size=int(item.get("lot_size", 1)),
                tick_size=float(item.get("tick_size", 0.01)),
                sector=item.get("sector"),
                tags=item.get("tags", []),
                active_from=item.get("active_from"),
                active_to=item.get("active_to"),
                metadata=item.get("metadata", {}),
            )
            self._instruments_by_symbol[inst.symbol] = inst
            self._instruments_by_id[inst.security_id] = inst

        self._baskets = data.get("baskets", {})

    def get(self, symbol: str) -> Optional[Instrument]:
        """Lookup instrument by symbol."""
        return self._instruments_by_symbol.get(symbol)

    def get_by_id(self, security_id: str) -> Optional[Instrument]:
        """Lookup instrument by immutable security ID."""
        return self._instruments_by_id.get(security_id)

    def get_basket(self, name: str) -> List[Instrument]:
        """Return list of instruments belonging to named basket."""
        symbols = self._baskets.get(name, [])
        return [self._instruments_by_symbol[s] for s in symbols if s in self._instruments_by_symbol]

    def list_baskets(self) -> List[str]:
        """Return all registered basket names."""
        return list(self._baskets.keys())

    def list_all(self) -> List[Instrument]:
        """Return all registered instruments."""
        return list(self._instruments_by_symbol.values())

    def query(
        self,
        asset_class: Optional[AssetClass] = None,
        tag: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> List[Instrument]:
        """Declarative filtering over registered instruments."""
        res = list(self._instruments_by_symbol.values())
        if asset_class is not None:
            res = [i for i in res if i.asset_class == asset_class]
        if tag is not None:
            res = [i for i in res if tag in i.tags]
        if sector is not None:
            res = [i for i in res if i.sector == sector]
        return res
