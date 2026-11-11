"""Composable Declarative Universe Selection Engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set
from ravana.data.universe.models import AssetClass, Instrument
from ravana.data.universe.registry import AssetRegistry


class UniverseSelector(ABC):
    """Abstract composable universe filter."""

    @abstractmethod
    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        """Resolve universe rule into concrete list of instruments."""
        pass

    def __or__(self, other: UniverseSelector) -> UniverseSelector:
        return UnionUniverse(self, other)

    def __and__(self, other: UniverseSelector) -> UniverseSelector:
        return IntersectionUniverse(self, other)


class BasketUniverse(UniverseSelector):
    """Selects all instruments in a predefined named basket."""

    def __init__(self, basket_name: str) -> None:
        self.basket_name = basket_name

    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        return registry.get_basket(self.basket_name)


class TagUniverse(UniverseSelector):
    """Selects all instruments containing a specific tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        return registry.query(tag=self.tag)


class AssetClassUniverse(UniverseSelector):
    """Selects all instruments belonging to an asset class."""

    def __init__(self, asset_class: AssetClass) -> None:
        self.asset_class = asset_class

    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        return registry.query(asset_class=self.asset_class)


class SymbolUniverse(UniverseSelector):
    """Explicit static list of ticker symbols."""

    def __init__(self, symbols: List[str]) -> None:
        self.symbols = symbols

    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        return [inst for s in self.symbols if (inst := registry.get(s)) is not None]


class UnionUniverse(UniverseSelector):
    """Combines two universe selectors via union (OR)."""

    def __init__(self, left: UniverseSelector, right: UniverseSelector) -> None:
        self.left = left
        self.right = right

    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        res_left = self.left.resolve(registry)
        res_right = self.right.resolve(registry)
        seen: Set[str] = set()
        out: List[Instrument] = []
        for inst in res_left + res_right:
            if inst.symbol not in seen:
                seen.add(inst.symbol)
                out.append(inst)
        return out


class IntersectionUniverse(UniverseSelector):
    """Filters two universe selectors via intersection (AND)."""

    def __init__(self, left: UniverseSelector, right: UniverseSelector) -> None:
        self.left = left
        self.right = right

    def resolve(self, registry: AssetRegistry) -> List[Instrument]:
        res_left = self.left.resolve(registry)
        right_symbols = {i.symbol for i in self.right.resolve(registry)}
        return [i for i in res_left if i.symbol in right_symbols]


def parse_universe_spec(spec: str, registry: AssetRegistry) -> List[Instrument]:
    """Parse a declarative universe expression into resolved instruments.

    Supported patterns:
      - 'basket:core_etfs'
      - 'tag:liquid'
      - 'class:ETF'
      - 'symbols:BANKBEES.NS,NIFTYBEES.NS'
      - 'basket:core_etfs | tag:nifty50'
    """
    spec = spec.strip()
    if "|" in spec:
        tokens = [t.strip() for t in spec.split("|")]
        selectors = [_parse_atomic_spec(t) for t in tokens]
        combined = selectors[0]
        for s in selectors[1:]:
            combined = combined | s
        return combined.resolve(registry)

    if "&" in spec:
        tokens = [t.strip() for t in spec.split("&")]
        selectors = [_parse_atomic_spec(t) for t in tokens]
        combined = selectors[0]
        for s in selectors[1:]:
            combined = combined & s
        return combined.resolve(registry)

    return _parse_atomic_spec(spec).resolve(registry)


def _parse_atomic_spec(token: str) -> UniverseSelector:
    """Parse atomic predicate token."""
    token = token.strip()
    if token.startswith("basket:"):
        return BasketUniverse(token.split(":", 1)[1].strip())
    elif token.startswith("tag:"):
        return TagUniverse(token.split(":", 1)[1].strip())
    elif token.startswith("class:"):
        c_str = token.split(":", 1)[1].strip().upper()
        return AssetClassUniverse(AssetClass(c_str))
    elif token.startswith("symbols:"):
        syms = [s.strip() for s in token.split(":", 1)[1].split(",") if s.strip()]
        return SymbolUniverse(syms)
    else:
        # Default fallback to single symbol
        return SymbolUniverse([token])
