"""Unit tests for Dynamic Universe Registry and Selection DSL."""
import unittest

from ravana.data.universe.models import AssetClass, ExchangeType, Instrument
from ravana.data.universe.registry import AssetRegistry
from ravana.data.universe.selector import (
    AssetClassUniverse,
    BasketUniverse,
    TagUniverse,
    parse_universe_spec,
)


class TestUniverseRegistry(unittest.TestCase):
    """Test suite for Security Master and Universe Selection."""

    def setUp(self) -> None:
        self.registry = AssetRegistry()

    def test_catalog_loaded_instruments(self):
        """Verify instruments are parsed with full metadata."""
        bankbees = self.registry.get("BANKBEES.NS")
        self.assertIsNotNone(bankbees)
        self.assertEqual(bankbees.security_id, "INE694G01026")
        self.assertEqual(bankbees.asset_class, AssetClass.ETF)
        self.assertEqual(bankbees.underlying_symbol, "^NSEBANK")
        self.assertEqual(bankbees.parity_ratio, 100.0)
        self.assertEqual(bankbees.exchange, ExchangeType.NSE)

    def test_basket_resolution(self):
        """Verify predefined baskets resolve to expected instruments."""
        core_etfs = self.registry.get_basket("core_etfs")
        symbols = [i.symbol for i in core_etfs]
        self.assertIn("BANKBEES.NS", symbols)
        self.assertIn("NIFTYBEES.NS", symbols)
        self.assertEqual(len(symbols), 2)

    def test_query_by_asset_class_and_tag(self):
        """Verify querying by asset class and tags."""
        etfs = self.registry.query(asset_class=AssetClass.ETF)
        self.assertGreaterEqual(len(etfs), 4)

        banking = self.registry.query(sector="Banking")
        symbols = [i.symbol for i in banking]
        self.assertIn("BANKBEES.NS", symbols)
        self.assertIn("HDFCBANK.NS", symbols)

    def test_composable_selectors(self):
        """Verify composable union and intersection DSL."""
        u_etf = AssetClassUniverse(AssetClass.ETF)
        u_liquid = TagUniverse("liquid")

        # Intersection: ETFs that are liquid
        inter = (u_etf & u_liquid).resolve(self.registry)
        symbols = [i.symbol for i in inter]
        self.assertIn("BANKBEES.NS", symbols)
        self.assertIn("NIFTYBEES.NS", symbols)

        # Union: Core ETFs or Nifty 50
        u_core = BasketUniverse("core_etfs")
        u_nifty = BasketUniverse("nifty50_top5")
        union_res = (u_core | u_nifty).resolve(self.registry)
        self.assertEqual(len(union_res), 7)

    def test_parse_universe_spec_string(self):
        """Verify string-based universe DSL for UI and CLI interaction."""
        res = parse_universe_spec("basket:core_etfs", self.registry)
        self.assertEqual(len(res), 2)

        res_union = parse_universe_spec("basket:core_etfs | tag:nifty50", self.registry)
        self.assertGreaterEqual(len(res_union), 5)
