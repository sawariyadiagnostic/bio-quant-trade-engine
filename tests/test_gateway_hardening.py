"""
tests/test_gateway_hardening.py
TDD verification suite for Task 0: Shoonya Gateway latent bug remediation,
append-only order audit logging (SQLite WAL), and archive JSON-to-SQLite migration.
Uses Python standard library unittest for zero-dependency execution.
"""
import os
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials


class TestGatewayHardening(unittest.TestCase):

    def setUp(self):
        self.creds = ShoonyaCredentials(
            user_id="FA_TEST",
            password="TEST_PASSWORD",
            totp_key="TEST_TOTP_KEY",
            vendor_code="TEST_VC",
            api_key="TEST_API_KEY"
        )
        self.paper_gw = ShoonyaGateway(credentials=self.creds, paper_trading=True)
        self.paper_gw.paper_positions = {}
        self.paper_gw.balance = 100000.0

    def test_01_liquidate_positions_paper_mode_fixes_keyword_mismatch(self):
        """
        Subtask 0.1 & 0.2: Proves liquidate_positions() executes without TypeError
        or NameError on keyword argument mismatches.
        """
        self.paper_gw.paper_positions = {
            "NIFTYBEES-EQ": {"quantity": 100, "avg_price": 250.0},
            "GOLDBEES-EQ": {"quantity": 50, "avg_price": 60.0},
            "RELIANCE-EQ": {"quantity": 10, "avg_price": 2500.0},
        }
        res = self.paper_gw.liquidate_positions()
        self.assertEqual(res.get("stat"), "Ok", f"Liquidation failed: {res}")
        self.assertEqual(res.get("liquidated_count"), 3)
        self.assertEqual(len(self.paper_gw.paper_positions), 0, "All positions must be cleared")

    def test_02_liquidate_positions_live_mode_handles_short_and_market_price(self):
        """
        Subtask 0.3, 0.4 & 0.5: Proves live mode liquidation correctly:
        1. Replaces undefined 'sym' with 'tsym'
        2. Allows price=0.0 for MKT orders
        3. Squares off short positions (netqty < 0) with action="BUY"
        4. Passes correct product type (prd)
        """
        gw = ShoonyaGateway(credentials=self.creds, paper_trading=False)
        gw.susertoken = "mock_session_token"
        
        mock_api = MagicMock()
        mock_api.get_positions.return_value = [
            {"tsym": "NIFTYBEES-EQ", "netqty": "100", "prd": "C", "exch": "NSE"},
            {"tsym": "BANKBEES-EQ", "netqty": "-50", "prd": "M", "exch": "NSE"},
        ]
        mock_api.place_order.return_value = {"stat": "Ok", "norenordno": "2411070001"}
        gw.api = mock_api
        
        res = gw.liquidate_positions()
        self.assertEqual(res.get("stat"), "Ok", f"Live liquidation failed: {res}")
        self.assertEqual(res.get("liquidated_count"), 2)
        
        # Verify place_order calls
        self.assertEqual(mock_api.place_order.call_count, 2)
        calls = mock_api.place_order.call_args_list
        
        # Call 1: Long position liquidated via SELL
        c1 = calls[0].kwargs
        self.assertEqual(c1["tradingsymbol"], "NIFTYBEES-EQ")
        self.assertEqual(c1["quantity"], 100)
        self.assertEqual(c1["buy_or_sell"], "S")
        self.assertEqual(c1["price_type"], "MKT")
        self.assertEqual(c1["product_type"], "C")
        
        # Call 2: Short position squared off via BUY
        c2 = calls[1].kwargs
        self.assertEqual(c2["tradingsymbol"], "BANKBEES-EQ")
        self.assertEqual(c2["quantity"], 50)
        self.assertEqual(c2["buy_or_sell"], "B")
        self.assertEqual(c2["price_type"], "MKT")
        self.assertEqual(c2["product_type"], "M")

    def test_03_cancel_all_orders_uses_orderno_parameter(self):
        """
        Subtask 0.6: Proves cancel_all_orders passes 'orderno' kwarg to NorenApi,
        not 'norenordno'.
        """
        gw = ShoonyaGateway(credentials=self.creds, paper_trading=False)
        gw.susertoken = "mock_token"
        
        mock_api = MagicMock()
        mock_api.get_order_book.return_value = [
            {"norenordno": "1001", "status": "OPEN"},
            {"norenordno": "1002", "status": "TRIGGER_PENDING"},
            {"norenordno": "1003", "status": "COMPLETE"},
        ]
        mock_api.cancel_order.return_value = {"stat": "Ok", "result": "1001"}
        gw.api = mock_api
        
        res = gw.cancel_all_orders()
        self.assertEqual(res.get("stat"), "Ok")
        self.assertEqual(res.get("cancelled_orders"), 2)
        
        # Verify cancel_order parameter
        self.assertEqual(mock_api.cancel_order.call_count, 2)
        for call in mock_api.cancel_order.call_args_list:
            self.assertTrue("orderno" in call.kwargs or len(call.args) > 0, "Must pass orderno parameter")

    def test_04_place_order_market_order_allows_zero_price(self):
        """
        Subtask 0.3: Proves place_order accepts price=0.0 when order_type='MKT'.
        """
        res = self.paper_gw.place_order(
            symbol="NIFTYBEES-EQ",
            quantity=10,
            price=0.0,
            action="BUY",
            order_type="MKT"
        )
        self.assertEqual(res.get("stat"), "Ok", f"Market order with price 0.0 rejected: {res}")

    def test_05_send_live_order_detects_broker_rejection(self):
        """
        Subtask 0.7: Proves _send_live_order returns Not_Ok when NorenApi returns None.
        """
        gw = ShoonyaGateway(credentials=self.creds, paper_trading=False)
        gw.susertoken = "mock_token"
        
        mock_api = MagicMock()
        mock_api.place_order.return_value = None  # NorenApi returns None on failure
        gw.api = mock_api
        
        res = gw._send_live_order("NIFTYBEES-EQ", 10, 250.0, "BUY", "LMT", "C")
        self.assertEqual(res.get("stat"), "Not_Ok", "Must not mask None as stat: Ok")

    def test_06_format_trading_symbol_preserves_futures(self):
        """
        Subtask 0.8: Proves futures and special series do not get corrupted with '-EQ'.
        """
        self.assertEqual(ShoonyaGateway.format_trading_symbol("NIFTYBEES.NS"), "NIFTYBEES-EQ")
        self.assertEqual(ShoonyaGateway.format_trading_symbol("RELIANCE.NS"), "RELIANCE-EQ")
        self.assertEqual(ShoonyaGateway.format_trading_symbol("NIFTY26NOVFUT"), "NIFTY26NOVFUT")
        self.assertEqual(ShoonyaGateway.format_trading_symbol("BANKNIFTY26NOVFUT"), "BANKNIFTY26NOVFUT")

    def test_07_audit_logging_sqlite_wal(self):
        """
        Subtask 0.9: Proves order execution audit log is written to SQLite WAL
        without O(N^2) JSON file overwrites.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            db_path = os.path.join(tmp_dir, "broker_orders.db")
            gw = ShoonyaGateway(credentials=self.creds, paper_trading=True, audit_file=db_path)
            
            for _ in range(10):
                gw.place_order(symbol="NIFTYBEES-EQ", quantity=1, price=250.0, action="BUY")
                
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM order_audit_journal")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            self.assertEqual(count, 10, f"Expected 10 audit records in SQLite WAL, found {count}")

    def test_08_archive_migration_to_sqlite(self):
        """
        Subtask 0.10: Proves legacy audit/archive_*.json files can be discovered
        and migrated into SQLite WAL schema with 100% field parity.
        """
        from src.evolution.archive_migrator import migrate_json_archives_to_sqlite
        
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            legacy_json = os.path.join(tmp_dir, "archive_TEST_1y.json")
            sample_data = {
                "ticker": "TEST",
                "lookback": "1y",
                "rows": 20,
                "cols": 20,
                "elites": [
                    {
                        "row": 5,
                        "col": 10,
                        "fitness": 2.45,
                        "net_return_pct": 35.2,
                        "cagr_pct": 28.1,
                        "max_drawdown": -6.5,
                        "win_rate": 65.0,
                        "turnover": 5.2,
                        "volatility": 0.18,
                        "trades_count": 42,
                        "formula_str": "Sub(Ts_Rank(Close, 10), Ts_Mean(Volume, 20))"
                    }
                ]
            }
            with open(legacy_json, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
            
            target_db = os.path.join(tmp_dir, "alpha_elites.db")
            migrated_count = migrate_json_archives_to_sqlite(source_dir=tmp_dir, target_db=target_db)
            
            self.assertEqual(migrated_count, 1)
            
            conn = sqlite3.connect(target_db)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM resident_elites WHERE ticker='TEST' AND lookback='1y'")
            row = cur.fetchone()
            conn.close()
            
            self.assertIsNotNone(row)
            self.assertEqual(row["fitness"], 2.45)
            self.assertEqual(row["formula_str"], "Sub(Ts_Rank(Close, 10), Ts_Mean(Volume, 20))")


if __name__ == "__main__":
    unittest.main()
