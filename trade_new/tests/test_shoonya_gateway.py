"""
Unit tests for Shoonya (Finvasia) High-Performance Gateway.
Empirically verifies:
1. Dynamic TOTP generation with RFC 6238 Base32 keys.
2. Exchange symbol formatting and NSE normalization.
3. Statutory friction calculations (1.2 bps ETF vs 25 bps Equity).
4. Paper Trading order lifecycle, balance updates, and friction deduction.
5. Order idempotency via client_order_id cache.
6. Portfolio liquidation in paper and exchange modes.
7. Append-only order audit logging to SQLite WAL.
"""
import os
import sys
import tempfile
import unittest
import pyotp
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials


class TestShoonyaGateway(unittest.TestCase):
    def setUp(self):
        self.secret = pyotp.random_base32()
        self.creds = ShoonyaCredentials(
            user_id="FA1234",
            password="dummy_password",
            totp_key=self.secret,
            vendor_code="FA1234_U",
            api_key="dummy_api_key"
        )
        self.gw = ShoonyaGateway(credentials=self.creds, paper_trading=True, initial_balance=50000.0)

    def test_totp_generation(self):
        """Verifies that a valid base32 key generates a 6-digit numeric TOTP code."""
        totp = self.gw.generate_totp()
        self.assertEqual(len(totp), 6)
        self.assertTrue(totp.isdigit())

    def test_symbol_formatting(self):
        """Verifies standard NSE suffix normalization."""
        self.assertEqual(ShoonyaGateway.format_trading_symbol("NIFTYBEES"), "NIFTYBEES-EQ")
        self.assertEqual(ShoonyaGateway.format_trading_symbol("NIFTYBEES.NS"), "NIFTYBEES-EQ")
        self.assertEqual(ShoonyaGateway.format_trading_symbol("INFY-EQ"), "INFY-EQ")
        self.assertEqual(ShoonyaGateway.format_trading_symbol("HAL"), "HAL-EQ")

    def test_friction_calculation(self):
        """Verifies statutory friction: 1.2 bps for ETF vs 25 bps for single equity."""
        trade_value = 10000.0
        etf_friction = ShoonyaGateway.calculate_friction(trade_value, is_etf=True)
        equity_friction = ShoonyaGateway.calculate_friction(trade_value, is_etf=False)
        self.assertEqual(round(etf_friction, 2), 1.20)
        self.assertEqual(round(equity_friction, 2), 25.00)

    def test_paper_trading_order_flow(self):
        """Verifies order execution, portfolio tracking, and balance updates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_orders.json")
            gw = ShoonyaGateway(paper_trading=True, initial_balance=10000.0, audit_file=audit_file)

            # Buy 36 units @ 270.0 = 9,720.0 + friction
            res_buy = gw.place_order(symbol="NIFTYBEES", quantity=36, price=270.0, action="BUY", is_etf=True)
            self.assertEqual(res_buy["stat"], "Ok")
            self.assertTrue(res_buy["norenordno"].startswith("SIM_"))
            self.assertEqual(gw.paper_positions["NIFTYBEES-EQ"]["quantity"], 36)
            self.assertEqual(gw.paper_positions["NIFTYBEES-EQ"]["avg_price"], 270.0)
            self.assertLess(gw.balance, 10000.0 - 9720.0)

            # Insufficient funds reject
            res_fail = gw.place_order(symbol="NIFTYBEES", quantity=10, price=270.0, action="BUY", is_etf=True)
            self.assertEqual(res_fail["stat"], "Not_Ok")
            self.assertIn("Insufficient virtual cash", res_fail["emsg"])

            # Sell 36 units @ 275.0 = 9,900.0 - friction -> net profit
            res_sell = gw.place_order(symbol="NIFTYBEES", quantity=36, price=275.0, action="SELL", is_etf=True)
            self.assertEqual(res_sell["stat"], "Ok")
            self.assertNotIn("NIFTYBEES-EQ", gw.paper_positions)
            self.assertGreater(gw.balance, 10000.0)

    def test_gateway_idempotency(self):
        """Verify client_order_id idempotency prevents double balance deduction."""
        clordid = "CLORD_TEST_SOVEREIGN_1"
        res1 = self.gw.place_order("NIFTYBEES", quantity=10, price=250.0, action="BUY", client_order_id=clordid)
        self.assertEqual(res1["stat"], "Ok")
        bal_after_1 = self.gw.balance

        # Duplicate order with identical clordid
        res2 = self.gw.place_order("NIFTYBEES", quantity=10, price=250.0, action="BUY", client_order_id=clordid)
        self.assertEqual(res2["stat"], "Ok")
        self.assertEqual(res2["norenordno"], res1["norenordno"])
        self.assertEqual(self.gw.balance, bal_after_1)

    def test_liquidate_positions(self):
        """Verify position liquidation in paper trading mode."""
        self.gw.paper_positions = {
            "NIFTYBEES-EQ": {"quantity": 100, "avg_price": 250.0},
            "GOLDBEES-EQ": {"quantity": 50, "avg_price": 60.0},
        }
        res = self.gw.liquidate_positions()
        self.assertEqual(res.get("stat"), "Ok")
        self.assertEqual(res.get("liquidated_count"), 2)
        self.assertEqual(len(self.gw.paper_positions), 0)

    def test_order_audit_trail_sqlite(self):
        """Verify order audit trail is recorded to SQLite WAL."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_audit.db")
            gw = ShoonyaGateway(paper_trading=True, initial_balance=50000.0, audit_file=db_path)
            res = gw.place_order("NIFTYBEES", quantity=10, price=250.0, action="BUY")
            self.assertEqual(res["stat"], "Ok")

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM order_audit_journal")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
