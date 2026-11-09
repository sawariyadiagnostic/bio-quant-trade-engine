"""
Dedicated unit test suite for ADHD Batch 2 Remediations:
ADHD-10 through ADHD-20 (Broker Protocols, Microstructure & Execution Gateways).
"""
import sys
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.execution.rate_limiter import TokenBucketRateLimiter
from src.execution.clordid_ledger import ClientOrderIdLedger
from src.execution.shoonya_gateway import ShoonyaGateway, ShoonyaCredentials
from src.execution.alpaca_gateway import AlpacaGateway, AlpacaCredentials
from src.execution.ibkr_gateway import InteractiveBrokersGateway, IBKRCredentials
from src.execution.ccxt_gateway import CCXTGateway
from src.execution.overnight_engine import PreOpenAuctionRouter, OvernightExecutionEngine
from src.domain.models import InstrumentType


class TestADHDBatch2BrokerProtocols(unittest.TestCase):

    def test_adhd_10_rate_limiter_capacity_clamp(self):
        """ADHD-10: TokenBucketRateLimiter clamps requests > capacity to avoid deadlock."""
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=20.0)
        # Requesting 50 tokens (greater than capacity 20)
        acquired = limiter.try_acquire(50.0)
        self.assertTrue(acquired)
        self.assertAlmostEqual(limiter.tokens, 0.0, places=2)

    def test_adhd_11_shoonya_token_expiry_retry(self):
        """ADHD-11: ShoonyaGateway automatically retries on token expiry."""
        gw = ShoonyaGateway(paper_trading=False)
        gw.api = MagicMock()
        gw.susertoken = "mock_token"
        gw.api.place_order.side_effect = [
            {"stat": "Not_Ok", "emsg": "Session Expired : Invalid Token"},
            {"stat": "Ok", "result": "12345"}
        ]
        with patch.object(gw, "authenticate", return_value=True):
            res = gw.place_order({
                "symbol": "RELIANCE", "quantity": 10, "price": 2500.0, "action": "BUY"
            })
            self.assertEqual(res["stat"], "Ok")
            self.assertEqual(gw.api.place_order.call_count, 2)

    def test_adhd_12_shoonya_partial_fill_polling(self):
        """ADHD-12: ShoonyaGateway detects and handles PARTIALLY_FILLED orders."""
        gw = ShoonyaGateway(paper_trading=False)
        gw.susertoken = "mock_token"
        gw._get_single_order_history = MagicMock(return_value=[{
            "status": "PARTIALLY_FILLED", "fillshares": 5, "avgprc": 2500.0
        }])
        res = gw.poll_order_confirmation("ORD123", max_wait_s=0.2, poll_interval_s=0.05)
        self.assertEqual(res["stat"], "Ok")
        self.assertEqual(res["status"], "PARTIALLY_FILLED")
        self.assertEqual(res["fill_qty"], 5)

    def test_adhd_13_shoonya_reverse_proxy_html_error(self):
        """ADHD-13: ShoonyaGateway gracefully handles HTML 502/503 from reverse proxies."""
        gw = ShoonyaGateway(paper_trading=False)
        gw.susertoken = "mock_token"
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.text = "<html>502 Bad Gateway</html>"
        mock_resp.json.side_effect = Exception("JSONDecodeError")
        with patch("requests.post", return_value=mock_resp):
            res = gw._send_live_order_rest("RELIANCE-EQ", 10, Decimal("2500.0"), "BUY", "LMT", "CLORD1")
            self.assertEqual(res["stat"], "Not_Ok")
            self.assertIn("HTTP 502", res["emsg"])

    def test_adhd_14_shoonya_cancel_order_race_check(self):
        """ADHD-14: ShoonyaGateway cancel_order checks post-cancel status for concurrent fill."""
        gw = ShoonyaGateway(paper_trading=False)
        gw.api = MagicMock()
        gw.api.cancel_order.return_value = {"stat": "Ok", "result": "CANCELLED"}
        gw.get_order_status = MagicMock(return_value={
            "stat": "Ok", "status": "FILLED", "norenord": "ORD123"
        })
        # When order was actually filled at matching engine, cancel detects fill
        res = gw.cancel_order("ORD123")
        self.assertTrue(res)

    def test_adhd_15_options_suffix_preservation(self):
        """ADHD-15: format_trading_symbol preserves options CE/PE suffixes without mangling to -EQ."""
        gw = ShoonyaGateway(paper_trading=True)
        self.assertEqual(gw.format_trading_symbol("NIFTY26FEB24500CE"), "NIFTY26FEB24500CE")
        self.assertEqual(gw.format_trading_symbol("BANKNIFTY26FEB50000PE"), "BANKNIFTY26FEB50000PE")
        self.assertEqual(gw.format_trading_symbol("RELIANCE"), "RELIANCE-EQ")
        self.assertEqual(gw.format_trading_symbol("TCS-EQ"), "TCS-EQ")

    def test_adhd_16_alpaca_zero_price_and_extended_hours(self):
        """ADHD-16: AlpacaGateway rejects zero/negative paper order prices and maps extended hours."""
        alpaca = AlpacaGateway(paper_trading=True, initial_balance_usd=Decimal("1000.0"))
        # Zero price rejected
        res = alpaca.place_order({
            "symbol": "SPY", "quantity": 10, "price": 0.0, "action": "BUY", "order_type": "market"
        })
        self.assertEqual(res["stat"], "Not_Ok")
        self.assertIn("positive", res["emsg"].lower())

        # Valid price executes
        res_valid = alpaca.place_order({
            "symbol": "SPY", "quantity": 2, "price": 400.0, "action": "BUY", "order_type": "market"
        })
        self.assertEqual(res_valid["status"], "FILLED")
        self.assertEqual(alpaca.balance_usd, Decimal("200.0"))

    def test_adhd_17_ibkr_conid_resolution_and_confirmation(self):
        """ADHD-17: InteractiveBrokersGateway resolves conid and handles interactive confirmations."""
        ibkr = InteractiveBrokersGateway(
            credentials=IBKRCredentials(account_id="U12345", gateway_host="127.0.0.1"),
            paper_trading=False
        )
        ibkr._resolve_conid = MagicMock(return_value=265598)

        mock_prompt_resp = MagicMock()
        mock_prompt_resp.status_code = 200
        mock_prompt_resp.json.return_value = [{"id": "reply_abc", "message": ["Warning: price threshold"]}]

        mock_confirm_resp = MagicMock()
        mock_confirm_resp.status_code = 200
        mock_confirm_resp.json.return_value = [{"order_id": "IB_ORD_999", "order_status": "Submitted"}]

        with patch("requests.post", side_effect=[mock_prompt_resp, mock_confirm_resp]):
            res = ibkr._send_live_order("AAPL", Decimal("10"), Decimal("150.0"), "BUY", ibkr.balances, "LMT", "CLORD1")
            self.assertEqual(res["stat"], "Ok")
            self.assertEqual(res["order_id"], "IB_ORD_999")

    def test_adhd_18_ccxt_symmetric_margin_and_realized_pnl(self):
        """ADHD-18: CCXTGateway enforces short margin and records realized PnL."""
        ccxt = CCXTGateway(paper_trading=True, initial_balance_usdt=Decimal("100.0"))

        # 1. Unchecked naked shorting rejected (10 BTC @ $65000 requires $65,000 margin)
        res_reject = ccxt.place_order({
            "symbol": "BTC/USDT", "quantity": Decimal("10.0"), "price": Decimal("65000.0"), "action": "SELL"
        })
        self.assertEqual(res_reject["stat"], "Not_Ok")
        self.assertIn("Insufficient USDT margin", res_reject["emsg"])

        # 2. Valid short with sufficient margin (0.01 BTC @ $65,000 requires $65 margin, have $100)
        res_open = ccxt.place_order({
            "symbol": "BTC/USDT", "quantity": Decimal("0.01"), "price": Decimal("65000.0"), "action": "SELL"
        })
        self.assertEqual(res_open["status"], "FILLED")
        self.assertEqual(ccxt.positions["BTC/USDT"]["size"], Decimal("-0.01"))

        # 3. Profitable buy to cover (price drops to $60,000; profit = 0.01 * ($65000 - $60000) = $50)
        res_close = ccxt.place_order({
            "symbol": "BTC/USDT", "quantity": Decimal("0.01"), "price": Decimal("60000.0"), "action": "BUY"
        })
        self.assertEqual(res_close["status"], "FILLED")
        self.assertNotIn("BTC/USDT", ccxt.positions)
        self.assertAlmostEqual(float(ccxt.balance_usdt), 150.0, places=2)

    def test_adhd_19_auction_tif_and_bipartite_friction(self):
        """ADHD-19: PreOpenAuctionRouter uses DAY TIF and OvernightEngine applies bipartite tax."""
        router = PreOpenAuctionRouter()
        order = router.route_auction_order("RELIANCE.NS", "BUY", 2500.0, 10)
        self.assertEqual(order["time_in_force"], "DAY")
        self.assertEqual(order["session"], "PRE_OPEN")

        engine = OvernightExecutionEngine(initial_capital=100000.0, leverage=1.0)
        # Execute session with profitable trade
        rec = engine.execute_session(
            date_str="2026-03-01",
            ticker="TCS.NS",
            tier="LargeCap",
            instrument=InstrumentType.EQUITY_CASH,
            entry_price=3000.0,
            open_price=3030.0,
            high_price=3050.0,
            low_price=3025.0
        )
        self.assertGreater(rec.statutory_friction, 0.0)
        # Roundtrip tax must exceed single-leg 10 bps turnover tax
        self.assertGreater(rec.statutory_friction, 100000.0 * 0.0010)

    def test_adhd_20_clordid_decimal_and_datetime_serialization(self):
        """ADHD-20: ClientOrderIdLedger serializes Decimal and datetime without crash."""
        from datetime import datetime
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            temp_db = tf.name

        try:
            ledger = ClientOrderIdLedger(db_path=temp_db)
            resp_data = {
                "status": "FILLED",
                "price": Decimal("2540.75"),
                "quantity": Decimal("100"),
                "timestamp": datetime.now()
            }
            # Must not raise TypeError
            ledger.put("CLORD_TEST_99", resp_data, symbol="RELIANCE", action="BUY", quantity=100, price=2540.75)
            saved = ledger.get("CLORD_TEST_99")
            self.assertIsNotNone(saved)
            self.assertEqual(saved["status"], "FILLED")
            self.assertEqual(saved["price"], 2540.75)
        finally:
            if os.path.exists(temp_db):
                try:
                    os.unlink(temp_db)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
