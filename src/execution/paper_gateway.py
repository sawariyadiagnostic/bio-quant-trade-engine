"""Event-Driven Paper Trading Execution Gateway (Ponytail Minimum Viable Code)."""
import os
import pandas as pd
from loguru import logger
from ..domain.models import OrderSignal


class PaperTradingGateway:
    def __init__(self, initial_capital: float = 50000.0, audit_dir: str = None):
        self.balance = float(initial_capital)
        self.initial_capital = float(initial_capital)
        self.audit_dir = audit_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "audit"))
        os.makedirs(self.audit_dir, exist_ok=True)
        self.csv_path = os.path.join(self.audit_dir, "paper_trades.csv")

    def _calculate_statutory_taxes(self, turn_val: float, is_delivery: bool = True) -> float:
        stt = turn_val * (0.001 if is_delivery else 0.00025)
        return float(stt + turn_val * (0.0000325 * 1.18 + 0.00015 + 0.0003))

    def execute_order(self, signal: OrderSignal, is_delivery: bool = True) -> bool:
        if signal.quantity <= 0:
            return False
        trade_val = signal.raw_price * signal.quantity
        taxes = self._calculate_statutory_taxes(trade_val, is_delivery)
        cost = trade_val + taxes
        if cost > self.balance:
            logger.warning(f"Gateway: Insufficient virtual balance ₹{self.balance:.2f} for ₹{cost:.2f}")
            return False

        self.balance -= cost
        logger.success(f"[PAPER] {signal.action} {signal.quantity} {signal.ticker} @ ₹{signal.raw_price:.2f} | Bal: ₹{self.balance:.2f}")
        rec = {
            "timestamp": signal.timestamp.isoformat(), "ticker": signal.ticker, "action": signal.action,
            "raw_price": signal.raw_price, "quantity": signal.quantity, "trade_value": round(trade_val, 2),
            "taxes_friction": round(taxes, 2), "stop_loss": signal.stop_loss, "target": signal.target,
            "regime_id": signal.regime_id, "remaining_balance": round(self.balance, 2)
        }
        pd.DataFrame([rec]).to_csv(self.csv_path, mode="a", header=not os.path.exists(self.csv_path), index=False)
        return True
