"""Multi-Asset Portfolio Paper Trading Engine (Wave 4: Task 4.1)."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


@dataclass
class PortfolioPaperTrader:
    """
    Virtual multi-asset broker managing cash balances, integer lot sizes,
    and statutory NSE friction deductions across simultaneous ETF positions.
    """
    initial_capital: float = 50000.0
    friction_bps: float = 1.2  # 0.00012 for ETF Delivery
    cash: float = field(init=False)
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    latest_prices: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.cash = float(self.initial_capital)

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculates mark-to-market total portfolio equity."""
        self.latest_prices.update(current_prices)
        pos_val = sum(
            pos["quantity"] * current_prices.get(sym, pos["avg_price"])
            for sym, pos in self.positions.items()
        )
        return float(self.cash + pos_val)

    def rebalance(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Executes a portfolio-wide target allocation rebalance.
        Sells are processed first to free up liquidity before executing buys.
        """
        self.latest_prices.update(current_prices)
        executed_trades: List[Dict[str, Any]] = []
        friction = self.friction_bps / 10000.0
        total_equity = self.get_total_equity(current_prices)

        # 1. First Pass: Sells / Liquidations
        for sym, pos in list(self.positions.items()):
            price = current_prices.get(sym, pos["avg_price"])
            target_pct = target_weights.get(sym, 0.0)
            target_val = total_equity * target_pct
            target_qty = int(target_val / price) if price > 0 else 0

            current_qty = pos["quantity"]
            if target_qty < current_qty:
                sell_qty = current_qty - target_qty
                turnover = sell_qty * price
                tax = turnover * friction
                net_credit = turnover - tax

                self.cash += net_credit
                pos["quantity"] = target_qty

                trade_rec = {
                    "timestamp": datetime.now().isoformat(),
                    "ticker": sym,
                    "action": "SELL",
                    "quantity": sell_qty,
                    "price": price,
                    "gross_turnover": round(turnover, 2),
                    "friction_fee": round(tax, 4),
                    "remaining_cash": round(self.cash, 2)
                }
                executed_trades.append(trade_rec)
                self.trade_history.append(trade_rec)

        # 2. Second Pass: Buys / Additions
        for sym, target_pct in target_weights.items():
            if target_pct <= 0.0 or sym not in current_prices:
                continue
            price = current_prices[sym]
            target_val = total_equity * target_pct
            current_qty = self.positions.get(sym, {}).get("quantity", 0)
            target_qty = int(target_val / price) if price > 0 else 0

            if target_qty > current_qty:
                buy_qty = target_qty - current_qty
                turnover = buy_qty * price
                tax = turnover * friction
                total_cost = turnover + tax

                if total_cost > self.cash:
                    # Clip to affordable integer shares
                    buy_qty = int(self.cash / (price * (1.0 + friction)))
                    turnover = buy_qty * price
                    tax = turnover * friction
                    total_cost = turnover + tax

                if buy_qty > 0 and total_cost <= self.cash:
                    self.cash -= total_cost
                    if sym not in self.positions:
                        self.positions[sym] = {"quantity": buy_qty, "avg_price": price}
                    else:
                        prev_qty = self.positions[sym]["quantity"]
                        prev_price = self.positions[sym]["avg_price"]
                        new_qty = prev_qty + buy_qty
                        new_avg = ((prev_qty * prev_price) + turnover) / new_qty
                        self.positions[sym] = {"quantity": new_qty, "avg_price": new_avg}

                    trade_rec = {
                        "timestamp": datetime.now().isoformat(),
                        "ticker": sym,
                        "action": "BUY",
                        "quantity": buy_qty,
                        "price": price,
                        "gross_turnover": round(turnover, 2),
                        "friction_fee": round(tax, 4),
                        "remaining_cash": round(self.cash, 2)
                    }
                    executed_trades.append(trade_rec)
                    self.trade_history.append(trade_rec)

        return executed_trades

    def earn_ledger_cash_interest(self) -> float:
        """SEBI Mandate: Uninvested ledger cash balances cannot earn interest. Yield is strictly 0.0."""
        return 0.0

    def auto_sweep_liquid_bees(
        self,
        liquid_bees_price: float = 1000.0,
        annual_yield: float = 0.068,
        reserve_ratio: float = 0.05
    ) -> Dict[str, Any]:
        """
        Sweeps idle unallocated cash into LIQUIDBEES earning 6.8% annualized TREPS yield.
        Leaves reserve_ratio (e.g. 5%) of cash as an immediate liquidity cushion.
        SEBI Compliance: Uninvested ledger cash balance earns strictly 0.0% interest.
        Yield is derived solely from the purchased LIQUIDBEES ETF units.
        """
        if self.cash <= liquid_bees_price:
            return {"status": "INSUFFICIENT_CASH", "swept_units": 0, "cash": self.cash}

        available_to_sweep = max(0.0, self.cash * (1.0 - reserve_ratio))
        swept_units = int(available_to_sweep / liquid_bees_price)
        if swept_units <= 0:
            return {"status": "ZERO_UNITS", "swept_units": 0, "cash": self.cash}

        sweep_cost = swept_units * liquid_bees_price
        self.cash -= sweep_cost

        prev_pos = self.positions.get("LIQUIDBEES", {"quantity": 0, "avg_price": liquid_bees_price})
        total_units = prev_pos["quantity"] + swept_units
        self.positions["LIQUIDBEES"] = {"quantity": total_units, "avg_price": liquid_bees_price}

        daily_yield_rate = annual_yield / 365.0
        daily_interest = (total_units * liquid_bees_price) * daily_yield_rate

        return {
            "status": "SWEPT_OK",
            "swept_units": swept_units,
            "total_liquid_bees_units": total_units,
            "cost": sweep_cost,
            "daily_yield_inr": round(daily_interest, 2),
            "remaining_cash": round(self.cash, 2)
        }
