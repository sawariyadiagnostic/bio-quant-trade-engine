"""Unified Production Execution Dispatcher (Ponytail MVC)."""
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import pandas as pd
from ..domain.models import CircuitBands
from ..domain.formula_ast import ASTNode, ASTEvaluator
from ..domain.formula_parser import FormulaParser
from .risk_controller import CircuitDamper, MicroCapitalController
from .shoonya_gateway import ShoonyaGateway


@dataclass
class DispatchDecision:
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'HALT'
    signal_value: float
    raw_price: float
    quantity: int
    regime_id: int
    circuit_bands: Optional[CircuitBands] = None
    rejection_reason: str = ""
    order_id: Optional[str] = None
    executed: bool = False


class ExecutionDispatcher:
    def __init__(self, gateway: ShoonyaGateway, risk_controller: MicroCapitalController, signal_threshold: float = 0.05):
        self.gateway, self.risk_controller, self.signal_threshold = gateway, risk_controller, signal_threshold

    def evaluate_and_dispatch(
        self, ticker: str, df: pd.DataFrame, formula: Union[ASTNode, str], regime_id: int = 0, is_etf: bool = True
    ) -> DispatchDecision:
        if df.empty:
            return DispatchDecision(ticker, "HALT", 0.0, 0.0, 0, regime_id, rejection_reason="EMPTY_DATAFRAME")

        col = next((c for c in ("Close_Raw", "Close_Adj", "Close") if c in df.columns), df.columns[0])
        p, prev_p = float(df[col].iloc[-1]), float(df[col].iloc[-2]) if len(df) > 1 else float(df[col].iloc[-1])
        if not np.isfinite(p) or p <= 0:
            return DispatchDecision(ticker, "HALT", 0.0, 0.0, 0, regime_id, rejection_reason="INVALID_PRICE")

        bands = CircuitDamper.calculate_bands(p, prev_p)
        if regime_id == -1:
            return DispatchDecision(ticker, "HALT", 0.0, round(p, 2), 0, regime_id, bands, "REGIME_CRASH_PROTECT_NOISE")

        ast_root = FormulaParser.parse(formula) if isinstance(formula, str) else formula
        signals = ASTEvaluator.eval_node(ast_root, df, current_regime=regime_id)
        sig = float(signals[-1]) if len(signals) > 0 else 0.0

        action = "BUY" if sig > self.signal_threshold else ("SELL" if sig < -self.signal_threshold else "HOLD")
        if action == "HOLD":
            return DispatchDecision(ticker, "HOLD", round(sig, 4), round(p, 2), 0, regime_id, bands, "SIGNAL_BELOW_THRESHOLD")

        if action == "SELL":
            tsym = self.gateway.format_trading_symbol(ticker)
            held = self.gateway.paper_positions.get(tsym, {}).get("quantity", 0)
            if held > 0:
                res = self.gateway.place_order(ticker, held, p, action="SELL", is_etf=is_etf)
                ok = res.get("stat") == "Ok" or res.get("status") in ("COMPLETE", "PLACED", "OPEN")
                return DispatchDecision(
                    ticker, "SELL", round(sig, 4), round(p, 2), held, regime_id, bands,
                    order_id=res.get("norenordno") or res.get("order_id"), executed=ok,
                    rejection_reason="" if ok else str(res.get("emsg") or "SELL_FAILED")
                )
            return DispatchDecision(ticker, "SELL", round(sig, 4), round(p, 2), 0, regime_id, bands, "NO_POSITION_HELD")

        units, reason = self.risk_controller.allocate(p, is_etf, regime_id, bands)
        if units <= 0:
            return DispatchDecision(ticker, "HALT", round(sig, 4), round(p, 2), 0, regime_id, bands, reason)

        res = self.gateway.place_order(ticker, units, p, action="BUY", is_etf=is_etf)
        ok = res.get("stat") == "Ok" or res.get("status") in ("COMPLETE", "PLACED", "OPEN")
        return DispatchDecision(
            ticker, "BUY", round(sig, 4), round(p, 2), units, regime_id, bands,
            order_id=res.get("norenordno") or res.get("order_id"), executed=ok,
            rejection_reason="" if ok else str(res.get("emsg") or res.get("reason", "ORDER_ROUTING_FAILED"))
        )
