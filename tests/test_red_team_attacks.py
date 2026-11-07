"""
Adversarial Red Team Attack & Fuzzing Test Suite.
Systematically tests edge cases, numeric explosions, NaN poisoning,
recursion bombs, division by zero, and capital breach attempts.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.formula_parser import FormulaParser
from src.domain.formula_ast import ASTNode, ASTEvaluator, TreeGenerator, GeneticCrossover
from src.execution.risk_controller import CircuitDamper, MicroCapitalController, PositionSizer
from src.execution.shoonya_gateway import ShoonyaGateway
from src.execution.execution_dispatcher import ExecutionDispatcher


def test_parser_scientific_notation_adversarial():
    """Attacks parser with scientific floats and operators."""
    # 1. Scientific float in denominator
    node = FormulaParser.parse("True_Range + 1e-8")
    assert node.op == "ADD"
    assert node.children[1].val == "1e-8"

    # 2. Standalone scientific notation
    node_sci = FormulaParser.parse("2.5e-5")
    assert node_sci.op == "FEATURE"
    assert node_sci.val == "2.5e-5"

    # 3. Scientific notation with positive exponent
    node_pos = FormulaParser.parse("1e+4 * Vol_ZScore")
    assert node_pos.op == "MUL"
    assert node_pos.children[0].val == "1e+4"


def test_parser_recursion_and_malformed_bomb():
    """Attacks parser with deep nesting and malformed syntax."""
    # Recursion bomb: 60 nested parentheses
    bomb = "(" * 60 + "Dist_VWAP" + ")" * 60
    node = FormulaParser.parse(bomb)
    assert node is not None

    # Empty string
    node_empty = FormulaParser.parse("   ")
    assert node_empty.op == "FEATURE"
    assert node_empty.val == "0.0"

    # Unbalanced parentheses
    node_unbal = FormulaParser.parse("((EMA_10(Dist_VWAP)")
    assert node_unbal is not None


def test_ast_numerical_stability_adversarial():
    """Attacks ASTEvaluator with division by zero, NaNs, and infinities."""
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0],
        "Zero_Col": [0.0, 0.0, 0.0],
        "Nan_Col": [np.nan, np.nan, np.nan],
        "Inf_Col": [np.inf, -np.inf, np.nan]
    })

    # Division by zero
    div_zero = ASTNode(
        op="DIV_SAFE",
        children=[
            ASTNode(op="FEATURE", val="Close"),
            ASTNode(op="FEATURE", val="Zero_Col")
        ]
    )
    res = ASTEvaluator.eval_node(div_zero, df, current_regime=0)
    assert np.all(np.isfinite(res))
    assert np.all(res == 0.0)

    # Missing column
    missing = ASTNode(op="FEATURE", val="NON_EXISTENT_FEATURE_NAME")
    res_missing = ASTEvaluator.eval_node(missing, df, current_regime=0)
    assert np.all(res_missing == 0.0)


def test_risk_controller_nan_and_infinite_price_attacks():
    """Attacks risk controller with NaN, Inf, and negative prices."""
    ctrl = MicroCapitalController(total_capital=10000.0)

    # NaN price
    units_nan, reason_nan = ctrl.allocate(current_price=float("nan"))
    assert units_nan == 0
    assert reason_nan == "INVALID_PRICE"

    # Infinite price
    units_inf, reason_inf = ctrl.allocate(current_price=float("inf"))
    assert units_inf == 0
    assert reason_inf == "INVALID_PRICE"

    # Negative price
    units_neg, reason_neg = ctrl.allocate(current_price=-150.0)
    assert units_neg == 0
    assert reason_neg == "INVALID_PRICE"

    # Circuit damper with NaN
    bands_nan = CircuitDamper.calculate_bands(current_price=float("nan"), prev_close=100.0)
    assert bands_nan.is_locked is True


def test_shoonya_gateway_nan_poisoning_attack():
    """Attacks ShoonyaGateway balance with NaN prices to attempt state corruption."""
    gw = ShoonyaGateway(paper_trading=True, initial_balance=10000.0)

    # Attempt to inject NaN price
    res_nan = gw.place_order(symbol="NIFTYBEES", quantity=10, price=float("nan"))
    assert res_nan["stat"] == "Not_Ok"
    assert "Price must be a positive finite number" in res_nan["emsg"]
    assert gw.balance == 10000.0  # Balance was NOT poisoned with NaN!

    # Attempt to pass float quantity
    res_qty = gw.place_order(symbol="NIFTYBEES", quantity=5.5, price=250.0) # type: ignore
    assert res_qty["stat"] == "Not_Ok"
    assert "Quantity must be a positive integer" in res_qty["emsg"]

    # Attempt to buy beyond balance
    res_over = gw.place_order(symbol="NIFTYBEES", quantity=1000000, price=250.0)
    assert res_over["stat"] == "Not_Ok"
    assert "Insufficient virtual cash" in res_over["emsg"]
    assert gw.balance == 10000.0


def test_crossover_epigenetic_histone_preservation():
    """Verifies that subtree crossover preserves epigenetic histone active_regimes."""
    p1 = ASTNode(op="ADD", active_regimes={0, 1, 2}, children=[
        ASTNode(op="FEATURE", val="Dist_VWAP", active_regimes={0, 1}),
        ASTNode(op="FEATURE", val="RSI_14", active_regimes={2, 3})
    ])
    p2 = ASTNode(op="SUB", active_regimes={4, 5, 6}, children=[
        ASTNode(op="FEATURE", val="SP500_Ret", active_regimes={4, 5}),
        ASTNode(op="FEATURE", val="ADR_Ret", active_regimes={6, 7})
    ])

    c1, c2 = GeneticCrossover.crossover(p1, p2)
    assert c1 is not None and c2 is not None
    # All subnodes have valid non-empty active_regimes
    for node in c1.get_all_subnodes() + c2.get_all_subnodes():
        assert len(node.active_regimes) > 0
