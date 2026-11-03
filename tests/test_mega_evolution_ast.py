"""Unit tests for Mega Evolution AST Node (Wave 2: Task 2.1)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_mega_evolution_ast_evaluation():
    """Verify MegaEvolutionNode dynamically switches between base and mega forms based on condition."""
    from src.domain.formula_ast import MegaEvolutionNode, ASTNode, ASTEvaluator

    # Condition: Vol_ZScore - 2.0 (triggers when Vol_ZScore > 2.0)
    cond = ASTNode("SUB", children=[ASTNode("FEATURE", "Vol_ZScore"), ASTNode("FEATURE", "2.0")])
    base = ASTNode("FEATURE", "Close_Adj")
    mega = ASTNode("MUL", children=[ASTNode("FEATURE", "Parkinson_Vol"), ASTNode("FEATURE", "10.0")])

    mega_node = MegaEvolutionNode(condition=cond, base=base, mega=mega)

    df_calm = pd.DataFrame({
        "Vol_ZScore": [0.5, 1.2, 1.8],
        "Close_Adj": [250.0, 252.0, 255.0],
        "Parkinson_Vol": [0.05, 0.06, 0.07]
    })
    # Calm market: condition <= 0 -> evaluates base (Close_Adj)
    res_calm = ASTEvaluator.eval_node(mega_node, df_calm)
    assert np.allclose(res_calm, [250.0, 252.0, 255.0])

    df_crash = pd.DataFrame({
        "Vol_ZScore": [2.5, 3.1, 4.0],
        "Close_Adj": [240.0, 230.0, 220.0],
        "Parkinson_Vol": [0.2, 0.3, 0.4]
    })
    # Crash / Volatility spike: condition > 0 -> evaluates mega (Parkinson_Vol * 10)
    res_crash = ASTEvaluator.eval_node(mega_node, df_crash)
    assert np.allclose(res_crash, [2.0, 3.0, 4.0])
    print("[PASS] test_mega_evolution_ast_evaluation")


def test_mega_evolution_formula_string_and_parsing():
    """Verify MegaEvolution formula serializes to string and parses back to identical AST."""
    from src.domain.formula_ast import MegaEvolutionNode, ASTNode
    from src.domain.formula_parser import FormulaParser

    cond = ASTNode("FEATURE", "Vol_ZScore")
    base = ASTNode("FEATURE", "Close_Adj")
    mega = ASTNode("FEATURE", "Parkinson_Vol")
    mega_node = MegaEvolutionNode(condition=cond, base=base, mega=mega)

    formula_str = mega_node.to_formula_str()
    assert "MEGA_SWITCH" in formula_str

    parsed_node = FormulaParser.parse(formula_str)
    assert parsed_node.op == "MEGA_SWITCH"
    assert len(parsed_node.children) == 3
    assert parsed_node.children[0].val == "Vol_ZScore"
    assert parsed_node.children[1].val == "Close_Adj"
    assert parsed_node.children[2].val == "Parkinson_Vol"
    print("[PASS] test_mega_evolution_formula_string_and_parsing")


if __name__ == "__main__":
    test_mega_evolution_ast_evaluation()
    test_mega_evolution_formula_string_and_parsing()
