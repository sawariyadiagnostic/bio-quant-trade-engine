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


def test_ast_canonical_hash_and_bounded_math():
    """Verify ASTNode canonical hash is invariant to commutative symmetry and math is clamped."""
    from src.domain.formula_ast import ASTNode, ASTEvaluator

    # Commutative test: ADD(A, B) vs ADD(B, A)
    node1 = ASTNode("ADD", children=[ASTNode("FEATURE", "A"), ASTNode("FEATURE", "B")])
    node2 = ASTNode("ADD", children=[ASTNode("FEATURE", "B"), ASTNode("FEATURE", "A")])
    assert node1.canonical_hash() == node2.canonical_hash()

    # Non-commutative test: SUB(A, B) vs SUB(B, A)
    sub1 = ASTNode("SUB", children=[ASTNode("FEATURE", "A"), ASTNode("FEATURE", "B")])
    sub2 = ASTNode("SUB", children=[ASTNode("FEATURE", "B"), ASTNode("FEATURE", "A")])
    assert sub1.canonical_hash() != sub2.canonical_hash()

    # Math clamping: EXP_SAFE and LOG_SAFE prevent overflow/underflow
    df = pd.DataFrame({"X": [100.0, -100.0, 0.0]})
    exp_node = ASTNode("EXP_SAFE", children=[ASTNode("FEATURE", "X")])
    exp_res = ASTEvaluator.eval_node(exp_node, df)
    assert np.all(np.isfinite(exp_res))
    assert exp_res[0] <= np.exp(20.0) + 1.0

    log_node = ASTNode("LOG_SAFE", children=[ASTNode("FEATURE", "X")])
    log_res = ASTEvaluator.eval_node(log_node, df)
    assert np.all(np.isfinite(log_res))
    print("[PASS] test_ast_canonical_hash_and_bounded_math")


def test_cross_sectional_operators():
    """Verify CS_Rank and CS_Demean compute cross-sectional alpha across multi-asset slices."""
    from src.domain.formula_ast import cs_rank, cs_demean, OPERATORS_CROSS_SECTIONAL

    assert "CS_RANK" in OPERATORS_CROSS_SECTIONAL
    assert "CS_DEMEAN" in OPERATORS_CROSS_SECTIONAL

    # 3 assets (N=3), 4 time bars (T=4)
    raw_slice = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [20.0, 10.0, 10.0, 50.0],
        [30.0, 30.0, 20.0, 10.0]
    ], dtype=np.float64)

    ranked = cs_rank(raw_slice)
    assert ranked.shape == (3, 4)
    # At t=0: values are 10, 20, 30 -> ranks should be 0.5/3, 1.5/3, 2.5/3
    assert np.isclose(ranked[0, 0], 0.5 / 3.0)
    assert np.isclose(ranked[1, 0], 1.5 / 3.0)
    assert np.isclose(ranked[2, 0], 2.5 / 3.0)

    demeaned = cs_demean(raw_slice)
    assert demeaned.shape == (3, 4)
    # Cross-sectional mean at every time step t must be exactly 0.0
    cs_sums = np.sum(demeaned, axis=0)
    assert np.allclose(cs_sums, [0.0, 0.0, 0.0, 0.0])
    print("[PASS] test_cross_sectional_operators")


if __name__ == "__main__":
    test_mega_evolution_ast_evaluation()
    test_mega_evolution_formula_string_and_parsing()
    test_ast_canonical_hash_and_bounded_math()
    test_cross_sectional_operators()
