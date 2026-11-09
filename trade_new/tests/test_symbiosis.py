"""Unit tests for Symbiogenesis & Chimeric Cross-Breeding (Wave 2: Task 2.3)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_symbiotic_chimera_fusion():
    """Verify SymbioticBreeder categorizes castes and fuses Sensor + Actuator into a chimera."""
    from src.evolution.symbiosis import SymbioticBreeder, FunctionalCaste
    from src.domain.formula_parser import FormulaParser

    sensor_ast = FormulaParser.parse("Vol_ZScore / (Garman_Klass_Vol + 1e-6)")
    actuator_ast = FormulaParser.parse("DELTA_5(Close_Adj) - Open")

    # 1. Caste Detection
    caste_sensor = SymbioticBreeder.classify_caste(sensor_ast)
    caste_actuator = SymbioticBreeder.classify_caste(actuator_ast)

    assert caste_sensor == FunctionalCaste.SENSOR
    assert caste_actuator == FunctionalCaste.ACTUATOR

    # 2. Chimeric Fusion
    chimera = SymbioticBreeder.fuse(sensor=sensor_ast, actuator=actuator_ast, mode="multiplicative_gate")
    formula_str = chimera.to_formula_str()

    assert "Vol_ZScore" in formula_str
    assert "Close_Adj" in formula_str
    assert chimera.depth() <= max(sensor_ast.depth(), actuator_ast.depth()) + 2
    print("[PASS] test_symbiotic_chimera_fusion")


def test_symbiotic_regime_switch_chimera():
    """Verify SymbioticBreeder can fuse into a conditional MegaEvolution chimera."""
    from src.evolution.symbiosis import SymbioticBreeder
    from src.domain.formula_parser import FormulaParser

    sensor = FormulaParser.parse("Vol_ZScore - 2.0")
    actuator_base = FormulaParser.parse("EMA_10(Close_Adj)")
    actuator_mega = FormulaParser.parse("Parkinson_Vol * 5.0")

    mega_chimera = SymbioticBreeder.fuse_mega(sensor, actuator_base, actuator_mega)
    formula_str = mega_chimera.to_formula_str()

    assert "MEGA_SWITCH" in formula_str
    assert mega_chimera.op == "MEGA_SWITCH"
    assert len(mega_chimera.children) == 3
    print("[PASS] test_symbiotic_regime_switch_chimera")


def load_tests(loader, standard_tests, pattern):
    import inspect
    import unittest
    suite = unittest.TestSuite()
    current_module = sys.modules[__name__]
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(func))
    return suite


if __name__ == "__main__":
    import unittest
    unittest.main()

