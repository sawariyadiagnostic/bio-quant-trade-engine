"""Unit tests for Multi-Asset Phenotype Vector Extractor (Wave 3: Task 3.1)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_phenotype_vector_extraction():
    """Verify PhenotypeExtractor evaluates formula across multiple asset lakes into a concatenated binary vector."""
    from src.clustering.phenotype_extractor import PhenotypeExtractor
    from src.domain.formula_parser import FormulaParser

    # Initialize extractor with 3 mock asset lakes of 100 bars each
    extractor = PhenotypeExtractor.create_mock(n_assets=3, n_bars=100)
    ast = FormulaParser.parse("Close_Adj > Open")

    vec = extractor.extract_vector(ast)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    # 3 assets x 100 bars = 300 binary position decisions
    assert vec.shape == (300,)
    assert set(np.unique(vec)).issubset({0.0, 1.0})
    print("[PASS] test_phenotype_vector_extraction")


def test_phenotype_matrix_batch_extraction():
    """Verify batch extraction across multiple formulas produces a 2D [N, D] phenotype matrix."""
    from src.clustering.phenotype_extractor import PhenotypeExtractor
    from src.domain.formula_parser import FormulaParser

    extractor = PhenotypeExtractor.create_mock(n_assets=2, n_bars=50)
    f1 = FormulaParser.parse("Close_Adj > Open")
    f2 = FormulaParser.parse("Vol_ZScore > 0.0")

    matrix = extractor.extract_batch([f1, f2])
    assert matrix.shape == (2, 100)
    assert set(np.unique(matrix)).issubset({0.0, 1.0})
    print("[PASS] test_phenotype_matrix_batch_extraction")


if __name__ == "__main__":
    test_phenotype_vector_extraction()
    test_phenotype_matrix_batch_extraction()
