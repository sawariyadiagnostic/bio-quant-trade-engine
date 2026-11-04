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


def test_phenotype_dense_1024_embedding():
    """Verify continuous 1024D protein-style latent embedding extraction."""
    from src.clustering.phenotype_extractor import PhenotypeExtractor
    from src.domain.formula_parser import FormulaParser

    extractor = PhenotypeExtractor.create_mock(n_assets=3, n_bars=100)
    ast = FormulaParser.parse("MEGA_SWITCH(Garman_Klass_Vol > 0.02, Close_Adj, Open)")

    emb = extractor.extract_dense_embedding(ast, target_dim=1024)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1024,)
    # Verify L2 unit norm
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

    batch_emb = extractor.extract_dense_batch([ast, ast], target_dim=1024)
    assert batch_emb.shape == (2, 1024)
    # Cosine similarity between identical strategy embeddings must be 1.0
    cos_sim = np.dot(batch_emb[0], batch_emb[1])
    assert np.isclose(cos_sim, 1.0, atol=1e-4)
    print("[PASS] test_phenotype_dense_1024_embedding")


def test_phenotype_simhash_and_crowding_tripwire():
    """Verify 1-bit SimHash bitmask extraction and crowding tripwire detection."""
    from src.clustering.phenotype_extractor import PhenotypeExtractor
    from src.domain.formula_parser import FormulaParser

    extractor = PhenotypeExtractor.create_mock(n_assets=2, n_bars=50)
    ast1 = FormulaParser.parse("MEGA_SWITCH(Garman_Klass_Vol > 0.02, Close_Adj, Open)")
    ast2 = FormulaParser.parse("MEGA_SWITCH(Garman_Klass_Vol > 0.02, Close_Adj, Open)")

    emb1 = extractor.extract_dense_embedding(ast1, target_dim=1024)
    emb2 = extractor.extract_dense_embedding(ast2, target_dim=1024)

    mask1 = extractor.extract_simhash_bitmask(emb1, n_bits=64)
    mask2 = extractor.extract_simhash_bitmask(emb2, n_bits=64)

    assert mask1.shape == (64,)
    # Identical strategies must have zero Hamming distance
    ham_dist = extractor.compute_simhash_hamming_distance(mask1, mask2)
    assert ham_dist == 0

    # Test crowding tripwire on clone embeddings
    matrix = np.stack([emb1, emb2], axis=0)
    max_sim, status = extractor.calculate_crowding_tripwire(matrix, threshold=0.88)
    assert np.isclose(max_sim, 1.0, atol=1e-3)
    assert status == "TRIPWIRE (Triggered)"
    print("[PASS] test_phenotype_simhash_and_crowding_tripwire")


if __name__ == "__main__":
    test_phenotype_vector_extraction()
    test_phenotype_matrix_batch_extraction()
    test_phenotype_dense_1024_embedding()
    test_phenotype_simhash_and_crowding_tripwire()
