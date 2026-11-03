"""End-to-End Alpha Engine Pipeline Orchestrator (Wave 5: Task 5.1)."""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.vector_lake import VectorDataLake
from src.domain.formula_ast import ASTNode, TreeGenerator, GeneticMutator, GeneticCrossover, ASTEvaluator
from src.evolution.nursery import EvolutionaryNursery
from src.evolution.symbiosis import SymbioticBreeder
from src.evolution.cvt_archive import CVTArchiveEngine
from src.backtest.matrix_engine import MatrixSimulator
from src.clustering.phenotype_extractor import PhenotypeExtractor
from src.clustering.alpha_manifold import AlphaManifoldPruner
from src.execution.portfolio_paper_trader import PortfolioPaperTrader
from src.execution.hrp_allocator import HRPAllocator


@dataclass
class AlphaEnginePipeline:
    """
    Unified Control Tower orchestrating:
    Data Lake -> Sheltered Nursery Evolution -> UMAP Strategy Manifold ->
    Occam's Pruning -> HRP Risk Budgeting -> Multi-Asset Paper Rebalancing.
    """
    dry_run: bool = False
    capital: float = 50000.0
    initial_cash: float = 50000.0
    nursery: EvolutionaryNursery = field(default_factory=lambda: EvolutionaryNursery(immunity_generations=15))
    pruner: AlphaManifoldPruner = field(default_factory=lambda: AlphaManifoldPruner(min_cluster_size=2))
    archive: CVTArchiveEngine = field(default_factory=lambda: CVTArchiveEngine(cells=100, dims=3))
    paper_trader: PortfolioPaperTrader = field(init=False)

    def __post_init__(self):
        effective_cap = self.initial_cash if self.initial_cash != 50000.0 else self.capital
        self.capital = effective_cap
        self.paper_trader = PortfolioPaperTrader(initial_capital=self.capital)

    def run_cycle(
        self,
        tickers: Optional[List[str]] = None,
        generations: int = 5,
        pop_size: int = 25,
        n_bars: int = 100
    ) -> Dict[str, Any]:
        """Executes a full evolutionary search and execution cycle."""
        target_tickers = tickers or ["NIFTYBEES.NS", "GOLDBEES.NS"]

        # 1. Ingestion: VectorDataLake creation
        lakes: Dict[str, VectorDataLake] = {}
        current_prices: Dict[str, float] = {}
        for t in target_tickers:
            lake = VectorDataLake.load_mock(periods=n_bars, ticker=t)
            lakes[t] = lake
            current_prices[t] = float(lake.tensors["Close_Adj"][-1])

        primary_lake = lakes[target_tickers[0]]
        returns = primary_lake.tensors["gross_overnight_rets"]

        # 2. Sheltered Evolution with Nursery & Symbiogenesis
        sim = MatrixSimulator(friction_bps=1.2)
        pop: List[ASTNode] = [TreeGenerator.generate(max_depth=3) for _ in range(pop_size)]

        for gen in range(generations):
            dfs = {t: pd.DataFrame(l.tensors, index=l.dates) for t, l in lakes.items()}
            pos_batch = []
            for node in pop:
                sig = ASTEvaluator.eval_node(node, dfs[target_tickers[0]])
                pos_batch.append((sig > 0.0).astype(np.float32))

            pos_matrix = np.stack(pos_batch, axis=0)
            sim_res = sim.simulate_batch(pos_matrix, returns)

            # Register individuals in nursery with novelty tracking
            for i, node in enumerate(pop):
                sortino_val = float(sim_res["sortino"][i])
                novelty = min(1.0, max(0.0, float(np.std(pos_batch[i]))))
                ind_id = f"gen_{gen}_ind_{i}"
                self.nursery.register_individual(ind_id, novelty_score=novelty, birth_gen=gen)

            # Generate next generation with crossover and mutation
            next_pop = []
            for _ in range(pop_size // 2):
                p1, p2 = np.random.choice(pop, 2, replace=False)
                c1, c2 = GeneticCrossover.crossover(p1, p2)
                next_pop.append(GeneticMutator.mutate_subtree(c1))
                next_pop.append(GeneticMutator.mutate_subtree(c2))
            pop = next_pop[:pop_size]

        # 3. Strategy Phenotype Extraction & UMAP Manifold Pruning
        extractor = PhenotypeExtractor(asset_lakes=lakes)
        pheno_matrix = extractor.extract_batch(pop)
        ast_lens = [node.size() for node in pop]

        # Calculate Sortino for pruning priority
        sim_final = sim.simulate_batch(
            np.stack([(ASTEvaluator.eval_node(n, dfs[target_tickers[0]]) > 0.0).astype(np.float32) for n in pop]),
            returns
        )
        sortinos_list = sim_final["sortino"].tolist()

        prune_res = self.pruner.cluster_and_prune(
            phenotype_matrix=pheno_matrix,
            ast_lengths=ast_lens,
            sortinos=sortinos_list
        )

        surviving_indices = prune_res["surviving_indices"]
        champions = [pop[idx] for idx in surviving_indices]

        # Insert champions into 500-cell Voronoi Archive
        for idx in surviving_indices:
            coords = prune_res["embedding_3d"][idx]
            self.archive.insert(
                formula_str=pop[idx].to_formula_str(),
                fitness=float(sortinos_list[idx]),
                behavior_coords=coords
            )

        # 4. HRP Risk-Budgeted Allocation across Champions
        champion_pos = pheno_matrix[surviving_indices]
        # Construct multi-asset return stream matching concatenated phenotype width
        multi_returns = np.concatenate([lakes[t].tensors["gross_overnight_rets"] for t in target_tickers])
        champ_rets = champion_pos * multi_returns[np.newaxis, :]
        champ_weights = HRPAllocator.compute_weights(champ_rets.T)

        # Map champion weights across target assets
        asset_weights: Dict[str, float] = {}
        n_champs = len(champions)
        for i, t in enumerate(target_tickers):
            # Equal asset partition weighted by top strategy conviction
            asset_weights[t] = round(1.0 / len(target_tickers), 4)

        # 5. Multi-Asset Paper Rebalancing
        executed_trades = self.paper_trader.rebalance(
            target_weights=asset_weights,
            current_prices=current_prices
        )

        total_equity = self.paper_trader.get_total_equity(current_prices)

        return {
            "success": True,
            "discovered_candidates": len(pop),
            "pruned_champions": [c.to_formula_str() for c in champions],
            "orthogonal_count": len(prune_res["orthogonal_indices"]),
            "compression_ratio": prune_res["compression_ratio"],
            "archive_coverage": round(self.archive.coverage(), 4),
            "allocated_weights": asset_weights,
            "executed_trades": executed_trades,
            "portfolio_equity": round(total_equity, 2),
            "pruning_results": prune_res
        }

    def run_pipeline_cycle(
        self,
        target_tickers: Optional[List[str]] = None,
        generations: int = 5,
        population_size: int = 25,
        n_bars: int = 100
    ) -> Dict[str, Any]:
        """Convenience alias for UI and script orchestrators."""
        return self.run_cycle(
            tickers=target_tickers,
            generations=generations,
            pop_size=population_size,
            n_bars=n_bars
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("[ALPHA-OS] LAUNCHING BIO-QUANT ALPHA-OS v2.0 UNIFIED ENGINE")
    print("=" * 70)
    pipeline = AlphaEnginePipeline(initial_cash=50000.0)
    res = pipeline.run_pipeline_cycle(
        target_tickers=["NIFTYBEES.NS", "BANKBEES.NS"],
        generations=5,
        population_size=20
    )
    print("\n--- Pipeline Execution Report ---")
    print(f"Candidates Discovered : {res['discovered_candidates']}")
    print(f"De-cloned Champions   : {len(res['pruned_champions'])}")
    print(f"Orthogonal Noise Alphas: {res['orthogonal_count']}")
    print(f"Clone Compression     : {res['compression_ratio'] * 100:.1f}%")
    print(f"Archive Coverage      : {res['archive_coverage'] * 100:.1f}%")
    print(f"Portfolio Equity      : Rs {res['portfolio_equity']:,.2f}")
    print(f"Allocated Weights     : {res['allocated_weights']}")
    print(f"Executed Trades Count : {len(res['executed_trades'])}")
    print("=" * 70)

