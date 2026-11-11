# Graph Report - ravana  (2026-11-11)

## Corpus Check
- 104 files · ~50,965 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: .ipynb 1)

## Summary
- 1580 nodes · 3025 edges · 93 communities (28 shown, 65 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 405 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e18e582f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- torch
- MarketDataLoader
- .parse
- BatchPipelineRunner
- TestDataResamplingEngine
- TestRiskEngine
- 4. Pending Improvements for Future Muscles (To Implement Next)
- NSESessionGatekeeper
- ASTNode
- TimeGridReindexer
- .validate
- RollingADF
- .compare_partition
- TestDataStorageEngine
- formula_ast.py
- numpy
- TestInstitutionalFormulaAST
- .read
- technical_primitives.py
- KaggleBurstDispatcher
- GeneticAlphaEngine
- OvernightTargetBuilder
- EventDrivenBacktester
- ResamplingEngine
- backtest/engine.py
- .upsert
- .generate
- IdeaGraveyard
- AssetRegistry
- CleanedDataset
- RollingADF
- FlashSpikeFilter
- .evaluate_single
- TestDataQualityEngine
- metrics.py
- TestBacktestEngine
- FeatureMatrixEngine
- ResampledDataset
- IngestedDataset
- SessionAwareResampler
- Individual
- deflated_sharpe.py
- instruments
- ETFBUSpreadAligner
- .evaluate_population
- DeliveryFrictionModel
- FeatureRegistry
- HierarchicalRiskParity
- .resolve_path
- schema.py
- orders.py
- TestMultiObjectiveFitness
- MicrostructureSlippageModel
- DataCleaningEngine
- .calculate_friction
- PortfolioRiskEngine
- pareto.py
- numpy
- UniverseSelector
- AssetClassUniverse
- TestUniverseRegistry
- BasketUniverse
- DrawdownState
- fdr.py
- _parse_atomic_spec

## God Nodes (most connected - your core abstractions)
1. `ASTNode` - 85 edges
2. `AssetRegistry` - 30 edges
3. `TestDataResamplingEngine` - 25 edges
4. `GeneticAlphaEngine` - 25 edges
5. `Instrument` - 24 edges
6. `EventDrivenBacktester` - 23 edges
7. `LakeManager` - 22 edges
8. `TestDataStorageEngine` - 21 edges
9. `CleanedDataset` - 21 edges
10. `FeatureMatrixEngine` - 21 edges

## Surprising Connections (you probably didn't know these)
- `1.2 ETF iNAV & Tracking Spread Arbitrage` --references--> `IngestedDataset`  [INFERRED]
  ravana/MICROSTRUCTURE_ALPHA_AND_ROADMAP_VAULT.md → ravana/data/ingestion/dataset.py
- `Muscle 6: Sovereign Columnar Parquet Data Lake (`ravana/data/lake/`)` --references--> `LakeManager`  [INFERRED]
  ravana/MICROSTRUCTURE_ALPHA_AND_ROADMAP_VAULT.md → ravana/data/storage/manager.py
- `Muscle 4 Implementation: Data Quality, Time-Grid Regularization & Spread Aligner (Completed & Deep-Audited)` --references--> `TimeGridReindexer`  [INFERRED]
  ravana/MICROSTRUCTURE_ALPHA_AND_ROADMAP_VAULT.md → ravana/data/cleaner/grid_reindexer.py
- `Phase 1: Data Pipeline & Microstructure Layer (Muscles 3, 4, 5) — 100% Complete & Sealed` --references--> `TimeGridReindexer`  [INFERRED]
  ravana/MICROSTRUCTURE_ALPHA_AND_ROADMAP_VAULT.md → ravana/data/cleaner/grid_reindexer.py
- `Muscle 5 Implementation: Causal Multi-Timeframe Resampler (Completed & Battle-Hardened)` --references--> `ResamplingEngine`  [INFERRED]
  ravana/MICROSTRUCTURE_ALPHA_AND_ROADMAP_VAULT.md → ravana/data/resampler/resampler.py

## Import Cycles
- None detected.

## Communities (93 total, 65 thin omitted)

### Community 0 - "torch"
Cohesion: 0.11
Nodes (19): div_safe_tensor(), _flatten_for_conv(), _get_decay_weights(), _get_ema_weights(), _norm_device_str(), _restore_from_conv(), ts_argmax(), ts_argmin() (+11 more)

### Community 1 - "MarketDataLoader"
Cohesion: 0.07
Nodes (4): IngestionConfig, CandlestickSanityValidator, MarketDataLoader, TestMarketDataLoader

### Community 2 - ".parse"
Cohesion: 0.09
Nodes (5): ComputeDispatcher, EvaluationContext, FormulaParser, TestComputeDispatcher, TestMuscle1Muscle2Coordination

### Community 3 - "BatchPipelineRunner"
Cohesion: 0.06
Nodes (6): enrich_intraday_for_m7(), make_meta(), sanitize_filename(), BatchPipelineRunner, main(), PipelineDAG

### Community 6 - "TestRiskEngine"
Cohesion: 0.08
Nodes (3): RiskConfig, RiskGatekeeper, TestRiskEngine

### Community 7 - "4. Pending Improvements for Future Muscles (To Implement Next)"
Cohesion: 0.07
Nodes (27): 1.1 The 14:00 – 15:15 IST Mutual Fund / DII Inflow Mechanism, 1.2 ETF iNAV & Tracking Spread Arbitrage, 1.2 The 09:15 – 10:00/10:30 IST Morning Opening Momentum & Gap-and-Go Drive, 1.3 Concessional Indian Statutory Friction Schedule for ETFs, 1.4 Exchange Hours & Microstructure Conventions, 1.5 The Depository Participant (DP) Flat Charge Reality & Viable Capital Sizing, 1.6 Single-Asset Directional Alpha vs. Cross-Sectional Relative Value, 1.7 Order Book Crossing & Opening Auction Slippage Realities (+19 more)

### Community 10 - "ASTNode"
Cohesion: 0.07
Nodes (3): TensorASTEvaluator, ASTNode, TestTensorASTEvaluator

### Community 15 - ".compare_partition"
Cohesion: 0.12
Nodes (4): ColumnDiff, LakeDiffInspector, LakePartitionDiff, TestLakeDiffInspector

### Community 19 - "numpy"
Cohesion: 0.10
Nodes (3): fetch_and_prepare_5m_data(), compute_real_features(), load_real_data_tensor()

### Community 20 - "TestInstitutionalFormulaAST"
Cohesion: 0.10
Nodes (5): ASTEvaluator, _fast_ts_decay_linear(), _fast_ts_rank_20(), run_benchmark(), TestInstitutionalFormulaAST

### Community 22 - "technical_primitives.py"
Cohesion: 0.09
Nodes (12): bollinger_pct(), garman_klass_volatility(), lagged_volume_zscore(), log_returns(), macd_normalized(), normalized_atr(), parkinson_volatility(), realized_volatility() (+4 more)

### Community 23 - "KaggleBurstDispatcher"
Cohesion: 0.08
Nodes (4): KaggleBurstConfig, KaggleBurstDispatcher, KaggleQuotaInfo, TestKaggleRunner

### Community 24 - "GeneticAlphaEngine"
Cohesion: 0.16
Nodes (3): EvolutionConfig, GeneticAlphaEngine, TestGeneticAlphaEvolution

### Community 25 - "OvernightTargetBuilder"
Cohesion: 0.15
Nodes (3): OvernightTargetBuilder, TargetConfig, TestOvernightTargetBuilder

### Community 26 - "EventDrivenBacktester"
Cohesion: 0.14
Nodes (4): EventDrivenBacktester, Order, Position, Trade

### Community 31 - ".generate"
Cohesion: 0.11
Nodes (6): FeatureDimension, GeneticCrossover, GeneticMutator, TreeGenerator, run_intraday_benchmark(), run_benchmark()

### Community 36 - "FlashSpikeFilter"
Cohesion: 0.16
Nodes (6): FlashSpikeFilter, 3. Master Chronological Improvement & Implementation Log, Muscle 4 Implementation: Data Quality, Time-Grid Regularization & Spread Aligner (Completed & Deep-Audited), Phase 0: Foundations & Domain AST (Muscle 1) — 100% Complete, Phase 0: High-Performance GPU Tensor Engine (Muscle 2) — 100% Complete & Cloud-Certified, Phase 1: Data Pipeline & Microstructure Layer (Muscles 3, 4, 5) — 100% Complete & Sealed

### Community 37 - ".evaluate_single"
Cohesion: 0.18
Nodes (3): AlphaMetrics, FitnessConfig, MultiObjectiveFitnessEngine

### Community 39 - "metrics.py"
Cohesion: 0.18
Nodes (5): compute_annualized_sharpe(), compute_ic_and_rank_ic(), compute_max_drawdown(), compute_strategy_returns(), compute_turnover()

### Community 42 - "FeatureMatrixEngine"
Cohesion: 0.21
Nodes (3): FeatureConfig, FeatureMatrixEngine, TestFeatureMatrixEngine

### Community 47 - "deflated_sharpe.py"
Cohesion: 0.27
Nodes (4): compute_dsr(), compute_moments(), compute_psr(), expected_maximum_sharpe()

### Community 48 - "instruments"
Cohesion: 0.11
Nodes (18): baskets, all_etfs, banking_sector, core_etfs, macro_drivers, nifty50_top5, instruments, asset_class (+10 more)

### Community 55 - ".resolve_path"
Cohesion: 0.12
Nodes (3): StorageConfig, LakeManager, IncrementalUpsertEngine

### Community 64 - "orders.py"
Cohesion: 0.21
Nodes (4): FifoLot, OrderSide, OrderStatus, RoundTripTrade

### Community 72 - ".calculate_friction"
Cohesion: 0.15
Nodes (5): AssetType, ExecutionTiming, TradingRegime, FrictionBreakdown, IndianFrictionModel

### Community 78 - "UniverseSelector"
Cohesion: 0.24
Nodes (3): IntersectionUniverse, UnionUniverse, UniverseSelector

### Community 80 - "AssetClassUniverse"
Cohesion: 0.23
Nodes (3): AssetClass, ExchangeType, AssetClassUniverse

### Community 89 - "_parse_atomic_spec"
Cohesion: 0.25
Nodes (3): _parse_atomic_spec(), parse_universe_spec(), SymbolUniverse

## Knowledge Gaps
- **65 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ASTNode` connect `ASTNode` to `torch`, `TestMultiObjectiveFitness`, `.parse`, `dataclasses`, `.evaluate_single`, `.simplify`, `Individual`, `formula_ast.py`, `.evaluate_population`, `TestInstitutionalFormulaAST`, `.count_nodes`, `GeneticAlphaEngine`, `.to_infix`, `._generate_random_tree`, `.from_dict`, `.generate`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `ASTNode` (e.g. with `ComputeDispatcher` and `TensorASTEvaluator`) actually correct?**
  _`ASTNode` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Should `torch` be split into smaller, more focused modules?**
  _Cohesion score 0.11400966183574879 - nodes in this community are weakly interconnected._
- **Why does `GeneticAlphaEngine` connect `GeneticAlphaEngine` to `IdeaGraveyard`, `.parse`, `BatchPipelineRunner`, `ASTNode`, `._compute_ic`, `Individual`, `.evaluate_population`, `numpy`, `FeatureRegistry`, `._generate_random_tree`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Should `MarketDataLoader` be split into smaller, more focused modules?**
  _Cohesion score 0.07399577167019028 - nodes in this community are weakly interconnected._
- **Why does `IngestedDataset` connect `IngestedDataset` to `MarketDataLoader`, `BatchPipelineRunner`, `dataclasses`, `DataCleaningEngine`, `4. Pending Improvements for Future Muscles (To Implement Next)`, `TestDataQualityEngine`, `TimeGridReindexer`, `.clean`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Should `.parse` be split into smaller, more focused modules?**
  _Cohesion score 0.09291521486643438 - nodes in this community are weakly interconnected._