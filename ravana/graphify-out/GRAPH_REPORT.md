# Graph Report - ravana  (2026-11-15)

## Corpus Check
- 129 files · ~72,339 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 127 file(s) not represented in the graph (top: .parquet 124, .pt 1, .log 1)
- WARNING: 127 of 256 files seen (50%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 1439 nodes · 2775 edges · 115 communities (23 shown, 92 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 514 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `95b366fb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- torch
- MarketDataLoader
- .parse
- BatchPipelineRunner
- TestDataResamplingEngine
- Order
- TestPhase567Chain
- ShoonyaBrokerGateway
- IntradayFrictionModel
- ASTNode
- ingestion/config.py
- RollingADF
- BrokerOrderRouter
- test_data_cleaner.py
- TestInstitutionalFormulaAST
- .eval_node
- technical_primitives.py
- TestMarketDependencies
- TargetConfig
- .run
- TestValidationEngines
- ObservabilityDriftMonitor
- .generate
- MonteCarloStressTester
- WalkForwardResult
- TestMultiHorizonTargetBuilder
- TestDataQualityEngine
- .run_walk_forward
- pipeline_runner.py
- EventDrivenBacktester
- FeatureConfig
- GeneticAlphaEngine
- numpy
- instruments
- ETFBUSpreadAligner
- ValidationPipeline
- DeliveryFrictionModel
- features/config.py
- WalkForwardEngine
- TestFormulaASTCrossSectional
- FeatureRegistry
- CPCVResult
- CircuitBreakerKillSwitch
- TokenBucketRateLimiter
- TestIntradayMISSignalReversal
- TestBacktestEngine
- TestIndianAssetCatalog
- TestTechnicalPrimitives
- DataCleaningEngine
- TestAssetCatalogUniverse
- TradingRegime
- RiskGatekeeper
- TargetHorizon
- Trade
- FeatureMatrixEngine
- TestAuditRemediations
- OvernightTargetBuilder

## God Nodes (most connected - your core abstractions)
1. `ASTNode` - 85 edges
2. `ShoonyaBrokerGateway` - 40 edges
3. `EventDrivenBacktester` - 33 edges
4. `Order` - 32 edges
5. `FeatureMatrixEngine` - 31 edges
6. `TestShoonyaBrokerGateway` - 31 edges
7. `GeneticAlphaEngine` - 23 edges
8. `OvernightTargetBuilder` - 23 edges
9. `TestAuditRemediations` - 23 edges
10. `TestPhase567Chain` - 23 edges

## Surprising Connections (you probably didn't know these)
- `_discover()` --calls--> `GeneticAlphaEngine`  [INFERRED]
  validation/walk_forward.py → evolution/engine.py
- `PaperBrokerGateway` --uses--> `AssetType`  [INFERRED]
  execution/router.py → backtest/config.py
- `ShoonyaBrokerGateway` --uses--> `AssetType`  [INFERRED]
  execution/shoonya_adapter.py → backtest/config.py
- `ShoonyaGatewayConfig` --uses--> `AssetType`  [INFERRED]
  execution/shoonya_adapter.py → backtest/config.py
- `TestBacktestEngine` --uses--> `AssetType`  [INFERRED]
  tests/test_backtest_engine.py → backtest/config.py

## Import Cycles
- None detected.

## Communities (115 total, 92 thin omitted)

### Community 0 - "torch"
Cohesion: 0.06
Nodes (23): cs_rank(), cs_zscore(), div_safe_tensor(), _flatten_for_conv(), _get_decay_weights(), _get_ema_weights(), _norm_device_str(), _restore_from_conv() (+15 more)

### Community 1 - "MarketDataLoader"
Cohesion: 0.06
Nodes (3): IngestionConfig, MarketDataLoader, NSESessionGatekeeper

### Community 3 - "BatchPipelineRunner"
Cohesion: 0.08
Nodes (5): enrich_intraday_for_m7(), make_meta(), sanitize_filename(), BatchPipelineRunner, EvolutionConfig

### Community 5 - "Order"
Cohesion: 0.11
Nodes (3): Order, ShoonyaGatewayConfig, TestShoonyaBrokerGateway

### Community 6 - "TestPhase567Chain"
Cohesion: 0.11
Nodes (6): AllocationResult, DynamicVolatilitySizer, VolatilitySizedAllocation, TestPhase567Chain, _create_sample_sized_allocation(), ValidationCertificate

### Community 8 - "IntradayFrictionModel"
Cohesion: 0.10
Nodes (3): IntradayFrictionModel, TestIntradayFrictionModel, TestIntradayFrictionModel

### Community 13 - "RollingADF"
Cohesion: 0.10
Nodes (3): RollingADF, RollingOUHalfLife, TestFeatureStationarity

### Community 16 - "BrokerOrderRouter"
Cohesion: 0.15
Nodes (6): OrderSide, OrderStatus, BrokerOrderRouter, ExecutionAuditRecord, RoutingReceipt, CircuitBreakerStatus

### Community 20 - ".eval_node"
Cohesion: 0.15
Nodes (5): ASTEvaluator, _fast_ts_corr(), _fast_ts_decay_linear(), _fast_ts_rank_20(), _fast_ts_zscore()

### Community 22 - "technical_primitives.py"
Cohesion: 0.10
Nodes (10): bollinger_pct(), garman_klass_volatility(), lagged_volume_zscore(), macd_normalized(), normalized_atr(), parkinson_volatility(), realized_volatility(), relative_strength_index() (+2 more)

### Community 27 - "TestValidationEngines"
Cohesion: 0.15
Nodes (3): TestValidationEngines, DSRGuardConfig, OverfittingGuardEngine

### Community 28 - "ObservabilityDriftMonitor"
Cohesion: 0.14
Nodes (3): FeatureDriftMetric, ObservabilityDriftMonitor, SystemHealthReport

### Community 31 - ".generate"
Cohesion: 0.10
Nodes (4): FeatureDimension, GeneticCrossover, GeneticMutator, TreeGenerator

### Community 33 - "MonteCarloStressTester"
Cohesion: 0.21
Nodes (3): MonteCarloConfig, MonteCarloReport, MonteCarloStressTester

### Community 39 - ".run_walk_forward"
Cohesion: 0.08
Nodes (8): MultiObjectiveFitnessEngine, compute_annualized_sharpe(), compute_ic_and_rank_ic(), compute_max_drawdown(), compute_oos_strategy_returns(), compute_strategy_returns(), compute_turnover(), _discover()

### Community 40 - "pipeline_runner.py"
Cohesion: 0.06
Nodes (3): StorageConfig, main(), PipelineDAG

### Community 47 - "numpy"
Cohesion: 0.15
Nodes (4): compute_dsr(), compute_moments(), compute_psr(), expected_maximum_sharpe()

### Community 48 - "instruments"
Cohesion: 0.11
Nodes (18): baskets, all_etfs, banking_sector, core_etfs, macro_drivers, nifty50_top5, instruments, asset_class (+10 more)

### Community 50 - "ValidationPipeline"
Cohesion: 0.19
Nodes (4): EvolutionResult, CPCVConfig, CombinatorialPurgedCV, ValidationPipeline

### Community 55 - "WalkForwardEngine"
Cohesion: 0.24
Nodes (3): WalkForwardConfig, WalkForwardFold, WalkForwardEngine

### Community 64 - "CircuitBreakerKillSwitch"
Cohesion: 0.10
Nodes (3): FifoLot, RoundTripTrade, CircuitBreakerKillSwitch

### Community 72 - "TradingRegime"
Cohesion: 0.29
Nodes (3): AssetType, ExecutionTiming, TradingRegime

### Community 78 - "Trade"
Cohesion: 0.12
Nodes (3): Trade, BrokerGateway, PaperBrokerGateway

## Knowledge Gaps
- **92 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ASTNode` connect `ASTNode` to `torch`, `.parse`, `WalkForwardResult`, `.run_walk_forward`, `.simplify`, `GeneticAlphaEngine`, `TestAuditRemediations`, `TestInstitutionalFormulaAST`, `.eval_node`, `ValidationPipeline`, `WalkForwardEngine`, `.count_nodes`, `TestFormulaASTCrossSectional`, `TestValidationEngines`, `CPCVResult`, `.from_dict`, `.generate`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `ASTNode` (e.g. with `TensorASTEvaluator` and `FormulaParser`) actually correct?**
  _`ASTNode` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Should `torch` be split into smaller, more focused modules?**
  _Cohesion score 0.0590990990990991 - nodes in this community are weakly interconnected._
- **Why does `TestAuditRemediations` connect `TestAuditRemediations` to `MarketDataLoader`, `DataCleaningEngine`, `TradingRegime`, `EventDrivenBacktester`, `ASTNode`, `RiskGatekeeper`, `RollingADF`, `.eval_node`, `technical_primitives.py`, `OvernightTargetBuilder`, `test_audit_remediation.py`, `TargetConfig`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `ShoonyaBrokerGateway` (e.g. with `AssetType` and `Order`) actually correct?**
  _`ShoonyaBrokerGateway` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Should `MarketDataLoader` be split into smaller, more focused modules?**
  _Cohesion score 0.06448979591836734 - nodes in this community are weakly interconnected._
- **Why does `FeatureMatrixEngine` connect `FeatureMatrixEngine` to `BatchPipelineRunner`, `FeatureConfig`, `RollingADF`, `features/config.py`, `test_features_matrix.py`, `TestMarketDependencies`, `FeatureRegistry`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._