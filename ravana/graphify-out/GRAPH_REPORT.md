# Graph Report - ravana  (2026-11-16)

## Corpus Check
- 162 files · ~108,111 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 13 file(s) not represented in the graph (top: .parquet 12, .ipynb 1)

## Summary
- 2522 nodes · 4964 edges · 159 communities (35 shown, 124 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 864 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3d38e6d0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- torch
- MarketDataLoader
- .parse
- BatchPipelineRunner
- TestDataResamplingEngine
- ShoonyaBrokerGateway
- TestPhase567Chain
- IntradayFrictionModel
- ASTNode
- ingestion/config.py
- RollingADF
- BrokerOrderRouter
- TestInstitutionalFormulaAST
- .eval_node
- technical_primitives.py
- TestMarketDependencies
- TargetConfig
- .run
- TestValidationEngines
- ObservabilityDriftMonitor
- .generate
- TestFastEngine
- TestMultiHorizonTargetBuilder
- TestDataQualityEngine
- .run_walk_forward
- TestBacktestEngine
- FeatureConfig
- GeneticAlphaEngine
- numpy
- instruments
- ETFBUSpreadAligner
- ValidationPipeline
- DeliveryFrictionModel
- FeatureName
- UniverseAlphaCrown
- WalkForwardEngine
- SectorDataFeeder
- TestFormulaASTCrossSectional
- CircuitBreakerKillSwitch
- AutonomousExecutionDaemon
- TestElasticString
- TriggerSignal
- TestIndianAssetCatalog
- TestStructuralStrategiesMaster
- DataCleaningEngine
- TestAssetCatalogUniverse
- TradingRegime
- RiskGatekeeper
- StrategyBanditAllocator
- TargetHorizon
- Order
- FeatureMatrixEngine
- TestAuditRemediations
- TestZeroGivebackStopEngine
- DataFrame
- OvernightTargetBuilder
- TwoWayExpiryGammaTrigger
- TestStructuralTriggers
- BoundedASTModulator
- NSESessionGatekeeper
- TriggerAction
- TwoWaySectorDriftTrigger
- TwoWayVWAPAbsorptionTrigger
- TestTacticalExecutionAgent
- TestSQLiteStateLedger
- TestDecisionRiskShieldAdaptiveStop
- TestStrategyRiskBridge
- TwoWayLeadLagTrigger
- TwoWayMorningORBTrigger
- DecisionAuditRecord
- AllWeatherOvernightAllocator
- RateLimiter
- TwoWayINAVSpreadTrigger
- StorageConfig
- Question
- TwoWayAfternoonEuropeTrigger
- TestBoundedASTModulator
- .check_hrp_correlation_ceiling
- StrategyRiskBridge
- OrderAuditRecord
- SQLiteStateLedger
- DecisionRiskShield
- .calculate_risk_at_stop_size
- dag.py
- orders.py
- DecisionResult
- BacktestConfig
- TestIntradayMISSignalReversal
- make_meta

## God Nodes (most connected - your core abstractions)
1. `ASTNode` - 103 edges
2. `TriggerSignal` - 66 edges
3. `TestStructuralStrategiesMaster` - 45 edges
4. `FeatureMatrixEngine` - 45 edges
5. `SQLiteStateLedger` - 43 edges
6. `ShoonyaBrokerGateway` - 42 edges
7. `TriggerAction` - 37 edges
8. `Question` - 34 edges
9. `EventDrivenBacktester` - 33 edges
10. `GeminiDecisionClient` - 30 edges

## Surprising Connections (you probably didn't know these)
- `_discover()` --calls--> `GeneticAlphaEngine`  [INFERRED]
  validation/walk_forward.py → evolution/engine.py
- `AutonomousExecutionDaemon` --uses--> `UniverseAlphaCrown`  [INFERRED]
  daemon/autonomous_daemon.py → decisions/universe_allocator.py
- `AutonomousExecutionDaemon` --uses--> `DecisionVetoGuard`  [INFERRED]
  daemon/autonomous_daemon.py → decisions/veto_guard.py
- `AutonomousExecutionDaemon` --uses--> `DecisionRiskShield`  [INFERRED]
  daemon/autonomous_daemon.py → risk/decision_shield.py
- `AutonomousExecutionDaemon` --uses--> `SQLiteStateLedger`  [INFERRED]
  daemon/autonomous_daemon.py → strategies/state_ledger.py

## Import Cycles
- None detected.

## Communities (159 total, 124 thin omitted)

### Community 0 - "torch"
Cohesion: 0.06
Nodes (23): cs_rank(), cs_zscore(), div_safe_tensor(), _flatten_for_conv(), _get_decay_weights(), _get_ema_weights(), _norm_device_str(), _restore_from_conv() (+15 more)

### Community 1 - "MarketDataLoader"
Cohesion: 0.11
Nodes (3): IngestionConfig, CandlestickSanityValidator, MarketDataLoader

### Community 3 - "BatchPipelineRunner"
Cohesion: 0.13
Nodes (3): sanitize_filename(), BatchPipelineRunner, EvolutionConfig

### Community 5 - "ShoonyaBrokerGateway"
Cohesion: 0.07
Nodes (4): ShoonyaBrokerGateway, ShoonyaGatewayConfig, TokenBucketRateLimiter, TestShoonyaBrokerGateway

### Community 6 - "TestPhase567Chain"
Cohesion: 0.07
Nodes (10): PaperBrokerGateway, AllocationResult, DynamicVolatilitySizer, VolatilitySizedAllocation, TestPhase567Chain, _create_sample_sized_allocation(), CPCVResult, DSRGuardReport (+2 more)

### Community 8 - "IntradayFrictionModel"
Cohesion: 0.10
Nodes (3): IntradayFrictionModel, TestIntradayFrictionModel, TestIntradayFrictionModel

### Community 13 - "RollingADF"
Cohesion: 0.09
Nodes (3): RollingADF, RollingOUHalfLife, TestFeatureStationarity

### Community 16 - "BrokerOrderRouter"
Cohesion: 0.15
Nodes (4): BrokerOrderRouter, ExecutionAuditRecord, RoutingReceipt, CircuitBreakerStatus

### Community 20 - ".eval_node"
Cohesion: 0.15
Nodes (5): ASTEvaluator, _fast_ts_corr(), _fast_ts_decay_linear(), _fast_ts_rank_20(), _fast_ts_zscore()

### Community 22 - "technical_primitives.py"
Cohesion: 0.08
Nodes (11): bollinger_pct(), garman_klass_volatility(), lagged_volume_zscore(), macd_normalized(), normalized_atr(), parkinson_volatility(), realized_volatility(), relative_strength_index() (+3 more)

### Community 28 - "ObservabilityDriftMonitor"
Cohesion: 0.14
Nodes (3): FeatureDriftMetric, ObservabilityDriftMonitor, SystemHealthReport

### Community 31 - ".generate"
Cohesion: 0.08
Nodes (5): ASTNode, FeatureDimension, GeneticCrossover, GeneticMutator, TreeGenerator

### Community 33 - "TestFastEngine"
Cohesion: 0.06
Nodes (6): compute_fast_atr(), FastBacktester, _log_trade(), _run_numba_fast_loop(), _update_ratchet_stops(), TestFastEngine

### Community 39 - ".run_walk_forward"
Cohesion: 0.12
Nodes (7): MultiObjectiveFitnessEngine, compute_annualized_sharpe(), compute_ic_and_rank_ic(), compute_max_drawdown(), compute_oos_strategy_returns(), compute_strategy_returns(), compute_turnover()

### Community 47 - "numpy"
Cohesion: 0.13
Nodes (4): compute_dsr(), compute_moments(), compute_psr(), expected_maximum_sharpe()

### Community 48 - "instruments"
Cohesion: 0.11
Nodes (18): baskets, all_etfs, banking_sector, core_etfs, macro_drivers, nifty50_top5, instruments, asset_class (+10 more)

### Community 50 - "ValidationPipeline"
Cohesion: 0.13
Nodes (8): EvolutionResult, TestValidationEngines, CPCVConfig, MonteCarloConfig, WalkForwardConfig, CombinatorialPurgedCV, MonteCarloStressTester, ValidationPipeline

### Community 53 - "UniverseAlphaCrown"
Cohesion: 0.08
Nodes (4): DaemonConfig, _extract_bar_val(), UniverseAlphaCrown, TestUniverseAlphaCrown

### Community 55 - "WalkForwardEngine"
Cohesion: 0.16
Nodes (4): WalkForwardFold, WalkForwardResult, WalkForwardEngine, _discover()

### Community 64 - "CircuitBreakerKillSwitch"
Cohesion: 0.10
Nodes (3): RoundTripTrade, BrokerGateway, CircuitBreakerKillSwitch

### Community 66 - "TestElasticString"
Cohesion: 0.06
Nodes (5): check_snap_release(), compute_basis_tension(), ElasticPairConfig, ElasticStringEngine, TestElasticString

### Community 67 - "TriggerSignal"
Cohesion: 0.09
Nodes (4): TriggerSignal, TriggerCompositeStrategy, ThursdayExpiryGammaTrigger, TestTriggerCompositeStrategy

### Community 72 - "TradingRegime"
Cohesion: 0.29
Nodes (3): AssetType, ExecutionTiming, TradingRegime

### Community 74 - "StrategyBanditAllocator"
Cohesion: 0.08
Nodes (3): BanditArmStats, StrategyBanditAllocator, TestStrategyBanditAllocator

### Community 83 - "TestZeroGivebackStopEngine"
Cohesion: 0.07
Nodes (4): StopLayerState, ZeroGivebackStopEngine, TestBaseAbstractions, TestZeroGivebackStopEngine

### Community 85 - "DataFrame"
Cohesion: 0.14
Nodes (6): ClosingVWAPSnapbackTrigger, CrossSectorLeadLagTrigger, extract_ofi_proxy(), extract_session_vwap(), extract_spread_metrics(), extract_variance_ratio()

### Community 115 - "TestStructuralTriggers"
Cohesion: 0.08
Nodes (3): AfternoonBondEuropeTWAPTrigger, is_weekly_expiry(), TestStructuralTriggers

### Community 118 - "TriggerAction"
Cohesion: 0.13
Nodes (8): StructuralTrigger, TriggerAction, DIISIPVWAPAbsorptionTrigger, ICTUnicornStopSweepTrigger, INAVSpreadMeanReversionTrigger, OpeningRangeBreakoutTrigger, OvernightContinuationTrigger, SectorPEADInstitutionalDriftTrigger

### Community 120 - "TwoWayVWAPAbsorptionTrigger"
Cohesion: 0.12
Nodes (3): TwoWayLiquidityStopSweepTrigger, TwoWayVWAPAbsorptionTrigger, TestTwoWayAbsorptionTriggers

### Community 121 - "TestTacticalExecutionAgent"
Cohesion: 0.14
Nodes (4): StrategicIntent, TacticalExecutionAgent, TacticalExecutionResult, TestTacticalExecutionAgent

### Community 128 - "AllWeatherOvernightAllocator"
Cohesion: 0.15
Nodes (3): AllWeatherOvernightAllocator, OvernightAllocationTarget, TestAllWeatherOvernightAllocator

### Community 132 - "Question"
Cohesion: 0.06
Nodes (7): GeminiDecisionClient, DecisionType, Question, TestGeminiDecisionLiveApi, TestGeminiDecisionMockMode, TestGeminiDecisionParsing, TestGeminiDecisionSchema

### Community 138 - "SQLiteStateLedger"
Cohesion: 0.06
Nodes (3): PositionRecord, SQLiteStateLedger, TestDecisionRiskShieldPreDump

### Community 143 - "orders.py"
Cohesion: 0.27
Nodes (3): FifoLot, OrderSide, OrderStatus

### Community 144 - "DecisionResult"
Cohesion: 0.07
Nodes (5): BatchResponse, DecisionResult, DecisionVetoGuard, TestDecisionRiskShieldOvernightContinuation, TestDecisionVetoGuard

## Knowledge Gaps
- **124 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestStructuralStrategiesMaster` connect `TestStructuralStrategiesMaster` to `TestFastEngine`, `TriggerSignal`, `ShoonyaBrokerGateway`, `TestPhase567Chain`, `StrategyRiskBridge`, `StrategyBanditAllocator`, `SQLiteStateLedger`, `numpy`, `BrokerOrderRouter`, `TestZeroGivebackStopEngine`, `BoundedASTModulator`, `TestStructuralTriggers`, `TriggerAction`, `DataFrame`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Are the 38 inferred relationships involving `ASTNode` (e.g. with `TensorASTEvaluator` and `FormulaParser`) actually correct?**
  _`ASTNode` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Should `torch` be split into smaller, more focused modules?**
  _Cohesion score 0.0590990990990991 - nodes in this community are weakly interconnected._
- **Why does `ASTNode` connect `.generate` to `torch`, `.parse`, `TestBoundedASTModulator`, `TestPhase567Chain`, `.simplify`, `ASTNode`, `TestInstitutionalFormulaAST`, `.eval_node`, `.run_walk_forward`, `GeneticAlphaEngine`, `ValidationPipeline`, `WalkForwardEngine`, `TestFormulaASTCrossSectional`, `.from_dict`, `CPCVResult`, `TriggerSignal`, `TestStructuralStrategiesMaster`, `TestAuditRemediations`, `BoundedASTModulator`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `TriggerSignal` (e.g. with `TriggerCompositeStrategy` and `ElasticStringEngine`) actually correct?**
  _`TriggerSignal` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Should `MarketDataLoader` be split into smaller, more focused modules?**
  _Cohesion score 0.11494252873563218 - nodes in this community are weakly interconnected._
- **Why does `SQLiteStateLedger` connect `SQLiteStateLedger` to `AutonomousExecutionDaemon`, `TestStructuralStrategiesMaster`, `DecisionRiskShield`, `autonomous_daemon.py`, `DecisionResult`, `UniverseAlphaCrown`, `TestSQLiteStateLedger`, `TestDecisionRiskShieldAdaptiveStop`, `DecisionAuditRecord`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._