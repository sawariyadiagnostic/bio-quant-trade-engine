"""RAVANA Structural Strategies, Risk Adapters, and State Machine Package."""

from ravana.strategies.base import (
    StopLayerState,
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
    ZeroGivebackStopEngine,
)
from ravana.strategies.elastic_string import (
    CORE_ELASTIC_PAIRS,
    CORE_ELASTIC_PAIRS_MAP,
    ElasticPairConfig,
    ElasticStringEngine,
    ElasticStringTrigger,
    check_snap_release,
    compute_basis_tension,
    generate_signals,
)
from ravana.strategies.composite import (
    ALL_TRIGGER_NAMES,
    BUCKET_1_NAMES,
    BUCKET_2_NAMES,
    BUCKET_3_NAMES,
    BanditArmStats,
    StrategyBanditAllocator,
    TriggerCompositeStrategy,
)
from ravana.strategies.modulator import (
    BoundedASTModulator,
)
from ravana.strategies.risk_adapter import (
    CorrelationCheckResult,
    RiskBridgeDecision,
    SizingResult,
    StrategyRiskBridge,
)
from ravana.strategies.state_ledger import (
    OrderAuditRecord,
    PositionRecord,
    SQLiteStateLedger,
)
from ravana.strategies.two_way_orb import TwoWayMorningORBTrigger
from ravana.strategies.two_way_inav_spread import TwoWayINAVSpreadTrigger
from ravana.strategies.two_way_pead_drift import TwoWaySectorDriftTrigger
from ravana.strategies.two_way_absorption import (
    TwoWayVWAPAbsorptionTrigger,
    TwoWayLiquidityStopSweepTrigger,
)
from ravana.strategies.two_way_afternoon_europe import TwoWayAfternoonEuropeTrigger
from ravana.strategies.two_way_chop_reversion import TwoWayChopMeanReversionTrigger
from ravana.strategies.volatility_scaler import AdaptiveThresholdScaler
from ravana.strategies.two_way_expiry_gamma import TwoWayExpiryGammaTrigger
from ravana.strategies.two_way_lead_lag import TwoWayLeadLagTrigger
from ravana.strategies.triggers import (
    AfternoonBondEuropeTWAPTrigger,
    ClosingVWAPSnapbackTrigger,
    CrossSectorLeadLagTrigger,
    DIISIPVWAPAbsorptionTrigger,
    ICTUnicornStopSweepTrigger,
    INAVSpreadMeanReversionTrigger,
    OpeningRangeBreakoutTrigger,
    OvernightContinuationTrigger,
    SectorPEADInstitutionalDriftTrigger,
    ThursdayExpiryGammaTrigger,
    extract_ofi_proxy,
    extract_session_vwap,
    extract_spread_metrics,
    extract_variance_ratio,
)

__all__ = [
    # Base abstractions
    "TriggerAction",
    "StopLayerState",
    "TriggerSignal",
    "StructuralTrigger",
    "ZeroGivebackStopEngine",
    # Hardened Two-Way Triggers
    "TwoWayMorningORBTrigger",
    "TwoWayINAVSpreadTrigger",
    "TwoWaySectorDriftTrigger",
    "TwoWayVWAPAbsorptionTrigger",
    "TwoWayLiquidityStopSweepTrigger",
    "TwoWayAfternoonEuropeTrigger",
    "TwoWayExpiryGammaTrigger",
    "TwoWayLeadLagTrigger",
    "TwoWayChopMeanReversionTrigger",
    "AdaptiveThresholdScaler",
    # 10 Hardened Structural Triggers
    "OpeningRangeBreakoutTrigger",
    "ThursdayExpiryGammaTrigger",
    "CrossSectorLeadLagTrigger",
    "SectorPEADInstitutionalDriftTrigger",
    "INAVSpreadMeanReversionTrigger",
    "ClosingVWAPSnapbackTrigger",
    "DIISIPVWAPAbsorptionTrigger",
    "ICTUnicornStopSweepTrigger",
    "AfternoonBondEuropeTWAPTrigger",
    "OvernightContinuationTrigger",
    # Modulator & Composite
    "BoundedASTModulator",
    "BanditArmStats",
    "StrategyBanditAllocator",
    "TriggerCompositeStrategy",
    "ALL_TRIGGER_NAMES",
    "BUCKET_1_NAMES",
    "BUCKET_2_NAMES",
    "BUCKET_3_NAMES",
    # Microstructure Extractor Helpers
    "extract_variance_ratio",
    "extract_ofi_proxy",
    "extract_session_vwap",
    "extract_spread_metrics",
    # Risk Adapter & State Ledger (Worker 4)
    "CorrelationCheckResult",
    "RiskBridgeDecision",
    "SizingResult",
    "StrategyRiskBridge",
    "PositionRecord",
    "OrderAuditRecord",
    "SQLiteStateLedger",
    # Elastic String Arbitrage Engine (Worker 2)
    "ElasticPairConfig",
    "CORE_ELASTIC_PAIRS",
    "CORE_ELASTIC_PAIRS_MAP",
    "ElasticStringEngine",
    "ElasticStringTrigger",
    "compute_basis_tension",
    "check_snap_release",
    "generate_signals",
]
