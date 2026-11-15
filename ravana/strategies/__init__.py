"""RAVANA Structural Strategies, Risk Adapters, and State Machine Package."""

from ravana.strategies.base import (
    StopLayerState,
    StructuralTrigger,
    TriggerAction,
    TriggerSignal,
    ZeroGivebackStopEngine,
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
]
