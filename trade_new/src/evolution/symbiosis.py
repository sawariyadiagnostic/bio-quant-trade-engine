"""Symbiogenesis & Chimeric Cross-Breeding Engine (Wave 2: Task 2.3)."""
from enum import Enum
from typing import Set
from ..domain.formula_ast import ASTNode, MegaEvolutionNode

SENSOR_FEATURES: Set[str] = {
    "Vol_ZScore", "Parkinson_Vol", "Garman_Klass_Vol", "Downside_Vol_Ratio",
    "Dist_VWAP", "Spread", "True_Range", "TR_Expansion", "EMA_Slope",
    "Dist_EMA20", "RSI_14"
}

ACTUATOR_FEATURES: Set[str] = {
    "Close_Adj", "Open", "High", "Low", "Volume", "Vol_Price_Flow",
    "SP500_Ret", "ADR_Ret", "Global_Confluence",
    "gross_overnight_rets", "overnight_auction_jump",
    "gross_intraday_rets", "intraday_core_return",
    "gross_close_to_close_rets", "c2c"
}


class FunctionalCaste(str, Enum):
    """Functional ecological role within the alpha evolutionary collective."""
    SENSOR = "SENSOR"      # Volatility, regime, flow, and risk filters
    ACTUATOR = "ACTUATOR"  # Directional triggers, trend momentum, and entry signals
    HYBRID = "HYBRID"      # Chimeric composite organisms


class SymbioticBreeder:
    """
    Implements symbiogenesis (Lynn Margulis endosymbiosis).
    Fuses specialized Sensor and Actuator clades into high-potency chimeras.
    """
    @staticmethod
    def classify_caste(node: ASTNode) -> FunctionalCaste:
        """Classifies an AST formula into Sensor, Actuator, or Hybrid caste."""
        subnodes = node.get_all_subnodes()
        sensor_hits = 0
        actuator_hits = 0

        for n in subnodes:
            if n.op == "FEATURE":
                val_str = str(n.val)
                if val_str in SENSOR_FEATURES:
                    sensor_hits += 1
                elif val_str in ACTUATOR_FEATURES or val_str in ("Close", "Close_Raw"):
                    actuator_hits += 1

        if sensor_hits > 0 and actuator_hits == 0:
            return FunctionalCaste.SENSOR
        if actuator_hits > 0 and sensor_hits == 0:
            return FunctionalCaste.ACTUATOR
        return FunctionalCaste.HYBRID

    @staticmethod
    def fuse(
        sensor: ASTNode,
        actuator: ASTNode,
        mode: str = "multiplicative_gate"
    ) -> ASTNode:
        """Fuses a Sensor formula and an Actuator formula into a single symbiotic chimera."""
        s_clone = sensor.clone()
        a_clone = actuator.clone()

        shared_regimes = set(sensor.active_regimes) & set(actuator.active_regimes)
        effective_regimes = shared_regimes if len(shared_regimes) > 0 else (set(sensor.active_regimes) | set(actuator.active_regimes))
        s_clone.active_regimes = set(effective_regimes)
        a_clone.active_regimes = set(effective_regimes)

        if mode == "safe_ratio_gate":
            return ASTNode(op="DIV_SAFE", active_regimes=effective_regimes, children=[a_clone, s_clone])
        elif mode == "additive_blend":
            return ASTNode(op="ADD", active_regimes=effective_regimes, children=[a_clone, s_clone])
        # Default: Multiplicative gating (Signal modulated by risk filter)
        return ASTNode(op="MUL", active_regimes=effective_regimes, children=[s_clone, a_clone])

    @staticmethod
    def fuse_mega(
        sensor_condition: ASTNode,
        base_actuator: ASTNode,
        mega_actuator: ASTNode
    ) -> ASTNode:
        """Fuses formulas into a dual-state Mega Evolution chimera."""
        shared_regimes = set(sensor_condition.active_regimes) & (set(base_actuator.active_regimes) | set(mega_actuator.active_regimes))
        effective_regimes = shared_regimes if len(shared_regimes) > 0 else (set(sensor_condition.active_regimes) | set(base_actuator.active_regimes) | set(mega_actuator.active_regimes))
        return MegaEvolutionNode(
            condition=sensor_condition.clone(),
            base=base_actuator.clone(),
            mega=mega_actuator.clone(),
            active_regimes=effective_regimes
        )
