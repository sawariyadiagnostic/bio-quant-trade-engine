"""Muscle 7: Market Vocabulary Feature Matrix Engine.

Exports:
- FeatureMatrixEngine: Main orchestrator for 2D matrices and 3D tensors.
- FeatureConfig: Hyperparameter and lookback window configuration.
- FeatureRegistry: Registry and definitions of 21 stationary features.
- FeatureName: Enum of feature column identifiers.
- RollingADF: Augmented Dickey-Fuller rolling stationarity t-statistic estimator.
- RollingOUHalfLife: Ornstein-Uhlenbeck rolling mean-reversion half-life estimator.
"""

from ravana.features.config import FeatureConfig, FeatureName, FeatureRegistry
from ravana.features.matrix_engine import FeatureMatrixEngine
from ravana.features.stationarity import RollingADF, RollingOUHalfLife

__all__ = [
    "FeatureConfig",
    "FeatureName",
    "FeatureRegistry",
    "FeatureMatrixEngine",
    "RollingADF",
    "RollingOUHalfLife",
]
