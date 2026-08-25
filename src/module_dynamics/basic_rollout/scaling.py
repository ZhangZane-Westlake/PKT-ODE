"""Leakage-safe feature scaling for module eigengene states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureScaler:
    """Per-feature scaler fit only on 1D, 4D, and 8D.

    Args:
        mean: Training-day feature means.
        scale: Training-day population standard deviations.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        """Fit statistics on the first three model days.

        Args:
            values: Treatment-by-day-by-feature values.

        Returns:
            Fitted scaler.
        """

        if values.ndim != 3 or values.shape[1] < 3:
            raise ValueError("values must be treatment x at-least-three-days x feature")
        training = values[:, :3].reshape(-1, values.shape[-1])
        mean = training.mean(axis=0)
        scale = training.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean.astype(np.float64), scale.astype(np.float64))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "FeatureScaler":
        """Fit directly on a pooled two-dimensional training matrix.

        Args:
            matrix: Rows of training feature vectors with modules on axis -1.

        Returns:
            Fitted scaler.
        """

        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("matrix must be a non-empty two-dimensional array")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean.astype(np.float64), scale.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardize values on their final feature axis.

        Args:
            values: Raw feature values.

        Returns:
            Standardized values.
        """

        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Restore raw values.

        Args:
            values: Standardized feature values.

        Returns:
            Raw values.
        """

        return values * self.scale + self.mean

    def to_dict(self, feature_name: str, feature_ids: Sequence[str]) -> dict[str, object]:
        """Serialize scaler statistics with an explicit feature axis.

        Args:
            feature_name: Axis label such as modules.
            feature_ids: Ordered feature identifiers.

        Returns:
            JSON-compatible scaler record.
        """

        return {
            "fit_days": ["1D", "4D", "8D"],
            feature_name: list(feature_ids),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }
