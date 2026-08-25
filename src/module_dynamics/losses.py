"""Shared target-loss modes for module dynamics.

Both rollout packages delegate the per-``(treatment, time)`` loss term here via
``--loss-mode``:

* ``mean`` collapses the three replicates to their mean and uses MSE.
* ``replicate`` keeps all three replicates, confines the prediction to their
  componentwise spread, and pulls it toward the **single nearest replicate**
  using whole-vector distance (one replicate for all modules, never a
  per-module cherry-pick).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn


LossMode = Literal["mean", "replicate"]
VALID_LOSS_MODES: tuple[str, ...] = ("mean", "replicate")


def training_fit_matrix(
    mean_values: np.ndarray,
    replicate_values: np.ndarray,
    mode: LossMode,
) -> np.ndarray:
    """Build the leakage-safe scaler fit matrix for a loss mode.

    Uses only the first three days (1D/4D/8D). In ``replicate`` mode the three
    replicates are pooled so the scaler sees the true biological spread.

    Args:
        mean_values: Treatment-by-day-by-module replicate means.
        replicate_values: Treatment-by-day-by-replicate-by-module scores.
        mode: Target loss mode.

    Returns:
        Two-dimensional fit matrix with modules on the final axis.

    Raises:
        ValueError: If ``mode`` is unsupported or shapes are inconsistent.
    """

    if mode not in VALID_LOSS_MODES:
        raise ValueError(f"unsupported loss mode: {mode}")
    if mean_values.ndim != 3:
        raise ValueError("mean_values must be treatment x day x module")
    module_dim = mean_values.shape[-1]
    if replicate_values.shape[-1] != module_dim:
        raise ValueError("replicate and mean module axes disagree")
    if mode == "replicate":
        if replicate_values.ndim != 4:
            raise ValueError("replicate_values must be treatment x day x rep x module")
        return replicate_values[:, :3].reshape(-1, module_dim)
    return mean_values[:, :3].reshape(-1, module_dim)


def module_target_loss(
    prediction: torch.Tensor,
    mean_target: torch.Tensor,
    replicate_target: torch.Tensor | None,
    mode: LossMode,
    lambda_box: float,
) -> torch.Tensor:
    """Compute the per-``(treatment, time)`` module target loss.

    Args:
        prediction: Predicted module vectors of shape ``[B, M]``.
        mean_target: Replicate-mean target vectors of shape ``[B, M]``.
        replicate_target: Replicate target tensors of shape ``[B, R, M]``
            (required for ``replicate`` mode).
        mode: ``mean`` or ``replicate``.
        lambda_box: Soft box-barrier strength for ``replicate`` mode.

    Returns:
        Scalar loss comparable in scale to MSE.

    Raises:
        ValueError: If ``mode`` is unsupported or replicates are missing.
    """

    if mode == "mean":
        return nn.functional.mse_loss(prediction, mean_target)
    if mode != "replicate":
        raise ValueError(f"unsupported loss mode: {mode}")
    if replicate_target is None:
        raise ValueError("replicate mode requires replicate targets")
    module_dim = prediction.shape[-1]
    diff = prediction.unsqueeze(1) - replicate_target  # [B, R, M]
    nearest = diff.pow(2).sum(dim=-1).min(dim=1).values  # [B] whole-vector nearest
    lower = replicate_target.min(dim=1).values  # [B, M]
    upper = replicate_target.max(dim=1).values  # [B, M]
    over = torch.relu(prediction - upper)
    under = torch.relu(lower - prediction)
    box = (over.pow(2) + under.pow(2)).mean(dim=-1)  # [B]
    loss = nearest / module_dim + lambda_box * box
    return loss.mean()


def mean_mse_for_reporting(
    prediction: np.ndarray,
    mean_target: np.ndarray,
) -> float:
    """Compute a plain MSE against replicate means for cross-model reporting.

    Args:
        prediction: Predicted module vectors.
        mean_target: Replicate-mean target vectors.

    Returns:
        Mean squared error.
    """

    return float(np.mean(np.square(prediction - mean_target)))


__all__ = [
    "LossMode",
    "VALID_LOSS_MODES",
    "mean_mse_for_reporting",
    "module_target_loss",
    "training_fit_matrix",
]
