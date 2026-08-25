"""Tests for the shared loss-mode target loss."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from src.module_dynamics.losses import module_target_loss, training_fit_matrix


class LossModeTest(unittest.TestCase):
    """Mean and replicate target-loss semantics."""

    def test_mean_equals_mse(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(4, 3)
        mean = torch.randn(4, 3)
        loss = module_target_loss(pred, mean, None, "mean", 1.0)
        expected = nn.functional.mse_loss(pred, mean)
        self.assertTrue(torch.allclose(loss, expected))

    def test_replicate_uses_whole_vector_nearest(self) -> None:
        # Replicates r0=[1,0], r1=[0,1]; prediction=[1,1].
        # Whole-vector nearest distance is 1 (to either replicate); a per-module
        # cherry-pick would yield 0. The box [0,0]-[1,1] contains [1,1] so box=0.
        replicates = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        pred = torch.tensor([[1.0, 1.0]])
        mean = torch.tensor([[0.5, 0.5]])
        loss = module_target_loss(pred, mean, replicates, "replicate", 1.0)
        # near = 1, /M=2 -> 0.5; box = 0 -> total 0.5
        self.assertAlmostEqual(float(loss), 0.5, places=5)

    def test_replicate_zero_when_prediction_is_a_replicate(self) -> None:
        replicates = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
        pred = torch.tensor([[1.0, 1.0]])
        mean = torch.tensor([[2.0, 2.0]])
        loss = module_target_loss(pred, mean, replicates, "replicate", 1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_replicate_box_penalizes_outside_points(self) -> None:
        replicates = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])
        mean = torch.tensor([[1.0, 1.0]])
        inside = torch.tensor([[1.0, 1.0]])
        outside = torch.tensor([[5.0, 5.0]])
        loss_in = module_target_loss(inside, mean, replicates, "replicate", 1.0)
        loss_out = module_target_loss(outside, mean, replicates, "replicate", 1.0)
        self.assertGreater(float(loss_out), float(loss_in))

    def test_training_fit_matrix_shape_by_mode(self) -> None:
        means = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
        reps = np.broadcast_to(means[:, :, None, :], (2, 5, 3, 3)).copy()
        mean_matrix = training_fit_matrix(means, reps, "mean")
        rep_matrix = training_fit_matrix(means, reps, "replicate")
        self.assertEqual(mean_matrix.shape, (6, 3))
        self.assertEqual(rep_matrix.shape, (18, 3))


if __name__ == "__main__":
    unittest.main()
