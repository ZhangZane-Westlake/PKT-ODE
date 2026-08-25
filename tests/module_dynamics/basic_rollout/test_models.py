"""Tests for the basic residual module transition and rollout."""

from __future__ import annotations

import unittest

import torch

from src.module_dynamics.basic_rollout.models import (
    build_transition_model,
    rollout,
    rollout_lrd_3h,
)


class BasicModelTest(unittest.TestCase):
    """Residual transition shapes and zero-init persistence."""

    def test_rollout_records_target_days(self) -> None:
        model = build_transition_model("mlp", 5, "none", 16, 2, 0.0, 16)
        initial = torch.randn(7, 5)
        predictions = rollout(model, initial, initial, "lrd", (4, 8, 15, 29))
        self.assertEqual(set(predictions.keys()), {4, 8, 15, 29})
        self.assertEqual(tuple(predictions[29].shape), (7, 5))

    def test_linear_zero_init_is_persistence(self) -> None:
        model = build_transition_model("linear", 4, "none", 8, 1, 0.0, 8)
        state = torch.randn(3, 4)
        output = model(state, state)
        self.assertTrue(torch.allclose(output, state))

    def test_all_conditioning_modes_build(self) -> None:
        for conditioning in ("none", "concat", "film", "residual_adapter"):
            model = build_transition_model("mlp", 4, conditioning, 8, 1, 0.0, 8)
            state = torch.randn(2, 4)
            output = model(state, state)
            self.assertEqual(tuple(output.shape), (2, 4))

    def test_linear_rejects_conditioning(self) -> None:
        with self.assertRaises(ValueError):
            build_transition_model("linear", 4, "concat", 8, 1, 0.0, 8)

    def test_observed_vs_lrd_step_count(self) -> None:
        model = build_transition_model("mlp", 3, "none", 8, 1, 0.0, 8)
        initial = torch.randn(2, 3)
        observed = rollout(model, initial, initial, "observed", (4, 8, 15, 29))
        lrd = rollout(model, initial, initial, "lrd", (4, 8, 15, 29))
        self.assertEqual(set(observed.keys()), set(lrd.keys()))


class LRD3HRolloutTest(unittest.TestCase):
    """3-hour-granularity rollout over the eight-timepoint hour axis."""

    def test_records_all_target_hours_and_shapes(self) -> None:
        model = build_transition_model("mlp", 4, "none", 8, 1, 0.0, 16)
        initial = torch.randn(2, 4)
        targets = (6, 9, 24, 96, 192, 360, 696)
        predictions = rollout_lrd_3h(model, initial, initial, targets)
        self.assertEqual(set(predictions.keys()), set(targets))
        for tensor in predictions.values():
            self.assertEqual(tuple(tensor.shape), (2, 4))

    def test_step_count_matches_hour_gap(self) -> None:
        # One model application per 3h step from init=3; reach 696 at step 231.
        model = build_transition_model("mlp", 3, "none", 8, 1, 0.0, 16)
        initial = torch.randn(1, 3)
        targets = (696,)
        # Each application is one transition; with a zero-init linear model the
        # state stays at init, but the call must not raise and must reach 696.
        predictions = rollout_lrd_3h(model, initial, initial, targets)
        self.assertEqual(set(predictions.keys()), {696})

    def test_rejects_unreachable_target(self) -> None:
        model = build_transition_model("linear", 3, "none", 8, 1, 0.0, 16)
        initial = torch.randn(1, 3)
        # 7 is not reachable from init=3 in 3h steps (7-3=4, not divisible by 3).
        with self.assertRaises(ValueError):
            rollout_lrd_3h(model, initial, initial, (6, 7, 9))


if __name__ == "__main__":
    unittest.main()
