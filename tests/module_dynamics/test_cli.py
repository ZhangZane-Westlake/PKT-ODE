"""Tests for module-dynamics CLI helpers and parser defaults."""

from __future__ import annotations

import argparse
import unittest

from src.module_dynamics.basic_rollout.cli import (
    build_parser as basic_parser,
    derive_repeat_seed,
    parse_train_day_weights,
)


class RepeatSeedTest(unittest.TestCase):
    """Deterministic repeat-seed derivation."""

    def test_step_is_1009(self) -> None:
        self.assertEqual(derive_repeat_seed(42, 1), 42)
        self.assertEqual(derive_repeat_seed(42, 2), 1051)
        self.assertEqual(derive_repeat_seed(42, 3), 2060)


class TrainDayWeightsTest(unittest.TestCase):
    """4D/8D weight parsing."""

    def test_valid_weights(self) -> None:
        self.assertEqual(parse_train_day_weights("4D:1,8D:2"), (1.0, 2.0))

    def test_missing_key_raises(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_train_day_weights("4D:1")

    def test_negative_weight_raises(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_train_day_weights("4D:-1,8D:2")


class ParserDefaultsTest(unittest.TestCase):
    """Basic comparator parser defaults."""

    def test_basic_parser_defaults(self) -> None:
        parser = basic_parser()
        namespace = parser.parse_args(
            ["train", "--reduction-dir", "x", "--model", "mlp", "--dynamics", "lrd"]
        )
        self.assertEqual(namespace.loss_mode, "mean")
        self.assertEqual(namespace.lambda_box, 1.0)
        self.assertEqual(namespace.conditioning, "none")

if __name__ == "__main__":
    unittest.main()
