"""Tests for the shared reduction-dir loader and scope parser."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.module_dynamics.common import (
    BASIC_MODEL_DAYS,
    DAY_TO_INDEX,
    five_day_view,
    load_reduction_trajectories,
    module_axis_sha256,
    parse_reduction_scope,
)
from tests.module_dynamics._fixtures import write_synthetic_reduction


class ScopeParseTest(unittest.TestCase):
    """Reduction-dir path parsing."""

    def test_compound_scope(self) -> None:
        path = Path(
            "output/gene_module_reduction/gene_selection_20260731/liver/"
            "compounds/fenofibrate/tight_stage/reductions/r2-0p8_x"
        )
        scope = parse_reduction_scope(path)
        self.assertTrue(scope.is_compound_scope)
        self.assertEqual(scope.organ, "liver")
        self.assertEqual(scope.compound_slug, "fenofibrate")
        self.assertEqual(scope.split_id, "tight_stage")
        self.assertEqual(scope.config_id, "r2-0p8_x")
        self.assertEqual(scope.scope_tag, "compounds/fenofibrate")

    def test_organ_scope(self) -> None:
        path = Path(
            "output/gene_module_reduction/gene_selection_20260731/kidney/"
            "loose_stage/reductions/r2-0p8_y"
        )
        scope = parse_reduction_scope(path)
        self.assertFalse(scope.is_compound_scope)
        self.assertEqual(scope.organ, "kidney")
        self.assertIsNone(scope.compound_slug)
        self.assertEqual(scope.split_id, "loose_stage")
        self.assertEqual(scope.scope_tag, "kidney")

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_reduction_scope(Path("not/a/reduction/dir"))


class TrajectoryLoadTest(unittest.TestCase):
    """End-to-end load of a synthetic reduction directory."""

    def test_load_compound_scope(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reduction_dir = write_synthetic_reduction(Path(tmp), n_compounds=1)
            data = load_reduction_trajectories(reduction_dir)
            self.assertEqual(len(data.treatment_ids), 2)
            self.assertEqual(len(data.module_ids), 3)
            self.assertEqual(len(data.model_times), 8)
            self.assertEqual(data.mean_values.shape, (2, 8, 3))
            self.assertEqual(data.replicate_values.shape, (2, 8, 3, 3))
            self.assertTrue(np.isfinite(data.mean_values).all())
            self.assertEqual(len(module_axis_sha256(data)), 64)

    def test_five_day_view(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reduction_dir = write_synthetic_reduction(Path(tmp), n_compounds=2)
            data = load_reduction_trajectories(reduction_dir)
            means, reps, sds = five_day_view(data)
            self.assertEqual(means.shape, (4, 5, 3))
            self.assertEqual(reps.shape, (4, 5, 3, 3))
            self.assertEqual(sds.shape, (4, 5, 3))
            self.assertEqual(BASIC_MODEL_DAYS, ("1D", "4D", "8D", "15D", "29D"))
            self.assertEqual(DAY_TO_INDEX, {1: 0, 4: 1, 8: 2, 15: 3, 29: 4})


if __name__ == "__main__":
    unittest.main()
