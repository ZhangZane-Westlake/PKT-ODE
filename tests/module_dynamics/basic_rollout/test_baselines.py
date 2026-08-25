"""Tests for the configurable fit-day module baselines."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.module_dynamics.basic_rollout.baselines import run_baselines
from src.module_dynamics.basic_rollout.data import ModuleTrajectoryMatrix
from src.module_dynamics.common import ReductionScope


def _synthetic_matrix(seed: int = 0) -> ModuleTrajectoryMatrix:
    """Build a small five-day module trajectory matrix.

    Args:
        seed: Random seed for the synthetic values.

    Returns:
        Synthetic module trajectory matrix with a strictly-linear module so the
        trend baselines are exactly recoverable.
    """

    rng = np.random.default_rng(seed)
    treatment_ids = [f"T{index}" for index in range(3)]
    scope = ReductionScope(
        organ="liver",
        compound_slug="fenofibrate",
        split_id="tight_stage",
        config_id="r2-test",
        is_compound_scope=True,
        run_name="gene_selection_TEST",
    )
    values = rng.normal(size=(3, 5, 3))
    # Module 0 is perfectly linear in day: y = 2*day + 1 (days 1,4,8,15,29).
    days = np.asarray([1, 4, 8, 15, 29], dtype=float)
    values[:, :, 0] = 2.0 * days + 1.0
    return ModuleTrajectoryMatrix(
        scope=scope,
        treatment_ids=treatment_ids,
        module_ids=["M001", "M002", "M003"],
        values=values,
        replicate_values=rng.normal(size=(3, 5, 3, 3)),
        treatment_metadata=pd.DataFrame({"treatment_id": treatment_ids}),
    )


class FitDaysBaselineTest(unittest.TestCase):
    """Configurable fit-day windows produce the expected names and targets."""

    def test_default_fit_predicts_validation_and_test(self) -> None:
        """Default fit_days=(1,4,8) predicts the 15D and 29D days."""

        matrix = _synthetic_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "default"
            metrics = run_baselines(matrix, run_dir)
            labels = list(np.load(run_dir / "predictions.npz")["model_labels"])
            self.assertIn("Early mean (1D4D8D)", labels)
            self.assertIn("Persistence (8D)", labels)
            npz_days = list(np.load(run_dir / "predictions.npz")["days"])
            self.assertEqual(npz_days, [15, 29])
            splits = set(zip(metrics["split"], metrics["day"]))
            self.assertEqual(splits, {("validation", 15), ("test", 29)})

    def test_folding_15d_into_fit_leaves_only_29d(self) -> None:
        """fit_days=(1,4,8,15) folds 15D into the fit so only 29D is held out."""

        matrix = _synthetic_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "fifteen"
            metrics = run_baselines(matrix, run_dir, fit_days=(1, 4, 8, 15))
            archive = np.load(run_dir / "predictions.npz")
            labels = list(archive["model_labels"])
            self.assertEqual(list(archive["days"]), [29])
            self.assertIn("Early mean (1D4D8D15D)", labels)
            self.assertIn("Linear trend (1D4D8D15D)", labels)
            self.assertIn("Log-time trend (1D4D8D15D)", labels)
            self.assertIn("Persistence (15D)", labels)
            splits = set(zip(metrics["split"], metrics["day"]))
            self.assertEqual(splits, {("test", 29)})
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["fit_days"], ["1D", "4D", "8D", "15D"])
            self.assertEqual(config["prediction_days"], ["29D"])
            self.assertIsNone(config["validation_day"])
            self.assertEqual(config["test_day"], "29D")
            self.assertTrue(config["uses_15d_to_predict_29d"])

    def test_linear_trend_with_15d_fit_extrapolates_to_29d(self) -> None:
        """The linear trend fit on 1D/4D/8D/15D recovers the 29D value exactly."""

        matrix = _synthetic_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "linear"
            run_baselines(matrix, run_dir, fit_days=(1, 4, 8, 15))
            archive = np.load(run_dir / "predictions.npz")
            labels = list(archive["model_labels"])
            idx = labels.index("Linear trend (1D4D8D15D)")
            # prediction_raw shape: (n_models, n_days=1, n_treat, n_module)
            predicted_29d = archive["prediction_raw"][idx, 0, :, 0]
            # Module 0 is y = 2*day + 1 -> at day 29 the value is 59.
            np.testing.assert_allclose(predicted_29d, 2.0 * 29 + 1.0)

    def test_persistence_anchors_on_latest_fit_day(self) -> None:
        """Persistence reports only the latest fit day as its anchor."""

        matrix = _synthetic_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "persist"
            run_baselines(matrix, run_dir, fit_days=(1, 4, 8, 15))
            archive = np.load(run_dir / "predictions.npz")
            labels = list(archive["model_labels"])
            idx = labels.index("Persistence (15D)")
            # Persistence at 29D must equal the observed 15D value.
            np.testing.assert_allclose(
                archive["prediction_raw"][idx, 0, :, 0],
                matrix.values[:, 3, 0],  # 15D lives at DAY_TO_INDEX[15] == 3
            )

    def test_invalid_fit_days_raise(self) -> None:
        """Unknown or all-consuming fit-day sets are rejected."""

        matrix = _synthetic_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsupported fit days"):
                run_baselines(matrix, Path(tmp) / "bad", fit_days=(1, 99))
            with self.assertRaisesRegex(ValueError, "at least one held out"):
                run_baselines(matrix, Path(tmp) / "all", fit_days=(1, 4, 8, 15, 29))


if __name__ == "__main__":
    unittest.main()
