"""Smoke and leakage tests for the basic module training loop."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.module_dynamics.basic_rollout.data import ModuleTrajectoryMatrix
from src.module_dynamics.basic_rollout.scaling import FeatureScaler
from src.module_dynamics.basic_rollout.training import TrainingConfig, train_module_model
from src.module_dynamics.common import ReductionScope


def _synthetic_matrix() -> ModuleTrajectoryMatrix:
    """Build a small five-day module trajectory matrix.

    Returns:
        Synthetic module trajectory matrix.
    """

    rng = np.random.default_rng(0)
    treatment_ids = [f"T{index}" for index in range(4)]
    scope = ReductionScope(
        organ="liver",
        compound_slug="fenofibrate",
        split_id="tight_stage",
        config_id="r2-test",
        is_compound_scope=True,
        run_name="gene_selection_TEST",
    )
    return ModuleTrajectoryMatrix(
        scope=scope,
        treatment_ids=treatment_ids,
        module_ids=["M001", "M002", "M003"],
        values=rng.normal(size=(4, 5, 3)),
        replicate_values=rng.normal(size=(4, 5, 3, 3)),
        treatment_metadata=pd.DataFrame({"treatment_id": treatment_ids}),
    )


class ScalerLeakageTest(unittest.TestCase):
    """The FeatureScaler must only read train days."""

    def test_fit_uses_first_three_days_only(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(size=(3, 5, 4))
        scaler = FeatureScaler.fit(values)
        expected_mean = values[:, :3].reshape(-1, 4).mean(axis=0)
        self.assertTrue(np.allclose(scaler.mean, expected_mean))
        # Validation/test days must not move the mean.
        values[:, 3] = 1000.0
        values[:, 4] = -1000.0
        scaler2 = FeatureScaler.fit(values)
        self.assertTrue(np.allclose(scaler2.mean, expected_mean))


class TrainingSmokeTest(unittest.TestCase):
    """Two-epoch CPU training for both loss modes."""

    def _train(self, loss_mode: str) -> None:
        matrix = _synthetic_matrix()
        config = TrainingConfig(
            organ=matrix.scope.organ,
            scope_tag=matrix.scope.scope_tag,
            model="mlp",
            dynamics="lrd",
            conditioning="none",
            loss_mode=loss_mode,
            device="cpu",
            max_epochs=2,
            patience=5,
            log_every=1,
            deterministic=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            logger = logging.getLogger(f"smoke.{loss_mode}.{tmp}")
            logger.handlers.clear()
            artifacts = train_module_model(matrix, config, run_dir, logger)
        self.assertEqual(len(artifacts.history), 2)
        self.assertTrue(np.isfinite(artifacts.metrics["mse"]).all())
        self.assertGreater(artifacts.best_epoch, 0)

    def test_mean_mode(self) -> None:
        self._train("mean")

    def test_replicate_mode(self) -> None:
        self._train("replicate")


if __name__ == "__main__":
    unittest.main()
