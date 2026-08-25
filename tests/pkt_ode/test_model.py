"""Numerical tests for PKT-ODE simulation."""

from pathlib import Path

import numpy as np

from src.pkt_ode.data import load_module_parameters, load_trajectories
from src.pkt_ode.evaluation import evaluate_splits, evaluate_statistical_baselines
from src.pkt_ode.model import (
    ModuleParameters,
    bateman_concentration,
    hill_effect,
    simulate_trajectories,
)


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORIES = ROOT / "data/processed/pkt_ode_input.npz"
PROJECTION_TRAJECTORIES = (
    ROOT
    / "data/processed/fenofibrate_reduction/projection/treatment_module_trajectories.npz"
)
PARAMETERS = ROOT / "data/processed/published_parameters.tsv"


def test_driver_is_nonnegative_and_hill_is_bounded() -> None:
    """Concentration must be non-negative and the Hill signal bounded."""

    times = np.linspace(-1.0, 3.0, 200)
    concentration = bateman_concentration(times, 100.0)
    effect = hill_effect(concentration)
    assert np.min(concentration) >= 0.0
    assert np.min(effect) >= 0.0
    assert np.max(effect) <= 1.0


def test_zero_effect_preserves_baseline() -> None:
    """A module with beta_1=0 remains at beta_0."""

    parameters = {"M000": ModuleParameters(k_per_day=2.0, beta_0=1.5, beta_1=0.0)}
    prediction = simulate_trajectories([10.0], np.asarray([0.125, 1.0, 29.0]), ["M000"], parameters)
    np.testing.assert_allclose(prediction, 1.5, atol=1e-12)


def test_published_parameters_reproduce_endpoint_correlation() -> None:
    """Rounded published parameters recover the reported endpoint correlation."""

    data = load_trajectories(TRAJECTORIES)
    parameters = load_module_parameters(PARAMETERS)
    prediction = simulate_trajectories(
        data.doses, data.times, data.module_ids, parameters, regimens=data.regimens
    )
    metrics = evaluate_splits(data.times, data.replicate_values, prediction).set_index("split")
    assert np.isclose(metrics.loc["test", "mse"], 0.14818, atol=1e-4)
    assert 0.95 < metrics.loc["test", "pearson_r"] < 0.97
    assert int(metrics.loc["test", "n_values"]) == 63


def test_statistical_baselines_match_snapshot() -> None:
    """The four deterministic baselines reproduce the stored benchmark."""

    with np.load(PROJECTION_TRAJECTORIES, allow_pickle=False) as archive:
        mean_values = np.asarray(archive["mean_values"], dtype=float)
    times = np.asarray([0.125, 0.25, 0.375, 1.0, 4.0, 8.0, 15.0, 29.0])
    metrics = evaluate_statistical_baselines(times, mean_values)
    day29 = metrics[metrics["day"] == 29].set_index("model")
    assert np.isclose(day29.loc["linear_trend", "mse"], 4.57387, atol=1e-4)
    assert np.isclose(day29.loc["persistence", "pearson_r"], 0.920083, atol=1e-5)
