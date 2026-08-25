"""Data-contract tests for the public trajectory archive."""

from pathlib import Path

import numpy as np

from src.pkt_ode.data import load_trajectories


ROOT = Path(__file__).resolve().parents[2]


def test_public_trajectory_contract() -> None:
    """The snapshot contains three doses, eight times, three replicates, and seven modules."""

    data = load_trajectories(ROOT / "data/processed/pkt_ode_input.npz")
    assert data.replicate_values.shape == (3, 8, 3, 7)
    assert data.mean_values.shape == (3, 8, 7)
    np.testing.assert_array_equal(data.doses, np.asarray([10.0, 100.0, 1000.0]))
    assert data.module_ids == tuple(f"M00{index}" for index in range(1, 8))
    assert int(data.used_for_fit.sum()) == 54
    assert data.regimens == (
        "Single",
        "Single",
        "Single",
        "Single",
        "Repeat",
        "Repeat",
        "Repeat",
        "Repeat",
    )
    np.testing.assert_allclose(
        data.replicate_values,
        data.raw_replicate_values / data.mad_fit,
        atol=2e-6,
    )
