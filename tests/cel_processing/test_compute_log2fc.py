"""Tests for control-pool eligibility and log2FC calculations."""

from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.cel_processing.compute_log2fc import (
    add_log2fc_eligibility,
    compute_log2fc_chunk,
)


def sample_row(
    sample_id: str,
    route: str,
    is_control: bool,
    baseline_group_id: str,
) -> dict[str, object]:
    """Create one synthetic RMA manifest row.

    Args:
        sample_id: Sample identifier.
        route: Administration route.
        is_control: Whether the sample is a control.
        baseline_group_id: Expected baseline group identifier.

    Returns:
        Synthetic manifest row.
    """

    return {
        "sample_id": sample_id,
        "organ": "Liver",
        "time_label": "3H",
        "administration_route": route,
        "is_control": is_control,
        "is_healthy_control": is_control,
        "include_in_rma": True,
        "baseline_group_id": baseline_group_id,
    }


class ComputeLog2fcTests(unittest.TestCase):
    """Validate exact-route baseline pools and subtraction semantics."""

    def setUp(self) -> None:
        """Build a three-control eligible pool and one undersized pool."""

        gavage_group = "ORGAN-LIVER__TIME-3H__ROUTE-GAVAGE"
        inject_group = "ORGAN-LIVER__TIME-3H__ROUTE-INTRAVENOUS-INJECTION"
        self.manifest = pd.DataFrame(
            [
                sample_row("C1", "Gavage", True, gavage_group),
                sample_row("C2", "Gavage", True, gavage_group),
                sample_row("C3", "Gavage", True, gavage_group),
                sample_row("T1", "Gavage", False, gavage_group),
                sample_row("C4", "intravenous injection", True, inject_group),
                sample_row("T2", "intravenous injection", False, inject_group),
            ]
        )

    def test_minimum_control_pool_and_route_matching(self) -> None:
        """Only samples in the exact-route pool with three controls should pass."""

        audit, groups = add_log2fc_eligibility(self.manifest, min_controls=3)
        by_id = audit.set_index("sample_id")
        self.assertTrue(by_id.loc["C1", "include_in_log2fc"])
        self.assertTrue(by_id.loc["T1", "include_in_log2fc"])
        self.assertFalse(by_id.loc["C4", "include_in_log2fc"])
        self.assertFalse(by_id.loc["T2", "include_in_log2fc"])
        counts = groups.set_index("administration_route")["n_controls"].to_dict()
        self.assertEqual(counts, {"Gavage": 3, "intravenous injection": 1})

    def test_log2fc_and_control_centering(self) -> None:
        """Treated values should subtract the mean of three matched controls."""

        audit, _ = add_log2fc_eligibility(self.manifest, min_controls=3)
        expression = pd.DataFrame(
            {
                "C1": [4.0, 8.0],
                "C2": [5.0, 10.0],
                "C3": [6.0, 12.0],
                "T1": [7.0, 16.0],
                "C4": [2.0, 3.0],
                "T2": [4.0, 7.0],
            },
            index=["GENE1", "GENE2"],
        )
        log2fc, baseline = compute_log2fc_chunk(expression, audit)
        expected = pd.DataFrame(
            {
                "C1": [-1.0, -2.0],
                "C2": [0.0, 0.0],
                "C3": [1.0, 2.0],
                "T1": [2.0, 6.0],
            },
            index=expression.index,
        )
        assert_frame_equal(log2fc, expected)
        self.assertTrue((log2fc[["C1", "C2", "C3"]].mean(axis=1).abs() < 1e-12).all())
        self.assertEqual(baseline.iloc[:, 0].tolist(), [5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
