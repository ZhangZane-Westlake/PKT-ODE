"""Tests for eight-timepoint treatment eligibility."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Optional

import pandas as pd

from src.gene_module_reduction.eligibility import (
    FIT_MODEL_TIMES,
    build_treatment_eligibility,
    filter_eligibility_for_compound,
)


SOURCE_TIMES = (
    ("3H", "Single"),
    ("6H", "Single"),
    ("9H", "Single"),
    ("24H", "Single"),
    ("4D", "Repeat"),
    ("8D", "Repeat"),
    ("15D", "Repeat"),
    ("29D", "Repeat"),
)


def _manifest_treatment(
    compound_no: str,
    missing_times: Optional[set[str]] = None,
) -> list[dict[str, object]]:
    """Build one synthetic exact-dose treatment.

    Args:
        compound_no: Compound identifier.
        missing_times: Source time labels to omit.

    Returns:
        Manifest rows.
    """

    missing = missing_times or set()
    rows: list[dict[str, object]] = []
    for time_label, regimen in SOURCE_TIMES:
        if time_label in missing:
            continue
        for replicate in range(3):
            rows.append(
                {
                    "sample_id": f"C{compound_no}_{time_label}_{replicate}",
                    "compound_no": compound_no,
                    "compound_name": f"compound_{compound_no}",
                    "dose": 10,
                    "dose_unit": "mg/kg",
                    "dose_level": "HIGH",
                    "time_label": time_label,
                    "organ": "Kidney",
                    "regimen": regimen,
                    "administration_route": "Gavage",
                    "is_control": False,
                    "include_in_log2fc": True,
                }
            )
    return rows


class TreatmentEligibilityTests(unittest.TestCase):
    """Verify exact-three and eight-timepoint eligibility boundaries."""

    def test_missing_15d_29d_excludes_six_timepoint_complete_treatment(self) -> None:
        """Six fit timepoints alone must not establish cohort eligibility."""

        manifest = pd.DataFrame(
            _manifest_treatment("00001")
            + _manifest_treatment("00002", missing_times={"15D", "29D"})
        )
        result = build_treatment_eligibility(manifest, "kidney")
        self.assertEqual(int(result.treatment_summary["treatment_eligible"].sum()), 1)
        self.assertEqual(len(result.eligible_samples), 8 * 3)
        excluded = result.treatment_eligibility[
            result.treatment_eligibility["compound_no"].eq("00002")
        ]
        self.assertFalse(excluded["treatment_eligible"].any())
        self.assertEqual(
            set(excluded.loc[~excluded["timepoint_valid"], "model_time"]),
            {"15D", "29D"},
        )

    def test_filter_compound_runs_after_full_timepoint_eligibility(self) -> None:
        """Name/number matching must retain only an eight-timepoint eligible compound."""

        manifest = pd.DataFrame(
            _manifest_treatment("00001")
            + _manifest_treatment("00002", missing_times={"29D"})
        )
        result = build_treatment_eligibility(manifest, "kidney")
        filtered, compound_no, compound_name = filter_eligibility_for_compound(
            result, "COMPOUND_00001"
        )
        self.assertEqual(compound_no, "00001")
        self.assertEqual(compound_name, "compound_00001")
        self.assertEqual(set(filtered.eligible_samples["compound_no"]), {"00001"})
        self.assertEqual(len(filtered.eligible_samples), 8 * 3)

        filtered_by_number, _, _ = filter_eligibility_for_compound(result, "C00001")
        pd.testing.assert_frame_equal(
            filtered.eligible_samples, filtered_by_number.eligible_samples
        )
        with self.assertRaisesRegex(ValueError, "no treatment with exactly three replicates"):
            filter_eligibility_for_compound(result, "00002")

    def test_current_manifest_has_expected_complete_cohorts(self) -> None:
        """The saved project manifest must reproduce the audited cohort counts."""

        path = Path("data/expression/log2fc/log2fc_sample_manifest.tsv")
        if not path.is_file():
            self.skipTest("Project log2FC manifest is unavailable")
        manifest = pd.read_csv(path, sep="\t", low_memory=False)
        expected = {"kidney": (34, 204), "liver": (139, 834)}
        for organ, (n_treatments, n_conditions) in expected.items():
            result = build_treatment_eligibility(manifest, organ)
            self.assertEqual(
                int(result.treatment_summary["treatment_eligible"].sum()), n_treatments
            )
            fit = result.eligible_samples[result.eligible_samples["used_for_reduction_fit"]]
            self.assertEqual(fit["condition_id"].nunique(), n_conditions)
            self.assertEqual(n_conditions, n_treatments * len(FIT_MODEL_TIMES))


if __name__ == "__main__":
    unittest.main()
