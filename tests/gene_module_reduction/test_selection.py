"""Tests for broad and specific Gene selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.gene_module_reduction.eligibility import FIT_MODEL_TIMES
from src.gene_module_reduction.selection import (
    SelectionConfig,
    _specific_support,
    select_genes,
    write_selection,
)


def _selection_fixture(directory: Path, late_value: float = 0.0) -> tuple[Path, pd.DataFrame]:
    """Create a selection matrix with broad, specific, both, and excluded Genes.

    Args:
        directory: Temporary output directory.
        late_value: Value assigned only to non-fit 15D samples.

    Returns:
        Log2FC path and eligible sample metadata.
    """

    directory.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, object]] = []
    fit_sample_ids: list[str] = []
    for compound_index in range(21):
        compound_no = str(compound_index + 1).zfill(5)
        for model_time in FIT_MODEL_TIMES:
            treatment_id = f"T{compound_no}"
            condition_id = f"{treatment_id}_{model_time}"
            for replicate in range(3):
                sample_id = f"{condition_id}_R{replicate}"
                fit_sample_ids.append(sample_id)
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "condition_id": condition_id,
                        "treatment_id": treatment_id,
                        "compound_no": compound_no,
                        "compound_name": f"compound_{compound_no}",
                        "dose": "10",
                        "dose_unit": "mg/kg",
                        "model_time": model_time,
                        "partition": "train",
                        "used_for_reduction_fit": True,
                    }
                )
    late_ids: list[str] = []
    for replicate in range(3):
        sample_id = f"T00001_15D_R{replicate}"
        late_ids.append(sample_id)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "condition_id": "T00001_15D",
                "treatment_id": "T00001",
                "compound_no": "00001",
                "compound_name": "compound_00001",
                "dose": "10",
                "dose_unit": "mg/kg",
                "model_time": "15D",
                "partition": "train",
                "used_for_reduction_fit": False,
            }
        )
    metadata = pd.DataFrame(sample_rows)
    n_conditions = 21 * len(FIT_MODEL_TIMES)
    offsets = np.asarray([-0.01, 0.0, 0.01])
    broad = np.repeat(np.full(n_conditions, 0.6), 3) + np.tile(offsets, n_conditions)
    specific_medians = np.zeros(n_conditions)
    specific_medians[:2] = 1.2
    specific = np.repeat(specific_medians, 3) + np.tile(offsets, n_conditions)
    both_medians = np.full(n_conditions, 0.6)
    both_medians[:2] = 1.2
    both = np.repeat(both_medians, 3) + np.tile(offsets, n_conditions)
    excluded_medians = np.zeros(n_conditions)
    excluded_medians[0] = 1.2
    excluded = np.repeat(excluded_medians, 3) + np.tile(offsets, n_conditions)
    non_directional = np.tile(np.asarray([-0.1, 0.6, 0.7]), n_conditions)
    low_snr_directional = np.tile(np.asarray([0.01, 0.6, 1.19]), n_conditions)
    opposite_medians = np.zeros(n_conditions)
    opposite_medians[0] = 1.2
    opposite_medians[1] = -1.2
    opposite = np.repeat(opposite_medians, 3) + np.tile(offsets, n_conditions)
    rows = []
    for gene_id, fit_values in (
        ("G_BROAD", broad),
        ("G_SPECIFIC", specific),
        ("G_BOTH", both),
        ("G_EXCLUDED", excluded),
        ("G_NON_DIRECTIONAL", non_directional),
        ("G_LOW_SNR_DIRECTIONAL", low_snr_directional),
        ("G_OPPOSITE", opposite),
    ):
        row: dict[str, object] = {"gene_id": gene_id}
        row.update(dict(zip(fit_sample_ids, fit_values.tolist())))
        row.update({sample_id: late_value for sample_id in late_ids})
        rows.append(row)
    path = directory / "log2fc.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path, metadata


class GeneSelectionTests(unittest.TestCase):
    """Verify dual-route selection and the 15D/29D leakage boundary."""

    def test_broad_specific_and_both_classification(self) -> None:
        """Selection routes must remain independent before their union."""

        self.assertEqual(SelectionConfig().broad_rate, 0.10)
        with tempfile.TemporaryDirectory() as temporary:
            path, metadata = _selection_fixture(Path(temporary))
            result = select_genes(path, metadata, SelectionConfig(chunk_size=2))
        classes = result.gene_metrics.set_index("gene_id")["selection_class"].to_dict()
        self.assertEqual(classes["G_BROAD"], "broad")
        self.assertEqual(classes["G_SPECIFIC"], "specific")
        self.assertEqual(classes["G_BOTH"], "both")
        self.assertEqual(classes["G_EXCLUDED"], "excluded")
        self.assertEqual(classes["G_NON_DIRECTIONAL"], "excluded")
        self.assertEqual(classes["G_LOW_SNR_DIRECTIONAL"], "broad")
        self.assertEqual(classes["G_OPPOSITE"], "excluded")
        self.assertEqual(
            result.selected_gene_ids,
            ["G_BROAD", "G_SPECIFIC", "G_BOTH", "G_LOW_SNR_DIRECTIONAL"],
        )

    def test_directionality_and_same_sign_specific_boundaries(self) -> None:
        """All replicates must agree, and opposite strong medians must not combine."""

        with tempfile.TemporaryDirectory() as temporary:
            path, metadata = _selection_fixture(Path(temporary))
            result = select_genes(path, metadata, SelectionConfig(chunk_size=3))
        metrics = result.gene_metrics.set_index("gene_id")
        n_conditions = len(result.condition_metadata)
        self.assertEqual(metrics.loc["G_NON_DIRECTIONAL", "n_directional_conditions"], 0)
        self.assertEqual(
            metrics.loc["G_LOW_SNR_DIRECTIONAL", "n_directional_conditions"],
            n_conditions,
        )
        self.assertLess(metrics.loc["G_LOW_SNR_DIRECTIONAL", "max_robust_snr"], 2.0)
        self.assertEqual(
            metrics.loc["G_OPPOSITE", "max_strong_conditions_within_compound"],
            1,
        )
        self.assertAlmostEqual(
            metrics.loc[
                "G_SPECIFIC", "max_same_direction_strong_rate_within_compound"
            ],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            metrics.loc[
                "G_OPPOSITE", "max_same_direction_strong_rate_within_compound"
            ],
            1.0 / 6.0,
        )
        self.assertEqual(metrics.loc["G_SPECIFIC", "specific_supporting_directions"], "00001:+")

    def test_specific_rate_scales_with_compound_condition_count(self) -> None:
        """One-third must require 2/6, 4/12, or 6/18 same-sign conditions."""

        compounds = ["A"] * 6 + ["B"] * 12 + ["C"] * 18
        directions = np.zeros((5, len(compounds)), dtype=np.int8)
        directions[0, :2] = 1
        directions[1, 6:9] = 1
        directions[2, 6:10] = -1
        directions[3, 18:23] = 1
        directions[4, 18:24] = 1
        specific, _, signed_support, max_counts, max_rates = _specific_support(
            directions,
            compounds,
            required_rate=1.0 / 3.0,
        )
        np.testing.assert_array_equal(specific, [True, False, True, False, True])
        np.testing.assert_array_equal(max_counts, [2, 3, 4, 5, 6])
        np.testing.assert_allclose(
            max_rates,
            [1.0 / 3.0, 1.0 / 4.0, 1.0 / 3.0, 5.0 / 18.0, 1.0 / 3.0],
        )
        self.assertEqual(signed_support, ["A:+", "", "B:-", "", "C:+"])

    def test_late_values_do_not_change_selection(self) -> None:
        """15D values must not enter any selection statistic."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path, first_metadata = _selection_fixture(root / "first", late_value=0.0)
            second_path, second_metadata = _selection_fixture(root / "second", late_value=999.0)
            first = select_genes(first_path, first_metadata, SelectionConfig())
            second = select_genes(second_path, second_metadata, SelectionConfig())
        pd.testing.assert_frame_equal(first.gene_metrics, second.gene_metrics)
        np.testing.assert_allclose(
            first.selected_condition_medians, second.selected_condition_medians
        )
        np.testing.assert_allclose(
            first.selected_fit_replicates, second.selected_fit_replicates
        )

    def test_single_compound_uses_inclusive_broad_only_rule(self) -> None:
        """Single-compound mode must use >=0.5, >=1/3, and never Specific."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, metadata = _selection_fixture(root)
            one_compound = metadata[metadata["compound_no"].eq("00001")].copy()
            matrix = pd.read_csv(path, sep="\t")
            exact_effect: dict[str, object] = {"gene_id": "G_EXACT_EFFECT"}
            for sample_id in matrix.columns[1:]:
                exact_effect[sample_id] = 0.5
            fit_ids = one_compound.loc[
                one_compound["used_for_reduction_fit"], "sample_id"
            ].astype(str)
            for condition_index in range(len(FIT_MODEL_TIMES)):
                condition_ids = fit_ids.iloc[condition_index * 3 : condition_index * 3 + 3]
                for sample_id, value in zip(condition_ids, (0.49, 0.5, 0.51)):
                    exact_effect[sample_id] = value
            matrix = pd.concat([matrix, pd.DataFrame([exact_effect])], ignore_index=True)
            matrix.to_csv(path, sep="\t", index=False)
            result = select_genes(
                path,
                one_compound,
                SelectionConfig(broad_rate=1.0 / 3.0, chunk_size=2),
                selection_mode="single_compound",
            )
            write_selection(result, root / "single_selection")
            summary = (root / "single_selection" / "summary.txt").read_text(
                encoding="utf-8"
            )

        metrics = result.gene_metrics.set_index("gene_id")
        self.assertEqual(
            result.selected_gene_ids,
            [
                "G_BROAD",
                "G_SPECIFIC",
                "G_BOTH",
                "G_LOW_SNR_DIRECTIONAL",
                "G_OPPOSITE",
                "G_EXACT_EFFECT",
            ],
        )
        self.assertTrue(bool(metrics.loc["G_SPECIFIC", "selected"]))
        self.assertAlmostEqual(metrics.loc["G_SPECIFIC", "broad_response_rate"], 1 / 3)
        self.assertTrue(bool(metrics.loc["G_EXACT_EFFECT", "selected"]))
        self.assertEqual(set(metrics["selection_class"]), {"broad", "excluded"})
        self.assertEqual(result.metadata["selection_mode"], "single_compound")
        self.assertEqual(result.metadata["selection_rules"]["specific"], "disabled")
        self.assertIn("broad_rate_requirement: >= 0.3333333333333333", summary)
        self.assertIn("specific_rate_requirement: disabled", summary)

    def test_replicate_fit_matrix_preserves_individual_log2fc(self) -> None:
        """WGCNA input must retain each selected Gene's replicate-level values."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, metadata = _selection_fixture(root)
            result = select_genes(path, metadata, SelectionConfig())
            source = pd.read_csv(path, sep="\t").set_index("gene_id")
            sample_ids = result.fit_sample_metadata["sample_id"].astype(str).tolist()
            expected = source.loc[result.selected_gene_ids, sample_ids].to_numpy().T
            np.testing.assert_allclose(result.selected_fit_replicates, expected)
            self.assertEqual(
                result.selected_fit_replicates.shape[0],
                result.selected_condition_medians.shape[0] * 3,
            )
            write_selection(result, root / "selection")
            with np.load(
                root / "selection" / "fit_replicate_values.npz", allow_pickle=False
            ) as archive:
                np.testing.assert_allclose(archive["values"], expected)
                self.assertEqual(archive["sample_ids"].astype(str).tolist(), sample_ids)
            summary = (root / "selection" / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("selection_mode: organ", summary)
            self.assertIn("n_selected_genes: 4", summary)
            self.assertIn("broad_rate_requirement: >= 0.1", summary)
            self.assertIn("specific_rate: 0.3333333333333333", summary)
            self.assertIn(
                "specific_rate_requirement: >= 0.3333333333333333",
                summary,
            )


if __name__ == "__main__":
    unittest.main()
