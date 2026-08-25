"""Tests for fixed module projection and audits."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.gene_module_reduction.audit import (
    calculate_module_hallmark_purity,
    calculate_module_stability,
    load_gene_sets,
)
from src.gene_module_reduction.projection import (
    ModuleBundle,
    ProjectionFitResult,
    fit_fixed_projection,
    project_values,
)


class ProjectionAuditTests(unittest.TestCase):
    """Verify fixed PC1 reconstruction, grey handling, and Hallmark purity."""

    def _fit_fixture(self, root: Path) -> tuple[
        np.ndarray,
        Path,
        pd.DataFrame,
        ProjectionFitResult,
    ]:
        """Create and fit a deterministic two-module fixture."""

        sample_ids = [f"S{index}" for index in range(12)]
        gene_ids = ["G1", "G2", "G3", "G4", "G_GREY"]
        first = np.linspace(-2.0, 2.0, 12)
        second = np.sin(np.linspace(0.0, 2.0 * np.pi, 12))
        values = np.column_stack(
            [first, first * 0.9 + 0.1, second, second * 1.1 - 0.1, np.arange(12)]
        )
        matrix_path = root / "fit_replicate_values.npz"
        np.savez_compressed(
            matrix_path,
            values=values,
            sample_ids=np.asarray(sample_ids, dtype=str),
            gene_ids=np.asarray(gene_ids, dtype=str),
        )
        modules = pd.DataFrame(
            {
                "gene_id": gene_ids,
                "module_id": ["M001", "M001", "M002", "M002", "grey"],
            }
        )
        metrics = pd.DataFrame(
            {
                "gene_id": gene_ids,
                "selection_class": ["broad", "specific", "both", "broad", "specific"],
            }
        )
        return values, matrix_path, modules, fit_fixed_projection(matrix_path, modules, metrics)

    def test_fixed_projection_reproduces_training_scores(self) -> None:
        """Applying the saved parameters must reproduce fitted PC1 scores exactly."""

        with tempfile.TemporaryDirectory() as temporary:
            values, _, _, result = self._fit_fixture(Path(temporary))
        raw, standardized = project_values(values, result.bundle)
        np.testing.assert_allclose(raw, result.fit_sample_scores_raw, atol=1e-10)
        np.testing.assert_allclose(
            standardized, result.fit_sample_scores_standardized, atol=1e-10
        )
        self.assertEqual(result.bundle.module_ids, ["M001", "M002"])
        self.assertEqual(len(result.gene_modules), 5)
        self.assertIn("grey", set(result.gene_modules["module_id"]))
        self.assertTrue(
            (result.module_summary["pc1_mean_expression_correlation"] >= 0).all()
        )

    def test_projection_rejects_non_finite_matrix_product(self) -> None:
        """Finite inputs that overflow during projection must fail explicitly."""

        bundle = ModuleBundle(
            gene_ids=["G1"],
            module_ids=["M001"],
            gene_center=np.asarray([0.0]),
            gene_scale=np.asarray([1.0]),
            loadings=np.asarray([[1e308]]),
            module_center=np.asarray([0.0]),
            module_scale=np.asarray([1.0]),
        )
        with self.assertRaisesRegex(ValueError, "produced non-finite values"):
            project_values(np.asarray([[1e308]]), bundle)

    def test_stability_and_versioned_hallmark_purity(self) -> None:
        """Purity must divide each Hallmark overlap by module Gene count."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, matrix_path, modules, result = self._fit_fixture(root)
            fit_sample_metadata = pd.DataFrame(
                {
                    "sample_id": [f"S{index}" for index in range(12)],
                    "compound_no": [str(index // 2).zfill(5) for index in range(12)],
                    "model_time": ["3H", "4D"] * 6,
                }
            )
            stability = calculate_module_stability(
                matrix_path,
                fit_sample_metadata,
                result.bundle,
                n_subsamples=3,
                compound_fraction=0.8,
            )
            hallmark = root / "hallmark.tsv"
            gene_sets_table = pd.DataFrame(
                {
                    "term_id": ["T1", "T1", "T1", "T2"],
                    "term_name": ["term1", "term1", "term1", "term2"],
                    "gene_id": ["G1", "G2", "G3", "G3"],
                }
            )
            gene_sets_table.to_csv(hallmark, sep="\t", index=False)
            digest = hashlib.sha256(hallmark.read_bytes()).hexdigest()
            manifest = root / "manifest.tsv"
            pd.DataFrame(
                {
                    "source": ["hallmark"],
                    "version": ["test-v1"],
                    "path": [hallmark.name],
                    "sha256": [digest],
                }
            ).to_csv(manifest, sep="\t", index=False)
            collection = load_gene_sets(manifest, allow_partial=True)
            purity = calculate_module_hallmark_purity(
                modules,
                collection,
            )
        self.assertEqual(set(stability["module_id"]), {"M001", "M002"})
        self.assertEqual(purity.index.tolist(), ["M001", "M002"])
        self.assertEqual(purity.columns.tolist(), ["term1", "term2"])
        expected_purity = pd.DataFrame(
            {
                "term1": [1.0, 0.5],
                "term2": [0.0, 0.5],
            },
            index=pd.Index(["M001", "M002"], name="module_id"),
        )
        pd.testing.assert_frame_equal(purity, expected_purity)

    def test_stability_accepts_a_single_compound_scope(self) -> None:
        """Compound-specific reduction must audit its one-compound fit matrix."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, matrix_path, _, result = self._fit_fixture(root)
            fit_sample_metadata = pd.DataFrame(
                {
                    "sample_id": [f"S{index}" for index in range(12)],
                    "compound_no": ["00079"] * 12,
                    "model_time": ["3H", "4D"] * 6,
                }
            )
            stability = calculate_module_stability(
                matrix_path,
                fit_sample_metadata,
                result.bundle,
                n_subsamples=3,
                compound_fraction=0.8,
            )
        self.assertEqual(set(stability["module_id"]), {"M001", "M002"})
        self.assertTrue(
            np.isfinite(stability["median_subsample_loading_cosine"]).all()
        )


if __name__ == "__main__":
    unittest.main()
