"""Tests for Gene-module reduction command-line scope and output paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.gene_module_reduction.cli import (
    _reduction_config_id,
    _reduction_root,
    _run_root,
    build_parser,
)


class GeneModuleCliTests(unittest.TestCase):
    """Verify scope paths and parameter-isolated reduction paths."""

    def test_compound_scope_has_an_isolated_output_tree(self) -> None:
        """Every stage must resolve the same normalized compound path."""

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "select",
                    "--organ",
                    "liver",
                    "--compound",
                    "Feno fibrate",
                    "--run-name",
                    "wgcna_v1",
                    "--output-root",
                    str(output_root),
                ]
            )
            self.assertEqual(args.broad_rate, None)
            self.assertEqual(
                _run_root(args),
                output_root.resolve()
                / "wgcna_v1"
                / "liver"
                / "compounds"
                / "feno-fibrate"
                / "loose_stage",
            )

    def test_organ_scope_keeps_the_existing_output_tree(self) -> None:
        """Omitting compound must preserve the established path contract."""

        parser = build_parser()
        args = parser.parse_args(
            ["select", "--organ", "kidney", "--run-name", "wgcna_v1"]
        )
        self.assertEqual(
            _run_root(args),
            args.output_root.resolve() / "wgcna_v1" / "kidney" / "loose_stage",
        )

    def test_reduction_configuration_is_parameter_readable(self) -> None:
        """Fit parameters must produce one stable, readable output component."""

        parser = build_parser()
        args = parser.parse_args(
            [
                "fit-wgcna",
                "--organ",
                "kidney",
                "--run-name",
                "wgcna_v1",
                "--scale-free-r2",
                "0.8",
                "--deep-split",
                "4",
                "--min-module-size",
                "15",
                "--merge-cut-height",
                "0.15",
            ]
        )
        config_id = _reduction_config_id(args)
        self.assertEqual(
            config_id,
            "r2-0p8_conn-5_ds-4_mms-15_merge-0p15_block-5000_outlier-0p1_seed-42",
        )
        self.assertEqual(
            _reduction_root(args),
            _run_root(args) / "reductions" / config_id,
        )

    def test_transform_resolves_the_same_parameterized_reduction(self) -> None:
        """Transform must locate a fit using the same explicit WGCNA parameters."""

        parser = build_parser()
        common = [
            "--organ",
            "liver",
            "--compound",
            "fenofibrate",
            "--run-name",
            "wgcna_v1",
            "--scale-free-r2",
            "0.8",
            "--deep-split",
            "3",
            "--min-module-size",
            "20",
            "--merge-cut-height",
            "0.2",
        ]
        fit_args = parser.parse_args(["fit-wgcna", *common])
        transform_args = parser.parse_args(["transform", *common])
        self.assertEqual(_reduction_root(fit_args), _reduction_root(transform_args))


if __name__ == "__main__":
    unittest.main()
