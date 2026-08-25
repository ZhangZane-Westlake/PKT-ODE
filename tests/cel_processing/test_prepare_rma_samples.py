"""Tests for Sample ID construction and rat-level filtering."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from src.cel_processing.common import (
    build_dataset_summary,
    build_sample_manifest,
    build_sample_id,
)


def attribute_row(
    barcode: str,
    exp_id: str,
    group_id: str,
    individual_id: str,
    organ: str,
    dose_level: str,
    *,
    alt: str = "20",
    ast: str = "60",
    bun: str = "12",
    cre: str = "0.2",
) -> dict[str, str]:
    """Create one minimal synthetic AllAttribute row.

    Args:
        barcode: CEL Barcode.
        exp_id: Experiment identifier.
        group_id: Group identifier.
        individual_id: Rat identifier within the group.
        organ: Organ name.
        dose_level: Control or treated dose level.
        alt: ALT value.
        ast: AST value.
        bun: BUN value.
        cre: CRE value.

    Returns:
        Synthetic attribute row.
    """

    return {
        "BARCODE": barcode,
        "ARR_DESIGN": "Rat230_2",
        "EXP_ID": exp_id,
        "GROUP_ID": group_id,
        "INDIVIDUAL_ID": individual_id,
        "ORGAN_ID": organ,
        "COMPOUND_NAME": "acetaminophen",
        "COMPOUND Abbr.": "APAP",
        "COMPOUND_NO": "00001",
        "TEST_TYPE": "in vivo",
        "SIN_REP_TYPE": "Single",
        "ADM_ROUTE_TYPE": "Gavage",
        "SACRI_PERIOD": "3 hr",
        "DOSE": "0" if dose_level == "Control" else "300",
        "DOSE_UNIT": "mg/kg",
        "DOSE_LEVEL": dose_level,
        "ALT(IU/L)": alt,
        "AST(IU/L)": ast,
        "BUN(mg/dL)": bun,
        "CRE(mg/dL)": cre,
    }


def pathology_row(
    exp_id: str,
    group_id: str,
    individual_id: str,
    sp_flag: str,
) -> dict[str, str]:
    """Create one minimal synthetic pathology row.

    Args:
        exp_id: Experiment identifier.
        group_id: Group identifier.
        individual_id: Rat identifier within the group.
        sp_flag: Pathology SP flag.

    Returns:
        Synthetic pathology row.
    """

    return {
        "BARCODE": "No ChipData",
        "EXP_ID": exp_id,
        "GROUP_ID": group_id,
        "INDIVIDUAL_ID": individual_id,
        "SP_FLG": sp_flag,
    }


class PrepareRmaSamplesTests(unittest.TestCase):
    """Validate naming, rat mapping, and filtering contracts."""

    def test_complete_sample_id_format(self) -> None:
        """Sample IDs should contain all accepted fields in a stable order."""

        row: dict[str, Any] = {
            "compound_no": "00001",
            "compound_name": "acetaminophen",
            "dose_level": "Control",
            "time_label": "3H",
            "organ": "Liver",
            "regimen": "Single",
            "administration_route": "Gavage",
            "rat_id": "E0040-G01-I1",
            "barcode": "003017644018",
        }
        self.assertEqual(
            build_sample_id(row),
            "C00001-ACETAMINOPHEN__DOSE-CONTROL__TIME-3H__ORGAN-LIVER"
            "__REGIMEN-SINGLE__ROUTE-GAVAGE__RAT-E0040-G01-I1"
            "__BARCODE-003017644018",
        )

    def test_rat_level_filtering_and_control_rules(self) -> None:
        """SP, pathology, and biochemistry rules should operate at rat level."""

        rows = [
            attribute_row("000000000001", "0001", "01", "1", "Liver", "Control"),
            attribute_row("000000000002", "0001", "01", "1", "Kidney", "Control"),
            attribute_row("000000000003", "0002", "01", "1", "Liver", "Control"),
            attribute_row(
                "000000000004", "0003", "01", "1", "Liver", "Control", alt="50"
            ),
            attribute_row("000000000005", "0004", "01", "1", "Liver", "High"),
        ]
        pathology = pd.DataFrame(
            [
                pathology_row("0001", "01", "1", "true"),
                pathology_row("0004", "01", "1", "false"),
            ]
        )
        cel_paths = {
            row["BARCODE"]: Path("data/raw") / f"{row['BARCODE']}.CEL" for row in rows
        }
        manifest = build_sample_manifest(
            attributes=pd.DataFrame(rows),
            pathology=pathology,
            cel_paths=cel_paths,
            project_root=Path("."),
            min_controls=1,
        )
        by_barcode = manifest.set_index("barcode")

        self.assertEqual(by_barcode.loc["000000000001", "rat_id"], "E0001-G01-I1")
        self.assertTrue(by_barcode.loc["000000000001", "has_sp_pathology"])
        self.assertTrue(by_barcode.loc["000000000002", "has_sp_pathology"])
        self.assertFalse(by_barcode.loc["000000000001", "include_in_rma"])
        self.assertFalse(by_barcode.loc["000000000002", "include_in_rma"])
        self.assertTrue(by_barcode.loc["000000000003", "is_healthy_control"])
        self.assertTrue(by_barcode.loc["000000000003", "include_in_rma"])
        self.assertEqual(
            by_barcode.loc["000000000004", "rma_exclusion_reason"],
            "control_biochemistry_above_threshold",
        )
        self.assertTrue(by_barcode.loc["000000000005", "has_any_pathology"])
        self.assertFalse(by_barcode.loc["000000000005", "has_sp_pathology"])
        self.assertTrue(by_barcode.loc["000000000005", "include_in_rma"])
        self.assertEqual(manifest["sample_id"].nunique(), len(manifest))

    def test_missing_control_biochemistry_is_excluded(self) -> None:
        """A missing control measurement should fail the healthy-control rule."""

        row = attribute_row(
            "000000000010", "0010", "01", "1", "Liver", "Control", cre=""
        )
        manifest = build_sample_manifest(
            attributes=pd.DataFrame([row]),
            pathology=pd.DataFrame(
                columns=["BARCODE", "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "SP_FLG"]
            ),
            cel_paths={row["BARCODE"]: Path("data/raw/000000000010.CEL")},
            project_root=Path("."),
            min_controls=1,
        )
        self.assertEqual(
            manifest.loc[0, "rma_exclusion_reason"], "control_biochemistry_missing"
        )
        self.assertFalse(manifest.loc[0, "include_in_rma"])

    def test_dataset_summary_tracks_all_stages(self) -> None:
        """Dataset summaries should contain input, RMA, and log2FC counts."""

        manifest = pd.DataFrame(
            {
                "organ": ["Liver", "Kidney"],
                "administration_route": ["Gavage", "Gavage"],
                "time_label": ["3H", "3H"],
                "is_control": [True, False],
                "include_in_rma": [True, True],
                "include_in_log2fc": [True, False],
                "rma_exclusion_reason": ["", ""],
            }
        )
        summary = build_dataset_summary(manifest)
        overall = summary[
            (summary["section"] == "overall")
            & (summary["sample_class"] == "all")
        ].set_index("stage")["count"]
        self.assertEqual(overall.to_dict(), {"input": 2, "rma": 2, "log2fc": 1})


if __name__ == "__main__":
    unittest.main()

