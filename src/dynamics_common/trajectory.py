"""Shared exact-dose trajectory selection contracts."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Final, Mapping

import numpy as np
import pandas as pd


MODEL_DAYS: Final[tuple[str, ...]] = ("1D", "4D", "8D", "15D", "29D")
DAY_NUMBER: Final[dict[str, int]] = {
    "1D": 1,
    "4D": 4,
    "8D": 8,
    "15D": 15,
    "29D": 29,
}
TIME_TO_MODEL_DAY: Final[dict[str, str]] = {
    "24H": "1D",
    "4D": "4D",
    "8D": "8D",
    "15D": "15D",
    "29D": "29D",
}
REQUIRED_MANIFEST_COLUMNS: Final[set[str]] = {
    "sample_id",
    "compound_no",
    "compound_name",
    "dose",
    "dose_unit",
    "dose_level",
    "time_label",
    "organ",
    "regimen",
    "administration_route",
    "is_control",
    "include_in_log2fc",
}
SAMPLE_METADATA_COLUMNS: Final[list[str]] = [
    "sample_id",
    "treatment_id",
    "compound_no",
    "compound_name",
    "dose",
    "dose_unit",
    "dose_level",
    "model_day",
    "day",
    "organ",
]
TRAJECTORY_METADATA_COLUMNS: Final[list[str]] = [
    "treatment_id",
    "compound_no",
    "compound_name",
    "dose",
    "dose_unit",
    "dose_levels",
    "model_day",
    "day",
    "organ",
    "n_replicates",
    "sample_ids",
]


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    """Validate a table contract.

    Args:
        frame: Input table.
        required: Required columns.
        source: Human-readable source name.

    Raises:
        ValueError: If required columns are absent.
    """

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _parse_bool_series(values: pd.Series, field_name: str) -> pd.Series:
    """Parse a manifest boolean column without string truthiness.

    Args:
        values: Values to parse.
        field_name: Column name used in errors.

    Returns:
        Parsed boolean series.
    """

    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(f"{field_name} contains invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def canonical_dose(value: object) -> str:
    """Convert a dose to the established exact-dose identifier token.

    Args:
        value: Raw numeric dose.

    Returns:
        Canonical decimal dose string.
    """

    text = str(value).strip()
    try:
        dose = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Dose must be numeric, got {value!r}") from exc
    if not dose.is_finite():
        raise ValueError(f"Dose must be finite, got {value!r}")
    canonical = format(dose.normalize(), "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    return canonical or "0"


def build_treatment_id(row: Mapping[str, object]) -> str:
    """Build the established organ/compound/exact-dose treatment ID.

    Args:
        row: Manifest-like mapping.

    Returns:
        Stable treatment identifier.
    """

    organ = str(row["organ"]).strip().upper()
    compound_no = str(row["compound_no"]).strip().upper().zfill(5)
    dose = canonical_dose(row["dose"])
    dose_unit = re.sub(
        r"[^A-Z0-9]+", "-", str(row["dose_unit"]).strip().upper()
    ).strip("-")
    return f"ORGAN-{organ}__C{compound_no}__DOSE-{dose}-{dose_unit}"


# Backward-compatible private aliases for existing callers and saved workflows.
_canonical_dose = canonical_dose
_build_treatment_id = build_treatment_id


def build_eligibility_audit(
    manifest: pd.DataFrame,
    organ: str,
    expected_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exact-replicate and complete-five-day trajectory filters.

    Args:
        manifest: Full log2FC sample manifest.
        organ: Organ to prepare.
        expected_replicates: Exact replicate count required per model day.

    Returns:
        Eligible sample rows and the treatment-day audit table.
    """

    _require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "log2FC manifest")
    frame = manifest.copy()
    frame["is_control"] = _parse_bool_series(frame["is_control"], "is_control")
    frame["include_in_log2fc"] = _parse_bool_series(
        frame["include_in_log2fc"], "include_in_log2fc"
    )
    organ_mask = frame["organ"].astype(str).str.casefold() == organ.casefold()
    route_mask = (
        frame["administration_route"].astype(str).str.strip().str.casefold()
        == "gavage"
    )
    time_mask = frame["time_label"].astype(str).isin(TIME_TO_MODEL_DAY)
    later_repeat = (
        frame["time_label"].astype(str).eq("24H")
        | frame["regimen"].astype(str).str.strip().str.casefold().eq("repeat")
    )
    candidate = frame.loc[
        organ_mask
        & route_mask
        & time_mask
        & later_repeat
        & ~frame["is_control"]
        & frame["include_in_log2fc"]
    ].copy()
    if candidate.empty:
        raise ValueError(f"No eligible Gavage candidate samples found for {organ}")
    candidate["model_day"] = candidate["time_label"].map(TIME_TO_MODEL_DAY)
    candidate["day"] = candidate["model_day"].map(DAY_NUMBER).astype(int)
    candidate["canonical_dose"] = candidate["dose"].map(canonical_dose)
    candidate["treatment_id"] = [
        build_treatment_id(row) for row in candidate.to_dict(orient="records")
    ]
    duplicated_samples = candidate["sample_id"].astype(str).duplicated(keep=False)
    if duplicated_samples.any():
        duplicates = sorted(candidate.loc[duplicated_samples, "sample_id"].astype(str))
        raise ValueError(f"Candidate manifest has duplicate sample IDs: {duplicates[:5]}")

    treatment_metadata = candidate[
        ["treatment_id", "compound_no", "compound_name", "canonical_dose", "dose_unit"]
    ].drop_duplicates()
    if treatment_metadata["treatment_id"].duplicated(keep=False).any():
        raise ValueError("A treatment ID maps to conflicting compound or dose metadata")

    treatment_ids = sorted(candidate["treatment_id"].unique())
    index = pd.MultiIndex.from_product(
        [treatment_ids, MODEL_DAYS], names=["treatment_id", "model_day"]
    )
    counts = (
        candidate.groupby(["treatment_id", "model_day"], observed=True)
        .size()
        .reindex(index, fill_value=0)
        .rename("n_replicates")
        .reset_index()
    )
    counts["timepoint_valid"] = counts["n_replicates"].eq(expected_replicates)
    treatment_valid = counts.groupby("treatment_id", observed=True)[
        "timepoint_valid"
    ].all()
    counts["treatment_eligible"] = counts["treatment_id"].map(treatment_valid)
    counts["exclusion_reason"] = np.select(
        [
            ~counts["timepoint_valid"],
            counts["timepoint_valid"] & ~counts["treatment_eligible"],
        ],
        ["replicate_count_not_exact", "incomplete_valid_timepoint_set"],
        default="",
    )
    counts["expected_replicates"] = expected_replicates
    counts = counts.merge(
        treatment_metadata.rename(columns={"canonical_dose": "dose"}),
        on="treatment_id",
        how="left",
        validate="many_to_one",
    )
    counts["day"] = counts["model_day"].map(DAY_NUMBER).astype(int)
    counts["organ"] = organ.capitalize()
    counts = counts.sort_values(["treatment_id", "day"]).reset_index(drop=True)

    eligible_ids = set(treatment_valid[treatment_valid].index)
    eligible = candidate[candidate["treatment_id"].isin(eligible_ids)].copy()
    expected_samples = len(eligible_ids) * len(MODEL_DAYS) * expected_replicates
    if len(eligible) != expected_samples:
        raise AssertionError(
            f"Eligibility invariant failed: expected {expected_samples} samples, "
            f"found {len(eligible)}"
        )
    return eligible, counts
