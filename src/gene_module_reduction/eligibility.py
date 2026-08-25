"""Build the eight-timepoint exact-dose treatment eligibility contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd

from ..dynamics_common.trajectory import build_treatment_id, canonical_dose


ELIGIBILITY_MODEL_TIMES: Final[tuple[str, ...]] = (
    "3H",
    "6H",
    "9H",
    "1D",
    "4D",
    "8D",
    "15D",
    "29D",
)
FIT_MODEL_TIMES: Final[tuple[str, ...]] = ("3H", "6H", "9H", "1D", "4D", "8D")
EARLY_MODEL_TIMES: Final[tuple[str, ...]] = ("3H", "6H", "9H", "1D")
REPEAT_MODEL_TIMES: Final[tuple[str, ...]] = ("4D", "8D", "15D", "29D")
SOURCE_TO_MODEL_TIME: Final[dict[str, str]] = {
    "3H": "3H",
    "6H": "6H",
    "9H": "9H",
    "24H": "1D",
    "4D": "4D",
    "8D": "8D",
    "15D": "15D",
    "29D": "29D",
}
MODEL_TIME_HOURS: Final[dict[str, int]] = {
    "3H": 3,
    "6H": 6,
    "9H": 9,
    "1D": 24,
    "4D": 96,
    "8D": 192,
    "15D": 360,
    "29D": 696,
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


@dataclass(frozen=True)
class EligibilityResult:
    """Eight-timepoint eligibility artifacts.

    Args:
        eligible_samples: Samples from treatments with exact replicate coverage.
        treatment_eligibility: One audit row per treatment and required timepoint.
        treatment_summary: One row per candidate treatment.
    """

    eligible_samples: pd.DataFrame
    treatment_eligibility: pd.DataFrame
    treatment_summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    """Require columns in a table.

    Args:
        frame: Table to validate.
        required: Required column names.
        source: Human-readable source label.

    Raises:
        ValueError: If required columns are absent.
    """

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _parse_bool(values: pd.Series, name: str) -> pd.Series:
    """Parse a strict boolean series.

    Args:
        values: Values to parse.
        name: Column name used in validation errors.

    Returns:
        Boolean values.
    """

    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(f"{name} contains invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def _normalize_compound_no(values: pd.Series) -> pd.Series:
    """Normalize compound identifiers.

    Args:
        values: Compound identifier values.

    Returns:
        Five-character identifiers.
    """

    normalized = values.astype(str).str.strip().str.upper().str.removeprefix("C")
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized.str.zfill(5)


def read_split_manifest(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Read an optional compound-grouped split manifest.

    Args:
        path: TSV with ``compound_no`` and ``partition`` or ``None``.

    Returns:
        Normalized split manifest or ``None``.

    Raises:
        ValueError: If the split contract is invalid.
    """

    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    split = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    _require_columns(split, {"compound_no", "partition"}, str(path))
    split = split[["compound_no", "partition"]].copy()
    split["compound_no"] = _normalize_compound_no(split["compound_no"])
    split["partition"] = split["partition"].str.strip().str.casefold()
    allowed = {"train", "validation", "test"}
    invalid = sorted(set(split["partition"]).difference(allowed))
    if invalid:
        raise ValueError(f"split manifest has invalid partitions: {invalid}")
    if split["compound_no"].duplicated(keep=False).any():
        duplicates = sorted(split.loc[split["compound_no"].duplicated(False), "compound_no"])
        raise ValueError(f"split manifest has duplicate compounds: {duplicates[:10]}")
    return split.sort_values("compound_no").reset_index(drop=True)


def build_treatment_eligibility(
    manifest: pd.DataFrame,
    organ: str,
    expected_replicates: int = 3,
    split_manifest: Optional[pd.DataFrame] = None,
) -> EligibilityResult:
    """Filter exact-dose treatments with eight complete timepoints.

    Args:
        manifest: Complete log2FC sample manifest.
        organ: Kidney or liver.
        expected_replicates: Exact replicate count required at every timepoint.
        split_manifest: Optional compound-grouped partition table.

    Returns:
        Eligibility samples and audit tables.

    Raises:
        ValueError: If input contracts or eligibility invariants fail.
    """

    normalized_organ = organ.strip().casefold()
    if normalized_organ not in {"kidney", "liver"}:
        raise ValueError(f"organ must be kidney or liver, got {organ!r}")
    if expected_replicates != 3:
        raise ValueError("Gene-module reduction requires exactly three replicates")
    _require_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "log2FC manifest")
    frame = manifest.copy()
    frame["is_control"] = _parse_bool(frame["is_control"], "is_control")
    frame["include_in_log2fc"] = _parse_bool(
        frame["include_in_log2fc"], "include_in_log2fc"
    )
    frame["compound_no"] = _normalize_compound_no(frame["compound_no"])
    regimen = frame["regimen"].astype(str).str.strip().str.casefold()
    time_label = frame["time_label"].astype(str).str.strip().str.upper()
    single_valid = time_label.isin({"3H", "6H", "9H", "24H"}) & regimen.eq("single")
    repeat_valid = time_label.isin({"4D", "8D", "15D", "29D"}) & regimen.eq("repeat")
    candidate = frame.loc[
        frame["organ"].astype(str).str.casefold().eq(normalized_organ)
        & frame["administration_route"].astype(str).str.strip().str.casefold().eq("gavage")
        & ~frame["is_control"]
        & frame["include_in_log2fc"]
        & (single_valid | repeat_valid)
    ].copy()
    if candidate.empty:
        raise ValueError(f"No eligible Gavage candidate samples found for {organ}")
    candidate["time_label"] = candidate["time_label"].astype(str).str.strip().str.upper()
    candidate["model_time"] = candidate["time_label"].map(SOURCE_TO_MODEL_TIME)
    candidate["time_hours"] = candidate["model_time"].map(MODEL_TIME_HOURS).astype(int)
    candidate["canonical_dose"] = candidate["dose"].map(canonical_dose)
    candidate["treatment_id"] = [
        build_treatment_id(row) for row in candidate.to_dict(orient="records")
    ]
    candidate["condition_id"] = candidate["treatment_id"] + "__TIME-" + candidate["model_time"]
    candidate["sample_id"] = candidate["sample_id"].astype(str)
    if candidate["sample_id"].duplicated(keep=False).any():
        duplicates = sorted(candidate.loc[candidate["sample_id"].duplicated(False), "sample_id"])
        raise ValueError(f"candidate manifest has duplicate sample IDs: {duplicates[:10]}")

    treatment_metadata = candidate[
        [
            "treatment_id",
            "compound_no",
            "compound_name",
            "canonical_dose",
            "dose_unit",
        ]
    ].drop_duplicates()
    if treatment_metadata["treatment_id"].duplicated(keep=False).any():
        raise ValueError("A treatment ID maps to conflicting compound or dose metadata")
    if split_manifest is None:
        treatment_metadata["partition"] = "train"
    else:
        _require_columns(split_manifest, {"compound_no", "partition"}, "split manifest")
        split = split_manifest.copy()
        split["compound_no"] = _normalize_compound_no(split["compound_no"])
        treatment_metadata = treatment_metadata.merge(
            split, on="compound_no", how="left", validate="many_to_one"
        )

    treatment_ids = sorted(treatment_metadata["treatment_id"].astype(str))
    audit_index = pd.MultiIndex.from_product(
        [treatment_ids, ELIGIBILITY_MODEL_TIMES], names=["treatment_id", "model_time"]
    )
    counts = (
        candidate.groupby(["treatment_id", "model_time"], observed=True)
        .size()
        .reindex(audit_index, fill_value=0)
        .rename("n_replicates")
        .reset_index()
    )
    counts["timepoint_valid"] = counts["n_replicates"].eq(expected_replicates)
    valid_by_treatment = counts.groupby("treatment_id", observed=True)["timepoint_valid"].all()
    counts["treatment_eligible"] = counts["treatment_id"].map(valid_by_treatment)
    counts["exclusion_reason"] = np.select(
        [
            ~counts["timepoint_valid"],
            counts["timepoint_valid"] & ~counts["treatment_eligible"],
        ],
        ["replicate_count_not_exact", "incomplete_eight_timepoint_set"],
        default="",
    )
    counts["expected_replicates"] = expected_replicates
    counts["time_hours"] = counts["model_time"].map(MODEL_TIME_HOURS).astype(int)
    counts["used_for_reduction_fit"] = counts["model_time"].isin(FIT_MODEL_TIMES)
    counts = counts.merge(
        treatment_metadata.rename(columns={"canonical_dose": "dose"}),
        on="treatment_id",
        how="left",
        validate="many_to_one",
    )
    counts["organ"] = normalized_organ.capitalize()
    counts = counts.sort_values(["treatment_id", "time_hours"]).reset_index(drop=True)

    eligible_ids = set(valid_by_treatment[valid_by_treatment].index)
    eligible_samples = candidate[candidate["treatment_id"].isin(eligible_ids)].copy()
    eligible_samples = eligible_samples.merge(
        treatment_metadata[["treatment_id", "partition"]],
        on="treatment_id",
        how="left",
        validate="many_to_one",
    )
    if eligible_samples["partition"].isna().any():
        missing = sorted(eligible_samples.loc[eligible_samples["partition"].isna(), "compound_no"].unique())
        raise ValueError(f"eligible compounds are missing from split manifest: {missing}")
    eligible_samples["used_for_reduction_fit"] = (
        eligible_samples["partition"].eq("train")
        & eligible_samples["model_time"].isin(FIT_MODEL_TIMES)
    )
    eligible_samples["dose"] = eligible_samples["canonical_dose"]
    sample_columns = [
        "sample_id",
        "condition_id",
        "treatment_id",
        "compound_no",
        "compound_name",
        "dose",
        "dose_unit",
        "dose_level",
        "time_label",
        "model_time",
        "time_hours",
        "regimen",
        "organ",
        "partition",
        "used_for_reduction_fit",
    ]
    eligible_samples = eligible_samples[sample_columns].sort_values(
        ["partition", "treatment_id", "time_hours", "sample_id"]
    ).reset_index(drop=True)
    expected_sample_count = len(eligible_ids) * len(ELIGIBILITY_MODEL_TIMES) * expected_replicates
    if len(eligible_samples) != expected_sample_count:
        raise AssertionError(
            f"eligibility invariant failed: expected {expected_sample_count}, "
            f"found {len(eligible_samples)}"
        )

    summary = counts.groupby("treatment_id", observed=True).first().reset_index()
    summary = summary[
        [
            "treatment_id",
            "compound_no",
            "compound_name",
            "dose",
            "dose_unit",
            "organ",
            "partition",
            "treatment_eligible",
        ]
    ]
    return EligibilityResult(
        eligible_samples=eligible_samples,
        treatment_eligibility=counts,
        treatment_summary=summary,
    )


def filter_eligibility_for_compound(
    result: EligibilityResult,
    compound: str,
) -> tuple[EligibilityResult, str, str]:
    """Restrict an organ eligibility result to one resolved compound.

    Matching is exact and case-insensitive for ``compound_name``. Numeric values,
    with an optional ``C`` prefix, are also matched against ``compound_no``.
    The full organ eligibility calculation must therefore finish before this
    function is called.

    Args:
        result: Complete organ-level eligibility result.
        compound: Requested compound name or identifier.

    Returns:
        Filtered eligibility result, resolved compound number, and resolved name.

    Raises:
        ValueError: If the request is empty, missing, ambiguous, or has no
            eight-timepoint eligible treatment.
    """

    query = compound.strip()
    if not query:
        raise ValueError("compound must not be empty")
    summary = result.treatment_summary.copy()
    compound_numbers = summary["compound_no"].astype(str).str.strip().str.zfill(5)
    compound_names = summary["compound_name"].astype(str).str.strip()
    name_match = compound_names.str.casefold().eq(query.casefold())
    identifier = query.upper().removeprefix("C")
    identifier = re.sub(r"\.0$", "", identifier)
    number_match = (
        compound_numbers.eq(identifier.zfill(5))
        if identifier.isdigit()
        else pd.Series(False, index=summary.index)
    )
    matches = summary.loc[name_match | number_match].copy()
    if matches.empty:
        available = sorted(compound_names.drop_duplicates().tolist())
        preview = ", ".join(available[:10])
        raise ValueError(
            f"Compound {compound!r} was not found in the organ candidates; "
            f"available examples: {preview}"
        )
    matched_numbers = sorted(
        matches["compound_no"].astype(str).str.strip().str.zfill(5).unique()
    )
    if len(matched_numbers) != 1:
        raise ValueError(
            f"Compound {compound!r} is ambiguous across compound numbers: {matched_numbers}"
        )
    resolved_number = matched_numbers[0]
    resolved_names = sorted(
        matches.loc[
            matches["compound_no"].astype(str).str.strip().str.zfill(5).eq(resolved_number),
            "compound_name",
        ]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if len(resolved_names) != 1:
        raise ValueError(
            f"Compound number {resolved_number} maps to conflicting names: {resolved_names}"
        )
    resolved_name = resolved_names[0]

    def select_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """Select rows for the resolved compound while preserving column types."""

        mask = frame["compound_no"].astype(str).str.strip().str.zfill(5).eq(
            resolved_number
        )
        return frame.loc[mask].reset_index(drop=True)

    filtered = EligibilityResult(
        eligible_samples=select_rows(result.eligible_samples),
        treatment_eligibility=select_rows(result.treatment_eligibility),
        treatment_summary=select_rows(result.treatment_summary),
    )
    eligible_count = int(filtered.treatment_summary["treatment_eligible"].sum())
    if eligible_count == 0 or filtered.eligible_samples.empty:
        raise ValueError(
            f"Compound {resolved_name} ({resolved_number}) has no treatment with "
            "exactly three replicates at all eight required timepoints"
        )
    return filtered, resolved_number, resolved_name


def write_eligibility(result: EligibilityResult, directory: Path) -> None:
    """Write eligibility artifacts without overwriting existing files.

    Args:
        result: Eligibility result.
        directory: Output directory.

    Raises:
        FileExistsError: If a target already exists.
    """

    targets = {
        "audit": directory / "treatment_eligibility.tsv",
        "samples": directory / "eligible_samples.tsv.gz",
        "summary": directory / "eligible_treatments.tsv",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Eligibility outputs already exist: " + ", ".join(existing))
    directory.mkdir(parents=True, exist_ok=True)
    result.treatment_eligibility.to_csv(targets["audit"], sep="\t", index=False)
    result.eligible_samples.to_csv(targets["samples"], sep="\t", index=False)
    result.treatment_summary.to_csv(targets["summary"], sep="\t", index=False)
