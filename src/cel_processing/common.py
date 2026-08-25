"""Shared metadata and naming logic for Open TG-GATEs CEL preprocessing."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAT_KEY_COLUMNS: Final[list[str]] = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
BASELINE_COLUMNS: Final[list[str]] = [
    "organ",
    "time_label",
    "administration_route",
]
BIOCHEMISTRY_COLUMNS: Final[dict[str, str]] = {
    "ALT(IU/L)": "alt_iu_l",
    "AST(IU/L)": "ast_iu_l",
    "BUN(mg/dL)": "bun_mg_dl",
    "CRE(mg/dL)": "cre_mg_dl",
}
BIOCHEMISTRY_THRESHOLDS: Final[dict[str, float]] = {
    "alt_iu_l": 37.84,
    "ast_iu_l": 81.72,
    "bun_mg_dl": 16.92,
    "cre_mg_dl": 0.3684,
}
REQUIRED_ATTRIBUTE_COLUMNS: Final[set[str]] = {
    "BARCODE",
    "ARR_DESIGN",
    "EXP_ID",
    "GROUP_ID",
    "INDIVIDUAL_ID",
    "ORGAN_ID",
    "COMPOUND_NAME",
    "COMPOUND Abbr.",
    "COMPOUND_NO",
    "TEST_TYPE",
    "SIN_REP_TYPE",
    "ADM_ROUTE_TYPE",
    "SACRI_PERIOD",
    "DOSE",
    "DOSE_UNIT",
    "DOSE_LEVEL",
    *BIOCHEMISTRY_COLUMNS.keys(),
}
REQUIRED_PATHOLOGY_COLUMNS: Final[set[str]] = {
    "EXP_ID",
    "GROUP_ID",
    "INDIVIDUAL_ID",
    "SP_FLG",
}
IN_VIVO_DIRECTORIES: Final[tuple[str, ...]] = (
    "rat-in-vivo-liver-single",
    "rat-in-vivo-liver-repeat",
    "rat-in-vivo-kidney-single",
    "rat-in-vivo-kidney-repeat",
)
DOSE_RANK: Final[dict[str, int]] = {
    "CONTROL": 0,
    "LOW": 1,
    "MIDDLE": 2,
    "HIGH": 3,
}
ORGAN_RANK: Final[dict[str, int]] = {"Liver": 0, "Kidney": 1}
REGIMEN_RANK: Final[dict[str, int]] = {"Single": 0, "Repeat": 1}


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    """Validate that a table contains all required columns.

    Args:
        frame: Input table.
        required: Required column names.
        source: Human-readable source name for error messages.

    Raises:
        ValueError: If one or more required columns are missing.
    """

    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def canonical_integer(value: object, width: int, field_name: str) -> str:
    """Convert an integer-like metadata value to a zero-padded string.

    Args:
        value: Metadata value.
        width: Minimum output width.
        field_name: Field name for validation errors.

    Returns:
        Zero-padded integer string.

    Raises:
        ValueError: If the value is empty or not integer-like.
    """

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"{field_name} must be integer-like, got {value!r}")
    return text.zfill(width)


def canonical_code(value: object, width: int, field_name: str) -> str:
    """Normalize an alphanumeric metadata code without discarding letters.

    Args:
        value: Metadata code.
        width: Minimum output width.
        field_name: Field name for validation errors.

    Returns:
        Uppercase, zero-padded alphanumeric code.

    Raises:
        ValueError: If the value contains characters outside letters and digits.
    """

    text = str(value).strip().upper()
    if not text or re.fullmatch(r"[A-Z0-9]+", text) is None:
        raise ValueError(f"{field_name} must be alphanumeric, got {value!r}")
    return text.zfill(width)


def slugify(value: object) -> str:
    """Convert metadata text to a stable, uppercase ASCII identifier token.

    Args:
        value: Source metadata value.

    Returns:
        Identifier-safe token containing ASCII letters, digits, and hyphens.

    Raises:
        ValueError: If no identifier characters remain after normalization.
    """

    text = str(value).strip()
    replacements = {
        "%": " PCT ",
        "+": " PLUS ",
        "α": " ALPHA ",
        "Α": " ALPHA ",
        "β": " BETA ",
        "Β": " BETA ",
        "γ": " GAMMA ",
        "Γ": " GAMMA ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii").upper()
    token = re.sub(r"[^A-Z0-9]+", "-", text).strip("-")
    if not token:
        raise ValueError(f"Cannot build identifier token from {value!r}")
    return token


def format_time(period: object) -> tuple[str, int]:
    """Normalize an Open TG-GATEs sacrifice period.

    Args:
        period: SACRI_PERIOD value such as ``3 hr`` or ``4 day``.

    Returns:
        A ``(time_label, time_hours)`` tuple.

    Raises:
        ValueError: If the period is not an integer number of hours or days.
    """

    text = str(period).strip()
    match = re.fullmatch(r"(\d+)\s*(hr|day)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unsupported SACRI_PERIOD value: {period!r}")
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "hr":
        return f"{amount}H", amount
    return f"{amount}D", amount * 24


def build_rat_id(exp_id: object, group_id: object, individual_id: object) -> str:
    """Build the stable rat identity used to propagate pathology flags.

    Args:
        exp_id: Open TG-GATEs experiment identifier.
        group_id: Open TG-GATEs group identifier.
        individual_id: Rat identifier within the group.

    Returns:
        Rat identifier such as ``E0040-G01-I1``.
    """

    exp = canonical_integer(exp_id, 4, "EXP_ID")
    group = canonical_integer(group_id, 2, "GROUP_ID")
    individual = canonical_integer(individual_id, 1, "INDIVIDUAL_ID")
    return f"E{exp}-G{group}-I{individual}"


def build_sample_id(row: Mapping[str, object]) -> str:
    """Build a complete, parseable Sample ID from normalized metadata.

    Args:
        row: Mapping containing normalized sample metadata.

    Returns:
        Full Sample ID containing compound, dose, time, organ, regimen, route,
        rat identity, and CEL Barcode.
    """

    compound_no = canonical_code(row["compound_no"], 5, "COMPOUND_NO")
    compound = slugify(row["compound_name"])
    barcode = canonical_integer(row["barcode"], 12, "BARCODE")
    fields = [
        f"C{compound_no}-{compound}",
        f"DOSE-{slugify(row['dose_level'])}",
        f"TIME-{row['time_label']}",
        f"ORGAN-{slugify(row['organ'])}",
        f"REGIMEN-{slugify(row['regimen'])}",
        f"ROUTE-{slugify(row['administration_route'])}",
        f"RAT-{row['rat_id']}",
        f"BARCODE-{barcode}",
    ]
    return "__".join(fields)


def read_attribute_table(path: Path) -> pd.DataFrame:
    """Read and filter the complete Open TG-GATEs attribute table.

    Args:
        path: Path to ``Open-tggates_AllAttribute.tsv``.

    Returns:
        CEL-backed Rat230_2 in-vivo rows.
    """

    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
    )
    require_columns(frame, REQUIRED_ATTRIBUTE_COLUMNS, str(path))
    test_type = frame["TEST_TYPE"].str.strip().str.casefold()
    barcode = frame["BARCODE"].str.strip()
    design = frame["ARR_DESIGN"].str.strip()
    return frame.loc[
        test_type.eq("in vivo")
        & design.eq("Rat230_2")
        & barcode.ne("")
        & barcode.ne("No ChipData")
    ].copy()


def read_pathology_table(path: Path) -> pd.DataFrame:
    """Read the Shift-JIS Open TG-GATEs pathology table.

    Args:
        path: Path to ``open_tggates_pathology.csv``.

    Returns:
        Pathology rows, including rows without chip data.
    """

    frame = pd.read_csv(
        path,
        dtype=str,
        encoding="shift_jis",
        keep_default_na=False,
    )
    require_columns(frame, REQUIRED_PATHOLOGY_COLUMNS, str(path))
    return frame


def index_cel_files(raw_dir: Path) -> dict[str, Path]:
    """Index in-vivo CEL files by Barcode.

    Args:
        raw_dir: Open TG-GATEs raw-data root.

    Returns:
        Barcode-to-path mapping.

    Raises:
        ValueError: If a Barcode occurs in multiple CEL paths.
    """

    indexed: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for directory_name in IN_VIVO_DIRECTORIES:
        directory = raw_dir / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing in-vivo directory: {directory}")
        for cel_path in directory.rglob("*.CEL"):
            barcode = cel_path.stem
            if barcode in indexed:
                duplicates.setdefault(barcode, [indexed[barcode]]).append(cel_path)
            else:
                indexed[barcode] = cel_path
    if duplicates:
        examples = {key: [str(path) for path in value] for key, value in list(duplicates.items())[:5]}
        raise ValueError(f"Duplicate CEL Barcodes found: {examples}")
    return indexed


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    """Return a project-relative path when possible.

    Args:
        path: Input path.
        project_root: Project root used for relative paths.

    Returns:
        POSIX path string.
    """

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_rat_invariants(frame: pd.DataFrame) -> None:
    """Validate fields that must be constant within one rat.

    Args:
        frame: Normalized sample metadata with ``rat_id``.

    Raises:
        ValueError: If a rat maps to conflicting experimental metadata.
    """

    invariant_columns = [
        "compound_no",
        "compound_name",
        "dose_level",
        "time_label",
        "regimen",
        "administration_route",
        *BIOCHEMISTRY_THRESHOLDS.keys(),
    ]
    conflicts: dict[str, list[str]] = {}
    for column in invariant_columns:
        counts = frame.groupby("rat_id", sort=False)[column].nunique(dropna=False)
        bad = counts[counts > 1]
        if not bad.empty:
            conflicts[column] = bad.index[:5].tolist()
    if conflicts:
        raise ValueError(f"Conflicting metadata within rat_id groups: {conflicts}")


def build_sample_manifest(
    attributes: pd.DataFrame,
    pathology: pd.DataFrame,
    cel_paths: Mapping[str, Path],
    project_root: Path,
    min_controls: int = 3,
) -> pd.DataFrame:
    """Build the complete sample manifest and filtering audit.

    Args:
        attributes: CEL-backed in-vivo attribute rows.
        pathology: Complete pathology table, including ``No ChipData`` rows.
        cel_paths: Barcode-to-CEL path mapping.
        project_root: Project root for portable CEL paths.
        min_controls: Minimum healthy controls required for a log2FC pool.

    Returns:
        One manifest row per CEL-backed in-vivo sample.

    Raises:
        ValueError: If metadata and CEL Barcodes do not match one-to-one or if
            required invariants are violated.
    """

    require_columns(attributes, REQUIRED_ATTRIBUTE_COLUMNS, "attributes")
    require_columns(pathology, REQUIRED_PATHOLOGY_COLUMNS, "pathology")
    if min_controls < 1:
        raise ValueError("min_controls must be at least 1")

    metadata_barcodes = set(attributes["BARCODE"].str.strip())
    cel_barcodes = set(cel_paths)
    missing_cels = sorted(metadata_barcodes.difference(cel_barcodes))
    missing_metadata = sorted(cel_barcodes.difference(metadata_barcodes))
    if missing_cels or missing_metadata:
        raise ValueError(
            "Metadata/CEL Barcode mismatch: "
            f"metadata_without_cel={missing_cels[:10]}, "
            f"cel_without_metadata={missing_metadata[:10]}"
        )

    normalized = pd.DataFrame(index=attributes.index)
    normalized["barcode"] = attributes["BARCODE"].str.strip()
    normalized["compound_no"] = attributes["COMPOUND_NO"].map(
        lambda value: canonical_code(value, 5, "COMPOUND_NO")
    )
    normalized["compound_name"] = attributes["COMPOUND_NAME"].str.strip()
    normalized["compound_abbreviation"] = attributes["COMPOUND Abbr."].str.strip()
    normalized["dose"] = attributes["DOSE"].str.strip()
    normalized["dose_unit"] = attributes["DOSE_UNIT"].str.strip()
    normalized["dose_level"] = attributes["DOSE_LEVEL"].str.strip().str.title()
    times = attributes["SACRI_PERIOD"].map(format_time)
    normalized["time_label"] = times.map(lambda value: value[0])
    normalized["time_hours"] = times.map(lambda value: value[1]).astype(int)
    normalized["organ"] = attributes["ORGAN_ID"].str.strip().str.title()
    normalized["regimen"] = attributes["SIN_REP_TYPE"].str.strip().str.title()
    normalized["administration_route"] = attributes["ADM_ROUTE_TYPE"].str.strip()
    normalized["exp_id"] = attributes["EXP_ID"].map(
        lambda value: canonical_integer(value, 4, "EXP_ID")
    )
    normalized["group_id"] = attributes["GROUP_ID"].map(
        lambda value: canonical_integer(value, 2, "GROUP_ID")
    )
    normalized["individual_id"] = attributes["INDIVIDUAL_ID"].map(
        lambda value: canonical_integer(value, 1, "INDIVIDUAL_ID")
    )
    normalized["rat_id"] = [
        build_rat_id(exp_id, group_id, individual_id)
        for exp_id, group_id, individual_id in zip(
            normalized["exp_id"],
            normalized["group_id"],
            normalized["individual_id"],
        )
    ]
    for source_column, output_column in BIOCHEMISTRY_COLUMNS.items():
        normalized[output_column] = pd.to_numeric(
            attributes[source_column].replace("", pd.NA), errors="coerce"
        )
    normalized["cel_path"] = normalized["barcode"].map(
        lambda barcode: _relative_or_absolute(cel_paths[barcode], project_root)
    )
    _validate_rat_invariants(normalized)

    pathology_normalized = pathology.copy()
    pathology_normalized["rat_id"] = [
        build_rat_id(exp_id, group_id, individual_id)
        for exp_id, group_id, individual_id in zip(
            pathology_normalized["EXP_ID"],
            pathology_normalized["GROUP_ID"],
            pathology_normalized["INDIVIDUAL_ID"],
        )
    ]
    pathology_rat_ids = set(pathology_normalized["rat_id"])
    sp_mask = pathology_normalized["SP_FLG"].str.strip().str.casefold().eq("true")
    sp_rat_ids = set(pathology_normalized.loc[sp_mask, "rat_id"])

    normalized["has_any_pathology"] = normalized["rat_id"].isin(pathology_rat_ids)
    normalized["has_sp_pathology"] = normalized["rat_id"].isin(sp_rat_ids)
    normalized["is_control"] = normalized["dose_level"].str.upper().eq("CONTROL")
    biochemistry_complete = normalized[list(BIOCHEMISTRY_THRESHOLDS)].notna().all(axis=1)
    biochemistry_negative = pd.Series(True, index=normalized.index, dtype=bool)
    for column, threshold in BIOCHEMISTRY_THRESHOLDS.items():
        biochemistry_negative &= normalized[column].le(threshold)
    normalized["biochemistry_complete"] = biochemistry_complete
    normalized["biochemistry_negative"] = biochemistry_complete & biochemistry_negative
    normalized["is_healthy_control"] = (
        normalized["is_control"]
        & ~normalized["has_any_pathology"]
        & normalized["biochemistry_negative"]
    )

    exclusion_reason = pd.Series("", index=normalized.index, dtype="object")
    exclusion_reason.loc[normalized["has_sp_pathology"]] = "rat_has_sp_pathology"
    remaining_control = normalized["is_control"] & exclusion_reason.eq("")
    exclusion_reason.loc[remaining_control & normalized["has_any_pathology"]] = (
        "control_rat_has_pathology"
    )
    remaining_control = normalized["is_control"] & exclusion_reason.eq("")
    exclusion_reason.loc[remaining_control & ~normalized["biochemistry_complete"]] = (
        "control_biochemistry_missing"
    )
    remaining_control = normalized["is_control"] & exclusion_reason.eq("")
    exclusion_reason.loc[remaining_control & ~normalized["biochemistry_negative"]] = (
        "control_biochemistry_above_threshold"
    )
    normalized["rma_exclusion_reason"] = exclusion_reason
    normalized["include_in_rma"] = exclusion_reason.eq("")

    normalized["baseline_group_id"] = normalized.apply(
        lambda row: "__".join(
            [
                f"ORGAN-{slugify(row['organ'])}",
                f"TIME-{row['time_label']}",
                f"ROUTE-{slugify(row['administration_route'])}",
            ]
        ),
        axis=1,
    )
    healthy_controls = normalized[
        normalized["include_in_rma"] & normalized["is_healthy_control"]
    ]
    control_counts = healthy_controls.groupby(BASELINE_COLUMNS, sort=False)["barcode"].nunique()
    normalized["baseline_n_controls"] = pd.Series(
        [
            control_counts.get(
                (row.organ, row.time_label, row.administration_route),
                0,
            )
            for row in normalized.itertuples()
        ],
        index=normalized.index,
        dtype="int64",
    )
    normalized["include_in_log2fc"] = (
        normalized["include_in_rma"] & normalized["baseline_n_controls"].ge(min_controls)
    )
    log2fc_reason = pd.Series("", index=normalized.index, dtype="object")
    log2fc_reason.loc[~normalized["include_in_rma"]] = "not_in_rma"
    log2fc_reason.loc[
        normalized["include_in_rma"] & ~normalized["include_in_log2fc"]
    ] = "baseline_pool_lt_min_controls"
    normalized["log2fc_exclusion_reason"] = log2fc_reason
    normalized["sample_id"] = normalized.apply(build_sample_id, axis=1)

    if normalized["sample_id"].duplicated().any():
        duplicate_ids = normalized.loc[
            normalized["sample_id"].duplicated(keep=False), "sample_id"
        ].head(10)
        raise ValueError(f"Duplicate Sample IDs generated: {duplicate_ids.tolist()}")
    if normalized["barcode"].duplicated().any():
        raise ValueError("Duplicate Barcodes found in normalized metadata")

    normalized["_compound_sort"] = normalized["compound_no"]
    normalized["_regimen_sort"] = normalized["regimen"].map(REGIMEN_RANK).fillna(99)
    normalized["_organ_sort"] = normalized["organ"].map(ORGAN_RANK).fillna(99)
    normalized["_dose_sort"] = (
        normalized["dose_level"].str.upper().map(DOSE_RANK).fillna(99)
    )
    normalized = normalized.sort_values(
        [
            "_compound_sort",
            "_regimen_sort",
            "time_hours",
            "_organ_sort",
            "_dose_sort",
            "exp_id",
            "group_id",
            "individual_id",
            "barcode",
        ],
        kind="stable",
    ).drop(columns=["_compound_sort", "_regimen_sort", "_organ_sort", "_dose_sort"])

    output_columns = [
        "sample_id",
        "barcode",
        "cel_path",
        "rat_id",
        "compound_no",
        "compound_name",
        "compound_abbreviation",
        "dose",
        "dose_unit",
        "dose_level",
        "time_label",
        "time_hours",
        "organ",
        "regimen",
        "administration_route",
        "exp_id",
        "group_id",
        "individual_id",
        "alt_iu_l",
        "ast_iu_l",
        "bun_mg_dl",
        "cre_mg_dl",
        "has_any_pathology",
        "has_sp_pathology",
        "is_control",
        "biochemistry_complete",
        "biochemistry_negative",
        "is_healthy_control",
        "include_in_rma",
        "rma_exclusion_reason",
        "baseline_group_id",
        "baseline_n_controls",
        "include_in_log2fc",
        "log2fc_exclusion_reason",
    ]
    return normalized[output_columns].reset_index(drop=True)


def build_dataset_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Create machine-readable sample counts for documentation and auditing.

    Args:
        manifest: Complete sample manifest.

    Returns:
        Long-form summary covering input, RMA, and log2FC stages.
    """

    stages: Sequence[tuple[str, pd.Series]] = (
        ("input", pd.Series(True, index=manifest.index)),
        ("rma", manifest["include_in_rma"].astype(bool)),
        ("log2fc", manifest["include_in_log2fc"].astype(bool)),
    )
    rows: list[dict[str, object]] = []
    for stage, mask in stages:
        subset = manifest.loc[mask]
        sample_classes = {
            "all": pd.Series(True, index=subset.index),
            "control": subset["is_control"].astype(bool),
            "treated": ~subset["is_control"].astype(bool),
        }
        for sample_class, class_mask in sample_classes.items():
            rows.append(
                {
                    "section": "overall",
                    "stage": stage,
                    "organ": "ALL",
                    "administration_route": "ALL",
                    "time_label": "ALL",
                    "sample_class": sample_class,
                    "count": int(class_mask.sum()),
                }
            )
        for organ, group in subset.groupby("organ", sort=True):
            rows.append(
                {
                    "section": "organ",
                    "stage": stage,
                    "organ": organ,
                    "administration_route": "ALL",
                    "time_label": "ALL",
                    "sample_class": "all",
                    "count": len(group),
                }
            )
        for route, group in subset.groupby("administration_route", sort=True):
            rows.append(
                {
                    "section": "route",
                    "stage": stage,
                    "organ": "ALL",
                    "administration_route": route,
                    "time_label": "ALL",
                    "sample_class": "all",
                    "count": len(group),
                }
            )
        for (time_label, organ), group in subset.groupby(
            ["time_label", "organ"], sort=False
        ):
            rows.append(
                {
                    "section": "time_organ",
                    "stage": stage,
                    "organ": organ,
                    "administration_route": "ALL",
                    "time_label": time_label,
                    "sample_class": "all",
                    "count": len(group),
                }
            )
    for reason, group in manifest.loc[~manifest["include_in_rma"]].groupby(
        "rma_exclusion_reason", sort=True
    ):
        rows.append(
            {
                "section": "rma_exclusion",
                "stage": "rma",
                "organ": "ALL",
                "administration_route": "ALL",
                "time_label": "ALL",
                "sample_class": str(reason),
                "count": len(group),
            }
        )
    return pd.DataFrame(rows)


def parse_bool_series(values: pd.Series, column_name: str) -> pd.Series:
    """Parse a TSV boolean column without accepting ambiguous values.

    Args:
        values: Input Series.
        column_name: Column name for errors.

    Returns:
        Boolean Series.

    Raises:
        ValueError: If a value is not an accepted boolean token.
    """

    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(f"Invalid boolean values in {column_name}: {invalid}")
    return normalized.map(mapping).astype(bool)
