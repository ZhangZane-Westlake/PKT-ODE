"""Human-readable reporting for module-eigengene dynamics."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import pandas as pd

from .data import ModuleTrajectoryMatrix


EXPECTED_REPLICATES_PER_TREATMENT_DAY = 3


def _format_parameter_value(value: object) -> str:
    """Format a configuration value without losing nested structure.

    Args:
        value: Configuration value.

    Returns:
        Human-readable scalar or JSON representation.
    """

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def format_parameter_section(parameters: Mapping[str, object]) -> list[str]:
    """Format the complete configuration.

    Args:
        parameters: Ordered configuration.

    Returns:
        Summary lines headed by ``Parameter configuration``.
    """

    return [
        "Parameter configuration:",
        *(f"- {name}: {_format_parameter_value(value)}" for name, value in parameters.items()),
    ]


def _split_size_lines(
    label: str,
    days: Sequence[int],
    trajectories: ModuleTrajectoryMatrix,
    role: str,
) -> list[str]:
    """Format sizes and roles for one temporal split.

    Args:
        label: Split heading.
        days: Days represented by the split.
        trajectories: Prepared module trajectories.
        role: Modeling role of the split.

    Returns:
        Split-specific summary lines.
    """

    n_treatments = len(trajectories.treatment_ids)
    n_modules = len(trajectories.module_ids)
    n_treatment_day_rows = n_treatments * len(days)
    return [
        f"{label}:",
        f"- days: {', '.join(f'{day}D' for day in days)}",
        f"- role: {role}",
        f"- treatments: {n_treatments}",
        f"- treatment_day_rows: {n_treatment_day_rows}",
        f"- module_values: {n_treatment_day_rows * n_modules}",
        f"- underlying_biological_replicates: {n_treatment_day_rows * EXPECTED_REPLICATES_PER_TREATMENT_DAY}",
    ]


def format_baseline_data_section(
    trajectories: ModuleTrajectoryMatrix,
    fit_days: Sequence[int] = (1, 4, 8),
    predict_days: Sequence[int] = (15, 29),
) -> list[str]:
    """Format dataset identities and baseline temporal splits.

    Args:
        trajectories: Prepared module trajectories.
        fit_days: Days the baselines consume (training/input split).
        predict_days: Held-out days; the latest is the test split and any earlier
            days form the validation split.

    Returns:
        Complete dataset and split summary lines.
    """

    n_treatments = len(trajectories.treatment_ids)
    n_modules = len(trajectories.module_ids)
    scope = trajectories.scope
    fit_uses_15d = 15 in fit_days
    validation_days = tuple(day for day in predict_days[:-1])
    test_days = (predict_days[-1],)
    lines = [
        "Dataset configuration:",
        f"- scope: {scope.scope_tag} ({scope.split_id}/{scope.config_id})",
        f"- n_treatments: {n_treatments}",
        f"- n_modules: {n_modules}",
        "- administration_route: Gavage",
        "- trajectory_days: 1D, 4D, 8D, 15D, 29D",
        f"- expected_replicates_per_treatment_day: {EXPECTED_REPLICATES_PER_TREATMENT_DAY}",
        "- module_unit: train-standardized fixed WGCNA module PC1 score",
        f"- total_treatment_day_rows: {n_treatments * 5}",
        f"- total_module_values: {n_treatments * 5 * n_modules}",
        f"- total_underlying_biological_replicates: {n_treatments * 5 * EXPECTED_REPLICATES_PER_TREATMENT_DAY}",
        "- split_axis: timepoint",
        "- same_treatments_across_train_validation_test: true",
        "- task_scope: known-treatment temporal extrapolation",
        "",
        "Split details:",
        *_split_size_lines(
            "Training/input split",
            fit_days,
            trajectories,
            "parameter-free calculation or independent per-module curve fitting; no optimizer",
        ),
        *(
            _split_size_lines(
                "Validation split",
                validation_days,
                trajectories,
                "evaluation only; no baseline tuning or model selection",
            )
            if validation_days
            else ["Validation split:", "- days: none (every non-test day is a fit day)"]
        ),
        *_split_size_lines(
            "Test split",
            test_days,
            trajectories,
            (
                "final evaluation; no held-out day is used as an input"
                if fit_uses_15d
                else f"final evaluation; {validation_days[0] if validation_days else 15}D is not used as a {test_days[0]}D input"
            ),
        ),
        "- scaler_fit_days_for_standardized_metrics: 1D, 4D, 8D",
        "- each_treatment_module_is_fit_independently: true",
        "- uses_other_modules_or_treatments: false",
        "",
        f"Treatment IDs ({n_treatments}):",
        *(f"- {treatment_id}" for treatment_id in trajectories.treatment_ids),
        "",
        f"Module IDs ({n_modules}):",
        *(f"- {module_id}" for module_id in trajectories.module_ids),
    ]
    return lines


def format_learned_data_section(trajectories: ModuleTrajectoryMatrix) -> list[str]:
    """Format learned-model training, validation, and test data details.

    Args:
        trajectories: Prepared module trajectories.

    Returns:
        Complete dataset identities and temporal split lines.
    """

    n_treatments = len(trajectories.treatment_ids)
    n_modules = len(trajectories.module_ids)
    scope = trajectories.scope
    return [
        "Dataset configuration:",
        f"- scope: {scope.scope_tag} ({scope.split_id}/{scope.config_id})",
        f"- n_treatments: {n_treatments}",
        f"- n_modules: {n_modules}",
        "- administration_route: Gavage",
        "- trajectory_days: 1D, 4D, 8D, 15D, 29D",
        f"- expected_replicates_per_treatment_day: {EXPECTED_REPLICATES_PER_TREATMENT_DAY}",
        "- module_unit: train-standardized fixed WGCNA module PC1 score",
        f"- total_treatment_day_rows: {n_treatments * 5}",
        f"- total_module_values: {n_treatments * 5 * n_modules}",
        f"- total_underlying_biological_replicates: {n_treatments * 5 * EXPECTED_REPLICATES_PER_TREATMENT_DAY}",
        "- split_axis: timepoint",
        "- same_treatments_across_train_validation_test: true",
        "- task_scope: known-treatment temporal extrapolation",
        "",
        "Split details:",
        *_split_size_lines(
            "Training split",
            (1, 4, 8),
            trajectories,
            "observed 1D initialization; 4D and 8D rollout module-MSE targets",
        ),
        f"- loss_target_treatment_day_rows: {n_treatments * 2}",
        f"- loss_target_module_values: {n_treatments * 2 * n_modules}",
        *_split_size_lines(
            "Validation split",
            (15,),
            trajectories,
            "module MSE, scheduler, early stopping, and checkpoint selection",
        ),
        *_split_size_lines(
            "Test split",
            (29,),
            trajectories,
            "held-out evaluation after reloading the best 15D checkpoint",
        ),
        "- module_scaler_fit_days: 1D, 4D, 8D",
        "- rollout_initial_state: observed 1D module state only",
        "- learned_layer_input: standardized module state",
        "- teacher_forcing: false",
        "- observed_4D_8D_15D_module_states_are_never_injected_into_rollout: true",
        "",
        f"Treatment IDs ({n_treatments}):",
        *(f"- {treatment_id}" for treatment_id in trajectories.treatment_ids),
        "",
        f"Module IDs ({n_modules}):",
        *(f"- {module_id}" for module_id in trajectories.module_ids),
    ]


def format_metric_section(title: str, metrics: pd.DataFrame) -> list[str]:
    """Format a metric table with the same row layout for every model.

    Args:
        title: Section title.
        metrics: Aggregate metric table containing model, split, day, and scale.

    Returns:
        Human-readable metric lines.
    """

    lines = [title]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"- {row.model} | {row.split} | {int(row.day)}D | {row.scale} | "
            f"MSE={row.mse:.8f} | RMSE={row.rmse:.8f} | "
            f"MAE={row.mae:.8f} | Pearson={row.pearson:.6f}"
        )
    return lines


def format_repeat_metric_section(title: str, metrics: pd.DataFrame) -> list[str]:
    """Format repeat mean and standard-deviation metrics.

    Args:
        title: Section title.
        metrics: Aggregate table with ``*_mean`` and ``*_std`` columns.

    Returns:
        Human-readable repeat metric lines.
    """

    lines = [title]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"- {row.model} | {row.split} | {int(row.day)}D | {row.scale} | "
            f"MSE={row.mse_mean:.8f} +/- {row.mse_std:.8f} | "
            f"RMSE={row.rmse_mean:.8f} +/- {row.rmse_std:.8f} | "
            f"MAE={row.mae_mean:.8f} +/- {row.mae_std:.8f} | "
            f"Pearson={row.pearson_mean:.6f} +/- {row.pearson_std:.6f}"
        )
    return lines


def format_learned_repeat_summary(
    parameters: Mapping[str, object],
    trajectories: ModuleTrajectoryMatrix,
    metrics: pd.DataFrame,
    best_epoch: int,
    best_loss: float,
) -> list[str]:
    """Build a summary for one learned repeat.

    Args:
        parameters: Complete repeat configuration.
        trajectories: Prepared module trajectories.
        metrics: Aggregate module metric table.
        best_epoch: Epoch selected by 15D validation module MSE.
        best_loss: Best standardized 15D module MSE.

    Returns:
        Complete summary lines.
    """

    return [
        "Module Dynamics Basic-Rollout Summary",
        "",
        *format_parameter_section(parameters),
        "",
        *format_learned_data_section(trajectories),
        "",
        "Training outcome:",
        f"Best epoch: {best_epoch}",
        f"Best 15D standardized module MSE: {best_loss:.8f}",
        "",
        *format_metric_section("Module-level evaluation metrics:", metrics),
    ]


def format_learned_session_summary(
    parameters: Mapping[str, object],
    trajectories: ModuleTrajectoryMatrix,
    repeat_metrics: pd.DataFrame,
) -> list[str]:
    """Build a summary for a learned repeat session.

    Args:
        parameters: Complete session configuration including repeat seeds.
        trajectories: Prepared module trajectories.
        repeat_metrics: Module metrics aggregated across repeats.

    Returns:
        Complete session summary lines.
    """

    return [
        "Module Dynamics Basic-Rollout Repeat Summary",
        "",
        *format_parameter_section(parameters),
        "",
        *format_learned_data_section(trajectories),
        "",
        *format_repeat_metric_section(
            "Module-level evaluation metrics across repeats "
            "(mean +/- SD; SD=0 for one repeat):",
            repeat_metrics,
        ),
    ]


def format_artifact_section() -> list[str]:
    """Describe the stable baseline artifact contract.

    Returns:
        Output artifact summary lines.
    """

    return [
        "Output artifacts:",
        "- predictions.npz: model x predicted_day x treatment x module arrays; no pickle",
        "- metrics_summary.tsv: model x split x day x scale aggregate module metrics",
        "- treatment_metrics.tsv: model x split x day x treatment raw module metrics",
        "- module_metrics.tsv: model x split x day x module raw metrics",
        "- config.json: machine-readable run contract",
        "- summary.txt: human-readable run contract and aggregate metrics",
    ]
