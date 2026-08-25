"""Recreate the four main PKT-ODE figures from portable repository inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TrajectoryData, load_module_parameters, load_trajectories
from .evaluation import calculate_metrics
from .model import ModuleParameters, simulate_trajectories


PKT_COLOR = "#c0392b"
NEUTRAL_COLOR = "#8a94a6"
DOSE_COLORS = ("#9ecae1", "#3182bd", "#08306b")


def _save_figure(figure: plt.Figure, output_stem: Path) -> None:
    """Save one figure as 300-dpi PNG, PDF, and SVG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")


def make_benchmark_figure(benchmark_path: Path, output_dir: Path) -> None:
    """Create the model overview and day-29 benchmark figure."""

    benchmark = pd.read_csv(benchmark_path, sep="\t").sort_values("mse_29d", ascending=False)
    colors = [PKT_COLOR if value == "PKT-ODE" else NEUTRAL_COLOR for value in benchmark["model"]]
    y_positions = np.arange(len(benchmark))
    figure = plt.figure(figsize=(8.0, 5.6))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.1, 1.0), hspace=0.55, wspace=0.55)
    overview = figure.add_subplot(grid[0, :])
    overview.axis("off")
    overview.text(
        0.5,
        0.82,
        "Dose D(t)  ->  plasma concentration Cp(t)  ->  Hill effect H(t)  ->  module state z_i(t)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    overview.text(
        0.5,
        0.35,
        r"$dz_i/dt = k_i[\beta_{0,i} + \beta_{1,i}H(t) - z_i(t)]$",
        ha="center",
        va="center",
        fontsize=15,
    )
    mse_axis = figure.add_subplot(grid[1, 0])
    mse_axis.barh(y_positions, benchmark["mse_29d"], color=colors)
    mse_axis.set_yticks(y_positions, benchmark["model"])
    mse_axis.set_xscale("log")
    mse_axis.set_xlabel("Day-29 MSE (lower is better)")
    correlation_axis = figure.add_subplot(grid[1, 1])
    correlation_axis.barh(y_positions, benchmark["pearson_r_29d"], color=colors)
    correlation_axis.set_yticks(y_positions, benchmark["model"])
    correlation_axis.set_xlim(0.5, 1.0)
    correlation_axis.set_xlabel("Day-29 Pearson r (higher is better)")
    _save_figure(figure, output_dir / "Fig1_overview_benchmark")
    plt.close(figure)


def make_trajectory_figure(
    data: TrajectoryData,
    parameters: Mapping[str, ModuleParameters],
    output_dir: Path,
) -> None:
    """Create fitted trajectories for seven modules and three doses."""

    dense_times = np.linspace(0.125, 29.0, 400)
    dense_regimens = tuple("Single" if time <= 1.0 else "Repeat" for time in dense_times)
    dense_prediction = simulate_trajectories(
        data.doses, dense_times, data.module_ids, parameters, regimens=dense_regimens
    )
    figure, axes = plt.subplots(2, 4, figsize=(8.0, 4.6))
    for module_index, module_id in enumerate(data.module_ids):
        axis = axes.ravel()[module_index]
        axis.axvspan(8.0, 30.0, color="#f1f1f1")
        for dose_index, dose in enumerate(data.doses):
            color = DOSE_COLORS[dose_index]
            axis.plot(dense_times, dense_prediction[dose_index, :, module_index], color=color)
            axis.scatter(
                np.repeat(data.times, data.replicate_values.shape[2]),
                data.replicate_values[dose_index, :, :, module_index].reshape(-1),
                color=color,
                s=13,
                edgecolor="white",
                linewidth=0.4,
            )
        module_parameter = parameters[module_id]
        axis.set_title(
            f"{module_id}: k={module_parameter.k_per_day:.1f}/d, "
            f"beta1={module_parameter.beta_1:+.2f}",
            fontsize=8,
        )
        axis.set_xscale("log")
        axis.set_xlim(0.1, 32.0)
        axis.set_xticks((0.125, 1.0, 4.0, 8.0, 15.0, 29.0))
        axis.set_xticklabels(("3h", "1d", "4d", "8d", "15d", "29d"), fontsize=6)
        axis.set_ylabel("Directed PC1")
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    for dose, color in zip(data.doses, DOSE_COLORS):
        legend_axis.plot([], [], color=color, label=f"{dose:g} mg/kg/day")
    legend_axis.legend(loc="center")
    figure.tight_layout()
    _save_figure(figure, output_dir / "Fig2_fitted_trajectories")
    plt.close(figure)


def make_scatter_figure(
    data: TrajectoryData,
    parameters: Mapping[str, ModuleParameters],
    output_dir: Path,
) -> None:
    """Create day-29 predicted-versus-observed scatter plot."""

    day29_index = int(np.flatnonzero(np.isclose(data.times, 29.0))[0])
    prediction = simulate_trajectories(
        data.doses, data.times, data.module_ids, parameters, regimens=data.regimens
    )[:, day29_index, :]
    observed = data.replicate_values[:, day29_index, :, :]
    repeated_prediction = np.repeat(prediction[:, None, :], observed.shape[1], axis=1)
    metric = calculate_metrics(observed, repeated_prediction)
    figure, axis = plt.subplots(figsize=(4.2, 4.0))
    module_colors = plt.get_cmap("tab10")(np.arange(len(data.module_ids)))
    for module_index, module_id in enumerate(data.module_ids):
        axis.scatter(
            observed[:, :, module_index].reshape(-1),
            repeated_prediction[:, :, module_index].reshape(-1),
            label=module_id,
            color=module_colors[module_index],
            s=28,
        )
    limits = (
        min(float(np.min(observed)), float(np.min(repeated_prediction))) - 0.2,
        max(float(np.max(observed)), float(np.max(repeated_prediction))) + 0.2,
    )
    axis.plot(limits, limits, linestyle="--", color="#666666")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Observed day-29 directed PC1")
    axis.set_ylabel("Predicted day-29 directed PC1")
    axis.set_title(f"PKT-ODE: Pearson r={metric.pearson_r:.3f}")
    axis.legend(ncol=2, fontsize=7)
    figure.tight_layout()
    _save_figure(figure, output_dir / "Fig3_day29_scatter")
    plt.close(figure)


def make_biology_figure(hit_path: Path, output_dir: Path) -> None:
    """Create the descriptive PPAR-alpha target distribution figure."""

    hits = pd.read_csv(hit_path, sep="\t")
    counts = hits.groupby("module_id").size().sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(6.0, 3.5))
    axis.barh(counts.index, counts.values, color="#2c6fb3")
    for index, count in enumerate(counts.values):
        axis.text(float(count) + 0.2, index, str(int(count)), va="center")
    axis.set_xlabel("Literature-curated PPAR-alpha target genes in module")
    axis.set_title("Targets occur in modules with positive fitted drug effects")
    figure.tight_layout()
    _save_figure(figure, output_dir / "Fig4_module_biology")
    plt.close(figure)


def generate_all_figures(repository_root: Path, output_dir: Path) -> None:
    """Generate all main figures from repository-relative inputs.

    Args:
        repository_root: Root directory of the public snapshot.
        output_dir: Destination for regenerated figures.
    """

    data = load_trajectories(
        repository_root
        / "data/processed/pkt_ode_input.npz"
    )
    parameters = load_module_parameters(repository_root / "data/processed/published_parameters.tsv")
    make_benchmark_figure(repository_root / "data/processed/benchmark_metrics.tsv", output_dir)
    make_trajectory_figure(data, parameters, output_dir)
    make_scatter_figure(data, parameters, output_dir)
    make_biology_figure(repository_root / "data/reference/ppara_module_hits.tsv", output_dir)
