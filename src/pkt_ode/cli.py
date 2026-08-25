"""Command-line interface for verification, refitting, and figure generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import load_module_parameters, load_trajectories
from .evaluation import evaluate_splits, evaluate_statistical_baselines
from .model import ModuleParameters, fit_pkt_ode, simulate_trajectories
from .prepare_data import build_pkt_ode_input


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTION_TRAJECTORIES = Path(
    "data/processed/fenofibrate_reduction/projection/treatment_module_trajectories.npz"
)
DEFAULT_BUNDLE = Path("data/processed/fenofibrate_reduction/projection/module_bundle.npz")
DEFAULT_METADATA = Path(
    "data/processed/fenofibrate_reduction/projection/sample_metadata.tsv.gz"
)
DEFAULT_INPUT = Path("data/processed/pkt_ode_input.npz")
DEFAULT_PARAMETERS = Path("data/processed/published_parameters.tsv")


def _resolve(path: Path) -> Path:
    """Resolve a CLI path against the repository root when relative."""

    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _parameter_frame(parameters: dict[str, ModuleParameters]) -> pd.DataFrame:
    """Convert a parameter mapping into the public TSV schema."""

    return pd.DataFrame(
        [
            {
                "module_id": module_id,
                "k_per_day": value.k_per_day,
                "beta_0": value.beta_0,
                "beta_1": value.beta_1,
            }
            for module_id, value in parameters.items()
        ]
    )


def _write_table(table: pd.DataFrame, output: Path | None) -> None:
    """Print a table or write it to a requested relative path."""

    if output is None:
        print(table.to_csv(sep="\t", index=False), end="")
        return
    destination = _resolve(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the PKT-ODE command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-input", help="build aligned MAD-normalized input")
    prepare.add_argument("--trajectories", type=Path, default=DEFAULT_PROJECTION_TRAJECTORIES)
    prepare.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    prepare.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    prepare.add_argument("--output", type=Path, default=DEFAULT_INPUT)

    verify = subparsers.add_parser("verify", help="re-simulate published parameters")
    verify.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    verify.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    verify.add_argument("--output", type=Path)

    baselines = subparsers.add_parser("baselines", help="evaluate four statistical baselines")
    baselines.add_argument("--trajectories", type=Path, default=DEFAULT_PROJECTION_TRAJECTORIES)
    baselines.add_argument("--output", type=Path)

    fit = subparsers.add_parser("fit", help="refit all seven modules through day 8")
    fit.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    fit.add_argument("--parameters-output", type=Path, required=True)
    fit.add_argument("--metrics-output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=42)

    figures = subparsers.add_parser("figures", help="regenerate all four main figures")
    figures.add_argument("--output-dir", type=Path, default=Path("results/reproduced_figures"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one PKT-ODE command.

    Args:
        argv: Optional command-line tokens for tests or embedding.

    Returns:
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    if args.command == "prepare-input":
        build_pkt_ode_input(
            _resolve(args.trajectories),
            _resolve(args.bundle),
            _resolve(args.metadata),
            _resolve(args.output),
        )
        return 0
    if args.command == "figures":
        from .figures import generate_all_figures

        generate_all_figures(REPOSITORY_ROOT, _resolve(args.output_dir))
        return 0

    if args.command == "baselines":
        with np.load(_resolve(args.trajectories), allow_pickle=False) as archive:
            mean_values = np.asarray(archive["mean_values"], dtype=float)
            labels = tuple(str(value) for value in archive["model_times"])
        from .data import TIME_IN_DAYS

        times = np.asarray([TIME_IN_DAYS[label] for label in labels], dtype=float)
        _write_table(evaluate_statistical_baselines(times, mean_values), args.output)
        return 0

    data = load_trajectories(_resolve(args.input))

    if args.command == "verify":
        parameters = load_module_parameters(_resolve(args.parameters))
        predictions = simulate_trajectories(
            data.doses, data.times, data.module_ids, parameters, regimens=data.regimens
        )
        _write_table(evaluate_splits(data.times, data.replicate_values, predictions), args.output)
        return 0

    fitted = fit_pkt_ode(
        data.doses,
        data.times,
        data.replicate_values,
        data.module_ids,
        data.regimens,
        random_seed=args.seed,
    )
    parameter_output = _resolve(args.parameters_output)
    parameter_output.parent.mkdir(parents=True, exist_ok=True)
    _parameter_frame(fitted).to_csv(parameter_output, sep="\t", index=False)
    predictions = simulate_trajectories(
        data.doses, data.times, data.module_ids, fitted, regimens=data.regimens
    )
    metrics_output = _resolve(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    evaluate_splits(data.times, data.replicate_values, predictions).to_csv(
        metrics_output, sep="\t", index=False
    )
    return 0
