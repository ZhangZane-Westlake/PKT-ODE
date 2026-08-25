"""Command-line interface for basic module-eigengene dynamics."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..common import LRD3H_TEST_HOUR
from .baselines import run_baselines
from .data import (
    PROJECT_ROOT,
    ModuleTrajectoryMatrix,
    load_module_trajectory_matrix,
    write_module_cache,
)
from .reporting import (
    EXPECTED_REPLICATES_PER_TREATMENT_DAY,
    format_learned_session_summary,
)
from .training import TrainingConfig, train_module_model, train_module_model_lrd_3h


REPEAT_SEED_STEP = 1009


def parse_positive_int(value: str) -> int:
    """Parse a strictly positive integer.

    Args:
        value: CLI token.

    Returns:
        Positive integer.

    Raises:
        argparse.ArgumentTypeError: If invalid.
    """

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def derive_repeat_seed(base_seed: int, repeat_index: int) -> int:
    """Derive a deterministic one-based repeat seed.

    Args:
        base_seed: First repeat seed.
        repeat_index: One-based repeat index.

    Returns:
        Derived seed.
    """

    if repeat_index <= 0:
        raise ValueError("repeat_index must be positive")
    return base_seed + (repeat_index - 1) * REPEAT_SEED_STEP


def parse_train_day_weights(value: str) -> tuple[float, float]:
    """Parse ``4D:weight,8D:weight``.

    Args:
        value: CLI weight specification.

    Returns:
        4D and 8D weights.

    Raises:
        argparse.ArgumentTypeError: If malformed.
    """

    weights: dict[str, float] = {}
    try:
        for token in value.split(","):
            day, raw_weight = token.split(":", maxsplit=1)
            weights[day.strip().upper()] = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected 4D:W,8D:W") from exc
    if set(weights) != {"4D", "8D"}:
        raise argparse.ArgumentTypeError("weights must contain exactly 4D and 8D")
    if min(weights.values()) < 0 or sum(weights.values()) <= 0:
        raise argparse.ArgumentTypeError("weights must be non-negative with positive sum")
    return weights["4D"], weights["8D"]


def parse_fit_days(value: str) -> tuple[int, ...]:
    """Parse a comma-separated fit-day specification such as ``"1,4,8,15"``.

    Accepts either bare day numbers (``1,4,8``) or ``D``-suffixed tokens
    (``1D,4D,8D``). Order is preserved; duplicates are rejected.

    Args:
        value: CLI fit-day specification.

    Returns:
        Tuple of fit-day integers in the supplied order.

    Raises:
        argparse.ArgumentTypeError: If malformed or empty.
    """

    token_to_day = {"1": 1, "4": 4, "8": 8, "15": 15, "29": 29}
    days: list[int] = []
    for token in value.split(","):
        key = token.strip().upper().rstrip("D")
        if key not in token_to_day:
            raise argparse.ArgumentTypeError(
                f"invalid fit day '{token.strip()}'; expected one of 1,4,8,15,29"
            )
        day = token_to_day[key]
        if day in days:
            raise argparse.ArgumentTypeError(f"duplicate fit day {day}")
        days.append(day)
    if not days:
        raise argparse.ArgumentTypeError("fit-days must list at least one day")
    return tuple(days)


def _run_name(requested: str) -> str:
    """Validate or generate a single-directory run name.

    Args:
        requested: User-provided name or empty string.

    Returns:
        Safe run directory name.
    """

    if not requested:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    if "/" in requested or "\\" in requested or not re.fullmatch(r"[A-Za-z0-9_.-]+", requested):
        raise ValueError("run-name must be one safe directory component")
    return requested


def _default_cache_dir(scope_tag: str, config_id: str) -> Path:
    """Return the default prepared-cache directory for a reduction scope.

    Args:
        scope_tag: Scope label (organ or ``compounds/<slug>``).
        config_id: Reduction config directory name.

    Returns:
        Default cache path.
    """

    return (
        PROJECT_ROOT
        / "data"
        / "expression"
        / "module_dynamics"
        / "basic_rollout"
        / scope_tag
        / config_id
    )


def _configure_logger(directory: Path, filename: str, name: str) -> logging.Logger:
    """Create a console and file logger.

    Args:
        directory: Log directory.
        filename: Log filename.
        name: Unique logger name.

    Returns:
        Configured logger.
    """

    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(directory / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers.

    Args:
        logger: Logger to close.
    """

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _session_parameters(
    config: TrainingConfig,
    repeat_seeds: Sequence[int],
    trajectories: ModuleTrajectoryMatrix,
) -> dict[str, object]:
    """Build the complete learned-session reporting contract.

    Args:
        config: Base training configuration.
        repeat_seeds: Ordered repeat seeds.
        trajectories: Prepared dataset shared by all repeats.

    Returns:
        Machine-readable session parameters.
    """

    scope = trajectories.scope
    return {
        **asdict(config),
        "schema_version": 2,
        "pipeline": "module_dynamics/basic_rollout",
        "run_type": "learned_session",
        "administration_route": "Gavage",
        "organ": scope.organ,
        "scope_tag": scope.scope_tag,
        "is_compound_scope": scope.is_compound_scope,
        "reduction_run": scope.run_name,
        "split_id": scope.split_id,
        "reduction_config_id": scope.config_id,
        "repeat": len(repeat_seeds),
        "base_seed": config.seed,
        "repeat_seed_step": REPEAT_SEED_STEP,
        "repeat_seed_rule": "base_seed + (repeat_index - 1) * repeat_seed_step",
        "repeat_seeds": list(repeat_seeds),
        "input_level": "standardized module eigengene state",
        "target_level": "module eigengene residual",
        "n_treatments": len(trajectories.treatment_ids),
        "n_modules": len(trajectories.module_ids),
        "expected_replicates_per_treatment_day": EXPECTED_REPLICATES_PER_TREATMENT_DAY,
        "split_contract": {
            "input": "observed 1D module state only",
            "train_loss_days": ["4D", "8D"],
            "validation_day": "15D",
            "test_day": "29D",
            "split_axis": "timepoint",
            "same_treatments_across_splits": True,
            "teacher_forcing": False,
            "loss_mode": config.loss_mode,
            "lambda_box": config.lambda_box,
        },
    }


def _write_session_outputs(
    session_dir: Path,
    config: TrainingConfig,
    trajectories: ModuleTrajectoryMatrix,
    histories: list[pd.DataFrame],
    metrics: list[pd.DataFrame],
    manifest_rows: list[dict[str, object]],
) -> None:
    """Write repeat manifests and aggregate mean/SD metrics.

    Args:
        session_dir: Training session directory.
        config: Base configuration.
        trajectories: Prepared dataset.
        histories: Repeat histories.
        metrics: Repeat metrics.
        manifest_rows: Repeat metadata rows.
    """

    combined_history = pd.concat(histories, ignore_index=True)
    combined_metrics = pd.concat(metrics, ignore_index=True)
    combined_history.to_csv(session_dir / "training_history_all_repeats.csv", index=False)
    combined_metrics.to_csv(session_dir / "metrics_all_repeats.tsv", sep="\t", index=False)
    pd.DataFrame(manifest_rows).to_csv(session_dir / "repeat_manifest.tsv", sep="\t", index=False)
    group_columns = ["model", "split", "day", "scale"]
    metric_columns = ["mse", "rmse", "mae", "pearson"]
    aggregate = combined_metrics.groupby(group_columns, observed=True)[metric_columns].agg(
        ["mean", "std"]
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index().fillna(0.0)
    aggregate.to_csv(session_dir / "metrics_repeat_mean_sd.tsv", sep="\t", index=False)
    repeat_seeds = [int(row["seed"]) for row in manifest_rows]
    parameters = _session_parameters(config, repeat_seeds, trajectories)
    lines = format_learned_session_summary(parameters, trajectories, aggregate)
    (session_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="load and cache five-day module trajectories")
    prepare.add_argument("--reduction-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--dry-run", action="store_true")

    baseline = subparsers.add_parser("baseline", help="run four univariate module baselines")
    baseline.add_argument("--reduction-dir", type=Path, required=True)
    baseline.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "module_dynamics" / "basic_rollout",
    )
    baseline.add_argument("--run-name", default="")
    baseline.add_argument(
        "--fit-days",
        type=parse_fit_days,
        default=(1, 4, 8),
        metavar="1,4,8[,15[,29]]",
        help=(
            "comma-separated days the baselines consume (default 1,4,8). "
            "Every other trajectory day is a prediction target; e.g. "
            "--fit-days 1,4,8,15 folds 15D into the fit so only 29D is held out"
        ),
    )

    train = subparsers.add_parser("train", help="train a residual module transition")
    train.add_argument("--reduction-dir", type=Path, required=True)
    train.add_argument("--model", choices=("linear", "mlp"), required=True)
    train.add_argument("--dynamics", choices=("observed", "lrd", "lrd_3h"), required=True)
    train.add_argument(
        "--conditioning",
        choices=("none", "concat", "film", "residual_adapter"),
        default="none",
    )
    train.add_argument("--loss-mode", choices=("mean", "replicate"), default="mean")
    train.add_argument("--lambda-box", type=float, default=1.0)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--repeat", type=parse_positive_int, default=1)
    train.add_argument("--max-epochs", type=parse_positive_int, default=500)
    train.add_argument("--patience", type=parse_positive_int, default=50)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.add_argument("--batch-size", type=parse_positive_int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden-dim", type=parse_positive_int, default=128)
    train.add_argument("--num-layers", type=parse_positive_int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--adapter-dim", type=parse_positive_int, default=32)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument(
        "--train-day-weights",
        type=parse_train_day_weights,
        default=(1.0, 1.0),
        metavar="4D:W,8D:W",
    )
    train.add_argument("--scheduler", choices=("none", "plateau"), default="plateau")
    train.add_argument("--scheduler-patience", type=int, default=10)
    train.add_argument("--scheduler-factor", type=float, default=0.5)
    train.add_argument("--min-learning-rate", type=float, default=1e-6)
    train.add_argument("--log-every", type=parse_positive_int, default=10)
    train.add_argument("--non-deterministic", action="store_true")
    train.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "module_dynamics" / "basic_rollout",
    )
    train.add_argument("--run-name", default="")
    return parser


def _run_prepare(args: argparse.Namespace) -> int:
    """Execute the preparation command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit status.
    """

    matrix = load_module_trajectory_matrix(args.reduction_dir)
    output_dir = args.output_dir or _default_cache_dir(
        matrix.scope.scope_tag, matrix.scope.config_id
    )
    print(
        f"scope={matrix.scope.scope_tag} treatments={len(matrix.treatment_ids)} "
        f"modules={len(matrix.module_ids)}"
    )
    if args.dry_run:
        print("dry-run: no files written")
        return 0
    write_module_cache(matrix, args.reduction_dir, output_dir)
    print(f"wrote prepared module cache to {output_dir}")
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    """Execute the baseline command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit status.
    """

    matrix = load_module_trajectory_matrix(args.reduction_dir)
    run_dir = (
        args.output_root
        / matrix.scope.scope_tag
        / matrix.scope.split_id
        / matrix.scope.config_id
        / "baselines"
        / _run_name(args.run_name)
    )
    metrics = run_baselines(matrix, run_dir, fit_days=args.fit_days)
    print(metrics[metrics["scale"].eq("raw")].to_string(index=False))
    print(f"wrote baseline outputs to {run_dir}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Execute a serial multi-repeat training session.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit status.
    """

    matrix = load_module_trajectory_matrix(
        args.reduction_dir, eight_timepoint=(args.dynamics == "lrd_3h")
    )
    session_dir = (
        args.output_root
        / matrix.scope.scope_tag
        / matrix.scope.split_id
        / matrix.scope.config_id
        / args.model
        / args.dynamics
        / args.conditioning
        / args.loss_mode
        / _run_name(args.run_name)
    )
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {session_dir}")
    weight_4d, weight_8d = args.train_day_weights
    config = TrainingConfig(
        organ=matrix.scope.organ,
        scope_tag=matrix.scope.scope_tag,
        model=args.model,
        dynamics=args.dynamics,
        conditioning=args.conditioning,
        loss_mode=args.loss_mode,
        lambda_box=args.lambda_box,
        device=args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        adapter_dim=args.adapter_dim,
        grad_clip=args.grad_clip,
        train_weight_4d=weight_4d,
        train_weight_8d=weight_8d,
        scheduler=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_learning_rate=args.min_learning_rate,
        log_every=args.log_every,
        deterministic=not args.non_deterministic,
    )
    config.validate()
    test_key = LRD3H_TEST_HOUR if config.dynamics == "lrd_3h" else 29
    repeat_seeds = [derive_repeat_seed(args.seed, index) for index in range(1, args.repeat + 1)]
    session_dir.mkdir(parents=True, exist_ok=True)
    session_parameters = _session_parameters(config, repeat_seeds, matrix)
    (session_dir / "config.json").write_text(
        json.dumps(session_parameters, indent=2) + "\n",
        encoding="utf-8",
    )
    logger = _configure_logger(session_dir, "session.log", f"module_dynamics.basic.session.{session_dir}")
    histories: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    try:
        for repeat_index, repeat_seed in enumerate(repeat_seeds, start=1):
            repeat_name = f"repeat_{repeat_index:02d}_seed_{repeat_seed}"
            repeat_dir = session_dir / repeat_name
            repeat_logger = _configure_logger(
                repeat_dir,
                "run.log",
                f"module_dynamics.basic.repeat.{session_dir}.{repeat_index}",
            )
            try:
                train_fn = (
                    train_module_model_lrd_3h
                    if config.dynamics == "lrd_3h"
                    else train_module_model
                )
                artifacts = train_fn(
                    matrix,
                    replace(config, seed=repeat_seed),
                    repeat_dir,
                    repeat_logger,
                )
            finally:
                _close_logger(repeat_logger)
            history = artifacts.history.copy()
            history.insert(0, "seed", repeat_seed)
            history.insert(0, "repeat_index", repeat_index)
            repeat_metrics = artifacts.metrics.copy()
            repeat_metrics.insert(0, "seed", repeat_seed)
            repeat_metrics.insert(0, "repeat_index", repeat_index)
            test29_std = repeat_metrics[
                repeat_metrics["day"].eq(test_key) & repeat_metrics["scale"].eq("standardized")
            ].iloc[0]
            test29_raw = repeat_metrics[
                repeat_metrics["day"].eq(test_key) & repeat_metrics["scale"].eq("raw")
            ].iloc[0]
            histories.append(history)
            metrics.append(repeat_metrics)
            manifest_rows.append(
                {
                    "repeat_index": repeat_index,
                    "seed": repeat_seed,
                    "directory": repeat_name,
                    "best_epoch": artifacts.best_epoch,
                    "status": "complete",
                }
            )
            logger.info(
                "Repeat %d/%d complete: seed=%d best_epoch=%d "
                "test29_loss_standardized_mse=%.6f test29_raw_mse=%.6f",
                repeat_index,
                args.repeat,
                repeat_seed,
                artifacts.best_epoch,
                test29_std["mse"],
                test29_raw["mse"],
            )
        _write_session_outputs(
            session_dir, config, matrix, histories, metrics, manifest_rows
        )
        all_metrics = pd.concat(metrics, ignore_index=True)
        test_losses = all_metrics.loc[
            all_metrics["day"].eq(test_key) & all_metrics["scale"].eq("standardized"),
            "mse",
        ]
        test_std = float(test_losses.std(ddof=1)) if len(test_losses) > 1 else 0.0
        logger.info(
            "All %d repeat(s) completed: test29_loss_standardized_mse_mean=%.6f std=%.6f",
            args.repeat,
            float(test_losses.mean()),
            test_std,
        )
    except Exception:
        logger.exception("Training session failed")
        raise
    finally:
        _close_logger(logger)
    print(f"wrote training outputs to {session_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the basic module dynamics CLI.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Exit status.
    """

    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return _run_prepare(args)
    if args.command == "baseline":
        return _run_baseline(args)
    if args.command == "train":
        return _run_train(args)
    raise ValueError(f"Unsupported command: {args.command}")
