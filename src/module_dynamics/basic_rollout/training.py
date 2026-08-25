"""Deterministic training for residual module rollouts."""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import (
    HOUR_TO_INDEX,
    LRD3H_INIT_HOUR,
    LRD3H_STEP_HOURS,
    LRD3H_TEST_HOUR,
    LRD3H_TRAIN_HOURS,
    LRD3H_VAL_HOUR,
    SPLIT_BY_HOUR,
)
from ..losses import LossMode, module_target_loss, training_fit_matrix
from .data import ModuleTrajectoryMatrix
from .evaluation import build_module_metric_tables, write_prediction_archive
from .models import (
    ConditioningMode,
    DynamicsMode,
    ModelName,
    ModuleTransition,
    build_transition_model,
    rollout,
    rollout_lrd_3h,
)
from .reporting import (
    EXPECTED_REPLICATES_PER_TREATMENT_DAY,
    format_learned_repeat_summary,
)
from .scaling import FeatureScaler


DAY_TO_INDEX: dict[int, int] = {1: 0, 4: 1, 8: 2, 15: 3, 29: 4}
SPLIT_BY_DAY: dict[int, str] = {
    4: "train",
    8: "train",
    15: "validation",
    29: "test",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Public configuration for one module-model repeat.

    Args:
        organ: Reduction-scope organ.
        scope_tag: Filesystem-safe scope label.
        model: Linear or MLP transition family.
        dynamics: Observed-step or daily LRD rollout.
        conditioning: 1D-anchor conditioning design.
        loss_mode: Target loss mode.
        lambda_box: Replicate box-barrier strength.
        device: Torch device string.
        seed: Base repeat seed.
        max_epochs: Maximum training epochs.
        patience: Early-stopping patience.
        min_delta: Minimum validation improvement.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: AdamW weight decay.
        hidden_dim: MLP hidden width.
        num_layers: MLP hidden-layer count.
        dropout: MLP dropout probability.
        adapter_dim: Drug-adapter bottleneck width.
        grad_clip: Gradient-norm clip.
        train_weight_4d: 4D loss weight.
        train_weight_8d: 8D loss weight.
        scheduler: Learning-rate scheduler.
        scheduler_patience: Plateau scheduler patience.
        scheduler_factor: Plateau scheduler factor.
        min_learning_rate: Minimum learning rate.
        log_every: Epoch logging cadence.
        deterministic: Whether to request deterministic Torch algorithms.
    """

    organ: str
    scope_tag: str
    model: ModelName
    dynamics: DynamicsMode
    conditioning: ConditioningMode
    loss_mode: LossMode = "mean"
    lambda_box: float = 1.0
    device: str = "cuda:0"
    seed: int = 42
    max_epochs: int = 500
    patience: int = 50
    min_delta: float = 1e-5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    adapter_dim: int = 32
    grad_clip: float = 1.0
    train_weight_4d: float = 1.0
    train_weight_8d: float = 1.0
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    log_every: int = 10
    deterministic: bool = True

    def validate(self) -> None:
        """Validate model and optimization settings.

        Raises:
            ValueError: If a setting is invalid.
        """

        if self.model not in {"linear", "mlp"}:
            raise ValueError("model must be linear or mlp")
        if self.dynamics not in {"observed", "lrd", "lrd_3h"}:
            raise ValueError("dynamics must be observed, lrd, or lrd_3h")
        if self.conditioning not in {"none", "concat", "film", "residual_adapter"}:
            raise ValueError("unsupported conditioning mode")
        if self.loss_mode not in {"mean", "replicate"}:
            raise ValueError("loss_mode must be mean or replicate")
        if self.lambda_box < 0:
            raise ValueError("lambda_box must be non-negative")
        if self.model == "linear" and self.conditioning != "none":
            raise ValueError("The linear model only supports conditioning=none")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("max_epochs, patience, and batch_size must be positive")
        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.adapter_dim <= 0:
            raise ValueError("MLP dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("optimizer parameters are invalid")
        if self.min_delta < 0 or self.train_weight_4d < 0 or self.train_weight_8d < 0:
            raise ValueError("loss settings must be non-negative")
        if self.train_weight_4d + self.train_weight_8d <= 0:
            raise ValueError("at least one training day weight must be positive")
        if self.scheduler not in {"none", "plateau"}:
            raise ValueError("scheduler must be none or plateau")
        if self.scheduler_patience < 0 or not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler settings are invalid")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate is invalid")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")


@dataclass(frozen=True)
class TrainingArtifacts:
    """In-memory artifacts returned to callers and tests."""

    history: pd.DataFrame
    metrics: pd.DataFrame
    best_epoch: int


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure deterministic random state.

    Args:
        seed: Global seed.
        deterministic: Whether to request deterministic Torch algorithms.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def resolve_device(requested: str) -> torch.device:
    """Resolve a device without silent CUDA fallback.

    Args:
        requested: Torch device string.

    Returns:
        Validated device.

    Raises:
        RuntimeError: If requested CUDA is unavailable.
    """

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested} requested, but CUDA is unavailable")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {requested} requested, but only {torch.cuda.device_count()} devices exist"
            )
    return device


def _weighted_training_loss(
    predictions: dict[int, torch.Tensor],
    batch_means: torch.Tensor,
    batch_replicates: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the weighted 4D/8D module loss for the active mode.

    Args:
        predictions: Predicted module states by day.
        batch_means: Standardized replicate-mean targets.
        batch_replicates: Standardized replicate targets.
        config: Training configuration.

    Returns:
        Total, 4D, and 8D losses.
    """

    loss_4d = module_target_loss(
        predictions[4],
        batch_means[:, DAY_TO_INDEX[4]],
        batch_replicates[:, DAY_TO_INDEX[4]],
        config.loss_mode,
        config.lambda_box,
    )
    loss_8d = module_target_loss(
        predictions[8],
        batch_means[:, DAY_TO_INDEX[8]],
        batch_replicates[:, DAY_TO_INDEX[8]],
        config.loss_mode,
        config.lambda_box,
    )
    weight_sum = config.train_weight_4d + config.train_weight_8d
    total = (config.train_weight_4d * loss_4d + config.train_weight_8d * loss_8d) / weight_sum
    return total, loss_4d, loss_8d


def _predict_standardized(
    model: ModuleTransition,
    values: torch.Tensor,
    dynamics: DynamicsMode,
) -> dict[int, np.ndarray]:
    """Run full end-to-end inference from observed 1D.

    Args:
        model: Trained module transition.
        values: Standardized targets; only 1D is read as input.
        dynamics: Rollout mode.

    Returns:
        Predictions for 4D, 8D, 15D, and 29D.
    """

    model.eval()
    with torch.no_grad():
        prediction = rollout(model, values[:, DAY_TO_INDEX[1]], values[:, DAY_TO_INDEX[1]], dynamics, (4, 8, 15, 29))
    return {day: tensor.detach().cpu().numpy() for day, tensor in prediction.items()}


def train_module_model(
    trajectories: ModuleTrajectoryMatrix,
    config: TrainingConfig,
    run_dir: Path,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Train, checkpoint, evaluate, and export one module-model repeat.

    Args:
        trajectories: Prepared module dataset.
        config: Validated training configuration.
        run_dir: Empty repeat output directory.
        logger: Repeat logger.

    Returns:
        Training history, global metrics, and best epoch.
    """

    config.validate()
    existing = [path for path in run_dir.iterdir() if path.name != "run.log"] if run_dir.exists() else []
    if existing:
        raise FileExistsError(f"Run directory already contains outputs: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_reproducibility(config.seed, config.deterministic)
    device = resolve_device(config.device)
    fit_matrix = training_fit_matrix(
        trajectories.values, trajectories.replicate_values, config.loss_mode
    )
    module_scaler = FeatureScaler.from_matrix(fit_matrix)
    standardized_means = module_scaler.transform(trajectories.values).astype(np.float32)
    standardized_replicates = module_scaler.transform(trajectories.replicate_values).astype(np.float32)
    means_tensor = torch.from_numpy(standardized_means)
    replicates_tensor = torch.from_numpy(standardized_replicates)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(means_tensor, replicates_tensor),
        batch_size=min(config.batch_size, len(means_tensor)),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    module_dim = len(trajectories.module_ids)
    model = build_transition_model(
        config.model,
        module_dim,
        config.conditioning,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
        config.adapter_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
        )
    full_means = means_tensor.to(device)
    full_replicates = replicates_tensor.to(device)
    checkpoint_path = run_dir / "best_model.pt"
    history_rows: list[dict[str, float | int | bool]] = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    logger.info(
        "Starting %s/%s/%s loss=%s on %s: treatments=%d modules=%d seed=%d",
        config.model,
        config.dynamics,
        config.conditioning,
        config.loss_mode,
        device,
        len(trajectories.treatment_ids),
        module_dim,
        config.seed,
    )
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_sum = 0.0
        loss_4d_sum = 0.0
        loss_8d_sum = 0.0
        n_seen = 0
        for batch_means_cpu, batch_reps_cpu in loader:
            batch_means = batch_means_cpu.to(device)
            batch_reps = batch_reps_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = rollout(
                model,
                batch_means[:, DAY_TO_INDEX[1]],
                batch_means[:, DAY_TO_INDEX[1]],
                config.dynamics,
                (4, 8),
            )
            total_loss, loss_4d, loss_8d = _weighted_training_loss(
                predictions, batch_means, batch_reps, config
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_size = int(batch_means.shape[0])
            n_seen += batch_size
            total_sum += float(total_loss.detach()) * batch_size
            loss_4d_sum += float(loss_4d.detach()) * batch_size
            loss_8d_sum += float(loss_8d.detach()) * batch_size
        train_loss = total_sum / n_seen
        train_4d = loss_4d_sum / n_seen
        train_8d = loss_8d_sum / n_seen
        model.eval()
        with torch.no_grad():
            validation = rollout(
                model,
                full_means[:, DAY_TO_INDEX[1]],
                full_means[:, DAY_TO_INDEX[1]],
                config.dynamics,
                (4, 8, 15),
            )
            val_loss = float(
                module_target_loss(
                    validation[15],
                    full_means[:, DAY_TO_INDEX[15]],
                    full_replicates[:, DAY_TO_INDEX[15]],
                    config.loss_mode,
                    config.lambda_box,
                ).detach()
            )
        if scheduler is not None:
            scheduler.step(val_loss)
        improved = val_loss < best_loss - config.min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_4d_loss": train_4d,
                "train_8d_loss": train_8d,
                "val_15d_loss": val_loss,
                "learning_rate": current_lr,
                "is_best": improved,
                "patience_remaining": max(0, config.patience - stale_epochs),
            }
        )
        if epoch == 1 or epoch % config.log_every == 0:
            logger.info(
                "epoch=%d train=%.6f 4D=%.6f 8D=%.6f val15=%.6f lr=%.3g best_epoch=%d",
                epoch,
                train_loss,
                train_4d,
                train_8d,
                val_loss,
                current_lr,
                best_epoch,
            )
        if stale_epochs >= config.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break
    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without producing a checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    predictions_std_one = _predict_standardized(model, full_means, config.dynamics)
    predictions_raw_one = {
        day: module_scaler.inverse_transform(values) for day, values in predictions_std_one.items()
    }
    label = f"{config.model}:{config.dynamics}:{config.conditioning}:{config.loss_mode}"
    predictions_raw = {label: predictions_raw_one}
    predictions_std = {label: predictions_std_one}
    metrics, treatment_metrics, module_metrics = build_module_metric_tables(
        predictions_raw,
        predictions_std,
        trajectories.values,
        standardized_means,
        trajectories.treatment_ids,
        trajectories.module_ids,
        DAY_TO_INDEX,
        SPLIT_BY_DAY,
    )
    write_prediction_archive(
        run_dir / "predictions.npz",
        predictions_raw,
        predictions_std,
        trajectories.values,
        standardized_means,
        trajectories.treatment_ids,
        trajectories.module_ids,
        DAY_TO_INDEX,
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(run_dir / "training_history.csv", index=False)
    metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
    treatment_metrics.to_csv(run_dir / "treatment_metrics.tsv", sep="\t", index=False)
    module_metrics.to_csv(run_dir / "module_metrics.tsv", sep="\t", index=False)
    run_config: dict[str, object] = {
        **asdict(config),
        "schema_version": 2,
        "pipeline": "module_dynamics/basic_rollout",
        "run_type": "learned_repeat",
        "administration_route": "Gavage",
        "resolved_device": str(device),
        "input_level": "standardized module eigengene state",
        "target_level": "module eigengene residual",
        "n_treatments": len(trajectories.treatment_ids),
        "n_modules": module_dim,
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
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "scaler.json").write_text(
        json.dumps(module_scaler.to_dict("module_ids", trajectories.module_ids), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = format_learned_repeat_summary(
        run_config,
        trajectories,
        metrics,
        best_epoch,
        best_loss,
    )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    test_raw = metrics[metrics["day"].eq(29) & metrics["scale"].eq("raw")].iloc[0]
    test_std = metrics[metrics["day"].eq(29) & metrics["scale"].eq("standardized")].iloc[0]
    logger.info(
        "Best epoch=%d; test29_loss_standardized_mse=%.6f raw_mse=%.6f raw_pearson=%.6f",
        best_epoch,
        test_std["mse"],
        test_raw["mse"],
        test_raw["pearson"],
    )
    return TrainingArtifacts(history=history, metrics=metrics, best_epoch=best_epoch)


def _lrd3h_fit_matrix(
    values: np.ndarray,
    replicate_values: np.ndarray,
    mode: LossMode,
) -> np.ndarray:
    """Leakage-safe scaler fit matrix for ``lrd_3h``: init+train hours only.

    Args:
        values: Treatment-by-hour-by-module replicate means (8-hour axis).
        replicate_values: Treatment-by-hour-by-replicate-by-module scores.
        mode: Target loss mode.

    Returns:
        Two-dimensional fit matrix with modules on the final axis.
    """

    region_hours = (LRD3H_INIT_HOUR,) + LRD3H_TRAIN_HOURS
    indices = [HOUR_TO_INDEX[hour] for hour in region_hours]
    if mode == "replicate":
        sub = replicate_values[:, indices, :, :]
        return sub.reshape(-1, sub.shape[-1])
    sub = values[:, indices, :]
    return sub.reshape(-1, sub.shape[-1])


def _lrd3h_training_loss(
    predictions: dict[int, torch.Tensor],
    batch_means: torch.Tensor,
    batch_reps: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Equal-weight mean module loss over the ``lrd_3h`` train hours.

    Args:
        predictions: Predicted states keyed by elapsed hour.
        batch_means: Standardized replicate-mean targets.
        batch_reps: Standardized replicate targets.
        config: Training configuration.

    Returns:
        ``(total_loss, per_hour_losses)``.
    """

    per_hour: dict[int, torch.Tensor] = {}
    for hour in LRD3H_TRAIN_HOURS:
        per_hour[hour] = module_target_loss(
            predictions[hour],
            batch_means[:, HOUR_TO_INDEX[hour]],
            batch_reps[:, HOUR_TO_INDEX[hour]],
            config.loss_mode,
            config.lambda_box,
        )
    total = sum(per_hour.values()) / len(per_hour)
    return total, per_hour


def train_module_model_lrd_3h(
    trajectories: ModuleTrajectoryMatrix,
    config: TrainingConfig,
    run_dir: Path,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Train a residual rollout at 3-hour granularity over the 8 timepoints.

    Like :func:`train_module_model` but on the full eight-timepoint hour axis:
    init at 3H, train loss on 6H/9H/1D/4D/8D, validation at 15D (360 h), test
    at 29D (696 h), with :func:`rollout_lrd_3h` stepping every 3 hours. The
    5-day observed/lrd path is untouched.

    Args:
        trajectories: Prepared module dataset on the 8-hour axis.
        config: Validated training configuration (``dynamics == "lrd_3h"``).
        run_dir: Empty repeat output directory.
        logger: Repeat logger.

    Returns:
        Training history, global metrics, and best epoch.
    """

    config.validate()
    if config.dynamics != "lrd_3h":
        raise ValueError("train_module_model_lrd_3h requires dynamics == 'lrd_3h'")
    existing = [path for path in run_dir.iterdir() if path.name != "run.log"] if run_dir.exists() else []
    if existing:
        raise FileExistsError(f"Run directory already contains outputs: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_reproducibility(config.seed, config.deterministic)
    device = resolve_device(config.device)
    fit_matrix = _lrd3h_fit_matrix(
        trajectories.values, trajectories.replicate_values, config.loss_mode
    )
    module_scaler = FeatureScaler.from_matrix(fit_matrix)
    standardized_means = module_scaler.transform(trajectories.values).astype(np.float32)
    standardized_replicates = module_scaler.transform(trajectories.replicate_values).astype(np.float32)
    means_tensor = torch.from_numpy(standardized_means)
    replicates_tensor = torch.from_numpy(standardized_replicates)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(means_tensor, replicates_tensor),
        batch_size=min(config.batch_size, len(means_tensor)),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    module_dim = len(trajectories.module_ids)
    model = build_transition_model(
        config.model,
        module_dim,
        config.conditioning,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
        config.adapter_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
        )
    full_means = means_tensor.to(device)
    full_replicates = replicates_tensor.to(device)
    checkpoint_path = run_dir / "best_model.pt"
    history_rows: list[dict[str, float | int | bool]] = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    init_index = HOUR_TO_INDEX[LRD3H_INIT_HOUR]
    train_keys = LRD3H_TRAIN_HOURS
    val_keys = LRD3H_TRAIN_HOURS + (LRD3H_VAL_HOUR,)
    predict_keys = LRD3H_TRAIN_HOURS + (LRD3H_VAL_HOUR, LRD3H_TEST_HOUR)
    val_index = HOUR_TO_INDEX[LRD3H_VAL_HOUR]
    logger.info(
        "Starting %s/%s/%s loss=%s on %s: treatments=%d modules=%d seed=%d",
        config.model,
        config.dynamics,
        config.conditioning,
        config.loss_mode,
        device,
        len(trajectories.treatment_ids),
        module_dim,
        config.seed,
    )
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_sum = 0.0
        n_seen = 0
        for batch_means_cpu, batch_reps_cpu in loader:
            batch_means = batch_means_cpu.to(device)
            batch_reps = batch_reps_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = rollout_lrd_3h(
                model,
                batch_means[:, init_index],
                batch_means[:, init_index],
                train_keys,
                init_hour=LRD3H_INIT_HOUR,
                step_hours=LRD3H_STEP_HOURS,
            )
            total_loss, _per_hour = _lrd3h_training_loss(
                predictions, batch_means, batch_reps, config
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_size = int(batch_means.shape[0])
            n_seen += batch_size
            total_sum += float(total_loss.detach()) * batch_size
        train_loss = total_sum / n_seen
        model.eval()
        with torch.no_grad():
            validation = rollout_lrd_3h(
                model,
                full_means[:, init_index],
                full_means[:, init_index],
                val_keys,
                init_hour=LRD3H_INIT_HOUR,
                step_hours=LRD3H_STEP_HOURS,
            )
            val_loss = float(
                module_target_loss(
                    validation[LRD3H_VAL_HOUR],
                    full_means[:, val_index],
                    full_replicates[:, val_index],
                    config.loss_mode,
                    config.lambda_box,
                ).detach()
            )
        if scheduler is not None:
            scheduler.step(val_loss)
        improved = val_loss < best_loss - config.min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "is_best": improved,
                "patience_remaining": max(0, config.patience - stale_epochs),
            }
        )
        if epoch == 1 or epoch % config.log_every == 0:
            logger.info(
                "epoch=%d train=%.6f val15=%.6f lr=%.3g best_epoch=%d",
                epoch,
                train_loss,
                val_loss,
                current_lr,
                best_epoch,
            )
        if stale_epochs >= config.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break
    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without producing a checkpoint")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        prediction_std = rollout_lrd_3h(
            model,
            full_means[:, init_index],
            full_means[:, init_index],
            predict_keys,
            init_hour=LRD3H_INIT_HOUR,
            step_hours=LRD3H_STEP_HOURS,
        )
    predictions_std_one = {
        hour: tensor.detach().cpu().numpy() for hour, tensor in prediction_std.items()
    }
    predictions_raw_one = {
        hour: module_scaler.inverse_transform(values)
        for hour, values in predictions_std_one.items()
    }
    label = f"{config.model}:{config.dynamics}:{config.conditioning}:{config.loss_mode}"
    predictions_raw = {label: predictions_raw_one}
    predictions_std = {label: predictions_std_one}
    metrics, treatment_metrics, module_metrics = build_module_metric_tables(
        predictions_raw,
        predictions_std,
        trajectories.values,
        standardized_means,
        trajectories.treatment_ids,
        trajectories.module_ids,
        HOUR_TO_INDEX,
        SPLIT_BY_HOUR,
    )
    write_prediction_archive(
        run_dir / "predictions.npz",
        predictions_raw,
        predictions_std,
        trajectories.values,
        standardized_means,
        trajectories.treatment_ids,
        trajectories.module_ids,
        HOUR_TO_INDEX,
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(run_dir / "training_history.csv", index=False)
    metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
    treatment_metrics.to_csv(run_dir / "treatment_metrics.tsv", sep="\t", index=False)
    module_metrics.to_csv(run_dir / "module_metrics.tsv", sep="\t", index=False)
    run_config: dict[str, object] = {
        **asdict(config),
        "schema_version": 2,
        "pipeline": "module_dynamics/basic_rollout",
        "run_type": "learned_repeat",
        "administration_route": "Gavage",
        "resolved_device": str(device),
        "input_level": "standardized module eigengene state",
        "target_level": "module eigengene residual",
        "n_treatments": len(trajectories.treatment_ids),
        "n_modules": module_dim,
        "expected_replicates_per_treatment_day": EXPECTED_REPLICATES_PER_TREATMENT_DAY,
        "split_contract": {
            "input": "observed 3H module state only",
            "time_axis": "eight_timepoint_hours",
            "init_hour": LRD3H_INIT_HOUR,
            "step_hours": LRD3H_STEP_HOURS,
            "train_loss_hours": list(LRD3H_TRAIN_HOURS),
            "validation_hour": LRD3H_VAL_HOUR,
            "test_hour": LRD3H_TEST_HOUR,
            "split_axis": "timepoint",
            "same_treatments_across_splits": True,
            "teacher_forcing": False,
            "loss_mode": config.loss_mode,
            "lambda_box": config.lambda_box,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "scaler.json").write_text(
        json.dumps(module_scaler.to_dict("module_ids", trajectories.module_ids), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = format_learned_repeat_summary(
        run_config, trajectories, metrics, best_epoch, best_loss
    )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    test_raw = metrics[metrics["day"].eq(LRD3H_TEST_HOUR) & metrics["scale"].eq("raw")].iloc[0]
    test_std = metrics[metrics["day"].eq(LRD3H_TEST_HOUR) & metrics["scale"].eq("standardized")].iloc[0]
    logger.info(
        "Best epoch=%d; test29_loss_standardized_mse=%.6f raw_mse=%.6f raw_pearson=%.6f",
        best_epoch,
        test_std["mse"],
        test_raw["mse"],
        test_raw["pearson"],
    )
    return TrainingArtifacts(history=history, metrics=metrics, best_epoch=best_epoch)
