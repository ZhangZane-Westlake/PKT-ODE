# Statistical and learned comparators

This package contains the comparator models evaluated alongside PKT-ODE. It does not implement PKT-ODE itself.

## Models

- Statistical: early mean, linear trend, log-time trend, and day-8 persistence.
- Learned: shared residual linear and multilayer-perceptron transitions.

The learned models do not receive administered dose or concentration as an explicit physical input. Their configurations and checkpoints must be selected by raw-scale day-15 MSE before day-29 test values are read.

## Data contract

The loader consumes a reduction directory containing:

```text
projection/
├── treatment_module_trajectories.npz
├── module_bundle.npz
└── sample_metadata.tsv.gz
```

The public snapshot also includes `reduction_scope.json`, which supplies portable scope metadata without encoding a local filesystem layout.

## Examples

```bash
REDUCTION_DIR=data/processed/fenofibrate_reduction

python3 -m src.module_dynamics.basic_rollout baseline \
  --reduction-dir "$REDUCTION_DIR" \
  --run-name reproduction

python3 -m src.module_dynamics.basic_rollout train \
  --reduction-dir "$REDUCTION_DIR" \
  --model linear --dynamics observed --conditioning none \
  --loss-mode mean --device cpu --run-name linear_reproduction
```

Outputs are new run artifacts. The manuscript benchmark snapshot is stored separately in `data/processed/benchmark_metrics.tsv`.

