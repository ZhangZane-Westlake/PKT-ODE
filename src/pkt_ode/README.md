# PKT-ODE implementation

This package contains the manuscript-specific model. It is independent of the learned residual-transition comparators in `src/module_dynamics`.

## Modules

- `prepare_data.py`: rebuilds the aligned, fit-window MAD-normalized PKT-ODE input from the frozen projection.
- `data.py`: validates and loads the compact replicate-level PKT-ODE input and parameter table.
- `model.py`: implements the Bateman PK profile, daily superposition, Hill effect, semi-analytic turnover update, and 12-start L-BFGS-B fitting.
- `evaluation.py`: calculates split metrics and the four deterministic statistical baselines.
- `figures.py`: recreates the four main figures from repository-relative inputs.
- `cli.py`: exposes `prepare-input`, `verify`, `baselines`, `fit`, and `figures` commands.

## Split contract

- PKT-ODE fitting: 3H, 6H, 9H, 1D, 4D, and 8D.
- Learned-comparator selection: 15D.
- Common test endpoint: 29D.

Neither 15D nor 29D enters PKT-ODE parameter fitting or normalization. Fitting uses 54 profiles: three doses by six early times by three biological replicates. Each held-out endpoint metric flattens three doses by three replicates by seven modules, giving 63 descriptive values.

Module scores are divided by the module-wise fit-window MAD multiplied by 1.4826. The archive also preserves raw scores and conventional fit-window standard deviations for auditability.

The `baselines` command intentionally reads the standardized projection condition means to reproduce the manuscript comparator rows. Those 21-value comparator metrics and the 63-value MAD-normalized PKT-ODE metrics preserve their original pipelines and should not be interpreted as sharing an identical MSE scale.

## Commands

```bash
python3 -m src.pkt_ode prepare-input
python3 -m src.pkt_ode verify
python3 -m src.pkt_ode baselines
python3 -m src.pkt_ode figures
python3 -m src.pkt_ode fit \
  --parameters-output results/reproduction/fitted_parameters.tsv \
  --metrics-output results/reproduction/fitted_metrics.tsv
```

CLI paths are resolved relative to the repository root unless an explicit path is supplied.
