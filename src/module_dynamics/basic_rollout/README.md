# Basic module rollout

This subpackage implements four fixed statistical rules and residual-transition neural models on the frozen seven-module state.

## Temporal roles

- Initial observed state: day 1 for the standard rollout.
- Training loss: days 4 and 8.
- Validation and configuration selection: day 15.
- Test: day 29.

The `lrd_3h` mode retains the full 3H-to-29D axis and advances in 3-hour increments. The publication benchmark uses a validation-selected configuration; day-29 labels must never be used for checkpoint or hyperparameter selection.

## Entry points

```bash
python3 -m src.module_dynamics.basic_rollout prepare --help
python3 -m src.module_dynamics.basic_rollout baseline --help
python3 -m src.module_dynamics.basic_rollout train --help
```

All Python functions use typed interfaces and document the expected array axes. See the parent README for publication-scoped examples.

