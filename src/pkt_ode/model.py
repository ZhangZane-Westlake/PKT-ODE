"""PKT-ODE pharmacokinetic driver, turnover solver, and parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class PKParameters:
    """Fixed rat fenofibrate pharmacokinetic and Hill parameters."""

    absorption_rate_per_day: float = 11.45
    elimination_rate_per_day: float = 2.64
    apparent_volume_l_per_kg: float = 0.441
    active_parent_mass_ratio: float = 318.75 / 360.83
    ec50_ug_per_ml: float = 5.42
    hill_coefficient: float = 1.0


@dataclass(frozen=True)
class ModuleParameters:
    """Module-specific first-order turnover parameters."""

    k_per_day: float
    beta_0: float
    beta_1: float


def bateman_concentration(
    time_since_dose: np.ndarray,
    dose_mg_per_kg: float,
    parameters: PKParameters = PKParameters(),
) -> np.ndarray:
    """Evaluate a one-compartment oral Bateman concentration profile.

    Args:
        time_since_dose: Time since one dose in days.
        dose_mg_per_kg: Administered parent-compound dose.
        parameters: Fixed pharmacokinetic parameters.

    Returns:
        Active-metabolite plasma concentration in micrograms per millilitre.
    """

    times = np.asarray(time_since_dose, dtype=float)
    concentration = np.zeros_like(times)
    positive = times > 0.0
    t_pos = times[positive]
    ka = parameters.absorption_rate_per_day
    kel = parameters.elimination_rate_per_day
    scale = (
        dose_mg_per_kg
        * parameters.active_parent_mass_ratio
        / parameters.apparent_volume_l_per_kg
        * ka
        / (ka - kel)
    )
    concentration[positive] = scale * (np.exp(-kel * t_pos) - np.exp(-ka * t_pos))
    return concentration


def repeated_dose_concentration(
    times: np.ndarray,
    dose_mg_per_kg: float,
    parameters: PKParameters = PKParameters(),
) -> np.ndarray:
    """Superimpose single-dose profiles for once-daily administration.

    Args:
        times: Times since the first dose in days.
        dose_mg_per_kg: Daily administered dose.
        parameters: Fixed pharmacokinetic parameters.

    Returns:
        Active-metabolite concentration at every requested time.
    """

    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1 or sample_times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if np.min(sample_times) < 0.0:
        raise ValueError("times cannot be negative")
    total = np.zeros_like(sample_times)
    for dosing_day in range(int(np.floor(np.max(sample_times))) + 1):
        total += bateman_concentration(
            sample_times - float(dosing_day), dose_mg_per_kg, parameters
        )
    return total


def plasma_concentration(
    times: np.ndarray,
    dose_mg_per_kg: float,
    regimen: str,
    parameters: PKParameters = PKParameters(),
) -> np.ndarray:
    """Evaluate the concentration profile for one experimental regimen.

    Args:
        times: Times since the first dose in days.
        dose_mg_per_kg: Administered dose in mg/kg.
        regimen: ``Single`` or once-daily ``Repeat`` administration.
        parameters: Fixed pharmacokinetic parameters.

    Returns:
        Active-metabolite concentration at every requested time.
    """

    if regimen == "Single":
        return bateman_concentration(times, dose_mg_per_kg, parameters)
    if regimen == "Repeat":
        return repeated_dose_concentration(times, dose_mg_per_kg, parameters)
    raise ValueError(f"Unknown regimen: {regimen}")


def hill_effect(
    concentration: np.ndarray,
    parameters: PKParameters = PKParameters(),
) -> np.ndarray:
    """Map concentration to a bounded Hill drug-effect signal.

    Args:
        concentration: Non-negative concentrations.
        parameters: Hill parameters stored with the fixed PK values.

    Returns:
        Drug-effect values in the closed interval [0, 1].
    """

    cp = np.maximum(np.asarray(concentration, dtype=float), 0.0)
    n_hill = parameters.hill_coefficient
    numerator = cp**n_hill
    return numerator / (parameters.ec50_ug_per_ml**n_hill + numerator)


def simulate_module(
    observation_times: np.ndarray,
    dose_mg_per_kg: float,
    module_parameters: ModuleParameters,
    pk_parameters: PKParameters = PKParameters(),
    step_days: float = 0.05,
    regimen: str = "Repeat",
) -> np.ndarray:
    """Simulate one module using the manuscript's semi-analytic update.

    Args:
        observation_times: Requested times in days.
        dose_mg_per_kg: Daily administered dose.
        module_parameters: Module turnover, baseline, and signed effect.
        pk_parameters: Fixed PK and Hill parameters.
        step_days: Integration grid spacing in days.
        regimen: ``Single`` or once-daily ``Repeat`` administration.

    Returns:
        Directed module PC1 scores at ``observation_times``.
    """

    times = np.asarray(observation_times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("observation_times must be one-dimensional and non-empty")
    if step_days <= 0.0:
        raise ValueError("step_days must be positive")
    if module_parameters.k_per_day <= 0.0:
        raise ValueError("k_per_day must be positive")
    end = max(30.5, float(np.max(times)))
    grid = np.arange(0.0, end + step_days * 0.5, step_days)
    effect = hill_effect(
        plasma_concentration(grid, dose_mg_per_kg, regimen, pk_parameters),
        pk_parameters,
    )
    target = module_parameters.beta_0 + module_parameters.beta_1 * effect
    state = np.empty_like(grid)
    state[0] = module_parameters.beta_0
    k_rate = module_parameters.k_per_day
    decay = np.exp(-k_rate * step_days)
    linear_driver_coefficient = 1.0 - (1.0 - decay) / (k_rate * step_days)
    for index in range(1, len(grid)):
        state[index] = (
            decay * state[index - 1]
            + (1.0 - decay) * target[index - 1]
            + (target[index] - target[index - 1]) * linear_driver_coefficient
        )
    return np.interp(times, grid, state)


def simulate_trajectories(
    doses: Sequence[float],
    observation_times: np.ndarray,
    module_ids: Sequence[str],
    module_parameters: Mapping[str, ModuleParameters],
    pk_parameters: PKParameters = PKParameters(),
    step_days: float = 0.05,
    regimens: Sequence[str] | None = None,
) -> np.ndarray:
    """Simulate all dose, time, and module combinations.

    Args:
        doses: Daily doses in mg/kg.
        observation_times: Requested times in days.
        module_ids: Ordered module identifiers.
        module_parameters: Parameters keyed by module identifier.
        pk_parameters: Fixed PK and Hill parameters.
        step_days: Integration grid spacing in days.
        regimens: Optional regimen label aligned to every observation time.

    Returns:
        Array shaped dose by time by module.
    """

    missing = set(module_ids).difference(module_parameters)
    if missing:
        raise ValueError(f"Missing parameters for modules: {sorted(missing)}")
    times = np.asarray(observation_times, dtype=float)
    regimen_values = tuple(regimens) if regimens is not None else tuple("Repeat" for _ in times)
    if len(regimen_values) != len(times):
        raise ValueError("regimens must align with observation_times")
    predictions = np.empty((len(doses), len(times), len(module_ids)))
    for dose_index, dose in enumerate(doses):
        for module_index, module_id in enumerate(module_ids):
            for regimen in set(regimen_values):
                mask = np.asarray([value == regimen for value in regimen_values])
                predictions[dose_index, mask, module_index] = simulate_module(
                    times[mask],
                    float(dose),
                    module_parameters[module_id],
                    pk_parameters,
                    step_days,
                    regimen,
                )
    return predictions


def _fit_one_module(
    doses: np.ndarray,
    times: np.ndarray,
    regimens: np.ndarray,
    observed: np.ndarray,
    pk_parameters: PKParameters,
    random_seed: int,
    n_starts: int,
) -> ModuleParameters:
    """Fit one module using the original bounded multi-start optimization."""

    rng = np.random.default_rng(random_seed)
    scale = max(float(np.std(observed)), 0.1)
    log_rate_bounds = (float(np.log(0.03)), float(np.log(40.0)))
    log_rate_grid = np.log(np.asarray([0.05, 0.15, 0.5, 1.5, 5.0, 20.0]))
    grid = [(log_rate, direction) for log_rate in log_rate_grid for direction in (-1.0, 1.0)]

    effect_cache: dict[tuple[float, str], np.ndarray] = {}
    dense_grid = np.arange(0.0, 30.5 + 0.05 / 2.0, 0.05)
    for dose, regimen in set(zip(doses.tolist(), regimens.tolist())):
        effect_cache[(float(dose), str(regimen))] = hill_effect(
            plasma_concentration(dense_grid, float(dose), str(regimen), pk_parameters),
            pk_parameters,
        )

    def predict_rows(candidate: ModuleParameters) -> np.ndarray:
        """Predict every replicate-level observation row."""

        predictions = np.empty(len(times), dtype=float)
        for dose, regimen in set(zip(doses.tolist(), regimens.tolist())):
            key = (float(dose), str(regimen))
            mask = (doses == dose) & (regimens == regimen)
            target = candidate.beta_0 + candidate.beta_1 * effect_cache[key]
            state = np.empty_like(dense_grid)
            state[0] = candidate.beta_0
            rate_step = candidate.k_per_day * 0.05
            decay = np.exp(-rate_step)
            one_minus_decay = (1.0 - decay) if rate_step >= 1e-8 else rate_step
            slope_coefficient = 1.0 - one_minus_decay / max(rate_step, 1e-12)
            for index in range(len(dense_grid) - 1):
                state[index + 1] = (
                    decay * state[index]
                    + one_minus_decay * target[index]
                    + (target[index + 1] - target[index]) * slope_coefficient
                )
            predictions[mask] = np.interp(times[mask], dense_grid, state)
        return predictions

    def objective(theta: np.ndarray) -> float:
        """Return residual sum of squares for one optimization vector."""

        candidate = ModuleParameters(
            k_per_day=float(np.exp(np.clip(theta[0], *log_rate_bounds))),
            beta_0=float(theta[1]),
            beta_1=float(theta[2]),
        )
        predicted = predict_rows(candidate)
        return float(np.sum((predicted - observed) ** 2))

    best_result = None
    bounds = [log_rate_bounds, (-5.0 * scale, 5.0 * scale), (-8.0 * scale, 8.0 * scale)]
    for start_index in range(max(n_starts, len(grid))):
        if start_index < len(grid):
            log_rate, direction = grid[start_index]
            start = np.asarray(
                [log_rate, rng.normal(0.0, 0.3 * scale), direction * scale]
            )
        else:
            start = np.asarray(
                [
                    rng.choice(log_rate_grid) + rng.normal(0.0, 0.3),
                    rng.normal(0.0, 0.3 * scale),
                    rng.choice([-1.0, 1.0]) * scale * (0.5 + rng.uniform(0.0, 1.5)),
                ]
            )
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-11},
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    if best_result is None or not np.isfinite(best_result.fun):
        raise RuntimeError("No finite parameter fit was obtained")
    return ModuleParameters(
        k_per_day=float(np.exp(best_result.x[0])),
        beta_0=float(best_result.x[1]),
        beta_1=float(best_result.x[2]),
    )


def fit_pkt_ode(
    doses: Sequence[float],
    times: np.ndarray,
    replicate_values: np.ndarray,
    module_ids: Sequence[str],
    regimens: Sequence[str],
    pk_parameters: PKParameters = PKParameters(),
    training_end_day: float = 8.0,
    random_seed: int = 42,
    n_starts: int = 12,
) -> dict[str, ModuleParameters]:
    """Fit independent module parameters using only the training time window.

    Args:
        doses: Daily doses in mg/kg.
        times: Observation times in days.
        replicate_values: Dose by time by replicate by module observations.
        module_ids: Ordered module identifiers.
        regimens: Single- or repeat-dose label aligned to every time.
        pk_parameters: Fixed PK and Hill parameters.
        training_end_day: Last time allowed in parameter fitting.
        random_seed: Base seed for pseudo-random starts.
        n_starts: Number of multi-start optimization attempts; at least 12 are used.

    Returns:
        Fitted parameters keyed by module identifier.
    """

    times_array = np.asarray(times, dtype=float)
    values = np.asarray(replicate_values, dtype=float)
    expected_shape = (len(doses), len(times_array), 3, len(module_ids))
    if values.shape != expected_shape:
        raise ValueError(
            f"replicate_values must have shape {expected_shape}, got {values.shape}"
        )
    regimen_values = np.asarray(regimens, dtype=str)
    if regimen_values.shape != times_array.shape:
        raise ValueError("regimens must align with times")
    training_mask = times_array <= training_end_day
    if not training_mask.any():
        raise ValueError("No observation time lies in the training window")
    replicate_count = values.shape[2]
    training_doses = np.broadcast_to(
        np.asarray(doses, dtype=float)[:, None, None],
        (len(doses), int(training_mask.sum()), replicate_count),
    ).reshape(-1)
    training_times = np.broadcast_to(
        times_array[training_mask][None, :, None],
        (len(doses), int(training_mask.sum()), replicate_count),
    ).reshape(-1)
    training_regimens = np.broadcast_to(
        regimen_values[training_mask][None, :, None],
        (len(doses), int(training_mask.sum()), replicate_count),
    ).reshape(-1)
    return {
        module_id: _fit_one_module(
            training_doses,
            training_times,
            training_regimens,
            values[:, training_mask, :, module_index].reshape(-1),
            pk_parameters,
            random_seed + module_index,
            n_starts,
        )
        for module_index, module_id in enumerate(module_ids)
    }
