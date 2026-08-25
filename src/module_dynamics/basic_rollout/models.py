"""Residual linear and neural module transition models.

Direct module-to-module residual transitions (modules are the top-level state,
so no gene/pathway projection is needed at train time). The four 1D-anchor
conditioning modes and the observed/daily rollout mirror the pathway and gene
rollout packages.
"""

from __future__ import annotations

from typing import Final, Literal

import torch
from torch import nn


ConditioningMode = Literal["none", "concat", "film", "residual_adapter"]
ModelName = Literal["linear", "mlp"]
DynamicsMode = Literal["observed", "lrd", "lrd_3h"]
VALID_CONDITIONING: Final[tuple[str, ...]] = (
    "none",
    "concat",
    "film",
    "residual_adapter",
)


def _zero_output_layer(module: nn.Module) -> None:
    """Zero-initialize the final linear layer of a transition branch.

    Args:
        module: Module containing at least one linear layer.
    """

    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linear_layers:
        raise ValueError("Transition branch has no linear output layer")
    nn.init.zeros_(linear_layers[-1].weight)
    if linear_layers[-1].bias is not None:
        nn.init.zeros_(linear_layers[-1].bias)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Build a typed feed-forward transition branch.

    Args:
        input_dim: Input feature count.
        hidden_dim: Hidden feature count.
        output_dim: Output feature count.
        num_layers: Number of hidden layers.
        dropout: Dropout probability after each hidden activation.

    Returns:
        Sequential MLP with a zero-initialized output layer.
    """

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    network = nn.Sequential(*layers)
    _zero_output_layer(network)
    return network


class ModuleTransition(nn.Module):
    """Base class for a residual module transition."""

    def forward(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Advance one daily or observed transition.

        Args:
            state: Current standardized module activity.
            anchor: Fixed standardized 1D activity for the treatment.

        Returns:
            Next standardized module activity.
        """

        raise NotImplementedError


class LinearResidualTransition(ModuleTransition):
    """One affine residual transition without separate drug conditioning."""

    def __init__(self, module_dim: int) -> None:
        """Initialize the linear transition.

        Args:
            module_dim: Number of module features.
        """

        super().__init__()
        self.delta = nn.Linear(module_dim, module_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Return ``state + affine_delta(state)``.

        Args:
            state: Current standardized activity.
            anchor: Unused 1D anchor retained for a shared model interface.

        Returns:
            Next standardized activity.
        """

        del anchor
        return state + self.delta(state)


class MLPResidualTransition(ModuleTransition):
    """Residual MLP transition with configurable 1D-anchor conditioning."""

    def __init__(
        self,
        module_dim: int,
        conditioning: ConditioningMode,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        adapter_dim: int,
    ) -> None:
        """Initialize one of the four neural conditioning designs.

        Args:
            module_dim: Number of module features.
            conditioning: Drug-conditioning design.
            hidden_dim: Shared hidden dimension.
            num_layers: Number of hidden layers in standard branches.
            dropout: Hidden dropout probability.
            adapter_dim: Residual-adapter bottleneck dimension.
        """

        super().__init__()
        if conditioning not in VALID_CONDITIONING:
            raise ValueError(f"Unsupported conditioning mode: {conditioning}")
        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")
        self.conditioning = conditioning
        self.module_dim = module_dim
        if conditioning == "none":
            self.transition = build_mlp(
                module_dim, hidden_dim, module_dim, num_layers, dropout
            )
        elif conditioning == "concat":
            self.transition = build_mlp(
                module_dim * 2, hidden_dim, module_dim, num_layers, dropout
            )
        elif conditioning == "film":
            self.state_encoder = nn.Sequential(
                nn.Linear(module_dim, hidden_dim), nn.GELU()
            )
            self.film_generator = nn.Linear(module_dim, hidden_dim * 2)
            nn.init.zeros_(self.film_generator.weight)
            nn.init.zeros_(self.film_generator.bias)
            hidden_layers: list[nn.Module] = []
            for _ in range(max(0, num_layers - 1)):
                hidden_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
                if dropout > 0:
                    hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.Linear(hidden_dim, module_dim))
            self.film_transition = nn.Sequential(*hidden_layers)
            _zero_output_layer(self.film_transition)
        else:
            self.shared_transition = build_mlp(
                module_dim, hidden_dim, module_dim, num_layers, dropout
            )
            self.drug_adapter = build_mlp(
                module_dim * 2, adapter_dim, module_dim, 1, dropout
            )

    def delta_components(
        self, state: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shared and drug-specific residual components.

        Args:
            state: Current standardized activity.
            anchor: Fixed standardized 1D activity.

        Returns:
            A ``(shared_delta, drug_delta)`` tuple. Non-adapter modes return the
            entire transition as the shared term and a zero drug term.
        """

        if self.conditioning == "none":
            shared = self.transition(state)
            return shared, torch.zeros_like(shared)
        if self.conditioning == "concat":
            shared = self.transition(torch.cat([state, anchor], dim=-1))
            return shared, torch.zeros_like(shared)
        if self.conditioning == "film":
            hidden = self.state_encoder(state)
            gamma_raw, beta = self.film_generator(anchor).chunk(2, dim=-1)
            modulated = (1.0 + gamma_raw) * hidden + beta
            shared = self.film_transition(modulated)
            return shared, torch.zeros_like(shared)
        shared = self.shared_transition(state)
        drug = self.drug_adapter(torch.cat([state, anchor], dim=-1))
        return shared, drug

    def forward(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Advance one residual transition.

        Args:
            state: Current standardized activity.
            anchor: Fixed standardized 1D treatment activity.

        Returns:
            Next standardized activity.
        """

        shared, drug = self.delta_components(state, anchor)
        return state + shared + drug


def build_transition_model(
    model_name: ModelName,
    module_dim: int,
    conditioning: ConditioningMode,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    adapter_dim: int,
) -> ModuleTransition:
    """Construct and validate a transition model.

    Args:
        model_name: Linear or MLP transition family.
        module_dim: Number of module features.
        conditioning: Requested conditioning mode.
        hidden_dim: MLP hidden dimension.
        num_layers: MLP hidden-layer count.
        dropout: MLP dropout probability.
        adapter_dim: Drug adapter bottleneck dimension.

    Returns:
        Initialized residual transition model.

    Raises:
        ValueError: If the model/conditioning combination is invalid.
    """

    if model_name == "linear":
        if conditioning != "none":
            raise ValueError("The linear model only supports conditioning=none")
        return LinearResidualTransition(module_dim)
    if model_name == "mlp":
        return MLPResidualTransition(
            module_dim,
            conditioning,
            hidden_dim,
            num_layers,
            dropout,
            adapter_dim,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def rollout(
    model: ModuleTransition,
    initial_state: torch.Tensor,
    anchor: torch.Tensor,
    dynamics: DynamicsMode,
    target_days: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Roll a model forward from day 1 without teacher forcing.

    Args:
        model: Residual transition model.
        initial_state: Observed standardized 1D state.
        anchor: Fixed standardized 1D condition vector.
        dynamics: ``observed`` or daily ``lrd`` transitions.
        target_days: Increasing observed days to retain.

    Returns:
        Mapping from requested day number to predicted state.
    """

    if not target_days or tuple(sorted(target_days)) != target_days:
        raise ValueError("target_days must be a non-empty increasing tuple")
    if target_days[0] <= 1:
        raise ValueError("target_days must occur after day 1")
    state = initial_state
    predictions: dict[int, torch.Tensor] = {}
    if dynamics == "lrd":
        targets = set(target_days)
        for day in range(2, target_days[-1] + 1):
            state = model(state, anchor)
            if day in targets:
                predictions[day] = state
        return predictions
    if dynamics == "observed":
        for day in target_days:
            state = model(state, anchor)
            predictions[day] = state
        return predictions
    raise ValueError(f"Unsupported dynamics mode: {dynamics}")


def rollout_lrd_3h(
    model: ModuleTransition,
    initial_state: torch.Tensor,
    anchor: torch.Tensor,
    target_hours: tuple[int, ...],
    init_hour: int = 3,
    step_hours: int = 3,
) -> dict[int, torch.Tensor]:
    """Roll a model forward in 3-hour steps over the eight-timepoint hour axis.

    Same residual transition as :func:`rollout`, but applied once per ``step_hours``
    elapsed hour from the ``init_hour`` observation (3H) so the model can learn
    from and predict the 3H/6H/9H short-term responses. All eight timepoints
    (3/6/9/24/96/192/360/696 h) are reachable from 3 in 3-hour steps.

    Args:
        model: Residual transition model (one call = one ``step_hours`` step).
        initial_state: Observed standardized state at ``init_hour`` (3H).
        anchor: Fixed standardized condition vector.
        target_hours: Increasing elapsed-hour values to retain (all > init_hour
            and reachable in ``step_hours`` increments).
        init_hour: Elapsed hour of ``initial_state`` (default 3).
        step_hours: Step granularity in hours (default 3).

    Returns:
        Mapping from requested elapsed hour to predicted state.

    Raises:
        ValueError: If ``target_hours`` is empty/non-increasing, a target is not
            beyond ``init_hour``, or a target is unreachable in ``step_hours``
            increments from ``init_hour``.
    """

    if not target_hours or tuple(sorted(target_hours)) != target_hours:
        raise ValueError("target_hours must be a non-empty increasing tuple")
    if target_hours[0] <= init_hour:
        raise ValueError("target_hours must occur after init_hour")
    unreachable = [
        h for h in target_hours if (h - init_hour) % step_hours != 0
    ]
    if unreachable:
        raise ValueError(
            f"target_hours {unreachable} not reachable from init_hour="
            f"{init_hour} in {step_hours}h steps"
        )
    targets = set(target_hours)
    state = initial_state
    predictions: dict[int, torch.Tensor] = {}
    start = init_hour + step_hours
    for hour in range(start, target_hours[-1] + 1, step_hours):
        state = model(state, anchor)
        if hour in targets:
            predictions[hour] = state
    return predictions
