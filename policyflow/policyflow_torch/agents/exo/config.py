from dataclasses import dataclass


@dataclass
class ExoCfg:
    """Configuration for the Exo algorithm."""

    learning_rate: float = 2e-4
    """The initial learning rate."""

    discount_factor: float = 0.99
    """The discount factor."""

    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    time_limit_bootstrap: bool = True
    """Time limit bootstrap for Generalized Advantage Estimation (GAE)."""

    mini_batches: int = 4
    """The number of mini-batches per update."""

    learning_epochs: int = 5
    """The number of learning epochs per update."""

    beta: float = 5.0
    """The beta parameter for exponential soft-clipping of policy ratio."""

    kl_coeff: float = 0.8
    """The coefficient for KL divergence regularization."""

    entropy_loss_scale: float = 0.01
    """The coefficient for the entropy loss."""

    clip_range: float = 0.2
    """The base clip range for the exponential clipping."""

    value_loss_scale: float = 1.0
    """The coefficient for the value loss."""

    grad_norm_clip: float = 1.0
    """The maximum gradient norm."""


class ExoCfgInstance(ExoCfg):
    """Default instance of ExoCfg with sensible defaults (same as ExoCfg defaults)."""
    pass
