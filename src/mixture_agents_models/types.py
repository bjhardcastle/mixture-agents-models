"""
Type definitions and protocols for mixture agents models.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.stats

logger = logging.getLogger(__name__)

# Type aliases for better readability
DistributionType = scipy.stats.rv_continuous | scipy.stats.rv_discrete

ArrayLike = npt.NDArray | list | tuple


# Forward declarations for circular imports
class RatData(Protocol):
    """Protocol for behavioral data objects."""

    choices: npt.NDArray[np.int_]
    rewards: npt.NDArray[np.float64]
    n_trials: int
    n_sessions: int


class Agent(Protocol):
    """Protocol defining the Agent interface."""

    @property
    def q0(self) -> npt.NDArray[np.float64]:
        """Initial Q values."""
        ...

    @property
    def beta_prior(self) -> list[DistributionType]:
        """Prior distributions for beta parameters."""
        ...

    def next_q(
        self, q: npt.NDArray[np.float64], data: RatData, t: int
    ) -> npt.NDArray[np.float64]:
        """Update Q values based on trial data."""
        ...

    def get_params(self) -> dict[str, float | int | str | bool]:
        """Get agent parameters."""
        ...


class ModelOptions(Protocol):
    """Protocol for model configuration options."""

    n_states: int
    max_iter: int
    tol: float


class SimOptions(Protocol):
    """Protocol for simulation configuration options."""

    n_trials: int
    n_reps: int
