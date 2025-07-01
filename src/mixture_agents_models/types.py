"""
Type definitions and protocols for mixture agents models.
"""

from typing import Protocol, Union, Any
import scipy.stats
import numpy as np


# Type aliases for better readability
DistributionType = Union[
    scipy.stats.rv_continuous,
    scipy.stats.rv_discrete, 
    scipy.stats._distn_infrastructure.rv_frozen
]

ArrayLike = Union[np.ndarray, list, tuple]

# Forward declarations for circular imports
class RatData(Protocol):
    """Protocol for behavioral data objects."""
    choices: np.ndarray
    rewards: np.ndarray
    n_trials: int
    n_sessions: int


class Agent(Protocol):
    """Protocol defining the Agent interface."""
    
    @property
    def q0(self) -> np.ndarray:
        """Initial Q values."""
        ...
    
    @property
    def beta_prior(self) -> list[DistributionType]:
        """Prior distributions for beta parameters."""
        ...
    
    def next_q(self, q: np.ndarray, data: RatData, t: int) -> np.ndarray:
        """Update Q values based on trial data."""
        ...
    
    def get_params(self) -> dict[str, Any]:
        """Get agent parameters."""
        ...


class ModelOptions(Protocol):
    """Protocol for model configuration options."""
    pass


class SimOptions(Protocol):  
    """Protocol for simulation configuration options."""
    pass
