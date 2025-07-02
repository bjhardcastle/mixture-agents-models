"""
Base agent classes.

This module defines the abstract Agent base class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from mixture_agents_models.types_ import DistributionType

if TYPE_CHECKING:
    from mixture_agents_models.types_ import SubjectData


@dataclass
class Agent:
    """
    Base class for all behavioral agents.

    Agents represent individual cognitive strategies that can be mixed
    in hidden Markov models to explain complex behavioral patterns.
    """
    
    q0: npt.NDArray[np.float64]
    beta_prior: list[DistributionType]
    color: str
    color_lite: str

    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """
        Update Q values based on trial data.

        The default implementation returns the Q-values unchanged.

        Args:
            q: Current Q values
            data: Behavioral data object
            t: Trial index

        Returns:
            Updated Q values
        """
        return q

    def get_params(self) -> dict[str, Any]:
        """
        Get dictionary of agent parameters.

        Returns a dictionary of the agent's public parameters.
        """
        return {
            k: v
            for k, v in asdict(self).items()
            if not k.startswith("_") and k not in ["q0", "beta_prior", "color", "color_lite"]
        }

    def set_params(self, **kwargs: Any) -> Agent:
        """
        Return new agent instance with updated parameters.
        """
        return replace(self, **kwargs)

    @property
    def beta_title(self) -> str:
        """Title for the beta parameter plot."""
        return f"β({self.__class__.__name__})"

    @property
    def atick(self) -> str:
        """Tick label for plots."""
        return self.__class__.__name__

    @property
    def agent2string(self) -> str:
        """String representation of the agent."""
        return self.__class__.__name__
