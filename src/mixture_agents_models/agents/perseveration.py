"""
Perseveration agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.stats
from typing_extensions import override

from mixture_agents_models.types_ import DistributionType
from mixture_agents_models.agents.base import Agent

if TYPE_CHECKING:
    from mixture_agents_models.types_ import SubjectData


@dataclass
class Perseveration(Agent):
    """
    Perseveration agent that learns action values based on choice history.

    Represents tendency to repeat previous actions independent of outcomes.
    """

    alpha: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_scale: int = 1
    alpha_prior: DistributionType = field(  # type: ignore
        default_factory=lambda: scipy.stats.beta(1, 1)
    )
    
    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.0, 0.0])
    )
    beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]  # type: ignore
    )
    color: str = "orange"
    color_lite: str = "moccasin"

    @override
    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """Update Q values based on choice history."""
        q_new = q * (1 - self.alpha)
        q_new[data.choices[t]] += self.alpha
        return q_new

    @property
    def alpha_title(self) -> str:
        return "α(Persev)"

    @property
    @override
    def beta_title(self) -> str:
        return "β(Persev)"
