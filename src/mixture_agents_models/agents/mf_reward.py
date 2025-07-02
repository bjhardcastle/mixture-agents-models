"""
Model-free reward learning agent.
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
class MFReward(Agent):
    """
    Model-free reward learning agent.

    Implements simple temporal difference learning for
    reward-based value updates.
    """

    alpha: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_scale: int = 1
    alpha_prior: DistributionType = field(
        default_factory=lambda: scipy.stats.beta(1, 1)  # type: ignore
    )
    
    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5])
    )
    beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]  # type: ignore
    )
    color: str = "seagreen"
    color_lite: str = "palegreen"

    @override
    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """Update Q values using model-free reward learning rule."""
        q_new = q * (1 - self.alpha)
        q_new[data.choices[t]] += self.alpha * data.rewards[t]
        return q_new

    @property
    def alpha_title(self) -> str:
        return "α(MFreward)"

    @property
    @override
    def beta_title(self) -> str:
        return "β(MF)"
