"""
Intercept agent.
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
class Intercept(Agent):
    """
    Agent defining a fixed intercept across trials.
    
    Half the magnitude of the Bias agent.
    """

    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, -0.5, 0.0, 0.0])
    )
    beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]  # type: ignore
    )
    color: str = "black"
    color_lite: str = "gray"

    @override
    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """Intercept agent sets specific Q values."""
        q_new = q.copy()
        q_new[0] = 1.0
        q_new[1] = -1.0
        return q_new
