"""
Reward agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.stats

from mixture_agents_models.types_ import DistributionType
from mixture_agents_models.agents.base import Agent
from typing_extensions import override

if TYPE_CHECKING:
    from mixture_agents_models.types_ import SubjectData


@dataclass
class Reward(Agent):
    """
    Agent capturing the effect of previous reward observed at a specified trial lag.
    """

    nback: int = 1

    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.0, 0.0])
    )
    beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]  # type: ignore
    )
    color: str = "royalblue"
    color_lite: str = "salmon"

    def __post_init__(self):
        if self.nback <= 0:
            raise ValueError("nback must be greater than 0")

    @override
    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """Update Q values based on reward history."""
        q_new = np.zeros_like(q)
        tn = t - self.nback + 1
        if tn >= 0:
            if not np.any(data.new_sess[tn + 1 : t + 1]):
                q_new[data.choices[tn]] = data.rewards[tn]
        return q_new

    @property
    @override
    def beta_title(self) -> str:
        return f"β(Reward[t-{self.nback}])"

    @property
    @override
    def atick(self) -> str:
        return f"Reward[t-{self.nback}]"

    @property
    @override
    def agent2string(self) -> str:
        return f"R{self.nback}"
