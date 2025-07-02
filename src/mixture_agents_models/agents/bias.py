"""
Static bias agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.stats
from typing_extensions import override

from mixture_agents_models.types_ import DistributionType
from mixture_agents_models.agents.base import Agent

if TYPE_CHECKING:
    from mixture_agents_models.types_ import SubjectData

@dataclass
class Bias(Agent):
    """
    Static bias agent that maintains constant choice preferences.

    Represents persistent tendencies independent of learning or rewards.
    """

    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, -1.0, 0.0, 0.0])
    )
    beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]  # type: ignore
    )
    color: str = "gray"
    color_lite: str = "lightgray"

    @override
    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """Bias agent maintains constant values."""
        return q

    def get_params(self) -> dict[str, Any]:
        return {}

    def set_params(self, **kwargs: Any) -> Bias:
        return Bias()
