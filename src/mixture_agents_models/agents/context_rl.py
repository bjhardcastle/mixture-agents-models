"""
Context-sensitive reinforcement learning agent.
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
class ContextRL(Agent):
    """
    Context-sensitive reinforcement learning agent.

    Specialized for dynamic routing tasks with context-dependent learning.
    Integrates with the existing dynamic routing RL framework.
    """

    q0: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5])
    )
    beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]  # type: ignore
    )
    color: str = "purple"
    color_lite: str = "lavender"
    alpha_context: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_reinforcement: float = field(default_factory=lambda: np.random.beta(5, 5))
    tau_context: float = field(default=120.0)
    alpha_context_scale: int = 1
    alpha_reinforcement_scale: int = 1
    alpha_context_prior: DistributionType = field(
        default_factory=lambda: scipy.stats.beta(1, 1)  # type: ignore
    )
    alpha_reinforcement_prior: DistributionType = field(
        default_factory=lambda: scipy.stats.beta(1, 1)  # type: ignore
    )

    @override
    def next_q(
        self, q: npt.NDArray[np.float64], data: SubjectData, t: int
    ) -> npt.NDArray[np.float64]:
        """
        Update Q values using context-sensitive learning rule.

        Implements core logic from dynamic routing RL model for
        context-dependent value updates.
        """
        # This would integrate with the existing dynamic routing logic
        # For now, implement a simplified version
        q_new = q.copy()

        # Context-dependent learning rate
        context_weight = self.alpha_context
        reinforcement_weight = self.alpha_reinforcement

        # Update based on reward and context
        if hasattr(data, "contexts") and hasattr(data, "rewards"):
            contexts = data.contexts  # type: ignore
            rewards = data.rewards  # type: ignore
            choices = data.choices  # type: ignore
            
            context = contexts[t] if t < len(contexts) else 0
            reward = rewards[t] if t < len(rewards) else 0
            choice = choices[t] if t < len(choices) else 0

            # Context-weighted update
            q_new[choice] = (1 - context_weight * reinforcement_weight) * q[
                choice
            ] + context_weight * reinforcement_weight * reward

        return q_new

    @override
    def get_params(self) -> dict[str, Any]:
        return {
            "alpha_context": self.alpha_context,
            "alpha_reinforcement": self.alpha_reinforcement,
            "tau_context": self.tau_context,
            "alpha_context_scale": self.alpha_context_scale,
            "alpha_reinforcement_scale": self.alpha_reinforcement_scale,
            "alpha_context_prior": self.alpha_context_prior,
            "alpha_reinforcement_prior": self.alpha_reinforcement_prior,
        }

    @override
    def set_params(self, **kwargs: Any) -> ContextRL:
        params = self.get_params()
        params.update(kwargs)
        # Extract base class params
        base_params = {
            "q0": self.q0,
            "beta_prior": self.beta_prior,
            "color": self.color,
            "color_lite": self.color_lite,
        }
        base_params.update(params)
        return ContextRL(**base_params)

    def alpha_title(self) -> str:
        return "α(Context)"

    @override
    def beta_title(self) -> str:
        return "β(ContextRL)"
