"""
Gambler's Fallacy agent (B version).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override

from mixture_agents_models.agents.gambler import Gambler


@dataclass
class GamblerB(Gambler):
    """
    Agent computing values from the perspective of a Gambler's Fallacy.

    This is a variant of the Gambler agent with different titles for plotting.
    """

    @property
    @override
    def alpha_title(self) -> str:
        return "α(GamblerB)"

    @property
    @override
    def beta_title(self) -> str:
        return "β(GamblerB)"

    @property
    @override
    def agent2string(self) -> str:
        return "GamblerB"
