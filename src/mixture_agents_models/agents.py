"""
Base agent classes and core agent implementations.

This module defines the abstract Agent base class and implements
common reinforcement learning agents for behavioral modeling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Union, Optional, Dict, Any, Tuple
import numpy as np
import scipy.stats

from .types import DistributionType


class Agent(ABC):
    """
    Abstract base class for all behavioral agents.
    
    Agents represent individual cognitive strategies that can be mixed
    in hidden Markov models to explain complex behavioral patterns.
    """
    
    @property
    @abstractmethod
    def q0(self) -> np.ndarray:
        """Initial Q values with length 4 for compatibility."""
        pass
    
    @property 
    @abstractmethod
    def beta_prior(self) -> list[DistributionType]:
        """Prior distributions for beta parameters."""
        pass
    
    @property
    @abstractmethod  
    def color(self) -> str:
        """Color for plotting this agent."""
        pass
    
    @property
    @abstractmethod
    def color_lite(self) -> str:
        """Lighter color for plotting this agent.""" 
        pass
    
    @abstractmethod
    def next_q(self, q: np.ndarray, data: 'RatData', t: int) -> np.ndarray:
        """
        Update Q values based on trial data.
        
        Args:
            q: Current Q values
            data: Behavioral data object
            t: Trial index
            
        Returns:
            Updated Q values
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get dictionary of agent parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **kwargs: Any) -> 'Agent':
        """Return new agent instance with updated parameters."""
        pass


@dataclass
class MBReward(Agent):
    """
    Model-based reward learning agent.
    
    Implements value learning based on reward predictions using 
    model-based reinforcement learning principles.
    """
    
    alpha: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_scale: int = 1
    alpha_prior: DistributionType = field(default_factory=lambda: scipy.stats.beta(1, 1))
    _q0: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5]))
    _beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]
    )
    _color: str = "blue"
    _color_lite: str = "lightblue"
    
    @property
    def q0(self) -> np.ndarray:
        return self._q0
    
    @property 
    def beta_prior(self) -> list[DistributionType]:
        return self._beta_prior
    
    @property
    def color(self) -> str:
        return self._color
    
    @property
    def color_lite(self) -> str:
        return self._color_lite
    
    def next_q(self, q: np.ndarray, data: 'RatData', t: int) -> np.ndarray:
        """Update Q values using model-based reward learning rule."""
        q_new = q * (1 - self.alpha)
        q_new[data.choices[t]] += self.alpha * data.rewards[t]
        return q_new
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "alpha_scale": self.alpha_scale,
            "alpha_prior": self.alpha_prior
        }
    
    def set_params(self, **kwargs: Any) -> 'MBReward':
        params = self.get_params()
        params.update(kwargs)
        return MBReward(**params)
    
    def alpha_title(self) -> str:
        return "α(MBreward)"
    
    def beta_title(self) -> str:
        return "β(MB)"


@dataclass  
class MFReward(Agent):
    """
    Model-free reward learning agent.
    
    Implements simple temporal difference learning for 
    reward-based value updates.
    """
    
    alpha: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_scale: int = 1
    alpha_prior: DistributionType = field(default_factory=lambda: scipy.stats.beta(1, 1))
    _q0: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5]))
    _beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]
    )
    _color: str = "seagreen"
    _color_lite: str = "palegreen"
    
    @property
    def q0(self) -> np.ndarray:
        return self._q0
    
    @property
    def beta_prior(self) -> list[DistributionType]:
        return self._beta_prior
    
    @property
    def color(self) -> str:
        return self._color
    
    @property
    def color_lite(self) -> str:
        return self._color_lite
    
    def next_q(self, q: np.ndarray, data: 'RatData', t: int) -> np.ndarray:
        """Update Q values using model-free reward learning rule.""" 
        q_new = q * (1 - self.alpha)
        q_new[data.choices[t]] += self.alpha * data.rewards[t]
        return q_new
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "alpha_scale": self.alpha_scale, 
            "alpha_prior": self.alpha_prior
        }
    
    def set_params(self, **kwargs: Any) -> 'MFReward':
        params = self.get_params()
        params.update(kwargs)
        return MFReward(**params)
    
    def alpha_title(self) -> str:
        return "α(MFreward)"
    
    def beta_title(self) -> str:
        return "β(MF)"


@dataclass
class Bias(Agent):
    """
    Static bias agent that maintains constant choice preferences.
    
    Represents persistent tendencies independent of learning or rewards.
    """
    
    _q0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    _beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]
    )
    _color: str = "gray"
    _color_lite: str = "lightgray"
    
    @property
    def q0(self) -> np.ndarray:
        return self._q0
    
    @property
    def beta_prior(self) -> list[DistributionType]:
        return self._beta_prior
    
    @property
    def color(self) -> str:
        return self._color
    
    @property
    def color_lite(self) -> str:
        return self._color_lite
    
    def next_q(self, q: np.ndarray, data: 'RatData', t: int) -> np.ndarray:
        """Bias agent maintains constant values."""
        return q
    
    def get_params(self) -> Dict[str, Any]:
        return {}
    
    def set_params(self, **kwargs: Any) -> 'Bias':
        return Bias()
    
    def beta_title(self) -> str:
        return "β(Bias)"


@dataclass
class Perseveration(Agent):
    """
    Perseveration agent that learns action values based on choice history.
    
    Represents tendency to repeat previous actions independent of outcomes.
    """
    
    alpha: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_scale: int = 1
    alpha_prior: DistributionType = field(default_factory=lambda: scipy.stats.beta(1, 1))
    _q0: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.0, 0.0]))
    _beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]
    )
    _color: str = "orange"
    _color_lite: str = "moccasin"
    
    @property
    def q0(self) -> np.ndarray:
        return self._q0
    
    @property
    def beta_prior(self) -> list[DistributionType]:
        return self._beta_prior
    
    @property
    def color(self) -> str:
        return self._color
    
    @property
    def color_lite(self) -> str:
        return self._color_lite
    
    def next_q(self, q: np.ndarray, data: 'RatData', t: int) -> np.ndarray:
        """Update Q values based on choice history."""
        q_new = q * (1 - self.alpha)
        q_new[data.choices[t]] += self.alpha
        return q_new
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "alpha_scale": self.alpha_scale,
            "alpha_prior": self.alpha_prior
        }
    
    def set_params(self, **kwargs: Any) -> 'Perseveration':
        params = self.get_params()
        params.update(kwargs)
        return Perseveration(**params)
    
    def alpha_title(self) -> str:
        return "α(Persev)"
    
    def beta_title(self) -> str:
        return "β(Persev)"


@dataclass
class ContextRL(Agent):
    """
    Context-sensitive reinforcement learning agent.
    
    Specialized for dynamic routing tasks with context-dependent learning.
    Integrates with the existing dynamic routing RL framework.
    """
    
    alpha_context: float = field(default_factory=lambda: np.random.beta(5, 5))
    alpha_reinforcement: float = field(default_factory=lambda: np.random.beta(5, 5))
    tau_context: float = field(default=120.0)
    alpha_context_scale: int = 1
    alpha_reinforcement_scale: int = 1
    alpha_context_prior: DistributionType = field(default_factory=lambda: scipy.stats.beta(1, 1))
    alpha_reinforcement_prior: DistributionType = field(default_factory=lambda: scipy.stats.beta(1, 1))
    _q0: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5]))
    _beta_prior: list[DistributionType] = field(
        default_factory=lambda: [scipy.stats.norm(0, 10)]
    )
    _color: str = "purple"
    _color_lite: str = "lavender"
    
    @property
    def q0(self) -> np.ndarray:
        return self._q0
    
    @property
    def beta_prior(self) -> list[DistributionType]:
        return self._beta_prior
    
    @property
    def color(self) -> str:
        return self._color
    
    @property
    def color_lite(self) -> str:
        return self._color_lite
    
    def next_q(self, q: np.ndarray, data: 'RatData', t: int) -> np.ndarray:
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
        if hasattr(data, 'contexts') and hasattr(data, 'rewards'):
            context = data.contexts[t] if t < len(data.contexts) else 0
            reward = data.rewards[t] if t < len(data.rewards) else 0
            choice = data.choices[t] if t < len(data.choices) else 0
            
            # Context-weighted update
            q_new[choice] = (1 - context_weight * reinforcement_weight) * q[choice] + \
                           context_weight * reinforcement_weight * reward
        
        return q_new
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha_context": self.alpha_context,
            "alpha_reinforcement": self.alpha_reinforcement,
            "tau_context": self.tau_context,
            "alpha_context_scale": self.alpha_context_scale,
            "alpha_reinforcement_scale": self.alpha_reinforcement_scale,
            "alpha_context_prior": self.alpha_context_prior,
            "alpha_reinforcement_prior": self.alpha_reinforcement_prior
        }
    
    def set_params(self, **kwargs: Any) -> 'ContextRL':
        params = self.get_params()
        params.update(kwargs)
        return ContextRL(**params)
    
    def alpha_title(self) -> str:
        return "α(Context)"
    
    def beta_title(self) -> str:
        return "β(ContextRL)"
