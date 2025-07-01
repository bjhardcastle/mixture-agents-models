"""
Mixture Agents Models: Python implementation of MoA-HMM framework.

This package provides computational models for analyzing choice behavior using 
mixture-of-agents frameworks with hidden Markov model transitions.
"""

from .agents import (
    Agent, MBReward, MFReward, Bias, Perseveration, ContextRL
)
from .models import (
    ModelHMM, ModelDrift, ModelOptionsHMM, ModelOptionsDrift, AgentOptions,
    optimize, simulate, choice_accuracy
)
from .tasks import (
    GenericData, TwoStepData, DynamicRoutingData, load_generic_data
)
from .integration import (
    fit_dynamic_routing_model, convert_from_dynamic_routing,
    create_dynamic_routing_agents
)
from .plotting import plot_model, plot_recovery, plot_comparison
from .utils import (
    cross_validate, parameter_recovery, model_compare,
    smooth, onehot, compute_agent_strings, beta_title, alpha_title
)

__version__ = "0.1.0"
__author__ = "Translated from Julia implementation"

# Core exports for convenience
__all__ = [
    # Core model types
    "ModelHMM",
    "ModelDrift", 
    "ModelOptionsHMM",
    "ModelOptionsDrift",
    "AgentOptions",
    
    # Agent types
    "Agent",
    "MBReward",
    "MFReward", 
    "Bias",
    "Perseveration",
    "ContextRL",
    
    # Data types
    "GenericData",
    "TwoStepData",
    "DynamicRoutingData",
    
    # Core functions
    "optimize",
    "simulate", 
    "choice_accuracy",
    "load_generic_data",
    "fit_dynamic_routing_model",
    "convert_from_dynamic_routing",
    "create_dynamic_routing_agents",
    
    # Analysis functions
    "cross_validate",
    "parameter_recovery", 
    "model_compare",
    
    # Utility functions
    "smooth",
    "onehot",
    "compute_agent_strings",
    "beta_title",
    "alpha_title",
    
    # Plotting functions
    "plot_model",
    "plot_recovery",
    "plot_comparison",
]
