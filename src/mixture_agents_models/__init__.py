"""
Mixture Agents Models: Python implementation of MoA-HMM framework.

This package provides computational models for analyzing choice behavior using
mixture-of-agents frameworks with hidden Markov model transitions.
"""

from mixture_agents_models.agents import Agent, Bias, ContextRL, MBReward, MFReward, Perseveration
from mixture_agents_models.integration import (
    convert_from_dynamic_routing,
    create_dynamic_routing_agents,
    fit_dynamic_routing_model,
)
from mixture_agents_models.models import (
    AgentOptions,
    ModelDrift,
    ModelHMM,
    ModelOptionsDrift,
    ModelOptionsHMM,
    choice_accuracy,
    optimize,
    simulate,
)
from mixture_agents_models.plotting import plot_comparison, plot_model, plot_recovery
from mixture_agents_models.tasks import DynamicRoutingData, GenericData, TwoStepData, load_generic_data
from mixture_agents_models.utils import (
    alpha_title,
    beta_title,
    compute_agent_strings,
    cross_validate,
    model_compare,
    onehot,
    parameter_recovery,
    smooth,
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
