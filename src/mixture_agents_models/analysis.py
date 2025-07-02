"""
Model analysis and comparison functions.

This module provides functions for systematic model analysis including
agent comparison studies and model selection tools.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import numpy.typing as npt

from mixture_agents_models.agents import Agent
from mixture_agents_models.models import (
    AgentOptions,
    ModelOptionsHMM,
    ModelOptionsDrift
)


def agents_comparison(
    model_options: ModelOptionsHMM | ModelOptionsDrift,
    agent_options: AgentOptions,
) -> tuple[list[ModelOptionsHMM | ModelOptionsDrift], list[AgentOptions]]:
    """
    Create model configurations by systematically leaving out each agent.
    
    Implements the Julia agents_comparison function that creates a set of models
    where each model excludes one agent from the original configuration. The first
    model includes all agents (original model), and subsequent models systematically
    remove each agent one at a time.
    
    Args:
        model_options: Base model configuration
        agent_options: Base agent configuration
        
    Returns:
        Tuple of (model_options_list, agent_options_list) where:
        - model_options_list[0] is the original model with all agents
        - model_options_list[i] is the model with agent i-1 removed
        - agent_options_list follows the same pattern
    """
    agents = agent_options.agents
    n_agents = len(agents)
    n_models = n_agents + 1
    
    # Initialize output arrays
    model_ops = []
    agent_ops = []
    
    # First model: original configuration with all agents
    model_ops.append(copy.deepcopy(model_options))
    agent_ops.append(copy.deepcopy(agent_options))
    
    # Create models with each agent removed
    for agent_idx in range(n_agents):
        # Remove agent from list
        agents_tmp = [agent for i, agent in enumerate(agents) if i != agent_idx]
        
        # Handle parameter configuration updates
        if agent_options.fit_symbols is not None and agent_options.fit_params is not None:
            fit_params_tmp = [p for i, p in enumerate(agent_options.fit_params) if i != agent_idx]
            
            # Adjust parameter indices after agent removal
            fit_params_adjusted = []
            for p in fit_params_tmp:
                if p > agent_idx + 1:  # Convert to 1-based indexing for comparison
                    fit_params_adjusted.append(p - 1)
                elif p == agent_idx + 1:
                    fit_params_adjusted.append(0)  # Mark as non-fitted
                else:
                    fit_params_adjusted.append(p)
            
            # Get unique parameter indices that are still being fit
            unique_params = sorted(set(p for p in fit_params_adjusted if p > 0))
            
            if unique_params:
                # Create new parameter mapping
                param_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_params, 1)}
                fit_params_final = [param_map.get(p, 0) for p in fit_params_adjusted]
                
                # Update symbols to match remaining parameters
                fit_symbols_tmp = []
                fit_scales_tmp = []
                fit_priors_tmp = []
                
                for param_idx in unique_params:
                    if param_idx <= len(agent_options.fit_symbols):
                        fit_symbols_tmp.append(agent_options.fit_symbols[param_idx - 1])
                        if agent_options.fit_scales and param_idx <= len(agent_options.fit_scales):
                            fit_scales_tmp.append(agent_options.fit_scales[param_idx - 1])
                        if agent_options.fit_priors and param_idx <= len(agent_options.fit_priors):
                            fit_priors_tmp.append(agent_options.fit_priors[param_idx - 1])
            else:
                # No parameters left to fit
                fit_symbols_tmp = None
                fit_params_final = None
                fit_scales_tmp = None
                fit_priors_tmp = None
        else:
            # No fitting parameters in original configuration
            fit_symbols_tmp = None
            fit_params_final = None
            fit_scales_tmp = None
            fit_priors_tmp = None
        
        # Create new agent options
        new_agent_options = AgentOptions(
            agents=agents_tmp,
            fit_symbols=fit_symbols_tmp,
            fit_params=fit_params_final,
            fit_scales=fit_scales_tmp or [],
            fit_priors=fit_priors_tmp or [],
            scale_x=agent_options.scale_x
        )
        agent_ops.append(new_agent_options)
        
        # Update model options if needed
        if hasattr(model_options, 'beta_0') and isinstance(model_options.beta_0, np.ndarray):
            if model_options.beta_0.ndim > 1 and model_options.beta_0.shape[0] > 1:
                # Remove corresponding agent dimension from beta_0
                beta_0_tmp = np.delete(model_options.beta_0, agent_idx, axis=0)
                new_model_options = copy.deepcopy(model_options)
                new_model_options.beta_0 = beta_0_tmp
            else:
                new_model_options = copy.deepcopy(model_options)
        else:
            new_model_options = copy.deepcopy(model_options)
        
        model_ops.append(new_model_options)
    
    return model_ops, agent_ops


def n_states_comparison(
    base_options: ModelOptionsHMM,
    n_states_range: range | list[int] = range(1, 7)
) -> list[ModelOptionsHMM]:
    """
    Create model configurations with different numbers of latent states.
    
    Args:
        base_options: Base model configuration
        n_states_range: Range of state numbers to test
        
    Returns:
        List of model options with varying numbers of states
    """
    if isinstance(n_states_range, range):
        n_states_range = list(n_states_range)
    
    model_ops = []
    
    for n_states in n_states_range:
        new_options = copy.deepcopy(base_options)
        new_options.n_states = n_states
        model_ops.append(new_options)
    
    return model_ops


def get_fit_params_structure(agent_options: AgentOptions) -> list[int]:
    """
    Reconstruct the original fit_params structure from processed AgentOptions.
    
    This function helps restore the user-input-formatted fit_params for
    compatibility with agent comparison functions.
    
    Args:
        agent_options: Processed agent options
        
    Returns:
        Original fit_params structure
    """
    if agent_options.fit_params is None:
        return []
    
    n_agents = len(agent_options.agents)
    fit_params = [0] * n_agents
    
    if hasattr(agent_options, '_param_to_agents'):
        # Use the parameter sharing mapping
        for param_idx, agent_indices in agent_options._param_to_agents.items():
            for agent_idx in agent_indices:
                if agent_idx < n_agents:
                    fit_params[agent_idx] = param_idx
    else:
        # Fallback to simple mapping
        for i, agent_idx in enumerate(agent_options.fit_indices or []):
            if agent_idx < n_agents and i < len(agent_options.fit_params):
                fit_params[agent_idx] = agent_options.fit_params[i]
    
    return fit_params
