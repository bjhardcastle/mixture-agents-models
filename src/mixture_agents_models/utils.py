"""
Utility functions for mixture agents models.

This module provides helper functions for data processing, 
model analysis, cross-validation, and parameter recovery.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import scipy.stats

from .tasks import GenericData
from .models import ModelHMM, ModelOptionsHMM, AgentOptions
from .agents import Agent


def smooth(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to data.
    
    Args:
        data: Input data array
        window_size: Size of smoothing window
        
    Returns:
        Smoothed data array
    """
    if len(data) < window_size:
        return data
    
    # Pad data at edges
    padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed


def cross_validate(
    data: GenericData,
    model_options: ModelOptionsHMM,
    agent_options: AgentOptions,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation for model evaluation.
    
    Args:
        data: Behavioral data
        model_options: Model configuration
        agent_options: Agent configuration
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with cross-validation metrics
    """
    if data.n_sessions < n_folds:
        raise ValueError(f"Need at least {n_folds} sessions for {n_folds}-fold CV")
    
    # Create session-based folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    session_indices = np.unique(data.session_indices)
    
    results = {
        'train_ll': np.zeros(n_folds),
        'test_ll': np.zeros(n_folds),
        'train_accuracy': np.zeros(n_folds),
        'test_accuracy': np.zeros(n_folds)
    }
    
    for fold, (train_sessions, test_sessions) in enumerate(kf.split(session_indices)):
        print(f"Cross-validation fold {fold + 1}/{n_folds}")
        
        # Create train/test masks
        train_mask = np.isin(data.session_indices, session_indices[train_sessions])
        test_mask = np.isin(data.session_indices, session_indices[test_sessions])
        
        # Split data
        train_data = _subset_data(data, train_mask)
        test_data = _subset_data(data, test_mask)
        
        try:
            # Import here to avoid circular imports
            from .models import optimize, choice_accuracy, compute_log_likelihood
            
            # Fit model on training data
            model, agents, train_ll = optimize(
                data=train_data,
                model_options=model_options,
                agent_options=agent_options,
                verbose=False
            )
            
            # Evaluate on training data
            train_accuracy = choice_accuracy(model, agents, train_data)
            
            # Evaluate on test data
            test_accuracy = choice_accuracy(model, agents, test_data)
            test_ll = compute_log_likelihood(model, agents, test_data)
            
            results['train_ll'][fold] = train_ll
            results['test_ll'][fold] = test_ll
            results['train_accuracy'][fold] = train_accuracy
            results['test_accuracy'][fold] = test_accuracy
            
        except Exception as e:
            print(f"  Fold {fold + 1} failed: {e}")
            results['train_ll'][fold] = np.nan
            results['test_ll'][fold] = np.nan
            results['train_accuracy'][fold] = np.nan
            results['test_accuracy'][fold] = np.nan
    
    return results


def model_compare(
    data: GenericData,
    model_configs: List[Tuple[ModelOptionsHMM, AgentOptions]],
    model_names: Optional[List[str]] = None,
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        data: Behavioral data
        model_configs: List of (model_options, agent_options) tuples
        model_names: Optional names for models
        cv_folds: Number of cross-validation folds
        
    Returns:
        DataFrame with comparison results
    """
    if data.n_trials < 10:
        # Too few trials for meaningful comparison
        print("Warning: Very few trials for model comparison")
    
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_configs))]
    
    results = []
    
    for i, (model_options, agent_options) in enumerate(model_configs):
        print(f"\nEvaluating {model_names[i]}...")
        
        try:
            cv_results = cross_validate(
                data=data,
                model_options=model_options,
                agent_options=agent_options,
                n_folds=cv_folds
            )
            
            # Compute summary statistics
            mean_test_ll = np.nanmean(cv_results['test_ll'])
            std_test_ll = np.nanstd(cv_results['test_ll'])
            mean_test_acc = np.nanmean(cv_results['test_accuracy'])
            std_test_acc = np.nanstd(cv_results['test_accuracy'])
            
            # Count parameters for complexity penalty
            n_params = _count_model_parameters(model_options, agent_options)
            aic = -2 * mean_test_ll + 2 * n_params
            bic = -2 * mean_test_ll + np.log(data.n_trials) * n_params
            
            results.append({
                'model': model_names[i],
                'mean_log_likelihood': mean_test_ll,
                'std_log_likelihood': std_test_ll,
                'mean_accuracy': mean_test_acc,
                'std_accuracy': std_test_acc,
                'n_parameters': n_params,
                'aic': aic,
                'bic': bic
            })
            
        except Exception as e:
            print(f"  Failed to evaluate {model_names[i]}: {e}")
            results.append({
                'model': model_names[i],
                'mean_log_likelihood': np.nan,
                'std_log_likelihood': np.nan,
                'mean_accuracy': np.nan,
                'std_accuracy': np.nan,
                'n_parameters': np.nan,
                'aic': np.nan,
                'bic': np.nan
            })
    
    return pd.DataFrame(results)


def parameter_recovery(
    true_agents: List[Agent],
    true_model: ModelHMM,
    n_trials: int = 1000,
    n_sims: int = 10,
    model_options: Optional[ModelOptionsHMM] = None,
    noise_level: float = 0.1
) -> Dict[str, Any]:
    """
    Test parameter recovery by fitting to simulated data.
    
    Args:
        true_agents: True agent parameters
        true_model: True model parameters
        n_trials: Number of trials per simulation
        n_sims: Number of simulations
        model_options: Model fitting options
        noise_level: Amount of noise to add to simulations
        
    Returns:
        Dictionary with recovery results
    """
    from .models import simulate, optimize, AgentOptions
    
    if model_options is None:
        model_options = ModelOptionsHMM()
    
    # Create agent options that match true agents
    agent_options = AgentOptions(
        agents=true_agents
    )
    
    recovery_results = {
        'true_beta': [],
        'recovered_beta': [],
        'true_pi': [],
        'recovered_pi': [],
        'true_A': [],
        'recovered_A': [],
        'log_likelihoods': [],
        'correlations': {},
        'mse': {}
    }
    
    for sim in range(n_sims):
        print(f"Parameter recovery simulation {sim + 1}/{n_sims}")
        
        try:
            # Simulate data from true parameters
            sim_data = simulate(
                agents=true_agents,
                model=true_model,
                n_trials=n_trials,
                noise_level=noise_level
            )
            
            # Fit model to simulated data
            recovered_model, recovered_agents, ll = optimize(
                data=sim_data,
                model_options=model_options,
                agent_options=agent_options,
                verbose=False
            )
            
            # Store true and recovered parameters
            recovery_results['true_beta'].append([agent.beta for agent in true_agents])
            recovery_results['recovered_beta'].append([agent.beta for agent in recovered_agents])
            
            recovery_results['true_pi'].append(true_model.pi)
            recovery_results['recovered_pi'].append(recovered_model.pi)
            
            recovery_results['true_A'].append(true_model.A)
            recovery_results['recovered_A'].append(recovered_model.A)
            
            recovery_results['log_likelihoods'].append(ll)
            
        except Exception as e:
            print(f"  Simulation {sim + 1} failed: {e}")
    
    # Compute recovery statistics
    recovery_results['correlations'] = _compute_recovery_correlations(recovery_results)
    recovery_results['mse'] = _compute_recovery_mse(recovery_results)
    
    return recovery_results


def _subset_data(data: GenericData, mask: np.ndarray) -> GenericData:
    """Create a subset of data based on boolean mask."""
    return GenericData(
        choices=data.choices[mask],
        rewards=data.rewards[mask],
        n_trials=int(np.sum(mask)),
        n_sessions=len(np.unique(data.session_indices[mask])),
        session_indices=data.session_indices[mask],
        trial_types=data.trial_types[mask] if data.trial_types is not None else None,
        contexts=data.contexts[mask] if data.contexts is not None else None,
        stimuli=data.stimuli[mask] if data.stimuli is not None else None,
        reaction_times=data.reaction_times[mask] if data.reaction_times is not None else None,
        metadata=data.metadata.copy()
    )


def _count_model_parameters(model_options: ModelOptionsHMM, agent_options: AgentOptions) -> int:
    """Count the number of free parameters in the model."""
    n_params = 0
    
    # Agent parameters (beta for each agent)
    n_params += agent_options.n_agents
    
    # Model parameters
    n_states = model_options.n_states
    
    # Initial state probabilities (n_states - 1 free parameters)
    n_params += n_states - 1
    
    # Transition matrix (n_states * (n_states - 1) free parameters)
    n_params += n_states * (n_states - 1)
    
    return n_params


def _compute_recovery_correlations(recovery_results: Dict[str, List]) -> Dict[str, float]:
    """Compute correlations between true and recovered parameters."""
    correlations = {}
    
    for param in ['beta', 'pi', 'A']:
        true_key = f'true_{param}'
        recovered_key = f'recovered_{param}'
        
        if true_key in recovery_results and len(recovery_results[true_key]) > 0:
            true_params = np.concatenate([np.array(p).flatten() for p in recovery_results[true_key]])
            recovered_params = np.concatenate([np.array(p).flatten() for p in recovery_results[recovered_key]])
            
            if len(true_params) > 0 and len(recovered_params) > 0:
                correlations[param] = scipy.stats.pearsonr(true_params, recovered_params)[0]
            else:
                correlations[param] = np.nan
    
    return correlations


def _compute_recovery_mse(recovery_results: Dict[str, List]) -> Dict[str, float]:
    """Compute mean squared error between true and recovered parameters."""
    mse = {}
    
    for param in ['beta', 'pi', 'A']:
        true_key = f'true_{param}'
        recovered_key = f'recovered_{param}'
        
        if true_key in recovery_results and len(recovery_results[true_key]) > 0:
            true_params = np.concatenate([np.array(p).flatten() for p in recovery_results[true_key]])
            recovered_params = np.concatenate([np.array(p).flatten() for p in recovery_results[recovered_key]])
            
            if len(true_params) > 0 and len(recovered_params) > 0:
                mse[param] = np.mean((true_params - recovered_params) ** 2)
            else:
                mse[param] = np.nan
    
    return mse


def compute_model_evidence(
    data: GenericData,
    model_options: ModelOptionsHMM,
    agent_options: AgentOptions,
    n_samples: int = 1000
) -> Dict[str, float]:
    """
    Compute model evidence using importance sampling.
    
    Args:
        data: Behavioral data
        model_options: Model configuration
        agent_options: Agent configuration
        n_samples: Number of importance samples
        
    Returns:
        Dictionary with evidence estimates
    """
    from .models import optimize, compute_log_likelihood
    
    # Fit model to get MAP estimate
    model, agents, map_ll = optimize(
        data=data,
        model_options=model_options,
        agent_options=agent_options
    )
    
    # Simple approximation using BIC
    n_params = _count_model_parameters(model_options, agent_options)
    bic = -2 * map_ll + np.log(data.n_trials) * n_params
    
    return {
        'map_log_likelihood': map_ll,
        'bic': bic,
        'log_model_evidence_approx': -0.5 * bic
    }


def onehot(x: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """Convert integer array to one-hot encoding."""
    if n_classes is None:
        n_classes = int(x.max()) + 1
    
    n_samples = len(x)
    onehot_matrix = np.zeros((n_samples, n_classes))
    onehot_matrix[np.arange(n_samples), x.astype(int)] = 1
    
    return onehot_matrix


def compute_agent_strings(agents: List[Agent]) -> List[str]:
    """Get string representations of agent types."""
    return [type(agent).__name__ for agent in agents]


def beta_title(agent: Agent) -> str:
    """Get beta parameter title for plotting."""
    if hasattr(agent, 'beta_title'):
        return agent.beta_title()
    else:
        return f"β({type(agent).__name__})"


def alpha_title(agent: Agent) -> str:
    """Get alpha parameter title for plotting."""
    if hasattr(agent, 'alpha_title'):
        return agent.alpha_title()
    elif hasattr(agent, 'alpha'):
        return f"α({type(agent).__name__})"
    else:
        return f"param({type(agent).__name__})"
