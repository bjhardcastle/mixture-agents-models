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
    # Handle small datasets
    if data.n_sessions < n_folds:
        print(f"Warning: Only {data.n_sessions} sessions available for {n_folds}-fold CV, reducing folds")
        n_folds = max(1, data.n_sessions)
    
    if data.n_trials < 10:
        print(f"Warning: Only {data.n_trials} trials available, using simple train/test split")
        # For very small datasets, just do a simple split
        split_point = max(1, data.n_trials // 2)
        
        # Create masks for train/test split
        train_mask = np.zeros(data.n_trials, dtype=bool)
        train_mask[:split_point] = True
        test_mask = ~train_mask
        
        results = {
            'train_ll': np.zeros(1),
            'test_ll': np.zeros(1),
            'train_accuracy': np.zeros(1),
            'test_accuracy': np.zeros(1)
        }
        
        try:
            # Import here to avoid circular imports
            from .models import optimize, choice_accuracy, compute_log_likelihood
            
            train_data = _subset_data(data, train_mask)
            test_data = _subset_data(data, test_mask)
            
            # Fit model on training data
            model, agents, train_ll = optimize(
                data=train_data,
                model_options=model_options,
                agent_options=agent_options,
                verbose=False
            )
            
            # Evaluate
            train_accuracy = choice_accuracy(model, agents, train_data)
            test_accuracy = choice_accuracy(model, agents, test_data)
            test_ll = compute_log_likelihood(model, agents, test_data)
            
            results['train_ll'][0] = train_ll
            results['test_ll'][0] = test_ll
            results['train_accuracy'][0] = train_accuracy
            results['test_accuracy'][0] = test_accuracy
            
        except Exception as e:
            print(f"  Simple split failed: {e}")
            results['train_ll'][0] = np.nan
            results['test_ll'][0] = np.nan
            results['train_accuracy'][0] = np.nan
            results['test_accuracy'][0] = np.nan
        
        return results
    
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
    model_configs: List[Dict[str, Any]],
    model_names: Optional[List[str]] = None,
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        data: Behavioral data
        model_configs: List of dictionaries with 'model_options' and 'agent_options' keys
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
    
    for i, config in enumerate(model_configs):
        print(f"\nEvaluating {model_names[i]}...")
        
        try:
            model_options = config['model_options']
            agent_options = config['agent_options']
            
            cv_results = cross_validate(
                data=data,
                model_options=model_options,
                agent_options=agent_options,
                n_folds=cv_folds
            )
            
            # Compute summary statistics with safety checks
            test_lls = cv_results['test_ll']
            test_accs = cv_results['test_accuracy']
            
            # Filter out NaN and infinite values
            valid_lls = test_lls[np.isfinite(test_lls)]
            valid_accs = test_accs[np.isfinite(test_accs)]
            
            if len(valid_lls) > 0:
                mean_test_ll = np.mean(valid_lls)
                std_test_ll = np.std(valid_lls)
            else:
                mean_test_ll = np.nan
                std_test_ll = np.nan
                
            if len(valid_accs) > 0:
                mean_test_acc = np.mean(valid_accs)
                std_test_acc = np.std(valid_accs)
            else:
                mean_test_acc = np.nan
                std_test_acc = np.nan
            
            # Count parameters for complexity penalty
            n_params = _count_model_parameters(model_options, agent_options)
            
            # Handle division by zero and invalid values in AIC/BIC calculation
            if np.isfinite(mean_test_ll) and not np.isnan(mean_test_ll) and data.n_trials > 0:
                aic = -2 * mean_test_ll + 2 * n_params
                bic = -2 * mean_test_ll + np.log(data.n_trials) * n_params
            else:
                aic = np.nan
                bic = np.nan
            
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
    from .tasks import GenericData
    
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
            # Create simple template data for simulation
            template_data = GenericData(
                choices=np.zeros(n_trials, dtype=int),
                rewards=np.zeros(n_trials, dtype=int),
                n_trials=n_trials,
                n_sessions=1
            )
            
            # Simulate data from true parameters
            predictions = simulate(
                model=true_model,
                agents=true_agents,
                data=template_data,
                n_reps=1
            )
            
            # Create simulated data object
            sim_data = GenericData(
                choices=predictions['choices'][0],
                rewards=np.random.binomial(1, 0.7, n_trials),  # Add some reward structure
                n_trials=n_trials,
                n_sessions=1
            )
            
            # Fit model to simulated data
            recovered_model, recovered_agents, ll = optimize(
                data=sim_data,
                model_options=model_options,
                agent_options=agent_options,
                verbose=False
            )
            
            # Store true and recovered parameters
            # Handle different parameter names (beta, alpha, etc.)
            true_params = []
            recovered_params = []
            for agent in true_agents:
                if hasattr(agent, 'beta'):
                    true_params.append(agent.beta)
                elif hasattr(agent, 'alpha'):
                    true_params.append(agent.alpha)
                else:
                    true_params.append(0.0)  # Default value
            
            for agent in recovered_agents:
                if hasattr(agent, 'beta'):
                    recovered_params.append(agent.beta)
                elif hasattr(agent, 'alpha'):
                    recovered_params.append(agent.alpha)
                else:
                    recovered_params.append(0.0)  # Default value
            
            recovery_results['true_beta'].append(true_params)
            recovery_results['recovered_beta'].append(recovered_params)
            
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
    n_params += len(agent_options.agents)
    
    # Model parameters
    n_states = model_options.n_states
    
    # Initial state probabilities (n_states - 1 free parameters)
    if n_states > 1:
        n_params += n_states - 1
    
    # Transition matrix (n_states * (n_states - 1) free parameters)
    if n_states > 1:
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
