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

from .models import ModelHMM, ModelOptionsHMM, AgentOptions, optimize, simulate
from .agents import Agent
from .tasks import GenericData


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
        
        # Split data by sessions
        train_mask = np.isin(data.session_indices, session_indices[train_sessions])
        test_mask = np.isin(data.session_indices, session_indices[test_sessions])
        
        train_data = _subset_data(data, train_mask)
        test_data = _subset_data(data, test_mask)
        
        # Fit model on training data
        model, agents, train_ll = optimize(
            data=train_data,
            model_options=model_options,
            agent_options=agent_options,
            verbose=False
        )
        
        # Evaluate on test data
        from .models import compute_log_likelihood, choice_accuracy
        test_ll = compute_log_likelihood(model, agents, test_data)
        
        train_accuracy = choice_accuracy(model, agents, train_data)
        test_accuracy = choice_accuracy(model, agents, test_data)
        
        results['train_ll'][fold] = train_ll
        results['test_ll'][fold] = test_ll
        results['train_accuracy'][fold] = train_accuracy
        results['test_accuracy'][fold] = test_accuracy
    
    return results


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
        n_trials: Number of trials to simulate
        n_sims: Number of simulation repetitions
        model_options: Fitting options (uses defaults if None)
        noise_level: Amount of noise to add to parameters
        
    Returns:
        Dictionary with recovery results
    """
    if model_options is None:
        model_options = ModelOptionsHMM(n_states=len(true_model.pi), verbose=False)
    
    recovery_results = {
        'true_beta': [],
        'recovered_beta': [],
        'true_pi': [],
        'recovered_pi': [],
        'true_A': [],
        'recovered_A': [],
        'correlations': {},
        'mse': {}
    }
    
    for sim in range(n_sims):
        print(f"Parameter recovery simulation {sim + 1}/{n_sims}")
        
        # Create dummy data structure for simulation
        dummy_data = GenericData(
            choices=np.zeros(n_trials, dtype=int),
            rewards=np.zeros(n_trials, dtype=float),
            n_trials=n_trials,
            n_sessions=1
        )
        
        # Simulate data from true model
        sim_results = simulate(true_model, true_agents, dummy_data, n_reps=1)
        
        # Create data object with simulated choices
        sim_data = GenericData(
            choices=sim_results['choices'][0],
            rewards=np.random.binomial(1, 0.7, n_trials),  # Random rewards
            n_trials=n_trials,
            n_sessions=1
        )
        
        # Add noise to true parameters for initialization
        noisy_agents = _add_noise_to_agents(true_agents, noise_level)
        agent_options = AgentOptions(agents=noisy_agents)
        
        # Fit model to simulated data
        try:
            recovered_model, recovered_agents, _ = optimize(
                data=sim_data,
                model_options=model_options,
                agent_options=agent_options,
                verbose=False
            )
            
            # Store results
            recovery_results['true_beta'].append(true_model.beta)
            recovery_results['recovered_beta'].append(recovered_model.beta)
            recovery_results['true_pi'].append(true_model.pi)
            recovery_results['recovered_pi'].append(recovered_model.pi)
            recovery_results['true_A'].append(true_model.A)
            recovery_results['recovered_A'].append(recovered_model.A)
            
        except Exception as e:
            print(f"  Simulation {sim + 1} failed: {e}")
    
    # Compute recovery statistics
    if len(recovery_results['true_beta']) > 0:
        recovery_results['correlations'] = _compute_recovery_correlations(recovery_results)
        recovery_results['mse'] = _compute_recovery_mse(recovery_results)
    
    return recovery_results


def model_compare(
    data: GenericData,
    model_configs: List[Dict[str, Any]],
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare different model configurations using information criteria.
    
    Args:
        data: Behavioral data
        model_configs: List of model configuration dictionaries
        model_names: Names for each model
        
    Returns:
        DataFrame with model comparison results
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_configs))]
    
    results = []
    
    for i, config in enumerate(model_configs):
        print(f"Fitting {model_names[i]}...")
        
        try:
            model, agents, log_likelihood = optimize(
                data=data,
                model_options=config['model_options'],
                agent_options=config['agent_options'],
                verbose=False
            )
            
            # Calculate model complexity
            n_states = len(model.pi)
            n_agents = len(agents)
            n_params = n_agents * n_states + n_states + n_states * (n_states - 1)
            
            # Calculate information criteria
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(data.n_trials) * n_params
            
            from .models import choice_accuracy
            accuracy = choice_accuracy(model, agents, data)
            
            results.append({
                'model': model_names[i],
                'log_likelihood': log_likelihood,
                'n_params': n_params,
                'aic': aic,
                'bic': bic,
                'accuracy': accuracy,
                'n_states': n_states,
                'n_agents': n_agents
            })
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'model': model_names[i],
                'log_likelihood': np.nan,
                'n_params': np.nan,
                'aic': np.nan,
                'bic': np.nan,
                'accuracy': np.nan,
                'error': str(e)
            })
    
    df = pd.DataFrame(results)
    
    # Add model rankings
    for metric in ['aic', 'bic']:
        if metric in df.columns:
            df[f'{metric}_rank'] = df[metric].rank()
    
    return df.sort_values('aic')


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing."""
    if window <= 1:
        return x
    
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


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


def _subset_data(data: GenericData, mask: np.ndarray) -> GenericData:
    """Extract subset of data based on boolean mask."""
    return GenericData(
        choices=data.choices[mask],
        rewards=data.rewards[mask],
        n_trials=int(mask.sum()),
        n_sessions=len(np.unique(data.session_indices[mask])),
        session_indices=data.session_indices[mask] if data.session_indices is not None else None,
        trial_types=data.trial_types[mask] if data.trial_types is not None else None,
        contexts=data.contexts[mask] if data.contexts is not None else None,
        stimuli=data.stimuli[mask] if data.stimuli is not None else None,
        reaction_times=data.reaction_times[mask] if data.reaction_times is not None else None,
        metadata=data.metadata.copy()
    )


def _add_noise_to_agents(agents: List[Agent], noise_level: float) -> List[Agent]:
    """Add noise to agent parameters for parameter recovery testing."""
    noisy_agents = []
    
    for agent in agents:
        # Create copy with noisy parameters
        params = agent.get_params()
        noisy_params = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float)) and 'scale' not in key and 'prior' not in key:
                # Add Gaussian noise
                noisy_value = value + np.random.normal(0, noise_level * abs(value))
                # Clip to reasonable bounds
                if 'alpha' in key:
                    noisy_value = np.clip(noisy_value, 0.01, 0.99)
                noisy_params[key] = noisy_value
            else:
                noisy_params[key] = value
        
        noisy_agents.append(agent.set_params(**noisy_params))
    
    return noisy_agents


def _compute_recovery_correlations(recovery_results: Dict[str, List]) -> Dict[str, float]:
    """Compute correlations between true and recovered parameters."""
    correlations = {}
    
    for param in ['beta', 'pi', 'A']:
        true_key = f'true_{param}'
        recovered_key = f'recovered_{param}'
        
        if true_key in recovery_results and len(recovery_results[true_key]) > 0:
            true_params = np.concatenate([p.flatten() for p in recovery_results[true_key]])
            recovered_params = np.concatenate([p.flatten() for p in recovery_results[recovered_key]])
            
            correlations[param] = scipy.stats.pearsonr(true_params, recovered_params)[0]
    
    return correlations


def _compute_recovery_mse(recovery_results: Dict[str, List]) -> Dict[str, float]:
    """Compute mean squared error between true and recovered parameters."""
    mse = {}
    
    for param in ['beta', 'pi', 'A']:
        true_key = f'true_{param}'
        recovered_key = f'recovered_{param}'
        
        if true_key in recovery_results and len(recovery_results[true_key]) > 0:
            true_params = np.concatenate([p.flatten() for p in recovery_results[true_key]])
            recovered_params = np.concatenate([p.flatten() for p in recovery_results[recovered_key]])
            
            mse[param] = np.mean((true_params - recovered_params) ** 2)
    
    return mse
