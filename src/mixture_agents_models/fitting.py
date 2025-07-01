"""
EM algorithm implementation for mixture-of-agents HMM fitting.

This module provides the core expectation-maximization algorithm
for parameter estimation in mixture-of-agents hidden Markov models.
"""

from typing import Tuple, List, Optional
import numpy as np
import scipy.special
import scipy.optimize
from sklearn.preprocessing import StandardScaler

from .models import ModelHMM, ModelOptionsHMM, AgentOptions
from .agents import Agent
from .tasks import GenericData


def fit_hmm_em(
    y: np.ndarray,
    x: np.ndarray,
    data: GenericData,
    model_options: ModelOptionsHMM,
    agent_options: AgentOptions,
    verbose: bool = True
) -> Tuple[ModelHMM, List[Agent], float]:
    """
    Fit HMM using expectation-maximization algorithm.
    
    Args:
        y: Choice matrix (n_trials x n_choices)
        x: Design matrix (n_trials x n_agents)  
        data: Behavioral data
        model_options: Model configuration
        agent_options: Agent configuration
        verbose: Whether to print progress
        
    Returns:
        Tuple of (fitted_model, fitted_agents, log_likelihood)
    """
    n_trials, n_agents = x.shape
    n_states = model_options.n_states
    
    best_ll = -np.inf
    best_model = None
    best_agents = None
    
    for start in range(model_options.n_starts):
        if verbose:
            print(f"EM start {start + 1}/{model_options.n_starts}")
        
        # Initialize parameters
        model = _initialize_hmm_parameters(model_options, n_agents)
        agents = _initialize_agents(agent_options)
        
        # EM iterations
        ll_prev = -np.inf
        
        for iteration in range(model_options.max_iter):
            # E-step: compute posterior state probabilities
            log_alpha, log_beta, log_gamma, log_xi = _forward_backward(
                y, x, model, agents
            )
            
            # M-step: update parameters
            model = _update_hmm_parameters(
                model, log_gamma, log_xi, model_options
            )
            
            agents = _update_agent_parameters(
                agents, y, x, log_gamma, agent_options
            )
            
            # Compute log-likelihood
            ll = scipy.special.logsumexp(log_alpha[-1])
            
            if verbose and (iteration + 1) % model_options.disp_iter == 0:
                print(f"  Iteration {iteration + 1}: LL = {ll:.4f}")
            
            # Check convergence
            if ll - ll_prev < model_options.tol:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
            
            ll_prev = ll
        
        # Track best fit across starts
        if ll > best_ll:
            best_ll = ll
            best_model = model
            best_agents = agents
    
    return best_model, best_agents, best_ll


def _initialize_hmm_parameters(
    options: ModelOptionsHMM, 
    n_agents: int
) -> ModelHMM:
    """Initialize HMM parameters."""
    n_states = options.n_states
    
    # Initialize beta (agent weights)
    if options.beta_0 is not None:
        if options.beta_0.shape == (1, n_states):
            # Broadcast single row to all agents
            beta = np.tile(options.beta_0, (n_agents, 1))
        else:
            beta = options.beta_0.copy()
    else:
        # Random initialization
        beta = np.random.normal(0, 1, (n_agents, n_states))
    
    # Initialize pi (initial state probabilities)
    pi = options.pi_0.copy() if options.pi_0 is not None else \
         np.ones(n_states) / n_states
    
    # Initialize A (transition matrix)
    A = options.A_0.copy() if options.A_0 is not None else \
        np.eye(n_states) * 0.9 + (1 - np.eye(n_states)) * 0.1 / (n_states - 1)
    
    return ModelHMM(beta=beta, pi=pi, A=A)


def _initialize_agents(options: AgentOptions) -> List[Agent]:
    """Initialize agent parameters."""
    # For now, return copy of input agents
    # In full implementation, would handle parameter initialization
    return [agent for agent in options.agents]


def _forward_backward(
    y: np.ndarray,
    x: np.ndarray, 
    model: ModelHMM,
    agents: List[Agent]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-backward algorithm for HMM inference.
    
    Returns:
        log_alpha: Forward probabilities
        log_beta: Backward probabilities  
        log_gamma: State probabilities
        log_xi: Transition probabilities
    """
    n_trials, n_states = y.shape[0], len(model.pi)
    
    # Forward pass
    log_alpha = np.zeros((n_trials, n_states))
    
    # Initial step
    for s in range(n_states):
        emission_ll = _compute_emission_log_likelihood(y[0], x[0], model.beta[:, s])
        log_alpha[0, s] = np.log(model.pi[s]) + emission_ll
    
    # Forward recursion
    for t in range(1, n_trials):
        for s in range(n_states):
            emission_ll = _compute_emission_log_likelihood(y[t], x[t], model.beta[:, s])
            log_alpha[t, s] = emission_ll + \
                            scipy.special.logsumexp(log_alpha[t-1] + np.log(model.A[:, s]))
    
    # Backward pass
    log_beta = np.zeros((n_trials, n_states))
    log_beta[-1, :] = 0  # Log of 1
    
    for t in range(n_trials - 2, -1, -1):
        for s in range(n_states):
            next_emissions = np.array([
                _compute_emission_log_likelihood(y[t+1], x[t+1], model.beta[:, s_next])
                for s_next in range(n_states)
            ])
            log_beta[t, s] = scipy.special.logsumexp(
                np.log(model.A[s, :]) + next_emissions + log_beta[t+1, :]
            )
    
    # Compute state and transition probabilities
    log_likelihood = scipy.special.logsumexp(log_alpha[-1])
    
    log_gamma = log_alpha + log_beta - log_likelihood
    
    log_xi = np.zeros((n_trials - 1, n_states, n_states))
    for t in range(n_trials - 1):
        for s1 in range(n_states):
            for s2 in range(n_states):
                emission_ll = _compute_emission_log_likelihood(
                    y[t+1], x[t+1], model.beta[:, s2]
                )
                log_xi[t, s1, s2] = log_alpha[t, s1] + \
                                   np.log(model.A[s1, s2]) + \
                                   emission_ll + \
                                   log_beta[t+1, s2] - \
                                   log_likelihood
    
    return log_alpha, log_beta, log_gamma, log_xi


def _compute_emission_log_likelihood(
    y_t: np.ndarray, 
    x_t: np.ndarray, 
    beta_s: np.ndarray
) -> float:
    """Compute emission log-likelihood for a single trial and state."""
    logit = np.dot(x_t, beta_s)
    
    # Sigmoid choice probability  
    choice_prob = 1 / (1 + np.exp(-logit))
    choice_prob = np.clip(choice_prob, 1e-10, 1 - 1e-10)  # Avoid log(0)
    
    # Binary choice likelihood
    if y_t[0] == 1:  # Assuming binary choice in first position
        return np.log(choice_prob)
    else:
        return np.log(1 - choice_prob)


def _update_hmm_parameters(
    model: ModelHMM,
    log_gamma: np.ndarray,
    log_xi: np.ndarray, 
    options: ModelOptionsHMM
) -> ModelHMM:
    """Update HMM parameters in M-step."""
    n_trials, n_states = log_gamma.shape
    
    # Update initial state probabilities
    pi_new = np.exp(log_gamma[0])
    pi_new /= pi_new.sum()
    
    # Update transition matrix
    A_new = np.zeros((n_states, n_states))
    for s1 in range(n_states):
        for s2 in range(n_states):
            numerator = scipy.special.logsumexp(log_xi[:, s1, s2])
            denominator = scipy.special.logsumexp(log_gamma[:-1, s1])
            A_new[s1, s2] = np.exp(numerator - denominator)
        
        # Normalize row
        A_new[s1] /= A_new[s1].sum()
    
    # Add Dirichlet prior if specified
    if options.alpha_A is not None:
        A_new = (A_new + options.alpha_A - 1) / \
                (A_new.sum(axis=1, keepdims=True) + options.alpha_A.sum(axis=1, keepdims=True) - n_states)
    
    return ModelHMM(beta=model.beta, pi=pi_new, A=A_new)


def _update_agent_parameters(
    agents: List[Agent],
    y: np.ndarray, 
    x: np.ndarray,
    log_gamma: np.ndarray,
    options: AgentOptions
) -> List[Agent]:
    """Update agent parameters in M-step."""
    n_trials, n_states = log_gamma.shape
    n_agents = len(agents)
    
    # For each state, update beta weights using weighted logistic regression
    beta_new = np.zeros((n_agents, n_states))
    
    for s in range(n_states):
        # Extract binary choices (assuming first column of y)
        choices = y[:, 0]
        
        # State-specific weights
        weights = np.exp(log_gamma[:, s])
        
        # Weighted logistic regression
        try:
            result = scipy.optimize.minimize(
                _weighted_logistic_loss,
                x0=np.zeros(n_agents),
                args=(x, choices, weights),
                method='BFGS'
            )
            beta_new[:, s] = result.x
        except:
            # Fallback to simple weighted least squares
            X_weighted = x * np.sqrt(weights[:, np.newaxis])
            y_weighted = choices * np.sqrt(weights)
            beta_new[:, s] = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
    
    # Create new model with updated beta
    # Note: In full implementation, would also update individual agent parameters
    return agents  # For now, return unchanged agents


def _weighted_logistic_loss(
    beta: np.ndarray,
    x: np.ndarray, 
    y: np.ndarray,
    weights: np.ndarray
) -> float:
    """Weighted logistic regression loss function."""
    logits = np.dot(x, beta)
    probs = 1 / (1 + np.exp(-logits))
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    
    log_likelihood = weights * (y * np.log(probs) + (1 - y) * np.log(1 - probs))
    return -np.sum(log_likelihood)
