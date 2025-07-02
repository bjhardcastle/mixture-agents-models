"""
EM algorithm implementation for mixture-of-agents HMM fitting.

This module provides the core expectation-maximization algorithm
for parameter estimation in mixture-of-agents hidden Markov models.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.special

from .agents import Agent
from .models import AgentOptions, ModelHMM, ModelOptionsHMM
from .tasks import GenericData

logger = logging.getLogger(__name__)


def fit_hmm_em(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    data: GenericData,
    model_options: ModelOptionsHMM,
    agent_options: AgentOptions,
    verbose: bool = True,
) -> tuple[ModelHMM, list[Agent], float]:
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
    logger.info("Starting HMM EM fitting process")
    logger.debug(f"Data shape: {x.shape}, States: {model_options.n_states}")

    n_trials, n_agents = x.shape
    n_states = model_options.n_states

    best_ll = -np.inf
    best_model = None
    best_agents = None

    for start in range(model_options.n_starts):
        logger.info(f"EM start {start + 1}/{model_options.n_starts}")
        if verbose:
            print(f"EM start {start + 1}/{model_options.n_starts}")

        # Initialize parameters
        logger.debug("Initializing HMM parameters")
        model = _initialize_hmm_parameters(model_options, n_agents)
        agents = _initialize_agents(agent_options)

        # EM iterations
        ll_prev = -np.inf

        for iteration in range(model_options.max_iter):
            logger.debug(f"EM iteration {iteration + 1}")

            # E-step: compute posterior state probabilities
            log_alpha, log_beta, log_gamma, log_xi = _forward_backward(
                y, x, model, agents
            )

            # M-step: update parameters
            model = _update_hmm_parameters(model, log_gamma, log_xi, model_options)

            agents = _update_agent_parameters(agents, y, x, log_gamma, agent_options)

            # Compute log-likelihood
            ll: float = scipy.special.logsumexp(log_alpha[-1])

            if verbose and (iteration + 1) % model_options.disp_iter == 0:
                logger.info(f"Iteration {iteration + 1}: LL = {ll:.4f}")
                print(f"  Iteration {iteration + 1}: LL = {ll:.4f}")

            # Check convergence
            if ll - ll_prev < model_options.tol:
                logger.info(
                    f"Converged after {iteration + 1} iterations (LL improvement: {ll - ll_prev:.6f})"
                )
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

            ll_prev = ll

        # Track best fit across starts
        if ll > best_ll:
            logger.debug(f"New best log-likelihood: {ll:.4f} (previous: {best_ll:.4f})")
            best_ll = ll
            best_model = model
            best_agents = agents

    logger.info(f"EM fitting completed. Final log-likelihood: {best_ll:.4f}")
    return best_model, best_agents, best_ll


def _initialize_hmm_parameters(options: ModelOptionsHMM, n_agents: int) -> ModelHMM:
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
    pi = (
        options.pi_0.copy()
        if options.pi_0 is not None
        else np.ones(n_states) / n_states
    )

    # Initialize A (transition matrix)
    A = (
        options.A_0.copy()
        if options.A_0 is not None
        else np.eye(n_states) * 0.9 + (1 - np.eye(n_states)) * 0.1 / (n_states - 1)
    )

    return ModelHMM(beta=beta, pi=pi, A=A)


def _initialize_agents(options: AgentOptions) -> list[Agent]:
    """Initialize agent parameters."""
    # For now, return copy of input agents
    # In full implementation, would handle parameter initialization
    return [agent for agent in options.agents]


def _forward_backward(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    model: ModelHMM,
    agents: list[Agent],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
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
            log_alpha[t, s] = emission_ll + scipy.special.logsumexp(
                log_alpha[t - 1] + np.log(model.A[:, s])
            )

    # Backward pass
    log_beta = np.zeros((n_trials, n_states))
    log_beta[-1, :] = 0  # Log of 1

    for t in range(n_trials - 2, -1, -1):
        for s in range(n_states):
            next_emissions = np.array(
                [
                    _compute_emission_log_likelihood(
                        y[t + 1], x[t + 1], model.beta[:, s_next]
                    )
                    for s_next in range(n_states)
                ]
            )
            log_beta[t, s] = scipy.special.logsumexp(
                np.log(model.A[s, :]) + next_emissions + log_beta[t + 1, :]
            )

    # Compute state and transition probabilities
    log_likelihood = scipy.special.logsumexp(log_alpha[-1])

    log_gamma = log_alpha + log_beta - log_likelihood

    log_xi = np.zeros((n_trials - 1, n_states, n_states))
    for t in range(n_trials - 1):
        for s1 in range(n_states):
            for s2 in range(n_states):
                emission_ll = _compute_emission_log_likelihood(
                    y[t + 1], x[t + 1], model.beta[:, s2]
                )
                log_xi[t, s1, s2] = (
                    log_alpha[t, s1]
                    + np.log(model.A[s1, s2])
                    + emission_ll
                    + log_beta[t + 1, s2]
                    - log_likelihood
                )

    return log_alpha, log_beta, log_gamma, log_xi


def _compute_emission_log_likelihood(
    y_t: npt.NDArray[np.float64],
    x_t: npt.NDArray[np.float64],
    beta_s: npt.NDArray[np.float64],
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
    log_gamma: npt.NDArray[np.float64],
    log_xi: npt.NDArray[np.float64],
    options: ModelOptionsHMM,
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
        A_new = (A_new + options.alpha_A - 1) / (
            A_new.sum(axis=1, keepdims=True)
            + options.alpha_A.sum(axis=1, keepdims=True)
            - n_states
        )

    return ModelHMM(beta=model.beta, pi=pi_new, A=A_new)


def _update_agent_parameters(
    agents: list[Agent],
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    log_gamma: npt.NDArray[np.float64],
    options: AgentOptions,
) -> list[Agent]:
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
                method="BFGS",
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
    beta: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> float:
    """Weighted logistic regression loss function."""
    logits = np.dot(x, beta)
    probs = 1 / (1 + np.exp(-logits))
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    log_likelihood = weights * (y * np.log(probs) + (1 - y) * np.log(1 - probs))
    return -np.sum(log_likelihood)
