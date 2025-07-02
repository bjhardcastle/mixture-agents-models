"""
Model structures and fitting algorithms for mixture-of-agents frameworks.

This module implements the core HMM and drift models along with
expectation-maximization algorithms for parameter estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.stats

from mixture_agents_models.agents import Agent
from mixture_agents_models.tasks import GenericData
from mixture_agents_models.types_ import DistributionType

logger = logging.getLogger(__name__)


@dataclass
class ModelHMM:
    """
    Hidden Markov Model for mixture-of-agents.

    Represents the latent state dynamics and agent mixture weights
    in a mixture-of-agents hidden Markov model.
    """

    beta: npt.NDArray[np.float64]  # (n_agents x n_states) agent weights
    pi: npt.NDArray[np.float64]  # (n_states,) initial state probabilities
    A: npt.NDArray[np.float64]  # (n_states x n_states) transition matrix

    def __post_init__(self) -> None:
        """Validate model parameters."""
        logger.debug("Validating ModelHMM parameters")
        n_agents, n_states = self.beta.shape

        if len(self.pi) != n_states:
            raise ValueError("pi must have length n_states")

        if self.A.shape != (n_states, n_states):
            raise ValueError("A must be (n_states x n_states)")

        # Ensure probabilities are valid
        if not np.allclose(self.pi.sum(), 1.0):
            raise ValueError(
                f"Initial state probabilities do not sum to 1: {self.pi.sum()}"
            )

        if not np.allclose(self.A.sum(axis=1), 1.0):
            raise ValueError(
                f"Transition matrix rows do not sum to 1: {self.A.sum(axis=1)}"
            )

        logger.debug(f"ModelHMM validated: {n_agents} agents, {n_states} states")


@dataclass
class ModelDrift:
    """
    Drift model for time-varying agent mixture weights.

    Alternative to HMM where agent weights drift smoothly over time
    rather than switching between discrete latent states.
    """

    beta: npt.NDArray[np.float64]  # (n_agents x n_timepoints) time-varying weights
    drift_variance: float  # variance of random walk drift

    def __post_init__(self) -> None:
        """Validate drift model parameters."""
        logger.debug("Validating ModelDrift parameters")
        if self.drift_variance <= 0:
            logger.error(f"Invalid drift variance: {self.drift_variance}")
            raise ValueError("drift_variance must be positive")

        n_agents, n_timepoints = self.beta.shape
        logger.debug(
            f"ModelDrift validated: {n_agents} agents, {n_timepoints} timepoints"
        )


@dataclass
class ModelOptionsHMM:
    """
    Configuration options for fitting HMM models.

    Specifies model structure, initialization parameters,
    and optimization settings for EM algorithm.
    """

    n_states: int = 2
    max_iter: int = 100
    tol: float = 1e-4
    n_starts: int = 1
    beta_0: npt.NDArray[np.float64] | None = None
    pi_0: npt.NDArray[np.float64] | None = None
    A_0: npt.NDArray[np.float64] | None = None
    alpha_A: npt.NDArray[np.float64] | None = None  # Dirichlet prior for A
    use_beta_prior: bool = True
    init_beta: bool = True
    verbose: bool = True
    disp_iter: int = 1

    def __post_init__(self) -> None:
        """Set default initialization values if not provided."""
        logger.debug(f"Initializing ModelOptionsHMM with {self.n_states} states")

        if self.pi_0 is None:
            self.pi_0 = np.ones(self.n_states) / self.n_states
            logger.debug("Created uniform initial state probabilities")

        if self.A_0 is None:
            # Default to high persistence (diagonal dominant)
            if self.n_states == 1:
                # Special case for 1 state - transition matrix is just [1.0]
                self.A_0 = np.ones((1, 1))
            else:
                self.A_0 = np.eye(self.n_states) * 0.9 + (1 - np.eye(self.n_states)) * (
                    0.1 / (self.n_states - 1)
                )
            logger.debug("Created default transition matrix with high persistence")

        if self.alpha_A is None:
            self.alpha_A = np.ones((self.n_states, self.n_states))
            logger.debug("Created uniform Dirichlet priors for transition matrix")


@dataclass
class ModelOptionsDrift:
    """Configuration options for fitting drift models."""

    max_iter: int = 100
    tol: float = 1e-4
    drift_prior_variance: float = 1.0
    init_beta: bool = True
    verbose: bool = True


@dataclass
class AgentOptions:
    """
    Configuration for agents and fitting parameters.

    Specifies which agents to include in mixture and which
    parameters should be fit versus held fixed.
    """

    agents: list[Agent]
    fit_symbols: list[str] | None = None
    fit_params: list[int] | None = None
    symbol_indices: list[int] | None = None
    param_indices: list[int] | None = None
    fit_indices: list[int] | None = None
    fit_scales: list[str] | None = None
    fit_priors: list[DistributionType] | None = None
    scale_x: bool = False

    def __post_init__(self) -> None:
        """Process agent options and compute derived fields."""
        if self.fit_symbols is not None and self.fit_params is not None:
            self._process_fit_specification()

    def _process_fit_specification(self) -> None:
        """Process fit symbols and parameters to create index mappings."""
        if self.fit_symbols is None or self.fit_params is None:
            return

        # Create mappings between symbols, parameters, and agents
        self.symbol_indices = []
        self.param_indices = []
        self.fit_indices = []
        self.fit_scales = []
        self.fit_priors = []

        for i, (symbol, param_idx) in enumerate(zip(self.fit_symbols, self.fit_params)):
            if param_idx > 0:  # 0 means don't fit this parameter
                agent_idx = param_idx - 1  # Convert to 0-indexed
                if agent_idx < len(self.agents):
                    agent = self.agents[agent_idx]
                    if hasattr(agent, symbol):
                        self.symbol_indices.append(i)
                        self.param_indices.append(agent_idx)
                        self.fit_indices.append(agent_idx)

                        # Get scale information
                        scale_attr = f"{symbol}_scale"
                        scale = (
                            getattr(agent, scale_attr, 0)
                            if hasattr(agent, scale_attr)
                            else 0
                        )
                        self.fit_scales.append(scale)

                        # Get prior information
                        prior_attr = f"{symbol}_prior"
                        prior = (
                            getattr(agent, prior_attr, None)
                            if hasattr(agent, prior_attr)
                            else None
                        )
                        self.fit_priors.append(prior)

        # Remove duplicates from fit_indices
        self.fit_indices = list(set(self.fit_indices))


def initialize_y(data: GenericData) -> np.ndarray:
    """
    Initialize choice matrix for model fitting.

    Args:
        data: Behavioral data

    Returns:
        Binary choice matrix (n_trials x n_choices)
    """
    n_trials = data.n_trials
    n_choices = int(data.choices.max()) + 1

    y = np.zeros((n_trials, n_choices))
    y[np.arange(n_trials), data.choices.astype(int)] = 1

    return y


def initialize_x(
    data: GenericData, agents: list[Agent], scale_x: bool = False
) -> npt.NDArray[np.float64]:
    """
    Initialize design matrix with agent Q-values.

    Args:
        data: Behavioral data
        agents: List of agents
        scale_x: Whether to z-score the design matrix

    Returns:
        Design matrix (n_trials x n_agents)
    """
    n_trials = data.n_trials
    n_agents = len(agents)

    x = np.zeros((n_trials, n_agents))

    for agent_idx, agent in enumerate(agents):
        q = agent.q0.copy()

        for t in range(n_trials):
            # Q-values for this trial
            x[t, agent_idx] = q[data.choices[t]] if len(q) > data.choices[t] else 0

            # Update Q-values for next trial
            if t < n_trials - 1:
                q = agent.next_q(q, data, t)

    if scale_x:
        # Z-score each agent column
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    return x


def optimize(
    data: GenericData,
    model_options: ModelOptionsHMM,
    agent_options: AgentOptions,
    verbose: bool = True,
) -> tuple[ModelHMM, list[Agent], float]:
    """
    Fit mixture-of-agents HMM using expectation-maximization.

    Args:
        data: Behavioral data
        model_options: HMM fitting options
        agent_options: Agent configuration options
        verbose: Whether to print progress

    Returns:
        Tuple of (fitted_model, fitted_agents, log_likelihood)
    """
    from mixture_agents_models.fitting import fit_hmm_em

    # Initialize data matrices
    y = initialize_y(data)
    x = initialize_x(data, agent_options.agents, agent_options.scale_x)

    # Fit model using EM algorithm
    model, agents, log_likelihood = fit_hmm_em(
        y=y,
        x=x,
        data=data,
        model_options=model_options,
        agent_options=agent_options,
        verbose=verbose,
    )

    return model, agents, log_likelihood


def simulate(
    model: ModelHMM, agents: list[Agent], data: GenericData, n_reps: int = 1
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Simulate choice behavior using fitted model.

    Args:
        model: Fitted HMM model
        agents: Fitted agents
        data: Behavioral data structure (for trial structure)
        n_reps: Number of simulation repetitions

    Returns:
        Dictionary with simulated choices, states, and probabilities
    """
    n_trials = data.n_trials
    n_states = len(model.pi)

    results = {
        "choices": np.zeros((n_reps, n_trials), dtype=int),
        "states": np.zeros((n_reps, n_trials), dtype=int),
        "choice_probs": np.zeros((n_reps, n_trials)),
        "state_probs": np.zeros((n_reps, n_trials, n_states)),
    }

    for rep in range(n_reps):
        # Initialize agent Q-values
        agent_qs = [agent.q0.copy() for agent in agents]

        # Sample initial state
        state = np.random.choice(n_states, p=model.pi)
        results["states"][rep, 0] = state

        for t in range(n_trials):
            # Get agent contributions for current state
            agent_values = np.array([q[0] if len(q) > 0 else 0 for q in agent_qs])

            # Compute choice probability using softmax with state-specific weights
            beta_state = model.beta[:, state]
            logits = np.dot(beta_state, agent_values)
            choice_prob = 1 / (1 + np.exp(-logits))

            # Sample choice
            choice = int(np.random.random() < choice_prob)
            results["choices"][rep, t] = choice
            results["choice_probs"][rep, t] = choice_prob

            # Update agent Q-values
            for i, agent in enumerate(agents):
                agent_qs[i] = agent.next_q(agent_qs[i], data, t)

            # Transition to next state
            if t < n_trials - 1:
                state = np.random.choice(n_states, p=model.A[state])
                results["states"][rep, t + 1] = state

    return results


def compute_log_likelihood(
    model: ModelHMM, agents: list[Agent], data: GenericData
) -> float:
    """
    Compute log-likelihood of data given model.

    Args:
        model: HMM model
        agents: List of agents
        data: Behavioral data

    Returns:
        Log-likelihood value
    """
    # Initialize matrices
    y = initialize_y(data)
    x = initialize_x(data, agents)

    n_trials, n_states = len(data.choices), len(model.pi)

    # Forward algorithm to compute likelihood
    log_alpha = np.zeros((n_trials, n_states))

    # Initial probabilities
    for s in range(n_states):
        beta_s = model.beta[:, s]
        logit = np.dot(x[0], beta_s)
        choice_prob = 1 / (1 + np.exp(-logit))

        if data.choices[0] == 1:
            emission_prob = choice_prob
        else:
            emission_prob = 1 - choice_prob

        log_alpha[0, s] = np.log(model.pi[s]) + np.log(emission_prob + 1e-10)

    # Forward recursion
    for t in range(1, n_trials):
        for s in range(n_states):
            beta_s = model.beta[:, s]
            logit = np.dot(x[t], beta_s)
            choice_prob = 1 / (1 + np.exp(-logit))

            if data.choices[t] == 1:
                emission_prob = choice_prob
            else:
                emission_prob = 1 - choice_prob

            # Sum over previous states
            log_alpha[t, s] = np.log(emission_prob + 1e-10) + scipy.special.logsumexp(
                log_alpha[t - 1] + np.log(model.A[:, s] + 1e-10)
            )

    # Total log-likelihood
    return scipy.special.logsumexp(log_alpha[-1])


def choice_accuracy(model: ModelHMM, agents: list[Agent], data: GenericData) -> float:
    """
    Compute choice prediction accuracy.

    Args:
        model: Fitted model
        agents: Fitted agents
        data: Behavioral data

    Returns:
        Prediction accuracy (fraction correct)
    """
    if data.n_trials == 0:
        return 0.0

    predictions = simulate(model, agents, data, n_reps=1)
    predicted_choices = predictions["choices"][0]

    return np.mean(predicted_choices == data.choices)
