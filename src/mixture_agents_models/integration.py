"""
Integration module for dynamic routing experiments.

This module provides specialized functionality for integrating
mixture-of-agents models with dynamic routing behavioral experiments,
building on the existing RLmodelHPC.py framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

from mixture_agents_models.agents import Agent, Bias, ContextRL, MFReward, Perseveration
from mixture_agents_models.models import AgentOptions, ModelHMM, ModelOptionsHMM, optimize
from mixture_agents_models.tasks import DynamicRoutingData

logger = logging.getLogger(__name__)


def _convert_single_session(session_data: DynRoutData) -> DynamicRoutingData:
    """
    Convert dynamic routing session data to mixture agents format.

    Args:
        session_data: DynRoutData

    Returns:
        DynamicRoutingData object for mixture agents analysis
    """
    # Extract core behavioral data
    choices = np.array(session_data.trialResponse, dtype=int)

    # Compute rewards based on choice and rewarded stimulus
    rewards = np.zeros_like(choices, dtype=float)
    for t in range(len(choices)):
        if hasattr(session_data, "autoRewardScheduled"):
            if session_data.autoRewardScheduled[t]:
                rewards[t] = 1.0

        # Check if choice matches rewarded stimulus
        if (
            choices[t] == 1
            and hasattr(session_data, "rewardedStim")
            and hasattr(session_data, "trialStim")
        ):
            stim = session_data.trialStim[t]
            if stim == session_data.rewardedStim[t]:
                rewards[t] = 1.0

    # Extract trial context information
    contexts = np.zeros(len(choices), dtype=int)
    if hasattr(session_data, "trialStim"):
        # Convert stimulus names to context indices
        for t, stim in enumerate(session_data.trialStim):
            if "vis" in str(stim):
                contexts[t] = 0  # Visual context
            elif "sound" in str(stim):
                contexts[t] = 1  # Auditory context

    # Extract other trial information
    trial_stim = (
        np.array(session_data.trialStim)
        if hasattr(session_data, "trialStim")
        else np.array([])
    )
    trial_block = (
        np.array(session_data.trialBlock)
        if hasattr(session_data, "trialBlock")
        else np.array([])
    )
    trial_opto_label = (
        np.array(session_data.trialOptoLabel)
        if hasattr(session_data, "trialOptoLabel")
        else np.array([])
    )
    auto_reward_scheduled = (
        np.array(session_data.autoRewardScheduled)
        if hasattr(session_data, "autoRewardScheduled")
        else np.array([])
    )
    rewarded_stim = (
        np.array(session_data.rewardedStim)
        if hasattr(session_data, "rewardedStim")
        else np.array([])
    )
    stim_start_times = (
        np.array(session_data.stimStartTimes)
        if hasattr(session_data, "stimStartTimes")
        else np.array([])
    )

    return DynamicRoutingData(
        choices=choices,
        rewards=rewards,
        n_trials=len(choices),
        n_sessions=1,
        contexts=contexts,
        trial_stim=trial_stim,
        trial_block=trial_block,
        trial_opto_label=trial_opto_label,
        auto_reward_scheduled=auto_reward_scheduled,
        rewarded_stim=rewarded_stim,
        stim_start_times=stim_start_times,
        mouse_id=getattr(session_data, "subjectName", None),
        session_start_time=getattr(session_data, "startTime", None),
        metadata={"source": "dynamic_routing", "converted": True},
    )


def create_dynamic_routing_agents(
    vis_confidence: float = 0.8,
    aud_confidence: float = 0.8,
    alpha_context: float = 0.5,
    alpha_reinforcement: float = 0.4,
    alpha_perseveration: float = 0.3,
    bias_action: float = 0.0,
) -> list[Agent]:
    """
    Create standard agent set for dynamic routing experiments.

    Args:
        vis_confidence: Visual stimulus confidence parameter
        aud_confidence: Auditory stimulus confidence parameter
        alpha_context: Context learning rate
        alpha_reinforcement: Reinforcement learning rate
        alpha_perseveration: Perseveration learning rate
        bias_action: Action bias parameter

    Returns:
        List of agents configured for dynamic routing
    """
    agents = [
        ContextRL(alpha_context=alpha_context, alpha_reinforcement=alpha_reinforcement),
        MFReward(alpha=alpha_reinforcement),
        Perseveration(alpha=alpha_perseveration),
        Bias(),
    ]

    return agents


def create_dynamic_routing_agents() -> list[Agent]:
    """
    Create a default set of agents suitable for dynamic routing experiments.

    This provides a standard agent configuration that parallels the
    reinforcement learning components in the original dynamic routing model.

    Returns:
        List of Agent objects configured for dynamic routing analysis
    """
    agents = [
        ContextRL(alpha_context=0.3, alpha_reinforcement=0.2),
        MFReward(alpha=0.4),
        Perseveration(alpha=0.2),
        Bias(),
    ]

    return agents


def fit_dynamic_routing_model(
    data: DynamicRoutingData,
    agents: list[Agent] | None = None,
    n_states: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Fit mixture-of-agents model to dynamic routing data.

    Args:
        data: Dynamic routing behavioral data
        agents: List of agents (uses default if None)
        n_states: Number of hidden states
        **kwargs: Additional arguments for model fitting

    Returns:
        Dictionary with fitting results
    """
    if agents is None:
        agents = create_dynamic_routing_agents()

    # Configure model options
    model_options = ModelOptionsHMM(
        n_states=n_states,
        max_iter=kwargs.get("max_iter", 100),
        tol=kwargs.get("tol", 1e-4),
        n_starts=kwargs.get("n_starts", 1),
        verbose=kwargs.get("verbose", True),
    )

    # Configure agent options
    agent_options = AgentOptions(agents=agents, scale_x=kwargs.get("scale_x", False))

    # Fit model
    model, fitted_agents, log_likelihood = optimize(
        data=data,
        model_options=model_options,
        agent_options=agent_options,
        verbose=kwargs.get("verbose", True),
    )

    # Compute additional metrics
    from mixture_agents_models.models import choice_accuracy, simulate

    accuracy = choice_accuracy(model, fitted_agents, data)
    predictions = simulate(model, fitted_agents, data, n_reps=1)

    return {
        "model": model,
        "agents": fitted_agents,
        "log_likelihood": log_likelihood,
        "accuracy": accuracy,
        "predictions": predictions,
        "data": data,
        "model_options": model_options,
        "agent_options": agent_options,
    }


@dataclass
class DynamicRoutingResults:
    """
    Container for dynamic routing analysis results.

    Provides structured access to model fitting results and
    comparison metrics for dynamic routing experiments.
    """

    model: ModelHMM
    agents: list[Agent]
    log_likelihood: float
    accuracy: float
    predictions: dict[str, np.ndarray]
    data: DynamicRoutingData
    model_options: ModelOptionsHMM
    agent_options: AgentOptions
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_state_probabilities(self) -> np.ndarray:
        """Get posterior state probabilities over time."""
        return self.predictions.get("state_probs", np.array([]))

    def get_agent_contributions(self) -> dict[str, np.ndarray]:
        """Get agent contribution weights over time."""
        state_probs = self.get_state_probabilities()
        if len(state_probs) == 0:
            return {}

        # Compute expected agent weights across states
        n_trials = state_probs.shape[1]
        n_agents = self.model.beta.shape[0]

        agent_weights = np.zeros((n_trials, n_agents))
        for t in range(n_trials):
            agent_weights[t] = np.dot(self.model.beta, state_probs[0, t])

        agent_names = [type(agent).__name__ for agent in self.agents]
        return {name: agent_weights[:, i] for i, name in enumerate(agent_names)}

    def compute_context_sensitivity(self) -> dict[str, float]:
        """Compute metrics of context sensitivity."""
        if not hasattr(self.data, "contexts") or self.data.contexts is None:
            return {}

        contexts = self.data.contexts
        choices = self.data.choices

        # Compute choice probability by context
        vis_trials = contexts == 0
        aud_trials = contexts == 1

        metrics = {}
        if np.sum(vis_trials) > 0:
            metrics["visual_choice_rate"] = np.mean(choices[vis_trials])
        if np.sum(aud_trials) > 0:
            metrics["auditory_choice_rate"] = np.mean(choices[aud_trials])

        if "visual_choice_rate" in metrics and "auditory_choice_rate" in metrics:
            metrics["context_bias"] = (
                metrics["visual_choice_rate"] - metrics["auditory_choice_rate"]
            )

        return metrics

    def plot_results(self, save_path: str | None = None) -> None:
        """Generate standard plots for dynamic routing results."""
        from mixture_agents_models.plotting import plot_dynamic_routing_results

        plot_dynamic_routing_results(self, save_path=save_path)


def compare_dynamic_routing_models(
    data: DynamicRoutingData,
    agent_sets: list[list[Agent]],
    model_names: list[str] | None = None,
    n_states_range: list[int] = [1, 2, 3],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Compare different agent combinations and state numbers.

    Args:
        data: Dynamic routing behavioral data
        agent_sets: List of agent combinations to compare
        model_names: Names for each agent set
        n_states_range: Range of state numbers to test
        **kwargs: Additional fitting arguments

    Returns:
        DataFrame with model comparison results
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(agent_sets))]

    results = []

    for i, agents in enumerate(agent_sets):
        for n_states in n_states_range:
            print(f"Fitting {model_names[i]} with {n_states} states...")

            try:
                result = fit_dynamic_routing_model(
                    data=data, agents=agents, n_states=n_states, verbose=False, **kwargs
                )

                results.append(
                    {
                        "model_name": model_names[i],
                        "n_states": n_states,
                        "n_agents": len(agents),
                        "log_likelihood": result["log_likelihood"],
                        "accuracy": result["accuracy"],
                        "aic": -2 * result["log_likelihood"]
                        + 2 * (len(agents) * n_states + n_states + n_states**2),
                        "bic": -2 * result["log_likelihood"]
                        + np.log(data.n_trials)
                        * (len(agents) * n_states + n_states + n_states**2),
                        "agents": [type(agent).__name__ for agent in agents],
                    }
                )

            except Exception as e:
                print(f"  Failed: {e}")
                results.append(
                    {
                        "model_name": model_names[i],
                        "n_states": n_states,
                        "n_agents": len(agents),
                        "log_likelihood": np.nan,
                        "accuracy": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "agents": [type(agent).__name__ for agent in agents],
                        "error": str(e),
                    }
                )

    return pd.DataFrame(results)


def integrate_with_rl_model_hpc(
    mouse_id: int, session_start_time: str, mixture_results: DynamicRoutingResults
) -> dict[str, Any]:
    """
    Integrate mixture agents results back with RLmodelHPC framework.

    Args:
        mouse_id: Mouse identifier
        session_start_time: Session start time
        mixture_results: Results from mixture agents fitting

    Returns:
        Dictionary with integrated analysis results
    """
    # This would integrate with the existing dynamic routing pipeline
    # For now, return structured results that can be used downstream

    integration_results = {
        "mouse_id": mouse_id,
        "session_start_time": session_start_time,
        "mixture_model": {
            "log_likelihood": mixture_results.log_likelihood,
            "accuracy": mixture_results.accuracy,
            "n_states": len(mixture_results.model.pi),
            "agent_weights": mixture_results.model.beta,
            "state_transitions": mixture_results.model.A,
            "initial_state_probs": mixture_results.model.pi,
        },
        "agent_contributions": mixture_results.get_agent_contributions(),
        "context_sensitivity": mixture_results.compute_context_sensitivity(),
        "predictions": mixture_results.predictions,
        "compatibility": {
            "rl_model_hpc": True,
            "conversion_successful": True,
            "data_format": "dynamic_routing",
        },
    }

    return integration_results


def convert_from_dynamic_routing(
    session_data: DynRoutData | Iterable[DynRoutData]
) -> DynamicRoutingData:
    """
    Convert dynamic routing session data to mixture agents format.

    Handles both single sessions and multiple sessions, combining them
    into a unified DynamicRoutingData object for cross-session analysis.

    Args:
        session_data: Single DynRoutData object or iterable of DynRoutData objects

    Returns:
        DynamicRoutingData object for mixture agents analysis
    """
    # Handle single session case
    if isinstance(session_data, DynRoutData):
        return _convert_single_session(session_data)

    # Handle multiple sessions case
    sessions = list(session_data)
    if len(sessions) == 0:
        raise ValueError("Empty session data provided")

    if len(sessions) == 1:
        return _convert_single_session(sessions[0])

    # Convert each session individually
    converted_sessions = [_convert_single_session(session) for session in sessions]

    # Combine all sessions into a single DynamicRoutingData object
    all_choices = np.concatenate([session.choices for session in converted_sessions])
    all_rewards = np.concatenate([session.rewards for session in converted_sessions])
    all_contexts = np.concatenate([session.contexts for session in converted_sessions])

    # Combine other trial data arrays
    all_trial_stim = np.concatenate([
        session.trial_stim for session in converted_sessions
        if len(session.trial_stim) > 0
    ]) if any(len(session.trial_stim) > 0 for session in converted_sessions) else np.array([])

    all_trial_block = np.concatenate([
        session.trial_block for session in converted_sessions
        if len(session.trial_block) > 0
    ]) if any(len(session.trial_block) > 0 for session in converted_sessions) else np.array([])

    all_trial_opto_label = np.concatenate([
        session.trial_opto_label for session in converted_sessions
        if len(session.trial_opto_label) > 0
    ]) if any(len(session.trial_opto_label) > 0 for session in converted_sessions) else np.array([])

    all_auto_reward_scheduled = np.concatenate([
        session.auto_reward_scheduled for session in converted_sessions
        if len(session.auto_reward_scheduled) > 0
    ]) if any(len(session.auto_reward_scheduled) > 0 for session in converted_sessions) else np.array([])

    all_rewarded_stim = np.concatenate([
        session.rewarded_stim for session in converted_sessions
        if len(session.rewarded_stim) > 0
    ]) if any(len(session.rewarded_stim) > 0 for session in converted_sessions) else np.array([])

    all_stim_start_times = np.concatenate([
        session.stim_start_times for session in converted_sessions
        if len(session.stim_start_times) > 0
    ]) if any(len(session.stim_start_times) > 0 for session in converted_sessions) else np.array([])

    # Create session indices to track which trials belong to which session
    session_indices = np.concatenate([
        np.full(session.n_trials, i) for i, session in enumerate(converted_sessions)
    ])

    # Collect metadata from all sessions
    mouse_ids = [session.mouse_id for session in converted_sessions if session.mouse_id is not None]
    if len(set(mouse_ids)) != 1:
        logger.warning('Sessions to be combined are from different subjects - the id and start time on the resulting object will only reflect the first session.')
    session_start_times = [session.session_start_time for session in converted_sessions if session.session_start_time is not None]

    combined_metadata = {
        "source": "dynamic_routing",
        "converted": True,
        "n_source_sessions": len(sessions),
        "mouse_ids": mouse_ids,
        "session_start_times": session_start_times,
    }

    return DynamicRoutingData(
        choices=all_choices,
        rewards=all_rewards,
        n_trials=len(all_choices),
        n_sessions=len(sessions),
        session_indices=session_indices,
        contexts=all_contexts,
        trial_stim=all_trial_stim,
        trial_block=all_trial_block,
        trial_opto_label=all_trial_opto_label,
        auto_reward_scheduled=all_auto_reward_scheduled,
        rewarded_stim=all_rewarded_stim,
        stim_start_times=all_stim_start_times,
        mouse_id=mouse_ids[0] if mouse_ids else None,
        session_start_time=session_start_times[0] if session_start_times else None,
        metadata=combined_metadata,
    )
