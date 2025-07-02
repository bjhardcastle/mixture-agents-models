"""
Plotting functions for mixture agents models.

This module provides visualization functions for model results,
parameter recovery, and model comparison.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .agents import Agent
from .models import ModelHMM
from .tasks import GenericData


def plot_model(
    model: ModelHMM,
    agents: list[Agent],
    data: GenericData,
    sessions_to_plot: list[int] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot model fit with state probabilities and agent weights.

    Args:
        model: Fitted HMM model
        agents: Fitted agents
        data: Behavioral data
        sessions_to_plot: Which sessions to plot (plots first 3 if None)
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    if sessions_to_plot is None:
        sessions_to_plot = list(range(min(3, data.n_sessions)))

    n_sessions = len(sessions_to_plot)
    n_states = len(model.pi)
    n_agents = len(agents)

    fig, axes = plt.subplots(n_sessions, 3, figsize=(15, 4 * n_sessions))
    if n_sessions == 1:
        axes = axes.reshape(1, -1)

    for i, session_idx in enumerate(sessions_to_plot):
        session_data = data.get_session_data(session_idx)

        # Simulate to get state probabilities
        from .models import simulate

        predictions = simulate(model, agents, session_data, n_reps=1)

        trials = np.arange(session_data.n_trials)

        # Plot choices and rewards
        axes[i, 0].scatter(
            trials, session_data.choices, c="blue", alpha=0.6, s=20, label="Choices"
        )
        axes[i, 0].scatter(
            trials, session_data.rewards, c="red", alpha=0.6, s=20, label="Rewards"
        )
        axes[i, 0].set_ylabel("Choice/Reward")
        axes[i, 0].set_title(f"Session {session_idx + 1}: Behavior")
        axes[i, 0].legend()
        axes[i, 0].set_ylim(-0.1, 1.1)

        # Plot state probabilities
        if "state_probs" in predictions and len(predictions["state_probs"]) > 0:
            state_probs = predictions["state_probs"][0]  # First rep
            for state in range(n_states):
                axes[i, 1].plot(
                    trials,
                    state_probs[:, state],
                    label=f"State {state + 1}",
                    linewidth=2,
                )
        axes[i, 1].set_ylabel("State Probability")
        axes[i, 1].set_title("Hidden State Probabilities")
        axes[i, 1].legend()
        axes[i, 1].set_ylim(0, 1)

        # Plot agent weights (expected across states)
        if "state_probs" in predictions and len(predictions["state_probs"]) > 0:
            state_probs = predictions["state_probs"][0]
            agent_weights = np.zeros((session_data.n_trials, n_agents))
            for t in range(session_data.n_trials):
                agent_weights[t] = np.dot(model.beta, state_probs[t])

            colors = [agent.color for agent in agents]
            for agent_idx in range(n_agents):
                axes[i, 2].plot(
                    trials,
                    agent_weights[:, agent_idx],
                    color=colors[agent_idx],
                    label=type(agents[agent_idx]).__name__,
                    linewidth=2,
                )

        axes[i, 2].set_ylabel("Agent Weight")
        axes[i, 2].set_title("Agent Contributions")
        axes[i, 2].legend()
        axes[i, 2].set_xlabel("Trial")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_recovery(
    recovery_results: dict[str, Any], save_path: str | None = None
) -> plt.Figure:
    """
    Plot parameter recovery results.

    Args:
        recovery_results: Results from parameter_recovery function
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot beta recovery
    if len(recovery_results["true_beta"]) > 0:
        true_beta = np.concatenate([b.flatten() for b in recovery_results["true_beta"]])
        recovered_beta = np.concatenate(
            [b.flatten() for b in recovery_results["recovered_beta"]]
        )

        axes[0].scatter(true_beta, recovered_beta, alpha=0.6)
        axes[0].plot(
            [true_beta.min(), true_beta.max()],
            [true_beta.min(), true_beta.max()],
            "r--",
            alpha=0.8,
        )
        axes[0].set_xlabel("True β")
        axes[0].set_ylabel("Recovered β")
        axes[0].set_title(
            f'β Recovery (r = {recovery_results["correlations"].get("beta", 0):.3f})'
        )
        axes[0].grid(True, alpha=0.3)

    # Plot pi recovery
    if len(recovery_results["true_pi"]) > 0:
        true_pi = np.concatenate([p for p in recovery_results["true_pi"]])
        recovered_pi = np.concatenate([p for p in recovery_results["recovered_pi"]])

        axes[1].scatter(true_pi, recovered_pi, alpha=0.6)
        axes[1].plot([0, 1], [0, 1], "r--", alpha=0.8)
        axes[1].set_xlabel("True π")
        axes[1].set_ylabel("Recovered π")
        axes[1].set_title(
            f'π Recovery (r = {recovery_results["correlations"].get("pi", 0):.3f})'
        )
        axes[1].grid(True, alpha=0.3)

    # Plot A recovery
    if len(recovery_results["true_A"]) > 0:
        true_A = np.concatenate([A.flatten() for A in recovery_results["true_A"]])
        recovered_A = np.concatenate(
            [A.flatten() for A in recovery_results["recovered_A"]]
        )

        axes[2].scatter(true_A, recovered_A, alpha=0.6)
        axes[2].plot([0, 1], [0, 1], "r--", alpha=0.8)
        axes[2].set_xlabel("True A")
        axes[2].set_ylabel("Recovered A")
        axes[2].set_title(
            f'A Recovery (r = {recovery_results["correlations"].get("A", 0):.3f})'
        )
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_comparison(
    comparison_df, metric: str = "aic", save_path: str | None = None
) -> plt.Figure:
    """
    Plot model comparison results.

    Args:
        comparison_df: DataFrame from model_compare function
        metric: Metric to plot ('aic', 'bic', 'accuracy', 'log_likelihood')
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot
    models = comparison_df["model"]
    values = comparison_df[metric]

    bars = ax.bar(models, values)

    # Color bars by rank if available
    if f"{metric}_rank" in comparison_df.columns:
        ranks = comparison_df[f"{metric}_rank"]
        colors = plt.cm.RdYlBu_r(ranks / ranks.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison: {metric.upper()}")

    # Rotate x-axis labels if needed
    if len(models) > 5:
        plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_dynamic_routing_results(results, save_path: str | None = None) -> plt.Figure:
    """
    Plot results specific to dynamic routing experiments.

    Args:
        results: DynamicRoutingResults object
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    data = results.data
    model = results.model
    predictions = results.predictions

    trials = np.arange(data.n_trials)

    # Plot 1: Choices and rewards over time
    axes[0, 0].scatter(trials, data.choices, c="blue", alpha=0.6, s=10, label="Choices")
    axes[0, 0].scatter(trials, data.rewards, c="red", alpha=0.6, s=10, label="Rewards")
    if hasattr(data, "contexts") and data.contexts is not None:
        # Color background by context
        vis_trials = data.contexts == 0
        aud_trials = data.contexts == 1
        axes[0, 0].fill_between(
            trials[vis_trials], 0, 1, alpha=0.1, color="purple", label="Visual"
        )
        axes[0, 0].fill_between(
            trials[aud_trials], 0, 1, alpha=0.1, color="orange", label="Auditory"
        )

    axes[0, 0].set_ylabel("Choice/Reward")
    axes[0, 0].set_title("Behavior Over Time")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(-0.1, 1.1)

    # Plot 2: State probabilities
    if "state_probs" in predictions and len(predictions["state_probs"]) > 0:
        state_probs = predictions["state_probs"][0]
        n_states = state_probs.shape[1]
        colors = plt.cm.Set1(np.linspace(0, 1, n_states))

        for state in range(n_states):
            axes[0, 1].plot(
                trials,
                state_probs[:, state],
                color=colors[state],
                label=f"State {state + 1}",
                linewidth=2,
            )

    axes[0, 1].set_ylabel("State Probability")
    axes[0, 1].set_title("Hidden State Probabilities")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)

    # Plot 3: Agent contributions
    agent_contributions = results.get_agent_contributions()
    if agent_contributions:
        for i, (name, weights) in enumerate(agent_contributions.items()):
            color = results.agents[i].color if i < len(results.agents) else f"C{i}"
            axes[0, 2].plot(trials, weights, color=color, label=name, linewidth=2)

    axes[0, 2].set_ylabel("Agent Weight")
    axes[0, 2].set_title("Agent Contributions")
    axes[0, 2].legend()

    # Plot 4: Context-dependent choice rates
    if hasattr(data, "contexts") and data.contexts is not None:
        context_metrics = results.compute_context_sensitivity()

        contexts = ["Visual", "Auditory"]
        choice_rates = [
            context_metrics.get("visual_choice_rate", 0),
            context_metrics.get("auditory_choice_rate", 0),
        ]

        bars = axes[1, 0].bar(contexts, choice_rates, color=["purple", "orange"])
        axes[1, 0].set_ylabel("Choice Rate")
        axes[1, 0].set_title("Context-Dependent Choice Rates")
        axes[1, 0].set_ylim(0, 1)

        # Add value labels
        for bar, rate in zip(bars, choice_rates):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{rate:.3f}",
                ha="center",
                va="bottom",
            )

    # Plot 5: Model transition matrix
    im = axes[1, 1].imshow(model.A, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_xlabel("To State")
    axes[1, 1].set_ylabel("From State")
    axes[1, 1].set_title("State Transition Matrix")

    # Add text annotations
    n_states = model.A.shape[0]
    for i in range(n_states):
        for j in range(n_states):
            axes[1, 1].text(j, i, f"{model.A[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(im, ax=axes[1, 1])

    # Plot 6: Agent weights by state
    im2 = axes[1, 2].imshow(model.beta, cmap="RdBu_r", aspect="auto")
    axes[1, 2].set_xlabel("State")
    axes[1, 2].set_ylabel("Agent")
    axes[1, 2].set_title("Agent Weights by State")

    # Set agent labels
    agent_names = [type(agent).__name__ for agent in results.agents]
    axes[1, 2].set_yticks(range(len(agent_names)))
    axes[1, 2].set_yticklabels(agent_names)

    # Add text annotations
    for i in range(model.beta.shape[0]):
        for j in range(model.beta.shape[1]):
            axes[1, 2].text(j, i, f"{model.beta[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(im2, ax=axes[1, 2])

    # Add overall title with model performance
    fig.suptitle(
        f"Dynamic Routing Model Results (LL = {results.log_likelihood:.2f}, "
        f"Accuracy = {results.accuracy:.3f})",
        fontsize=16,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_agent_weights_matrix(
    model: ModelHMM, agents: list[Agent], save_path: str | None = None
) -> plt.Figure:
    """
    Plot agent weights matrix as heatmap.

    Args:
        model: Fitted HMM model
        agents: List of agents
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(model.beta, cmap="RdBu_r", aspect="auto")

    # Set labels
    agent_names = [type(agent).__name__ for agent in agents]
    state_names = [f"State {i+1}" for i in range(model.beta.shape[1])]

    ax.set_xticks(range(len(state_names)))
    ax.set_xticklabels(state_names)
    ax.set_yticks(range(len(agent_names)))
    ax.set_yticklabels(agent_names)

    # Add value annotations
    for i in range(model.beta.shape[0]):
        for j in range(model.beta.shape[1]):
            ax.text(
                j,
                i,
                f"{model.beta[i, j]:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.set_title("Agent Weights by Hidden State", fontsize=14)
    ax.set_xlabel("Hidden State")
    ax.set_ylabel("Agent")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weight")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
