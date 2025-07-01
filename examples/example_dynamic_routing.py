"""
Example: Fitting mixture-of-agents HMM to dynamic routing data.

This example demonstrates how to load dynamic routing behavioral data,
convert it to the mixture agents format, and fit a MoA-HMM model.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import mixture agents models
import mixture_agents_models as mam

# Import dynamic routing utilities  
import sys
sys.path.append('../dynamicrouting')
from RLmodelHPC import getSessionData


def main():
    """Run dynamic routing MoA-HMM fitting example."""
    
    # Example parameters (replace with actual data loading)
    mouse_id = 12345
    session_start_time = "2024-01-15_10:30:00"
    
    print("Dynamic Routing MoA-HMM Example")
    print("=" * 40)
    
    # Step 1: Create simulated dynamic routing data for demonstration
    print("1. Creating simulated dynamic routing data...")
    
    # Simulate 500 trials of dynamic routing behavior
    n_trials = 500
    
    # Simulate context switches every ~50 trials
    contexts = np.repeat([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 50)[:n_trials]
    
    # Simulate choices with context bias
    choice_probs = np.where(contexts == 0, 0.7, 0.3)  # Visual bias
    choices = np.random.binomial(1, choice_probs)
    
    # Simulate rewards (70% correct in matching context)
    reward_probs = np.where(
        (contexts == 0) & (choices == 1), 0.8,  # Visual context, choice 1
        np.where((contexts == 1) & (choices == 0), 0.8, 0.2)  # Audio context, choice 0
    )
    rewards = np.random.binomial(1, reward_probs)
    
    # Create simulated trial stimuli
    trial_stim = np.where(contexts == 0, 'vis1', 'sound1')
    trial_block = np.repeat(np.arange(10) + 1, 50)[:n_trials]
    
    # Create dynamic routing data object
    data = mam.DynamicRoutingData(
        choices=choices,
        rewards=rewards,
        n_trials=n_trials,
        n_sessions=1,
        contexts=contexts,
        trial_stim=trial_stim,
        trial_block=trial_block,
        mouse_id=str(mouse_id),
        session_start_time=session_start_time
    )
    
    print(f"   Created data with {n_trials} trials")
    print(f"   Choice rate: {np.mean(choices):.3f}")
    print(f"   Reward rate: {np.mean(rewards):.3f}")
    
    # Step 2: Define agents for mixture model
    print("\n2. Defining mixture agents...")
    
    agents = mam.create_dynamic_routing_agents(
        alpha_context=0.6,
        alpha_reinforcement=0.5,
        alpha_perseveration=0.3
    )
    
    agent_names = [type(agent).__name__ for agent in agents]
    print(f"   Agents: {', '.join(agent_names)}")
    
    # Step 3: Fit 2-state HMM model
    print("\n3. Fitting 2-state mixture-of-agents HMM...")
    
    results = mam.fit_dynamic_routing_model(
        data=data,
        agents=agents,
        n_states=2,
        max_iter=50,
        verbose=True
    )
    
    print(f"   Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"   Accuracy: {results['accuracy']:.3f}")
    
    # Step 4: Analyze results
    print("\n4. Analyzing results...")
    
    model = results['model']
    fitted_agents = results['agents']
    predictions = results['predictions']
    
    print("\n   State transition matrix:")
    print(model.A)
    
    print("\n   Agent weights by state:")
    print("   State 1:", model.beta[:, 0])
    print("   State 2:", model.beta[:, 1])
    
    # Compute context sensitivity metrics
    dr_results = mam.DynamicRoutingResults(**results)
    context_metrics = dr_results.compute_context_sensitivity()
    
    print("\n   Context sensitivity:")
    for metric, value in context_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    # Step 5: Model comparison
    print("\n5. Comparing different model configurations...")
    
    # Define different agent combinations to test
    agent_sets = [
        [mam.ContextRL(), mam.Bias()],  # Context + Bias only
        [mam.MFReward(), mam.Perseveration(), mam.Bias()],  # Classic RL agents
        agents,  # Full dynamic routing set
    ]
    
    model_names = [
        "Context+Bias",
        "Classic RL",
        "Dynamic Routing"
    ]
    
    comparison_df = mam.compare_dynamic_routing_models(
        data=data,
        agent_sets=agent_sets,
        model_names=model_names,
        n_states_range=[1, 2, 3],
        max_iter=30,
        verbose=False
    )
    
    print("\n   Model comparison results:")
    print(comparison_df[['model_name', 'n_states', 'aic', 'bic', 'accuracy']].to_string(index=False))
    
    # Step 6: Plotting
    print("\n6. Generating plots...")
    
    # Plot main results
    fig1 = dr_results.plot_results()
    plt.show()
    
    # Plot model comparison
    fig2 = mam.plot_comparison(comparison_df, metric='aic')
    plt.show()
    
    # Plot agent weights matrix
    fig3 = mam.plot_agent_weights_matrix(model, fitted_agents)
    plt.show()
    
    # Step 7: Integration with existing pipeline
    print("\n7. Integration with dynamic routing pipeline...")
    
    integration_results = mam.integrate_with_rl_model_hpc(
        mouse_id=mouse_id,
        session_start_time=session_start_time,
        mixture_results=dr_results
    )
    
    print("   Integration successful!")
    print(f"   Compatibility: {integration_results['compatibility']}")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nNext steps:")
    print("- Load real dynamic routing data using getSessionData()")
    print("- Experiment with different agent combinations")
    print("- Perform parameter recovery analysis")
    print("- Cross-validate model performance")


if __name__ == "__main__":
    main()
