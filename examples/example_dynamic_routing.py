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
# Add the dynamicrouting directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "dynamicrouting"))
try:
    from RLmodelHPC import getSessionData
    HAVE_RL_MODEL = True
except ImportError:
    print("Note: RLmodelHPC not found - will use simulated data only")
    HAVE_RL_MODEL = False


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
    
    agents = mam.create_dynamic_routing_agents()
    
    agent_names = [type(agent).__name__ for agent in agents]
    print(f"   Agents: {', '.join(agent_names)}")
    
    # Step 3: Fit 2-state HMM model
    print("\n3. Fitting 2-state mixture-of-agents HMM...")
    
    model_options = mam.ModelOptionsHMM(
        n_states=2,
        max_iter=50,
        tol=1e-4,
        n_starts=1,
        verbose=True
    )
    
    agent_options = mam.AgentOptions(
        agents=agents,
        scale_x=False
    )
    
    model, fitted_agents, log_likelihood = mam.optimize(
        data=data,
        model_options=model_options,
        agent_options=agent_options,
        verbose=True
    )
    
    print(f"   Log-likelihood: {log_likelihood:.2f}")
    
    # Compute choice accuracy
    accuracy = mam.choice_accuracy(model, fitted_agents, data)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Step 4: Analyze results
    print("\n4. Analyzing results...")
    
    print("\n   State transition matrix:")
    print(model.A)
    
    print("\n   Agent weights by state:")
    print("   State 1:", model.beta[:, 0])
    print("   State 2:", model.beta[:, 1])
    
    # Step 5: Generate plots
    print("\n5. Generating plots...")
    
    # Plot main results
    fig1 = mam.plot_model(model, fitted_agents, data)
    plt.show()
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nThis example demonstrated:")
    print("- Creating dynamic routing behavioral data")
    print("- Fitting mixture-of-agents HMM models")
    print("- Analyzing context-dependent behavior")
    print("\nFor real data integration:")
    if HAVE_RL_MODEL:
        print("- Use getSessionData() to load actual experimental data")
        print("- Apply convert_from_dynamic_routing() for data conversion")
    else:
        print("- Install RLmodelHPC.py for real data integration")


if __name__ == "__main__":
    main()
