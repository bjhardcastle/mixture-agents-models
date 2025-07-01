"""
Example: Basic mixture-of-agents HMM fitting.

This example demonstrates the core functionality of the mixture agents
framework using simulated two-step task data.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mixture_agents_models as mam


def main():
    """Run basic MoA-HMM fitting example."""
    
    print("Basic Mixture-of-Agents HMM Example")
    print("=" * 40)
    
    # Step 1: Simulate behavioral data
    print("1. Simulating behavioral data...")
    
    n_trials = 1000
    n_sessions = 5
    
    # Create realistic choice and reward sequences
    # Simulate learning with some model-based and model-free components
    choices = []
    rewards = []
    session_indices = []
    
    for session in range(n_sessions):
        session_trials = n_trials // n_sessions
        
        # Simulate learning curve within session
        for t in range(session_trials):
            # Choice probability that evolves over time
            prob = 0.3 + 0.4 * (t / session_trials)  # Learning curve
            choice = int(np.random.random() < prob)
            
            # Reward probability depends on choice
            reward_prob = 0.8 if choice == 1 else 0.2
            reward = int(np.random.random() < reward_prob)
            
            choices.append(choice)
            rewards.append(reward)
            session_indices.append(session)
    
    # Create data object
    data = mam.GenericData(
        choices=np.array(choices),
        rewards=np.array(rewards),
        n_trials=len(choices),
        n_sessions=n_sessions,
        session_indices=np.array(session_indices)
    )
    
    print(f"   Created {data.n_trials} trials across {data.n_sessions} sessions")
    print(f"   Overall choice rate: {np.mean(data.choices):.3f}")
    print(f"   Overall reward rate: {np.mean(data.rewards):.3f}")
    
    # Step 2: Define agents
    print("\n2. Defining agents...")
    
    agents = [
        mam.MBReward(alpha=0.6),
        mam.MFReward(alpha=0.4),
        mam.Perseveration(alpha=0.3),
        mam.Bias()
    ]
    
    agent_names = [type(agent).__name__ for agent in agents]
    print(f"   Agents: {', '.join(agent_names)}")
    
    # Step 3: Configure and fit model
    print("\n3. Fitting 2-state HMM...")
    
    model_options = mam.ModelOptionsHMM(
        n_states=2,
        max_iter=100,
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
    
    print(f"\n   Final log-likelihood: {log_likelihood:.2f}")
    
    # Step 4: Analyze fitted model
    print("\n4. Analyzing fitted model...")
    
    print("\n   Initial state probabilities:")
    print(f"   π = {model.pi}")
    
    print("\n   State transition matrix:")
    print("   A =")
    for i, row in enumerate(model.A):
        print(f"   State {i+1}: {row}")
    
    print("\n   Agent weights by state:")
    for i in range(model.beta.shape[1]):
        print(f"   State {i+1}: {model.beta[:, i]}")
    
    # Compute choice accuracy
    accuracy = mam.choice_accuracy(model, fitted_agents, data)
    print(f"\n   Choice prediction accuracy: {accuracy:.3f}")
    
    # Step 5: Model simulation
    print("\n5. Simulating from fitted model...")
    
    predictions = mam.simulate(model, fitted_agents, data, n_reps=3)
    
    print(f"   Generated {predictions['choices'].shape[1]} simulated trials")
    print(f"   Mean simulated choice rate: {np.mean(predictions['choices']):.3f}")
    
    # Step 6: Cross-validation
    print("\n6. Cross-validation...")
    
    cv_results = mam.cross_validate(
        data=data,
        model_options=model_options,
        agent_options=agent_options,
        n_folds=5
    )
    
    print(f"   Mean test accuracy: {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
    print(f"   Mean test log-likelihood: {np.mean(cv_results['test_ll']):.2f} ± {np.std(cv_results['test_ll']):.2f}")
    
    # Step 7: Model comparison
    print("\n7. Model comparison...")
    
    # Compare different numbers of states
    model_configs = []
    for n_states in [1, 2, 3]:
        model_configs.append({
            'model_options': mam.ModelOptionsHMM(n_states=n_states, max_iter=50, verbose=False),
            'agent_options': agent_options
        })
    
    comparison_df = mam.model_compare(
        data=data,
        model_configs=model_configs,
        model_names=[f"{i+1}-State" for i in range(3)]
    )
    
    print("\n   Model comparison results:")
    print(comparison_df[['model', 'n_states', 'aic', 'bic', 'accuracy']].to_string(index=False))
    
    # Step 8: Parameter recovery
    print("\n8. Parameter recovery test...")
    
    # Test recovery with smaller simulation for speed
    recovery_results = mam.parameter_recovery(
        true_agents=fitted_agents,
        true_model=model,
        n_trials=500,
        n_sims=5,
        model_options=mam.ModelOptionsHMM(n_states=2, max_iter=30, verbose=False)
    )
    
    print("   Parameter recovery correlations:")
    for param, correlation in recovery_results['correlations'].items():
        print(f"   {param}: r = {correlation:.3f}")
    
    # Step 9: Plotting
    print("\n9. Generating plots...")
    
    # Plot model fit
    fig1 = mam.plot_model(model, fitted_agents, data)
    plt.show()
    
    # Plot model comparison
    fig2 = mam.plot_comparison(comparison_df, metric='aic')
    plt.show()
    
    # Plot parameter recovery
    if len(recovery_results['true_beta']) > 0:
        fig3 = mam.plot_recovery(recovery_results)
        plt.show()
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nKey results:")
    print(f"- Best model: {comparison_df.iloc[0]['model']} (AIC = {comparison_df.iloc[0]['aic']:.1f})")
    print(f"- Cross-validation accuracy: {np.mean(cv_results['test_accuracy']):.3f}")
    print(f"- Parameter recovery: β r = {recovery_results['correlations'].get('beta', 0):.3f}")


if __name__ == "__main__":
    main()
