"""
Quick start script for mixture agents models.

This script demonstrates the basic workflow and can be used to verify
that the package is working correctly after installation.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Try importing the package
try:
    import mixture_agents_models as mam
    print("✓ Successfully imported mixture_agents_models")
except ImportError as e:
    print(f"✗ Failed to import mixture_agents_models: {e}")
    print("Make sure to install the package first:")
    print("  pip install -e .")
    exit(1)

def quick_demo():
    """Run a quick demonstration of the package functionality."""
    
    print("\nMixture Agents Models - Quick Demo")
    print("=" * 35)
    
    # 1. Create simple test data
    print("1. Creating test data...")
    np.random.seed(42)
    
    n_trials = 200
    choices = np.random.binomial(1, 0.6, n_trials)  # 60% choice rate
    rewards = np.random.binomial(1, 0.7, n_trials)  # 70% reward rate
    
    data = mam.GenericData(
        choices=choices,
        rewards=rewards,
        n_trials=n_trials,
        n_sessions=1
    )
    
    print(f"   Created {n_trials} trials")
    print(f"   Choice rate: {np.mean(choices):.3f}")
    print(f"   Reward rate: {np.mean(rewards):.3f}")
    
    # 2. Create agents
    print("\n2. Creating agents...")
    
    agents = [
        mam.MBReward(alpha=0.5),
        mam.MFReward(alpha=0.4),
        mam.Bias()
    ]
    
    print(f"   Created {len(agents)} agents: {[type(a).__name__ for a in agents]}")
    
    # 3. Fit model
    print("\n3. Fitting model...")
    
    model_options = mam.ModelOptionsHMM(
        n_states=2,
        max_iter=20,  # Keep short for demo
        verbose=False
    )
    
    agent_options = mam.AgentOptions(agents=agents)
    
    try:
        model, fitted_agents, log_likelihood = mam.optimize(
            data=data,
            model_options=model_options,
            agent_options=agent_options,
            verbose=False
        )
        
        print(f"   ✓ Model fitted successfully")
        print(f"   Log-likelihood: {log_likelihood:.2f}")
        
    except Exception as e:
        print(f"   ✗ Model fitting failed: {e}")
        return False
    
    # 4. Analyze results
    print("\n4. Analyzing results...")
    
    try:
        accuracy = mam.choice_accuracy(model, fitted_agents, data)
        print(f"   Choice accuracy: {accuracy:.3f}")
        
        print(f"   Initial state probs: {model.pi}")
        print(f"   Transition matrix shape: {model.A.shape}")
        print(f"   Agent weights shape: {model.beta.shape}")
        
    except Exception as e:
        print(f"   ✗ Analysis failed: {e}")
        return False
    
    # 5. Test simulation
    print("\n5. Testing simulation...")
    
    try:
        predictions = mam.simulate(model, fitted_agents, data, n_reps=1)
        sim_choices = predictions['choices'][0]
        
        print(f"   ✓ Simulation successful")
        print(f"   Simulated choice rate: {np.mean(sim_choices):.3f}")
        
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        return False
    
    # 6. Test plotting
    print("\n6. Testing plotting...")
    
    try:
        fig = mam.plot_model(model, fitted_agents, data)
        plt.close(fig)  # Close without showing
        
        print(f"   ✓ Plotting successful")
        
    except Exception as e:
        print(f"   ✗ Plotting failed: {e}")
        return False
    
    # 7. Test dynamic routing integration
    print("\n7. Testing dynamic routing integration...")
    
    try:
        dr_agents = mam.create_dynamic_routing_agents()
        print(f"   ✓ Dynamic routing agents created: {[type(a).__name__ for a in dr_agents]}")
        
        # Create minimal dynamic routing data
        dr_data = mam.DynamicRoutingData(
            choices=choices[:50],  # Smaller for speed
            rewards=rewards[:50],
            n_trials=50,
            n_sessions=1,
            contexts=np.random.randint(0, 2, 50)
        )
        
        # Test conversion and fitting functions exist
        print(f"   ✓ Dynamic routing data created: {dr_data.n_trials} trials")
        
    except Exception as e:
        print(f"   ✗ Dynamic routing integration failed: {e}")
        return False
    
    print("\n" + "=" * 35)
    print("✓ All tests passed! Package is working correctly.")
    print("\nNext steps:")
    print("- Run: python examples/example_basic_fitting.py")
    print("- Run: python examples/example_dynamic_routing.py")
    print("- Check out the documentation in README_python.md")
    
    return True

if __name__ == "__main__":
    success = quick_demo()
    exit(0 if success else 1)
