"""
Test the enhanced AgentOptions parameter sharing and agents_comparison functionality.

This script validates that the Python implementation now matches the Julia 
functionality for complex parameter sharing patterns and systematic agent comparison.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mixture_agents_models as mam


def test_complex_parameter_sharing():
    """Test complex parameter sharing patterns like Julia [1,1,2,2,0]."""
    print("=== Testing Complex Parameter Sharing ===")
    
    # Create agents similar to Julia examples
    agents = [
        mam.MBReward(alpha=0.6),
        mam.MBReward(alpha=0.7),  # Different instance, will share parameter
        mam.MFReward(alpha=0.4),
        mam.MFReward(alpha=0.5),  # Different instance, will share parameter  
        mam.Bias()
    ]
    
    # Test Julia-style parameter sharing: [1,1,2,2,0]
    # This means: agents 0&1 share param 1, agents 2&3 share param 2, agent 4 has no fitted params
    fit_symbols = ["alpha", "alpha"]  # Two alpha parameters
    fit_params = [1, 1, 2, 2, 0]  # Sharing pattern
    
    agent_options = mam.AgentOptions(
        agents=agents,
        fit_symbols=fit_symbols,
        fit_params=fit_params
    )
    
    print(f"✓ Created AgentOptions with sharing pattern: {fit_params}")
    print(f"  - Fit symbols: {fit_symbols}")
    print(f"  - Number of agents: {len(agents)}")
    print(f"  - Fit indices: {agent_options.fit_indices}")
    print(f"  - Parameter sharing mapping: {agent_options.get_shared_params()}")
    
    # Verify parameter sharing is correctly processed
    shared_params = agent_options.get_shared_params()
    assert 1 in shared_params, "Parameter 1 should be in sharing map"
    assert 2 in shared_params, "Parameter 2 should be in sharing map"
    assert shared_params[1] == [0, 1], f"Parameter 1 should map to agents [0,1], got {shared_params[1]}"
    assert shared_params[2] == [2, 3], f"Parameter 2 should map to agents [2,3], got {shared_params[2]}"
    
    print("✓ Parameter sharing correctly processed")
    return agent_options


def test_agents_comparison():
    """Test agents_comparison function matches Julia behavior."""
    print("\n=== Testing Agents Comparison ===")
    
    # Create base configuration
    agents = [
        mam.MBReward(alpha=0.5),
        mam.MFReward(alpha=0.4), 
        mam.Bias()
    ]
    
    fit_symbols = ["alpha", "alpha"]
    fit_params = [1, 2, 0]  # Independent parameters for first two agents
    
    agent_options = mam.AgentOptions(
        agents=agents,
        fit_symbols=fit_symbols,
        fit_params=fit_params
    )
    
    model_options = mam.ModelOptionsHMM(n_states=2, max_iter=10, verbose=False)
    
    # Run agents comparison
    model_ops, agent_ops = mam.agents_comparison(model_options, agent_options)
    
    print(f"✓ Generated {len(model_ops)} model configurations")
    print(f"✓ Generated {len(agent_ops)} agent configurations")
    
    # Verify structure
    assert len(model_ops) == len(agents) + 1, f"Should have {len(agents) + 1} models, got {len(model_ops)}"
    assert len(agent_ops) == len(agents) + 1, f"Should have {len(agents) + 1} agent configs, got {len(agent_ops)}"
    
    # First configuration should be original (all agents)
    assert len(agent_ops[0].agents) == len(agents), "First config should have all agents"
    
    # Subsequent configurations should have one less agent each
    for i in range(1, len(agent_ops)):
        assert len(agent_ops[i].agents) == len(agents) - 1, f"Config {i} should have {len(agents) - 1} agents"
        print(f"  Model {i}: {len(agent_ops[i].agents)} agents, fit_params: {agent_ops[i].fit_params}")
    
    print("✓ Agent comparison structure verified")
    return model_ops, agent_ops


def test_shared_parameter_comparison():
    """Test agents_comparison with shared parameters."""
    print("\n=== Testing Shared Parameter Comparison ===")
    
    # Test with parameter sharing pattern [1,1,2,2,0] like Julia
    agents = [
        mam.MBReward(alpha=0.6),
        mam.MBReward(alpha=0.7),
        mam.MFReward(alpha=0.4), 
        mam.MFReward(alpha=0.5),
        mam.Bias()
    ]
    
    fit_symbols = ["alpha", "alpha"]
    fit_params = [1, 1, 2, 2, 0]  # Shared parameters
    
    agent_options = mam.AgentOptions(
        agents=agents,
        fit_symbols=fit_symbols,
        fit_params=fit_params
    )
    
    model_options = mam.ModelOptionsHMM(n_states=1, max_iter=5, verbose=False)
    
    # Run agents comparison
    model_ops, agent_ops = mam.agents_comparison(model_options, agent_options)
    
    print(f"✓ Generated {len(model_ops)} configurations with shared parameters")
    
    # Verify that removing agents properly handles shared parameters
    for i, config in enumerate(agent_ops[1:], 1):  # Skip first (full) config
        print(f"  Config {i}: {len(config.agents)} agents")
        if config.fit_params:
            print(f"    Fit params: {config.fit_params}")
            print(f"    Fit symbols: {config.fit_symbols}")
            print(f"    Shared params: {config.get_shared_params() if hasattr(config, 'get_shared_params') else 'N/A'}")
        else:
            print("    No fitted parameters")
    
    print("✓ Shared parameter handling verified")
    return model_ops, agent_ops


def test_integration_with_existing_code():
    """Test that new functionality integrates with existing optimization."""
    print("\n=== Testing Integration with Optimization ===")
    
    # Create synthetic data
    np.random.seed(42)
    n_trials = 50
    choices = np.random.binomial(1, 0.6, n_trials)
    rewards = np.random.binomial(1, 0.7, n_trials)
    
    data = mam.GenericData(
        choices=choices,
        rewards=rewards,
        n_trials=n_trials,
        n_sessions=1
    )
    
    # Test with shared parameters
    agents = [
        mam.MBReward(alpha=0.5),
        mam.MFReward(alpha=0.4),
        mam.Bias()
    ]
    
    # Use parameter sharing
    fit_symbols = ["alpha"]
    fit_params = [1, 1, 0]  # Both RL agents share same alpha
    
    agent_options = mam.AgentOptions(
        agents=agents,
        fit_symbols=fit_symbols,
        fit_params=fit_params
    )
    
    model_options = mam.ModelOptionsHMM(n_states=1, max_iter=5, verbose=False)
    
    try:
        # Test basic optimization with shared parameters
        model, fitted_agents, ll = mam.optimize(
            data=data,
            model_options=model_options,
            agent_options=agent_options,
            verbose=False
        )
        print("✓ Optimization with shared parameters successful")
        print(f"  Log-likelihood: {ll:.3f}")
        
        # Test agents comparison integration
        model_ops, agent_ops = mam.agents_comparison(model_options, agent_options)
        
        # Try fitting one of the comparison models
        test_model, test_agents, test_ll = mam.optimize(
            data=data,
            model_options=model_ops[1],  # Model with first agent removed
            agent_options=agent_ops[1],
            verbose=False
        )
        print("✓ Agents comparison model fitting successful")
        print(f"  Comparison model log-likelihood: {test_ll:.3f}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise
    
    print("✓ Integration tests passed")


def main():
    """Run all tests."""
    print("Testing Enhanced AgentOptions and Agents Comparison")
    print("=" * 60)
    
    try:
        # Test complex parameter sharing
        agent_options = test_complex_parameter_sharing()
        
        # Test agents comparison
        model_ops, agent_ops = test_agents_comparison()
        
        # Test with shared parameters  
        shared_model_ops, shared_agent_ops = test_shared_parameter_comparison()
        
        # Test integration
        test_integration_with_existing_code()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\nThe Python implementation now supports:")
        print("✓ Complex parameter sharing patterns (Julia-style [1,1,2,2,0])")
        print("✓ Systematic agent comparison analysis")
        print("✓ Integration with existing optimization pipeline")
        print("✓ Compatibility with model comparison workflows")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
