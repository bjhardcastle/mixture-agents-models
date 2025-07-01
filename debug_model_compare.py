#!/usr/bin/env python3

import sys
sys.path.insert(0, '/root/capsule/code/MixtureAgentsModels/src')
import mixture_agents_models as mam
import numpy as np

# Create minimal test data
print("Creating test data...")
data = mam.GenericData(
    choices=np.array([0, 1, 0, 1, 0]),
    rewards=np.array([1, 0, 1, 0, 1]),
    n_trials=5,
    n_sessions=1
)

# Create agents
print("Creating agents...")
agents = [mam.MBReward(), mam.Bias()]
agent_options = mam.AgentOptions(agents=agents)

# Test model comparison
print("Testing model comparison...")
try:
    model_configs = [
        {'model_options': mam.ModelOptionsHMM(n_states=1, max_iter=5, verbose=False), 
         'agent_options': agent_options},
        {'model_options': mam.ModelOptionsHMM(n_states=2, max_iter=5, verbose=False), 
         'agent_options': agent_options}
    ]
    
    comparison_df = mam.model_compare(
        data=data,
        model_configs=model_configs,
        model_names=['1-State', '2-State'],
        cv_folds=2  # Reduce folds for small dataset
    )
    print(f"Model comparison successful: {len(comparison_df)} models compared")
    print(comparison_df)
except Exception as e:
    import traceback
    print(f"Model comparison failed: {e}")
    print("Full traceback:")
    traceback.print_exc()
