#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Testing mixture_agents_models import...")
    import mixture_agents_models as mam
    print("✓ Import successful")
    
    print(f"Available attributes: {len([x for x in dir(mam) if not x.startswith('_')])}")
    
    # Test basic functionality
    print("Testing basic data creation...")
    import numpy as np
    
    data = mam.GenericData(
        choices=np.array([0, 1, 1, 0, 1]),
        rewards=np.array([0, 1, 1, 0, 1]),
        n_trials=5,
        n_sessions=1
    )
    print(f"✓ Created data with {data.n_trials} trials")
    
    print("Testing agent creation...")
    agents = [
        mam.MBReward(alpha=0.5),
        mam.Bias()
    ]
    print(f"✓ Created {len(agents)} agents")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
