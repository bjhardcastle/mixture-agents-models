#!/bin/bash

# Mixture Agents Models - Python Package Installation
# This script sets up the Python translation of the Julia MixtureAgentsModels

echo "Setting up Mixture Agents Models Python Package"
echo "================================================"

# Create virtual environment (optional but recommended)
echo "1. Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "2. Installing dependencies..."
pip install --upgrade pip
pip install numpy scipy matplotlib seaborn pandas scikit-learn

# Install package in development mode
echo "3. Installing package..."
pip install -e .

# Run tests
echo "4. Running basic tests..."
python quick_start.py

echo "5. Package structure:"
echo "   src/mixture_agents_models/    - Main package"
echo "   examples/                     - Usage examples" 
echo "   dynamicrouting/               - Integration with existing RL model"
echo "   tests/                        - Test suite"

echo ""
echo "Installation complete!"
echo ""
echo "Quick start:"
echo "  python quick_start.py                    # Basic functionality test"
echo "  python examples/example_basic_fitting.py # Full example"
echo "  python examples/example_dynamic_routing.py # DR integration"
echo ""
echo "Key imports:"
echo "  import mixture_agents_models as mam"
echo "  agents = [mam.MBReward(), mam.MFReward(), mam.Bias()]"
echo "  model, fitted_agents, ll = mam.optimize(data, model_options, agent_options)"
