# Mixture Agents Models - Python

A Python implementation of Mixture-of-Agents Hidden Markov Models (MoA-HMM) for behavioral data analysis, translated from the original Julia codebase and integrated with dynamic routing experiments.

## Overview

This package implements computational models for analyzing choice behavior using mixture-of-agents frameworks. The core innovation is decomposing complex decision-making into interpretable cognitive strategies (agents) that can transition between latent behavioral states.

## Key Components

- **Agents**: Individual cognitive strategies (model-based, model-free, bias, perseveration, etc.)
- **HMM Framework**: Hidden Markov models for state transitions between agent mixtures
- **Dynamic Routing Integration**: Specialized components for multi-modal decision tasks
- **Fitting & Analysis**: Parameter estimation, cross-validation, and model comparison tools

## Installation

```bash
# Using UV (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

```python
import mixture_agents_models as mam
from dynamicrouting.rl_model_hpc import load_session_data

# Load behavioral data
data = mam.load_generic_data("path/to/data.csv")

# Define agents for mixture model
agents = [
    mam.MBReward(alpha=0.6),
    mam.MFReward(alpha=0.4), 
    mam.Bias()
]

# Configure HMM with 2 hidden states
model_options = mam.ModelOptionsHMM(n_states=2, max_iter=100)
agent_options = mam.AgentOptions(agents=agents)

# Fit model
model, fitted_agents, log_likelihood = mam.optimize(
    data=data,
    model_options=model_options, 
    agent_options=agent_options
)

# Generate predictions and analyze results
predictions = mam.simulate(model, fitted_agents, data)
mam.plot_model(model, fitted_agents, data)
```

## Integration with Dynamic Routing

The package includes specialized integration with dynamic routing experiments:

```python
from dynamicrouting.rl_model_hpc import get_session_data
import mixture_agents_models as mam

# Load dynamic routing session
mouse_id = 12345
session_data = get_session_data(mouse_id, "2024-01-15")

# Convert to mixture agents format
data = mam.convert_from_dynamic_routing(session_data)

# Define context-sensitive agents
agents = [
    mam.ContextRL(alpha_context=0.7),
    mam.MFReward(alpha=0.5),
    mam.Perseveration(alpha=0.3)
]

# Fit and analyze
results = mam.fit_dynamic_routing_model(data, agents)
```

## Architecture

The package follows these design principles:

- **Dataclasses**: Structured data containers for agents, models, and options
- **Type Safety**: Full type hints throughout the codebase  
- **Modular Design**: Composable agents and model components
- **Integration Ready**: Compatible with existing dynamic routing analysis pipeline

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=mixture_agents_models tests/
```
