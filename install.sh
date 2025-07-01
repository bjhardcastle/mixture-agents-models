#!/bin/bash

# Install mixture agents models Python package
# This script sets up the Python environment and installs dependencies

echo "Setting up Mixture Agents Models Python package..."
echo "=================================================="

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using UV for package management"
    PACKAGE_MANAGER="uv"
else
    echo "UV not found, using pip"
    PACKAGE_MANAGER="pip"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        uv venv venv
    else
        python -m venv venv
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install package in development mode
echo "Installing mixture agents models package..."
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi

# Run tests to verify installation
echo "Running tests to verify installation..."
python -m pytest tests/ -v

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation completed successfully!"
    echo ""
    echo "To get started:"
    echo "  source venv/bin/activate"
    echo "  python examples/example_basic_fitting.py"
    echo "  python examples/example_dynamic_routing.py"
    echo ""
    echo "To run tests:"
    echo "  python -m pytest tests/"
    echo ""
    echo "To deactivate virtual environment:"
    echo "  deactivate"
else
    echo ""
    echo "✗ Installation failed. Check error messages above."
    exit 1
fi
