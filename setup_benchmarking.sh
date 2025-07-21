#!/bin/bash
# Setup script for FLUX.1-Kontext benchmarking suite

echo "=== FLUX.1-Kontext Benchmarking Setup ==="
echo

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No virtual environment detected!"
    echo "Please activate your environment first:"
    echo "  source flux_env/bin/activate"
    exit 1
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"

# Install benchmarking dependencies
echo
echo "Installing benchmarking dependencies..."
pip install -r requirements_benchmark.txt

# Create output directories
echo
echo "Creating output directories..."
mkdir -p benchmark_results
mkdir -p benchmark_plots
mkdir -p quality_plots
mkdir -p cost_analysis
mkdir -p demo_results

# Check GPU
echo
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('✗ No GPU detected!')
"

# Check HF token
echo
echo "Checking HuggingFace token..."
if [[ "$HF_TOKEN" == "" ]]; then
    echo "⚠️  HF_TOKEN not set!"
    echo "Please set: export HF_TOKEN='your_token_here'"
else
    echo "✓ HF_TOKEN is set"
fi

echo
echo "=== Setup Complete ==="
echo
echo "To run benchmarks:"
echo "1. Quick demo (30 min):    python run_benchmark_demo.py"
echo "2. Full suite (2-3 hours): python benchmark_suite.py"
echo
echo "After benchmarking:"
echo "- Visualizations: python benchmark_visualizer.py"
echo "- Quality check:  python quality_assessment.py"
echo "- Cost analysis:  python cost_calculator.py"