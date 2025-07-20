#!/bin/bash
# Setup script for FLUX.1 Kontext on remote server

echo "Setting up FLUX.1 Kontext environment..."

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU support may not be available."
else
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv flux_env
source flux_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p outputs
mkdir -p ~/.cache/flux_endpoint

# Set up environment variables
cat > .env << EOF
# HuggingFace token (required for FLUX.1-Kontext-dev)
HF_TOKEN=your_huggingface_token_here

# Optimization settings
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
EOF

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your HuggingFace token"
echo "2. Activate the virtual environment: source flux_env/bin/activate"
echo "3. Run the server: python flux_kontext_standalone.py"
echo ""
echo "The API will be available at http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"