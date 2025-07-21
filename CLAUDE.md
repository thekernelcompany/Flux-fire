# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
# First-time setup
./setup_flux.sh

# Activate virtual environment
source flux_env/bin/activate

# Set HuggingFace token (required)
export HF_TOKEN="your_huggingface_token_here"
```

### Running the Server
```bash
# Run the main standalone server with benchmarking
python flux_kontext_standalone.py

# Alternative endpoint implementations
python flux_endpoint.py
python flux_kontext_script.py
```

### Installing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# PyTorch must be installed with CUDA support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Architecture Overview

This is a high-performance FLUX.1-Kontext image generation service with the following key components:

### Core Scripts
- **flux_kontext_standalone.py**: Main server with automatic benchmarking, FastAPI endpoints, and full optimization pipeline
- **flux_endpoint.py**: Alternative endpoint implementation
- **flux_kontext_script.py**: Script version without server functionality
- **utils.py**: Para-attention caching utilities and tensor operations

### Key Optimizations
The codebase implements a 2.63x speedup through:
1. **Para-Attention Caching**: Caches attention computations across inference steps using the `para-attn` library
2. **Torch Compilation**: Uses `torch.compile` with max-autotune mode for optimized kernels
3. **Memory Format Optimization**: Converts models to channels_last format
4. **QKV Projection Fusion**: Fuses query, key, value projections in attention layers
5. **Mega-Cache System**: Saves compiled artifacts to `.cache/flux_endpoint/` for faster subsequent loads

### API Structure
FastAPI server exposes:
- `POST /inference`: Main generation endpoint accepting prompts, dimensions, steps, etc.
- `GET /health`: Health check endpoint
- Supports both text-to-image and image-to-image generation
- Returns base64-encoded images and saves to `outputs/` directory

### Dependencies
- **Deep Learning**: PyTorch 2.7.0 with CUDA, transformers, diffusers (from git)
- **Optimization**: accelerate, para-attn for attention caching
- **Web Framework**: FastAPI with uvicorn
- **Image Processing**: Pillow, OpenCV

### Important Notes
- Requires CUDA-capable GPU with 24GB+ VRAM
- HuggingFace token required for model access (set via HF_TOKEN env var)
- No test suite or linting configuration present
- Generated images saved to `outputs/` directory
- Cache artifacts stored in `.cache/flux_endpoint/`

## Dev Guidelines

### Workflow Notes
- Make sure to not change the flux kontext standalone file and we are only gonna work with it and nothing else in the repo