# FLUX.1-Kontext - High-Performance Image Generation

Professional implementation of FLUX.1-Kontext with modular architecture and advanced optimizations achieving up to 3.5x speedup.

> **🚀 NEW**: The project has been refactored with a clean, professional structure. See [README_NEW.md](README_NEW.md) for the updated documentation or continue below for the original scripts.

## Available Scripts

### flux_kontext_standalone.py
A complete, self-contained server that:
- **Automatically benchmarks performance** on first run, comparing optimized vs unoptimized inference
- **Runs a FastAPI server** for REST API access to the model
- **Implements cutting-edge optimizations** including para-attention caching and torch compilation
- **Manages caching** for faster subsequent startups
- **Supports both text-to-image and image-to-image generation**

### flux_kontext_optimized_refactored.py (NEW)
A refactored, modular version of the optimization stack that:
- **Separates concerns** into specialized modules for better maintainability
- **Provides the same optimizations** as the original but with cleaner architecture
- **Enables easier testing and debugging** of individual components
- **Supports profiling and benchmarking** with detailed performance analysis

## Modular Architecture (NEW)

The codebase has been refactored into modular components:

### Core Modules
- **`flux_optimizations.py`**: Optimization implementations
  - FlashAttention 3 integration
  - MXFP4 quantization
  - H100 FP8 optimization
  - TensorRT compilation
  - Model data type management

- **`flux_schedulers.py`**: Scheduler and CUDA graphs
  - DPM-Solver++ scheduler setup
  - Euler scheduler configuration
  - CUDA graphs management
  - Scheduler compatibility patches

- **`flux_compilation.py`**: Compilation utilities
  - torch.compile configuration
  - Model compilation with fallback strategies
  - Mega-cache management for faster loads

- **`flux_utils.py`**: Shared utilities
  - Profiling and timing tools
  - Performance analysis
  - Image processing
  - Benchmarking framework

## Important Setup Requirement

**CRITICAL**: Before running `flux_kontext_standalone.py`, you MUST replace the utils.py file in the para_attn package:

```bash
# Replace the para_attn utils.py with the one from this repo
cp utils.py flux-env/lib/python*/site-packages/para_attn/first_block_cache/utils.py
```

This step is essential for FLUX.1-Kontext compatibility. The provided `utils.py` includes necessary modifications for the model to work correctly.

## Quick Start

### Option 1: Run the Standalone Server
```bash
# 1. Setup environment (first time only)
./setup_flux.sh

# 2. Activate environment
source flux-env/bin/activate

# 3. Replace para_attn utils.py (REQUIRED)
cp utils.py flux-env/lib/python*/site-packages/para_attn/first_block_cache/utils.py
# Reason here: https://chatgpt.com/share/687e5e7b-7df4-8010-a1e6-b7aafb55dd2f

# 4. Set HuggingFace token (required for model download)
export HF_TOKEN="your_huggingface_token_here"

# 5. Run the standalone server
python flux_kontext_standalone.py
```

### Option 2: Run the Refactored Script with Profiling
```bash
# Steps 1-4 same as above, then:

# 5. Run the refactored script with modular optimizations
python flux_kontext_optimized_refactored.py

# Or run with custom optimizations
python -c "
from flux_kontext_optimized_refactored import OptimizedFluxService

service = OptimizedFluxService()
service.load_model(enable_optimizations={
    'flash_attention': False,  # Set True if you have FlashAttention 3
    'mxfp4': True,            # MXFP4 quantization
    'scheduler': 'dpm_solver', # or 'euler'
    'cuda_graphs': True,       # CUDA graph optimization
    'para_attention': False,   # Set True if no torch.compile conflicts
    'tensorrt': False         # Set True for TensorRT backend
})

# Run inference
result = service.inference(
    prompt='A beautiful sunset over mountains',
    height=1024,
    width=1024,
    steps=14
)
"
```

## What Happens When You Run It

When you execute `python flux_kontext_standalone.py`, the script will:

1. **Run Performance Benchmarks** (first run):
   - Tests unoptimized model performance (3 runs)
   - Tests optimized model performance (3 runs)
   - Shows speedup comparison (typically 2.63x faster)
   - Saves a sample image to `outputs/` directory

2. **Start FastAPI Server**:
   - Loads the optimized model
   - Applies para-attention caching
   - Compiles with torch.compile for maximum performance
   - Starts server on `http://localhost:8000`

## API Endpoints

### Generate Images
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "num_images": 1
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Main text description |
| `prompt2` | string | null | Additional prompt (optional) |
| `negative_prompt` | string | null | What to avoid in generation |
| `image_url` | string | null | URL for image-to-image generation |
| `width` | int | 1024 | Image width (256-1024, multiple of 16) |
| `height` | int | 1024 | Image height (256-1024, multiple of 16) |
| `steps` | int | 28 | Inference steps (1-50) |
| `guidance_scale` | float | 3.5 | Prompt adherence (0-20) |
| `true_cfg_scale` | float | 1.0 | True CFG scale (0-20) |
| `num_images` | int | 1 | Batch size (1-4) |
| `seed` | int | null | Random seed for reproducibility |
| `output_format` | string | "PNG" | Output format (PNG/JPG/WEBP) |
| `output_quality` | int | 90 | JPEG/WEBP quality (1-100) |

## Performance Optimizations

Both scripts implement several advanced optimizations:

1. **Para-Attention Caching**: Caches attention computations across inference steps
2. **Torch Compilation**: Uses `torch.compile` with max-autotune mode
3. **Memory Format Optimization**: Uses channels_last format for better performance
4. **QKV Projection Fusion**: Fuses query, key, value projections
5. **Mega-Cache System**: Saves compiled artifacts for faster subsequent loads
6. **FlashAttention 3** (optional): Hardware-optimized attention mechanism
7. **MXFP4 Quantization**: Reduces memory usage while maintaining quality
8. **H100 FP8 Support**: Leverages H100 GPU capabilities for maximum performance
9. **TensorRT Backend** (optional): NVIDIA's high-performance deep learning inference optimizer
10. **Advanced Schedulers**: DPM-Solver++ for fewer steps with better quality

## System Requirements

- **GPU**: CUDA-capable GPU with 24GB+ VRAM (e.g., RTX 3090, A5000, or better)
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for model and cache files

## Output

Generated images are:
- Returned as base64-encoded data URLs in the API response
- Saved to the `outputs/` directory with unique filenames
- Available in PNG, JPG, or WEBP formats

## Using the Modular Components

You can use individual optimization components separately:

```python
from flux_optimizations import FlashAttentionOptimizer, MXFP4Quantizer
from flux_schedulers import SchedulerManager
from flux_compilation import ModelCompiler

# Apply specific optimizations
flash_opt = FlashAttentionOptimizer(is_h100=True)
flash_opt.apply_patch(pipe)

# Setup custom scheduler
scheduler_mgr = SchedulerManager(pipe)
scheduler_mgr.setup_scheduler('dpm_solver')

# Compile with custom settings
compiler = ModelCompiler(pipe)
compiler.compile_model()
```

## Performance Benchmarking

The refactored script includes comprehensive profiling:

```python
from flux_kontext_optimized_refactored import OptimizedFluxService

service = OptimizedFluxService()
service.load_model()

# Run profiled inference
result = service.profile_inference(
    prompt="A detailed landscape",
    output_dir=Path("profiling_results")
)

# View recommendations
print(result['recommendations'])
```

## Troubleshooting

1. **"No HuggingFace token"**: Set `HF_TOKEN` environment variable
2. **Out of memory**: Reduce batch size or image dimensions
3. **Slow first run**: Initial compilation takes time, subsequent runs are faster
4. **Import errors**: Ensure all dependencies are installed via `setup_flux.sh`
5. **flash-attn build failure**: If you encounter a build error with flash-attn, install torch first and use the `--no-build-isolation` flag:
   ```bash
   source flux-env/bin/activate
   uv pip install torch
   uv pip install -r requirements.txt --no-build-isolation
   ```
   This resolves the issue where flash-attn requires torch to be installed before building.
6. **Module import errors**: Ensure all new module files are in the same directory
7. **Compilation conflicts**: Disable para_attention if torch.compile fails
