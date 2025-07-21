# FLUX.1-Kontext Standalone Server

High-performance FLUX.1-Kontext image generation with automatic performance benchmarking and 2.63x speedup through advanced optimizations.

## About flux_kontext_standalone.py

The `flux_kontext_standalone.py` script is a complete, self-contained server that:
- **Automatically benchmarks performance** on first run, comparing optimized vs unoptimized inference
- **Runs a FastAPI server** for REST API access to the model
- **Implements cutting-edge optimizations** including para-attention caching and torch compilation
- **Manages caching** for faster subsequent startups
- **Supports both text-to-image and image-to-image generation**

## Important Setup Requirement

**CRITICAL**: Before running `flux_kontext_standalone.py`, you MUST replace the utils.py file in the para_attn package:

```bash
# Replace the para_attn utils.py with the one from this repo
cp utils.py flux-env/lib/python*/site-packages/para_attn/first_block_cache/utils.py
```

This step is essential for FLUX.1-Kontext compatibility. The provided `utils.py` includes necessary modifications for the model to work correctly.

## Quick Start

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

The standalone script implements several advanced optimizations:

1. **Para-Attention Caching**: Caches attention computations across inference steps
2. **Torch Compilation**: Uses `torch.compile` with max-autotune mode
3. **Memory Format Optimization**: Uses channels_last format for better performance
4. **QKV Projection Fusion**: Fuses query, key, value projections
5. **Mega-Cache System**: Saves compiled artifacts for faster subsequent loads

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

## Troubleshooting

1. **"No HuggingFace token"**: Set `HF_TOKEN` environment variable
2. **Out of memory**: Reduce batch size or image dimensions
3. **Slow first run**: Initial compilation takes time, subsequent runs are faster
4. **Import errors**: Ensure all dependencies are installed via `setup_flux.sh`
