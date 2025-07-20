# FLUX.1-Kontext Image Generation

High-performance text-to-image generation using FLUX.1-Kontext model with 2.63x speedup through compilation and parallel attention optimizations.

## Features

- **Fast inference**: Para-attention caching and torch compilation
- **REST API**: FastAPI server for easy integration
- **Cloud deployment**: Modal.com scripts with Cloudflare R2 storage
- **Flexible generation**: Text-to-image and image-to-image support
- **Batch processing**: Generate 1-4 images per request

## Quick Start

```bash
# Setup environment
./setup_flux.sh

# Activate environment
source flux-env/bin/activate

# Run server
python flux_kontext_standalone.py
```

## API Usage

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "num_images": 1,
    "width": 1024,
    "height": 1024
  }'
```

## Key Parameters

- `prompt`: Main text description
- `prompt2`: Additional prompt (optional)
- `negative_prompt`: What to avoid
- `num_inference_steps`: Quality vs speed (default: 50)
- `guidance_scale`: Prompt adherence (default: 7.5)
- `seed`: Reproducible generation

## Requirements

- CUDA-capable GPU (24GB+ VRAM recommended)
- Python 3.10+
- HuggingFace token (set in .env)

## Performance

Optimizations include:
- Para-attention caching across inference steps
- Torch compilation with max-autotune
- Memory format optimization
- QKV projection fusion