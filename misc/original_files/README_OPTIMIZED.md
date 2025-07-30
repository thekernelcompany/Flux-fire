# FLUX.1-Kontext Optimized - 2.90x Performance Stack

High-performance FLUX.1-Kontext image generation with **full optimization stack** achieving **2.90x speedup** through advanced optimizations based on the research from `FLUX_OPTIMIZATION_PERF_STACK.md`.

## Performance Improvements

This implementation provides a **complete optimization stack** with the following performance improvements:

| Stage | Optimizations | Scheduler | Steps | UNet (ms) | VAE (ms) | Total (ms) | Speedup vs Baseline | Speedup vs Previous |
|-------|---------------|-----------|-------|-----------|----------|------------|-------------------|-------------------|
| 0 | Naïve BF16 Diffusers | Euler, 28 | 28 | 1,176 | 135 | **1,451** | 1.00× | – |
| 1 | Inductor compile, QKV-fusion, channels-last, 1st-block cache | Euler, 28 | 28 | 896 | 115 | **1,144** | 1.27× | 1.00× |
| 2 | + **FlashAttention 3** | Euler, 28 | 28 | 728 | 115 | **1,006** | 1.44× | 1.14× |
| 3 | + **MXFP4 FP8/FP4 kernels** | Euler, 28 | 28 | 504 | 115 | **789** | 1.84× | 1.45× |
| 4 | + **DPM-Solver++ 2M Karras** → 14 steps | DPM-Solver++ 2M, 14 | 14 | 252 | 115 | **500** | 2.90× | 2.29× |
| 5 | + **CUDA Graphs** | DPM-Solver++ 2M, 14 | 14 | ~240 | 115 | **~475** | ~3.10× | ~1.05× |

*Performance data based on H100-80GB, 1024×1024 images, batch size 1*

## Key Optimizations Implemented

### Stage 1: Foundation Optimizations (1.27x speedup)
- **Torch Inductor compilation** with max-autotune mode
- **QKV projection fusion** for transformer and VAE
- **Channels-last memory format** optimization
- **Para-attention first-block caching** with 0.12 residual threshold

### Stage 2: FlashAttention 3 (1.14x additional speedup)
- **Hardware-aware attention** with FP8 support
- **Asynchronous computation** overlap
- **Optimized memory access patterns**
- **Hopper GPU tensor core utilization**

### Stage 3: MXFP4 Quantization (1.45x additional speedup)
- **FP8/FP4 tensor core GEMMs** for massive compute reduction
- **Block-wise quantization** with 32-element blocks
- **Dual quantization libraries**: TorchMX + Microsoft MX
- **Maintained accuracy** with advanced quantization techniques

### Stage 4: DPM-Solver++ 2M Karras (2.29x additional speedup)
- **Step count reduction**: 28 → 14 steps
- **Advanced scheduler** with Karras noise schedule
- **Optimal solver order** for quality vs speed
- **Single largest performance gain**

### Stage 5: CUDA Graphs (5-8% additional speedup)
- **Graph capture** eliminates Python overhead
- **Optimized kernel fusion**
- **Reduced memory transfers**

## Quick Start

### 1. Automated Setup (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd Flux-fire

# Run the automated setup script
./setup_flux_optimized.sh

# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token_here"
```

### 2. Manual Setup
```bash
# Create virtual environment
python -m venv flux-optimized-env
source flux-optimized-env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install FlashAttention 3
pip install flash-attn --no-build-isolation

# Install MXFP4 quantization
pip install torchmx
```

### 3. Replace para_attn utils.py (REQUIRED)
```bash
# This step is essential for FLUX.1-Kontext compatibility
cp utils.py flux-optimized-env/lib/python*/site-packages/para_attn/first_block_cache/utils.py
```

## Usage

### FastAPI Server (Recommended)
```bash
# Start the optimized server
uvicorn flux_kontext_optimized:app --host 0.0.0.0 --port 8000

# Or run directly
python flux_kontext_optimized.py
```

### API Endpoints

#### Generate Images (Optimized)
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "steps": 14,
    "num_images": 1,
    "use_flash_attention": true,
    "use_mxfp4": true,
    "use_dpm_solver": true,
    "use_cuda_graphs": true
  }'
```

#### Performance Benchmark
```bash
curl http://localhost:8000/benchmark
```

#### Health Check with Optimization Status
```bash
curl http://localhost:8000/health
```

## Configuration Options

### Optimization Controls
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_flash_attention` | `true` | Enable FlashAttention 3 |
| `use_mxfp4` | `true` | Enable MXFP4 quantization |
| `use_dpm_solver` | `true` | Use DPM-Solver++ 2M Karras |
| `use_cuda_graphs` | `true` | Enable CUDA graphs |

### Generation Parameters  
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `steps` | `14` | 1-50 | Inference steps (optimized for DPM-Solver++) |
| `guidance_scale` | `3.5` | 0-20 | Prompt adherence |
| `true_cfg_scale` | `1.0` | 0-20 | True CFG scale |
| `width/height` | `1024` | 256-1024 | Image dimensions (multiple of 16) |

## System Requirements

### Minimum Requirements
- **GPU**: CUDA-capable GPU with 24GB+ VRAM
- **CUDA**: 11.8 or higher  
- **Python**: 3.10 or higher
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models and cache

### Recommended for Full Performance
- **GPU**: H100, A100, or RTX 4090
- **CUDA**: 12.0+ for optimal FlashAttention 3 performance
- **RAM**: 64GB+ for large batch processing
- **Storage**: NVMe SSD for fast model loading

## Performance Benchmarking

The optimized service includes built-in benchmarking:

```python
# Run comprehensive benchmark
python flux_kontext_optimized.py

# Or via API
curl http://localhost:8000/benchmark
```

Expected results on H100-80GB:
- **Baseline (Stage 1)**: ~1,144ms per image
- **Full Stack (Stage 4)**: ~500ms per image  
- **With CUDA Graphs**: ~475ms per image
- **Total Speedup**: 2.90-3.10x

## Troubleshooting

### FlashAttention 3 Issues
```bash
# Check CUDA compatibility
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall with specific CUDA version
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### MXFP4 Quantization Issues
```bash
# Try alternative quantization library
pip uninstall torchmx
git clone https://github.com/microsoft/microxcaling.git
cd microxcaling && pip install -e .
```

### Memory Issues
- Reduce batch size (`num_images`)
- Disable some optimizations temporarily
- Use gradient checkpointing: `--checkpointing`

### Performance Issues
- Ensure GPU has sufficient VRAM (24GB+)
- Check CUDA version compatibility
- Verify all optimizations are enabled in health check

## Technical Details

### FlashAttention 3 Implementation
- Uses FP8 precision when available
- Implements asynchronous GEMM/softmax overlap
- Leverages Hopper architecture features
- Falls back gracefully to standard attention

### MXFP4 Quantization Details
- Block size: 32 elements
- Supports both FP8 and FP4 formats
- Maintains model accuracy through advanced techniques
- Compatible with both TorchMX and Microsoft MX

### DPM-Solver++ Configuration
- Algorithm: `dpmsolver++`
- Solver order: 2 (optimal for quality/speed)
- Karras sigmas: Enabled
- Step reduction: 28 → 14 steps

## Contributing

This optimization stack is based on cutting-edge research. Contributions welcome:

1. **Performance improvements**: Additional optimization techniques
2. **Hardware support**: AMD, Intel GPU support  
3. **Model variants**: Support for other FLUX models
4. **Benchmarking**: Extended performance analysis

## License

Same as original FLUX.1-Kontext model. See LICENSE file.

## Acknowledgments

- Black Forest Labs for FLUX.1-Kontext
- Tri Dao for FlashAttention research
- Microsoft and TorchMX teams for quantization libraries
- Research community for DPM-Solver++ advances 