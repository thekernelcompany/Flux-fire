# FLUX.1-Kontext - Professional Project Structure

High-performance FLUX.1-Kontext image generation with modular architecture and advanced optimizations achieving up to 3.5x speedup.

## 🚀 New Professional Structure

The project has been refactored into a clean, modular architecture:

```
Flux-fire/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   └── service.py     # Main FLUX service implementation
│   ├── optimizations/     # Optimization modules
│   │   ├── attention.py   # FlashAttention 3 optimization
│   │   ├── quantization.py # MXFP4 & H100 FP8 quantization
│   │   ├── tensorrt.py    # TensorRT compilation
│   │   ├── dtype.py       # Data type optimization
│   │   ├── schedulers.py  # Scheduler management & CUDA graphs
│   │   └── compilation.py # torch.compile & mega-cache
│   ├── utils/             # Utility functions
│   │   └── flux_utils.py  # Profiling, benchmarking, image processing
│   └── api/               # API implementation
│       ├── models.py      # Pydantic models
│       └── server.py      # FastAPI server
├── scripts/               # Executable scripts
│   ├── run_server.py     # Start the API server
│   ├── generate_image.py # CLI image generation
│   └── benchmark.py      # Run performance benchmarks
├── config/               # Configuration
│   └── settings.py       # Application settings
├── examples/             # Example usage
├── tests/               # Unit tests
├── docs/                # Documentation
├── outputs/             # Generated images
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore rules
├── requirements.txt     # Python dependencies
└── README_ME.md           # This file
```

## 🎯 Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
git clone <repository>
cd Flux-fire

# Setup virtual environment
./setup_flux.sh
source flux-env/bin/activate

# CRITICAL: Replace para_attn utils.py
cp utils.py flux-env/lib/python*/site-packages/para_attn/first_block_cache/utils.py

# Copy and configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 2. Use the Scripts

#### Run API Server
```bash
python scripts/run_server.py

# Or with custom settings
HOST=0.0.0.0 PORT=8080 python scripts/run_server.py
```

#### Generate Images (CLI)
```bash
# Basic usage
python scripts/generate_image.py "A beautiful sunset over mountains"

# With options
python scripts/generate_image.py "A majestic dragon" \
  --width 768 \
  --height 768 \
  --steps 20 \
  --num-images 2 \
  --seed 42

# Disable specific optimizations
python scripts/generate_image.py "A futuristic city" \
  --no-flash-attention \
  --scheduler euler
```

#### Run Benchmarks
```bash
python scripts/benchmark.py
```

## 📦 Module Overview

### Core Module (`src/core/`)
- **`service.py`**: Main `OptimizedFluxService` class that orchestrates all optimizations

### Optimization Modules (`src/optimizations/`)
- **`attention.py`**: FlashAttention 3 implementation
- **`quantization.py`**: MXFP4 and H100 FP8 quantization
- **`tensorrt.py`**: TensorRT backend compilation
- **`dtype.py`**: Model data type management
- **`schedulers.py`**: DPM-Solver++, Euler schedulers, and CUDA graphs
- **`compilation.py`**: torch.compile configuration and mega-cache

### Utilities (`src/utils/`)
- **`flux_utils.py`**: Shared utilities for profiling, benchmarking, image processing

### API (`src/api/`)
- **`server.py`**: FastAPI server implementation
- **`models.py`**: Request/response models

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
HF_TOKEN=your_huggingface_token

# Server
HOST=0.0.0.0
PORT=8000

# Optimizations
ENABLE_FLASH_ATTENTION=false
ENABLE_MXFP4=true
ENABLE_TENSORRT=false
SCHEDULER_TYPE=dpm_solver

# Paths
FLUX_CACHE_DIR=/path/to/cache
```

### Python Configuration

```python
from config.settings import settings

# Access settings
print(settings.MODEL_ID)
print(settings.get_optimization_config())
print(settings.is_h100)
```

## 🚀 API Usage

### Start Server
```bash
python scripts/run_server.py
```

### Generate Images
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene lake at dawn",
    "width": 1024,
    "height": 1024,
    "steps": 14,
    "num_images": 1
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## 🧩 Using Individual Components

```python
# Import specific optimizers
from src.optimizations.attention import FlashAttentionOptimizer
from src.optimizations.quantization import MXFP4Quantizer
from src.optimizations.schedulers import SchedulerManager

# Use in your own pipeline
flash_opt = FlashAttentionOptimizer(is_h100=True)
flash_opt.apply_patch(your_pipe)

scheduler_mgr = SchedulerManager(your_pipe)
scheduler_mgr.setup_scheduler('dpm_solver')
```

## 📊 Performance Optimizations

The modular structure implements these optimizations:

1. **torch.compile** with inductor backend
2. **Para-attention** first-block caching
3. **FlashAttention 3** (optional)
4. **MXFP4 Quantization** for memory efficiency
5. **H100 FP8** optimization (when available)
6. **DPM-Solver++** for fewer inference steps
7. **CUDA graphs** for kernel launch optimization
8. **TensorRT** backend (optional)
9. **Channels-last** memory format
10. **QKV projection fusion**

## 🔍 Profiling & Benchmarking

```python
from src.core.service import OptimizedFluxService

service = OptimizedFluxService()
service.load_model()

# Run with profiling
result = service.profile_inference(
    prompt="A detailed landscape",
    output_dir=Path("profiling_results")
)

# View bottlenecks and recommendations
print(result['recommendations'])
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_optimizations.py

# With coverage
pytest --cov=src tests/
```

## 📝 Development

### Adding New Optimizations

1. Create new file in `src/optimizations/`
2. Implement optimizer class with standard interface
3. Import in `src/core/service.py`
4. Add configuration option in `config/settings.py`

### Project Standards

- Use type hints for all functions
- Follow PEP 8 style guide
- Add docstrings to all classes and functions
- Keep modules focused and single-purpose
- Write tests for new features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-optimization`)
3. Commit changes (`git commit -m 'Add amazing optimization'`)
4. Push to branch (`git push origin feature/amazing-optimization`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- FLUX.1-Kontext model by Black Forest Labs
- Optimization techniques from various research papers
- Community contributions and feedback