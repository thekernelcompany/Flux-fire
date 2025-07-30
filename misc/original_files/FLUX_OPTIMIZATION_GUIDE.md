# FLUX.1-Kontext Ultra-Fast Optimization Guide

## Goal: Sub-1 Second Inference

This guide details the most impactful optimizations for achieving sub-1 second inference with FLUX.1-Kontext, ranked by expected performance gain.

## 1. FP8 Quantization (30-40% speedup)

### Why it's impactful:
- **Memory bandwidth reduction**: FP8 uses half the memory of FP16, doubling effective memory bandwidth
- **Compute throughput**: Modern GPUs (H100, RTX 4090) have dedicated FP8 tensor cores with 2x throughput vs FP16
- **Cache efficiency**: More weights fit in L2 cache, reducing memory latency

### Implementation:
```python
import torch
from torch.ao.quantization import quantize_dynamic

# Option 1: Dynamic quantization
pipe.transformer = quantize_dynamic(
    pipe.transformer,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.float8_e4m3fn  # or torch.float8_e5m2
)

# Option 2: Static quantization with calibration
from quanto import quantize, freeze, qfloat8

# Quantize and calibrate
quantize(pipe.transformer, weights=qfloat8, activations=qfloat8)
# Run calibration samples
with torch.no_grad():
    for prompt in calibration_prompts:
        pipe(prompt=prompt, height=512, width=512)
freeze(pipe.transformer)
```

### Trade-offs:
- Minimal quality loss (< 1% in most metrics)
- Requires GPU with FP8 support (Ada Lovelace, Hopper, or newer)

## 2. Advanced Schedulers (50-75% speedup)

### Why it's impactful:
- **Fewer denoising steps**: Reduce from 28 steps to 10-15 with minimal quality loss
- **Direct time reduction**: Linear speedup with step reduction
- **No hardware requirements**: Works on any GPU

### Implementation:
```python
from diffusers import DPMSolverMultistepScheduler, DEISMultistepScheduler

# DPM-Solver++ (recommended)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True
)

# Now use only 10-15 steps instead of 28
result = pipe(prompt=prompt, num_inference_steps=12)
```

### Optimal configurations:
- **DPM-Solver++**: 12-15 steps for high quality
- **DEIS**: 10-12 steps for speed priority
- **UniPC**: 8-10 steps for extreme speed

## 3. Flash Attention 3 (20-30% speedup)

### Why it's impactful:
- **IO efficiency**: Fuses attention operations to minimize memory transfers
- **FP8 native support**: FA3 supports FP8 operations directly
- **Block-sparse patterns**: Automatically optimizes attention patterns

### Implementation:
```python
# Install Flash Attention 3
# pip install flash-attn>=3.0.0

from flash_attn import flash_attn_func
from diffusers.models.attention_processor import Attention

class FlashAttentionProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Prepare Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states or hidden_states)
        value = attn.to_v(encoder_hidden_states or hidden_states)
        
        # Reshape for flash attention
        query = query.view(batch_size, seq_len, attn.heads, -1)
        key = key.view(batch_size, key.shape[1], attn.heads, -1)
        value = value.view(batch_size, value.shape[1], attn.heads, -1)
        
        # Apply Flash Attention 3
        attn_output = flash_attn_func(
            query, key, value,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),  # Use sliding window for long sequences
        )
        
        return attn.to_out(attn_output.reshape(batch_size, seq_len, -1))

# Apply to all attention layers
pipe.transformer.set_attn_processor(FlashAttentionProcessor())
```

## 4. TensorRT Conversion (2x additional speedup)

### Why it's impactful:
- **Kernel fusion**: Automatically fuses hundreds of operations
- **FP8/INT8 optimization**: Native support for reduced precision
- **Hardware optimization**: Tailored to specific GPU architecture

### Implementation:
```python
import torch_tensorrt

# Convert transformer to TensorRT
trt_transformer = torch_tensorrt.compile(
    pipe.transformer,
    inputs=[
        torch_tensorrt.Input(
            shape=(1, 4, 64, 64),  # latent shape
            dtype=torch.float16
        ),
        torch_tensorrt.Input(
            shape=(1, 77, 4096),  # text embedding shape
            dtype=torch.float16
        )
    ],
    enabled_precisions={torch.float8_e4m3fn, torch.float16},
    workspace_size=1 << 30,
    truncate_long_and_double=True,
)

pipe.transformer = trt_transformer
```

## 5. Token Merging (ToMe) (30-50% speedup)

### Why it's impactful:
- **Reduces computational load**: Merges similar tokens, reducing total tokens by 30-50%
- **Maintains quality**: Intelligent merging preserves important details
- **Scales with resolution**: More effective at higher resolutions

### Implementation:
```python
import tome_diffusers

# Apply token merging
tome_diffusers.apply_patch(
    pipe.transformer,
    ratio=0.5,  # Merge 50% of tokens
    max_downsample=2,
    sx=2, sy=2,
    use_rand=False,
    merge_attn=True,
    merge_crossattn=True,
    merge_mlp=True
)
```

## Combined Implementation Strategy

### Phase 1: Software Optimizations (No hardware requirements)
1. Implement advanced schedulers (12 steps)
2. Apply token merging (30% reduction)
3. Enable existing optimizations

**Expected result**: 2.5-3x speedup

### Phase 2: Precision Optimization (Requires modern GPU)
1. Apply FP8 quantization
2. Implement Flash Attention 3
3. Fine-tune memory formats

**Expected result**: Additional 1.5-2x speedup

### Phase 3: Compilation & Deployment (Production ready)
1. Convert to TensorRT
2. Implement dynamic shape optimization
3. Add request batching

**Expected result**: Final 1.5-2x speedup

## Total Expected Speedup: 5-8x

With all optimizations, expected inference times:
- 512x512: 0.6-0.8 seconds
- 768x768: 1.2-1.5 seconds
- 1024x1024: 2.0-2.5 seconds

## Memory Optimization Bonus

```python
# Enable memory efficient attention
pipe.enable_xformers_memory_efficient_attention()

# Use CPU offloading for large batches
pipe.enable_sequential_cpu_offload()

# Optimize CUDA memory allocation
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## Benchmarking Code

```python
def benchmark_optimizations(pipe, prompt, steps=12, runs=5):
    import time
    
    # Warmup
    pipe(prompt=prompt, height=512, width=512, num_inference_steps=steps)
    
    # Benchmark
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        pipe(prompt=prompt, height=512, width=512, num_inference_steps=steps)
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"Throughput: {1/avg_time:.2f} images/second")
    return avg_time
```

## Hardware Recommendations

### Minimum for sub-1s @ 512x512:
- RTX 4070 Ti / RTX 3090
- 16GB VRAM
- PCIe 4.0

### Optimal for consistent sub-1s:
- RTX 4090 / A100 / H100
- 24GB+ VRAM
- NVLink (for multi-GPU)

## Quality vs Speed Trade-offs

| Optimization | Speed Gain | Quality Impact | Reversible |
|-------------|------------|----------------|------------|
| Fewer steps (28→12) | 2.3x | Minimal | Yes |
| FP8 Quantization | 1.4x | < 1% loss | Yes |
| Token Merging 50% | 1.5x | 2-3% loss | Yes |
| Flash Attention | 1.3x | None | Yes |
| TensorRT | 2.0x | None | No |

## Implementation Priority

1. **Start with schedulers** - Immediate 2x+ speedup, no quality loss
2. **Add Flash Attention** - Free performance, no downsides
3. **Implement FP8** - Significant speedup, minimal quality impact
4. **Apply token merging** - Tunable speed/quality trade-off
5. **Deploy with TensorRT** - Maximum performance for production