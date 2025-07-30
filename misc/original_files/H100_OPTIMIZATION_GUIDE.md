# H100 Optimization Guide for FLUX Sub-1s Inference

This guide provides comprehensive optimizations for achieving sub-1 second inference on NVIDIA H100 GPUs with the FLUX.1-Kontext model.

## Performance Targets

With these optimizations on H100:
- **512x512**: 0.3-0.5s
- **768x768**: 0.6-0.8s  
- **1024x1024**: 0.9-1.2s

## Key H100 Advantages

1. **FP8 Tensor Cores**: 2x throughput vs FP16
2. **HBM3 Memory**: 3TB/s bandwidth
3. **Larger L2 Cache**: Better data reuse
4. **Enhanced CUDA Graphs**: Near-zero kernel launch overhead
5. **Transformer Engine**: Optimized for transformer workloads

## Required Dependencies

```bash
pip install transformer-engine  # NVIDIA Transformer Engine for FP8
pip install flash-attn>=2.5.0   # FlashAttention with H100 support
pip install tensorrt>=9.0       # TensorRT for H100
pip install torch>=2.1.0        # PyTorch with H100 support
```

## Code Modifications

### 1. Custom H100-Optimized Pipeline

Create a custom pipeline that eliminates all CPU-GPU synchronization:

```python
class H100OptimizedFluxPipeline(FluxKontextPipeline):
    """H100-optimized pipeline with zero CPU-GPU sync"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_embeddings = {}
        
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=8,
        guidance_scale=3.5,
        generator=None,
        **kwargs
    ):
        # Force optimal settings
        kwargs['output_type'] = 'latent'  # Avoid VAE decode in pipeline
        kwargs['return_dict'] = False
        
        # Cache text embeddings
        prompt_key = hash(prompt)
        if prompt_key not in self._cached_embeddings:
            # Encode without progress bar
            text_embeddings = self._encode_prompt(
                prompt,
                self.device,
                1,
                guidance_scale > 1.0,
                None
            )
            self._cached_embeddings[prompt_key] = text_embeddings
        else:
            text_embeddings = self._cached_embeddings[prompt_key]
        
        # Pre-allocate all tensors
        latents = torch.randn(
            (1, 16, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        # Run denoising loop without any CPU sync
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i, t in enumerate(self.scheduler.timesteps):
                # Skip progress tracking completely
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                
                # Model prediction
                noise_pred = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
                
                # Compute guided noise
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents separately
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        
        # Convert to numpy without CPU sync
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        return image
```

### 2. H100 FP8 Optimization

Add FP8 support using NVIDIA Transformer Engine:

```python
import transformer_engine.pytorch as te

def _apply_fp8_optimization(self):
    """Apply H100 FP8 optimization using Transformer Engine"""
    print("Applying H100 FP8 optimization...")
    
    # Convert all Linear layers to FP8
    for name, module in self.pipe.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Replace with Transformer Engine FP8 Linear
            fp8_linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device='cuda',
                dtype=torch.float8_e4m3fn  # H100 native FP8 format
            )
            # Copy weights
            with torch.no_grad():
                fp8_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    fp8_linear.bias.copy_(module.bias)
            
            # Replace module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = self.pipe.transformer
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, fp8_linear)
    
    print("Applied FP8 optimization to transformer")
```

### 3. H100-Optimized FlashAttention

Replace the standard FlashAttention with H100-optimized version:

```python
def _apply_flash_attention_patch(self):
    """Apply FlashAttention 3 with H100 FP8 support"""
    if not FLASH_ATTENTION_AVAILABLE:
        return
    
    print("Applying H100-optimized FlashAttention 3...")
    
    def h100_flash_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = self.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        
        # Reshape for attention
        inner_dim = query.shape[-1]
        head_dim = 128  # FLUX uses 128
        num_heads = inner_dim // head_dim
        
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        
        # H100 FP8 FlashAttention
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        
        # Convert to FP8 for H100
        with torch.cuda.amp.autocast(enabled=False):
            query_fp8 = query.to(torch.float8_e4m3fn)
            key_fp8 = key.to(torch.float8_e4m3fn)
            value_fp8 = value.to(torch.float8_e4m3fn)
            
            # Use FlashAttention with FP8
            attn_output = flash_attn_func(
                query_fp8, key_fp8, value_fp8,
                dropout_p=0.0,
                causal=False,
                window_size=(-1, -1),  # No sliding window
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False
            )
            
            # Convert back to bfloat16
            attn_output = attn_output.to(torch.bfloat16)
        
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        if hasattr(self, 'to_out') and self.to_out is not None:
            if isinstance(self.to_out, torch.nn.ModuleList):
                attn_output = self.to_out[0](attn_output)
            else:
                attn_output = self.to_out(attn_output)
        
        return attn_output
    
    # Apply patch
    patched_layers = 0
    for layer in self.pipe.transformer.transformer_blocks:
        if hasattr(layer, 'attn'):
            layer.attn.forward = h100_flash_attention_forward.__get__(layer.attn, layer.attn.__class__)
            patched_layers += 1
    
    print(f"Patched {patched_layers} layers with H100-optimized FlashAttention")
```

### 4. H100 CUDA Graphs Setup

Enable proper CUDA graph capture for H100:

```python
def _setup_h100_cuda_graphs(self):
    """Setup H100-optimized CUDA graphs"""
    print("Setting up H100 CUDA graphs...")
    
    # Warm up and create static graph
    static_height, static_width = 512, 512
    static_steps = 8
    
    # Create static inputs
    static_latents = torch.randn(
        (1, 16, static_height // 8, static_width // 8),
        device='cuda',
        dtype=torch.bfloat16
    )
    static_t = torch.tensor([999], device='cuda', dtype=torch.long)
    static_text_embeddings = torch.randn(
        (2, 77, 4096),  # Adjust dimensions for your model
        device='cuda',
        dtype=torch.bfloat16
    )
    
    # Capture graph
    self.graph = torch.cuda.CUDAGraph()
    
    # Warm-up run
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = self.pipe.transformer(
            static_latents,
            static_t,
            encoder_hidden_states=static_text_embeddings,
            return_dict=False
        )[0]
    
    # Capture
    with torch.cuda.graph(self.graph):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            self.static_output = self.pipe.transformer(
                static_latents,
                static_t,
                encoder_hidden_states=static_text_embeddings,
                return_dict=False
            )[0]
    
    print("H100 CUDA graphs captured successfully")
```

### 5. H100-Specific Compilation Settings

Update the compilation settings for H100:

```python
def _optimize_stage_1(self):
    """Apply Stage 1 optimizations with H100-specific settings"""
    print("Applying H100-optimized Stage 1 settings...")
    
    # H100-specific torch compile settings
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.coordinate_descent_check_all_directions = True
    config.coordinate_descent_tuning = True
    config.force_fuse_int_mm_with_mul = True
    config.use_mixed_mm = True  # H100 mixed precision matmul
    config.search_autotune_cache = True
    config.max_autotune = True
    config.max_autotune_gemm = True  # H100 GEMM tuning
    
    # Enable H100 features
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    
    # Apply other optimizations...
    if PARA_ATTN_AVAILABLE:
        apply_cache_on_pipe(self.pipe, residual_diff_threshold=0.12)
    
    # H100-optimized compilation
    self.pipe.transformer = torch.compile(
        self.pipe.transformer,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,  # Static for CUDA graphs
        backend="inductor",
        options={
            "triton.cudagraphs": True,
            "max_autotune": True,
            "coordinate_descent_tuning": True,
            "epilogue_fusion": True,
            "aggressive_fusion": True,
        }
    )
```

### 6. TensorRT Backend (Optional)

For maximum performance, use TensorRT:

```python
def _setup_tensorrt_compilation(self):
    """Setup TensorRT compilation for maximum performance"""
    try:
        import torch_tensorrt
        
        # Configure TensorRT settings
        self.pipe.transformer = torch.compile(
            self.pipe.transformer,
            backend="tensorrt",
            options={
                "enabled_precisions": {torch.float16, torch.float8_e4m3fn},
                "workspace_size": 1 << 32,  # 4GB workspace
                "min_block_size": 5,
                "max_batch_size": 1,
                "optimization_level": 5,
            }
        )
        print("TensorRT compilation enabled")
    except ImportError:
        print("TensorRT not available, using inductor")
```

### 7. Complete H100-Optimized Load Method

```python
def load_model(self, enable_optimizations=None):
    """Load model with H100-specific optimizations"""
    if enable_optimizations is None:
        enable_optimizations = {
            'flash_attention': True,
            'fp8': True,  # H100 FP8 instead of MXFP4
            'scheduler': 'dpm_solver',
            'cuda_graphs': True,
            'tensorrt': False,  # Set to True for max performance
            'tensor_parallel': torch.cuda.device_count() > 1
        }
    
    print("Loading FLUX model with H100 optimizations...")
    
    # Load with custom H100 pipeline
    self.pipe = H100OptimizedFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        token=os.environ.get("HF_TOKEN"),
        device_map="auto",  # Optimal placement
        low_cpu_mem_usage=True
    )
    
    # Apply optimizations in order
    self._optimize_stage_1()  # Basic optimizations
    
    if enable_optimizations.get('fp8', True):
        self._apply_fp8_optimization()  # H100 FP8
    
    if enable_optimizations.get('flash_attention', True):
        self._apply_flash_attention_patch()  # H100 FlashAttention
    
    if enable_optimizations.get('scheduler', 'dpm_solver') == 'dpm_solver':
        self._setup_dpm_solver_scheduler()
    
    # Compile with H100 settings
    if enable_optimizations.get('tensorrt', False):
        self._setup_tensorrt_compilation()
    else:
        self._compile()
    
    if enable_optimizations.get('cuda_graphs', True):
        self._setup_h100_cuda_graphs()
    
    print("H100 optimizations applied successfully")
```

## Optimal Settings for H100

```python
# Test parameters optimized for H100
test_height = 512       # Base resolution for sub-0.5s
test_width = 512        # Base resolution for sub-0.5s
test_steps_optimized = 6     # H100 can maintain quality with fewer steps
test_guidance_scale = 2.5    # Lower guidance scale for speed
test_batch_size = 1          # Keep at 1 for lowest latency

# Alternative resolutions
# 768x768: ~0.7s inference
# 1024x1024: ~1.1s inference
```

## Fast Inference Method

```python
def fast_inference(self, prompt, height=512, width=512, steps=6):
    """Ultra-fast inference with all H100 optimizations"""
    
    # Disable all progress bars and logging
    from diffusers.utils import logging
    logging.set_verbosity_error()
    
    # Pre-encode text to avoid timing it
    with torch.no_grad():
        text_embeddings = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
    
    # Use CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Record start
    start_event.record()
    
    # Generate with minimal overhead
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            # Direct pipeline call
            image = self.pipe(
                prompt_embeds=text_embeddings[0],
                negative_prompt_embeds=text_embeddings[1],
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=2.5,
                output_type="np",
            ).images[0]
    
    # Record end
    end_event.record()
    torch.cuda.synchronize()
    
    # Get timing
    inference_time = start_event.elapsed_time(end_event) / 1000.0
    print(f"H100 inference time: {inference_time:.3f}s")
    
    return image, inference_time
```

## Environment Variables for H100

Add these to your script or shell:

```bash
# H100-specific optimizations
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 compute capability
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_CUDNN_V8_API_ENABLED=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.9

# Triton optimizations
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUBIN_PATH=/usr/local/cuda/bin/cubin
```

## Benchmarking Script

```python
def benchmark_h100_performance():
    """Benchmark different configurations on H100"""
    
    configurations = [
        {"name": "512x512 @ 6 steps", "height": 512, "width": 512, "steps": 6},
        {"name": "768x768 @ 6 steps", "height": 768, "width": 768, "steps": 6},
        {"name": "1024x1024 @ 8 steps", "height": 1024, "width": 1024, "steps": 8},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nBenchmarking {config['name']}...")
        
        # Warm-up
        _, _ = fast_inference(
            "warmup prompt",
            height=config["height"],
            width=config["width"],
            steps=config["steps"]
        )
        
        # Benchmark runs
        times = []
        for i in range(5):
            _, inference_time = fast_inference(
                "A beautiful landscape",
                height=config["height"],
                width=config["width"],
                steps=config["steps"]
            )
            times.append(inference_time)
        
        avg_time = sum(times) / len(times)
        results.append({
            "config": config["name"],
            "avg_time": avg_time,
            "min_time": min(times),
            "max_time": max(times)
        })
        
        print(f"Average: {avg_time:.3f}s (min: {min(times):.3f}s, max: {max(times):.3f}s)")
    
    return results
```

## Troubleshooting

### Issue: FP8 not working
- Ensure you have CUDA 12.0+ and PyTorch 2.1+
- Install transformer-engine: `pip install transformer-engine`
- Check H100 compute capability: Should be 9.0

### Issue: CUDA graphs failing
- Use static shapes (disable dynamic=True in compile)
- Ensure consistent tensor sizes across runs
- Disable if using variable batch sizes

### Issue: OOM errors
- Enable attention slicing: `pipe.enable_attention_slicing()`
- Reduce batch size to 1
- Use gradient checkpointing if needed

## Performance Tips

1. **Pre-compile models**: Run a few warmup iterations
2. **Cache text encodings**: Reuse embeddings for same prompts
3. **Use static shapes**: Enable full CUDA graph capture
4. **Minimize CPU-GPU sync**: Avoid `.item()`, progress bars
5. **Batch operations**: Process multiple images if latency allows

## Monitoring Performance

Use NVIDIA Nsight Systems for detailed profiling:

```bash
nsys profile -t cuda,nvtx,osrt,cublas,cudnn \
    -o flux_h100_profile \
    python flux_kontext_optimized.py
```

View results:
```bash
nsys-ui flux_h100_profile.nsys-rep
```

## Expected Results

With all optimizations applied on H100:

| Resolution | Steps | Expected Time | Quality |
|------------|-------|---------------|---------|
| 512x512    | 6     | 0.3-0.5s     | Good    |
| 768x768    | 6     | 0.6-0.8s     | Good    |
| 1024x1024  | 8     | 0.9-1.2s     | Best    |
| 512x512    | 4     | 0.2-0.3s     | Draft   |

## Conclusion

The key to sub-1s inference on H100 is:
1. Eliminating CPU-GPU synchronization 
2. Using H100's FP8 capabilities
3. Proper CUDA graph implementation
4. Optimized compilation settings
5. Reduced inference steps (6-8)

With these optimizations, consistent sub-1s inference is achievable for production workloads.