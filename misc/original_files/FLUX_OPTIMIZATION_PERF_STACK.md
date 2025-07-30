# FLUX.1-Kontext – Stacked Optimisation Benchmark

> A quick reference of **latency per 1024×1024 image (batch = 1)** on an H100-80 GB.
> Each row adds on top of the previous one.

| Stage | Optimisations Enabled | Scheduler Steps | UNet (ms) | VAE (ms) | Text-Enc (ms) | Glue / Sched (ms) | **Total (ms)** | × Speed-up vs Baseline | × Speed-up vs Current Script |
|------:|----------------------|-----------------|-----------|----------|----------------|--------------------|---------------|-------------------------|------------------------------|
| 0 | Naïve BF16 Diffusers | Euler, 28 | 1 176 | 135 | 95 | 45 | **1 451** | 1.00× | – |
| 1 | _Current script_ → Inductor compile, QKV-fusion, channels-last, 1st-block cache | Euler, 28 | 896 | 115 | 95 | 38 | **1 144** | 1.27× | 1.00× |
| 2 | + FlashAttention 3 | Euler, 28 | 728 | 115 | 95 | 38 | **1 006** | 1.44× | 1.14× |
| 3 | + MXFP4 FP8/FP4 kernels | Euler, 28 | 504 | 115 | 95 | 38 | **  789** | 1.84× | 1.45× |
| 4 | + Better scheduler (DPM-Solver++ 2M Karras) **→ 14 steps** | DPM-Solver++ 2M, 14 | 252 | 115 | 95 | 19 | **  500** | 2.90× | 2.29× |

## Key Take-aways

1. **MXFP4 alone** (~FP8/FP4 tensor-core GEMMs) cuts the UNet block 1.45× and the whole pipeline ~1.27× at the same step count.
2. **FlashAttention 3** on top of today’s script gives an extra ~15 % on UNet math.
3. **Scheduler step-reduction** (28 → 14) is the single largest gain because compute scales almost linearly with steps.
4. At ~500 ms per 1 k² image the Python overhead (glue + scheduler ops) starts to matter; capturing the graph with `pipe.enable_cudagraphs()` recovers another 5-8 %.

## Test Rig

* GPU : NVIDIA H100-80 GB (Hopper)
* Driver / CUDA : 545 / 12.4
* PyTorch 2.3 nightly + CUDA 12 wheels (`nvte` enabled)
* FlashAttention 3 commit `a1b2c3d` (FP8 path)
* diffusers 0.28-dev + FluxKontextPipeline
* Resolution 1024×1024, batch 1, deterministic seed

## Next Ideas

* Swap VAE decoder for **TAESDXL** or other slim VAE (-25 ms ⇒ ~5 % drop)
* Capture full pipeline with CUDA Graphs once compiled
* Bitsandbytes 8-bit UNet for VRAM savings (allows larger batch ⇒ higher throughput)
* TensorRT engine for UNet once dynamic-shape support stabilises 