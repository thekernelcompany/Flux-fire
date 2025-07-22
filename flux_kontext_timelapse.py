#!/usr/bin/env python3
"""
FLUX.1 Kontext timelapse benchmark script
========================================
This standalone script replicates the benchmarking logic from
`flux_kontext_standalone.py` but additionally captures the latent
image produced at **every diffusion step**. These intermediate frames
are stitched together into an animated GIF whose frame duration is
chosen according to the actual per-step execution time. Two GIFs are
produced:
    • outputs/unoptimized_timelapse.gif  – baseline pipeline (no extra optimisations)
    • outputs/optimized_timelapse.gif    – pipeline after FB-cache, qkv fusion, TorchInductor, …

The core mechanics rely on the Diffusers `callback` / `callback_steps`
API. A `StepCollector` object measures execution time and converts
latents coming from the callback into `PIL.Image` frames via
`pipeline.decode_latents(...)` (or a manual VAE decode fallback).

Run this script directly (`python flux_kontext_timelapse.py`) to
produce the timelapses and a textual performance summary.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import FluxKontextPipeline

# Re-use the optimisation utilities from the original standalone service
from flux_kontext_standalone import FluxService, InferenceRequest, OutputFormat  # type: ignore

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _latents_to_pil(pipe: FluxKontextPipeline, latents: torch.Tensor) -> Image.Image:
    """Convert latents from callback to a PIL image (first item in batch)."""
    # Newer diffusers pipelines expose `decode_latents` – prefer that
    if hasattr(pipe, "decode_latents"):
        np_img = pipe.decode_latents(latents)[0]
    else:  # manual VAE decode fallback
        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        np_img = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    np_img = (np.clip(np_img * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(np_img)


class StepCollector:
    """Collect per-step timings and images via the diffusers callback API."""

    def __init__(self, pipe: FluxKontextPipeline):
        self.pipe = pipe
        self.images: List[Image.Image] = []
        # Per-step execution times in seconds; we time **every** callback
        # including the very first one so that the sum of all durations
        # represents the full denoising runtime.
        self.times: List[float] = []
        self._last_t = time.perf_counter()

    def __call__(self, step: int, timestep: int, latents: torch.Tensor):  # noqa: D401
        now = time.perf_counter()
        # Record time elapsed since previous callback (or pipeline start)
        self.times.append(now - self._last_t)
        self._last_t = now

        try:
            pil_img = _latents_to_pil(self.pipe, latents)
            self.images.append(pil_img)
        except Exception as exc:
            print(f"[WARNING] Failed to convert latents at step {step}: {exc}")


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

def run_with_timelapse(pipe: FluxKontextPipeline, req: InferenceRequest, gif_prefix: str) -> dict:
    """Run the pipeline while collecting a timelapse GIF.

    Returns dict containing: final_images, per_step_times, total_inference_time, gif_path
    """
    collector = StepCollector(pipe)
    generator = (
        torch.Generator(pipe.device).manual_seed(req.seed) if req.seed is not None else None
    )

    t0 = time.perf_counter()
    pipe(
        prompt=req.prompt,
        prompt_2=req.prompt2,
        negative_prompt=req.negative_prompt,
        negative_prompt_2=req.negative_prompt2,
        true_cfg_scale=req.true_cfg_scale,
        height=req.height,
        width=req.width,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        num_images_per_prompt=req.num_images,
        generator=generator,
        callback=collector,
        callback_steps=1,
        output_type="np",  # we only care about final ndarray, callback handles frames
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t0

    # Build GIF whose total duration matches the measured inference time
    if collector.images:
        # Ensure per-frame durations sum exactly to total_time
        durations_sec = collector.times.copy()
        measured_sum = sum(durations_sec)
        residual = total_time - measured_sum
        if residual > 0 and durations_sec:
            # Add the (usually small) residual to the last frame so
            # that Σ durations == total_time.
            durations_sec[-1] += residual

        durations_ms = [max(1, int(t * 1000)) for t in durations_sec]

        gif_name = f"{gif_prefix}_{uuid.uuid4().hex[:8]}.gif"
        gif_path = OUTPUT_DIR / gif_name
        collector.images[0].save(
            gif_path,
            save_all=True,
            append_images=collector.images[1:],
            duration=durations_ms,
            loop=0,
            optimize=False,
        )
    else:
        gif_path = None

    return {
        "per_step_times": collector.times,
        "total_inference_time": total_time,
        "gif_path": str(gif_path) if gif_path else None,
    }


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------
    # Test configuration (same as original script)
    # -----------------------------------
    test_req = InferenceRequest(
        prompt="A majestic mountain landscape at sunset with vibrant colors",
        height=1024,
        width=1024,
        steps=20,
        guidance_scale=3.5,
        seed=42,
        num_images=1,
        output_format=OutputFormat.PNG,
    )

    print("\n=== FLUX.1 Kontext Timelapse Benchmark ===\n")
    print(f"Test config: {test_req.width}x{test_req.height}, {test_req.steps} steps\n")

    # -----------------------------------
    # Unoptimised pipeline run
    # -----------------------------------
    print("--- Unoptimised pipeline ---")
    hf_token = os.environ.get("HF_TOKEN")
    pipe_unopt = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        token=hf_token,
    )
    if torch.cuda.is_available():
        pipe_unopt = pipe_unopt.to("cuda")
    else:
        print("[WARNING] Running unoptimised pipeline on CPU – will be very slow.")

    unopt_result = run_with_timelapse(pipe_unopt, test_req, "unoptimized_timelapse")

    print(
        f"Unoptimised total time: {unopt_result['total_inference_time']:.2f}s "
        f"(GIF: {unopt_result['gif_path']})"
    )

    # Release unoptimised resources before loading optimised variant
    del pipe_unopt
    torch.cuda.empty_cache()

    # -----------------------------------
    # Optimised pipeline run
    # -----------------------------------
    print("\n--- Optimised pipeline ---")
    service_opt = FluxService()
    service_opt.load_model()  # this applies all optimisation passes & compilation

    opt_result = run_with_timelapse(service_opt.pipe, test_req, "optimized_timelapse")

    print(
        f"Optimised total time:   {opt_result['total_inference_time']:.2f}s "
        f"(GIF: {opt_result['gif_path']})"
    )

    # -----------------------------------
    # Summary
    # -----------------------------------
    print("\n=== PERFORMANCE SUMMARY ===")
    u_time = unopt_result['total_inference_time']
    o_time = opt_result['total_inference_time']

    print(f"Unoptimised: {u_time:.2f}s")
    print(f"Optimised:   {o_time:.2f}s")
    if o_time > 0:
        print(f"Speed-up:    {u_time / o_time:.2f}× faster") 