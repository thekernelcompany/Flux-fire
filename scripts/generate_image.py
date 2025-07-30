#!/usr/bin/env python3
"""
Generate images using FLUX.1-Kontext from command line
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.service import OptimizedFluxService


def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX.1-Kontext")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--steps", type=int, default=14, help="Number of inference steps (default: 14)")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-flash-attention", action="store_true", help="Disable FlashAttention optimization")
    parser.add_argument("--no-mxfp4", action="store_true", help="Disable MXFP4 quantization")
    parser.add_argument("--scheduler", type=str, default="dpm_solver", choices=["dpm_solver", "euler"], 
                       help="Scheduler type (default: dpm_solver)")
    
    args = parser.parse_args()
    
    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Resolution: {args.width}x{args.height}, Steps: {args.steps}")
    
    # Initialize service
    service = OptimizedFluxService()
    
    # Load model with optimizations
    enable_optimizations = {
        'flash_attention': not args.no_flash_attention,
        'mxfp4': not args.no_mxfp4,
        'scheduler': args.scheduler,
        'cuda_graphs': True,
        'para_attention': False,  # Disabled by default
        'tensorrt': False
    }
    
    print("Loading model...")
    service.load_model(enable_optimizations=enable_optimizations)
    
    # Generate images
    print("Generating images...")
    result = service.inference(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images=args.num_images,
        seed=args.seed
    )
    
    print(f"\nGeneration complete!")
    print(f"Inference time: {result['inference_time']:.2f}s")
    print(f"Images saved to: {', '.join(result['saved_files'])}")


if __name__ == "__main__":
    main()