#!/usr/bin/env python3
"""
Run benchmarks for FLUX.1-Kontext optimizations
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.service import OptimizedFluxService
from src.utils.flux_utils import BenchmarkRunner


def main():
    """Run benchmark comparison"""
    
    # Test parameters
    test_prompt = "A majestic mountain landscape at sunset with vibrant colors"
    test_height = 1024
    test_width = 1024
    test_steps_optimized = 14  # DPM-Solver++ optimal
    test_steps_baseline = 28   # Euler baseline
    test_seed = 42
    
    print("=== FLUX.1-Kontext Benchmark Comparison ===\n")
    print(f"Test prompt: '{test_prompt}'")
    print(f"Resolution: {test_width}x{test_height}")
    print(f"Seed: {test_seed}")
    print()
    
    # Benchmark different optimization stages
    optimization_configs = [
        {
            "name": "Baseline (Minimal optimizations)", 
            "config": {"flash_attention": False, "mxfp4": False, "scheduler": None, "cuda_graphs": False},
            "steps": test_steps_baseline
        },
        {
            "name": "Stage 1 (Compilation + Para-attention)", 
            "config": {"flash_attention": False, "mxfp4": False, "scheduler": None, "cuda_graphs": False, "para_attention": True},
            "steps": test_steps_baseline
        },
        {
            "name": "Stage 1+2 (+ FlashAttention)", 
            "config": {"flash_attention": True, "mxfp4": False, "scheduler": None, "cuda_graphs": False},
            "steps": test_steps_baseline
        },
        {
            "name": "Stage 1+2+3 (+ MXFP4 Quantization)", 
            "config": {"flash_attention": True, "mxfp4": True, "scheduler": None, "cuda_graphs": False},
            "steps": test_steps_baseline
        },
        {
            "name": "Stage 1+2+3+4 (+ DPM-Solver++)", 
            "config": {"flash_attention": True, "mxfp4": True, "scheduler": "dpm_solver", "cuda_graphs": False},
            "steps": test_steps_optimized
        },
        {
            "name": "Full Stack (All optimizations)", 
            "config": {"flash_attention": True, "mxfp4": True, "scheduler": "dpm_solver", "cuda_graphs": True, "para_attention": False},
            "steps": test_steps_optimized
        }
    ]
    
    results = BenchmarkRunner.run_benchmark_comparison(
        OptimizedFluxService, test_prompt, test_height, test_width, test_seed, optimization_configs
    )
    
    # Print summary
    BenchmarkRunner.print_benchmark_summary(results)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()