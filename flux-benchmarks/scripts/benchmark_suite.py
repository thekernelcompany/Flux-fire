#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for FLUX.1-Kontext
Measures performance, quality, and cost metrics for blog post analysis
"""

import os
import sys
import time
import json
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from collections import defaultdict
import statistics

# Import the flux service
from flux_kontext_standalone import FluxService, InferenceRequest

@dataclass
class BenchmarkResult:
    """Store individual benchmark results"""
    test_name: str
    resolution: Tuple[int, int]
    steps: int
    batch_size: int
    optimization_level: str
    inference_time: float
    total_time: float
    memory_used: float
    memory_allocated: float
    gpu_utilization: float
    temperature: float
    power_draw: float
    prompt: str
    seed: int
    timestamp: datetime

@dataclass
class OptimizationConfig:
    """Configuration for different optimization levels"""
    name: str
    use_compile: bool
    use_para_attn: bool
    use_channels_last: bool
    use_qkv_fusion: bool
    use_mega_cache: bool

class GPUMonitor:
    """Monitor GPU metrics during inference"""
    
    def __init__(self):
        self.has_nvidia_ml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_nvidia_ml = True
            self.pynvml = pynvml
        except:
            print("Warning: pynvml not available, GPU monitoring limited")
    
    def get_metrics(self) -> Dict:
        """Get current GPU metrics"""
        metrics = {
            'memory_used': 0,
            'memory_allocated': 0,
            'utilization': 0,
            'temperature': 0,
            'power_draw': 0
        }
        
        # PyTorch memory stats
        if torch.cuda.is_available():
            metrics['memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics['memory_used'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # NVIDIA ML stats
        if self.has_nvidia_ml:
            try:
                metrics['utilization'] = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                metrics['temperature'] = self.pynvml.nvmlDeviceGetTemperature(self.handle, self.pynvml.NVML_TEMPERATURE_GPU)
                metrics['power_draw'] = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Watts
            except:
                pass
        
        return metrics

class BenchmarkSuite:
    """Comprehensive benchmarking suite for FLUX.1-Kontext"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.gpu_monitor = GPUMonitor()
        
        # Test prompts for diversity
        self.test_prompts = [
            "A serene mountain landscape at sunset with vibrant colors",
            "A futuristic cyberpunk cityscape with neon lights and flying cars",
            "A detailed portrait of a wise elderly wizard with flowing beard",
            "An abstract composition of geometric shapes in bold colors",
            "A photorealistic still life of fresh fruits on a wooden table",
            "A whimsical underwater scene with colorful coral and tropical fish",
            "A dramatic storm over an ancient castle on a cliff",
            "A minimalist architectural interior with natural lighting",
            "A dense jungle with exotic plants and hidden temples",
            "A cozy coffee shop interior with warm lighting and vintage decor"
        ]
        
        # Optimization configurations
        self.optimization_configs = [
            OptimizationConfig("baseline", False, False, False, False, False),
            OptimizationConfig("compile_only", True, False, False, False, False),
            OptimizationConfig("para_attn_only", False, True, False, False, False),
            OptimizationConfig("channels_last_only", False, False, True, False, False),
            OptimizationConfig("qkv_fusion_only", False, False, False, True, False),
            OptimizationConfig("standard_opts", True, False, True, True, False),
            OptimizationConfig("full_optimization", True, True, True, True, True)
        ]
    
    def create_optimized_service(self, config: OptimizationConfig) -> FluxService:
        """Create a FluxService with specific optimizations"""
        service = FluxService()
        
        # Override optimization methods based on config
        original_optimize = service._optimize
        original_compile = service._compile
        original_load_cache = service._load_mega_cache
        original_save_cache = service._save_mega_cache
        
        def custom_optimize(self):
            if config.use_para_attn:
                from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
                apply_cache_on_pipe(self.pipe, residual_diff_threshold=0.12)
            
            if config.use_qkv_fusion:
                self.pipe.transformer.fuse_qkv_projections()
                self.pipe.vae.fuse_qkv_projections()
            
            if config.use_channels_last:
                self.pipe.transformer.to(memory_format=torch.channels_last)
                self.pipe.vae.to(memory_format=torch.channels_last)
            
            if config.use_compile:
                # Configure torch compile
                config_torch = torch._inductor.config
                config_torch.conv_1x1_as_mm = True
                config_torch.coordinate_descent_check_all_directions = True
                config_torch.coordinate_descent_tuning = True
                config_torch.disable_progress = False
                config_torch.epilogue_fusion = False
                config_torch.shape_padding = True
                
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=True
                )
                
                self.pipe.vae.decode = torch.compile(
                    self.pipe.vae.decode,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=True
                )
        
        def custom_compile(self):
            if config.use_compile:
                original_compile()
        
        def custom_load_cache(self):
            if config.use_mega_cache:
                original_load_cache()
        
        def custom_save_cache(self):
            if config.use_mega_cache:
                original_save_cache()
        
        # Apply custom methods
        service._optimize = lambda: custom_optimize(service)
        service._compile = lambda: custom_compile(service)
        service._load_mega_cache = lambda: custom_load_cache(service)
        service._save_mega_cache = lambda: custom_save_cache(service)
        
        return service
    
    def warmup_gpu(self):
        """Warm up GPU to stable temperature"""
        print("Warming up GPU...")
        dummy = torch.randn(1000, 1000, device='cuda')
        for _ in range(100):
            dummy = torch.matmul(dummy, dummy)
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()
        time.sleep(5)  # Let GPU stabilize
    
    def run_single_benchmark(
        self, 
        service: FluxService,
        request: InferenceRequest,
        optimization_name: str,
        test_name: str,
        warmup: bool = True
    ) -> BenchmarkResult:
        """Run a single benchmark test"""
        
        try:
            # Warmup run
            if warmup:
                _ = service.inference(request)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Get initial GPU state
            gpu_metrics_start = self.gpu_monitor.get_metrics()
            
            # Run inference
            start_time = time.perf_counter()
            result = service.inference(request)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            
            # Get final GPU state
            gpu_metrics_end = self.gpu_monitor.get_metrics()
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                test_name=test_name,
                resolution=(request.width, request.height),
                steps=request.steps,
                batch_size=request.num_images,
                optimization_level=optimization_name,
                inference_time=result['inference_time'],
                total_time=total_time,
                memory_used=gpu_metrics_end['memory_used'],
                memory_allocated=gpu_metrics_end['memory_allocated'],
                gpu_utilization=gpu_metrics_end['utilization'],
                temperature=gpu_metrics_end['temperature'],
                power_draw=gpu_metrics_end['power_draw'],
                prompt=request.prompt,
                seed=request.seed,
                timestamp=datetime.now()
            )
            
            return benchmark_result
            
        except Exception as e:
            print(f"ERROR in benchmark: {e}")
            # Return a failed benchmark result
            return BenchmarkResult(
                test_name=test_name,
                resolution=(request.width, request.height),
                steps=request.steps,
                batch_size=request.num_images,
                optimization_level=optimization_name,
                inference_time=-1,  # Indicate failure
                total_time=-1,
                memory_used=0,
                memory_allocated=0,
                gpu_utilization=0,
                temperature=0,
                power_draw=0,
                prompt=request.prompt,
                seed=request.seed,
                timestamp=datetime.now()
            )
    
    def run_resolution_benchmarks(self):
        """Test different resolutions"""
        print("\n=== Resolution Benchmarks ===")
        
        resolutions = [
            (512, 512),
            (768, 768),
            (1024, 1024),
            (1280, 720),  # 16:9
            (1024, 768),  # 4:3
        ]
        
        # Validate resolutions
        MAX_RESOLUTION = 2048
        for width, height in resolutions:
            if width > MAX_RESOLUTION or height > MAX_RESOLUTION:
                print(f"WARNING: Skipping {width}x{height} - exceeds max resolution {MAX_RESOLUTION}")
                resolutions.remove((width, height))
        
        for config in self.optimization_configs:
            print(f"\nTesting {config.name}...")
            service = self.create_optimized_service(config)
            service.load_model()
            
            for width, height in resolutions:
                print(f"  Resolution: {width}x{height}")
                
                # Run multiple times for statistical significance
                for i in range(5):
                    request = InferenceRequest(
                        prompt=self.test_prompts[i % len(self.test_prompts)],
                        width=width,
                        height=height,
                        steps=20,
                        num_images=1,
                        seed=42 + i
                    )
                    
                    result = self.run_single_benchmark(
                        service, 
                        request, 
                        config.name,
                        f"resolution_{width}x{height}",
                        warmup=(i == 0)
                    )
                    self.results.append(result)
            
            # Clean up
            del service
            torch.cuda.empty_cache()
            gc.collect()
    
    def run_step_count_benchmarks(self):
        """Test different step counts"""
        print("\n=== Step Count Benchmarks ===")
        
        step_counts = [10, 20, 28, 50]
        
        for config in ["baseline", "full_optimization"]:  # Only test baseline vs full
            opt_config = next(c for c in self.optimization_configs if c.name == config)
            print(f"\nTesting {config}...")
            
            service = self.create_optimized_service(opt_config)
            service.load_model()
            
            for steps in step_counts:
                print(f"  Steps: {steps}")
                
                for i in range(5):
                    request = InferenceRequest(
                        prompt=self.test_prompts[i % len(self.test_prompts)],
                        width=1024,
                        height=1024,
                        steps=steps,
                        num_images=1,
                        seed=42 + i
                    )
                    
                    result = self.run_single_benchmark(
                        service,
                        request,
                        config,
                        f"steps_{steps}",
                        warmup=(i == 0)
                    )
                    self.results.append(result)
            
            del service
            torch.cuda.empty_cache()
            gc.collect()
    
    def run_batch_size_benchmarks(self):
        """Test different batch sizes"""
        print("\n=== Batch Size Benchmarks ===")
        
        batch_sizes = [1, 2, 4]
        
        for config in ["baseline", "full_optimization"]:
            opt_config = next(c for c in self.optimization_configs if c.name == config)
            print(f"\nTesting {config}...")
            
            service = self.create_optimized_service(opt_config)
            service.load_model()
            
            for batch_size in batch_sizes:
                print(f"  Batch size: {batch_size}")
                
                for i in range(5):
                    request = InferenceRequest(
                        prompt=self.test_prompts[0],  # Same prompt for consistency
                        width=1024,
                        height=1024,
                        steps=20,
                        num_images=batch_size,
                        seed=42
                    )
                    
                    result = self.run_single_benchmark(
                        service,
                        request,
                        config,
                        f"batch_{batch_size}",
                        warmup=(i == 0)
                    )
                    self.results.append(result)
            
            del service
            torch.cuda.empty_cache()
            gc.collect()
    
    def calculate_statistics(self, data: List[float]) -> Dict:
        """Calculate statistical metrics"""
        if not data:
            return {}
        
        # Remove outliers (top and bottom 10%)
        sorted_data = sorted(data)
        trim_count = max(1, len(data) // 10)
        if len(data) > 2 * trim_count:
            trimmed_data = sorted_data[trim_count:-trim_count]
        else:
            trimmed_data = sorted_data
        
        return {
            'mean': statistics.mean(trimmed_data),
            'median': statistics.median(trimmed_data),
            'std': statistics.stdev(trimmed_data) if len(trimmed_data) > 1 else 0,
            'min': min(trimmed_data),
            'max': max(trimmed_data),
            'samples': len(trimmed_data)
        }
    
    def save_results(self):
        """Save benchmark results to JSON and CSV"""
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw data
        df.to_csv(self.output_dir / "benchmark_raw_data.csv", index=False)
        
        # Calculate aggregated statistics
        agg_results = []
        
        # Group by test configuration
        for group_keys in [
            ['optimization_level', 'resolution', 'steps', 'batch_size'],
            ['optimization_level', 'test_name']
        ]:
            grouped = df.groupby(group_keys)
            
            for name, group in grouped:
                stats = self.calculate_statistics(group['inference_time'].tolist())
                if stats:
                    result = dict(zip(group_keys, name))
                    result.update({
                        'inference_time_mean': stats['mean'],
                        'inference_time_std': stats['std'],
                        'inference_time_median': stats['median'],
                        'memory_used_mean': group['memory_used'].mean(),
                        'samples': len(group)
                    })
                    agg_results.append(result)
        
        # Save aggregated results
        agg_df = pd.DataFrame(agg_results)
        agg_df.to_csv(self.output_dir / "benchmark_summary.csv", index=False)
        
        # Save as JSON for easy loading
        with open(self.output_dir / "benchmark_results.json", 'w') as f:
            json.dump({
                'raw_results': [asdict(r) for r in self.results],
                'summary': agg_results,
                'test_config': {
                    'prompts': self.test_prompts,
                    'optimization_levels': [asdict(c) for c in self.optimization_configs]
                }
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite"""
        print("Starting comprehensive FLUX.1-Kontext benchmark suite...")
        print(f"Results will be saved to: {self.output_dir}")
        
        # Warm up GPU
        if torch.cuda.is_available():
            self.warmup_gpu()
        
        # Run different benchmark categories
        self.run_resolution_benchmarks()
        self.run_step_count_benchmarks()
        self.run_batch_size_benchmarks()
        
        # Save results
        self.save_results()
        
        print("\nBenchmarking complete!")
        print(f"Total tests run: {len(self.results)}")

def main():
    # Check for GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. Benchmarking requires GPU.")
        sys.exit(1)
    
    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Please set: export HF_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Run benchmarks
    suite = BenchmarkSuite()
    suite.run_full_benchmark()

if __name__ == "__main__":
    main()