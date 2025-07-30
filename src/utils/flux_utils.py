#!/usr/bin/env python3
"""
FLUX.1-Kontext Utilities - Profiling, Timing, and Optimization Helpers

This module contains utility classes and functions for:
- Profiling context management
- Detailed timing analysis
- Performance recommendations
- System resource monitoring
- Memory analysis
- Optimization helpers
"""

import os
import time
import json
import psutil
import gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager
import torch
import torch.profiler
import numpy as np
from PIL import Image

# Check if profiling should be disabled
DISABLE_PROFILING = os.environ.get("DISABLE_PROFILING", "false").lower() == "true"


class ProfilingContext:
    """Context manager for detailed profiling"""
    
    def __init__(self, name: str, enable_torch_profiler: bool = True, 
                 enable_memory_profiling: bool = True):
        self.name = name
        self.enable_torch_profiler = enable_torch_profiler and not DISABLE_PROFILING
        self.enable_memory_profiling = enable_memory_profiling and not DISABLE_PROFILING
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.profiler = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        
        if self.enable_memory_profiling and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
            
        if self.enable_torch_profiler:
            try:
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    with_modules=True
                )
                self.profiler.__enter__()
            except Exception as e:
                print(f"Warning: Failed to start profiler for {self.name}: {e}")
                self.profiler = None
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        self.end_time = time.perf_counter()
        
        if self.enable_memory_profiling and torch.cuda.is_available():
            self.end_memory = torch.cuda.memory_allocated()
            
        if self.profiler:
            try:
                self.profiler.__exit__(exc_type, exc_val, exc_tb)
            except RuntimeError as e:
                if "Can't disable Kineto profiler when it's not running" in str(e):
                    print(f"Warning: Profiler was already stopped for {self.name}")
                else:
                    raise e
            
    def get_stats(self) -> Dict:
        """Get profiling statistics"""
        stats = {
            'name': self.name,
            'duration': self.end_time - self.start_time if self.end_time else None,
        }
        
        if self.start_memory is not None and self.end_memory is not None:
            stats['memory_delta'] = self.end_memory - self.start_memory
            stats['memory_delta_mb'] = (self.end_memory - self.start_memory) / 1024 / 1024
            
        return stats
        
    def save_trace(self, output_dir: Path):
        """Save profiler trace for analysis"""
        if self.profiler:
            try:
                trace_path = output_dir / f"{self.name}_trace.json"
                self.profiler.export_chrome_trace(str(trace_path))
                print(f"Profiler trace saved to: {trace_path}")
            except Exception as e:
                print(f"Warning: Failed to save profiler trace for {self.name}: {e}")
                return
            
            # Also save summary
            try:
                summary_path = output_dir / f"{self.name}_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(f"=== {self.name} Profiling Summary ===\n\n")
                    
                    # Top CPU operations
                    f.write("Top CPU Operations:\n")
                    f.write(self.profiler.key_averages().table(
                        sort_by="cpu_time_total", row_limit=20
                    ))
                    f.write("\n\n")
                    
                    # Top GPU operations (if available)
                    if torch.cuda.is_available():
                        f.write("Top GPU Operations:\n")
                        f.write(self.profiler.key_averages().table(
                            sort_by="cuda_time_total", row_limit=20
                        ))
                        f.write("\n\n")
                    
                    # Memory usage
                    f.write("Memory Usage:\n")
                    f.write(self.profiler.key_averages().table(
                        sort_by="self_cuda_memory_usage", row_limit=20
                    ))
                    
                print(f"Profiler summary saved to: {summary_path}")
            except Exception as e:
                print(f"Warning: Failed to save profiler summary for {self.name}: {e}")


class DetailedTimer:
    """Detailed timing for specific operations"""
    
    def __init__(self):
        self.timings = {}
        self.current_context = []
        
    @contextmanager
    def time(self, name: str):
        """Time a specific operation"""
        full_name = ".".join(self.current_context + [name])
        self.current_context.append(name)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            self.timings[full_name] = end_time - start_time
            self.current_context.pop()
            
    def get_report(self) -> str:
        """Get timing report"""
        if not self.timings:
            return "No timings recorded"
            
        # Sort by time descending
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        report = "=== Detailed Timing Report ===\n"
        total_time = sum(self.timings.values())
        
        for name, duration in sorted_timings:
            percentage = (duration / total_time) * 100
            report += f"{name:40} | {duration:8.4f}s | {percentage:6.2f}%\n"
            
        report += f"\nTotal measured time: {total_time:.4f}s\n"
        return report
        
    def clear(self):
        """Clear all timings"""
        self.timings.clear()


class PerformanceAnalyzer:
    """Analyze performance and generate recommendations"""
    
    @staticmethod
    def generate_performance_recommendations(profiler_stats: Dict, timings: Dict) -> List[str]:
        """Generate performance optimization recommendations based on profiling data"""
        recommendations = []
        
        total_time = timings.get('total_inference', 0)
        
        # Analyze timing bottlenecks
        if total_time > 1.0:
            recommendations.append(f"Target sub-1s inference: Current time is {total_time:.2f}s")
            
            # Check major components
            denoising_time = timings.get('denoising_loop', 0)
            if denoising_time > 0.7:
                recommendations.append(f"Denoising loop is {denoising_time:.2f}s - consider reducing steps or using faster scheduler")
            
            text_encoding_time = timings.get('text_encoding', 0)
            if text_encoding_time > 0.1:
                recommendations.append(f"Text encoding is {text_encoding_time:.2f}s - consider caching embeddings")
            
            vae_time = timings.get('vae_decoding', 0)
            if vae_time > 0.2:
                recommendations.append(f"VAE decoding is {vae_time:.2f}s - consider VAE optimization or lower precision")
        
        # Memory recommendations
        if torch.cuda.is_available():
            max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            if max_memory_gb > 20:
                recommendations.append(f"High memory usage ({max_memory_gb:.1f}GB) - consider model quantization")
            elif max_memory_gb < 8:
                recommendations.append("Low memory usage - could potentially use higher precision or larger batch sizes")
        
        # Specific optimization recommendations
        recommendations.extend([
            "Consider using TensorRT for further optimization",
            "Try reducing resolution to 512x512 for faster inference",
            "Use fewer denoising steps (8-10) with DPM-Solver++",
            "Enable mixed precision training if not already enabled",
            "Consider model distillation for smaller, faster models",
            "Use CUDA graphs for static input shapes",
            "Profile individual transformer blocks to identify specific bottlenecks",
            "Consider using fp16 instead of bfloat16 if supported",
            "Enable attention slicing for memory efficiency",
            "Use CPU offloading for components not in critical path"
        ])
        
        return recommendations
    
    @staticmethod
    def analyze_memory_usage(output_dir: Path) -> Dict:
        """Analyze GPU memory usage and save report"""
        if not torch.cuda.is_available():
            return {}
            
        memory_stats = torch.cuda.memory_stats()
        memory_summary = torch.cuda.memory_summary()
        
        memory_report_path = output_dir / "memory_analysis.txt"
        with open(memory_report_path, 'w') as f:
            f.write("=== GPU Memory Analysis ===\n\n")
            f.write(f"Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
            f.write(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n")
            f.write(f"Current reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")
            f.write(f"Max reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB\n\n")
            f.write("Detailed Memory Summary:\n")
            f.write(memory_summary)
            f.write("\n\nDetailed Memory Stats:\n")
            f.write(json.dumps(memory_stats, indent=2))
        
        print(f"Memory analysis saved to: {memory_report_path}")
        return memory_stats
    
    @staticmethod
    def analyze_system_resources(output_dir: Path) -> Dict:
        """Analyze system resource usage and save report"""
        system_stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / 1024**3,
        }
        
        system_report_path = output_dir / "system_analysis.txt"
        with open(system_report_path, 'w') as f:
            f.write("=== System Resource Analysis ===\n\n")
            f.write(f"CPU Usage: {system_stats['cpu_percent']:.1f}%\n")
            f.write(f"Memory Usage: {system_stats['memory_percent']:.1f}%\n")
            f.write(f"Available Memory: {system_stats['available_memory_gb']:.2f} GB\n")
        
        print(f"System analysis saved to: {system_report_path}")
        return system_stats


class ImageProcessor:
    """Handle image processing and saving operations"""
    
    @staticmethod
    def save_images(images: List[np.ndarray], output_dir: Path, prefix: str = "image") -> List[str]:
        """Save images to disk and return file paths"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for idx, image in enumerate(images):
            # Convert to uint8
            image_np = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_np)
            
            # Save to file
            filename = f"{prefix}_{idx+1}.png"
            filepath = output_dir / filename
            pil_image.save(filepath, format="PNG")
            saved_files.append(str(filepath))
            
            print(f"Image {idx+1} saved to: {filepath}")
        
        return saved_files
    
    @staticmethod
    def save_single_image(image: np.ndarray, output_dir: Path, filename: str) -> str:
        """Save a single image to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8
        image_np = (image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)
        
        # Save to file
        filepath = output_dir / filename
        pil_image.save(filepath, format="PNG")
        
        print(f"Image saved to: {filepath}")
        return str(filepath)


class CacheManager:
    """Manage torch compilation cache and mega-cache"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.mega_cache_bin_path = cache_dir / ".mega_cache" / "flux_torch_mega_optimized"
        self.mega_cache_bin_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_mega_cache(self):
        """Load torch mega-cache if available"""
        print("Loading torch mega-cache...")
        try:
            if self.mega_cache_bin_path.exists():
                with open(self.mega_cache_bin_path, "rb") as f:
                    artifact_bytes = f.read()
                
                if artifact_bytes:
                    torch.compiler.load_cache_artifacts(artifact_bytes)
                    print("Mega-cache loaded successfully")
            else:
                print("Mega-cache not found, will be generated")
        except Exception as e:
            print(f"Error loading mega-cache: {e}")
    
    def save_mega_cache(self):
        """Save torch mega-cache for faster subsequent loads"""
        print("Saving torch mega-cache...")
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            artifact_bytes, _ = artifacts
            
            with open(self.mega_cache_bin_path, "wb") as f:
                f.write(artifact_bytes)
            
            print("Mega-cache saved successfully")
        except Exception as e:
            print(f"Error saving mega-cache: {e}")


class EnvironmentSetup:
    """Setup environment variables and configurations for optimization"""
    
    @staticmethod
    def setup_optimization_env(cache_dir: Path):
        """Setup environment variables for optimization"""
        # Environment variables for optimization
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        os.environ["CUDA_CACHE_PATH"] = str(cache_dir / ".nv_cache")
        os.environ["HF_HUB_CACHE"] = str(cache_dir / ".hf_hub_cache")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / ".inductor_cache")
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir / ".triton_cache")
    
    @staticmethod
    def setup_h100_env():
        """Setup H100-specific environment variables"""
        if torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0):
            os.environ["CUDA_MODULE_LOADING"] = "LAZY"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"  # H100 compute capability
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.9"
            print("Detected H100 GPU - enabling H100-specific environment settings")
            return True
        return False
    
    @staticmethod
    def setup_torch_compile_config():
        """Setup torch.compile configuration"""
        config = torch._inductor.config
        config.conv_1x1_as_mm = True
        config.coordinate_descent_check_all_directions = True
        config.coordinate_descent_tuning = True
        config.disable_progress = False
        config.epilogue_fusion = False
        config.shape_padding = True
        
        # CRITICAL: Prevent torch.compile from changing data types
        config.force_fuse_int_mm_with_mul = False
        config.freezing = False  # Disable freezing to preserve data types
        
        # H100-specific settings
        if torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0):
            print("Applying H100-specific compile settings...")
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


class BenchmarkRunner:
    """Run performance benchmarks and comparisons"""
    
    @staticmethod
    def run_benchmark_comparison(service_class, test_prompt: str, test_height: int, 
                                test_width: int, test_seed: int, optimization_configs: List[Dict]) -> Dict:
        """Run benchmark comparison between different optimization configurations"""
        results = {}
        
        for config in optimization_configs:
            print(f"\n=== {config['name']} ===")
            print(f"Steps: {config['steps']}")
            
            # Initialize service
            service = service_class()
            service.load_model(enable_optimizations=config['config'])
            
            # Warm-up run
            print("Warm-up run...")
            _ = service.inference(
                prompt=test_prompt,
                height=test_height,
                width=test_width,
                steps=config['steps'],
                seed=test_seed
            )
            
            # Benchmark runs
            print("Benchmarking...")
            times = []
            for i in range(3):
                result = service.inference(
                    prompt=test_prompt,
                    height=test_height,
                    width=test_width,
                    steps=config['steps'],
                    seed=test_seed
                )
                times.append(result['inference_time'])
                print(f"Run {i+1}: {result['inference_time']:.2f}s")
            
            avg_time = sum(times) / len(times)
            results[config['name']] = avg_time
            print(f"Average inference time: {avg_time:.2f}s")
            
            # Clear memory
            del service
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    @staticmethod
    def print_benchmark_summary(results: Dict):
        """Print benchmark comparison summary"""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*70)
        
        baseline_time = results["Baseline (Minimal optimizations)"]
        
        for name, time in results.items():
            speedup = baseline_time / time
            sub_1s_status = "✓ SUB-1S!" if time < 1.0 else f"Need {time - 1.0:.2f}s reduction"
            print(f"{name:35} | {time:6.2f}s | {speedup:5.2f}x | {sub_1s_status}")
        
        print(f"\nBaseline time: {baseline_time:.2f}s")
        best_time = min(results.values())
        print(f"Best time: {best_time:.2f}s")
        print(f"Overall speedup: {baseline_time / best_time:.2f}x")
        
        if best_time < 1.0:
            print(f"🎉 ACHIEVED SUB-1S INFERENCE: {best_time:.3f}s")
        else:
            print(f"❌ Sub-1s target missed by: {best_time - 1.0:.3f}s")


# Export all utility classes and functions
__all__ = [
    'ProfilingContext',
    'DetailedTimer', 
    'PerformanceAnalyzer',
    'ImageProcessor',
    'CacheManager',
    'EnvironmentSetup',
    'BenchmarkRunner',
    'DISABLE_PROFILING'
] 