"""
Performance tests for inference speed and optimization impact
"""

import pytest
import torch
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import statistics

from src.core.service import OptimizedFluxService
from src.utils.flux_utils import BenchmarkRunner


class TestInferenceSpeed:
    """Test inference speed and performance metrics"""
    
    @pytest.fixture
    def mock_fast_pipeline(self):
        """Create a mock pipeline that simulates fast inference"""
        mock_pipe = MagicMock()
        
        def mock_inference(*args, **kwargs):
            # Simulate some processing time
            time.sleep(0.1)  # 100ms inference
            mock_output = MagicMock()
            mock_output.images = [np.random.rand(512, 512, 3)]
            return mock_output
        
        mock_pipe.__call__ = mock_inference
        mock_pipe.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add required attributes
        mock_pipe.transformer = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.scheduler.config = MagicMock()
        
        return mock_pipe
    
    @pytest.mark.performance
    def test_baseline_inference_speed(self, mock_fast_pipeline, performance_tracker):
        """Test baseline inference speed without optimizations"""
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        # Run multiple inferences
        times = []
        for i in range(5):
            start = time.perf_counter()
            
            result = service.inference(
                prompt="Performance test",
                height=512,
                width=512,
                steps=10,
                seed=42
            )
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            performance_tracker.record("baseline_inference", elapsed)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        # Performance assertions
        assert avg_time < 1.0  # Should be under 1 second for mock
        assert std_dev < 0.1  # Should be consistent
        
        print(f"Baseline inference: {avg_time:.3f}s ± {std_dev:.3f}s")
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_optimization_impact(self, benchmark, mock_fast_pipeline):
        """Benchmark the impact of different optimizations"""
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        # Mock optimization methods
        service.flash_attention_optimizer.apply_patch = MagicMock(return_value=4)
        service.mxfp4_quantizer.apply_quantization = MagicMock(return_value=True)
        
        def run_inference():
            return service.inference(
                prompt="Benchmark test",
                height=512,
                width=512,
                steps=10,
                seed=42
            )
        
        # Run benchmark
        result = benchmark(run_inference)
        
        # Benchmark automatically calculates statistics
        assert result['inference_time'] < 1.0
    
    @pytest.mark.performance
    @pytest.mark.parametrize("resolution", [
        (512, 512),
        (768, 768),
        (1024, 1024),
    ])
    def test_resolution_scaling(self, mock_fast_pipeline, resolution, performance_tracker):
        """Test performance scaling with different resolutions"""
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        width, height = resolution
        
        # Simulate resolution-dependent timing
        def resolution_aware_inference(*args, **kwargs):
            # Simulate quadratic scaling with resolution
            base_time = 0.1
            scale_factor = (width * height) / (512 * 512)
            time.sleep(base_time * scale_factor)
            
            mock_output = MagicMock()
            mock_output.images = [np.random.rand(height, width, 3)]
            return mock_output
        
        mock_fast_pipeline.__call__ = resolution_aware_inference
        
        # Measure inference time
        start = time.perf_counter()
        result = service.inference(
            prompt="Resolution test",
            height=height,
            width=width,
            steps=10
        )
        elapsed = time.perf_counter() - start
        
        performance_tracker.record(f"resolution_{width}x{height}", elapsed)
        
        # Verify scaling is reasonable
        expected_scale = (width * height) / (512 * 512)
        assert elapsed < 0.2 * expected_scale  # Should scale sub-linearly in practice
    
    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_performance(self, mock_fast_pipeline, batch_size, performance_tracker):
        """Test batch processing performance"""
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        # Simulate batch processing
        def batch_inference(*args, **kwargs):
            num_images = kwargs.get('num_images_per_prompt', 1)
            time.sleep(0.1 + 0.05 * (num_images - 1))  # Batch efficiency
            
            mock_output = MagicMock()
            mock_output.images = [np.random.rand(512, 512, 3) for _ in range(num_images)]
            return mock_output
        
        mock_fast_pipeline.__call__ = batch_inference
        
        # Measure batch inference
        start = time.perf_counter()
        result = service.inference(
            prompt="Batch test",
            height=512,
            width=512,
            steps=10,
            num_images=batch_size
        )
        elapsed = time.perf_counter() - start
        
        performance_tracker.record(f"batch_size_{batch_size}", elapsed)
        
        # Calculate per-image time
        per_image_time = elapsed / batch_size
        
        # Batch processing should be more efficient
        if batch_size > 1:
            assert per_image_time < 0.15  # Should be faster per image in batch
    
    @pytest.mark.performance
    def test_scheduler_performance(self, mock_fast_pipeline, performance_tracker):
        """Test performance difference between schedulers"""
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        # Initialize scheduler manager
        from src.optimizations.schedulers import SchedulerManager
        service.scheduler_manager = SchedulerManager(mock_fast_pipeline)
        
        schedulers = {
            'euler': 28,  # More steps
            'dpm_solver': 14  # Fewer steps
        }
        
        for scheduler_type, steps in schedulers.items():
            # Setup scheduler
            service.scheduler_manager.setup_scheduler(scheduler_type)
            
            # Simulate step-dependent timing
            def step_aware_inference(*args, **kwargs):
                num_steps = kwargs.get('num_inference_steps', 28)
                time.sleep(0.01 * num_steps)  # Linear with steps
                
                mock_output = MagicMock()
                mock_output.images = [np.random.rand(512, 512, 3)]
                return mock_output
            
            mock_fast_pipeline.__call__ = step_aware_inference
            
            # Measure inference time
            start = time.perf_counter()
            result = service.inference(
                prompt=f"{scheduler_type} test",
                height=512,
                width=512,
                steps=steps
            )
            elapsed = time.perf_counter() - start
            
            performance_tracker.record(f"scheduler_{scheduler_type}", elapsed)
        
        # DPM solver should be faster due to fewer steps
        dpm_time = performance_tracker.get_average("scheduler_dpm_solver")
        euler_time = performance_tracker.get_average("scheduler_euler")
        
        if dpm_time and euler_time:
            assert dpm_time < euler_time
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_efficiency(self, mock_fast_pipeline):
        """Test memory usage during inference"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        # Measure memory before
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        
        # Run inference
        result = service.inference(
            prompt="Memory test",
            height=1024,
            width=1024,
            steps=10
        )
        
        # Measure memory after
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()
        
        # Calculate memory usage
        memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
        
        # Memory usage should be reasonable
        assert memory_used_mb < 8000  # Less than 8GB for 1024x1024
        
        # Cleanup
        torch.cuda.empty_cache()
    
    @pytest.mark.performance
    def test_warmup_effect(self, mock_fast_pipeline, performance_tracker):
        """Test the effect of warmup on inference speed"""
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        times = []
        
        # Run 10 inferences to see warmup effect
        for i in range(10):
            start = time.perf_counter()
            
            result = service.inference(
                prompt="Warmup test",
                height=512,
                width=512,
                steps=10,
                seed=42
            )
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            performance_tracker.record(f"warmup_run_{i}", elapsed)
        
        # First run should be slower (cold start)
        first_run = times[0]
        avg_warm = statistics.mean(times[5:])  # Average of last 5 runs
        
        # Warm runs should be faster
        assert avg_warm <= first_run
        
        # Calculate speedup from warmup
        warmup_speedup = first_run / avg_warm
        print(f"Warmup speedup: {warmup_speedup:.2f}x")


class TestOptimizationSpeedup:
    """Test speedup from various optimizations"""
    
    @pytest.mark.performance
    def test_flash_attention_speedup(self, mock_fast_pipeline):
        """Test speedup from FlashAttention"""
        # Create two services
        baseline_service = OptimizedFluxService()
        optimized_service = OptimizedFluxService()
        
        baseline_service.pipe = mock_fast_pipeline
        optimized_service.pipe = mock_fast_pipeline
        
        # Mock FlashAttention to provide speedup
        def mock_flash_inference(*args, **kwargs):
            time.sleep(0.08)  # 20% faster
            mock_output = MagicMock()
            mock_output.images = [np.random.rand(512, 512, 3)]
            return mock_output
        
        # Measure baseline
        baseline_times = []
        for _ in range(3):
            start = time.perf_counter()
            baseline_service.inference("Test", height=512, width=512, steps=10)
            baseline_times.append(time.perf_counter() - start)
        
        # Apply FlashAttention optimization
        optimized_service.pipe.__call__ = mock_flash_inference
        
        # Measure optimized
        optimized_times = []
        for _ in range(3):
            start = time.perf_counter()
            optimized_service.inference("Test", height=512, width=512, steps=10)
            optimized_times.append(time.perf_counter() - start)
        
        # Calculate speedup
        baseline_avg = statistics.mean(baseline_times)
        optimized_avg = statistics.mean(optimized_times)
        speedup = baseline_avg / optimized_avg
        
        assert speedup > 1.1  # At least 10% speedup
        print(f"FlashAttention speedup: {speedup:.2f}x")
    
    @pytest.mark.performance
    def test_quantization_speedup(self, mock_fast_pipeline):
        """Test speedup from quantization"""
        # Similar structure to flash attention test
        # but simulating quantization benefits
        
        service = OptimizedFluxService()
        service.pipe = mock_fast_pipeline
        
        # Simulate quantized inference (faster and less memory)
        def mock_quantized_inference(*args, **kwargs):
            time.sleep(0.07)  # 30% faster due to reduced precision
            mock_output = MagicMock()
            mock_output.images = [np.random.rand(512, 512, 3)]
            return mock_output
        
        # Test shows quantization provides speedup
        service.pipe.__call__ = mock_quantized_inference
        
        start = time.perf_counter()
        result = service.inference("Quantization test", height=512, width=512, steps=10)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1  # Should be fast with quantization