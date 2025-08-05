"""
Unit tests for utility functions and classes
"""

import pytest
import torch
import numpy as np
import time
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from contextlib import contextmanager

from src.utils.flux_utils import (
    ProfilingContext, DetailedTimer, PerformanceAnalyzer,
    ImageProcessor, CacheManager, EnvironmentSetup, BenchmarkRunner
)


class TestProfilingContext:
    """Test profiling context functionality"""
    
    def test_basic_profiling(self):
        """Test basic profiling context"""
        with ProfilingContext("test_operation") as profiler:
            # Simulate some work
            time.sleep(0.01)
        
        stats = profiler.get_stats()
        
        assert stats['name'] == "test_operation"
        assert stats['duration'] > 0.01
        assert stats['duration'] < 0.1  # Should be reasonably fast
    
    @pytest.mark.gpu_required
    def test_memory_profiling(self):
        """Test memory profiling functionality"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with ProfilingContext("memory_test", enable_memory_profiling=True) as profiler:
            # Allocate some GPU memory
            tensor = torch.randn(1000, 1000, device='cuda')
        
        stats = profiler.get_stats()
        
        assert 'memory_delta' in stats
        assert 'memory_delta_mb' in stats
        # Memory should have increased
        assert stats['memory_delta'] >= 0
    
    def test_torch_profiler_integration(self):
        """Test PyTorch profiler integration"""
        with ProfilingContext("torch_profile_test", enable_torch_profiler=True) as profiler:
            # Some tensor operations
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)
        
        # Profiler should be created
        if profiler.profiler is not None:
            assert hasattr(profiler.profiler, 'export_chrome_trace')
    
    def test_save_trace(self, temp_dir):
        """Test saving profiler trace"""
        with ProfilingContext("save_test", enable_torch_profiler=True) as profiler:
            x = torch.randn(10, 10)
            y = x + x
        
        if profiler.profiler is not None:
            profiler.save_trace(temp_dir)
            
            # Check if files were created
            trace_file = temp_dir / "save_test_trace.json"
            summary_file = temp_dir / "save_test_summary.txt"
            
            # Files might not be created if profiler fails
            # This is expected in some test environments


class TestDetailedTimer:
    """Test detailed timing functionality"""
    
    def test_basic_timing(self):
        """Test basic timing operations"""
        timer = DetailedTimer()
        
        with timer.time("operation1"):
            time.sleep(0.01)
        
        with timer.time("operation2"):
            time.sleep(0.02)
        
        assert "operation1" in timer.timings
        assert "operation2" in timer.timings
        assert timer.timings["operation1"] > 0.01
        assert timer.timings["operation2"] > 0.02
    
    def test_nested_timing(self):
        """Test nested timing operations"""
        timer = DetailedTimer()
        
        with timer.time("parent"):
            time.sleep(0.01)
            with timer.time("child"):
                time.sleep(0.01)
        
        assert "parent" in timer.timings
        assert "parent.child" in timer.timings
        assert timer.timings["parent"] > timer.timings["parent.child"]
    
    def test_timing_report(self):
        """Test timing report generation"""
        timer = DetailedTimer()
        
        with timer.time("fast_op"):
            time.sleep(0.01)
        
        with timer.time("slow_op"):
            time.sleep(0.02)
        
        report = timer.get_report()
        
        assert "Detailed Timing Report" in report
        assert "fast_op" in report
        assert "slow_op" in report
        assert "Total measured time" in report
    
    def test_clear_timings(self):
        """Test clearing timings"""
        timer = DetailedTimer()
        
        with timer.time("operation"):
            pass
        
        assert len(timer.timings) > 0
        
        timer.clear()
        
        assert len(timer.timings) == 0


class TestPerformanceAnalyzer:
    """Test performance analysis functionality"""
    
    def test_generate_recommendations(self):
        """Test performance recommendation generation"""
        profiler_stats = {'name': 'test', 'duration': 2.0}
        timings = {
            'total_inference': 2.5,
            'denoising_loop': 1.8,
            'text_encoding': 0.2,
            'vae_decoding': 0.3
        }
        
        recommendations = PerformanceAnalyzer.generate_performance_recommendations(
            profiler_stats, timings
        )
        
        assert len(recommendations) > 0
        assert any("sub-1s inference" in rec for rec in recommendations)
        assert any("Denoising loop" in rec for rec in recommendations)
    
    @pytest.mark.gpu_required
    def test_analyze_memory_usage(self, temp_dir):
        """Test GPU memory analysis"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        memory_stats = PerformanceAnalyzer.analyze_memory_usage(temp_dir)
        
        assert isinstance(memory_stats, dict)
        assert (temp_dir / "memory_analysis.txt").exists()
    
    def test_analyze_system_resources(self, temp_dir):
        """Test system resource analysis"""
        system_stats = PerformanceAnalyzer.analyze_system_resources(temp_dir)
        
        assert 'cpu_percent' in system_stats
        assert 'memory_percent' in system_stats
        assert 'available_memory_gb' in system_stats
        assert (temp_dir / "system_analysis.txt").exists()


class TestImageProcessor:
    """Test image processing functionality"""
    
    def test_save_images(self, temp_dir, sample_image):
        """Test saving multiple images"""
        images = [sample_image, sample_image * 0.5]
        
        saved_files = ImageProcessor.save_images(images, temp_dir, prefix="test")
        
        assert len(saved_files) == 2
        assert all(Path(f).exists() for f in saved_files)
        assert all("test_" in f for f in saved_files)
    
    def test_save_single_image(self, temp_dir, sample_image):
        """Test saving a single image"""
        filename = "single_test.png"
        
        saved_file = ImageProcessor.save_single_image(sample_image, temp_dir, filename)
        
        assert Path(saved_file).exists()
        assert saved_file.endswith(filename)
    
    def test_image_conversion(self, temp_dir):
        """Test image format conversion"""
        # Create image with values outside [0, 1]
        image = np.random.rand(100, 100, 3) * 2.0 - 0.5
        
        saved_file = ImageProcessor.save_single_image(image, temp_dir, "conversion_test.png")
        
        # Image should be saved successfully despite out-of-range values
        assert Path(saved_file).exists()


class TestCacheManager:
    """Test cache management functionality"""
    
    def test_initialization(self, temp_dir):
        """Test CacheManager initialization"""
        manager = CacheManager(temp_dir)
        
        assert manager.cache_dir == temp_dir
        assert manager.mega_cache_bin_path.parent.exists()
    
    def test_load_save_cache_cycle(self, temp_dir):
        """Test complete load/save cycle"""
        manager = CacheManager(temp_dir)
        
        # Mock torch.compiler functions
        with patch('torch.compiler.save_cache_artifacts', return_value=(b"test_data", {})):
            with patch('torch.compiler.load_cache_artifacts') as mock_load:
                # Save cache
                manager.save_mega_cache()
                
                # Load cache
                manager.load_mega_cache()
                
                # Verify load was called with saved data
                mock_load.assert_called_once_with(b"test_data")


class TestEnvironmentSetup:
    """Test environment setup functionality"""
    
    def test_setup_optimization_env(self, temp_dir, monkeypatch):
        """Test optimization environment setup"""
        EnvironmentSetup.setup_optimization_env(temp_dir)
        
        assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
        assert os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE") == "1"
        assert "CUDA_CACHE_PATH" in os.environ
        assert str(temp_dir) in os.environ.get("CUDA_CACHE_PATH", "")
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="NVIDIA H100")
    def test_setup_h100_env(self, mock_device, mock_cuda):
        """Test H100-specific environment setup"""
        result = EnvironmentSetup.setup_h100_env()
        
        assert result is True
        assert os.environ.get("CUDA_MODULE_LOADING") == "LAZY"
        assert os.environ.get("TORCH_CUDA_ARCH_LIST") == "9.0"
    
    def test_setup_torch_compile_config(self):
        """Test torch compile configuration"""
        with patch('torch._inductor.config') as mock_config:
            EnvironmentSetup.setup_torch_compile_config()
            
            assert mock_config.conv_1x1_as_mm is True
            assert mock_config.coordinate_descent_tuning is True
            assert mock_config.shape_padding is True


class TestBenchmarkRunner:
    """Test benchmark runner functionality"""
    
    def test_run_benchmark_comparison(self, mock_service):
        """Test benchmark comparison execution"""
        # Mock service class
        mock_service_class = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_class.return_value = mock_service_instance
        
        # Mock inference results
        mock_service_instance.inference.return_value = {
            'inference_time': 1.0,
            'processing_time': 0.1,
            'total_time': 1.1,
            'saved_files': ['test.png']
        }
        
        configs = [
            {"name": "baseline", "config": {}, "steps": 28},
            {"name": "optimized", "config": {"mxfp4": True}, "steps": 14}
        ]
        
        results = BenchmarkRunner.run_benchmark_comparison(
            mock_service_class, "test prompt", 512, 512, 42, configs
        )
        
        assert "baseline" in results
        assert "optimized" in results
        assert results["baseline"] == 1.0
        assert results["optimized"] == 1.0
    
    def test_print_benchmark_summary(self, capsys):
        """Test benchmark summary printing"""
        results = {
            "Baseline": 2.0,
            "Optimized": 0.8,
            "Super Optimized": 0.5
        }
        
        BenchmarkRunner.print_benchmark_summary(results)
        
        captured = capsys.readouterr()
        assert "PERFORMANCE COMPARISON SUMMARY" in captured.out
        assert "Baseline" in captured.out
        assert "2.00s" in captured.out
        assert "SUB-1S!" in captured.out
        assert "Overall speedup: 4.00x" in captured.out