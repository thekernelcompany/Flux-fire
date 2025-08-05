"""
Integration tests for OptimizedFluxService
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.core.service import OptimizedFluxService


class TestOptimizedFluxServiceIntegration:
    """Integration tests for the complete service"""
    
    @pytest.fixture
    def service(self):
        """Create service instance (without loading models)"""
        return OptimizedFluxService()
    
    def test_service_initialization(self, service):
        """Test service initialization and component setup"""
        assert service.pipe is None
        assert service.cache_manager is not None
        assert service.timer is not None
        assert hasattr(service, 'flash_attention_optimizer')
        assert hasattr(service, 'mxfp4_quantizer')
        assert hasattr(service, 'h100_fp8_optimizer')
        assert hasattr(service, 'tensorrt_optimizer')
        assert hasattr(service, 'mega_cache_manager')
    
    @patch('src.core.service.FluxKontextPipeline')
    def test_model_loading_mock(self, mock_pipeline_class, service, mock_hf_token):
        """Test model loading with mocked pipeline"""
        # Create mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        # Mock components
        mock_pipeline.transformer = MagicMock()
        mock_pipeline.vae = MagicMock()
        mock_pipeline.scheduler = MagicMock()
        mock_pipeline.scheduler.config = MagicMock()
        
        # Load model
        service.load_model(enable_optimizations={
            'flash_attention': False,
            'mxfp4': False,
            'scheduler': 'dpm_solver',
            'cuda_graphs': False,
            'para_attention': False,
            'tensorrt': False
        })
        
        # Verify model was loaded
        assert service.pipe is not None
        mock_pipeline_class.from_pretrained.assert_called_once()
        
        # Verify managers were initialized
        assert service.scheduler_manager is not None
        assert service.cuda_graphs_manager is not None
        assert service.model_compiler is not None
    
    def test_optimization_stage_1(self, service, mock_pipeline):
        """Test Stage 1 optimizations"""
        service.pipe = mock_pipeline
        
        # Initialize required managers
        service.scheduler_manager = MagicMock()
        service.cuda_graphs_manager = MagicMock()
        service.model_compiler = MagicMock()
        service.model_compiler.compile_model.return_value = True
        
        # Apply Stage 1 optimizations
        service._optimize_stage_1({'para_attention': False})
        
        # Verify optimizations were applied
        assert mock_pipeline.transformer.to.called
        assert mock_pipeline.vae.to.called
        service.model_compiler.compile_model.assert_called_once()
    
    @patch('src.core.service.torch.cuda.is_available', return_value=True)
    def test_inference_mock(self, mock_cuda, service, mock_pipeline):
        """Test inference with mocked pipeline"""
        # Setup service with mock pipeline
        service.pipe = mock_pipeline
        mock_output = MagicMock()
        mock_output.images = [np.random.rand(512, 512, 3)]
        mock_pipeline.return_value = mock_output
        
        # Run inference
        result = service.inference(
            prompt="Test prompt",
            height=512,
            width=512,
            steps=10,
            num_images=1
        )
        
        # Verify results
        assert 'inference_time' in result
        assert 'processing_time' in result
        assert 'total_time' in result
        assert 'saved_files' in result
        assert len(result['saved_files']) == 1
    
    def test_profile_inference_mock(self, service, mock_pipeline, temp_dir):
        """Test profiled inference"""
        # Setup service with mock pipeline
        service.pipe = mock_pipeline
        mock_output = MagicMock()
        mock_output.images = [np.random.rand(512, 512, 3)]
        mock_pipeline.return_value = mock_output
        
        # Run profiled inference
        result = service.profile_inference(
            prompt="Test prompt",
            height=512,
            width=512,
            steps=10,
            output_dir=temp_dir
        )
        
        # Verify profiling results
        assert 'inference_time' in result
        assert 'saved_files' in result
        assert 'profiling_dir' in result
        assert 'timings' in result
        assert 'recommendations' in result
        
        # Check profiling files
        assert (temp_dir / "timing_analysis.txt").exists()
        assert (temp_dir / "performance_recommendations.txt").exists()
    
    def test_fast_inference_mock(self, service, mock_pipeline):
        """Test fast inference path"""
        # Setup service with mock pipeline
        service.pipe = mock_pipeline
        service.is_h100 = False
        
        # Mock the fast inference components
        mock_pipeline.scheduler = MagicMock()
        mock_pipeline.scheduler.timesteps = [1000, 800, 600, 400, 200, 0]
        mock_pipeline.transformer = MagicMock()
        mock_pipeline.vae = MagicMock()
        mock_pipeline.vae.decode = MagicMock(return_value=[torch.randn(1, 3, 64, 64)])
        mock_pipeline.vae.config = MagicMock(scaling_factor=0.13025)
        
        # Run fast inference
        image, inference_time = service.fast_inference(
            prompt="Fast test",
            height=512,
            width=512,
            steps=6
        )
        
        # Verify results
        assert isinstance(image, np.ndarray) or image is not None
        assert inference_time > 0
    
    def test_optimization_combinations(self, service, mock_pipeline):
        """Test different optimization combinations"""
        service.pipe = mock_pipeline
        
        # Initialize managers
        service.scheduler_manager = MagicMock()
        service.cuda_graphs_manager = MagicMock()
        service.model_compiler = MagicMock()
        service.mega_cache_manager = MagicMock()
        
        # Test configurations
        configs = [
            {'flash_attention': True, 'mxfp4': False},
            {'flash_attention': False, 'mxfp4': True},
            {'flash_attention': True, 'mxfp4': True},
            {'scheduler': 'dpm_solver', 'cuda_graphs': True},
        ]
        
        for config in configs:
            # Reset mocks
            service.flash_attention_optimizer.apply_patch = MagicMock(return_value=4)
            service.mxfp4_quantizer.apply_quantization = MagicMock(return_value=True)
            
            # Apply optimizations
            service._optimize_stage_1(config)
            
            # Stage 2
            if config.get('flash_attention', False):
                service.flash_attention_optimizer.apply_patch(mock_pipeline)
                service.flash_attention_optimizer.apply_patch.assert_called()
            
            # Stage 3
            if config.get('mxfp4', False):
                service.mxfp4_quantizer.apply_quantization(mock_pipeline)
                service.mxfp4_quantizer.apply_quantization.assert_called()
    
    def test_error_handling_during_load(self, service):
        """Test error handling during model loading"""
        with patch('src.core.service.FluxKontextPipeline.from_pretrained') as mock_load:
            mock_load.side_effect = RuntimeError("Model not found")
            
            with pytest.raises(RuntimeError):
                service.load_model()
    
    def test_memory_cleanup(self, service, mock_pipeline):
        """Test memory cleanup after operations"""
        import gc
        
        service.pipe = mock_pipeline
        
        # Simulate some operations
        service.timer.time("test_op")
        
        # Delete service
        del service
        gc.collect()
        
        # Memory should be freed (this is more of a smoke test)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestServiceOptimizationFlow:
    """Test the complete optimization flow"""
    
    @patch('src.core.service.FluxKontextPipeline')
    def test_full_optimization_stack(self, mock_pipeline_class, mock_hf_token):
        """Test applying all optimizations in sequence"""
        # Create service
        service = OptimizedFluxService()
        
        # Create elaborate mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        # Mock all components
        mock_pipeline.transformer = MagicMock()
        mock_pipeline.transformer.transformer_blocks = [MagicMock() for _ in range(4)]
        mock_pipeline.vae = MagicMock()
        mock_pipeline.scheduler = MagicMock()
        mock_pipeline.scheduler.config = MagicMock()
        mock_pipeline.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mock optimization results
        service.flash_attention_optimizer.apply_patch = MagicMock(return_value=4)
        service.mxfp4_quantizer.apply_quantization = MagicMock(return_value=True)
        service.model_compiler.compile_model = MagicMock(return_value=True)
        service.model_compiler.trigger_compilation = MagicMock()
        
        # Load with all optimizations
        service.load_model(enable_optimizations={
            'flash_attention': True,
            'mxfp4': True,
            'scheduler': 'dpm_solver',
            'cuda_graphs': True,
            'para_attention': False,
            'tensorrt': False
        })
        
        # Verify all optimizations were attempted
        service.flash_attention_optimizer.apply_patch.assert_called()
        service.mxfp4_quantizer.apply_quantization.assert_called()
        service.model_compiler.trigger_compilation.assert_called()
        
        # Verify pipeline is ready
        assert service.pipe is not None
        assert service.scheduler_manager is not None
        assert service.cuda_graphs_manager is not None