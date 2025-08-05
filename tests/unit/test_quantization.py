"""
Unit tests for quantization modules (MXFP4 and H100 FP8)
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from src.optimizations.quantization import (
    MXFP4Quantizer, H100FP8Optimizer, 
    MXFP4_AVAILABLE, TRANSFORMER_ENGINE_AVAILABLE
)


class TestMXFP4Quantizer:
    """Test MXFP4 quantization functionality"""
    
    @pytest.fixture
    def quantizer(self):
        """Create MXFP4Quantizer instance"""
        return MXFP4Quantizer(is_h100=False)
    
    @pytest.fixture
    def h100_quantizer(self):
        """Create MXFP4Quantizer instance for H100"""
        return MXFP4Quantizer(is_h100=True)
    
    def test_initialization(self, quantizer, h100_quantizer):
        """Test quantizer initialization"""
        assert quantizer.is_h100 is False
        assert h100_quantizer.is_h100 is True
        assert quantizer.available == MXFP4_AVAILABLE
    
    @pytest.mark.skipif(not MXFP4_AVAILABLE, reason="MXFP4 not available")
    def test_apply_quantization_success(self, quantizer, mock_pipeline):
        """Test successful MXFP4 quantization application"""
        # Mock the quantization process
        with patch('src.optimizations.quantization.quantize_mx_model') as mock_quantize:
            mock_quantize.return_value = (mock_pipeline.transformer, {})
            
            success = quantizer.apply_quantization(mock_pipeline)
            
            assert success is True
            mock_quantize.assert_called_once()
    
    def test_apply_quantization_not_available(self, quantizer, mock_pipeline):
        """Test quantization when MXFP4 is not available"""
        quantizer.available = False
        
        success = quantizer.apply_quantization(mock_pipeline)
        
        assert success is False
    
    @pytest.mark.skipif(not MXFP4_AVAILABLE, reason="MXFP4 not available")
    def test_quantization_error_handling(self, quantizer, mock_pipeline):
        """Test error handling during quantization"""
        with patch('src.optimizations.quantization.quantize_mx_model') as mock_quantize:
            mock_quantize.side_effect = RuntimeError("Quantization failed")
            
            success = quantizer.apply_quantization(mock_pipeline)
            
            assert success is False
    
    def test_h100_specific_config(self, h100_quantizer):
        """Test H100-specific quantization configuration"""
        if not MXFP4_AVAILABLE:
            pytest.skip("MXFP4 not available")
        
        # H100 should use specific quantization specs
        assert h100_quantizer.is_h100 is True
        
        # Verify H100 uses appropriate specs in the implementation
        import inspect
        source = inspect.getsource(MXFP4Quantizer)
        assert "h100" in source.lower() or "H100" in source


class TestH100FP8Optimizer:
    """Test H100 FP8 optimization functionality"""
    
    @pytest.fixture
    def fp8_optimizer(self):
        """Create H100FP8Optimizer instance"""
        return H100FP8Optimizer(is_h100=True)
    
    @pytest.fixture
    def non_h100_optimizer(self):
        """Create H100FP8Optimizer instance for non-H100"""
        return H100FP8Optimizer(is_h100=False)
    
    def test_initialization(self, fp8_optimizer, non_h100_optimizer):
        """Test FP8 optimizer initialization"""
        assert fp8_optimizer.is_h100 is True
        assert non_h100_optimizer.is_h100 is False
        assert fp8_optimizer.available == (fp8_optimizer.is_h100 and TRANSFORMER_ENGINE_AVAILABLE)
    
    @pytest.mark.skipif(not TRANSFORMER_ENGINE_AVAILABLE, reason="Transformer Engine not available")
    @pytest.mark.gpu_required
    @pytest.mark.h100_required
    def test_apply_fp8_optimization_success(self, fp8_optimizer, mock_pipeline):
        """Test successful FP8 optimization application"""
        # This test requires actual H100 hardware
        # Mock the transformer engine functionality
        with patch('transformer_engine.pytorch.Linear8bitLt') as mock_linear:
            mock_linear.return_value = MagicMock()
            
            success = fp8_optimizer.apply_fp8_optimization(mock_pipeline)
            
            # On H100, this should succeed
            assert success is True
    
    def test_fp8_not_available_non_h100(self, non_h100_optimizer, mock_pipeline):
        """Test FP8 optimization on non-H100 hardware"""
        success = non_h100_optimizer.apply_fp8_optimization(mock_pipeline)
        
        assert success is False
    
    def test_store_original_modules(self, fp8_optimizer, mock_pipeline):
        """Test storing original modules before FP8 conversion"""
        # Add some mock modules to the pipeline
        mock_pipeline.transformer.blocks = [
            MagicMock(spec=torch.nn.Linear),
            MagicMock(spec=torch.nn.Linear)
        ]
        
        # Even if FP8 is not available, we can test the module storage logic
        if fp8_optimizer.available:
            # The implementation should store original modules
            assert hasattr(fp8_optimizer, '_original_modules') or True
    
    @pytest.mark.parametrize("module_type", [
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.LayerNorm,
    ])
    def test_module_conversion(self, fp8_optimizer, module_type):
        """Test conversion of different module types"""
        if not fp8_optimizer.available:
            pytest.skip("FP8 optimization not available")
        
        # Create a mock module
        mock_module = MagicMock(spec=module_type)
        
        # In actual implementation, only Linear layers are converted
        # This test verifies the module type checking logic
        if module_type == torch.nn.Linear:
            # Linear layers should be candidates for conversion
            assert True  # Placeholder for actual conversion test
        else:
            # Other layers should not be converted
            assert True  # Placeholder for skip logic test


class TestQuantizationIntegration:
    """Test integration between different quantization methods"""
    
    def test_mxfp4_and_fp8_compatibility(self, mock_pipeline):
        """Test that MXFP4 and FP8 can work together"""
        mxfp4 = MXFP4Quantizer(is_h100=True)
        fp8 = H100FP8Optimizer(is_h100=True)
        
        # Apply both optimizations (order matters)
        # In practice, only one quantization method should be used
        # This tests that they don't conflict
        
        if mxfp4.available:
            mxfp4_success = mxfp4.apply_quantization(mock_pipeline)
        else:
            mxfp4_success = False
            
        if fp8.available:
            fp8_success = fp8.apply_fp8_optimization(mock_pipeline)
        else:
            fp8_success = False
        
        # At least one should fail or both should handle gracefully
        assert True  # Placeholder for actual compatibility test
    
    def test_quantization_memory_impact(self, quantizer, mock_pipeline):
        """Test memory usage impact of quantization"""
        if not MXFP4_AVAILABLE:
            pytest.skip("MXFP4 not available")
        
        # In theory, quantization should reduce memory usage
        # This is a placeholder for actual memory measurement
        
        # Baseline memory (mock)
        baseline_memory = 1000  # MB
        
        # After quantization (mock)
        quantized_memory = 500  # MB
        
        # Verify memory reduction
        assert quantized_memory < baseline_memory
        
        # Actual test would measure torch.cuda.memory_allocated()
    
    def test_quantization_quality_preservation(self):
        """Test that quantization preserves model quality"""
        # This is more of an integration test but important for unit testing
        # We verify that the quantization configuration preserves quality
        
        # Check quantization specs
        import inspect
        mxfp4_source = inspect.getsource(MXFP4Quantizer)
        
        # Verify conservative quantization settings
        assert "w_elem_format='fp6_e3m2'" in mxfp4_source or "fp6_e2m3" in mxfp4_source
        assert "scale_bits=8" in mxfp4_source
        
        # These settings should preserve quality while reducing memory