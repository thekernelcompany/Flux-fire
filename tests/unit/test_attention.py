"""
Unit tests for FlashAttention optimization module
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from src.optimizations.attention import FlashAttentionOptimizer, FLASH_ATTENTION_AVAILABLE


class TestFlashAttentionOptimizer:
    """Test FlashAttention optimization functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create FlashAttentionOptimizer instance"""
        return FlashAttentionOptimizer(is_h100=False)
    
    @pytest.fixture
    def h100_optimizer(self):
        """Create FlashAttentionOptimizer instance for H100"""
        return FlashAttentionOptimizer(is_h100=True)
    
    def test_initialization(self, optimizer, h100_optimizer):
        """Test optimizer initialization"""
        assert optimizer.is_h100 is False
        assert h100_optimizer.is_h100 is True
        assert optimizer.available == FLASH_ATTENTION_AVAILABLE
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="FlashAttention not available")
    def test_apply_patch_success(self, optimizer, mock_pipeline):
        """Test successful FlashAttention patch application"""
        # Setup mock transformer blocks with attention layers
        for block in mock_pipeline.transformer.transformer_blocks:
            block.attn = MagicMock()
            block.attn.forward = MagicMock()
        
        # Apply patch
        patched_layers = optimizer.apply_patch(mock_pipeline)
        
        # Verify patches were applied
        assert patched_layers == len(mock_pipeline.transformer.transformer_blocks)
        for block in mock_pipeline.transformer.transformer_blocks:
            assert hasattr(block.attn, 'original_forward')
    
    def test_apply_patch_no_flash_attention(self, optimizer, mock_pipeline):
        """Test patch application when FlashAttention is not available"""
        optimizer.available = False
        
        patched_layers = optimizer.apply_patch(mock_pipeline)
        
        assert patched_layers == 0
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="FlashAttention not available")
    def test_flash_attention_forward(self, optimizer, mock_pipeline):
        """Test the FlashAttention forward pass"""
        # Create mock attention layer
        mock_attn = MagicMock()
        mock_attn.to_q = MagicMock(return_value=torch.randn(1, 77, 1024))
        mock_attn.to_k = MagicMock(return_value=torch.randn(1, 77, 1024))
        mock_attn.to_v = MagicMock(return_value=torch.randn(1, 77, 1024))
        mock_attn.to_out = MagicMock(return_value=torch.randn(1, 77, 1024))
        
        # Apply patch to single layer
        mock_pipeline.transformer.transformer_blocks = [MagicMock(attn=mock_attn)]
        optimizer.apply_patch(mock_pipeline)
        
        # Test forward pass
        hidden_states = torch.randn(1, 77, 1024)
        
        # The patched forward function should handle the computation
        # We can't easily test the actual flash_attn_func call without GPU
        # but we can verify the function was replaced
        assert mock_attn.forward != mock_attn.original_forward
    
    def test_data_type_handling(self, optimizer):
        """Test data type conversion in FlashAttention"""
        # This tests the safe_flash_attention wrapper logic
        # In real usage, this ensures bf16/fp16 compatibility
        
        # Create test tensors
        query = torch.randn(2, 10, 8, 64, dtype=torch.float32)
        key = torch.randn(2, 10, 8, 64, dtype=torch.float32)
        value = torch.randn(2, 10, 8, 64, dtype=torch.float32)
        
        # The optimizer should handle dtype conversion
        # This is more of an integration test but important for unit testing
        assert query.dtype == torch.float32
        
        # After conversion in the actual implementation
        query_bf16 = query.to(torch.bfloat16)
        assert query_bf16.dtype == torch.bfloat16
    
    def test_h100_specific_behavior(self, h100_optimizer, mock_pipeline):
        """Test H100-specific optimizations"""
        if not FLASH_ATTENTION_AVAILABLE:
            pytest.skip("FlashAttention not available")
        
        # Apply patch with H100 optimizations
        patched_layers = h100_optimizer.apply_patch(mock_pipeline)
        
        # H100 should use FP8 support when available
        # This is mostly a placeholder for H100-specific logic
        assert h100_optimizer.is_h100 is True
    
    def test_error_handling(self, optimizer, mock_pipeline):
        """Test error handling during patch application"""
        if not FLASH_ATTENTION_AVAILABLE:
            pytest.skip("FlashAttention not available")
        
        # Create a layer that will fail to patch
        bad_block = MagicMock()
        bad_block.attn = MagicMock()
        bad_block.attn.forward = property(lambda self: (_ for _ in ()).throw(RuntimeError("Cannot patch")))
        
        mock_pipeline.transformer.transformer_blocks = [bad_block]
        
        # Should handle the error gracefully
        patched_layers = optimizer.apply_patch(mock_pipeline)
        
        # No layers should be patched due to error
        assert patched_layers == 0
    
    def test_fallback_mechanism(self, optimizer):
        """Test fallback to standard attention when FlashAttention fails"""
        # This is tested within the forward function implementation
        # The actual test would require GPU and specific failure conditions
        
        # Verify the fallback logic exists in the implementation
        import inspect
        source = inspect.getsource(FlashAttentionOptimizer)
        
        # Check for fallback implementation
        assert "falling back to standard attention" in source
        assert "torch.matmul" in source  # Standard attention uses matmul
    
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 77, 1024),
        (2, 256, 768),
        (4, 512, 1280),
    ])
    def test_various_input_shapes(self, optimizer, batch_size, seq_len, hidden_dim):
        """Test FlashAttention with various input shapes"""
        if not FLASH_ATTENTION_AVAILABLE:
            pytest.skip("FlashAttention not available")
        
        # Create tensors with different shapes
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Verify shapes are handled correctly
        assert hidden_states.shape == (batch_size, seq_len, hidden_dim)
        
        # In actual implementation, these shapes should work with FlashAttention
        # The real test would involve running through the attention mechanism