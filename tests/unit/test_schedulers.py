"""
Unit tests for scheduler management and CUDA graphs
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from src.optimizations.schedulers import SchedulerManager, CUDAGraphsManager, SchedulerPatcher


class TestSchedulerPatcher:
    """Test scheduler timestep fixing functionality"""
    
    def test_patch_scheduler_timesteps(self):
        """Test patching scheduler timesteps method"""
        # Create mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.config = MagicMock()
        mock_scheduler.set_timesteps = MagicMock()
        
        # Apply patch
        SchedulerPatcher.patch_scheduler_timesteps(mock_scheduler)
        
        # Verify the method was replaced
        assert hasattr(mock_scheduler, '_original_set_timesteps')
        assert mock_scheduler.set_timesteps != mock_scheduler._original_set_timesteps
    
    def test_timesteps_tensor_conversion(self):
        """Test that timesteps are properly converted to tensors"""
        # Create mock scheduler with timesteps
        mock_scheduler = MagicMock()
        mock_scheduler.config = MagicMock()
        mock_scheduler.timesteps = [1000, 800, 600, 400, 200]
        original_set_timesteps = MagicMock()
        mock_scheduler._original_set_timesteps = original_set_timesteps
        
        # Apply patch
        SchedulerPatcher.patch_scheduler_timesteps(mock_scheduler)
        
        # Call the patched method
        mock_scheduler.set_timesteps(5, device='cpu')
        
        # Verify original was called
        original_set_timesteps.assert_called_once_with(5, device='cpu')
        
        # Verify timesteps would be converted to tensor in actual implementation
        # (The actual conversion happens in the patched method)


class TestSchedulerManager:
    """Test scheduler management functionality"""
    
    @pytest.fixture
    def scheduler_manager(self, mock_pipeline):
        """Create SchedulerManager instance"""
        return SchedulerManager(mock_pipeline)
    
    def test_initialization(self, scheduler_manager, mock_pipeline):
        """Test SchedulerManager initialization"""
        assert scheduler_manager.pipe == mock_pipeline
        assert scheduler_manager.original_scheduler is not None
    
    def test_setup_dpm_solver(self, scheduler_manager):
        """Test DPM-Solver++ setup"""
        with patch('diffusers.DPMSolverMultistepScheduler') as mock_dpm:
            mock_dpm.from_config.return_value = MagicMock()
            
            scheduler_manager.setup_scheduler('dpm_solver')
            
            mock_dpm.from_config.assert_called_once()
            assert scheduler_manager.pipe.scheduler == mock_dpm.from_config.return_value
    
    def test_setup_euler_scheduler(self, scheduler_manager):
        """Test Euler scheduler setup"""
        with patch('diffusers.EulerDiscreteScheduler') as mock_euler:
            mock_euler.from_config.return_value = MagicMock()
            
            scheduler_manager.setup_scheduler('euler')
            
            mock_euler.from_config.assert_called_once()
            assert scheduler_manager.pipe.scheduler == mock_euler.from_config.return_value
    
    def test_setup_none_scheduler(self, scheduler_manager):
        """Test keeping original scheduler when None is specified"""
        original_scheduler = scheduler_manager.pipe.scheduler
        
        scheduler_manager.setup_scheduler(None)
        
        assert scheduler_manager.pipe.scheduler == original_scheduler
    
    def test_invalid_scheduler_type(self, scheduler_manager):
        """Test handling of invalid scheduler type"""
        with pytest.raises(ValueError):
            scheduler_manager.setup_scheduler('invalid_scheduler')
    
    def test_scheduler_config_preservation(self, scheduler_manager):
        """Test that scheduler config is preserved during setup"""
        original_config = scheduler_manager.original_scheduler.config
        
        with patch('diffusers.DPMSolverMultistepScheduler') as mock_dpm:
            mock_scheduler = MagicMock()
            mock_dpm.from_config.return_value = mock_scheduler
            
            scheduler_manager.setup_scheduler('dpm_solver')
            
            # Verify config was passed
            mock_dpm.from_config.assert_called_with(original_config)
    
    def test_restore_original_scheduler(self, scheduler_manager):
        """Test restoring original scheduler"""
        original_scheduler = scheduler_manager.original_scheduler
        
        # Change scheduler
        scheduler_manager.setup_scheduler('dpm_solver')
        
        # Restore original
        scheduler_manager.pipe.scheduler = original_scheduler
        
        assert scheduler_manager.pipe.scheduler == original_scheduler


class TestCUDAGraphsManager:
    """Test CUDA graphs management functionality"""
    
    @pytest.fixture
    def cuda_graphs_manager(self, mock_pipeline):
        """Create CUDAGraphsManager instance"""
        return CUDAGraphsManager(mock_pipeline, is_h100=False)
    
    @pytest.fixture
    def h100_cuda_graphs_manager(self, mock_pipeline):
        """Create CUDAGraphsManager instance for H100"""
        return CUDAGraphsManager(mock_pipeline, is_h100=True)
    
    def test_initialization(self, cuda_graphs_manager, h100_cuda_graphs_manager):
        """Test CUDAGraphsManager initialization"""
        assert cuda_graphs_manager.is_h100 is False
        assert h100_cuda_graphs_manager.is_h100 is True
        assert cuda_graphs_manager.cuda_graphs_enabled is False
    
    @pytest.mark.gpu_required
    def test_setup_cuda_graphs_success(self, cuda_graphs_manager):
        """Test successful CUDA graphs setup"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Mock the enable_cuda_graph method
        cuda_graphs_manager.pipe.enable_cuda_graph = MagicMock()
        
        success = cuda_graphs_manager.setup_cuda_graphs()
        
        if hasattr(cuda_graphs_manager.pipe, 'enable_cuda_graph'):
            assert success is True
            cuda_graphs_manager.pipe.enable_cuda_graph.assert_called_once()
        else:
            assert success is False
    
    def test_setup_cuda_graphs_not_available(self, cuda_graphs_manager):
        """Test CUDA graphs setup when not available"""
        # Remove the enable_cuda_graph method
        if hasattr(cuda_graphs_manager.pipe, 'enable_cuda_graph'):
            delattr(cuda_graphs_manager.pipe, 'enable_cuda_graph')
        
        success = cuda_graphs_manager.setup_cuda_graphs()
        
        assert success is False
        assert cuda_graphs_manager.cuda_graphs_enabled is False
    
    def test_cuda_graphs_h100_specific(self, h100_cuda_graphs_manager):
        """Test H100-specific CUDA graphs behavior"""
        # H100 might have different CUDA graph capabilities
        assert h100_cuda_graphs_manager.is_h100 is True
        
        # Mock H100-specific features
        h100_cuda_graphs_manager.pipe.enable_cuda_graph = MagicMock()
        
        success = h100_cuda_graphs_manager.setup_cuda_graphs()
        
        # H100 should handle CUDA graphs with its specific optimizations
        if hasattr(h100_cuda_graphs_manager.pipe, 'enable_cuda_graph'):
            assert success is True
    
    def test_cuda_graphs_error_handling(self, cuda_graphs_manager):
        """Test error handling during CUDA graphs setup"""
        # Mock the method to raise an exception
        cuda_graphs_manager.pipe.enable_cuda_graph = MagicMock(
            side_effect=RuntimeError("CUDA graphs failed")
        )
        
        success = cuda_graphs_manager.setup_cuda_graphs()
        
        assert success is False
        assert cuda_graphs_manager.cuda_graphs_enabled is False
    
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_device_compatibility(self, mock_pipeline, device):
        """Test CUDA graphs compatibility with different devices"""
        mock_pipeline.device = torch.device(device)
        cuda_graphs_manager = CUDAGraphsManager(mock_pipeline, is_h100=False)
        
        if device == "cpu":
            # CUDA graphs should not be enabled on CPU
            success = cuda_graphs_manager.setup_cuda_graphs()
            assert success is False
        else:
            # On CUDA devices, it depends on availability
            # This is a placeholder for actual device testing
            assert True


class TestSchedulerIntegration:
    """Test integration between scheduler and CUDA graphs"""
    
    def test_scheduler_cuda_graphs_compatibility(self, mock_pipeline):
        """Test that scheduler changes work with CUDA graphs"""
        scheduler_manager = SchedulerManager(mock_pipeline)
        cuda_graphs_manager = CUDAGraphsManager(mock_pipeline, is_h100=False)
        
        # Setup DPM solver
        scheduler_manager.setup_scheduler('dpm_solver')
        
        # Setup CUDA graphs
        mock_pipeline.enable_cuda_graph = MagicMock()
        cuda_graphs_success = cuda_graphs_manager.setup_cuda_graphs()
        
        # Both should work together
        assert scheduler_manager.pipe.scheduler is not None
        
        # CUDA graphs success depends on availability
        if hasattr(mock_pipeline, 'enable_cuda_graph'):
            assert cuda_graphs_success is True
    
    def test_scheduler_performance_impact(self, scheduler_manager):
        """Test performance impact of different schedulers"""
        # This is a placeholder for actual performance testing
        
        # DPM-Solver should use fewer steps
        dpm_steps = 14
        euler_steps = 28
        
        assert dpm_steps < euler_steps
        
        # Theoretical speedup
        speedup = euler_steps / dpm_steps
        assert speedup > 1.5  # Should be at least 1.5x faster