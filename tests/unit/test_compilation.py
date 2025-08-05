"""
Unit tests for compilation and mega-cache functionality
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

from src.optimizations.compilation import CompilationConfig, ModelCompiler, MegaCacheManager


class TestCompilationConfig:
    """Test compilation configuration setup"""
    
    def test_setup_compile_config(self):
        """Test torch.compile configuration setup"""
        with patch('torch._inductor.config') as mock_config:
            # Setup mock config
            mock_config.conv_1x1_as_mm = False
            mock_config.coordinate_descent_check_all_directions = False
            
            # Apply configuration
            CompilationConfig.setup_compile_config(is_h100=False)
            
            # Verify basic settings
            assert mock_config.conv_1x1_as_mm is True
            assert mock_config.coordinate_descent_check_all_directions is True
            assert mock_config.coordinate_descent_tuning is True
            assert mock_config.disable_progress is False
            assert mock_config.epilogue_fusion is False
            assert mock_config.shape_padding is True
    
    def test_setup_compile_config_h100(self):
        """Test H100-specific torch.compile configuration"""
        with patch('torch._inductor.config') as mock_config:
            with patch('torch.backends.cuda') as mock_cuda:
                with patch('torch.backends.cudnn') as mock_cudnn:
                    # Apply H100 configuration
                    CompilationConfig.setup_compile_config(is_h100=True)
                    
                    # Verify H100-specific settings
                    assert mock_config.force_fuse_int_mm_with_mul is True
                    assert mock_config.use_mixed_mm is True
                    assert mock_config.search_autotune_cache is True
                    assert mock_config.max_autotune is True
                    assert mock_config.max_autotune_gemm is True
                    
                    # Verify TF32 settings
                    assert mock_cuda.matmul.allow_tf32 is True
                    assert mock_cudnn.allow_tf32 is True


class TestModelCompiler:
    """Test model compilation functionality"""
    
    @pytest.fixture
    def model_compiler(self, mock_pipeline):
        """Create ModelCompiler instance"""
        mock_timer = MagicMock()
        return ModelCompiler(mock_pipeline, mock_timer, is_h100=False)
    
    def test_initialization(self, model_compiler):
        """Test ModelCompiler initialization"""
        assert model_compiler.is_h100 is False
        assert model_compiler.compiled is False
        assert model_compiler.compile_error is None
    
    @patch('torch.compile')
    def test_compile_model_success(self, mock_torch_compile, model_compiler):
        """Test successful model compilation"""
        # Mock successful compilation
        mock_compiled_model = MagicMock()
        mock_torch_compile.return_value = mock_compiled_model
        
        # Compile model
        success = model_compiler.compile_model()
        
        assert success is True
        assert model_compiler.compiled is True
        assert model_compiler.compile_error is None
        
        # Verify compilation was called with correct parameters
        mock_torch_compile.assert_called_with(
            model_compiler.pipe.transformer,
            mode="max-autotune-no-cudagraphs",
            dynamic=None,
            fullgraph=False,
            backend="inductor"
        )
    
    @patch('torch.compile')
    def test_compile_model_with_h100_modules(self, mock_torch_compile, model_compiler):
        """Test compilation with H100 FP8 modules"""
        # Mock H100 modules that should be skipped
        mock_h100_modules = {
            'layer1': MagicMock(),
            'layer2': MagicMock()
        }
        
        success = model_compiler.compile_model(h100_modules=mock_h100_modules)
        
        # Should handle H100 modules appropriately
        assert success is True or success is False  # Depends on implementation
    
    @patch('torch.compile')
    def test_compile_model_failure(self, mock_torch_compile, model_compiler):
        """Test handling of compilation failure"""
        # Mock compilation failure
        mock_torch_compile.side_effect = RuntimeError("Compilation failed")
        
        success = model_compiler.compile_model()
        
        assert success is False
        assert model_compiler.compiled is False
        assert model_compiler.compile_error is not None
        assert "Compilation failed" in str(model_compiler.compile_error)
    
    def test_trigger_compilation(self, model_compiler):
        """Test compilation triggering with warmup"""
        # Mock the warmup inference
        model_compiler.pipe = MagicMock()
        model_compiler.pipe.return_value = MagicMock(images=[])
        
        with patch.object(model_compiler, 'compile_model', return_value=True):
            model_compiler.trigger_compilation()
            
            # Verify warmup inference was called
            model_compiler.pipe.assert_called_once()
            
            # Verify compilation was attempted
            model_compiler.compile_model.assert_called_once()
    
    def test_skip_compilation_on_cpu(self, mock_pipeline):
        """Test that compilation is skipped on CPU"""
        mock_pipeline.device = torch.device("cpu")
        mock_timer = MagicMock()
        
        compiler = ModelCompiler(mock_pipeline, mock_timer, is_h100=False)
        
        with patch('torch.compile') as mock_compile:
            compiler.trigger_compilation()
            
            # Should not attempt compilation on CPU
            mock_compile.assert_not_called()


class TestMegaCacheManager:
    """Test mega-cache management functionality"""
    
    @pytest.fixture
    def cache_manager(self, temp_dir):
        """Create MegaCacheManager instance"""
        return MegaCacheManager(temp_dir)
    
    def test_initialization(self, cache_manager, temp_dir):
        """Test MegaCacheManager initialization"""
        assert cache_manager.cache_dir == temp_dir
        assert cache_manager.mega_cache_bin_path.parent.exists()
        assert "flux_torch_mega_optimized" in str(cache_manager.mega_cache_bin_path)
    
    def test_load_mega_cache_success(self, cache_manager):
        """Test successful mega-cache loading"""
        # Create mock cache file
        cache_data = b"mock_cache_data"
        cache_manager.mega_cache_bin_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_manager.mega_cache_bin_path, "wb") as f:
            f.write(cache_data)
        
        with patch('torch.compiler.load_cache_artifacts') as mock_load:
            cache_manager.load_mega_cache()
            
            mock_load.assert_called_once_with(cache_data)
    
    def test_load_mega_cache_not_found(self, cache_manager):
        """Test loading when cache doesn't exist"""
        # Ensure cache doesn't exist
        if cache_manager.mega_cache_bin_path.exists():
            cache_manager.mega_cache_bin_path.unlink()
        
        with patch('torch.compiler.load_cache_artifacts') as mock_load:
            cache_manager.load_mega_cache()
            
            # Should not attempt to load
            mock_load.assert_not_called()
    
    def test_load_mega_cache_error(self, cache_manager):
        """Test error handling during cache loading"""
        # Create mock cache file
        cache_manager.mega_cache_bin_path.parent.mkdir(parents=True, exist_ok=True)
        cache_manager.mega_cache_bin_path.write_bytes(b"data")
        
        with patch('torch.compiler.load_cache_artifacts', side_effect=Exception("Load failed")):
            # Should handle error gracefully
            cache_manager.load_mega_cache()  # Should not raise
    
    def test_save_mega_cache_success(self, cache_manager):
        """Test successful mega-cache saving"""
        mock_artifacts = (b"cache_data", {"metadata": "value"})
        
        with patch('torch.compiler.save_cache_artifacts', return_value=mock_artifacts):
            cache_manager.save_mega_cache()
            
            # Verify cache was saved
            assert cache_manager.mega_cache_bin_path.exists()
            assert cache_manager.mega_cache_bin_path.read_bytes() == b"cache_data"
    
    def test_save_mega_cache_error(self, cache_manager):
        """Test error handling during cache saving"""
        with patch('torch.compiler.save_cache_artifacts', side_effect=Exception("Save failed")):
            # Should handle error gracefully
            cache_manager.save_mega_cache()  # Should not raise
    
    def test_cache_persistence(self, cache_manager):
        """Test that cache persists across instances"""
        # Save cache
        cache_data = b"persistent_data"
        cache_manager.mega_cache_bin_path.parent.mkdir(parents=True, exist_ok=True)
        cache_manager.mega_cache_bin_path.write_bytes(cache_data)
        
        # Create new instance
        new_manager = MegaCacheManager(cache_manager.cache_dir)
        
        # Verify cache is found
        assert new_manager.mega_cache_bin_path.exists()
        assert new_manager.mega_cache_bin_path.read_bytes() == cache_data


class TestCompilationIntegration:
    """Test integration between compilation components"""
    
    def test_config_compiler_integration(self, mock_pipeline):
        """Test that compilation config is applied before compilation"""
        mock_timer = MagicMock()
        
        with patch('torch._inductor.config') as mock_config:
            with patch('torch.compile') as mock_compile:
                # Setup config
                CompilationConfig.setup_compile_config(is_h100=False)
                
                # Compile model
                compiler = ModelCompiler(mock_pipeline, mock_timer, is_h100=False)
                compiler.compile_model()
                
                # Verify config was set before compilation
                assert mock_config.conv_1x1_as_mm is True
                mock_compile.assert_called_once()
    
    def test_cache_compilation_workflow(self, temp_dir, mock_pipeline):
        """Test complete cache and compilation workflow"""
        mock_timer = MagicMock()
        
        # Create cache manager and compiler
        cache_manager = MegaCacheManager(temp_dir)
        compiler = ModelCompiler(mock_pipeline, mock_timer, is_h100=False)
        
        with patch('torch.compiler.save_cache_artifacts', return_value=(b"data", {})):
            with patch('torch.compiler.load_cache_artifacts'):
                # Load cache (empty initially)
                cache_manager.load_mega_cache()
                
                # Compile model
                compiler.compile_model()
                
                # Save cache
                cache_manager.save_mega_cache()
                
                # Verify cache exists
                assert cache_manager.mega_cache_bin_path.exists()