"""
Compilation utilities for FLUX.1-Kontext - Handles torch.compile and related optimizations
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any


class CompilationConfig:
    """Configuration for torch compilation"""
    
    @staticmethod
    def setup_compile_config(is_h100: bool = False):
        """Setup torch inductor config for compilation"""
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
        if is_h100:
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


class ModelCompiler:
    """Handles model compilation with torch.compile"""
    
    def __init__(self, pipe, timer=None, is_h100: bool = False):
        self.pipe = pipe
        self.timer = timer
        self.is_h100 = is_h100
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup environment variables for compilation"""
        os.environ["TORCHDYNAMO_DISABLE"] = "0"
        os.environ["TORCH_LOGS"] = "dynamo"
        
    def compile_model(self, original_modules: Optional[Dict] = None):
        """Compile transformer and VAE with conservative settings"""
        compile_dynamic = True  # Always use dynamic for better compatibility
        
        # Use more conservative compilation settings to avoid Linear.forward issues
        compile_options = {
            "max_autotune": True,
            "coordinate_descent_tuning": True,
            "epilogue_fusion": True,
        }
        
        # Temporarily disable H100 FP8 optimization for compilation
        if original_modules:
            print("Temporarily disabling H100 FP8 optimization for torch.compile compatibility...")
            self._restore_original_modules(original_modules)
        
        try:
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                dynamic=compile_dynamic,
                fullgraph=False,  # Disable fullgraph for better compatibility
                backend="inductor",
                options=compile_options
            )
            
            self.pipe.vae.decode = torch.compile(
                self.pipe.vae.decode,
                dynamic=compile_dynamic,
                backend="inductor",
                options=compile_options
            )
            
            print("torch.compile applied successfully")
            return True
            
        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("Continuing without torch.compile() - model will still work but may be slower")
            return False
    
    def _restore_original_modules(self, original_modules: Dict):
        """Restore original Linear layers for compilation"""
        for name, module in self.pipe.transformer.named_modules():
            if name in original_modules:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.pipe.transformer
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                # Restore original torch.nn.Linear
                original_linear = torch.nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
                with torch.no_grad():
                    original_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        original_linear.bias.copy_(module.bias)
                setattr(parent, child_name, original_linear)
    
    def trigger_compilation(self):
        """Trigger torch compilation with dummy inputs"""
        # Monkey-patch for para-attn compatibility
        from torch._inductor.fx_passes import post_grad
        
        if not hasattr(post_grad, "_orig_same_meta"):
            post_grad._orig_same_meta = post_grad.same_meta
            
            def _safe_same_meta(node1, node2):
                try:
                    return post_grad._orig_same_meta(node1, node2)
                except AttributeError as e:
                    if "SymFloat" in str(e) and "size" in str(e):
                        return False
                    raise
            
            post_grad.same_meta = _safe_same_meta
        
        print("Triggering torch compile...")
        
        # Temporarily disable FlashAttention for compilation
        original_attention_methods = {}
        if hasattr(self.pipe, 'transformer'):
            for layer in self.pipe.transformer.transformer_blocks:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'original_forward'):
                    original_attention_methods[layer.attn] = layer.attn.forward
                    layer.attn.forward = layer.attn.original_forward
        
        try:
            # First compilation attempt
            print("First compilation attempt...")
            self.pipe(prompt="dummy prompt", height=1024, width=1024, num_images_per_prompt=1)
            
            print("Recompiling for dynamic batch size...")
            self.pipe(prompt="dummy prompt", height=1024, width=1024, num_images_per_prompt=2)
            
            print("torch.compile successful!")
            
        except Exception as e:
            print(f"Compilation failed: {e}")
            self._handle_compilation_error(e)
        
        finally:
            # Restore FlashAttention after compilation
            if original_attention_methods:
                print("Restoring FlashAttention after compilation...")
                for layer in self.pipe.transformer.transformer_blocks:
                    if hasattr(layer, 'attn') and layer.attn in original_attention_methods:
                        layer.attn.forward = original_attention_methods[layer.attn]
    
    def _handle_compilation_error(self, error):
        """Handle specific compilation errors"""
        error_str = str(error)
        
        # Check for specific Linear.forward tracing issue
        if "Linear.forward" in error_str and "should not be traced" in error_str:
            print("Detected Linear.forward tracing issue - this is likely due to H100 FP8 optimization")
            print("Attempting to compile with more conservative settings...")
            
            # Try with more conservative compilation
            try:
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer,
                    dynamic=True,
                    fullgraph=False,
                    backend="inductor",
                    options={"max_autotune": False}
                )
                
                self.pipe.vae.decode = torch.compile(
                    self.pipe.vae.decode,
                    dynamic=True,
                    backend="inductor",
                    options={"max_autotune": False}
                )
                
                print("Conservative compilation successful")
                
            except Exception as e2:
                print(f"Conservative compilation also failed: {e2}")
                print("Disabling torch.compile() - model will still work but may be slower")
                return
        
        # Check if it's the para-attention unittest.mock issue
        elif "unittest.mock" in error_str or "_patch_object" in error_str:
            print("Detected para-attention unittest.mock conflict")
            print("Para-attention conflict detected. Consider setting para_attention=False")
            print("Continuing without compilation...")
        
        # Try disabling FlashAttention
        elif "flash_attention" in error_str.lower():
            print("FlashAttention compilation issue detected")
            print("Consider disabling FlashAttention for better compilation compatibility")
        else:
            print("Unknown compilation error - continuing without torch.compile()")
            print("Model will still work but may be slower")


class MegaCacheManager:
    """Manages torch mega-cache for faster loads"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.mega_cache_bin_path = cache_dir / "torch_mega_cache.bin"
        
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