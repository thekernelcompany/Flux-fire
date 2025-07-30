"""
TensorRT optimization for FLUX.1-Kontext
"""

import torch
from typing import Optional

try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class TensorRTOptimizer:
    """TensorRT compilation for maximum performance"""
    
    def __init__(self, is_h100: bool = False):
        self.is_h100 = is_h100
        self.available = TENSORRT_AVAILABLE
        
    def setup_compilation(self, pipe):
        """Setup TensorRT compilation for maximum performance"""
        if not self.available:
            print("TensorRT not available")
            return False
            
        try:
            import torch_tensorrt
            
            print("Setting up TensorRT compilation...")
            
            # Configure TensorRT settings based on GPU
            enabled_precisions = {torch.float16, torch.bfloat16}
            if self.is_h100 and hasattr(torch, 'float8_e4m3fn'):
                enabled_precisions.add(torch.float8_e4m3fn)
            
            # Create TensorRT-compiled transformer
            trt_transformer = torch.compile(
                pipe.transformer,
                backend="tensorrt",
                options={
                    "enabled_precisions": enabled_precisions,
                    "workspace_size": 1 << 32,  # 4GB workspace
                    "min_block_size": 5,
                    "max_batch_size": 1,
                    "optimization_level": 5,
                    "use_python_runtime": True,
                }
            )
            
            # Test compilation with dummy input
            print("Testing TensorRT compilation...")
            dummy_input = torch.randn(1, 16, 64, 64, device='cuda', dtype=torch.bfloat16)
            dummy_t = torch.tensor([999], device='cuda', dtype=torch.long)
            dummy_text = torch.randn(2, 256, 4096, device='cuda', dtype=torch.bfloat16)
            
            with torch.no_grad():
                _ = trt_transformer(dummy_input, dummy_t, encoder_hidden_states=dummy_text, return_dict=False)
            
            # Replace transformer if successful
            pipe.transformer = trt_transformer
            
            # Also try to compile VAE decoder
            try:
                trt_vae_decode = torch.compile(
                    pipe.vae.decode,
                    backend="tensorrt",
                    options={
                        "enabled_precisions": {torch.float16, torch.bfloat16},
                        "workspace_size": 1 << 30,  # 1GB workspace
                        "optimization_level": 5,
                    }
                )
                pipe.vae.decode = trt_vae_decode
                print("TensorRT compilation enabled for both transformer and VAE")
            except:
                print("TensorRT compilation enabled for transformer only")
            
            return True
            
        except Exception as e:
            print(f"TensorRT compilation failed: {e}")
            return False