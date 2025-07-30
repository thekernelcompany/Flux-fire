"""
Quantization optimizations for FLUX.1-Kontext
"""

import torch
from typing import Optional, Dict

# Try to import MXFP4 quantization
try:
    import torchmx
    from torchmx import MXTensor, dtypes
    MXFP4_AVAILABLE = True
except ImportError:
    try:
        import mx
        from mx.specs import MxSpecs
        MXFP4_AVAILABLE = True
    except ImportError:
        MXFP4_AVAILABLE = False

# Try to import H100-specific dependencies
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False


class MXFP4Quantizer:
    """Handles MXFP4 quantization for transformer and VAE"""
    
    def __init__(self, is_h100: bool = False):
        self.is_h100 = is_h100
        self.available = MXFP4_AVAILABLE
        
    def apply_quantization(self, pipe):
        """Apply MXFP4 quantization to transformer and VAE"""
        # Check if H100 FP8 is available and preferred
        if self.is_h100 and TRANSFORMER_ENGINE_AVAILABLE:
            print("H100 detected - using FP8 optimization instead of MXFP4")
            return H100FP8Optimizer(self.is_h100).apply_fp8_optimization(pipe)
            
        if not self.available:
            return
            
        print("Applying MXFP4 quantization...")
        
        try:
            if 'torchmx' in globals():
                self._apply_torchmx_quantization(pipe)
            else:
                self._apply_microsoft_mx_quantization(pipe)
        except Exception as e:
            print(f"MXFP4 quantization failed: {e}")
    
    def _apply_torchmx_quantization(self, pipe):
        """Apply TorchMX MXFP4 quantization"""
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight_mx = MXTensor.to_mx(
                    module.weight.data,
                    elem_dtype=dtypes.float4_e2m1,
                    block_size=32
                )
                module.weight.data = weight_mx.to_dtype(torch.bfloat16)
        
        print("Applied TorchMX MXFP4 quantization to transformer")
    
    def _apply_microsoft_mx_quantization(self, pipe):
        """Apply Microsoft MX MXFP4 quantization"""
        mx_specs = MxSpecs()
        mx_specs['scale_bits'] = 8
        mx_specs['w_elem_format'] = 'fp4_e2m1'
        mx_specs['a_elem_format'] = 'fp4_e2m1' 
        mx_specs['block_size'] = 32
        mx_specs['bfloat'] = 16
        mx_specs['custom_cuda'] = True
        
        from mx.linear import linear as mx_linear
        
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, torch.nn.Linear):
                original_forward = module.forward
                def quantized_forward(input):
                    return mx_linear(input, module.weight, module.bias, mx_specs)
                module.forward = quantized_forward
        
        print("Applied Microsoft MX MXFP4 quantization to transformer")


class H100FP8Optimizer:
    """H100-specific FP8 optimization using Transformer Engine"""
    
    def __init__(self, is_h100: bool = False):
        self.is_h100 = is_h100
        self.available = TRANSFORMER_ENGINE_AVAILABLE
        self.fp8_recipe = None
        self._original_modules = {}
        
    def apply_fp8_optimization(self, pipe):
        """Apply H100 FP8 optimization using Transformer Engine"""
        if not self.available:
            print("Transformer Engine not available, cannot apply FP8 optimization")
            return
            
        print("Applying H100 FP8 optimization...")
        
        # Create FP8 recipe for mixed precision
        self.fp8_recipe = recipe.DelayedScaling(
            fp8_format=recipe.Format.E4M3,
            amax_history_len=16,
            amax_compute_algo="max"
        )
        
        # Store original modules for torch.compile compatibility
        self._original_modules = {}
        
        # Convert transformer Linear layers to FP8
        replaced_count = 0
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    # Store original module
                    self._original_modules[name] = module
                    
                    # Get parent module and attribute name
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = pipe.transformer
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    
                    # Replace with Transformer Engine FP8 Linear
                    fp8_linear = te.Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        device='cuda',
                        params_dtype=torch.bfloat16,
                        parallel_mode=None
                    )
                    
                    # Copy weights
                    with torch.no_grad():
                        fp8_linear.weight.copy_(module.weight.to(torch.bfloat16))
                        if module.bias is not None:
                            fp8_linear.bias.copy_(module.bias.to(torch.bfloat16))
                    
                    # Replace module
                    setattr(parent, child_name, fp8_linear)
                    replaced_count += 1
                    
                except Exception as e:
                    print(f"Failed to replace {name}: {e}")
        
        print(f"Applied H100 FP8 optimization to {replaced_count} linear layers")
        return self._original_modules