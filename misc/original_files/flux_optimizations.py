"""
Optimization modules for FLUX.1-Kontext - Contains various optimization implementations
"""

import os
import torch
import inspect
from typing import Optional, Dict, Any
from pathlib import Path

# Try to import optimization dependencies
try:
    import flash_attn
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

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

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False

try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class FlashAttentionOptimizer:
    """Handles FlashAttention 3 optimization for transformer layers"""
    
    def __init__(self, is_h100: bool = False):
        self.is_h100 = is_h100
        self.available = FLASH_ATTENTION_AVAILABLE
        
    def apply_patch(self, pipe):
        """Apply FlashAttention 3 patch to transformer attention layers"""
        if not self.available:
            return 0
        
        print("Applying FlashAttention 3 optimization...")
        
        if self.is_h100:
            print("Using H100-optimized FlashAttention with FP8 support")
        
        # Create a safe FlashAttention wrapper
        def safe_flash_attention(query, key, value, **kwargs):
            """Wrapper for FlashAttention that ensures correct data types"""
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)
            
            try:
                return flash_attn_func(query, key, value, **kwargs)
            except RuntimeError as e:
                if "only support fp16 and bf16" in str(e):
                    print(f"FlashAttention data type error: {e}")
                    print("Converting to fp16 and retrying...")
                    query_fp16 = query.to(torch.float16)
                    key_fp16 = key.to(torch.float16)
                    value_fp16 = value.to(torch.float16)
                    result = flash_attn_func(query_fp16, key_fp16, value_fp16, **kwargs)
                    return result.to(torch.bfloat16)
                else:
                    raise e
        
        # Patch transformer attention layers
        def flash_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):
            batch_size, seq_len, _ = hidden_states.shape
            
            hidden_states = hidden_states.to(torch.bfloat16)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
            
            query = self.to_q(hidden_states).to(torch.bfloat16)
            key = self.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states).to(torch.bfloat16)
            value = self.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states).to(torch.bfloat16)
            
            inner_dim = query.shape[-1]
            head_dim = getattr(self, 'head_dim', None)
            
            if head_dim is None:
                if hasattr(self, 'processor') and hasattr(self.processor, 'head_dim'):
                    head_dim = self.processor.head_dim
                else:
                    head_dim = 128 if inner_dim % 128 == 0 else 64
            
            num_heads = inner_dim // head_dim
            
            query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            
            attn_output = None
            
            try:
                query = query.transpose(1, 2).contiguous()
                key = key.transpose(1, 2).contiguous()
                value = value.transpose(1, 2).contiguous()
                
                attn_output = safe_flash_attention(
                    query, key, value,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=False
                )
                
                attn_output = attn_output.view(batch_size, seq_len, -1)
                
            except Exception as e:
                print(f"FlashAttention 3 failed, falling back to standard attention: {e}")
                query = query.transpose(1, 2)
                key = key.transpose(1, 2) 
                value = value.transpose(1, 2)
                
                scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
                if attention_mask is not None:
                    scores = scores + attention_mask
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, value)
                attn_output = attn_output.view(batch_size, seq_len, -1)
            
            if attn_output is None:
                print("CRITICAL: attn_output is None, using zero tensor fallback")
                attn_output = torch.zeros(batch_size, seq_len, inner_dim, device=hidden_states.device, dtype=torch.bfloat16)
            
            if hasattr(self, 'to_out') and self.to_out is not None:
                if isinstance(self.to_out, torch.nn.ModuleList):
                    attn_output = self.to_out[0](attn_output)
                else:
                    attn_output = self.to_out(attn_output)
            
            return attn_output
        
        # Apply patch to all transformer blocks
        patched_layers = 0
        failed_patches = 0
        
        for layer in pipe.transformer.transformer_blocks:
            if hasattr(layer, 'attn'):
                try:
                    # Store original forward for potential restoration
                    layer.attn.original_forward = layer.attn.forward
                    layer.attn.forward = flash_attention_forward.__get__(layer.attn, layer.attn.__class__)
                    patched_layers += 1
                except Exception as e:
                    print(f"Failed to patch attention layer: {e}")
                    failed_patches += 1
        
        if failed_patches > 0:
            print(f"WARNING: {failed_patches} attention layers failed to patch")
        
        print(f"Patched {patched_layers} transformer layers with FlashAttention 3")
        
        if patched_layers == 0 and failed_patches > 0:
            print("CRITICAL: All FlashAttention patches failed.")
        
        return patched_layers


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


class ModelDTypeOptimizer:
    """Ensures consistent data types across model components"""
    
    @staticmethod
    def ensure_model_dtype(pipe, dtype):
        """Ensure all model components are in the specified data type"""
        print(f"Ensuring all model components are {dtype}...")
        
        # List of components to check - be selective about which ones to convert
        components = []
        if hasattr(pipe, 'transformer'):
            components.append(('transformer', pipe.transformer))
        if hasattr(pipe, 'vae'):
            components.append(('vae', pipe.vae))
        # Don't convert text_encoder to bf16 - it needs to stay in its original dtype
        if hasattr(pipe, 'unet'):
            components.append(('unet', pipe.unet))
        
        for name, component in components:
            try:
                # Convert component to dtype
                component = component.to(dtype)
                
                # Convert all parameters
                if hasattr(component, 'parameters'):
                    for param in component.parameters():
                        if param.dtype != dtype:
                            param.data = param.data.to(dtype)
                    
                    # Convert all buffers
                    for buffer in component.buffers():
                        if buffer.dtype != dtype:
                            buffer.data = buffer.data.to(dtype)
                
                print(f"  ✓ {name} converted to {dtype}")
                
            except Exception as e:
                print(f"  ⚠️  Failed to convert {name} to {dtype}: {e}")
        
        print(f"Model dtype conversion complete")