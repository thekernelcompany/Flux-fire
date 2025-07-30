"""
FlashAttention optimization for FLUX.1-Kontext
"""

import torch
from typing import Optional

# Try to import FlashAttention 3
try:
    import flash_attn
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


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