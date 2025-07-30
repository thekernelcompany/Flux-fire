"""
Data type optimization utilities for FLUX.1-Kontext
"""

import torch
from typing import Any


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