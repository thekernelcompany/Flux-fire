"""
Configuration settings for FLUX.1-Kontext
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class Settings:
    """Application settings with environment variable support"""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.CACHE_DIR = Path(os.environ.get(
            "FLUX_CACHE_DIR", 
            Path.home() / ".cache" / "flux_optimized"
        ))
        self.OUTPUTS_DIR = self.BASE_DIR / "outputs"
        
        # Create directories if they don't exist
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Model settings
        self.MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        self.DTYPE = os.environ.get("FLUX_DTYPE", "bfloat16")
        
        # Server settings
        self.HOST = os.environ.get("HOST", "0.0.0.0")
        self.PORT = int(os.environ.get("PORT", 8000))
        
        # GPU settings
        self.CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        # Optimization settings
        self.ENABLE_FLASH_ATTENTION = os.environ.get("ENABLE_FLASH_ATTENTION", "false").lower() == "true"
        self.ENABLE_MXFP4 = os.environ.get("ENABLE_MXFP4", "true").lower() == "true"
        self.ENABLE_TENSORRT = os.environ.get("ENABLE_TENSORRT", "false").lower() == "true"
        self.ENABLE_PARA_ATTENTION = os.environ.get("ENABLE_PARA_ATTENTION", "false").lower() == "true"
        self.SCHEDULER_TYPE = os.environ.get("SCHEDULER_TYPE", "dpm_solver")
        
        # Profiling settings
        self.DISABLE_PROFILING = os.environ.get("DISABLE_PROFILING", "false").lower() == "true"
        
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration as a dictionary"""
        return {
            'flash_attention': self.ENABLE_FLASH_ATTENTION,
            'mxfp4': self.ENABLE_MXFP4,
            'scheduler': self.SCHEDULER_TYPE,
            'cuda_graphs': True,  # Always enabled
            'para_attention': self.ENABLE_PARA_ATTENTION,
            'tensorrt': self.ENABLE_TENSORRT
        }
    
    def get_cache_paths(self) -> Dict[str, Path]:
        """Get all cache paths"""
        return {
            'mega_cache': self.CACHE_DIR / "torch_mega_cache.bin",
            'hf_hub': self.CACHE_DIR / "hf_hub_cache",
            'inductor': self.CACHE_DIR / "inductor_cache",
            'triton': self.CACHE_DIR / "triton_cache",
            'cuda': self.CACHE_DIR / "nv_cache"
        }
    
    @property
    def is_h100(self) -> bool:
        """Check if running on H100 GPU"""
        import torch
        return torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)


# Global settings instance
settings = Settings()