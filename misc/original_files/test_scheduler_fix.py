#!/usr/bin/env python3
"""
Test script to verify the scheduler compatibility fix for FluxKontextPipeline
"""

import os
import sys
import torch
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from flux_kontext_optimized import OptimizedFluxService

def test_scheduler_compatibility():
    """Test both DPM-Solver and Euler scheduler configurations"""
    
    print("Testing FluxKontext Scheduler Compatibility Fix")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            "name": "DPM-Solver (Fixed)",
            "config": {"flash_attention": False, "mxfp4": False, "scheduler": "dpm_solver", "cuda_graphs": False}
        },
        {
            "name": "Euler Scheduler",
            "config": {"flash_attention": False, "mxfp4": False, "scheduler": "euler", "cuda_graphs": False}
        },
        {
            "name": "Default Scheduler",
            "config": {"flash_attention": False, "mxfp4": False, "scheduler": None, "cuda_graphs": False}
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        
        try:
            # Initialize service
            service = OptimizedFluxService()
            service.load_model(enable_optimizations=config['config'])
            
            print(f"✅ {config['name']}: Successfully loaded!")
            print(f"   Scheduler type: {type(service.pipe.scheduler).__name__}")
            
            # Test a quick inference to make sure it works
            result = service.inference(
                prompt="A simple test image",
                height=512,
                width=512,
                steps=4,  # Very few steps for quick test
                num_images=1
            )
            
            print(f"✅ {config['name']}: Inference successful!")
            print(f"   Generated {len(result['images'])} image(s)")
            
            # Clean up
            del service
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {config['name']}: Failed with error:")
            print(f"   {str(e)}")
            continue
    
    print("\n" + "=" * 50)
    print("Scheduler compatibility test complete!")

if __name__ == "__main__":
    test_scheduler_compatibility() 