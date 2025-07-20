#!/usr/bin/env python3
"""
Standalone FLUX.1 Kontext inference script for running on remote servers with GPU.
Converted from Modal deployment to direct execution.
"""

import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional
from enum import Enum
import urllib.request
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from diffusers import FluxKontextPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn

# Configuration
CACHE_DIR = Path.home() / ".cache" / "flux_endpoint"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Environment variables for optimization
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["CUDA_CACHE_PATH"] = str(CACHE_DIR / ".nv_cache")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / ".hf_hub_cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR / ".inductor_cache")
os.environ["TRITON_CACHE_DIR"] = str(CACHE_DIR / ".triton_cache")

# Supported output formats
class OutputFormat(Enum):
    PNG = "PNG"
    JPG = "JPG"
    WEBP = "WEBP"

# Request model
class InferenceRequest(BaseModel):
    prompt: str
    prompt2: Optional[str] = None
    negative_prompt: Optional[str] = None
    negative_prompt2: Optional[str] = None
    image_url: Optional[str] = None
    true_cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0, multiple_of=0.1)
    height: int = Field(default=1024, ge=256, le=1024, multiple_of=16)
    width: int = Field(default=1024, ge=256, le=1024, multiple_of=16)
    steps: int = Field(default=28, ge=1, le=50)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0, multiple_of=0.1)
    num_images: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = None
    output_format: OutputFormat = Field(default=OutputFormat.PNG)
    output_quality: int = Field(default=90, ge=1, le=100)

class FluxService:
    def __init__(self):
        self.pipe = None
        self.mega_cache_bin_path = CACHE_DIR / ".mega_cache" / "flux_torch_mega"
        self.mega_cache_bin_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _optimize(self):
        """Apply optimization techniques to the pipeline"""
        # using FBcache from the ParaAttention paper:
        apply_cache_on_pipe(self.pipe, residual_diff_threshold=0.12),

        # Fuse qkv projections
        self.pipe.transformer.fuse_qkv_projections()
        self.pipe.vae.fuse_qkv_projections()
        
        # Use channels last memory format
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)
        
        # Torch compile configs
        config = torch._inductor.config
        config.conv_1x1_as_mm = True
        config.coordinate_descent_check_all_directions = True
        config.coordinate_descent_tuning = True
        config.disable_progress = False
        config.epilogue_fusion = False
        config.shape_padding = True
        
        # Compile transformer and VAE
        self.pipe.transformer = torch.compile(
            self.pipe.transformer,
            mode="max-autotune-no-cudagraphs",
            dynamic=True
        )
        
        self.pipe.vae.decode = torch.compile(
            self.pipe.vae.decode,
            mode="max-autotune-no-cudagraphs",
            dynamic=True
        )
    
    def _compile(self):
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
        self.pipe(prompt="dummy prompt", height=1024, width=1024, num_images_per_prompt=1)
        
        print("Recompiling for dynamic batch size...")
        self.pipe(prompt="dummy prompt", height=1024, width=1024, num_images_per_prompt=2)
    
    def _load_mega_cache(self):
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
    
    def _save_mega_cache(self):
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
    
    def load_model(self):
        """Load and prepare the FLUX model"""
        print("Loading FLUX.1-Kontext model...")
        
        # Check if HuggingFace token is set
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("Using HuggingFace token for model download")
        
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            token=hf_token
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU available, using CPU (will be very slow)")
            self.pipe = self.pipe.to("cpu")
        
        # Apply optimizations
        self._load_mega_cache()
        self._optimize()
        self._compile()
        self._save_mega_cache()
        
        print("Model ready for inference")
    
    def inference(self, request: InferenceRequest):
        """Run inference on the model"""
        generator = (
            torch.Generator(self.pipe.device).manual_seed(request.seed)
            if request.seed is not None
            else None
        )
        
        # Download and process input image if provided
        input_image = None
        if request.image_url:
            try:
                with urllib.request.urlopen(request.image_url) as response:
                    image_data = response.read()
                
                input_image = Image.open(io.BytesIO(image_data))
                
                if input_image.size != (request.width, request.height):
                    input_image = input_image.resize(
                        (request.width, request.height), 
                        Image.Resampling.LANCZOS
                    )
                
                print(f"Loaded input image: {input_image.size}")
            except Exception as e:
                print(f"Error loading input image: {e}")
                input_image = None
        
        # Time the inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        
        # Prepare pipeline arguments
        pipeline_kwargs = {
            "prompt": request.prompt,
            "prompt_2": request.prompt2,
            "negative_prompt": request.negative_prompt,
            "negative_prompt_2": request.negative_prompt2,
            "true_cfg_scale": request.true_cfg_scale,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "num_images_per_prompt": request.num_images,
            "generator": generator,
            "output_type": "np",
        }
        
        if input_image is not None:
            pipeline_kwargs["image"] = input_image
        
        # Generate images
        images = self.pipe(**pipeline_kwargs).images
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.perf_counter() - t0
        print(f"Inference time: {inference_time:.2f}s")
        
        # Process images
        t1 = time.perf_counter()
        image_results = []
        
        for idx, image in enumerate(images):
            # Convert to uint8
            image_np = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_np)
            
            # Save to bytes buffer
            buffer = BytesIO()
            pil_image.save(
                buffer, 
                format=request.output_format.value, 
                quality=request.output_quality
            )
            
            # Create data URL
            image_data = base64.b64encode(buffer.getvalue()).decode()
            data_url = f"data:image/{request.output_format.value.lower()};base64,{image_data}"
            
            # Save to file (optional)
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            filename = f"{uuid.uuid4()}.{request.output_format.value.lower()}"
            filepath = output_dir / filename
            pil_image.save(filepath, format=request.output_format.value, quality=request.output_quality)
            
            image_results.append({
                "data_url": data_url,
                "filename": str(filename),
                "filepath": str(filepath)
            })
        
        processing_time = time.perf_counter() - t1
        print(f"Image processing time: {processing_time:.2f}s")
        
        return {
            "images": image_results,
            "inference_time": inference_time,
            "processing_time": processing_time,
            "total_time": inference_time + processing_time
        }

# Initialize service
service = FluxService()

# Create FastAPI app
app = FastAPI(title="FLUX.1 Kontext API")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    service.load_model()

@app.post("/inference")
async def inference_endpoint(request: InferenceRequest):
    """Run inference on the FLUX model"""
    return service.inference(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

if __name__ == "__main__":
    # Run benchmark comparison
    print("\n=== FLUX.1 Kontext Speed Comparison ===\n")
    
    # Test parameters
    test_request = InferenceRequest(
        prompt="A majestic mountain landscape at sunset with vibrant colors",
        height=512,
        width=512,
        steps=20,
        guidance_scale=3.5,
        seed=42  # Fixed seed for consistent results
    )
    
    print(f"Test config: {test_request.width}x{test_request.height}, {test_request.steps} steps")
    print(f"Prompt: '{test_request.prompt}'\n")
    
    # Test 1: WITHOUT optimizations
    print("=== TEST 1: Without Optimizations ===")
    service_unoptimized = FluxService()
    
    # Load model without optimizations
    print("Loading model WITHOUT optimizations...")
    hf_token = os.environ.get("HF_TOKEN")
    service_unoptimized.pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        token=hf_token
    )
    if torch.cuda.is_available():
        service_unoptimized.pipe = service_unoptimized.pipe.to("cuda")
    print("Model loaded (unoptimized)\n")
    
    # Warm-up run
    print("Warm-up run...")
    _ = service_unoptimized.inference(test_request)
    
    # Benchmark unoptimized
    print("\nBenchmarking unoptimized model...")
    unopt_times = []
    for i in range(3):
        result = service_unoptimized.inference(test_request)
        unopt_times.append(result['inference_time'])
        print(f"Run {i+1}: {result['inference_time']:.2f}s")
    
    avg_unopt = sum(unopt_times) / len(unopt_times)
    print(f"\nAverage unoptimized inference time: {avg_unopt:.2f}s")
    
    # Clear memory
    del service_unoptimized
    torch.cuda.empty_cache()
    
    # Test 2: WITH optimizations
    print("\n=== TEST 2: With Optimizations ===")
    service_optimized = FluxService()
    service_optimized.load_model()
    
    # Benchmark optimized
    print("\nBenchmarking optimized model...")
    opt_times = []
    for i in range(3):
        result = service_optimized.inference(test_request)
        opt_times.append(result['inference_time'])
        print(f"Run {i+1}: {result['inference_time']:.2f}s")
    
    avg_opt = sum(opt_times) / len(opt_times)
    print(f"\nAverage optimized inference time: {avg_opt:.2f}s")
    
    # Results summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Unoptimized: {avg_unopt:.2f}s")
    print(f"Optimized:   {avg_opt:.2f}s")
    print(f"Speedup:     {avg_unopt/avg_opt:.2f}x faster")
    print(f"Time saved:  {avg_unopt - avg_opt:.2f}s per image")
    
    # Save final test image
    print(f"\nFinal image saved to: {result['images'][0]['filepath']}")
