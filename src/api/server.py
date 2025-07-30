"""
FastAPI server for FLUX.1-Kontext
"""

import os
import sys
import time
import torch
import base64
from pathlib import Path
from io import BytesIO
from typing import Optional
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.models import InferenceRequest, InferenceResponse, HealthResponse, OutputFormat
from src.core.service import OptimizedFluxService


def create_app(service: Optional[OptimizedFluxService] = None) -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="FLUX.1-Kontext API",
        description="High-performance image generation API using FLUX.1-Kontext-dev",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize service if not provided
    if service is None:
        service = OptimizedFluxService()
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup"""
        if not hasattr(service, 'pipe') or service.pipe is None:
            service.load_model()
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            gpu_available=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            model_loaded=hasattr(service, 'pipe') and service.pipe is not None
        )
    
    @app.post("/inference", response_model=InferenceResponse)
    async def generate_image(request: InferenceRequest):
        """Generate images based on the request"""
        try:
            start_time = time.time()
            
            # Download and process input image if provided
            input_image = None
            if request.image_url:
                try:
                    with urllib.request.urlopen(request.image_url) as response:
                        input_image = Image.open(BytesIO(response.read())).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
            
            # Generate images
            result = service.inference(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                num_images=request.num_images,
                seed=request.seed
            )
            
            # Convert images to base64
            encoded_images = []
            for img_path in result['saved_files']:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    encoded_images.append(f"data:image/{request.output_format.value.lower()};base64,{img_base64}")
            
            inference_time = time.time() - start_time
            
            return InferenceResponse(
                images=encoded_images,
                inference_time=inference_time,
                seed=request.seed
            )
            
        except Exception as e:
            return InferenceResponse(
                images=[],
                inference_time=0.0,
                error=str(e)
            )
    
    return app


def main():
    """Main entry point for the server"""
    # Initialize service
    service = OptimizedFluxService()
    
    # Create app
    app = create_app(service)
    
    # Run server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting FLUX.1-Kontext server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()