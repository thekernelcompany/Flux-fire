"""
Pydantic models for FLUX.1-Kontext API
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class OutputFormat(Enum):
    """Supported output formats for generated images"""
    PNG = "PNG"
    JPG = "JPG"
    WEBP = "WEBP"


class InferenceRequest(BaseModel):
    """Request model for image generation"""
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


class InferenceResponse(BaseModel):
    """Response model for image generation"""
    images: list[str]  # Base64 encoded images
    inference_time: float
    seed: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    model_loaded: bool