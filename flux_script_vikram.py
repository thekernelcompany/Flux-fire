"""
Modal script for serving the FLUX.1-dev text-to-image generation model.

This script defines a Modal application that sets up a containerized environment
with the FLUX.1-dev model from HuggingFace. It includes optimizations like
TorchInductor compilation, custom caching layers (para_attn), and a "mega-cache"
for compiled artifacts to ensure fast cold starts and efficient inference.

The application exposes a FastAPI endpoint (`/inference`) that accepts prompts
and various generation parameters, generates images using the FLUX model,
uploads them to a Cloudflare R2 bucket, and returns presigned URLs for accessing
the generated images.

Key Components:
- NVIDIA CUDA base image with Python 3.12.
- Dependencies for PyTorch, Diffusers, Transformers, FastAPI, etc.
- Modal Volume for persistent caching of model weights and compiled artifacts.
- Cloudflare R2 bucket integration for image output storage.
- `FluxService` class encapsulating model loading, optimization, compilation,
  and the inference logic.
- `InferenceRequest` Pydantic model for validating and structuring API requests.
- Optimizations including QKV projection fusion, channels-last memory format,
  and extensive TorchInductor configurations and caching.
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Container-mount directories
# ---------------------------------------------------------------------------
CONTAINER_CACHE_DIR = Path("/cache")
"""Directory path within the Modal container used for storing various caches.

This includes HuggingFace Hub cache, TorchInductor cache, Triton cache,
and custom mega-cache artifacts. It is typically mapped to a persistent
Modal Volume.
"""

CONTAINER_CLOUD_MOUNT_DIR = Path("/outputs")
"""Directory path within the Modal container used as a temporary mount point
for uploading generated images to a cloud storage bucket (e.g., Cloudflare R2).
Files written here are synced to the configured cloud bucket.
"""

# Modal volume for caching compiled model artifacts and other caches
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("flux_endpoint", create_if_missing=True)
"""A persistent Modal Volume used to store caches across container runs.

This helps in reducing cold start times by preserving downloaded model weights,
compiled TorchInductor artifacts (mega-cache), and other cacheable data.
The volume is named "flux_endpoint".
"""

# Cloudflare R2 bucket details for image storage
CLOUD_BUCKET_ACCOUNT_ID = "954ed10c1c6c0499480b64904329202f"
"""The Account ID for the Cloudflare R2 bucket where generated images are stored."""

CLOUD_BUCKET_NAME = "agents"
"""The name of the Cloudflare R2 bucket used for storing generated images."""

# ---------------------------------------------------------------------------
# Build the container image
# ---------------------------------------------------------------------------
cuda_version = "12.6.3"
"""Specifies the CUDA version to be used in the base NVIDIA Docker image."""
flavor = "devel"
"""Specifies the flavor (e.g., 'base', 'runtime', 'devel') of the CUDA image."""
operating_system = "ubuntu24.04"
"""Specifies the operating system of the CUDA image."""
tag = f"{cuda_version}-{flavor}-{operating_system}"
"""Constructed tag for the NVIDIA CUDA Docker image (e.g., '12.6.3-devel-ubuntu24.04')."""

nvidia_cuda_image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12").entrypoint(
    []
)
"""Modal Image definition for the base NVIDIA CUDA image.

Uses the specified tag and adds Python 3.12. The entrypoint is cleared.
"""

flux_endpoint_image = nvidia_cuda_image.pip_install(
    "accelerate==1.6.0",
    "boto3==1.37.35",
    "diffusers==0.33.1",
    "fastapi[standard]==0.115.12",
    "huggingface-hub[hf_transfer]==0.30.2",
    "numpy==2.2.4",
    "opencv-python-headless==4.11.0.86",
    "para-attn==0.3.32",
    "pydantic==2.11.4",
    "safetensors==0.5.3",
    "sentencepiece==0.2.0",
    "torch==2.7.0",
    "transformers==4.51.3",
).env(
    {
        # Disable Inductor multiprocessing: 1 == compile inside main proc
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "CUDA_CACHE_PATH": str(CONTAINER_CACHE_DIR / ".nv_cache"),
        "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".inductor_cache"),
        "TRITON_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".triton_cache"),
    }
)
"""Modal Image definition for the FLUX endpoint.

Builds upon the `nvidia_cuda_image` by:
- Installing necessary Python packages (PyTorch, Diffusers, FastAPI, etc.).
- Setting environment variables for TorchInductor, HuggingFace Hub,
  and various cache directories. These are pointed to locations within
  `CONTAINER_CACHE_DIR` to leverage the persistent Modal Volume.
"""

# ---------------------------------------------------------------------------
# Create the Modal app
# ---------------------------------------------------------------------------
app = modal.App("flux_endpoint", image=flux_endpoint_image)
"""The main Modal App instance for this FLUX endpoint.

It is named "flux_endpoint" and uses the `flux_endpoint_image`
which contains all necessary dependencies and configurations.
"""

with flux_endpoint_image.imports():
    import concurrent.futures
    from enum import Enum
    import os
    import time
    from typing import Optional
    import uuid

    import boto3
    import cv2
    import numpy as np
    import torch

    # ↓↓↓ Disable subprocess pool again for safety
    import torch._inductor.config as ind_cfg

    ind_cfg.compile_threads = 1
    # ↑↑↑

    from diffusers import FluxPipeline
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    from pydantic import BaseModel, Field
    from torch._inductor.fx_passes import post_grad

    # -----------------------------------------------------------------------
    # Supported output formats
    # -----------------------------------------------------------------------
    class OutputFormat(Enum):
        """Enumeration of supported image output formats."""

        PNG = "PNG"
        """Portable Network Graphics format. Good for lossless compression."""
        JPG = "JPG"
        """Joint Photographic Experts Group format. Good for lossy compression of photos."""
        WEBP = "WEBP"
        """WebP format. Offers both lossless and lossy compression, often with better
        compression ratios than PNG or JPG.
        """

    # -----------------------------------------------------------------------
    # Request / response model (Pydantic)
    # -----------------------------------------------------------------------
    class InferenceRequest(BaseModel):
        """
        Pydantic model defining the structure for inference requests to the FLUX endpoint.

        Attributes:
            prompt (str): The main positive text prompt for image generation.
            prompt2 (Optional[str]): An optional secondary positive prompt.
            negative_prompt (Optional[str]): An optional negative prompt to specify
                what to avoid in the image.
            negative_prompt2 (Optional[str]): An optional secondary negative prompt.
            true_cfg_scale (float): Classifier-Free Guidance scale. Controls how
                strongly the prompt influences the generation. (Default: 1.0)
            height (int): Height of the generated image in pixels. Must be a
                multiple of 16. (Default: 1024, Min: 256, Max: 1024)
            width (int): Width of the generated image in pixels. Must be a
                multiple of 16. (Default: 1024, Min: 256, Max: 1024)
            steps (int): Number of denoising steps. More steps can lead to higher
                quality but take longer. (Default: 28, Min: 1, Max: 50)
            guidance_scale (float): Guidance scale for the diffusion process.
                (Default: 3.5)
            num_images (int): Number of images to generate per prompt.
                (Default: 1, Min: 1, Max: 4)
            seed (Optional[int]): Random seed for reproducibility. If None, a random
                seed will be used.
            output_format (OutputFormat): The desired format for the output images
                (PNG, JPG, WEBP). (Default: OutputFormat.PNG)
            output_quality (int): Quality setting for JPG and WEBP formats,
                ranging from 1 to 100. Higher is better quality. (Default: 90)
        """

        prompt: str
        prompt2: Optional[str] = None
        negative_prompt: Optional[str] = None
        negative_prompt2: Optional[str] = None
        true_cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0, multiple_of=0.1)
        height: int = Field(default=1024, ge=256, le=1024, multiple_of=16)
        width: int = Field(default=1024, ge=256, le=1024, multiple_of=16)
        steps: int = Field(default=28, ge=1, le=50)
        guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0, multiple_of=0.1)
        num_images: int = Field(default=1, ge=1, le=4)
        seed: Optional[int] = None
        output_format: OutputFormat = Field(default=OutputFormat.PNG)
        output_quality: int = Field(default=90, ge=1, le=100)


# ---------------------------------------------------------------------------
# FluxService class – handles model loading / optimisation / inference
# ---------------------------------------------------------------------------
@app.cls(
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name(
            "r2-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ),
    ],
    gpu="H100",
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
        CONTAINER_CLOUD_MOUNT_DIR: modal.CloudBucketMount(
            bucket_name=CLOUD_BUCKET_NAME,
            bucket_endpoint_url=f"https://{CLOUD_BUCKET_ACCOUNT_ID}.r2.cloudflarestorage.com",
            secret=modal.Secret.from_name(
                "r2-secret",
                required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
            ),
        ),
    },
    min_containers=0,
    buffer_containers=0,
    scaledown_window=1200,  # 10 min
    timeout=3600,  # 1 h
    enable_memory_snapshot=True,
)
class FluxService:
    """
    Manages the FLUX.1-dev model, including loading, optimization, compilation,
    and serving inference requests via a FastAPI endpoint.

    This class leverages Modal's features for GPU acceleration, persistent storage
    for caches, and secrets management. It applies several optimizations to the
    FLUX pipeline for improved performance, including `para_attn` caching,
    QKV projection fusion, channels-last memory format, and extensive
    TorchInductor compilation with a custom "mega-cache".

    Modal Lifecycle:
    - `@modal.enter(snap=True)` (`load` method): Handles initial model loading
      from HuggingFace Hub. This runs once when the Modal image is snapshotted.
    - `@modal.enter(snap=False)` (`setup` method): Performs further setup like
      moving the model to GPU, applying optimizations, compiling components,
      and initializing the S3 client. This runs on every container (re)start.
    - `@modal.fastapi_endpoint` (`inference` method): Exposes the image generation
      functionality as a web endpoint.

    Attributes:
        pipe (FluxPipeline): The loaded FLUX.1-dev diffusion pipeline.
        mega_cache_bin_path (Path): Path to the TorchInductor mega-cache file.
        s3_client (boto3.client): Boto3 S3 client for interacting with Cloudflare R2.
    """

    # -----------------------------------------------------------------------
    # Private optimisation helpers
    # -----------------------------------------------------------------------
    def _optimize(self):
        """
        Applies various optimizations to the loaded FLUX pipeline.

        Optimizations include:
        - Applying `para_attn` caching for attention blocks.
        - Fusing QKV (Query, Key, Value) projections in the transformer and VAE.
        - Converting transformer and VAE to channels-last memory format.
        - Setting TorchInductor configuration flags for better performance
          (e.g., `conv_1x1_as_mm`, `coordinate_descent_tuning`).
        """
        apply_cache_on_pipe(
            self.pipe,
            residual_diff_threshold=0.12,
        )

        # Fuse qkv projections + channels-last
        self.pipe.transformer.fuse_qkv_projections()
        self.pipe.vae.fuse_qkv_projections()
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

        # Inductor config tweaks
        cfg = torch._inductor.config
        cfg.conv_1x1_as_mm = True
        cfg.coordinate_descent_check_all_directions = True
        cfg.coordinate_descent_tuning = True
        cfg.disable_progress = False
        cfg.epilogue_fusion = False
        cfg.shape_padding = True

        # Torch compile with dynamic shapes
        self.pipe.transformer = torch.compile(
            self.pipe.transformer,
            mode="max-autotune-no-cudagraphs",
            dynamic=True,
        )
        self.pipe.vae.decode = torch.compile(
            self.pipe.vae.decode,
            mode="max-autotune-no-cudagraphs",
            dynamic=True,
        )

    def _compile(self):
        """
        Triggers TorchInductor compilation for the transformer and VAE decode components.

        This method first applies a patch for a known SymFloat bug in a TorchInductor
        pass. It then runs dummy inference calls to trigger the compilation process,
        once with a fixed batch size and then again to ensure dynamic batch sizes
        are handled correctly by the compiled graph.
        """
        # Patch SymFloat bug in remove_noop_ops pass
        if not hasattr(post_grad, "_orig_same_meta"):
            post_grad._orig_same_meta = post_grad.same_meta

            def _safe_same_meta(n1, n2):
                try:
                    return post_grad._orig_same_meta(n1, n2)
                except AttributeError as e:
                    if "SymFloat" in str(e) and "size" in str(e):
                        return False
                    raise

            post_grad.same_meta = _safe_same_meta

        print("triggering torch compile pass")
        self.pipe("dummy prompt", height=1024, width=1024, num_images_per_prompt=1)
        print("recompiling for dynamic batch size")
        self.pipe("dummy prompt", height=1024, width=1024, num_images_per_prompt=1)

    # -----------------------------------------------------------------------
    # Mega-cache helpers
    # -----------------------------------------------------------------------
    def _load_mega_cache(self):
        """
        Loads TorchInductor compiled artifacts from a "mega-cache" file.

        This helps in significantly reducing compilation times on subsequent runs
        by reusing previously compiled graphs. If the cache file doesn't exist
        or an error occurs during loading, a message is printed, and compilation
        will proceed as normal (and a new cache will be saved later).
        The mega-cache is stored in the path defined by `self.mega_cache_bin_path`.
        """
        try:
            if self.mega_cache_bin_path.exists():
                with open(self.mega_cache_bin_path, "rb") as f:
                    bytes_ = f.read()
                if bytes_:
                    torch.compiler.load_cache_artifacts(bytes_)
            else:
                print("mega-cache missing: will regenerate")
        except Exception as e:
            print(f"mega-cache load error: {e}")

    def _save_mega_cache(self):
        """
        Saves the current TorchInductor compiled artifacts to the "mega-cache" file.

        After model components are compiled (e.g., during the `_compile` step),
        this method saves the resulting cache artifacts to `self.mega_cache_bin_path`
        and commits the changes to the `CONTAINER_CACHE_VOLUME` to persist them
        across container runs.
        """
        try:
            bytes_, _ = torch.compiler.save_cache_artifacts()
            with open(self.mega_cache_bin_path, "wb") as f:
                f.write(bytes_)
            CONTAINER_CACHE_VOLUME.commit()
        except Exception as e:
            print(f"mega-cache save error: {e}")

    # -----------------------------------------------------------------------
    # Memory snapshot entrypoints
    # -----------------------------------------------------------------------
    @modal.enter(snap=True)
    def load(self):
        """
        Modal entrypoint for initial model loading (runs once during image snapshot).

        This method is responsible for:
        - Passing the HuggingFace token for `hf_transfer` if available.
        - Downloading the FLUX.1-dev model weights from HuggingFace Hub.
          The model is initially loaded onto the CPU to be part of the snapshot.
        - Setting up the path for the TorchInductor mega-cache file.
        """
        # Pass HF token through (for HF Transfer)
        if "HUGGINGFACE_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

        print("Loading FLUX.1 weights …")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cpu")

        mega_cache_dir = CONTAINER_CACHE_DIR / ".mega_cache"
        mega_cache_dir.mkdir(parents=True, exist_ok=True)
        self.mega_cache_bin_path = mega_cache_dir / "flux_torch_mega"

    @modal.enter(snap=False)
    def setup(self):
        """
        Modal entrypoint for final setup (runs on every container start, after snapshot).

        This method performs tasks that require the GPU or are specific to a running
        container instance:
        - Moves the loaded `self.pipe` (FLUX pipeline) to the CUDA device.
        - Loads the pre-compiled "mega-cache" artifacts.
        - Applies optimizations (`self._optimize()`).
        - Triggers TorchInductor compilation (`self._compile()`).
        - Saves any newly compiled artifacts back to the "mega-cache".
        - Initializes the Boto3 S3 client for communication with Cloudflare R2,
          using credentials from Modal secrets.
        """
        self.pipe.to("cuda")
        self._load_mega_cache()
        self._optimize()
        self._compile()
        self._save_mega_cache()

        # R2 client
        self.s3_client = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{CLOUD_BUCKET_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    # -----------------------------------------------------------------------
    # FastAPI inference endpoint
    # -----------------------------------------------------------------------
    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest) -> list[str]:
        """
        FastAPI endpoint for generating images using the FLUX.1-dev model.

        Accepts an `InferenceRequest` Pydantic model containing the prompt,
        negative prompts, and various generation parameters.

        The process involves:
        1. Setting up a PyTorch CUDA generator with the provided seed (if any).
        2. Running the FLUX pipeline inference with the given parameters.
        3. Measuring the inference duration.
        4. Post-processing the generated images in parallel:
           - Converting images to the requested output format (PNG, JPG, WEBP).
           - Handling quality settings for JPG/WEBP.
           - Saving images temporarily to a directory mounted to Cloudflare R2.
        5. Uploading images to the R2 bucket.
        6. Generating presigned URLs for accessing the uploaded images.
        7. Measuring post-processing and upload duration.

        Args:
            request (InferenceRequest): The request object containing all parameters
                                        for image generation.

        Returns:
            List[str]: A list of presigned URLs pointing to the generated images
                       stored in Cloudflare R2. URLs are valid for 24 hours.
        """
        gen = (
            torch.Generator("cuda").manual_seed(request.seed) if request.seed is not None else None
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        images = self.pipe(
            prompt=request.prompt,
            prompt_2=request.prompt2,
            negative_prompt=request.negative_prompt,
            negative_prompt_2=request.negative_prompt2,
            true_cfg_scale=request.true_cfg_scale,
            height=request.height,
            width=request.width,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images,
            generator=gen,
            output_type="np",
        ).images

        torch.cuda.synchronize()
        print(f"inference: {time.perf_counter() - t0:.2f}s")
        t1 = time.perf_counter()

        # Parallel post-processing / R2 upload
        CONTAINER_CLOUD_MOUNT_DIR.mkdir(parents=True, exist_ok=True)

        def process(img):
            """
            Processes a single generated image: saves it in the specified format,
            uploads it to R2, and returns a presigned URL.

            This function is designed to be run in a thread pool for parallel
            processing of multiple generated images.

            Args:
                img (np.ndarray): The generated image as a NumPy array (RGB, float 0-1).

            Returns:
                str: A presigned URL for the uploaded image.
            """
            fname = f"{uuid.uuid4()}.{request.output_format.value.lower()}"
            out_path = CONTAINER_CLOUD_MOUNT_DIR / fname

            img_np = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if request.output_format is OutputFormat.JPG:
                params = [cv2.IMWRITE_JPEG_QUALITY, request.output_quality]
            elif request.output_format is OutputFormat.WEBP:
                params = [cv2.IMWRITE_WEBP_QUALITY, request.output_quality]
            else:
                params = []

            cv2.imwrite(str(out_path), img_bgr, params)

            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": CLOUD_BUCKET_NAME, "Key": fname},
                ExpiresIn=86_400,
            )

        with concurrent.futures.ThreadPoolExecutor() as pool:
            urls = list(pool.map(process, images))

        torch.cuda.synchronize()
        print(f"postproc+upload: {time.perf_counter() - t1:.2f}s")
        return urls
