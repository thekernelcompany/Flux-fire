"""
Shared pytest fixtures and configuration for all tests
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.service import OptimizedFluxService
from src.api.server import create_app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available"""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def is_h100():
    """Check if running on H100"""
    if torch.cuda.is_available():
        return "H100" in torch.cuda.get_device_name(0)
    return False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_pipeline():
    """Create a mock diffusers pipeline for testing"""
    mock_pipe = MagicMock()
    
    # Mock transformer
    mock_pipe.transformer = MagicMock()
    mock_pipe.transformer.to = MagicMock(return_value=mock_pipe.transformer)
    mock_pipe.transformer.transformer_blocks = [MagicMock() for _ in range(4)]
    
    # Mock VAE
    mock_pipe.vae = MagicMock()
    mock_pipe.vae.to = MagicMock(return_value=mock_pipe.vae)
    mock_pipe.vae.config.scaling_factor = 0.13025
    
    # Mock text encoders
    mock_pipe.text_encoder = MagicMock()
    mock_pipe.text_encoder_2 = MagicMock()
    
    # Mock scheduler
    mock_pipe.scheduler = MagicMock()
    mock_pipe.scheduler.config = MagicMock()
    
    # Mock device
    mock_pipe.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mock_pipe.to = MagicMock(return_value=mock_pipe)
    
    # Mock inference
    mock_output = MagicMock()
    mock_output.images = [np.random.rand(512, 512, 3)]
    mock_pipe.__call__ = MagicMock(return_value=mock_output)
    
    return mock_pipe


@pytest.fixture
def mock_service(mock_pipeline):
    """Create a mock OptimizedFluxService"""
    service = OptimizedFluxService()
    service.pipe = mock_pipeline
    return service


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    # Mock the service to avoid loading actual models
    mock_service = MagicMock()
    app = create_app(service=mock_service)
    return TestClient(app), mock_service


@pytest.fixture
def sample_image():
    """Generate a sample numpy image"""
    return np.random.rand(512, 512, 3).astype(np.float32)


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing"""
    return "A beautiful sunset over mountains"


@pytest.fixture
def optimization_configs():
    """Standard optimization configurations for testing"""
    return [
        {
            "name": "baseline",
            "config": {
                "flash_attention": False,
                "mxfp4": False,
                "scheduler": None,
                "cuda_graphs": False,
                "para_attention": False,
                "tensorrt": False
            }
        },
        {
            "name": "full_optimizations",
            "config": {
                "flash_attention": True,
                "mxfp4": True,
                "scheduler": "dpm_solver",
                "cuda_graphs": True,
                "para_attention": False,
                "tensorrt": False
            }
        }
    ]


@pytest.fixture
def mock_hf_token(monkeypatch):
    """Mock HuggingFace token"""
    monkeypatch.setenv("HF_TOKEN", "test_token_12345")


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("DISABLE_PROFILING", "true")
    monkeypatch.setenv("FLUX_CACHE_DIR", "/tmp/test_flux_cache")


# Skip markers for tests requiring specific hardware
def pytest_collection_modifyitems(config, items):
    """Add skip markers based on hardware availability"""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu_required" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip H100-specific tests if not on H100
    if not (torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)):
        skip_h100 = pytest.mark.skip(reason="H100 GPU not available")
        for item in items:
            if "h100_required" in item.keywords:
                item.add_marker(skip_h100)


# Performance tracking
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests"""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
        
        def record(self, name, value):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
        
        def get_average(self, name):
            if name in self.metrics and self.metrics[name]:
                return sum(self.metrics[name]) / len(self.metrics[name])
            return None
        
        def get_report(self):
            return {k: self.get_average(k) for k in self.metrics}
    
    return PerformanceTracker()


# Custom markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "h100_required: mark test to run only on H100 GPUs"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as requiring significant memory"
    )
    config.addinivalue_line(
        "markers", "network_required: mark test as requiring network access"
    )