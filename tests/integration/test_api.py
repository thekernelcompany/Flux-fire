"""
Integration tests for FastAPI server
"""

import pytest
import json
import base64
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.api.models import InferenceRequest, InferenceResponse


class TestAPIIntegration:
    """Test API endpoints and integration"""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock service for API testing"""
        service = MagicMock()
        
        # Mock inference method
        service.inference.return_value = {
            'inference_time': 1.5,
            'processing_time': 0.1,
            'total_time': 1.6,
            'saved_files': ['/tmp/test.png']
        }
        
        # Mock the actual image generation
        mock_image = np.random.rand(512, 512, 3)
        service.pipe = MagicMock()
        
        return service
    
    @pytest.fixture
    def client(self, mock_service):
        """Create test client with mock service"""
        app = create_app(service=mock_service)
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_inference_endpoint_basic(self, client, mock_service):
        """Test basic inference request"""
        request_data = {
            "prompt": "A beautiful sunset",
            "width": 512,
            "height": 512,
            "steps": 10
        }
        
        # Mock image data
        mock_image_data = base64.b64encode(b"fake_image_data").decode()
        mock_service.inference.return_value['saved_files'] = ['test.png']
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            with patch('pathlib.Path.exists', return_value=True):
                response = client.post("/inference", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "inference_time" in data
        assert "processing_time" in data
        assert "total_time" in data
        assert "images" in data
        assert "metadata" in data
    
    def test_inference_endpoint_validation(self, client):
        """Test request validation"""
        # Missing required field
        invalid_request = {
            "width": 512,
            "height": 512
        }
        
        response = client.post("/inference", json=invalid_request)
        assert response.status_code == 422
        
        # Invalid dimensions
        invalid_dims = {
            "prompt": "Test",
            "width": 100,  # Not multiple of 16
            "height": 512
        }
        
        response = client.post("/inference", json=invalid_dims)
        assert response.status_code == 422
    
    def test_inference_endpoint_full_request(self, client, mock_service):
        """Test inference with all parameters"""
        request_data = {
            "prompt": "Main prompt",
            "prompt2": "Secondary prompt",
            "negative_prompt": "Avoid this",
            "width": 1024,
            "height": 768,
            "steps": 20,
            "guidance_scale": 7.5,
            "true_cfg_scale": 1.5,
            "num_images": 2,
            "seed": 42,
            "output_format": "JPG",
            "output_quality": 85
        }
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            with patch('pathlib.Path.exists', return_value=True):
                response = client.post("/inference", json=request_data)
        
        assert response.status_code == 200
        
        # Verify service was called with correct parameters
        mock_service.inference.assert_called_once()
        call_args = mock_service.inference.call_args[1]
        assert call_args['prompt'] == "Main prompt"
        assert call_args['width'] == 1024
        assert call_args['height'] == 768
        assert call_args['steps'] == 20
        assert call_args['num_images'] == 2
        assert call_args['seed'] == 42
    
    def test_inference_error_handling(self, client, mock_service):
        """Test error handling in inference endpoint"""
        # Mock service to raise an error
        mock_service.inference.side_effect = RuntimeError("GPU out of memory")
        
        request_data = {
            "prompt": "Test prompt",
            "width": 512,
            "height": 512
        }
        
        response = client.post("/inference", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "GPU out of memory" in data["message"]
    
    @pytest.mark.parametrize("output_format", ["PNG", "JPG", "WEBP"])
    def test_different_output_formats(self, client, mock_service, output_format):
        """Test different image output formats"""
        request_data = {
            "prompt": "Test",
            "width": 512,
            "height": 512,
            "output_format": output_format
        }
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            with patch('pathlib.Path.exists', return_value=True):
                response = client.post("/inference", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that images are returned as data URLs
        if "images" in data and len(data["images"]) > 0:
            assert data["images"][0].startswith("data:image/")
    
    def test_concurrent_requests(self, client, mock_service):
        """Test handling concurrent requests"""
        import threading
        
        results = []
        errors = []
        
        def make_request():
            try:
                request_data = {
                    "prompt": "Concurrent test",
                    "width": 512,
                    "height": 512
                }
                
                with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
                    with patch('pathlib.Path.exists', return_value=True):
                        response = client.post("/inference", json=request_data)
                        results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify all requests succeeded
        assert len(errors) == 0
        assert all(status == 200 for status in results)
    
    def test_image_to_image_request(self, client, mock_service):
        """Test image-to-image generation"""
        request_data = {
            "prompt": "Enhance this image",
            "image_url": "https://example.com/image.jpg",
            "width": 512,
            "height": 512,
            "steps": 10
        }
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            with patch('pathlib.Path.exists', return_value=True):
                response = client.post("/inference", json=request_data)
        
        # Should handle image_url parameter
        assert response.status_code == 200
    
    def test_metadata_in_response(self, client, mock_service):
        """Test that metadata is included in response"""
        request_data = {
            "prompt": "Test metadata",
            "width": 512,
            "height": 512,
            "seed": 12345
        }
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            with patch('pathlib.Path.exists', return_value=True):
                response = client.post("/inference", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        metadata = data.get("metadata", {})
        assert metadata.get("prompt") == "Test metadata"
        assert metadata.get("width") == 512
        assert metadata.get("height") == 512
        assert metadata.get("seed") == 12345
        assert "model" in metadata
        assert "timestamp" in metadata


class TestAPIStartup:
    """Test API startup and initialization"""
    
    def test_app_creation_without_service(self):
        """Test creating app without pre-initialized service"""
        with patch('src.api.server.OptimizedFluxService') as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service
            
            app = create_app()
            client = TestClient(app)
            
            # Service should be created
            mock_service_class.assert_called_once()
            
            # Health check should work
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_app_creation_with_service(self):
        """Test creating app with pre-initialized service"""
        mock_service = MagicMock()
        
        app = create_app(service=mock_service)
        client = TestClient(app)
        
        # Health check should use provided service
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_startup_event(self):
        """Test startup event handler"""
        with patch('src.api.server.OptimizedFluxService') as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service
            
            app = create_app()
            
            # Simulate startup
            with TestClient(app) as client:
                # During startup, model should be loaded
                mock_service.load_model.assert_called_once()
    
    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        app = create_app(service=MagicMock())
        client = TestClient(app)
        
        response = client.options("/inference")
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers