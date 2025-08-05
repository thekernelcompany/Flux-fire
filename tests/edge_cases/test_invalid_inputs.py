"""
Edge case tests for invalid inputs and error handling
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.core.service import OptimizedFluxService
from src.api.server import create_app
from fastapi.testclient import TestClient


class TestInvalidInputs:
    """Test handling of invalid inputs"""
    
    @pytest.fixture
    def edge_service(self):
        """Create service for edge case testing"""
        return OptimizedFluxService()
    
    @pytest.mark.edge_case
    def test_empty_prompt(self, edge_service, mock_pipeline):
        """Test handling of empty prompt"""
        edge_service.pipe = mock_pipeline
        
        # Empty string prompt
        result = edge_service.inference(
            prompt="",
            height=512,
            width=512,
            steps=10
        )
        
        # Should handle empty prompt gracefully
        assert 'inference_time' in result
        
        # None prompt should raise error
        with pytest.raises(TypeError):
            edge_service.inference(
                prompt=None,
                height=512,
                width=512,
                steps=10
            )
    
    @pytest.mark.edge_case
    @pytest.mark.parametrize("width,height,should_fail", [
        (0, 512, True),          # Zero width
        (512, 0, True),          # Zero height
        (-512, 512, True),       # Negative width
        (512, -512, True),       # Negative height
        (100, 512, False),       # Non-standard but valid
        (513, 512, False),       # Not multiple of 16 but valid
        (16384, 512, True),      # Too large
        (512, 16384, True),      # Too large
    ])
    def test_invalid_dimensions(self, edge_service, mock_pipeline, width, height, should_fail):
        """Test handling of invalid image dimensions"""
        edge_service.pipe = mock_pipeline
        
        if should_fail:
            with pytest.raises((ValueError, RuntimeError, AssertionError)):
                edge_service.inference(
                    prompt="Dimension test",
                    height=height,
                    width=width,
                    steps=10
                )
        else:
            # Should handle non-standard dimensions
            result = edge_service.inference(
                prompt="Dimension test",
                height=height,
                width=width,
                steps=10
            )
            assert 'inference_time' in result
    
    @pytest.mark.edge_case
    @pytest.mark.parametrize("steps", [0, -1, 1000, 1.5, "10", None])
    def test_invalid_steps(self, edge_service, mock_pipeline, steps):
        """Test handling of invalid step counts"""
        edge_service.pipe = mock_pipeline
        
        if isinstance(steps, (int, float)) and 1 <= steps <= 100:
            # Valid range
            result = edge_service.inference(
                prompt="Steps test",
                height=512,
                width=512,
                steps=int(steps)
            )
            assert 'inference_time' in result
        else:
            # Invalid steps
            with pytest.raises((ValueError, TypeError, AssertionError)):
                edge_service.inference(
                    prompt="Steps test",
                    height=512,
                    width=512,
                    steps=steps
                )
    
    @pytest.mark.edge_case
    @pytest.mark.parametrize("guidance_scale", [-1.0, 0.0, 50.0, float('inf'), float('nan'), "7.5"])
    def test_invalid_guidance_scale(self, edge_service, mock_pipeline, guidance_scale):
        """Test handling of invalid guidance scale values"""
        edge_service.pipe = mock_pipeline
        
        if isinstance(guidance_scale, str) or not np.isfinite(guidance_scale):
            # Invalid type or non-finite
            with pytest.raises((ValueError, TypeError)):
                edge_service.inference(
                    prompt="Guidance test",
                    height=512,
                    width=512,
                    steps=10,
                    guidance_scale=guidance_scale
                )
        elif guidance_scale < 0:
            # Negative guidance scale might be rejected
            try:
                result = edge_service.inference(
                    prompt="Guidance test",
                    height=512,
                    width=512,
                    steps=10,
                    guidance_scale=guidance_scale
                )
                # Some implementations might allow it
                assert 'inference_time' in result
            except ValueError:
                # Others might reject it
                pass
    
    @pytest.mark.edge_case
    def test_extremely_long_prompt(self, edge_service, mock_pipeline):
        """Test handling of extremely long prompts"""
        edge_service.pipe = mock_pipeline
        
        # Create very long prompt
        long_prompt = "test " * 10000  # 50,000 characters
        
        # Should handle or truncate gracefully
        result = edge_service.inference(
            prompt=long_prompt,
            height=512,
            width=512,
            steps=10
        )
        
        assert 'inference_time' in result
    
    @pytest.mark.edge_case
    def test_special_characters_in_prompt(self, edge_service, mock_pipeline):
        """Test handling of special characters in prompt"""
        edge_service.pipe = mock_pipeline
        
        special_prompts = [
            "Test with emoji 🎨🖼️",
            "Test with unicode: ñáéíóú",
            "Test with symbols: @#$%^&*()",
            "Test with newlines:\nMultiple\nLines",
            "Test with tabs:\t\tIndented",
            'Test with quotes: "quoted" and \'single\'',
            "Test with backslash: \\escape",
            "Test with null char: \0",
        ]
        
        for prompt in special_prompts:
            result = edge_service.inference(
                prompt=prompt,
                height=512,
                width=512,
                steps=10
            )
            assert 'inference_time' in result
    
    @pytest.mark.edge_case
    def test_invalid_seed(self, edge_service, mock_pipeline):
        """Test handling of invalid seed values"""
        edge_service.pipe = mock_pipeline
        
        invalid_seeds = [
            -1,  # Negative seed
            2**64,  # Too large
            float('inf'),
            float('nan'),
            "random",
            [],
            {},
        ]
        
        for seed in invalid_seeds:
            if isinstance(seed, (int, float)) and not np.isfinite(seed):
                # Non-finite numbers should fail
                with pytest.raises((ValueError, TypeError, OverflowError)):
                    edge_service.inference(
                        prompt="Seed test",
                        height=512,
                        width=512,
                        steps=10,
                        seed=seed
                    )
            elif isinstance(seed, int) and seed < 0:
                # Negative seeds might be handled differently
                try:
                    result = edge_service.inference(
                        prompt="Seed test",
                        height=512,
                        width=512,
                        steps=10,
                        seed=seed
                    )
                    assert 'inference_time' in result
                except ValueError:
                    pass  # Some implementations reject negative seeds
            elif not isinstance(seed, int):
                # Wrong type
                with pytest.raises(TypeError):
                    edge_service.inference(
                        prompt="Seed test",
                        height=512,
                        width=512,
                        steps=10,
                        seed=seed
                    )
    
    @pytest.mark.edge_case
    def test_invalid_num_images(self, edge_service, mock_pipeline):
        """Test handling of invalid num_images values"""
        edge_service.pipe = mock_pipeline
        
        invalid_counts = [0, -1, 100, 1.5, "2", None]
        
        for num_images in invalid_counts:
            if isinstance(num_images, int) and 1 <= num_images <= 10:
                # Valid range
                result = edge_service.inference(
                    prompt="Batch test",
                    height=512,
                    width=512,
                    steps=10,
                    num_images=num_images
                )
                assert 'inference_time' in result
            else:
                # Invalid
                with pytest.raises((ValueError, TypeError)):
                    edge_service.inference(
                        prompt="Batch test",
                        height=512,
                        width=512,
                        steps=10,
                        num_images=num_images
                    )


class TestAPIEdgeCases:
    """Test API edge cases"""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client"""
        mock_service = MagicMock()
        mock_service.inference.return_value = {
            'inference_time': 1.0,
            'processing_time': 0.1,
            'total_time': 1.1,
            'saved_files': ['test.png']
        }
        
        app = create_app(service=mock_service)
        return TestClient(app), mock_service
    
    @pytest.mark.edge_case
    def test_malformed_json(self, api_client):
        """Test API handling of malformed JSON"""
        client, _ = api_client
        
        # Invalid JSON
        response = client.post(
            "/inference",
            data="{'invalid': json}",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.edge_case
    def test_missing_content_type(self, api_client):
        """Test API handling of missing content type"""
        client, _ = api_client
        
        response = client.post(
            "/inference",
            data='{"prompt": "test", "width": 512, "height": 512}'
        )
        
        # Should handle missing content type
        assert response.status_code in [200, 422]
    
    @pytest.mark.edge_case
    def test_extra_fields_in_request(self, api_client):
        """Test API handling of extra fields in request"""
        client, _ = api_client
        
        response = client.post("/inference", json={
            "prompt": "Test",
            "width": 512,
            "height": 512,
            "extra_field": "should be ignored",
            "another_extra": 123
        })
        
        # Should accept and ignore extra fields
        assert response.status_code == 200
    
    @pytest.mark.edge_case
    def test_unicode_in_api_request(self, api_client):
        """Test API handling of unicode in requests"""
        client, _ = api_client
        
        response = client.post("/inference", json={
            "prompt": "测试 テスト тест 🎨",
            "width": 512,
            "height": 512
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @pytest.mark.edge_case
    def test_large_api_request(self, api_client):
        """Test API handling of large requests"""
        client, _ = api_client
        
        # Create large prompt (1MB of text)
        large_prompt = "a" * (1024 * 1024)
        
        response = client.post("/inference", json={
            "prompt": large_prompt,
            "width": 512,
            "height": 512
        })
        
        # Should handle or reject gracefully
        assert response.status_code in [200, 413, 422]
    
    @pytest.mark.edge_case
    def test_concurrent_api_errors(self, api_client):
        """Test API handling of concurrent errors"""
        client, mock_service = api_client
        
        # Make service raise errors
        mock_service.inference.side_effect = RuntimeError("Test error")
        
        # Send multiple error-producing requests
        responses = []
        for i in range(5):
            response = client.post("/inference", json={
                "prompt": f"Error test {i}",
                "width": 512,
                "height": 512
            })
            responses.append(response)
        
        # All should return error status
        assert all(r.status_code == 500 for r in responses)
        assert all(r.json()["status"] == "error" for r in responses)


class TestResourceExhaustion:
    """Test behavior under resource exhaustion"""
    
    @pytest.mark.edge_case
    @pytest.mark.memory_intensive
    def test_out_of_memory_handling(self, mock_pipeline):
        """Test handling of out-of-memory errors"""
        service = OptimizedFluxService()
        service.pipe = mock_pipeline
        
        # Mock OOM error
        def raise_oom(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        
        mock_pipeline.__call__ = raise_oom
        
        # Should handle OOM gracefully
        with pytest.raises(torch.cuda.OutOfMemoryError):
            service.inference(
                prompt="OOM test",
                height=4096,  # Very large
                width=4096,
                steps=10
            )
    
    @pytest.mark.edge_case
    def test_model_not_loaded(self):
        """Test inference without loading model"""
        service = OptimizedFluxService()
        
        # Should raise appropriate error
        with pytest.raises((AttributeError, RuntimeError)):
            service.inference(
                prompt="No model test",
                height=512,
                width=512,
                steps=10
            )
    
    @pytest.mark.edge_case
    def test_corrupted_model_state(self, mock_pipeline):
        """Test handling of corrupted model state"""
        service = OptimizedFluxService()
        service.pipe = mock_pipeline
        
        # Simulate corrupted state
        mock_pipeline.transformer = None
        
        # Should handle gracefully
        with pytest.raises((AttributeError, RuntimeError)):
            service.inference(
                prompt="Corrupted model test",
                height=512,
                width=512,
                steps=10
            )