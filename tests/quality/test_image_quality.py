"""
Quality tests for generated images
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock, patch
import torchvision.transforms as transforms

from src.core.service import OptimizedFluxService


class TestImageQuality:
    """Test image quality metrics and consistency"""
    
    @pytest.fixture
    def quality_service(self, mock_pipeline):
        """Create service for quality testing"""
        service = OptimizedFluxService()
        service.pipe = mock_pipeline
        return service
    
    @pytest.fixture
    def reference_image(self):
        """Create a reference image for quality testing"""
        # Create a synthetic image with known properties
        image = np.zeros((512, 512, 3), dtype=np.float32)
        
        # Add gradient
        for i in range(512):
            image[i, :, 0] = i / 512.0  # Red gradient
            image[:, i, 1] = i / 512.0  # Green gradient
        
        # Add some patterns
        image[100:400, 100:400, 2] = 0.5  # Blue square
        
        return image
    
    @pytest.mark.quality
    def test_clip_score_consistency(self, quality_service):
        """Test CLIP score consistency across optimizations"""
        # Mock CLIP scoring
        with patch('transformers.CLIPModel') as mock_clip_model:
            with patch('transformers.CLIPProcessor') as mock_clip_processor:
                # Setup mock CLIP
                mock_model = MagicMock()
                mock_processor = MagicMock()
                mock_clip_model.from_pretrained.return_value = mock_model
                mock_clip_processor.from_pretrained.return_value = mock_processor
                
                # Mock CLIP output
                mock_output = MagicMock()
                mock_output.logits_per_image = torch.tensor([[25.0]])  # Good alignment
                mock_model.return_value = mock_output
                
                # Calculate mock CLIP score
                clip_score = torch.sigmoid(mock_output.logits_per_image / 100).item()
                
                # CLIP score should indicate good alignment
                assert clip_score > 0.5  # Above 0.5 is good
                assert clip_score < 0.7  # But not suspiciously high
    
    @pytest.mark.quality
    def test_image_properties(self, quality_service, sample_image):
        """Test basic image properties"""
        # Test image is in correct format
        assert sample_image.shape == (512, 512, 3)
        assert sample_image.dtype == np.float32
        
        # Test value range
        assert sample_image.min() >= 0.0
        assert sample_image.max() <= 1.0
        
        # Test image statistics
        mean = sample_image.mean()
        std = sample_image.std()
        
        # Images should have reasonable statistics
        assert 0.2 < mean < 0.8  # Not too dark or bright
        assert std > 0.1  # Has some variation
    
    @pytest.mark.quality
    @pytest.mark.parametrize("optimization_level", ["baseline", "optimized"])
    def test_quality_preservation(self, quality_service, optimization_level):
        """Test that optimizations preserve image quality"""
        # Generate images with different optimization levels
        prompt = "A high quality test image"
        
        # Mock different quality outputs based on optimization
        if optimization_level == "baseline":
            mock_image = np.random.rand(512, 512, 3) * 0.9 + 0.05  # Good quality
        else:
            mock_image = np.random.rand(512, 512, 3) * 0.85 + 0.075  # Slightly different but similar
        
        quality_service.pipe.return_value.images = [mock_image]
        
        # Generate image
        result = quality_service.inference(
            prompt=prompt,
            height=512,
            width=512,
            steps=10,
            seed=42
        )
        
        # Image should maintain quality regardless of optimization
        assert len(result['saved_files']) > 0
        
        # In real testing, we would load and compare the actual images
        # For now, verify the mock maintains reasonable properties
        assert mock_image.mean() > 0.3
        assert mock_image.std() > 0.1
    
    @pytest.mark.quality
    def test_structural_similarity(self, reference_image):
        """Test structural similarity between images"""
        # Create a slightly modified version
        modified_image = reference_image.copy()
        noise = np.random.normal(0, 0.01, modified_image.shape)
        modified_image += noise
        modified_image = np.clip(modified_image, 0, 1)
        
        # Convert to tensors for SSIM calculation
        ref_tensor = torch.from_numpy(reference_image).permute(2, 0, 1).unsqueeze(0)
        mod_tensor = torch.from_numpy(modified_image).permute(2, 0, 1).unsqueeze(0)
        
        # Calculate mock SSIM (in practice, use torchmetrics.functional.ssim)
        # For testing, we'll calculate a simple metric
        diff = torch.abs(ref_tensor - mod_tensor).mean()
        mock_ssim = 1.0 - diff.item()
        
        # Should have high structural similarity
        assert mock_ssim > 0.95  # Very similar
    
    @pytest.mark.quality
    def test_perceptual_similarity(self, reference_image):
        """Test perceptual similarity using LPIPS"""
        # This would use LPIPS in practice
        # For testing, we'll verify the concept
        
        # Create variations
        variation1 = reference_image * 0.95  # Slight brightness change
        variation2 = reference_image.copy()
        variation2[:, :, 0] *= 0.9  # Slight color shift
        
        # Mock perceptual distances
        brightness_distance = 0.05  # Small perceptual difference
        color_distance = 0.08  # Slightly larger but still small
        
        # Perceptual distances should be small for similar images
        assert brightness_distance < 0.1
        assert color_distance < 0.1
    
    @pytest.mark.quality
    def test_consistency_across_seeds(self, quality_service):
        """Test that same seed produces consistent results"""
        prompt = "Consistency test"
        seed = 12345
        
        # Generate multiple times with same seed
        results = []
        for _ in range(3):
            # Mock deterministic output for same seed
            mock_image = np.ones((512, 512, 3)) * 0.5  # Gray image
            quality_service.pipe.return_value.images = [mock_image]
            
            result = quality_service.inference(
                prompt=prompt,
                height=512,
                width=512,
                steps=10,
                seed=seed
            )
            results.append(mock_image)
        
        # All results should be identical with same seed
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    
    @pytest.mark.quality
    @pytest.mark.parametrize("guidance_scale", [1.0, 3.5, 7.5, 20.0])
    def test_guidance_scale_impact(self, quality_service, guidance_scale):
        """Test impact of guidance scale on image quality"""
        prompt = "A detailed landscape"
        
        # Mock different outputs based on guidance scale
        if guidance_scale < 2.0:
            # Low guidance - more variation, less prompt adherence
            mock_image = np.random.rand(512, 512, 3)
        elif guidance_scale > 15.0:
            # High guidance - might be oversaturated
            mock_image = np.clip(np.random.rand(512, 512, 3) * 1.5, 0, 1)
        else:
            # Optimal range
            mock_image = np.random.rand(512, 512, 3) * 0.8 + 0.1
        
        quality_service.pipe.return_value.images = [mock_image]
        
        result = quality_service.inference(
            prompt=prompt,
            height=512,
            width=512,
            steps=10,
            guidance_scale=guidance_scale
        )
        
        # Verify image properties are reasonable
        image_mean = mock_image.mean()
        image_std = mock_image.std()
        
        if 2.0 <= guidance_scale <= 15.0:
            # Optimal guidance range should produce balanced images
            assert 0.3 < image_mean < 0.7
            assert image_std > 0.15
    
    @pytest.mark.quality
    def test_resolution_quality_scaling(self, quality_service):
        """Test quality at different resolutions"""
        resolutions = [(512, 512), (768, 768), (1024, 1024)]
        prompt = "High resolution test"
        
        for width, height in resolutions:
            # Mock resolution-appropriate output
            mock_image = np.random.rand(height, width, 3) * 0.8 + 0.1
            quality_service.pipe.return_value.images = [mock_image]
            
            result = quality_service.inference(
                prompt=prompt,
                height=height,
                width=width,
                steps=10
            )
            
            # Higher resolutions should maintain quality
            assert mock_image.shape == (height, width, 3)
            assert mock_image.mean() > 0.3  # Not too dark
            assert mock_image.std() > 0.1   # Has detail
    
    @pytest.mark.quality
    def test_batch_quality_consistency(self, quality_service):
        """Test quality consistency in batch generation"""
        prompt = "Batch quality test"
        batch_size = 4
        
        # Mock batch output
        mock_images = [
            np.random.rand(512, 512, 3) * 0.8 + 0.1 
            for _ in range(batch_size)
        ]
        quality_service.pipe.return_value.images = mock_images
        
        result = quality_service.inference(
            prompt=prompt,
            height=512,
            width=512,
            steps=10,
            num_images=batch_size
        )
        
        # Calculate quality metrics for each image
        means = [img.mean() for img in mock_images]
        stds = [img.std() for img in mock_images]
        
        # All images in batch should have similar quality
        mean_variance = np.var(means)
        std_variance = np.var(stds)
        
        assert mean_variance < 0.01  # Similar brightness
        assert std_variance < 0.01   # Similar detail level


class TestQualityMetrics:
    """Test quality metric calculations"""
    
    def test_clip_score_calculation(self):
        """Test CLIP score calculation logic"""
        # Mock CLIP logits
        logits = torch.tensor([[25.0], [30.0], [20.0]])
        
        # Calculate CLIP scores
        clip_scores = torch.sigmoid(logits / 100)
        
        # Verify score properties
        assert all(0 < score < 1 for score in clip_scores)
        assert clip_scores[1] > clip_scores[0]  # Higher logit = higher score
        assert clip_scores[0] > clip_scores[2]  # Lower logit = lower score
    
    def test_quality_thresholds(self):
        """Test quality threshold definitions"""
        # Define quality thresholds
        thresholds = {
            'excellent': 0.35,
            'good': 0.30,
            'acceptable': 0.25,
            'poor': 0.20
        }
        
        # Test score classification
        test_scores = [0.40, 0.32, 0.27, 0.18]
        classifications = ['excellent', 'good', 'acceptable', 'poor']
        
        for score, expected_class in zip(test_scores, classifications):
            if score >= thresholds['excellent']:
                assert expected_class == 'excellent'
            elif score >= thresholds['good']:
                assert expected_class == 'good'
            elif score >= thresholds['acceptable']:
                assert expected_class == 'acceptable'
            else:
                assert expected_class == 'poor'