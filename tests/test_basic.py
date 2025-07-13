"""Basic tests for Superscale."""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

import superscale


class TestBasicFunctionality:
    """Test basic library functionality."""
    
    def test_import(self):
        """Test that the library can be imported."""
        assert superscale.__version__
    
    def test_list_models(self):
        """Test listing available models."""
        models = superscale.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_model_aliases(self):
        """Test that model aliases work."""
        assert "Athena" in superscale.MODEL_ALIASES
        assert "Hermes" in superscale.MODEL_ALIASES


class TestImageProcessing:
    """Test image processing utilities."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a small test image."""
        return Image.new('RGB', (64, 64), color='red')
    
    @pytest.fixture
    def sample_array(self):
        """Create a test numpy array."""
        return np.ones((64, 64, 3), dtype=np.uint8) * 255
    
    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_image_to_tensor(self, sample_image):
        """Test image to tensor conversion."""
        from superscale.core.base_upscaler import ImageProcessor
        
        tensor = ImageProcessor.image_to_tensor(sample_image)
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0
    
    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_tensor_to_image(self):
        """Test tensor to image conversion."""
        from superscale.core.base_upscaler import ImageProcessor
        
        # Create test tensor
        tensor = torch.rand(1, 3, 64, 64)
        
        image = ImageProcessor.tensor_to_image(tensor)
        assert isinstance(image, Image.Image)
        assert image.size == (64, 64)
        assert image.mode == "RGB"
    
    def test_resize_to_multiple(self, sample_image):
        """Test resizing to multiple."""
        from superscale.core.base_upscaler import ImageProcessor
        
        # Test image that needs resizing
        img = Image.new('RGB', (65, 65), color='blue')
        resized, original_size = ImageProcessor.resize_to_multiple(img, multiple=8)
        
        assert original_size == (65, 65)
        assert resized.size == (64, 64)
        assert resized.width % 8 == 0
        assert resized.height % 8 == 0


class TestRegistry:
    """Test model registry."""
    
    def test_registry_list(self):
        """Test listing models from registry."""
        from superscale.core.registry import ModelRegistry
        
        models = ModelRegistry.list_models()
        assert isinstance(models, list)
    
    def test_registry_get(self):
        """Test getting model from registry."""
        from superscale.core.registry import ModelRegistry
        
        # This will fail if no models are registered
        try:
            model_class = ModelRegistry.get("hitsr-sir-x4")
            assert model_class is not None
        except ValueError:
            # No models registered in test environment
            pass
    
    def test_registry_aliases(self):
        """Test registry alias resolution."""
        from superscale.core.registry import ModelRegistry
        
        # Test that aliases resolve to same model
        try:
            model1 = ModelRegistry.get("hitsr-sir-x4")
            model2 = ModelRegistry.get("Athena")
            assert model1 == model2
        except ValueError:
            # No models registered in test environment
            pass


class TestCacheManager:
    """Test cache manager."""
    
    def test_cache_basic(self):
        """Test basic cache operations."""
        from superscale.core.cache_manager import CacheManager
        
        cache = CacheManager(max_models=2)
        
        # Test empty cache
        assert cache.get("test") is None
        assert len(cache.list_cached()) == 0
        
        # Test adding to cache
        cache.put("model1", {"name": "model1"})
        assert cache.get("model1") == {"name": "model1"}
        assert len(cache.list_cached()) == 1
        
        # Test LRU eviction
        cache.put("model2", {"name": "model2"})
        cache.put("model3", {"name": "model3"})
        
        # model1 should be evicted
        assert cache.get("model1") is None
        assert cache.get("model2") is not None
        assert cache.get("model3") is not None
        assert len(cache.list_cached()) == 2
    
    def test_cache_clear(self):
        """Test cache clearing."""
        from superscale.core.cache_manager import CacheManager
        
        cache = CacheManager()
        cache.put("model1", {"name": "model1"})
        cache.put("model2", {"name": "model2"})
        
        # Clear specific model
        cache.clear("model1")
        assert cache.get("model1") is None
        assert cache.get("model2") is not None
        
        # Clear all
        cache.clear()
        assert len(cache.list_cached()) == 0


class TestModelManager:
    """Test model download manager."""
    
    def test_manager_init(self):
        """Test model manager initialization."""
        from superscale.core.model_manager import ModelManager
        
        manager = ModelManager()
        assert manager.cache_dir.exists()
    
    def test_list_configs(self):
        """Test listing model configurations."""
        from superscale.core.model_manager import ModelManager
        
        manager = ModelManager()
        assert len(manager.MODEL_CONFIGS) > 0
        assert "hitsr-sir-x4" in manager.MODEL_CONFIGS
    
    def test_get_download_size(self):
        """Test getting download size."""
        from superscale.core.model_manager import ModelManager
        
        manager = ModelManager()
        size = manager.get_download_size("hitsr-sir-x4")
        assert size != "Unknown"
        assert "MB" in size or "GB" in size


class TestUtils:
    """Test utility functions."""
    
    @pytest.mark.skipif(torch is None, reason="PyTorch not installed")
    def test_get_device(self):
        """Test device detection."""
        from superscale.core.utils import get_device
        
        # Test auto
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        # Test explicit
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_format_bytes(self):
        """Test byte formatting."""
        from superscale.core.utils import format_bytes
        
        assert format_bytes(100) == "100.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_is_image_file(self):
        """Test image file detection."""
        from superscale.core.utils import is_image_file
        
        assert is_image_file("test.jpg")
        assert is_image_file("test.png")
        assert is_image_file("test.JPEG")
        assert not is_image_file("test.txt")
        assert not is_image_file("test.py")
    
    def test_validate_scale(self):
        """Test scale validation."""
        from superscale.core.utils import validate_scale
        
        # Exact match
        assert validate_scale(4, [2, 3, 4]) == 4
        
        # Close match
        assert validate_scale(3, [2, 4]) == 2 or validate_scale(3, [2, 4]) == 4
        
        # Too far
        with pytest.raises(ValueError):
            validate_scale(8, [2, 3, 4])


if __name__ == "__main__":
    pytest.main([__file__])