"""
Stage B: Monocular Depth Estimation
Uses Depth Anything V2 for metric depth estimation from single RGB images.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import time


@dataclass
class DepthResult:
    """Container for depth estimation results."""
    depth_map: np.ndarray  # 2D array of depth values in meters
    min_depth: float
    max_depth: float
    mean_depth: float
    inference_time_ms: float
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """Get depth value at a specific pixel coordinate."""
        h, w = self.depth_map.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        return float(self.depth_map[y, x])
    
    def get_depth_in_region(
        self,
        x1: int, y1: int, x2: int, y2: int,
        method: str = "median"
    ) -> float:
        """
        Get depth value for a bounding box region.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            method: Aggregation method ('median', 'mean', 'center', 'min')
        """
        h, w = self.depth_map.shape
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        
        region = self.depth_map[int(y1):int(y2), int(x1):int(x2)]
        
        if region.size == 0:
            return 0.0
            
        if method == "median":
            return float(np.median(region))
        elif method == "mean":
            return float(np.mean(region))
        elif method == "center":
            cy, cx = region.shape[0] // 2, region.shape[1] // 2
            return float(region[cy, cx])
        elif method == "min":
            return float(np.min(region))
        else:
            raise ValueError(f"Unknown method: {method}")


class MonocularDepthEstimator:
    """
    Stage B: Monocular Depth Estimation using Depth Anything V2.
    
    Estimates metric depth from single RGB images.
    """
    
    SUPPORTED_MODELS = {
        "outdoor_small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "outdoor_base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "indoor_small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "indoor_base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    }
    
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        min_depth: float = 0.5,
        max_depth: float = 10.0,
        device: str = "cpu"
    ):
        """
        Initialize the depth estimator.
        
        Args:
            model_name: Hugging Face model name or key from SUPPORTED_MODELS
            min_depth: Minimum expected depth in meters
            max_depth: Maximum expected depth in meters
            device: Device to run inference on
        """
        from transformers import pipeline
        
        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[model_name]
            
        self.model_name = model_name
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = device
        
        print(f"[Stage B] Loading {model_name}...")
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if "cuda" in device else -1
        )
        print(f"[Stage B] Model loaded successfully")
        
    def estimate_depth(
        self,
        image: np.ndarray,
        normalize_to_metric: bool = True
    ) -> DepthResult:
        """
        Estimate depth from a single RGB image.
        
        Args:
            image: Input image (RGB, numpy array or PIL Image)
            normalize_to_metric: If True, normalize depth to metric range
            
        Returns:
            DepthResult containing the depth map and statistics
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Run inference
        start_time = time.perf_counter()
        result = self.pipe(pil_image)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Get raw depth
        depth_raw = np.array(result["depth"])
        
        # Normalize to metric range if requested
        if normalize_to_metric:
            depth_normalized = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
            depth_map = self.min_depth + depth_normalized * (self.max_depth - self.min_depth)
        else:
            depth_map = depth_raw.astype(np.float32)
        
        # Resize to match input image if needed
        target_h, target_w = (pil_image.size[1], pil_image.size[0])
        if depth_map.shape != (target_h, target_w):
            depth_pil = Image.fromarray(depth_map.astype(np.float32))
            depth_pil = depth_pil.resize((target_w, target_h), Image.BILINEAR)
            depth_map = np.array(depth_pil)
        
        return DepthResult(
            depth_map=depth_map,
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            mean_depth=float(depth_map.mean()),
            inference_time_ms=inference_time
        )
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "depth_range": (self.min_depth, self.max_depth)
        }


class DepthMapVisualizer:
    """Utilities for visualizing depth maps."""
    
    @staticmethod
    def to_colormap(
        depth_map: np.ndarray,
        colormap: str = "plasma",
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert depth map to a colored visualization.
        
        Args:
            depth_map: 2D depth array
            colormap: Matplotlib colormap name
            min_depth: Minimum depth for normalization
            max_depth: Maximum depth for normalization
            
        Returns:
            RGB image (H, W, 3) uint8
        """
        import matplotlib.pyplot as plt
        
        if min_depth is None:
            min_depth = depth_map.min()
        if max_depth is None:
            max_depth = depth_map.max()
            
        normalized = (depth_map - min_depth) / (max_depth - min_depth + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        
        cmap = plt.get_cmap(colormap)
        colored = cmap(normalized)[:, :, :3]  # Remove alpha channel
        return (colored * 255).astype(np.uint8)
    
    @staticmethod
    def overlay_depth_on_image(
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "plasma"
    ) -> np.ndarray:
        """
        Overlay depth visualization on the original image.
        
        Args:
            image: Original RGB image
            depth_map: 2D depth array
            alpha: Blend factor (0 = only image, 1 = only depth)
            colormap: Matplotlib colormap name
            
        Returns:
            Blended RGB image
        """
        depth_colored = DepthMapVisualizer.to_colormap(depth_map, colormap)
        
        # Resize if needed
        if depth_colored.shape[:2] != image.shape[:2]:
            depth_pil = Image.fromarray(depth_colored)
            depth_pil = depth_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
            depth_colored = np.array(depth_pil)
        
        blended = ((1 - alpha) * image + alpha * depth_colored).astype(np.uint8)
        return blended
