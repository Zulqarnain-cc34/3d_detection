import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import time


@dataclass
class DepthResult:
    depth_map: np.ndarray  # 2D array of depth values in meters
    max_depth: float       # Maximum depth in the scene (meters)
    mean_depth: float      # Mean depth (meters)
    inference_time_ms: float
    
    def get_depth_in_region(
        self,
        x1: int, y1: int, x2: int, y2: int,
        method: str = "median"
    ) -> float:
        """
        Get depth value within a bounding box region.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            method: Aggregation method ('median', 'mean', 'center', 'min', 'percentile')
            
        Returns:
            Depth value in meters
        """
        # Clip to image bounds
        h, w = self.depth_map.shape
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return self.mean_depth
            
        region = self.depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return self.mean_depth
        
        # Filter out invalid depths (zeros or very large values)
        valid_mask = (region > 0.1) & (region < 100.0)
        valid_depths = region[valid_mask]
        
        if valid_depths.size == 0:
            valid_depths = region.flatten()
            
        if method == "median":
            return float(np.median(valid_depths))
        elif method == "mean":
            return float(np.mean(valid_depths))
        elif method == "center":
            cy, cx = region.shape[0] // 2, region.shape[1] // 2
            return float(region[cy, cx])
        elif method == "min":
            return float(np.min(valid_depths))
        elif method == "percentile":
            # Use 30th percentile - objects are usually at front of bbox
            return float(np.percentile(valid_depths, 30))
        else:
            raise ValueError(f"Unknown method: {method}")


class MonocularDepthEstimator:
    """
    Stage B: Monocular Depth Estimation using Depth Anything V2.
    
    IMPORTANT: Uses metric depth models that output actual meters.
    The depth values are calibrated for indoor/outdoor scenes.
    """
    
    SUPPORTED_MODELS = {
        # Metric models - output actual depth in meters
        "indoor_small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "indoor_base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "outdoor_small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "outdoor_base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    }
    
    def __init__(
        self,
        model_name: str = "indoor_small",
        device: str = "cpu",
        depth_scale: float = 1.0,
        max_depth_clip: float = 20.0
    ):
        """
        Initialize the depth estimator.
        
        Args:
            model_name: Model key from SUPPORTED_MODELS or full HF model name
            device: Device to run inference on ('cpu', 'cuda:0')
            depth_scale: Scale factor to correct depth (use for calibration)
            max_depth_clip: Maximum depth to clip to (meters)
        """
        from transformers import pipeline
        
        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[model_name]
        
        self.model_name = model_name
        self.device = device
        self.depth_scale = depth_scale
        self.max_depth_clip = max_depth_clip
        
        print(f"[Stage B] Loading {model_name}...")
        print(f"[Stage B] Depth scale factor: {depth_scale}")
        
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if "cuda" in device else -1
        )
        print(f"[Stage B] Model loaded successfully")
        
    def estimate_depth(
        self,
        image: np.ndarray,
    ) -> DepthResult:
        """
        Estimate metric depth from a single RGB image.
        
        The Depth Anything V2 Metric models output depth in METERS directly.
        We do NOT normalize - we use the raw metric output.
        
        Args:
            image: Input image (RGB, numpy array or PIL Image)
            
        Returns:
            DepthResult containing the depth map in meters
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
        
        # Get depth map - this is ALREADY in meters for metric models
        depth_map = np.array(result["depth"]).astype(np.float32)
        
        # Apply calibration scale factor
        depth_map = depth_map * self.depth_scale
        
        # Clip to reasonable range
        depth_map = np.clip(depth_map, 0.1, self.max_depth_clip)
        
        # Resize to match input image if needed
        target_h, target_w = pil_image.size[1], pil_image.size[0]
        if depth_map.shape != (target_h, target_w):
            depth_pil = Image.fromarray(depth_map)
            depth_pil = depth_pil.resize((target_w, target_h), Image.BILINEAR)
            depth_map = np.array(depth_pil)
        
        return DepthResult(
            depth_map=depth_map,
            max_depth=float(depth_map.max()),
            mean_depth=float(depth_map.mean()),
            inference_time_ms=inference_time
        )
    
    def calibrate_depth_scale(
        self,
        measured_depth: float,
        estimated_depth: float
    ) -> float:
        """
        Calculate the depth scale factor for calibration.
        
        Usage:
            1. Measure actual distance to an object (e.g., 3 meters)
            2. Run detection and get estimated depth
            3. Call this function to get scale factor
            4. Set self.depth_scale to the returned value
        
        Args:
            measured_depth: Actual measured distance in meters
            estimated_depth: Depth value from the model
            
        Returns:
            Scale factor to apply to depth values
        """
        scale = measured_depth / estimated_depth
        print(f"[Stage B] Calibration: measured={measured_depth}m, estimated={estimated_depth}m")
        print(f"[Stage B] Recommended depth_scale: {scale:.3f}")
        return scale
    
    def set_depth_scale(self, scale: float) -> None:
        """Set the depth scale factor."""
        self.depth_scale = scale
        print(f"[Stage B] Depth scale set to: {scale}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "depth_scale": self.depth_scale,
            "max_depth_clip": self.max_depth_clip
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
            depth_map: 2D depth array (meters)
            colormap: Matplotlib colormap name
            min_depth: Minimum depth for normalization (default: auto)
            max_depth: Maximum depth for normalization (default: auto)
            
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
        """Overlay depth visualization on the original image."""
        depth_colored = DepthMapVisualizer.to_colormap(depth_map, colormap)
        
        if depth_colored.shape[:2] != image.shape[:2]:
            depth_pil = Image.fromarray(depth_colored)
            depth_pil = depth_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
            depth_colored = np.array(depth_pil)
        
        blended = ((1 - alpha) * image + alpha * depth_colored).astype(np.uint8)
        return blended
