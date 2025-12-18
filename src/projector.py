"""
Stage C: 3D Geometric Projection
Projects 2D detections to 3D space using depth information.

FIXED: Added depth_scale parameter and improved depth sampling
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import from local modules
from src.depth_estimator import DepthResult
from src.detector import Detection2D


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length X (pixels)
    fy: float  # Focal length Y (pixels)
    cx: float  # Principal point X (pixels)
    cy: float  # Principal point Y (pixels)
    
    # Image dimensions (optional, for validation)
    width: Optional[int] = None
    height: Optional[int] = None
    
    @classmethod
    def from_fov(cls, fov_degrees: float, width: int, height: int) -> 'CameraIntrinsics':
        """
        Create camera intrinsics from field of view.
        
        Args:
            fov_degrees: Horizontal field of view in degrees
            width: Image width in pixels
            height: Image height in pixels
        """
        fov_rad = np.radians(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    
    @classmethod
    def default_for_video(cls, width: int = 640, height: int = 480) -> 'CameraIntrinsics':
        """
        Create default intrinsics for typical webcam/video.
        Assumes ~60 degree horizontal FOV.
        """
        # Typical webcam has ~60-70 degree HFOV
        fov_degrees = 60.0
        return cls.from_fov(fov_degrees, width, height)
    
    def pixel_to_3d(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates + depth to 3D point.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            depth: Depth in meters
            
        Returns:
            3D point (X, Y, Z) in camera frame (meters)
        """
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return np.array([X, Y, Z])
    
    def to_dict(self) -> Dict:
        return {
            "fx": self.fx,
            "fy": self.fy, 
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height
        }


@dataclass
class BoundingBox3D:
    """3D bounding box representation."""
    center: np.ndarray        # (X, Y, Z) center in meters
    dimensions: np.ndarray    # (width, height, depth) in meters
    class_name: str
    confidence: float
    corners_3d: Optional[np.ndarray] = None  # 8x3 corner points
    bbox_2d: Optional[Tuple[float, float, float, float]] = None
    depth_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "center_3d": self.center.tolist(),
            "dimensions": self.dimensions.tolist(),
            "depth_meters": float(self.center[2]),
            "bbox_2d": list(self.bbox_2d) if self.bbox_2d else None,
            "corners_3d": self.corners_3d.tolist() if self.corners_3d is not None else None
        }
    
    def get_corners(self) -> np.ndarray:
        """Get the 8 corners of the 3D bounding box."""
        if self.corners_3d is not None:
            return self.corners_3d
            
        w, h, d = self.dimensions / 2
        cx, cy, cz = self.center
        
        corners = np.array([
            [cx - w, cy - h, cz - d],
            [cx + w, cy - h, cz - d],
            [cx + w, cy + h, cz - d],
            [cx - w, cy + h, cz - d],
            [cx - w, cy - h, cz + d],
            [cx + w, cy - h, cz + d],
            [cx + w, cy + h, cz + d],
            [cx - w, cy + h, cz + d],
        ])
        return corners


class Projector3D:
    """
    Stage C: Projects 2D detections to 3D space using depth information.
    
    FIXED: Added depth_scale parameter for calibration
    """
    
    # Approximate real-world sizes for common objects (width, height, depth) in meters
    OBJECT_SIZE_PRIORS = {
        # Furniture
        "chair": (0.5, 0.9, 0.5),
        "couch": (2.0, 0.9, 0.9),
        "sofa": (2.0, 0.9, 0.9),
        "table": (1.2, 0.75, 0.8),
        "dining table": (1.5, 0.75, 0.9),
        "desk": (1.2, 0.75, 0.6),
        "bed": (2.0, 0.6, 1.5),
        "bench": (1.5, 0.5, 0.4),
        
        # Electronics
        "tv": (1.2, 0.7, 0.1),
        "monitor": (0.5, 0.4, 0.15),
        "laptop": (0.35, 0.02, 0.25),
        "keyboard": (0.45, 0.03, 0.15),
        "mouse": (0.06, 0.03, 0.10),
        "cell phone": (0.07, 0.15, 0.01),
        "remote": (0.05, 0.18, 0.02),
        
        # Household items
        "lamp": (0.3, 0.5, 0.3),
        "clock": (0.3, 0.3, 0.05),
        "vase": (0.15, 0.3, 0.15),
        "potted plant": (0.4, 0.6, 0.4),
        "plant": (0.4, 0.6, 0.4),
        
        # Kitchen items
        "cup": (0.08, 0.10, 0.08),
        "mug": (0.08, 0.10, 0.08),
        "bottle": (0.08, 0.25, 0.08),
        "wine glass": (0.07, 0.20, 0.07),
        "bowl": (0.15, 0.08, 0.15),
        "plate": (0.25, 0.03, 0.25),
        "fork": (0.02, 0.18, 0.01),
        "knife": (0.02, 0.22, 0.01),
        "spoon": (0.03, 0.17, 0.01),
        
        # Books and office
        "book": (0.15, 0.22, 0.03),
        "backpack": (0.3, 0.45, 0.15),
        "handbag": (0.3, 0.25, 0.12),
        "suitcase": (0.45, 0.65, 0.25),
        
        # Structure
        "door": (0.9, 2.1, 0.1),
        "window": (1.0, 1.2, 0.1),
        "refrigerator": (0.7, 1.7, 0.7),
        "oven": (0.6, 0.9, 0.6),
        "microwave": (0.5, 0.3, 0.4),
        "sink": (0.6, 0.2, 0.5),
        "toilet": (0.4, 0.7, 0.65),
        
        # People and animals
        "person": (0.5, 1.7, 0.3),
        "cat": (0.3, 0.25, 0.4),
        "dog": (0.4, 0.5, 0.6),
        
        # Default
        "default": (0.5, 0.5, 0.5)
    }
    
    def __init__(
        self,
        camera: CameraIntrinsics,
        depth_scale: float = 1.0,
        depth_sampling_method: str = "percentile",
        use_size_priors: bool = True
    ):
        """
        Initialize the 3D projector.
        
        Args:
            camera: Camera intrinsic parameters
            depth_scale: Scale factor for depth correction (calibration)
            depth_sampling_method: How to sample depth ('median', 'mean', 'percentile')
            use_size_priors: Use prior knowledge of object sizes
        """
        self.camera = camera
        self.depth_scale = depth_scale
        self.depth_sampling_method = depth_sampling_method
        self.use_size_priors = use_size_priors
        
        print(f"[Stage C] Projector initialized with depth_scale={depth_scale}")
        
    def set_depth_scale(self, scale: float) -> None:
        """
        Set depth scale factor for calibration.
        
        If objects appear too far, decrease scale (e.g., 0.5)
        If objects appear too close, increase scale (e.g., 2.0)
        """
        self.depth_scale = scale
        print(f"[Stage C] Depth scale set to: {scale}")
    
    def project_detection(
        self,
        detection: Detection2D,
        depth_result: DepthResult
    ) -> BoundingBox3D:
        """
        Project a single 2D detection to 3D.
        
        Args:
            detection: 2D detection from Stage A
            depth_result: Depth estimation from Stage B
            
        Returns:
            3D bounding box with calibrated depth
        """
        x1, y1, x2, y2 = detection.bbox_xyxy
        cx, cy = detection.center
        
        # Get depth for this detection using configured method
        raw_depth = depth_result.get_depth_in_region(
            int(x1), int(y1), int(x2), int(y2),
            method=self.depth_sampling_method
        )
        
        # Apply depth scale calibration
        depth = raw_depth * self.depth_scale
        
        # Sanity check - clip to reasonable range
        depth = np.clip(depth, 0.3, 50.0)
        
        # Project center to 3D
        center_3d = self.camera.pixel_to_3d(cx, cy, depth)
        
        # Estimate 3D dimensions
        dimensions = self._estimate_dimensions(detection, depth)
        
        return BoundingBox3D(
            center=center_3d,
            dimensions=dimensions,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox_2d=detection.bbox_xyxy,
            depth_estimate=depth
        )
    
    def project_all_detections(
        self,
        detections: List[Detection2D],
        depth_result: DepthResult,
        debug: bool = False
    ) -> List[BoundingBox3D]:
        """
        Project all 2D detections to 3D.
        
        Args:
            detections: List of 2D detections
            depth_result: Depth estimation result
            debug: If True, print debug info for each detection
            
        Returns:
            List of 3D bounding boxes
        """
        boxes_3d = []
        
        for det in detections:
            box_3d = self.project_detection(det, depth_result)
            boxes_3d.append(box_3d)
            
            if debug:
                x1, y1, x2, y2 = det.bbox_xyxy
                raw_depth = depth_result.get_depth_in_region(
                    int(x1), int(y1), int(x2), int(y2),
                    method=self.depth_sampling_method
                )
                print(f"  {det.class_name}: raw_depth={raw_depth:.2f}m, "
                      f"scaled={box_3d.depth_estimate:.2f}m, "
                      f"position=({box_3d.center[0]:.2f}, {box_3d.center[1]:.2f}, {box_3d.center[2]:.2f})")
        
        return boxes_3d
    
    def _estimate_dimensions(
        self,
        detection: Detection2D,
        depth: float
    ) -> np.ndarray:
        """
        Estimate 3D dimensions of an object.
        
        Uses a combination of:
        1. Prior knowledge of typical object sizes
        2. 2D bbox dimensions projected to 3D
        """
        x1, y1, x2, y2 = detection.bbox_xyxy
        bbox_width_px = x2 - x1
        bbox_height_px = y2 - y1
        
        # Project 2D dimensions to 3D at the estimated depth
        width_3d = bbox_width_px * depth / self.camera.fx
        height_3d = bbox_height_px * depth / self.camera.fy
        
        # Get size prior if available
        class_lower = detection.class_name.lower()
        
        # Check for exact match or partial match
        prior = None
        if self.use_size_priors:
            if class_lower in self.OBJECT_SIZE_PRIORS:
                prior = self.OBJECT_SIZE_PRIORS[class_lower]
            else:
                # Try to find partial match
                for key in self.OBJECT_SIZE_PRIORS:
                    if key in class_lower or class_lower in key:
                        prior = self.OBJECT_SIZE_PRIORS[key]
                        break
        
        if prior is not None:
            # Blend projected size with prior
            # Trust projection more for width/height, prior for depth
            width_3d = 0.6 * width_3d + 0.4 * prior[0]
            height_3d = 0.6 * height_3d + 0.4 * prior[1]
            depth_3d = prior[2]  # Use prior for object depth
        else:
            # Estimate depth as fraction of min(width, height)
            depth_3d = min(width_3d, height_3d) * 0.5
            
        return np.array([width_3d, height_3d, depth_3d])
    
    def calibrate_from_known_distance(
        self,
        detection: Detection2D,
        depth_result: DepthResult,
        actual_distance: float
    ) -> float:
        """
        Calculate depth scale from a known distance measurement.
        
        Usage:
            1. Place an object at a known distance (e.g., 2 meters)
            2. Detect the object
            3. Call this function with the actual distance
            4. Use the returned scale factor
        
        Args:
            detection: Detection of the reference object
            depth_result: Depth estimation result
            actual_distance: Measured distance to object in meters
            
        Returns:
            Recommended depth scale factor
        """
        x1, y1, x2, y2 = detection.bbox_xyxy
        raw_depth = depth_result.get_depth_in_region(
            int(x1), int(y1), int(x2), int(y2),
            method=self.depth_sampling_method
        )
        
        scale = actual_distance / raw_depth
        
        print(f"[Calibration] Object: {detection.class_name}")
        print(f"[Calibration] Raw depth estimate: {raw_depth:.2f}m")
        print(f"[Calibration] Actual distance: {actual_distance:.2f}m")
        print(f"[Calibration] Recommended depth_scale: {scale:.3f}")
        
        return scale
    
    def generate_point_cloud_from_depth(
        self,
        depth_result: DepthResult,
        image: Optional[np.ndarray] = None,
        downsample: int = 4
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate a 3D point cloud from the depth map."""
        h, w = depth_result.depth_map.shape
        
        # Create pixel grid
        u = np.arange(0, w, downsample)
        v = np.arange(0, h, downsample)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        
        # Get depths and apply scale
        depths = depth_result.depth_map[v, u] * self.depth_scale
        
        # Filter invalid depths
        valid = (depths > 0.1) & (depths < 50.0)
        u, v, depths = u[valid], v[valid], depths[valid]
        
        # Back-project to 3D
        X = (u - self.camera.cx) * depths / self.camera.fx
        Y = (v - self.camera.cy) * depths / self.camera.fy
        Z = depths
        
        points = np.stack([X, Y, Z], axis=1)
        
        # Get colors if image provided
        colors = None
        if image is not None:
            colors = image[v, u] / 255.0
            
        return points, colors


def compute_3d_iou(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
    """
    Compute 3D Intersection over Union between two boxes.
    Assumes axis-aligned boxes (no rotation).
    """
    c1, d1 = box1.center, box1.dimensions / 2
    c2, d2 = box2.center, box2.dimensions / 2
    
    min1, max1 = c1 - d1, c1 + d1
    min2, max2 = c2 - d2, c2 + d2
    
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)
    
    vol1 = np.prod(box1.dimensions)
    vol2 = np.prod(box2.dimensions)
    union_vol = vol1 + vol2 - inter_vol
    
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol
