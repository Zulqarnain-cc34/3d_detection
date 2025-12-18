"""
Stage C: 3D Geometric Projection
Converts 2D detections + depth estimates into 3D bounding boxes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from src.detector import Detection2D
from src.depth_estimator import DepthResult


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int  # Image width
    height: int  # Image height
    
    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        fov_horizontal_deg: float = 60.0
    ) -> "CameraIntrinsics":
        """
        Create intrinsics from field of view.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_horizontal_deg: Horizontal field of view in degrees
        """
        fov_rad = np.radians(fov_horizontal_deg)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    
    @classmethod
    def default_640x480(cls) -> "CameraIntrinsics":
        """Default intrinsics for 640x480 images."""
        return cls(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    
    def scale_to_image(self, new_width: int, new_height: int) -> "CameraIntrinsics":
        """Scale intrinsics to a different image size."""
        scale_x = new_width / self.width
        scale_y = new_height / self.height
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=new_width,
            height=new_height
        )
    
    def pixel_to_ray(self, x: float, y: float) -> np.ndarray:
        """
        Convert pixel coordinates to a unit ray direction.
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            Unit vector [x, y, z] in camera frame
        """
        ray = np.array([
            (x - self.cx) / self.fx,
            (y - self.cy) / self.fy,
            1.0
        ])
        return ray / np.linalg.norm(ray)
    
    def pixel_to_3d(self, x: float, y: float, depth: float) -> np.ndarray:
        """
        Back-project a pixel to 3D coordinates.
        
        Args:
            x, y: Pixel coordinates
            depth: Depth value in meters
            
        Returns:
            3D point [X, Y, Z] in camera frame (meters)
        """
        X = (x - self.cx) * depth / self.fx
        Y = (y - self.cy) * depth / self.fy
        Z = depth
        return np.array([X, Y, Z])


@dataclass
class BoundingBox3D:
    """Represents a 3D bounding box in camera coordinates."""
    center: np.ndarray  # [X, Y, Z] in meters
    dimensions: np.ndarray  # [width, height, depth] in meters
    class_name: str
    confidence: float
    corners_3d: Optional[np.ndarray] = None  # 8x3 array of corner points
    
    # Original 2D detection info
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
        """
        Get the 8 corners of the 3D bounding box.
        
        Returns:
            8x3 array of corner coordinates
        """
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
    """
    
    # Approximate real-world sizes for common objects (meters)
    OBJECT_SIZE_PRIORS = {
        "chair": (0.5, 0.9, 0.5),
        "sofa": (2.0, 0.9, 0.9),
        "table": (1.2, 0.75, 0.8),
        "desk": (1.2, 0.75, 0.6),
        "bed": (2.0, 0.6, 1.5),
        "tv": (1.0, 0.6, 0.1),
        "monitor": (0.5, 0.4, 0.1),
        "lamp": (0.3, 0.5, 0.3),
        "door": (0.9, 2.1, 0.1),
        "window": (1.0, 1.2, 0.1),
        "plant": (0.4, 0.6, 0.4),
        "pillow": (0.5, 0.1, 0.5),
        "cup": (0.08, 0.12, 0.08),
        "bottle": (0.08, 0.25, 0.08),
        "book": (0.15, 0.02, 0.22),
        "laptop": (0.35, 0.02, 0.25),
        "phone": (0.07, 0.15, 0.01),
        "remote": (0.05, 0.18, 0.02),
        "person": (0.5, 1.7, 0.3),
        "default": (0.5, 0.5, 0.5)
    }
    
    def __init__(
        self,
        camera: CameraIntrinsics,
        depth_sampling_method: str = "median",
        use_size_priors: bool = True
    ):
        """
        Initialize the 3D projector.
        
        Args:
            camera: Camera intrinsic parameters
            depth_sampling_method: How to sample depth in bbox region
            use_size_priors: Use prior knowledge of object sizes
        """
        self.camera = camera
        self.depth_sampling_method = depth_sampling_method
        self.use_size_priors = use_size_priors
        
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
            3D bounding box
        """
        x1, y1, x2, y2 = detection.bbox_xyxy
        cx, cy = detection.center
        
        # Get depth for this detection
        depth = depth_result.get_depth_in_region(
            int(x1), int(y1), int(x2), int(y2),
            method=self.depth_sampling_method
        )
        
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
        depth_result: DepthResult
    ) -> List[BoundingBox3D]:
        """
        Project all 2D detections to 3D.
        
        Args:
            detections: List of 2D detections
            depth_result: Depth estimation result
            
        Returns:
            List of 3D bounding boxes
        """
        boxes_3d = []
        for det in detections:
            box_3d = self.project_detection(det, depth_result)
            boxes_3d.append(box_3d)
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
        if self.use_size_priors and class_lower in self.OBJECT_SIZE_PRIORS:
            prior = self.OBJECT_SIZE_PRIORS[class_lower]
            # Blend projected size with prior (favor prior for depth dimension)
            width_3d = 0.7 * width_3d + 0.3 * prior[0]
            height_3d = 0.7 * height_3d + 0.3 * prior[1]
            depth_3d = prior[2]  # Use prior for depth (can't estimate from single view)
        else:
            # Estimate depth as average of width and height
            depth_3d = (width_3d + height_3d) / 2
            
        return np.array([width_3d, height_3d, depth_3d])
    
    def generate_point_cloud_from_depth(
        self,
        depth_result: DepthResult,
        image: Optional[np.ndarray] = None,
        downsample: int = 4
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate a 3D point cloud from the depth map.
        
        Args:
            depth_result: Depth estimation result
            image: Optional RGB image for coloring points
            downsample: Downsample factor to reduce point count
            
        Returns:
            Tuple of (points Nx3, colors Nx3 or None)
        """
        h, w = depth_result.depth_map.shape
        
        # Create pixel grid
        u = np.arange(0, w, downsample)
        v = np.arange(0, h, downsample)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        
        # Get depths
        depths = depth_result.depth_map[v, u]
        
        # Filter invalid depths
        valid = depths > 0
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
    # Get min/max corners
    c1, d1 = box1.center, box1.dimensions / 2
    c2, d2 = box2.center, box2.dimensions / 2
    
    min1, max1 = c1 - d1, c1 + d1
    min2, max2 = c2 - d2, c2 + d2
    
    # Compute intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)
    
    # Compute union
    vol1 = np.prod(box1.dimensions)
    vol2 = np.prod(box2.dimensions)
    union_vol = vol1 + vol2 - inter_vol
    
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol
