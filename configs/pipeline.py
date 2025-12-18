"""
SP1 3D Object Detection Pipeline
Integrates all three stages: Detection → Depth → 3D Projection

FIXED: Added depth_scale parameter for calibration and proper metric depth
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import time
import json
import yaml

from src.detector import OpenVocabDetector, Detection2D
from src.depth_estimator import MonocularDepthEstimator, DepthResult
from src.projector import Projector3D, BoundingBox3D, CameraIntrinsics


@dataclass
class PipelineResult:
    """Complete result from the 3D detection pipeline."""
    # Input info
    image_path: Optional[str]
    image_shape: Tuple[int, int, int]
    query_classes: List[str]
    
    # Stage outputs
    detections_2d: List[Detection2D]
    depth_result: DepthResult
    detections_3d: List[BoundingBox3D]
    
    # Timing
    detection_time_ms: float
    depth_time_ms: float
    projection_time_ms: float
    total_time_ms: float
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "image_shape": list(self.image_shape),
            "query_classes": self.query_classes,
            "num_detections": len(self.detections_3d),
            "detections_3d": [d.to_dict() for d in self.detections_3d],
            "timing": {
                "detection_ms": self.detection_time_ms,
                "depth_ms": self.depth_time_ms,
                "projection_ms": self.projection_time_ms,
                "total_ms": self.total_time_ms
            },
            "depth_stats": {
                "min_depth": self.depth_result.min_depth,
                "max_depth": self.depth_result.max_depth,
                "mean_depth": self.depth_result.mean_depth
            }
        }
    
    def save_json(self, path: str) -> None:
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"=== SP1 3D Detection Results ===",
            f"Image: {self.image_path or 'N/A'}",
            f"Shape: {self.image_shape}",
            f"Query: {self.query_classes}",
            f"",
            f"Detections ({len(self.detections_3d)} objects):",
        ]
        
        for i, det in enumerate(self.detections_3d):
            lines.append(
                f"  [{i+1}] {det.class_name}: "
                f"depth={det.center[2]:.2f}m, "
                f"pos=({det.center[0]:.2f}, {det.center[1]:.2f}, {det.center[2]:.2f}), "
                f"conf={det.confidence:.2f}"
            )
        
        lines.extend([
            f"",
            f"Depth Stats:",
            f"  Min: {self.depth_result.min_depth:.2f}m",
            f"  Max: {self.depth_result.max_depth:.2f}m",
            f"  Mean: {self.depth_result.mean_depth:.2f}m",
            f"",
            f"Timing:",
            f"  Detection: {self.detection_time_ms:.1f}ms",
            f"  Depth: {self.depth_time_ms:.1f}ms",
            f"  Projection: {self.projection_time_ms:.1f}ms",
            f"  Total: {self.total_time_ms:.1f}ms",
            f"  FPS: {1000/self.total_time_ms:.1f}" if self.total_time_ms > 0 else "  FPS: N/A"
        ])
        
        return "\n".join(lines)


class SP1Pipeline:
    """
    Sub-Project 1: Complete 3D Object Detection Pipeline
    
    RGB-Only pipeline for open-vocabulary 3D object detection.
    
    Pipeline stages:
        A. Open-Vocabulary 2D Detection (YOLO-World)
        B. Monocular Depth Estimation (Depth Anything V2)
        C. 3D Geometric Projection
    
    IMPORTANT - Depth Calibration:
        The depth_scale parameter is critical for accurate metric depth.
        If objects appear too far, decrease depth_scale.
        If objects appear too close, increase depth_scale.
        
        To calibrate:
            1. Place an object at known distance (e.g., 2 meters)
            2. Run detection
            3. Check reported depth
            4. Adjust depth_scale = actual_distance / reported_distance
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        detector_model: str = "yolov8s-world",
        depth_model: str = "indoor_small",  # Use indoor model by default
        camera: Optional[CameraIntrinsics] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        depth_scale: float = 1.0,  # NEW: Depth calibration scale
        max_depth: float = 20.0
    ):
        """
        Initialize the SP1 pipeline.
        
        Args:
            config_path: Path to YAML config file (overrides other args)
            detector_model: YOLO-World model variant
            depth_model: Depth model key ('indoor_small', 'indoor_base', 'outdoor_small')
            camera: Camera intrinsics (auto-detected if None)
            device: Device for inference ('cpu', 'cuda:0')
            confidence_threshold: Min confidence for detections
            depth_scale: Scale factor for depth calibration (adjust if depths are wrong)
            max_depth: Maximum depth to clip to (meters)
        """
        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            detector_model = config.get('detector', {}).get('model_name', detector_model)
            depth_model = config.get('depth_estimator', {}).get('model_name', depth_model)
            device = config.get('detector', {}).get('device', device)
            confidence_threshold = config.get('detector', {}).get('confidence_threshold', confidence_threshold)
            depth_scale = config.get('projection', {}).get('depth_scale', depth_scale)
        
        self.device = device
        self.camera = camera
        self.depth_scale = depth_scale
        
        # Initialize stages
        print("=" * 50)
        print("Initializing SP1 3D Detection Pipeline")
        print("=" * 50)
        
        # Stage A: Detector
        self.detector = OpenVocabDetector(
            model_name=detector_model,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Stage B: Depth estimator (NO normalization - use raw metric output)
        self.depth_estimator = MonocularDepthEstimator(
            model_name=depth_model,
            device=device,
            depth_scale=1.0,  # We apply scale in projector instead
            max_depth_clip=max_depth
        )
        
        # Stage C: Projector (initialized per-image based on resolution)
        self.projector: Optional[Projector3D] = None
        
        print(f"[Pipeline] Depth scale factor: {depth_scale}")
        print("=" * 50)
        print("Pipeline initialized successfully!")
        print("=" * 50)
    
    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _ensure_projector(self, image_shape: Tuple[int, ...]) -> None:
        """Initialize or update projector for image dimensions."""
        h, w = image_shape[:2]
        
        if self.camera is not None:
            camera = self.camera
            # Scale if dimensions don't match
            if camera.width and camera.width != w:
                scale_x = w / camera.width
                scale_y = h / camera.height if camera.height else scale_x
                camera = CameraIntrinsics(
                    fx=camera.fx * scale_x,
                    fy=camera.fy * scale_y,
                    cx=camera.cx * scale_x,
                    cy=camera.cy * scale_y,
                    width=w,
                    height=h
                )
        else:
            # Auto-detect camera intrinsics based on typical webcam FOV
            camera = CameraIntrinsics.default_for_video(w, h)
        
        self.projector = Projector3D(
            camera=camera,
            depth_scale=self.depth_scale,
            depth_sampling_method="percentile"  # Use percentile for better accuracy
        )
    
    def set_depth_scale(self, scale: float) -> None:
        """
        Set the depth scale factor for calibration.
        
        If objects at 5m are reported as 15m, set scale = 5/15 = 0.33
        If objects at 5m are reported as 2m, set scale = 5/2 = 2.5
        """
        self.depth_scale = scale
        if self.projector:
            self.projector.set_depth_scale(scale)
        print(f"[Pipeline] Depth scale set to: {scale}")
    
    def calibrate_depth(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_class: str,
        actual_distance: float
    ) -> float:
        """
        Calibrate depth scale using a known distance measurement.
        
        Args:
            image: Image containing the reference object
            target_class: Class name of the reference object
            actual_distance: Measured distance to the object in meters
            
        Returns:
            Recommended depth_scale value
            
        Usage:
            # Place a person at 3 meters from camera
            scale = pipeline.calibrate_depth(image, "person", 3.0)
            pipeline.set_depth_scale(scale)
        """
        # Temporarily set scale to 1.0 to get raw estimate
        old_scale = self.depth_scale
        self.depth_scale = 1.0
        if self.projector:
            self.projector.set_depth_scale(1.0)
        
        # Run detection
        result = self.detect(image, [target_class])
        
        # Restore old scale
        self.depth_scale = old_scale
        if self.projector:
            self.projector.set_depth_scale(old_scale)
        
        if not result.detections_3d:
            print(f"[Calibration] No {target_class} detected!")
            return 1.0
        
        # Get the most confident detection
        det = max(result.detections_3d, key=lambda d: d.confidence)
        estimated_depth = det.center[2]
        
        # Calculate scale
        scale = actual_distance / estimated_depth
        
        print(f"[Calibration] Object: {target_class}")
        print(f"[Calibration] Estimated depth (raw): {estimated_depth:.2f}m")
        print(f"[Calibration] Actual distance: {actual_distance:.2f}m")
        print(f"[Calibration] Recommended depth_scale: {scale:.3f}")
        print(f"[Calibration] Call pipeline.set_depth_scale({scale:.3f}) to apply")
        
        return scale
    
    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        classes: List[str],
        debug: bool = False
    ) -> PipelineResult:
        """
        Run the complete 3D detection pipeline.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            classes: List of object classes to detect
            debug: If True, print debug information for each detection
            
        Returns:
            PipelineResult with all detection information
        """
        # Load image
        image_path = None
        if isinstance(image, str):
            image_path = image
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Ensure projector is initialized
        self._ensure_projector(image.shape)
        
        # === Stage A: 2D Detection ===
        detections_2d, detection_time = self.detector.detect(image, classes)
        
        # === Stage B: Depth Estimation ===
        depth_result = self.depth_estimator.estimate_depth(image)
        depth_time = depth_result.inference_time_ms
        
        # === Stage C: 3D Projection ===
        proj_start = time.perf_counter()
        detections_3d = self.projector.project_all_detections(
            detections_2d, depth_result, debug=debug
        )
        projection_time = (time.perf_counter() - proj_start) * 1000
        
        total_time = detection_time + depth_time + projection_time
        
        return PipelineResult(
            image_path=image_path,
            image_shape=image.shape,
            query_classes=classes,
            detections_2d=detections_2d,
            depth_result=depth_result,
            detections_3d=detections_3d,
            detection_time_ms=detection_time,
            depth_time_ms=depth_time,
            projection_time_ms=projection_time,
            total_time_ms=total_time
        )
    
    def detect_single_object(
        self,
        image: Union[str, np.ndarray, Image.Image],
        object_query: str
    ) -> Optional[BoundingBox3D]:
        """
        Detect a single object by name and return its 3D location.
        
        Args:
            image: Input image
            object_query: Object to find (e.g., "red chair", "laptop")
            
        Returns:
            3D bounding box of the most confident detection, or None
        """
        result = self.detect(image, [object_query])
        
        if not result.detections_3d:
            return None
            
        # Return highest confidence detection
        return max(result.detections_3d, key=lambda d: d.confidence)

    def get_waypoint(
        self,
        image: Union[str, np.ndarray, Image.Image],
        target_object: str,
        offset_distance: float = 0.5
    ) -> Optional[Dict]:
        """
        Get a navigation waypoint to approach a target object.
        """
        result = self.detect(image, [target_object])
        
        if not result.detections_3d:
            return None
        
        # Get highest confidence detection
        detection = max(result.detections_3d, key=lambda d: d.confidence)
        
        # Calculate waypoint in front of object
        obj_pos = detection.center
        
        # Handle zero position case
        if np.linalg.norm(obj_pos) < 0.001:
            return None
            
        direction = obj_pos / np.linalg.norm(obj_pos)
        waypoint = obj_pos - direction * offset_distance
        
        return {
            "target_object": target_object,
            "object_position": obj_pos.tolist(),
            "waypoint_position": waypoint.tolist(),
            "distance_to_object": float(np.linalg.norm(obj_pos)),
            "confidence": detection.confidence
        }
    
    def warmup(self, image_size: Tuple[int, int] = (640, 480), iterations: int = 3) -> None:
        """Warm up the pipeline with dummy inference."""
        print(f"Warming up pipeline ({iterations} iterations)...")
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        for i in range(iterations):
            _ = self.detect(dummy_image, ["object"])
            print(f"  Warmup {i+1}/{iterations} complete")
        
        print("Warmup complete!")
    
    def benchmark(
        self,
        image: Union[str, np.ndarray],
        classes: List[str],
        num_runs: int = 10
    ) -> Dict:
        """Run timing benchmark on the pipeline."""
        times = {
            'detection': [],
            'depth': [],
            'projection': [],
            'total': []
        }
        
        for _ in range(num_runs):
            result = self.detect(image, classes)
            times['detection'].append(result.detection_time_ms)
            times['depth'].append(result.depth_time_ms)
            times['projection'].append(result.projection_time_ms)
            times['total'].append(result.total_time_ms)
        
        return {
            'detection_ms': {'mean': np.mean(times['detection']), 'std': np.std(times['detection'])},
            'depth_ms': {'mean': np.mean(times['depth']), 'std': np.std(times['depth'])},
            'projection_ms': {'mean': np.mean(times['projection']), 'std': np.std(times['projection'])},
            'total_ms': {'mean': np.mean(times['total']), 'std': np.std(times['total'])},
            'fps': 1000 / np.mean(times['total'])
        }
