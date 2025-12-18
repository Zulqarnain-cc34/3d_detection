"""
SP1 3D Object Detection Pipeline
Integrates all three stages: Detection → Depth → 3D Projection
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
                f"conf={det.confidence:.2f}"
            )
        
        lines.extend([
            f"",
            f"Timing:",
            f"  Detection: {self.detection_time_ms:.1f}ms",
            f"  Depth: {self.depth_time_ms:.1f}ms",
            f"  Projection: {self.projection_time_ms:.1f}ms",
            f"  Total: {self.total_time_ms:.1f}ms",
            f"  FPS: {1000/self.total_time_ms:.1f}"
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
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        detector_model: str = "yolov8s-world",
        depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        camera: Optional[CameraIntrinsics] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        min_depth: float = 0.5,
        max_depth: float = 10.0
    ):
        """
        Initialize the SP1 pipeline.
        
        Args:
            config_path: Path to YAML config file (overrides other args)
            detector_model: YOLO-World model variant
            depth_model: Depth estimation model
            camera: Camera intrinsics (auto-detected if None)
            device: Device for inference ('cpu', 'cuda:0')
            confidence_threshold: Min confidence for detections
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
        """
        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            detector_model = config.get('detector', {}).get('model_name', detector_model)
            depth_model = config.get('depth_estimator', {}).get('model_name', depth_model)
            device = config.get('detector', {}).get('device', device)
            confidence_threshold = config.get('detector', {}).get('confidence_threshold', confidence_threshold)
            depth_range = config.get('depth_estimator', {}).get('depth_range', {})
            min_depth = depth_range.get('min_depth', min_depth)
            max_depth = depth_range.get('max_depth', max_depth)
        
        self.device = device
        self.camera = camera
        
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
        
        # Stage B: Depth estimator
        self.depth_estimator = MonocularDepthEstimator(
            model_name=depth_model,
            device=device
        )
        
        # Stage C: Projector (initialized per-image based on resolution)
        self.projector: Optional[Projector3D] = None
        
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
            camera = self.camera.scale_to_image(w, h)
        else:
            # Auto-detect camera intrinsics
            camera = CameraIntrinsics.from_fov(w, h, fov_horizontal_deg=60.0)
        
        self.projector = Projector3D(camera=camera)
    
    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        classes: List[str],
        return_visualization: bool = False
    ) -> PipelineResult:
        """
        Run the complete 3D detection pipeline.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            classes: List of object classes to detect
            return_visualization: If True, include visualization data
            
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
            detections_2d, depth_result
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
        # Run full detection with the target as a list
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
        """
        Warm up the pipeline with dummy inference.
        
        Useful for getting accurate timing measurements.
        """
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
        """
        Run timing benchmark on the pipeline.
        
        Args:
            image: Test image
            classes: Classes to detect
            num_runs: Number of inference runs
            
        Returns:
            Dict with timing statistics
        """
        times = {
            'detection': [],
            'depth': [],
            'projection': [],
            'total': []
        }
        
        # Warmup
        self.warmup()
        
        # Benchmark runs
        print(f"Running {num_runs} benchmark iterations...")
        for i in range(num_runs):
            result = self.detect(image, classes)
            times['detection'].append(result.detection_time_ms)
            times['depth'].append(result.depth_time_ms)
            times['projection'].append(result.projection_time_ms)
            times['total'].append(result.total_time_ms)
        
        # Compute statistics
        stats = {}
        for key, values in times.items():
            stats[key] = {
                'mean_ms': np.mean(values),
                'std_ms': np.std(values),
                'min_ms': np.min(values),
                'max_ms': np.max(values)
            }
        
        stats['fps'] = 1000 / stats['total']['mean_ms']
        
        return stats
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            "detector": self.detector.get_model_info(),
            "depth_estimator": self.depth_estimator.get_model_info(),
            "device": self.device
        }
