"""
Stage A: Open-Vocabulary 2D Object Detection
Uses YOLO-World for zero-shot object detection with text prompts.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import time


@dataclass
class Detection2D:
    """Represents a single 2D detection."""
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)
    area: float
    
    def to_dict(self) -> Dict:
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "center": list(self.center),
            "area": self.area
        }


class OpenVocabDetector:
    """
    Stage A: Open-Vocabulary 2D Object Detection using YOLO-World.
    
    Supports dynamic class queries without retraining.
    """
    
    SUPPORTED_MODELS = [
        "yolov8n-world",  # Nano - fastest
        "yolov8s-world",  # Small - balanced
        "yolov8m-world",  # Medium - more accurate
        "yolov8l-world",  # Large - highest accuracy
    ]
    
    def __init__(
        self,
        model_name: str = "yolov8s-world",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the open-vocabulary detector.
        
        Args:
            model_name: YOLO-World model variant
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda:0', etc.)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.current_classes: List[str] = []
        
        print(f"[Stage A] Loading {model_name}...")
        self.model = YOLO(f"{model_name}.pt")
        print(f"[Stage A] Model loaded successfully on {device}")
        
    def set_classes(self, classes: List[str]) -> None:
        """
        Set the target classes for open-vocabulary detection.
        
        Args:
            classes: List of class names to detect (e.g., ["chair", "table", "cup"])
        """
        self.current_classes = classes
        self.model.set_classes(classes)
        print(f"[Stage A] Classes set: {classes}")
        
    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        return_raw: bool = False
    ) -> Tuple[List[Detection2D], float]:
        """
        Run 2D object detection on an image.
        
        Args:
            image: Input image (RGB, numpy array)
            classes: Optional class list (overrides set_classes)
            return_raw: If True, also return raw YOLO results
            
        Returns:
            Tuple of (list of Detection2D objects, inference time in ms)
        """
        # Update classes if provided
        if classes is not None and classes != self.current_classes:
            self.set_classes(classes)
        
        if not self.current_classes:
            raise ValueError("No classes set. Call set_classes() first or provide classes parameter.")
        
        # Run inference
        start_time = time.perf_counter()
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = xyxy
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)
                
                detection = Detection2D(
                    class_name=result.names[cls_id],
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=center,
                    area=area
                )
                detections.append(detection)
        
        if return_raw:
            return detections, inference_time, results
        return detections, inference_time
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "current_classes": self.current_classes
        }


# Utility functions
def filter_detections_by_confidence(
    detections: List[Detection2D],
    min_confidence: float
) -> List[Detection2D]:
    """Filter detections by minimum confidence threshold."""
    return [d for d in detections if d.confidence >= min_confidence]


def filter_detections_by_class(
    detections: List[Detection2D],
    classes: List[str]
) -> List[Detection2D]:
    """Filter detections to include only specified classes."""
    return [d for d in detections if d.class_name in classes]


def get_largest_detection(
    detections: List[Detection2D],
    class_name: Optional[str] = None
) -> Optional[Detection2D]:
    """Get the detection with the largest bounding box area."""
    if class_name:
        detections = filter_detections_by_class(detections, [class_name])
    if not detections:
        return None
    return max(detections, key=lambda d: d.area)
