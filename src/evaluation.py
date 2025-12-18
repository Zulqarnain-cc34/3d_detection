"""
Evaluation metrics for SP1 3D Detection Pipeline

Provides quantitative assessment of:
- 2D Detection accuracy
- Depth estimation accuracy  
- 3D localization accuracy
- Runtime performance
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from src.projector import BoundingBox3D, compute_3d_iou
from src.detector import Detection2D


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    # Detection metrics
    num_predictions: int
    num_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    
    # Depth metrics
    depth_mae: float  # Mean Absolute Error
    depth_rmse: float  # Root Mean Square Error
    depth_abs_rel: float  # Absolute Relative Error
    
    # 3D metrics
    mean_3d_iou: float
    localization_error_m: float  # Mean 3D distance error
    
    # Performance metrics
    mean_inference_time_ms: float
    fps: float
    
    def to_dict(self) -> Dict:
        return {
            "detection": {
                "num_predictions": self.num_predictions,
                "num_ground_truth": self.num_ground_truth,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score
            },
            "depth": {
                "mae": self.depth_mae,
                "rmse": self.depth_rmse,
                "abs_rel": self.depth_abs_rel
            },
            "3d_localization": {
                "mean_iou": self.mean_3d_iou,
                "mean_error_m": self.localization_error_m
            },
            "performance": {
                "mean_inference_ms": self.mean_inference_time_ms,
                "fps": self.fps
            }
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "SP1 PIPELINE EVALUATION RESULTS",
            "=" * 50,
            "",
            "DETECTION METRICS:",
            f"  Predictions: {self.num_predictions}",
            f"  Ground Truth: {self.num_ground_truth}",
            f"  True Positives: {self.true_positives}",
            f"  Precision: {self.precision:.3f}",
            f"  Recall: {self.recall:.3f}",
            f"  F1 Score: {self.f1_score:.3f}",
            "",
            "DEPTH METRICS:",
            f"  MAE: {self.depth_mae:.3f} m",
            f"  RMSE: {self.depth_rmse:.3f} m",
            f"  Abs Relative: {self.depth_abs_rel:.3f}",
            "",
            "3D LOCALIZATION:",
            f"  Mean 3D IoU: {self.mean_3d_iou:.3f}",
            f"  Mean Error: {self.localization_error_m:.3f} m",
            "",
            "PERFORMANCE:",
            f"  Inference Time: {self.mean_inference_time_ms:.1f} ms",
            f"  FPS: {self.fps:.1f}",
            "=" * 50
        ]
        return "\n".join(lines)


class PipelineEvaluator:
    """
    Evaluator for the SP1 3D Detection Pipeline.
    
    Supports evaluation against ground truth data for:
    - Standard benchmarks (ScanNet, SUN RGB-D)
    - Custom annotated datasets
    """
    
    def __init__(
        self,
        iou_threshold_2d: float = 0.5,
        iou_threshold_3d: float = 0.25,
        depth_threshold_m: float = 0.5
    ):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold_2d: IoU threshold for 2D detection matching
            iou_threshold_3d: IoU threshold for 3D detection matching
            depth_threshold_m: Depth error threshold for correct match
        """
        self.iou_threshold_2d = iou_threshold_2d
        self.iou_threshold_3d = iou_threshold_3d
        self.depth_threshold_m = depth_threshold_m
        
        # Accumulated metrics
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_predictions = []
        self.all_ground_truth = []
        self.depth_errors = []
        self.iou_scores = []
        self.localization_errors = []
        self.inference_times = []
    
    def add_sample(
        self,
        predictions: List[BoundingBox3D],
        ground_truth: List[Dict],
        depth_pred: np.ndarray,
        depth_gt: Optional[np.ndarray] = None,
        inference_time_ms: float = 0
    ):
        """
        Add evaluation sample.
        
        Args:
            predictions: Predicted 3D bounding boxes
            ground_truth: Ground truth annotations (list of dicts with 'class', 'center_3d', 'dimensions')
            depth_pred: Predicted depth map
            depth_gt: Ground truth depth map (optional)
            inference_time_ms: Inference time for this sample
        """
        self.all_predictions.append(predictions)
        self.all_ground_truth.append(ground_truth)
        self.inference_times.append(inference_time_ms)
        
        # Compute depth metrics if GT available
        if depth_gt is not None:
            valid_mask = depth_gt > 0
            if valid_mask.sum() > 0:
                pred_valid = depth_pred[valid_mask]
                gt_valid = depth_gt[valid_mask]
                
                mae = np.mean(np.abs(pred_valid - gt_valid))
                rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
                abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
                
                self.depth_errors.append({
                    'mae': mae,
                    'rmse': rmse,
                    'abs_rel': abs_rel
                })
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            best_loc_error = float('inf')
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                if gt.get('class', '').lower() != pred.class_name.lower():
                    continue
                
                # Create GT bounding box
                gt_box = BoundingBox3D(
                    center=np.array(gt['center_3d']),
                    dimensions=np.array(gt.get('dimensions', [0.5, 0.5, 0.5])),
                    class_name=gt['class'],
                    confidence=1.0
                )
                
                iou = compute_3d_iou(pred, gt_box)
                loc_error = np.linalg.norm(pred.center - gt_box.center)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_loc_error = loc_error
            
            if best_iou > self.iou_threshold_3d:
                matched_gt.add(best_gt_idx)
                self.iou_scores.append(best_iou)
                self.localization_errors.append(best_loc_error)
    
    def compute_metrics(self) -> EvaluationResult:
        """Compute final evaluation metrics."""
        total_pred = sum(len(p) for p in self.all_predictions)
        total_gt = sum(len(g) for g in self.all_ground_truth)
        tp = len(self.iou_scores)
        fp = total_pred - tp
        fn = total_gt - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Depth metrics
        if self.depth_errors:
            depth_mae = np.mean([e['mae'] for e in self.depth_errors])
            depth_rmse = np.mean([e['rmse'] for e in self.depth_errors])
            depth_abs_rel = np.mean([e['abs_rel'] for e in self.depth_errors])
        else:
            depth_mae = depth_rmse = depth_abs_rel = 0
        
        # 3D metrics
        mean_iou = np.mean(self.iou_scores) if self.iou_scores else 0
        mean_loc_error = np.mean(self.localization_errors) if self.localization_errors else 0
        
        # Performance
        mean_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1000 / mean_time if mean_time > 0 else 0
        
        return EvaluationResult(
            num_predictions=total_pred,
            num_ground_truth=total_gt,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            depth_mae=depth_mae,
            depth_rmse=depth_rmse,
            depth_abs_rel=depth_abs_rel,
            mean_3d_iou=mean_iou,
            localization_error_m=mean_loc_error,
            mean_inference_time_ms=mean_time,
            fps=fps
        )


def compute_2d_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute 2D IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def create_synthetic_ground_truth(detections: List[BoundingBox3D], noise_level: float = 0.1) -> List[Dict]:
    """
    Create synthetic ground truth from detections (for testing).
    
    Useful when you don't have real GT but want to test the evaluation pipeline.
    """
    gt = []
    for det in detections:
        noise = np.random.randn(3) * noise_level
        gt.append({
            'class': det.class_name,
            'center_3d': (det.center + noise).tolist(),
            'dimensions': det.dimensions.tolist()
        })
    return gt


class DepthEvaluator:
    """Standalone evaluator for depth estimation quality."""
    
    @staticmethod
    def evaluate_depth_map(
        pred: np.ndarray,
        gt: np.ndarray,
        min_depth: float = 0.5,
        max_depth: float = 10.0
    ) -> Dict:
        """
        Evaluate depth map against ground truth.
        
        Standard metrics from depth estimation literature.
        """
        valid_mask = (gt > min_depth) & (gt < max_depth)
        
        if valid_mask.sum() == 0:
            return {'error': 'No valid pixels'}
        
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        # Compute metrics
        thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
        
        metrics = {
            'mae': float(np.mean(np.abs(pred_valid - gt_valid))),
            'rmse': float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2))),
            'abs_rel': float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid)),
            'sq_rel': float(np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)),
            'log_rmse': float(np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2))),
            'delta_1': float(np.mean(thresh < 1.25)),
            'delta_2': float(np.mean(thresh < 1.25 ** 2)),
            'delta_3': float(np.mean(thresh < 1.25 ** 3)),
        }
        
        return metrics
