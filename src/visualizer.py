"""
Visualization utilities for SP1 3D Detection Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from typing import List, Optional, Tuple, Dict
from PIL import Image

from src.detector import Detection2D
from src.depth_estimator import DepthResult, DepthMapVisualizer
from src.projector import BoundingBox3D


class PipelineVisualizer:
    """Visualization tools for the SP1 pipeline."""
    
    COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
        self.class_colors: Dict[str, str] = {}
        
    def _get_color(self, class_name: str) -> str:
        if class_name not in self.class_colors:
            idx = len(self.class_colors) % len(self.COLORS)
            self.class_colors[class_name] = self.COLORS[idx]
        return self.class_colors[class_name]
    
    def visualize_pipeline_result(
        self,
        image: np.ndarray,
        detections_2d: List[Detection2D],
        depth_result: DepthResult,
        detections_3d: List[BoundingBox3D],
        title: str = "SP1 3D Detection Pipeline Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comprehensive visualization of pipeline results."""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Original Image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(image)
        ax1.set_title("Input RGB Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. 2D Detections
        ax2 = fig.add_subplot(2, 3, 2)
        self._draw_2d_detections(ax2, image, detections_2d)
        ax2.set_title("Stage A: Open-Vocab 2D Detection", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Depth Map
        ax3 = fig.add_subplot(2, 3, 3)
        depth_vis = DepthMapVisualizer.to_colormap(depth_result.depth_map, 'plasma')
        ax3.imshow(depth_vis)
        ax3.set_title("Stage B: Monocular Depth Estimation", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        sm = plt.cm.ScalarMappable(
            cmap='plasma',
            norm=plt.Normalize(depth_result.min_depth, depth_result.max_depth)
        )
        cbar = fig.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (m)', fontsize=10)
        
        # 4. Bird's Eye View
        ax4 = fig.add_subplot(2, 3, 4)
        self._draw_birds_eye_view(ax4, detections_3d)
        ax4.set_title("Stage C: 3D Projection (Bird's Eye)", fontsize=12, fontweight='bold')
        
        # 5. Side View
        ax5 = fig.add_subplot(2, 3, 5)
        self._draw_side_view(ax5, detections_3d)
        ax5.set_title("Stage C: 3D Projection (Side View)", fontsize=12, fontweight='bold')
        
        # 6. Detection Summary
        ax6 = fig.add_subplot(2, 3, 6)
        self._draw_summary_table(ax6, detections_3d, depth_result)
        ax6.set_title("Detection Summary", fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig
    
    def _draw_2d_detections(self, ax: plt.Axes, image: np.ndarray, detections: List[Detection2D]) -> None:
        ax.imshow(image)
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = self._get_color(det.class_name)
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            label = f"{det.class_name}: {det.confidence:.2f}"
            ax.text(x1, y1 - 5, label, fontsize=9, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    
    def _draw_birds_eye_view(self, ax: plt.Axes, detections: List[BoundingBox3D]) -> None:
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m) - Depth', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.plot(0, 0, 'k^', markersize=12, label='Camera')
        ax.annotate('Camera', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=9)
        
        if detections:
            max_depth = max([d.center[2] for d in detections])
            fov_rad = np.radians(30)
            ax.plot([0, -max_depth * np.tan(fov_rad)], [0, max_depth], 'k--', alpha=0.3, linewidth=1)
            ax.plot([0, max_depth * np.tan(fov_rad)], [0, max_depth], 'k--', alpha=0.3, linewidth=1)
        
        for det in detections:
            x, _, z = det.center
            w, _, d = det.dimensions
            color = self._get_color(det.class_name)
            rect = Rectangle((x - w/2, z - d/2), w, d, linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            ax.plot(x, z, 'o', color=color, markersize=8)
            ax.annotate(det.class_name, (x, z), textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 12)
    
    def _draw_side_view(self, ax: plt.Axes, detections: List[BoundingBox3D]) -> None:
        ax.set_xlabel('Z (m) - Depth', fontsize=10)
        ax.set_ylabel('Y (m) - Height', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='brown', linewidth=2, label='Ground')
        ax.plot(0, 0, 'k^', markersize=12)
        ax.annotate('Camera', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=9)
        
        for det in detections:
            _, y, z = det.center
            _, h, d = det.dimensions
            color = self._get_color(det.class_name)
            rect = Rectangle((z - d/2, -y - h/2), d, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            ax.plot(z, -y, 'o', color=color, markersize=8)
            ax.annotate(det.class_name, (z, -y), textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)
        
        ax.set_xlim(0, 12)
        ax.set_ylim(-3, 3)
    
    def _draw_summary_table(self, ax: plt.Axes, detections: List[BoundingBox3D], depth_result: DepthResult) -> None:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        y_pos = 9.5
        ax.text(0.5, y_pos, "Object", fontsize=10, fontweight='bold')
        ax.text(3.5, y_pos, "Confidence", fontsize=10, fontweight='bold')
        ax.text(6.0, y_pos, "Depth (m)", fontsize=10, fontweight='bold')
        ax.text(8.0, y_pos, "Position", fontsize=10, fontweight='bold')
        
        y_pos -= 0.8
        ax.axhline(y=y_pos + 0.3, xmin=0.05, xmax=0.95, color='black', linewidth=1)
        
        for det in detections[:8]:
            color = self._get_color(det.class_name)
            ax.text(0.5, y_pos, det.class_name, fontsize=9, color=color)
            ax.text(3.5, y_pos, f"{det.confidence:.2f}", fontsize=9)
            ax.text(6.0, y_pos, f"{det.center[2]:.2f}", fontsize=9)
            ax.text(8.0, y_pos, f"({det.center[0]:.1f}, {det.center[1]:.1f})", fontsize=8)
            y_pos -= 0.7
        
        y_pos -= 0.5
        ax.axhline(y=y_pos + 0.3, xmin=0.05, xmax=0.95, color='gray', linewidth=0.5)
        ax.text(0.5, y_pos, f"Total Objects: {len(detections)}", fontsize=9, style='italic')
        ax.text(0.5, y_pos - 0.6, f"Depth Range: {depth_result.min_depth:.1f}m - {depth_result.max_depth:.1f}m", fontsize=9, style='italic')
    
    def create_3d_scene_plot(
        self,
        detections: List[BoundingBox3D],
        point_cloud: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create interactive 3D visualization."""
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter([0], [0], [0], c='black', marker='^', s=200, label='Camera')
        
        if point_cloud is not None:
            points, colors = point_cloud
            if colors is not None:
                ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=colors, s=1, alpha=0.3)
            else:
                ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c='gray', s=1, alpha=0.3)
        
        for det in detections:
            corners = det.get_corners()
            corners_plot = corners.copy()
            corners_plot[:, [1, 2]] = corners_plot[:, [2, 1]]
            corners_plot[:, 2] = -corners_plot[:, 2]
            color = self._get_color(det.class_name)
            
            faces = [
                [corners_plot[0], corners_plot[1], corners_plot[2], corners_plot[3]],
                [corners_plot[4], corners_plot[5], corners_plot[6], corners_plot[7]],
                [corners_plot[0], corners_plot[1], corners_plot[5], corners_plot[4]],
                [corners_plot[2], corners_plot[3], corners_plot[7], corners_plot[6]],
                [corners_plot[0], corners_plot[3], corners_plot[7], corners_plot[4]],
                [corners_plot[1], corners_plot[2], corners_plot[6], corners_plot[5]]
            ]
            
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor=color, linewidth=1))
            center_plot = det.center.copy()
            center_plot[[1, 2]] = center_plot[[2, 1]]
            center_plot[2] = -center_plot[2]
            ax.text(center_plot[0], center_plot[1], center_plot[2], det.class_name, fontsize=9, color=color)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m) - Depth')
        ax.set_zlabel('Y (m) - Height')
        ax.set_title('3D Scene Reconstruction', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def quick_visualize(result, image: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
    """Quick visualization helper function."""
    viz = PipelineVisualizer()
    return viz.visualize_pipeline_result(
        image=image,
        detections_2d=result.detections_2d,
        depth_result=result.depth_result,
        detections_3d=result.detections_3d,
        save_path=save_path
    )
