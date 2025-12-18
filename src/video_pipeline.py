"""
SP1 Video Pipeline - Real-time 3D Object Detection for Video
============================================================

Extends the SP1 image pipeline to support:
- Video file processing (.mp4, .avi, .mov, etc.)
- Real-time webcam/camera streaming
- Video output with 3D detection overlays
- Frame-by-frame generator for custom processing

Author: SP1 Team
Target: Jetson Orin 8GB / Desktop GPU

Usage:
    from src.pipeline import SP1Pipeline
    from src.video_pipeline import SP1VideoPipeline, VideoConfig
    
    # IMPORTANT: Create an INSTANCE of SP1Pipeline, not import the module
    pipeline = SP1Pipeline(device='cuda:0')  # This creates an instance
    
    config = VideoConfig(detection_classes=['person', 'chair', 'table'])
    video_pipeline = SP1VideoPipeline(pipeline, config)
    
    video_pipeline.process_video('input.mp4', 'output.mp4')
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable, Generator, Union
from pathlib import Path
from collections import deque


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    # Processing settings
    process_every_n_frames: int = 1  # Process every Nth frame (1 = all frames)
    detection_classes: List[str] = field(default_factory=lambda: [
        "chair", "table", "couch", "bed", "desk", "lamp",
        "tv", "laptop", "bottle", "cup", "book", "plant", "person"
    ])
    confidence_threshold: float = 0.25
    
    # Display settings
    show_depth_minimap: bool = True
    show_3d_overlay: bool = True
    show_fps: bool = True
    show_object_panel: bool = True
    depth_colormap: str = "plasma"  # plasma, viridis, magma, inferno, jet
    
    # Output settings
    output_fps: float = 30.0
    
    # Performance
    warmup_frames: int = 3


@dataclass 
class FrameResult:
    """Result for a single processed video frame."""
    frame_number: int
    timestamp_ms: float
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    depth_colormap: Optional[np.ndarray]
    detections_2d: List
    detections_3d: List
    detection_time_ms: float
    depth_time_ms: float
    projection_time_ms: float
    total_time_ms: float
    current_fps: float


class VideoOverlayRenderer:
    """Renders detection overlays on video frames using OpenCV."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.colors = {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent BGR color for object class."""
        if class_name not in self.colors:
            np.random.seed(hash(class_name) % 2**32)
            self.colors[class_name] = tuple(int(c) for c in np.random.randint(80, 255, 3))
        return self.colors[class_name]
    
    def render(
        self,
        frame: np.ndarray,
        detections_2d: List,
        detections_3d: List,
        depth_map: np.ndarray,
        fps: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render all overlays on a frame.
        
        Args:
            frame: BGR frame from OpenCV
            detections_2d: List of Detection2D objects
            detections_3d: List of BoundingBox3D objects
            depth_map: Depth map array
            fps: Current FPS for display
            
        Returns:
            Tuple of (annotated_frame, depth_colormap)
        """
        output = frame.copy()
        
        # Create depth colormap
        depth_vis = self._create_depth_colormap(depth_map)
        
        # Draw 3D detections on frame
        if self.config.show_3d_overlay and detections_3d:
            output = self._draw_detections(output, detections_2d, detections_3d)
        
        # Draw depth minimap in corner
        if self.config.show_depth_minimap:
            output = self._draw_depth_minimap(output, depth_vis)
        
        # Draw FPS
        if self.config.show_fps:
            output = self._draw_fps(output, fps)
        
        # Draw object info panel
        if self.config.show_object_panel and detections_3d:
            output = self._draw_object_panel(output, detections_3d)
        
        return output, depth_vis
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections_2d: List,
        detections_3d: List
    ) -> np.ndarray:
        """Draw bounding boxes with 3D information."""
        for det_2d, det_3d in zip(detections_2d, detections_3d):
            color = self._get_color(det_3d.class_name)
            x1, y1, x2, y2 = map(int, det_2d.bbox_xyxy)
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label with depth
            label = f"{det_3d.class_name} {det_3d.center[2]:.2f}m"
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), self.font, 0.6, (255, 255, 255), 2)
            
            # 3D position at bottom
            pos_label = f"X:{det_3d.center[0]:.1f} Y:{det_3d.center[1]:.1f}"
            cv2.putText(frame, pos_label, (x1 + 2, y2 - 4), self.font, 0.4, color, 1)
            
            # Depth indicator bar (green=close, red=far)
            depth_norm = np.clip((det_3d.center[2] - 0.5) / 9.5, 0, 1)
            bar_color = (0, int(255 * (1 - depth_norm)), int(255 * depth_norm))
            bar_w = int((x2 - x1) * (1 - depth_norm))
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 6), bar_color, -1)
        
        return frame
    
    def _create_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to colormap visualization."""
        # Normalize to 0-255
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_norm = ((depth_map - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        cmap_dict = {
            "plasma": cv2.COLORMAP_PLASMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
        }
        cmap = cmap_dict.get(self.config.depth_colormap, cv2.COLORMAP_PLASMA)
        return cv2.applyColorMap(depth_norm, cmap)
    
    def _draw_depth_minimap(self, frame: np.ndarray, depth_vis: np.ndarray) -> np.ndarray:
        """Draw depth minimap in top-right corner."""
        h, w = frame.shape[:2]
        mini_h, mini_w = h // 4, w // 4
        
        mini_depth = cv2.resize(depth_vis, (mini_w, mini_h))
        
        # Position with padding
        pad = 10
        x1, y1 = w - mini_w - pad, pad
        
        # Border
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x1 + mini_w + 2, y1 + mini_h + 2), (255, 255, 255), 2)
        
        # Overlay
        frame[y1:y1 + mini_h, x1:x1 + mini_w] = mini_depth
        
        # Label
        cv2.putText(frame, "Depth", (x1, y1 - 5), self.font, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter."""
        text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.7, 2)
        cv2.rectangle(frame, (8, 8), (16 + tw, 16 + th), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, 12 + th), self.font, 0.7, (0, 255, 0), 2)
        return frame
    
    def _draw_object_panel(self, frame: np.ndarray, detections_3d: List) -> np.ndarray:
        """Draw object list panel in bottom-left."""
        h, w = frame.shape[:2]
        
        num_objects = min(len(detections_3d), 6)
        panel_h = 30 + num_objects * 22
        panel_w = 180
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, h - panel_h - 8), (8 + panel_w, h - 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, f"Objects ({len(detections_3d)})", (14, h - panel_h + 16), 
                    self.font, 0.5, (255, 255, 255), 1)
        
        # List objects
        y = h - panel_h + 38
        for det in detections_3d[:num_objects]:
            color = self._get_color(det.class_name)
            text = f"{det.class_name}: {det.center[2]:.1f}m"
            cv2.putText(frame, text, (16, y), self.font, 0.42, color, 1)
            y += 22
        
        return frame


class SP1VideoPipeline:
    """
    Video processing extension for SP1 3D Object Detection Pipeline.
    
    IMPORTANT: You must pass an INSTANCE of SP1Pipeline, not the module!
    
    Correct usage:
        from src.pipeline import SP1Pipeline
        from src.video_pipeline import SP1VideoPipeline, VideoConfig
        
        pipeline = SP1Pipeline(device='cuda:0')  # Create instance!
        video_pipeline = SP1VideoPipeline(pipeline, config)
    
    Wrong usage:
        from src import pipeline  # This imports the MODULE
        video_pipeline = SP1VideoPipeline(pipeline, config)  # ERROR!
    """
    
    def __init__(self, pipeline, config: Optional[VideoConfig] = None):
        """
        Initialize the video pipeline.
        
        Args:
            pipeline: An INSTANCE of SP1Pipeline (not the module!)
            config: Video processing configuration
        """
        # Validate that pipeline is an instance, not a module
        if not hasattr(pipeline, 'detect'):
            raise TypeError(
                "ERROR: 'pipeline' must be an INSTANCE of SP1Pipeline, not the module!\n\n"
                "Correct usage:\n"
                "    from src.pipeline import SP1Pipeline\n"
                "    pipeline = SP1Pipeline(device='cuda:0')  # Create instance\n"
                "    video_pipeline = SP1VideoPipeline(pipeline, config)\n\n"
                "Wrong usage:\n"
                "    from src import pipeline  # This imports the MODULE\n"
                "    video_pipeline = SP1VideoPipeline(pipeline, config)  # This fails!"
            )
        
        self.pipeline = pipeline
        self.config = config or VideoConfig()
        self.renderer = VideoOverlayRenderer(self.config)
        
        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.is_running = False
        
        print("[SP1 Video] Pipeline initialized")
    
    def _open_source(self, source: Union[str, int]) -> Tuple[bool, Dict]:
        """Open video source and return properties."""
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"[SP1 Video] ERROR: Cannot open source: {source}")
            return False, {}
        
        props = {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "is_camera": isinstance(source, int)
        }
        
        print(f"[SP1 Video] Source opened: {props['width']}x{props['height']} @ {props['fps']:.1f} FPS")
        if props['total_frames'] > 0:
            duration = props['total_frames'] / max(props['fps'], 1)
            print(f"[SP1 Video] Duration: {duration:.1f}s ({props['total_frames']} frames)")
        
        return True, props
    
    def _setup_writer(self, output_path: str, width: int, height: int) -> None:
        """Setup video writer."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.config.output_fps, (width, height))
        print(f"[SP1 Video] Output: {output_path}")
    
    def _get_fps(self) -> float:
        """Get smoothed FPS."""
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0
    
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame through the SP1 pipeline.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            FrameResult with all detection information
        """
        start = time.perf_counter()
        
        # Convert BGR -> RGB for pipeline
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run SP1 pipeline - call detect() method on the pipeline INSTANCE
        result = self.pipeline.detect(frame_rgb, self.config.detection_classes)
        
        # Render overlays (keep BGR for OpenCV)
        annotated, depth_vis = self.renderer.render(
            frame,
            result.detections_2d,
            result.detections_3d,
            result.depth_result.depth_map,
            self._get_fps()
        )
        
        total_time = (time.perf_counter() - start) * 1000
        self.fps_history.append(1000 / total_time if total_time > 0 else 0)
        
        return FrameResult(
            frame_number=self.frame_count,
            timestamp_ms=self.cap.get(cv2.CAP_PROP_POS_MSEC) if self.cap else 0,
            original_frame=frame,
            annotated_frame=annotated,
            depth_colormap=depth_vis,
            detections_2d=result.detections_2d,
            detections_3d=result.detections_3d,
            detection_time_ms=result.detection_time_ms,
            depth_time_ms=result.depth_time_ms,
            projection_time_ms=result.projection_time_ms,
            total_time_ms=total_time,
            current_fps=self._get_fps()
        )
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        display: bool = False,
        callback: Optional[Callable[[FrameResult], bool]] = None
    ) -> Dict:
        """
        Process a video file and save annotated output.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            max_frames: Maximum frames to process (None = all)
            display: Show live preview window
            callback: Optional callback(FrameResult) -> bool, return False to stop
            
        Returns:
            Dict with processing statistics
        """
        success, props = self._open_source(input_path)
        if not success:
            return {"error": "Failed to open video"}
        
        self._setup_writer(output_path, props["width"], props["height"])
        
        return self._process_loop(
            max_frames=max_frames,
            display=display,
            callback=callback,
            total_frames=props["total_frames"]
        )
    
    def run_webcam(
        self,
        camera_id: int = 0,
        output_path: Optional[str] = None,
        callback: Optional[Callable[[FrameResult], bool]] = None
    ) -> Dict:
        """
        Run real-time detection on webcam.
        
        Args:
            camera_id: Camera device ID (default 0)
            output_path: Optional path to save output video
            callback: Optional callback function
            
        Returns:
            Dict with processing statistics
        """
        success, props = self._open_source(camera_id)
        if not success:
            return {"error": "Failed to open camera"}
        
        if output_path:
            self._setup_writer(output_path, props["width"], props["height"])
        
        return self._process_loop(display=True, callback=callback)
    
    def _process_loop(
        self,
        max_frames: Optional[int] = None,
        display: bool = True,
        callback: Optional[Callable[[FrameResult], bool]] = None,
        total_frames: int = 0
    ) -> Dict:
        """Main processing loop."""
        self.frame_count = 0
        self.fps_history.clear()
        self.is_running = True
        
        stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "total_detections": 0,
            "detection_times": [],
            "depth_times": [],
            "total_times": []
        }
        
        print("\n[SP1 Video] Starting... (Press 'q' to quit, 's' to screenshot, 'p' to pause)")
        
        # Warmup
        print("[SP1 Video] Warming up...")
        warmup_count = 0
        while warmup_count < self.config.warmup_frames:
            ret, frame = self.cap.read()
            if ret:
                _ = self.process_frame(frame)
                warmup_count += 1
            else:
                break
        
        # Reset for actual processing
        if total_frames > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        self.fps_history.clear()
        
        print("[SP1 Video] Processing...\n")
        paused = False
        result = None
        
        try:
            while self.is_running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("\n[SP1 Video] End of video")
                        break
                    
                    self.frame_count += 1
                    
                    # Check max frames
                    if max_frames and self.frame_count > max_frames:
                        break
                    
                    # Skip frames if configured
                    if self.frame_count % self.config.process_every_n_frames != 0:
                        stats["frames_skipped"] += 1
                        continue
                    
                    # Process frame
                    result = self.process_frame(frame)
                    
                    # Update stats
                    stats["frames_processed"] += 1
                    stats["total_detections"] += len(result.detections_3d)
                    stats["detection_times"].append(result.detection_time_ms)
                    stats["depth_times"].append(result.depth_time_ms)
                    stats["total_times"].append(result.total_time_ms)
                    
                    # Write output
                    if self.writer:
                        self.writer.write(result.annotated_frame)
                    
                    # Callback
                    if callback and not callback(result):
                        break
                    
                    # Progress
                    if stats["frames_processed"] % 30 == 0:
                        progress = ""
                        if total_frames > 0:
                            pct = self.frame_count / total_frames * 100
                            progress = f" ({pct:.1f}%)"
                        print(f"Frame {self.frame_count}{progress} | "
                              f"FPS: {result.current_fps:.1f} | "
                              f"Objects: {len(result.detections_3d)}")
                    
                    # Display
                    if display:
                        cv2.imshow("SP1 3D Detection", result.annotated_frame)
                        if result.depth_colormap is not None:
                            cv2.imshow("Depth Map", result.depth_colormap)
                
                # Keyboard handling
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[SP1 Video] Quit requested")
                        break
                    elif key == ord('s') and result is not None:
                        path = f"screenshot_{self.frame_count:06d}.png"
                        cv2.imwrite(path, result.annotated_frame)
                        print(f"[SP1 Video] Saved: {path}")
                    elif key == ord('p'):
                        paused = not paused
                        print(f"[SP1 Video] {'Paused' if paused else 'Resumed'}")
                    elif key == ord('d'):
                        self.config.show_depth_minimap = not self.config.show_depth_minimap
        
        except KeyboardInterrupt:
            print("\n[SP1 Video] Interrupted")
        
        finally:
            self._cleanup()
        
        # Final stats
        if stats["total_times"]:
            stats["avg_fps"] = 1000 / np.mean(stats["total_times"])
            stats["avg_detection_ms"] = np.mean(stats["detection_times"])
            stats["avg_depth_ms"] = np.mean(stats["depth_times"])
            stats["avg_total_ms"] = np.mean(stats["total_times"])
        
        self._print_stats(stats)
        return stats
    
    def _print_stats(self, stats: Dict) -> None:
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Frames skipped: {stats['frames_skipped']}")
        print(f"Total detections: {stats['total_detections']}")
        if "avg_fps" in stats:
            print(f"Average FPS: {stats['avg_fps']:.1f}")
            print(f"Avg detection time: {stats['avg_detection_ms']:.1f}ms")
            print(f"Avg depth time: {stats['avg_depth_ms']:.1f}ms")
            print(f"Avg total time: {stats['avg_total_ms']:.1f}ms")
        print("=" * 50)
    
    def _cleanup(self) -> None:
        """Release resources."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        cv2.destroyAllWindows()
        print("[SP1 Video] Cleanup complete")
    
    def generate_frames(
        self,
        source: Union[str, int]
    ) -> Generator[FrameResult, None, None]:
        """
        Generator that yields processed frames.
        
        Useful for custom processing loops or integration with other systems.
        
        Args:
            source: Video file path or camera ID
            
        Yields:
            FrameResult for each processed frame
        """
        success, _ = self._open_source(source)
        if not success:
            return
        
        self.frame_count = 0
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                if self.frame_count % self.config.process_every_n_frames != 0:
                    continue
                
                yield self.process_frame(frame)
        
        finally:
            self._cleanup()


def create_side_by_side(
    annotated: np.ndarray,
    depth_vis: np.ndarray
) -> np.ndarray:
    """Create side-by-side view of annotated frame and depth map."""
    h, w = annotated.shape[:2]
    depth_resized = cv2.resize(depth_vis, (w, h))
    
    combined = np.hstack([annotated, depth_resized])
    
    cv2.putText(combined, "RGB + 3D Detection", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Depth Estimation", (w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return combined
