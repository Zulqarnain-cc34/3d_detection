# SP1: 3D Open-Vocabulary Object Detection Pipeline

**RGB-Only 3D Object Detection for Semantic Navigation**

A complete pipeline for detecting and localizing objects in 3D space using only RGB images. Designed for deployment on **Jetson Orin 8GB** for humanoid robotics applications.

## 🎯 Overview

This pipeline enables robots to:
- **Recognize any object** using natural language queries (open-vocabulary)
- **Estimate 3D positions** without depth sensors (RGB-only)
- **Generate navigation waypoints** for semantic navigation tasks

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SP1 3D Detection Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Stage A    │    │   Stage B    │    │   Stage C    │      │
│  │              │    │              │    │              │      │
│  │  YOLO-World  │───►│ Depth Any V2 │───►│ 3D Projection│      │
│  │              │    │              │    │              │      │
│  │ Open-Vocab   │    │  Monocular   │    │  Geometric   │      │
│  │ 2D Detection │    │    Depth     │    │    Fusion    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│   2D Bounding Box      Depth Map         3D Bounding Box       │
│   + Class + Conf       (meters)         + 3D Position          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Features

- **Open-Vocabulary Detection**: Detect any object by text description
- **Monocular Depth**: No depth sensor required - works with RGB only
- **Real-Time Performance**: Optimized for edge deployment
- **Navigation Ready**: Generates 3D waypoints for robot navigation
- **Comprehensive Evaluation**: Built-in metrics and benchmarking

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd sp1_3d_detection

# Install dependencies
pip install -r requirements.txt

# Run the test
python run_test.py
```

### Basic Usage

```python
from src import SP1Pipeline

# Initialize pipeline
pipeline = SP1Pipeline(device='cuda:0')  # or 'cpu'

# Detect objects
result = pipeline.detect(
    image='path/to/image.jpg',
    classes=['chair', 'table', 'cup', 'laptop']
)

# View results
print(result.summary())

# Access 3D positions
for obj in result.detections_3d:
    print(f"{obj.class_name}: {obj.center} meters")
```

### Navigation Waypoint

```python
# Get waypoint to approach an object
waypoint = pipeline.get_waypoint(
    image='scene.jpg',
    target_object='red chair',
    offset_distance=0.5  # Stop 0.5m from object
)

print(f"Navigate to: {waypoint['waypoint_position']}")
```

## 📁 Project Structure

```
sp1_3d_detection/
├── src/
│   ├── __init__.py         # Package exports
│   ├── pipeline.py         # Main SP1Pipeline class
│   ├── detector.py         # Stage A: YOLO-World detection
│   ├── depth_estimator.py  # Stage B: Depth Anything V2
│   ├── projector.py        # Stage C: 3D projection
│   ├── visualizer.py       # Visualization utilities
│   └── evaluation.py       # Evaluation metrics
├── configs/
│   └── config.yaml         # Pipeline configuration
├── outputs/                # Generated outputs
├── data/                   # Test images
├── run_test.py            # Main test script
├── requirements.txt       # Dependencies
└── README.md
```

## 🧪 Running Tests

### Basic Test
```bash
python run_test.py
```

### Custom Image
```bash
python run_test.py --image path/to/your/image.jpg
```

### Performance Benchmark
```bash
python run_test.py --benchmark --benchmark-runs 20
```

### All Tests
```bash
python run_test.py --benchmark --multi-query --waypoint
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to test image | Downloads sample |
| `--classes` | Classes to detect | furniture list |
| `--device` | cpu or cuda:0 | cpu |
| `--benchmark` | Run timing benchmark | False |
| `--multi-query` | Test multiple queries | False |
| `--waypoint` | Test waypoint generation | False |
| `--confidence` | Detection threshold | 0.25 |

## 📊 Expected Performance

### Desktop (RTX 3080)
| Stage | Time |
|-------|------|
| Detection | ~30ms |
| Depth | ~50ms |
| Projection | ~1ms |
| **Total** | **~80ms (12 FPS)** |

### Jetson Orin 8GB (TensorRT FP16)
| Stage | Time |
|-------|------|
| Detection | ~50ms |
| Depth | ~80ms |
| Projection | ~2ms |
| **Total** | **~130ms (7.7 FPS)** |

## 🔧 Configuration

Edit `configs/config.yaml`:

```yaml
detector:
  model_name: "yolov8s-world"  # n/s/m/l variants
  confidence_threshold: 0.25
  device: "cuda:0"

depth_estimator:
  model_name: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
  depth_range:
    min_depth: 0.5
    max_depth: 10.0

projection:
  camera:
    fx: 525.0
    fy: 525.0
    cx: 320.0
    cy: 240.0
```

## 📈 Evaluation Metrics

The pipeline provides comprehensive metrics:

- **Detection**: Precision, Recall, F1-Score
- **Depth**: MAE, RMSE, Abs Relative Error
- **3D Localization**: Mean IoU, Position Error (meters)
- **Performance**: Inference time, FPS

## 🤖 Integration with Robot Navigation

```python
from src import SP1Pipeline

class NavigationController:
    def __init__(self):
        self.perception = SP1Pipeline(device='cuda:0')
    
    def find_and_approach(self, image, target_object):
        # Get 3D position of target
        detection = self.perception.detect_single_object(image, target_object)
        
        if detection:
            # Object position in camera frame
            x, y, z = detection.center
            
            # Convert to navigation command
            # (integrate with your navigation stack)
            return {
                'target_x': x,
                'target_y': y, 
                'target_z': z,
                'object_class': detection.class_name
            }
        return None
```

## 🔜 Roadmap to Production

1. **Current**: Python API with full functionality ✅
2. **Week 7-8**: TensorRT optimization for Jetson
3. **Optional**: ROS2 node wrapper for navigation stack

## 📝 API Reference

### SP1Pipeline

```python
class SP1Pipeline:
    def detect(image, classes) -> PipelineResult
    def detect_single_object(image, query) -> BoundingBox3D
    def get_waypoint(image, target, offset) -> Dict
    def benchmark(image, classes, num_runs) -> Dict
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    detections_2d: List[Detection2D]
    depth_result: DepthResult
    detections_3d: List[BoundingBox3D]
    detection_time_ms: float
    depth_time_ms: float
    total_time_ms: float
```

### BoundingBox3D

```python
@dataclass
class BoundingBox3D:
    center: np.ndarray      # [X, Y, Z] in meters
    dimensions: np.ndarray  # [W, H, D] in meters
    class_name: str
    confidence: float
```

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) for open-vocabulary detection
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
