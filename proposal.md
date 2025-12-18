### Executive Summary

| Requirement | Proposed Strategy | Status |
| :--- | :--- | :--- |
| **I. Project Goal** | Semantic Navigation via core cognitive capabilities. | **Targeted.** The combined output of 3D Object Bounding Boxes and 3D Room Layout Mesh provides the complete data structure for a downstream Semantic Navigation module. |
| **Constraint** | **RGB-Only Pipeline (No Depth/LiDAR).** | **Addressed.** Both sub-projects are built on a Monocular Vision (single RGB camera) deep learning and geometric projection stack. |
| **Constraint** | **Deployment on Jetson Orin (8GB).** | **Addressed.** All chosen models/architectures are optimized using **NVIDIA TensorRT** for FP16/INT8 precision to achieve real-time performance on the target edge device. |
| **Timeline** | **2 Months (Fully Deployable).** | The proposal is structured as a **Minimum Viable Deployment (MVD)** focusing *only* on the core deliverables for a successful hand-off. An optional Phase 2 for robustness is recommended. |

---

## Sub-Project 1: 3D Open-Vocabulary Object Detection (RGB-Only)

### I. Objectives

*   **Core:** To generate a 3D bounding box (location, dimensions, and orientation) for any user-queried object (open-vocabulary) from a single RGB camera stream.
*   **Metric:** Real-time inference on the Jetson Orin 8GB.
*   **Output:** A Python API function that accepts an RGB image and a text prompt (e.g., "blue chair," "remote control") and returns a list of 3D object poses/boxes in a local camera frame.

### II. Strategy: 2D Open-Vocabulary -> Monocular Depth -> 3D Projection

| Stage | Strategy | Rationale (Why this strategy?) |
| :--- | :--- | :--- |
| **A. Open-Vocabulary 2D Detection** | Use a highly efficient, pre-trained Vision-Language Model backbone like **YOLO-World-N (object detection + generalized conceptual knowledge of a Large Language Model)** | This models provide state-of-the-art open-vocabulary detection and are designed for speed, often outperforming traditional methods. YOLO-World, in particular, is highly suitable for real-time edge deployment. |
| **B. Monocular Depth Estimation** | Use a lightweight, single-image Monocular Depth Estimation (MDE) network **Depth Anything V2 (Fine-tuned for Metric Depth Estimation)**. It gives you the Z coordinate (depth) for every point. | This directly addresses the **RGB-Only** constraint by synthetically generating the necessary depth. |
| **C. 3D Pose Estimation** | Triangulate the 2D bounding box (from Stage A) with the estimated depth (from Stage B) and the known camera intrinsics (Calibration Task). A PnP (Perspective-n-Point) solver or a similar geometric projection method will generate the final 3D bounding box coordinates and orientation. **Stage C is where the raw data (depth map + 2D box) is structured into a complete 3D Bounding Box.** | This converts the 2D semantic information into the required 3D spatial coordinates, completing the core deliverable: ready for the navigation system. |

### III. Milestones and Tasks

| Milestone | Deliverables (Output) | Task Breakdown | Timeline (Weeks) |
| :--- | :--- | :--- | :--- |
| **M1: Technical Feasibility Study (SP1)** | Status of Core models (**YOLO-World-N**, **MDE**) and their suitability for the Orin confirmation. **(Validation of Core Strategy)** | Select, acquire, and **validate base models** for architectural compatibility and conceptual accuracy on test data. Basically how technically feasible it is. | **Weeks 1-2** |
| **M2: Camera Calibration & Depth Refinement** | Camera Intrinsics determined and Metric Depth Estimation (MDE) refined. | Implement 3D geometric projection (PnP solver/triangulation) and 3D bounding box fitting logic Perform Camera Calibration (Intrinsics) and establish the local camera coordinate frame. | **Weeks 3-4** |
| **M3: 3D Geometric Projection & Unoptimized API** | Unoptimized Python API for Monocular 3D Detection | T3.1: Integrate the 2D Detector, MDE, and 3D Projection into a single Python pipeline for end-to-end functionality. | **Weeks 5-6** |
| **M4: Orin Deployment & Final Test** | **Fully Deployable Python API (TensorRT Optimized) for SP1.** | T4.1: Convert all models (2D Detector, MDE) to ONNX format. T4.2: Optimize models for Jetson Orin using NVIDIA TensorRT (FP16/INT8). T4.3: Final latency/accuracy testing on the physical Jetson Orin 8GB device and SP1 documentation. | **Weeks 7-8** |

---

## Sub-Project 2: 3D Room Layout Estimation (RGB-Only)

### I. Objectives

*   **Core:** To reconstruct the structural elements of a room (walls, floor, ceiling, corners, openings/doors) into a simplified 3D mesh model, functionally similar to Apple's **RoomPlan**.
*   **Metric:** Real-time layout update on the Jetson Orin 8GB.
*   **Output:** A Python API function that accepts an RGB image and returns a parameterized 3D room layout structure (e.g., a dictionary of 3D vertices/planes).

### II. Strategy: Single-Image Layout -> Geometric Constraints -> Optimization

| Stage | Strategy | Rationale (Why this strategy?) |
| :--- | :--- | :--- |
| **A. 2D Structural Prediction** | Use a lightweight, end-to-end **FCNN/Transformer architecture** like **ST-RoomNet (ConvNext-Lite Backbone)**. This model is known for high-speed, direct prediction of perspective transformation parameters or key points from a single RGB image. | This model is an efficient, state-of-the-art CNN-based approach that is demonstrably capable of running (without TensorRT) on similar hardware, directly addressing the **RGB-Only** and **Real-Time** constraints for edge deployment. |
| **B. 3D Model Generation** | The network's 2D predictions (e.g., keypoints, boundary masks, or transformation parameters) are fed into a **constrained non-linear optimization** layer. This layer enforces **Manhattan World** assumptions **(Note: Assumes orthogonal walls and structural alignment along three perpendicular axes)** and geometric consistency. | This is the crucial step that generates the parameterized, clean 3D output (the "RoomPlan-like" feature) from the initial pixel-level/parameter predictions, converting the 2D semantics into a geometrically sound 3D mesh. |
| **C. Layout Refinement for Openings** | Incorporate a secondary, lightweight segmentation network based on an **FCN/MobileNetV2** backbone using **pre-trained weights from the ADE20K** semantic segmentation dataset. This provides pixel-accurate masks for the classes **'door'** and **'window'**. | This strategy leverages a confirmed, publicly available, and highly efficient semantic segmentation model architecture known for ADE20K training. It eliminates the need for custom fine-tuning (reducing MVD risk) and provides pixel-accurate masks, which are superior for clean geometric cut-outs in the 3D mesh. |

### III. Milestones and Tasks

| Milestone | Deliverables (Output) | Task Breakdown | Timeline (Weeks) |
| :--- | :--- | :--- | :--- |
| **M5: Technical Feasibility Study (SP2)** | Core Layout Network FCNN/Transformer architecture. **(Validation of 2D-to-3D Layout Approach)** | T5.1: Validate the feasibility of the model. | **Weeks 1-2** |
| **M6: 3D Core Layout Pipeline** | Functional 2D-to-3D Layout Generator (Base Room Mesh, No Openings). | T6.1: Implement the 3D non-linear optimization/fitting module (Manhattan World logic). T6.2: Integrate the 2D prediction (M5) with the 3D logic. | **Weeks 3-4** |
| **M7: Openings Integration & Full API** | Unoptimized Python API for 3D Room Layout (Mesh/Corner output) including openings. | **T7.1: Acquire pre-trained FCN/MobileNetV2 (ADE20K) model weights.** **T7.2: Integrate semantic segmentation output (pixel mask) into the 3D optimization for clean "cut-outs."** T7.3: Ensure a clean, structured output (e.g., a dictionary of 3D vertices/planes). | **Weeks 5-6** |
| **M8: Orin Deployment & Finalization** | **Fully Deployable Python API (TensorRT Optimized) for SP2 & Full System Integration.** | T8.1: Convert all models to ONNX format. T8.2: Optimize models for Jetson Orin using NVIDIA TensorRT (FP16). T8.3: Final System Integration Test (Object Detection + Layout) and final documentation/handover. | **Weeks 7-8** |

---

## IV. How the Two Sub-Projects Connect

In simple terms, these two parallel systems create a comprehensive digital representation of the world by defining a map and then populating that map with targets.

1.  **Sub-Project 2 (Room Layout)** is the **Container (The Map):** It uses the RGB image to build the unmoving, structural framework of the environment—the 3D walls, floor, ceiling, and the essential doors for passing through. This provides the stable, global coordinate system and defines the *traversable space*.

2.  **Sub-Project 1 (Object Detection)** is the **Content (The Targets):** It uses the same RGB image and a user prompt to find the specific, searchable items inside that framework (e.g., "remote control," "blue chair").

3.  **The Connection (Registration):** The final step is to precisely place the 3D locations of the objects (SP1 output) *inside* the 3D mesh of the room structure (SP2 output). This fusion allows the final Semantic Navigation module to accurately plan a path through the SP2 map *to* a specific object found by SP1.
