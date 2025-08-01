# 3D MCP Tools

This repository contains three Model Context Protocol (MCP) tools for 3D data processing and analysis:

## 1. Point Cloud Generation MCP (`mcp_point_cloud_generation.py`)

**Purpose**: Generates 3D point clouds from image sequences using the CUT3R (ARCroco3DStereo) model.

**Input**:

- `seq_path`: Path to the image sequence folder
- `model_path`: Path to the CUT3R model (default: "src/cut3r_512_dpt_4_64.pth")
- `size`: Image size for processing (default: 512)
- `vis_threshold`: Visibility threshold for point filtering (default: 1.5)
- `skip_frames`: Whether to skip frames during processing (default: True)
- `output_dir`: Output directory path (optional)

**Output**:

- A PLY file containing the generated 3D point cloud (`fused.ply`)
- Camera intrinsics and poses for each frame
- Depth maps and confidence for each frame
- Processed image frames
- Returns the path to the generated point cloud file and output folder

**Use Case**: Convert 2D image sequences (like video frames) into 3D point cloud representations for further processing.

## 2. Mesh Generation MCP (`mcp_mesh_generation.py`)

**Purpose**: Converts PLY point cloud files into 3D mesh surfaces using various reconstruction algorithms.

**Input**:

- `ply_path`: Path to the input PLY point cloud file
- `method`: Mesh reconstruction method (poisson, alpha, or ball_pivoting)
- `output_path`: Output path for the generated mesh (optional)
- `poisson_depth`: Depth parameter for Poisson reconstruction (default: 8)
- `alpha`: Alpha parameter for alpha shape reconstruction (default: 0.03)
- `ball_radii`: Ball radii for ball pivoting reconstruction (default: [0.005, 0.01, 0.02])

**Output**:

- A PLY file containing the generated 3D mesh surface
- Returns the path to the generated mesh file

**Methods Available**:

- **Poisson**: Surface reconstruction using Poisson surface reconstruction
- **Alpha**: Surface reconstruction using alpha shapes
- **Ball Pivoting**: Surface reconstruction using ball pivoting algorithm

**Use Case**: Convert sparse point clouds into continuous mesh surfaces for visualization, analysis, or 3D printing.

## 3. Floor Data MCP (`mcp_floor_data.py`)

**Purpose**: Generates 2D floor maps from PLY point cloud files for navigation and planning.

**Input**:

- `ply_path`: Path to the PLY point cloud file
- `output_path`: Output path for the generated floor map (optional)

**Output**:

- A 2D floor map representation of the 3D environment
- Returns the generated floor data

**Use Case**: Create navigable 2D representations of 3D environments for robotics, autonomous navigation, or spatial analysis.

## Installation and Usage

Each MCP tool can be run independently as a server on port 8000. The tools automatically clone and set up the required dependencies from their respective repositories:

- Point Cloud Generation: Uses CUT3R repository
- Mesh Generation: Uses solo-3d repository
- Floor Data: Uses solo-3d repository

All tools require CUDA support for optimal performance, especially the point cloud generation tool which uses deep learning models.
