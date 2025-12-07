# PyCAM3D

## Overview

PyCAM3D is a comprehensive CAM (Computer-Aided Manufacturing) software that generates CNC toolpaths from 3D models and scans. It integrates OpenCAMLib for toolpath algorithms, Trimesh for mesh manipulation, and Open3D for point cloud processing. The system supports both 3-axis and 4-axis machining, with advanced strategies including iso-scallop, trochoidal milling, and adaptive clearing. Users can input STL/OBJ/PLY files or point clouds, configure machining parameters, and export G-code for various CNC controllers.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Pipeline Architecture

The application follows a modular pipeline design with three main processing stages:

1. **Mesh Processing Layer** (`mesh.py`): Handles 3D model loading, point cloud reconstruction (Poisson, Ball Pivoting, Alpha Shape methods), mesh repair, and transformations. Uses Trimesh as the primary mesh manipulation library with Open3D for point cloud operations.

2. **Toolpath Generation Layer** (`toolpath.py`, `strategies.py`): Implements OpenCAMLib-based toolpath algorithms including drop-cutter, waterline, parallel, spiral, and advanced strategies (iso-scallop with curvature analysis, trochoidal milling). Supports multiple tool types (ball, flat, bull nose, cone) with automatic collision avoidance.

3. **G-code Export Layer** (`gcode.py`, `gcode_multiaxis.py`): Converts toolpaths to machine-specific G-code dialects (Generic, LinuxCNC, GRBL, Mach3, Fanuc, Haas) with support for 3-axis and 4-axis (rotary) operations.

### Multi-Axis Machining Support

The system includes specialized 4-axis rotary machining capabilities:

- **Pattern Generation** (`patterns/rotary.py`): Creates PARALLEL, PARALLELR, and HELIX toolpath patterns for cylindrical/revolution parts
- **Surface Sampling** (`collision/sampler_4axis.py`): Projects patterns onto mesh surfaces using ray casting with tool geometry compensation
- **Chunk-based Toolpaths** (`chunk.py`): Stores trajectory points with A/B/C axis rotations for multi-axis machines

### Intelligent Decision Engine

The `decision.py` module acts as an automated "brain" that:

- Analyzes part geometry to detect features, undercuts, and accessibility issues
- Recommends optimal machining strategies, tools, and parameters
- Integrates machine database (`machines.py`) with work envelope limits and post-processor configs
- Integrates materials database (`materials.py`) for feeds/speeds calculation
- Performs automatic part orientation optimization (`orientation.py`) to minimize setups

### Web Interface

FastAPI-based web server (`web.py`) provides:

- REST API for mesh upload and toolpath generation
- Three.js-based 3D visualization of meshes and toolpaths
- Interactive parameter configuration
- Real-time preview of machining operations

Entry point is `main.py` which starts the Uvicorn server on port 5000.

### CLI Interface

Command-line tools (`cli.py`) built with Click framework provide batch processing capabilities with Rich terminal output for progress tracking and results visualization.

### Supporting Systems

- **Stock Management** (`stock.py`): Defines stock geometry (rectangular, cylindrical, from mesh) and workholding fixtures with collision zones
- **Tool Library** (`tools.py`): Comprehensive database of cutting tools with materials, coatings, geometries, and tool life estimation
- **Simulation Engine** (`simulation.py`): Real-time toolpath animation, material removal visualization, and machining time estimation

## External Dependencies

### Core CAM Libraries

- **OpenCAMLib**: Toolpath generation algorithms (drop-cutter, waterline, push-cutter operations)
- **Trimesh**: Primary mesh loading, manipulation, repair, and analysis library
- **Open3D**: Point cloud processing and mesh reconstruction algorithms
- **NumPy**: Numerical array operations and geometric calculations

### Web Framework

- **FastAPI**: REST API server framework
- **Uvicorn**: ASGI server for hosting the web application
- **Pydantic**: Request/response data validation
- **python-multipart**: File upload handling

### CLI & Utilities

- **Click**: Command-line interface framework
- **Rich**: Terminal formatting and progress display
- **SciPy**: Spatial transformations and scientific computing (used in collision detection)

### Development Tools

- **pytest**: Test framework with coverage support
- **black**: Code formatting
- **ruff**: Fast Python linter
- **mypy**: Static type checking

### Data Format Support

The application handles multiple 3D file formats through Trimesh's format ecosystem:
- Mesh formats: STL, OBJ, PLY, GLTF/GLB
- Point cloud formats: PLY, PCD, XYZ, PTS
- G-code output: Various CNC controller dialects

### No Database Required

The application operates primarily on file I/O with in-memory processing. Machine and material databases are defined as Python enums and dataclasses rather than external database systems.