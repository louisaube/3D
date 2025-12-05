# PyCAM3D

CAM software integrating **OpenCAMLib**, **Trimesh**, and **Open3D** for complete 3D scan-to-gcode workflow.

## Features

- **Mesh Processing**: Load, clean, repair, and transform 3D meshes (STL, OBJ, PLY, GLTF)
- **Point Cloud Support**: Reconstruct meshes from 3D scans using Poisson, Ball Pivoting, or Alpha Shape
- **Toolpath Generation**: Drop-cutter, waterline, roughing, and finishing operations via OpenCAMLib
- **G-code Output**: Multi-machine support (Generic, LinuxCNC, GRBL, Mach3, Fanuc, Haas)
- **CLI Interface**: Easy-to-use command line tools

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    PyCAM3D                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐    ┌──────────────┐              │
│  │   Open3D     │───▶│   Trimesh    │              │
│  │ (scan input) │    │ (mesh clean) │              │
│  └──────────────┘    └──────┬───────┘              │
│                             │                       │
│                             ▼                       │
│                    ┌────────────────┐               │
│                    │  OpenCAMLib    │               │
│                    │ (toolpaths)    │               │
│                    └───────┬────────┘               │
│                            │                        │
│                            ▼                        │
│                    ┌────────────────┐               │
│                    │   G-code out   │               │
│                    └────────────────┘               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

### Dependencies

- **numpy**: Array operations
- **trimesh**: Mesh loading and manipulation
- **open3d**: Point cloud processing and reconstruction
- **opencamlib**: Toolpath algorithms
- **click**: CLI framework
- **rich**: Terminal UI

## Quick Start

### Command Line

```bash
# Process a mesh and generate G-code
pycam3d process model.stl -o output.nc --tool-diameter 6 --stepover 0.15

# Generate waterline toolpaths
pycam3d waterline model.stl -o waterline.nc --z-step 1.0

# Prepare mesh (repair, center, etc.)
pycam3d prepare model.stl -o cleaned.stl --repair --simplify 50000

# View mesh info
pycam3d info model.stl

# Simulate G-code
pycam3d simulate output.nc
```

### Python API

```python
from pycam3d import CAMPipeline, CAMJob, Tool

# Initialize pipeline
pipeline = CAMPipeline()

# Load and prepare mesh
pipeline.load_mesh("model.stl")
pipeline.prepare_mesh(repair=True, center=True, place_on_bed=True)

# Define CAM job
job = CAMJob(
    roughing_tool=Tool.flat(diameter=10.0),
    finishing_tool=Tool.ball(diameter=6.0),
    roughing_stepover=0.4,
    finishing_stepover=0.15,
    feed_rate=1200.0,
    spindle_rpm=12000,
)

# Generate toolpaths and G-code
result = pipeline.run(job)
result.gcode.save("output.nc")
```

### Point Cloud Workflow

```python
from pycam3d import CAMPipeline, CAMJob
from pycam3d.mesh import ReconstructionMethod

pipeline = CAMPipeline()

# Load point cloud and reconstruct mesh
pipeline.load_point_cloud(
    "scan.ply",
    method=ReconstructionMethod.POISSON,
    depth=9
)

# Prepare and generate toolpaths
pipeline.prepare_mesh(repair=True, smooth_iterations=2)
result = pipeline.run(CAMJob.default_3d_surfacing())
result.gcode.save("output.nc")
```

## Supported Tools

| Tool Type | Description | Use Case |
|-----------|-------------|----------|
| Ball | Ball nose end mill | 3D finishing, smooth surfaces |
| Flat | Flat end mill (cylindrical) | Roughing, flat surfaces |
| Bull | Bull nose (toroidal) | Combined roughing/finishing |
| Cone | Conical/tapered | V-carving, engraving |

## Toolpath Strategies

- **Parallel X/Y**: Unidirectional passes
- **Zigzag X/Y**: Bidirectional passes (faster)
- **Waterline**: Constant Z contours (for steep walls)
- **Roughing**: Layered material removal
- **Finishing**: Fine surface pass

## Machine Support

| Machine | Notes |
|---------|-------|
| Generic | Standard RS274/NGC G-code |
| LinuxCNC | Path blending (G64) |
| GRBL | Homing ($H), limited feed rates |
| Mach3 | Standard |
| Fanuc | Program numbers (Oxxxx) |
| Haas | Program numbers, tool change macros |

## Examples

See the `examples/` directory:

- `basic_workflow.py` - Complete STL to G-code
- `point_cloud_workflow.py` - 3D scan to G-code
- `advanced_strategies.py` - Different toolpath strategies

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests
```

## License

MIT
