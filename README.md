# PyCAM3D

CAM software integrating **OpenCAMLib**, **Trimesh**, and **Open3D** for complete 3D scan-to-gcode workflow.

## Features

- **Mesh Processing**: Load, clean, repair, and transform 3D meshes (STL, OBJ, PLY, GLTF)
- **Point Cloud Support**: Reconstruct meshes from 3D scans using Poisson, Ball Pivoting, or Alpha Shape
- **Toolpath Generation**: Drop-cutter, waterline, roughing, and finishing operations via OpenCAMLib
- **4-Axis Machining**: Rotary machining for cylindrical/conical parts (ported from Fabex)
- **G-code Output**: Multi-machine support (Generic, LinuxCNC, GRBL, Mach3, Fanuc, Haas, LUQUE L1530)
- **CLI Interface**: Easy-to-use command line tools

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         PyCAM3D                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐                        │
│  │   Open3D     │───▶│   Trimesh    │                        │
│  │ (scan input) │    │ (mesh clean) │                        │
│  └──────────────┘    └──────┬───────┘                        │
│                             │                                 │
│               ┌─────────────┴─────────────┐                  │
│               ▼                           ▼                   │
│      ┌────────────────┐         ┌──────────────────┐         │
│      │  OpenCAMLib    │         │  4-Axis Pipeline │         │
│      │ (3-axis paths) │         │  (rotary paths)  │         │
│      └───────┬────────┘         └────────┬─────────┘         │
│              │                           │                    │
│              └───────────┬───────────────┘                    │
│                          ▼                                    │
│                 ┌────────────────┐                            │
│                 │  G-code out    │                            │
│                 │ (3/4/5 axis)   │                            │
│                 └────────────────┘                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
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

# 4-Axis rotary machining
pycam3d 4axis cylinder.stl -s PARALLELR -a X -m LUQUE_L1530

# 4-Axis roughing (HELIX strategy)
pycam3d 4axis-roughing part.stl --stepdown 5 --stock-to-leave 0.5

# 4-Axis finishing (fine PARALLELR)
pycam3d 4axis-finishing part.stl --stepover 0.5
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

### 4-Axis Rotary Machining

```python
from pycam3d import Pipeline4Axis, Tool
from pycam3d.gcode_multiaxis import MachineType4Axis

# Load mesh and create tool
tool = Tool.ball(diameter=6.0)
pipeline = Pipeline4Axis("cylinder.stl", tool, MachineType4Axis.LUQUE_L1530)

# Generate 4-axis toolpath
result = pipeline.generate(
    strategy='PARALLELR',      # Circular passes around axis
    rotary_axis='X',           # Axis of rotation
    stepover=2.0,              # mm between passes
    feed_rate=1000,            # mm/min
    spindle_speed=12000        # RPM
)

# Save G-code with B-axis commands
pipeline.save_gcode(result, "output_4axis.nc")
print(f"Generated {result.total_points} points, est. {result.estimated_time_min:.1f} min")
```

#### Quick 4-Axis Generation

```python
from pycam3d import quick_4axis

# One-liner for simple jobs
result = quick_4axis(
    "part.stl",
    "output.nc",
    tool_diameter=6.0,
    tool_type='ball',
    strategy='PARALLELR',
    machine='LUQUE_L1530',
    feed_rate=1000
)
```

## Supported Tools

| Tool Type | Description | Use Case |
|-----------|-------------|----------|
| Ball | Ball nose end mill | 3D finishing, smooth surfaces |
| Flat | Flat end mill (cylindrical) | Roughing, flat surfaces |
| Bull | Bull nose (toroidal) | Combined roughing/finishing |
| Cone | Conical/tapered | V-carving, engraving |

## Toolpath Strategies

### 3-Axis Strategies

- **Parallel X/Y**: Unidirectional passes
- **Zigzag X/Y**: Bidirectional passes (faster)
- **Waterline**: Constant Z contours (for steep walls)
- **Roughing**: Layered material removal
- **Finishing**: Fine surface pass

### 4-Axis Strategies (Rotary)

| Strategy | Description | Use Case |
|----------|-------------|----------|
| PARALLELR | Circular passes around the rotary axis | Turning, cylindrical surfaces |
| PARALLEL | Linear passes along the rotary axis | Surfacing flanks |
| HELIX | Continuous spiral path | Fast roughing |
| CROSS | Both directions combined | Best surface finish |

```
PARALLELR (around axis)          PARALLEL (along axis)           HELIX (spiral)
    ┌─────────┐                     ════════════                 ╭──────────╮
   ╱│         │╲                    ════════════                ╱│          │
  ╱ │  ────▶  │ ╲                   ════════════               ╱ │  ╲       │
 │  │   ○     │  │                  ════════════              │  │   ╲      │
  ╲ │         │ ╱                   ════════════               ╲ │    ╲     │
   ╲│         │╱                    ════════════                ╲│     ╲    │
    └─────────┘                     ════════════                 ╰──────────╯
```

## Machine Support

### 3-Axis Machines

| Machine | Notes |
|---------|-------|
| Generic | Standard RS274/NGC G-code |
| LinuxCNC | Path blending (G64) |
| GRBL | Homing ($H), limited feed rates |
| Mach3 | Standard |
| Fanuc | Program numbers (Oxxxx) |
| Haas | Program numbers, tool change macros |

### 4-Axis Machines

| Machine | Rotary Axis | Mode | Notes |
|---------|-------------|------|-------|
| LUQUE L1530 | B (spindle tilt) | Simultaneous | -120° to +120° |
| Generic 4-Axis | A | Simultaneous | Full 360° |
| Rotary Table A | A | Indexed | Positional only |
| Rotary Table C | C | Simultaneous | Full 360° |
| Haas 4-Axis | A | Simultaneous | Fast indexing |
| Trunnion | A+C | Simultaneous | 5-axis capable |

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
