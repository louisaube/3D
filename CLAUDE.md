# CLAUDE.md - PyCAM3D Development Guide

## Project Overview

PyCAM3D is CAM (Computer Aided Manufacturing) software for generating CNC toolpaths and G-code from 3D models. It integrates **OpenCAMLib**, **Trimesh**, and **Open3D** to provide a complete 3D scan-to-gcode workflow.

**Key Capabilities:**
- Load and process 3D meshes (STL, OBJ, PLY, GLTF)
- Reconstruct meshes from point cloud scans
- Generate toolpaths using multiple strategies
- Output G-code for various CNC machines
- 4-axis rotary machining support
- Smart multi-tool strategy recommendations
- Web-based 3D visualization and toolpath preview

## Repository Structure

```
3D/
├── src/pycam3d/           # Main package source
│   ├── __init__.py        # Package exports and version
│   ├── cli.py             # Click CLI commands
│   ├── web.py             # FastAPI server + Three.js UI
│   ├── pipeline.py        # High-level CAM workflow orchestration
│   ├── mesh.py            # Mesh loading and processing (Trimesh + Open3D)
│   ├── toolpath.py        # Toolpath generation via OpenCAMLib
│   ├── gcode.py           # G-code generation and machine configs
│   ├── strategies.py      # Advanced strategies (iso-scallop, trochoidal, spiral)
│   ├── curvature.py       # Curvature analysis for adaptive stepover
│   ├── smart_strategy.py  # Multi-tool strategy recommendations
│   ├── machines.py        # CNC machine database
│   ├── materials.py       # Material database with feeds/speeds
│   ├── tools.py           # Tool library
│   ├── stock.py           # Stock/workholding definitions
│   ├── orientation.py     # Part orientation optimization
│   ├── decision.py        # Smart decision engine for auto settings
│   ├── simulation.py      # Toolpath simulation engine
│   ├── pipeline_4axis.py  # 4-axis machining pipeline
│   ├── chunk.py           # 4-axis path chunks
│   ├── gcode_multiaxis.py # Multi-axis G-code generation
│   ├── patterns/          # Rotary pattern generators
│   │   └── rotary.py
│   └── collision/         # Collision detection
│       └── sampler_4axis.py
├── tests/                 # pytest test suite
├── examples/              # Example workflows
├── main.py                # Web server entry point (Replit)
├── pyproject.toml         # Project configuration and dependencies
└── README.md              # User documentation
```

## Development Commands

### Setup and Installation

```bash
# Install package in development mode
pip install -e .

# Install with dev dependencies (testing, linting)
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_toolpath.py

# Run with coverage
pytest --cov=pycam3d
```

### Code Quality

```bash
# Format code with black
black src tests

# Lint with ruff
ruff check src tests

# Type checking
mypy src
```

### Running the Application

```bash
# CLI usage
pycam3d process model.stl -o output.nc
pycam3d waterline model.stl -o waterline.nc
pycam3d info model.stl
pycam3d simulate output.nc

# Web server
pycam3d serve --port 8000

# Or directly via main.py (Replit entry point)
python main.py
```

## Architecture Patterns

### Core Pipeline Pattern

The codebase uses a pipeline pattern where `CAMPipeline` orchestrates the workflow:

```python
from pycam3d import CAMPipeline, CAMJob, Tool

pipeline = CAMPipeline()
pipeline.load_mesh("model.stl")
pipeline.prepare_mesh(repair=True, center=True, place_on_bed=True)

job = CAMJob(
    roughing_tool=Tool.flat(diameter=10.0),
    finishing_tool=Tool.ball(diameter=6.0),
    roughing_stepover=0.4,
    finishing_stepover=0.15,
)

result = pipeline.run(job)
result.gcode.save("output.nc")
```

### Key Classes and Their Responsibilities

| Class | Module | Purpose |
|-------|--------|---------|
| `CAMPipeline` | pipeline.py | High-level workflow orchestration |
| `MeshProcessor` | mesh.py | Mesh loading, cleaning, repair, transforms |
| `ToolpathGenerator` | toolpath.py | OpenCAMLib-based toolpath generation |
| `GCodeWriter` | gcode.py | G-code generation with machine dialects |
| `Tool` | toolpath.py | Tool definition (ball, flat, bull, cone) |
| `Toolpath` | toolpath.py | Collection of toolpath points |
| `CurvatureAnalyzer` | curvature.py | Surface curvature computation |
| `SmartStrategy` | smart_strategy.py | Multi-tool machining recommendations |
| `DecisionEngine` | decision.py | Automatic parameter selection |
| `Pipeline4Axis` | pipeline_4axis.py | 4-axis rotary machining |
| `SimulationEngine` | simulation.py | Toolpath animation and validation |

### Toolpath Strategies

The codebase supports multiple toolpath strategies:

- **Parallel X/Y**: Unidirectional raster passes
- **Zigzag X/Y**: Bidirectional passes (faster)
- **Waterline**: Constant Z contours for steep walls
- **Iso-Scallop**: Adaptive stepover based on curvature (in `strategies.py`)
- **Spiral**: Continuous spiral from center outward
- **Trochoidal**: Circular motion for constant engagement

### 4-Axis Machining

For rotary parts, use the 4-axis pipeline:

```python
from pycam3d.pipeline_4axis import Pipeline4Axis
from pycam3d.toolpath import Tool

pipeline = Pipeline4Axis("part.stl", Tool.ball(6.0))
result = pipeline.generate(
    strategy="PARALLELR",  # or PARALLEL, HELIX, CROSS
    rotary_axis="X",
    stepover=2.0,
)
pipeline.save_gcode(result, "output.nc")
```

## Coding Conventions

### Python Style

- Python 3.9+ compatible
- Line length: 100 characters (configured in pyproject.toml)
- Use type hints for function signatures
- Docstrings in Google style
- Use `from __future__ import annotations` for modern type syntax

### Imports

```python
# Standard library
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import trimesh

# Local
from pycam3d.toolpath import Tool, Toolpath
```

### Error Handling

- Raise specific exceptions with descriptive messages
- Use logging for operational messages
- Validate inputs early with clear error messages

```python
if self._mesh is None:
    raise ValueError("No mesh loaded")
```

### Dataclasses

Use dataclasses for structured data:

```python
@dataclass
class MachineConfig:
    machine_type: MachineType = MachineType.GENERIC
    units: Units = Units.MM
    max_feed_rate: float = 5000.0
```

### Enums

Use Enums for fixed options:

```python
class ToolType(Enum):
    BALL = "ball"
    FLAT = "flat"
    BULL = "bull"
    CONE = "cone"
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and geometry calculations |
| `trimesh` | Mesh loading, manipulation, and export |
| `open3d` | Point cloud processing and mesh reconstruction |
| `opencamlib` | Toolpath generation algorithms (drop-cutter, waterline) |
| `click` | CLI framework |
| `rich` | Terminal UI and progress indicators |
| `fastapi` | Web API (in web.py) |
| `uvicorn` | ASGI server |

## Testing Patterns

Tests are in `tests/` using pytest:

```python
class TestTool:
    """Tests for Tool class."""

    def test_ball_tool(self):
        """Test creating ball nose tool."""
        tool = Tool.ball(diameter=6.0)

        assert tool.type == ToolType.BALL
        assert tool.diameter == 6.0

    def test_custom_name(self):
        """Test custom tool name."""
        tool = Tool(type=ToolType.BALL, diameter=6.0, name="Custom")
        assert tool.name == "Custom"
```

## Web Interface

The web interface (in `web.py`) uses:

- **FastAPI** for the REST API
- **Three.js** for 3D visualization (embedded in HTML template)
- **Endpoints:**
  - `POST /api/upload` - Upload mesh file
  - `POST /api/scale` - Scale mesh
  - `POST /api/analyze` - Smart strategy analysis
  - `POST /api/generate` - Generate toolpath
  - `GET /api/simulation/{mesh_id}` - Get animation data

## Common Tasks

### Adding a New Toolpath Strategy

1. Add strategy enum in `toolpath.py`:
   ```python
   class Strategy(Enum):
       NEW_STRATEGY = "new_strategy"
   ```

2. Implement generator in `strategies.py`:
   ```python
   class NewStrategyGenerator:
       def __init__(self, mesh, stl_surf, bounds):
           ...
       def generate(self, tool, **params) -> Toolpath:
           ...
   ```

3. Add CLI command in `cli.py`
4. Add tests in `tests/`

### Adding a New Machine Type

1. Add to `MachineType` enum in `gcode.py`
2. Handle dialect specifics in `GCodeWriter._generate_header()`
3. Add machine config to `machines.py` database

### Modifying G-code Output

G-code generation is in `gcode.py`. The `GCodeWriter` class handles:
- Machine-specific headers/footers
- Line formatting and numbering
- Coordinate precision
- Feed rate handling

## Important Notes

### OpenCAMLib Integration

- STL files are loaded via `ocl.STLSurf()` and `ocl.STLReader()`
- Use `PathDropCutter` for surface-following operations
- Use `Waterline` for constant-Z contours
- Tool types map to OCL cutters (BallCutter, CylCutter, BullCutter)

### Performance Considerations

- Curvature analysis uses vectorized numpy operations (see `curvature.py`)
- Large meshes should be simplified before toolpath generation
- Smart strategy has a `fast_mode` option for quick heuristic analysis
- Toolpath optimization removes collinear points

### Thread Safety

- Web server uses temporary mesh storage per session
- Each request gets its own processing pipeline

## File Formats

### Input Formats
- Mesh: STL, OBJ, PLY, GLTF/GLB, OFF
- Point Cloud: PLY, PCD, XYZ

### Output Formats
- G-code: .nc, .gcode, .tap (machine-specific dialects)
- Mesh: STL (for cleaned/prepared meshes)

## CLI Commands Reference

```
pycam3d process     # Full pipeline: mesh to G-code
pycam3d waterline   # Constant-Z contour toolpaths
pycam3d prepare     # Clean and repair mesh
pycam3d info        # Display mesh statistics
pycam3d simulate    # Simulate G-code
pycam3d iso-scallop # Adaptive stepover strategy
pycam3d spiral      # Spiral toolpath
pycam3d trochoidal  # Trochoidal milling for slots
pycam3d curvature   # Analyze surface curvature
pycam3d wizard      # Interactive workflow
pycam3d serve       # Start web server
pycam3d machines    # List machine database
pycam3d materials   # List material database
pycam3d tools       # List tool library
pycam3d auto        # Automatic settings based on machine/material
pycam3d 4axis       # 4-axis rotary machining
pycam3d analyze     # Part accessibility analysis
```

## Contributing Guidelines

1. Run tests before committing: `pytest`
2. Format code: `black src tests`
3. Check linting: `ruff check src tests`
4. Add tests for new features
5. Update docstrings and type hints
6. Keep commits focused and descriptive
