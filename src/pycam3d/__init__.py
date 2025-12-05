"""
PyCAM3D - CAM software integrating OpenCAMLib, Trimesh, and Open3D.

This package provides a complete pipeline for:
- Loading and processing 3D scans (point clouds, meshes)
- Mesh cleaning and repair
- Toolpath generation using OpenCAMLib
- G-code output for CNC machines

Advanced strategies include:
- Iso-Scallop: Adaptive stepover for constant scallop height
- Trochoidal: Constant engagement angle milling
- Spiral: Continuous spiral toolpaths
- Voronoi/Medial Axis: Optimal pocketing from center outward
"""

__version__ = "0.1.0"

from pycam3d.mesh import MeshProcessor
from pycam3d.toolpath import ToolpathGenerator, Tool, ToolType, Toolpath, Strategy
from pycam3d.gcode import GCodeWriter
from pycam3d.pipeline import CAMPipeline

# Advanced strategies
from pycam3d.curvature import (
    CurvatureAnalyzer,
    CurvatureField,
    compute_scallop_height,
    compute_stepover_for_scallop,
)
from pycam3d.strategies import (
    IsoScallopGenerator,
    TrochoidalGenerator,
    TrochoidalParams,
    SpiralGenerator,
    VoronoiPocketGenerator,
)

__all__ = [
    # Core
    "MeshProcessor",
    "ToolpathGenerator",
    "Tool",
    "ToolType",
    "Toolpath",
    "Strategy",
    "GCodeWriter",
    "CAMPipeline",
    # Curvature analysis
    "CurvatureAnalyzer",
    "CurvatureField",
    "compute_scallop_height",
    "compute_stepover_for_scallop",
    # Advanced strategies
    "IsoScallopGenerator",
    "TrochoidalGenerator",
    "TrochoidalParams",
    "SpiralGenerator",
    "VoronoiPocketGenerator",
]
