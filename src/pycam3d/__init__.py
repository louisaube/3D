"""
PyCAM3D - CAM software integrating OpenCAMLib, Trimesh, and Open3D.

This package provides a complete pipeline for:
- Loading and processing 3D scans (point clouds, meshes)
- Mesh cleaning and repair
- Toolpath generation using OpenCAMLib
- G-code output for CNC machines
"""

__version__ = "0.1.0"

from pycam3d.mesh import MeshProcessor
from pycam3d.toolpath import ToolpathGenerator, Tool, ToolType
from pycam3d.gcode import GCodeWriter
from pycam3d.pipeline import CAMPipeline

__all__ = [
    "MeshProcessor",
    "ToolpathGenerator",
    "Tool",
    "ToolType",
    "GCodeWriter",
    "CAMPipeline",
]
