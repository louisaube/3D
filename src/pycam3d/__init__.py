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

Smart features:
- Machine database with limits and post-processors
- Materials database with feeds/speeds calculation
- Automatic part orientation optimization
- Accessibility and undercut detection
- Smart decision engine for automatic settings
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

# Machine and material databases
from pycam3d.machines import (
    Machine,
    MachineDatabase,
    MachineType,
    PostProcessor,
    get_machine,
    list_machines,
)
from pycam3d.materials import (
    Material,
    MaterialDatabase,
    MaterialCategory,
    FeedsSpeedsCalculator,
    CuttingParameters,
    get_material,
    list_materials,
    calculate_feeds_speeds,
)

# Stock and workholding
from pycam3d.stock import (
    Stock,
    StockManager,
    WorkholdingSetup,
    WorkholdingType,
)

# Orientation and accessibility
from pycam3d.orientation import (
    Orientation,
    AccessibilityAnalyzer,
    AccessibilityAnalysis,
    OrientationOptimizer,
    find_optimal_orientation,
    analyze_mesh_accessibility,
)

# Decision engine
from pycam3d.decision import (
    DecisionEngine,
    MachiningIntent,
    MachiningPlan,
    auto_plan,
    quick_settings,
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
    # Machines
    "Machine",
    "MachineDatabase",
    "MachineType",
    "PostProcessor",
    "get_machine",
    "list_machines",
    # Materials
    "Material",
    "MaterialDatabase",
    "MaterialCategory",
    "FeedsSpeedsCalculator",
    "CuttingParameters",
    "get_material",
    "list_materials",
    "calculate_feeds_speeds",
    # Stock
    "Stock",
    "StockManager",
    "WorkholdingSetup",
    "WorkholdingType",
    # Orientation
    "Orientation",
    "AccessibilityAnalyzer",
    "AccessibilityAnalysis",
    "OrientationOptimizer",
    "find_optimal_orientation",
    "analyze_mesh_accessibility",
    # Decision
    "DecisionEngine",
    "MachiningIntent",
    "MachiningPlan",
    "auto_plan",
    "quick_settings",
]
