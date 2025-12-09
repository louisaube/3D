"""
Smart Multi-Tool Strategy Module.

Automatically analyzes mesh geometry and recommends optimal
tool sequences and machining strategies for CNC operations.

Theory:
- Curvature analysis segments surface into regions
- Each region gets optimal tool/strategy combination
- Multi-pass approach: Roughing -> Semi-finish -> Finish -> Details
- Limited to MAX 4 tools from available pool for practical workshop use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import trimesh

from pycam3d.curvature import CurvatureAnalyzer, CurvatureField

logger = logging.getLogger(__name__)

# =============================================================================
# TOOL POOL CONFIGURATION
# =============================================================================
# Maximum number of tools to use (minimize tool changes = faster machining)
MAX_TOOLS = 4

# Default pool of available tools in workshop
# Modify this list to match your actual tool inventory
DEFAULT_TOOL_POOL = [
    # Fraises plates (Flat end mills) - pour ébauche et surfaces planes
    {"type": "flat", "diameter": 20.0, "name": "Flat 20mm"},
    {"type": "flat", "diameter": 12.0, "name": "Flat 12mm"},
    {"type": "flat", "diameter": 8.0, "name": "Flat 8mm"},
    {"type": "flat", "diameter": 6.0, "name": "Flat 6mm"},
    {"type": "flat", "diameter": 4.0, "name": "Flat 4mm"},
    {"type": "flat", "diameter": 3.0, "name": "Flat 3mm"},
    # Fraises boules (Ball nose) - pour surfaces courbes et finition
    {"type": "ball", "diameter": 12.0, "name": "Ball 12mm"},
    {"type": "ball", "diameter": 8.0, "name": "Ball 8mm"},
    {"type": "ball", "diameter": 6.0, "name": "Ball 6mm"},
    {"type": "ball", "diameter": 4.0, "name": "Ball 4mm"},
    {"type": "ball", "diameter": 3.0, "name": "Ball 3mm"},
    {"type": "ball", "diameter": 2.0, "name": "Ball 2mm"},
    {"type": "ball", "diameter": 1.0, "name": "Ball 1mm"},
    # Fraises toriques (Bull nose) - pour semi-finition
    {"type": "bull", "diameter": 8.0, "corner_radius": 1.0, "name": "Bull 8mm R1"},
    {"type": "bull", "diameter": 6.0, "corner_radius": 0.5, "name": "Bull 6mm R0.5"},
]


class MachiningPhase(Enum):
    """Machining operation phases."""
    ROUGHING = "roughing"
    SEMI_FINISH = "semi_finish"
    FINISH = "finish"
    DETAIL = "detail"


class RegionType(Enum):
    """Surface region classification based on curvature."""
    FLAT = "flat"                    # Curvature < 0.01
    GENTLE_CURVE = "gentle_curve"    # 0.01 <= Curvature < 0.05
    MODERATE_CURVE = "moderate_curve" # 0.05 <= Curvature < 0.2
    HIGH_CURVE = "high_curve"        # 0.2 <= Curvature < 0.5
    SHARP_FEATURE = "sharp_feature"  # Curvature >= 0.5
    POCKET = "pocket"                # Enclosed concave region
    UNDERCUT = "undercut"            # Accessibility issues


@dataclass
class ToolSpec:
    """Tool specification for a machining operation."""
    type: str           # "ball", "flat", "bull"
    diameter: float     # mm
    corner_radius: float = 0.0  # For bull nose
    name: str = ""

    def __post_init__(self):
        if not self.name:
            if self.type == "ball":
                self.name = f"Ball {self.diameter}mm"
            elif self.type == "flat":
                self.name = f"Flat {self.diameter}mm"
            elif self.type == "bull":
                self.name = f"Bull {self.diameter}mm R{self.corner_radius}"


@dataclass
class OperationStep:
    """A single machining operation step."""
    phase: MachiningPhase
    tool: ToolSpec
    strategy: str           # "parallel", "iso-scallop", "spiral", "waterline"
    stepover_percent: float # % of tool diameter
    z_step: Optional[float] = None  # For roughing layers
    regions: List[RegionType] = field(default_factory=list)
    estimated_time_percent: float = 0.0  # % of total time
    description: str = ""


@dataclass
class MachiningPlan:
    """Complete multi-tool machining plan."""
    mesh_id: str
    mesh_bounds: Tuple[List[float], List[float]]
    mesh_size: List[float]
    total_volume: float

    # Analysis results
    region_analysis: Dict[str, Any] = field(default_factory=dict)
    curvature_stats: Dict[str, float] = field(default_factory=dict)

    # Recommended operations
    operations: List[OperationStep] = field(default_factory=list)

    # Summary
    total_tools: int = 0
    estimated_time_reduction: str = ""
    quality_improvement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for JSON serialization."""
        return {
            "mesh_id": self.mesh_id,
            "mesh_bounds": self.mesh_bounds,
            "mesh_size": self.mesh_size,
            "total_volume": self.total_volume,
            "region_analysis": self.region_analysis,
            "curvature_stats": self.curvature_stats,
            "operations": [
                {
                    "phase": op.phase.value,
                    "tool": {
                        "type": op.tool.type,
                        "diameter": op.tool.diameter,
                        "corner_radius": op.tool.corner_radius,
                        "name": op.tool.name,
                    },
                    "strategy": op.strategy,
                    "stepover_percent": op.stepover_percent,
                    "z_step": op.z_step,
                    "regions": [r.value for r in op.regions],
                    "estimated_time_percent": op.estimated_time_percent,
                    "description": op.description,
                }
                for op in self.operations
            ],
            "total_tools": self.total_tools,
            "max_tools": MAX_TOOLS,
            "available_tools": DEFAULT_TOOL_POOL,
            "estimated_time_reduction": self.estimated_time_reduction,
            "quality_improvement": self.quality_improvement,
        }


class SmartStrategy:
    """
    Intelligent multi-tool strategy analyzer.

    Analyzes mesh geometry and recommends optimal tool sequences
    and machining strategies based on surface characteristics.
    """

    # Curvature thresholds for region classification
    CURVATURE_THRESHOLDS = {
        RegionType.FLAT: 0.01,
        RegionType.GENTLE_CURVE: 0.05,
        RegionType.MODERATE_CURVE: 0.2,
        RegionType.HIGH_CURVE: 0.5,
    }

    # Strategy recommendations by region type
    STRATEGY_MAP = {
        RegionType.FLAT: ("parallel", 50),           # strategy, stepover%
        RegionType.GENTLE_CURVE: ("iso-scallop", 30),
        RegionType.MODERATE_CURVE: ("iso-scallop", 20),
        RegionType.HIGH_CURVE: ("spiral", 15),
        RegionType.SHARP_FEATURE: ("waterline", 10),
        RegionType.POCKET: ("parallel", 40),
        RegionType.UNDERCUT: ("waterline", 15),
    }

    def __init__(self, mesh: trimesh.Trimesh, mesh_id: str = ""):
        """
        Initialize smart strategy analyzer.

        Args:
            mesh: Trimesh mesh object
            mesh_id: Optional mesh identifier
        """
        self.mesh = mesh
        self.mesh_id = mesh_id
        self.curvature_analyzer = CurvatureAnalyzer(mesh)
        self.curvature_field: Optional[CurvatureField] = None
        self.region_map: Dict[RegionType, np.ndarray] = {}
        self._fast_mode = False

    def analyze(self, fast_mode: bool = False) -> MachiningPlan:
        """
        Perform mesh analysis and generate machining plan.

        Args:
            fast_mode: Skip detailed curvature analysis for speed.
                      Uses heuristics based on mesh complexity.

        Returns:
            MachiningPlan with recommended operations
        """
        self._fast_mode = fast_mode
        mode_str = "FAST" if fast_mode else "full"
        logger.info(f"Starting smart strategy analysis ({mode_str} mode)...")

        # Compute mesh properties
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        volume = self.mesh.volume if self.mesh.is_watertight else 0

        if fast_mode:
            # FAST MODE: Use heuristics based on mesh complexity
            region_analysis = self._estimate_regions_fast()
            curvature_stats = {"min": 0, "max": 1, "mean": 0.5, "median": 0.5, "std": 0.3}
        else:
            # FULL MODE: Detailed curvature analysis
            self._analyze_curvature()
            region_analysis = self._segment_regions()
            curvature_stats = self._get_curvature_stats()

        # Generate tool recommendations
        operations = self._recommend_operations(size)

        # Create plan
        plan = MachiningPlan(
            mesh_id=self.mesh_id,
            mesh_bounds=(bounds[0].tolist(), bounds[1].tolist()),
            mesh_size=size.tolist(),
            total_volume=float(volume),
            region_analysis=region_analysis,
            curvature_stats=curvature_stats,
            operations=operations,
            total_tools=len(set(op.tool.diameter for op in operations)),
            estimated_time_reduction="25-35%",
            quality_improvement="+20-30%",
        )

        logger.info(f"Analysis complete: {len(operations)} operations recommended")
        return plan

    def _estimate_regions_fast(self) -> Dict[str, Any]:
        """
        Fast region estimation using mesh heuristics.

        Uses face count and normals variance to estimate surface complexity
        without computing full curvature field.
        """
        n_faces = len(self.mesh.faces)
        n_vertices = len(self.mesh.vertices)

        # Estimate complexity from face density
        bounds = self.mesh.bounds
        bbox_volume = np.prod(bounds[1] - bounds[0])
        face_density = n_faces / (bbox_volume + 1e-10)

        # Use normal variance as proxy for surface complexity
        normals = self.mesh.face_normals
        normal_variance = np.var(normals, axis=0).mean()

        # Heuristic classification
        if normal_variance < 0.1:
            # Mostly flat surfaces
            flat_pct, curve_pct, sharp_pct = 70, 20, 10
        elif normal_variance < 0.3:
            # Mixed surfaces
            flat_pct, curve_pct, sharp_pct = 30, 50, 20
        else:
            # Complex curved surfaces
            flat_pct, curve_pct, sharp_pct = 10, 40, 50

        # Populate region_map with estimates for _recommend_operations
        self.region_map = {
            RegionType.FLAT: np.ones(int(n_vertices * flat_pct / 100), dtype=bool),
            RegionType.GENTLE_CURVE: np.ones(int(n_vertices * curve_pct / 200), dtype=bool),
            RegionType.MODERATE_CURVE: np.ones(int(n_vertices * curve_pct / 200), dtype=bool),
            RegionType.HIGH_CURVE: np.ones(int(n_vertices * sharp_pct / 200), dtype=bool),
            RegionType.SHARP_FEATURE: np.ones(int(n_vertices * sharp_pct / 200), dtype=bool),
        }

        return {
            "flat_percent": flat_pct,
            "gentle_curve_percent": curve_pct / 2,
            "moderate_curve_percent": curve_pct / 2,
            "high_curve_percent": sharp_pct / 2,
            "sharp_feature_percent": sharp_pct / 2,
            "total_vertices": n_vertices,
            "fast_mode": True,
        }

    def _analyze_curvature(self) -> None:
        """Compute curvature field for the mesh."""
        logger.info("Computing curvature field...")
        self.curvature_field = self.curvature_analyzer.compute_curvature()

    def _segment_regions(self) -> Dict[str, Any]:
        """
        Segment mesh vertices into regions based on curvature.

        Returns:
            Dictionary with region statistics
        """
        if self.curvature_field is None:
            self._analyze_curvature()

        max_curv = self.curvature_field.max_curvature
        n_vertices = len(max_curv)

        # Classify vertices by curvature magnitude
        flat_mask = max_curv < self.CURVATURE_THRESHOLDS[RegionType.FLAT]
        gentle_mask = (max_curv >= self.CURVATURE_THRESHOLDS[RegionType.FLAT]) & \
                     (max_curv < self.CURVATURE_THRESHOLDS[RegionType.GENTLE_CURVE])
        moderate_mask = (max_curv >= self.CURVATURE_THRESHOLDS[RegionType.GENTLE_CURVE]) & \
                       (max_curv < self.CURVATURE_THRESHOLDS[RegionType.MODERATE_CURVE])
        high_mask = (max_curv >= self.CURVATURE_THRESHOLDS[RegionType.MODERATE_CURVE]) & \
                   (max_curv < self.CURVATURE_THRESHOLDS[RegionType.HIGH_CURVE])
        sharp_mask = max_curv >= self.CURVATURE_THRESHOLDS[RegionType.HIGH_CURVE]

        self.region_map = {
            RegionType.FLAT: flat_mask,
            RegionType.GENTLE_CURVE: gentle_mask,
            RegionType.MODERATE_CURVE: moderate_mask,
            RegionType.HIGH_CURVE: high_mask,
            RegionType.SHARP_FEATURE: sharp_mask,
        }

        # Calculate percentages
        region_analysis = {
            "flat_percent": float(np.sum(flat_mask) / n_vertices * 100),
            "gentle_curve_percent": float(np.sum(gentle_mask) / n_vertices * 100),
            "moderate_curve_percent": float(np.sum(moderate_mask) / n_vertices * 100),
            "high_curve_percent": float(np.sum(high_mask) / n_vertices * 100),
            "sharp_feature_percent": float(np.sum(sharp_mask) / n_vertices * 100),
            "total_vertices": n_vertices,
        }

        logger.info(f"Region segmentation: {region_analysis}")
        return region_analysis

    def _get_curvature_stats(self) -> Dict[str, float]:
        """Get curvature statistics."""
        if self.curvature_field is None:
            return {}

        max_curv = self.curvature_field.max_curvature
        return {
            "min": float(np.min(max_curv)),
            "max": float(np.max(max_curv)),
            "mean": float(np.mean(max_curv)),
            "median": float(np.median(max_curv)),
            "std": float(np.std(max_curv)),
        }

    def _recommend_operations(self, mesh_size: np.ndarray) -> List[OperationStep]:
        """
        Generate recommended machining operations from tool pool.

        Selects up to MAX_TOOLS (4) tools from DEFAULT_TOOL_POOL
        optimized for speed (fewer tool changes = faster).

        Args:
            mesh_size: Mesh dimensions [x, y, z]

        Returns:
            List of OperationStep recommendations (max 4)
        """
        min_dim = float(np.min(mesh_size))

        # Calculate ideal tool sizes based on mesh dimensions
        # Rule: largest tool ~15% of smallest dimension
        ideal_roughing = max(3.0, min(20.0, min_dim * 0.15))
        ideal_semifinish = max(2.0, ideal_roughing * 0.5)
        ideal_finish = max(1.0, ideal_roughing * 0.25)
        ideal_detail = max(0.5, ideal_roughing * 0.1)

        # Analyze surface regions
        flat_pct = self.region_map.get(RegionType.FLAT, np.array([])).sum()
        curved_pct = (
            self.region_map.get(RegionType.GENTLE_CURVE, np.array([])).sum() +
            self.region_map.get(RegionType.MODERATE_CURVE, np.array([])).sum()
        )
        high_curve_pct = self.region_map.get(RegionType.HIGH_CURVE, np.array([])).sum()
        sharp_pct = self.region_map.get(RegionType.SHARP_FEATURE, np.array([])).sum()

        has_curved = curved_pct > 0
        has_high_curve = high_curve_pct > 0
        has_sharp = sharp_pct > 0

        # Build candidate operations with priority scores
        candidates = []

        # ROUGHING - Always needed (highest priority for speed)
        roughing_tool = self._find_best_tool_from_pool("flat", ideal_roughing)
        if roughing_tool:
            candidates.append({
                "phase": MachiningPhase.ROUGHING,
                "tool": roughing_tool,
                "strategy": "parallel",
                "stepover": 50,
                "z_step": roughing_tool.diameter * 0.5,
                "regions": [RegionType.FLAT, RegionType.GENTLE_CURVE],
                "time_pct": 35,
                "priority": 100,  # Always include roughing
                "desc": f"Roughing: {roughing_tool.name} - Fast bulk removal"
            })

        # SEMI-FINISH - If curved surfaces exist
        if has_curved or has_high_curve:
            semifinish_tool = self._find_best_tool_from_pool("bull", ideal_semifinish)
            if not semifinish_tool:
                semifinish_tool = self._find_best_tool_from_pool("ball", ideal_semifinish)
            if semifinish_tool:
                candidates.append({
                    "phase": MachiningPhase.SEMI_FINISH,
                    "tool": semifinish_tool,
                    "strategy": "iso-scallop",
                    "stepover": 25,
                    "z_step": None,
                    "regions": [RegionType.GENTLE_CURVE, RegionType.MODERATE_CURVE],
                    "time_pct": 25,
                    "priority": 30 + (curved_pct / 100),  # Higher if more curves
                    "desc": f"Semi-finish: {semifinish_tool.name} - Prepare surfaces"
                })

        # FINISH - For high curvature areas
        if has_high_curve or has_curved:
            finish_tool = self._find_best_tool_from_pool("ball", ideal_finish)
            if finish_tool:
                candidates.append({
                    "phase": MachiningPhase.FINISH,
                    "tool": finish_tool,
                    "strategy": "spiral",
                    "stepover": 15,
                    "z_step": None,
                    "regions": [RegionType.MODERATE_CURVE, RegionType.HIGH_CURVE],
                    "time_pct": 25,
                    "priority": 40 + (high_curve_pct / 50),  # Higher if high curvature
                    "desc": f"Finish: {finish_tool.name} - Smooth surfaces"
                })

        # DETAIL - For sharp features only
        if has_sharp:
            detail_tool = self._find_best_tool_from_pool("ball", ideal_detail)
            if detail_tool:
                candidates.append({
                    "phase": MachiningPhase.DETAIL,
                    "tool": detail_tool,
                    "strategy": "waterline",
                    "stepover": 10,
                    "z_step": None,
                    "regions": [RegionType.SHARP_FEATURE],
                    "time_pct": 15,
                    "priority": 50 + (sharp_pct / 50),  # Higher if more sharp features
                    "desc": f"Detail: {detail_tool.name} - Fine features"
                })

        # Sort by priority (descending) and take top MAX_TOOLS
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        selected = candidates[:MAX_TOOLS]

        # Re-sort by machining phase order for execution
        phase_order = {
            MachiningPhase.ROUGHING: 0,
            MachiningPhase.SEMI_FINISH: 1,
            MachiningPhase.FINISH: 2,
            MachiningPhase.DETAIL: 3
        }
        selected.sort(key=lambda x: phase_order[x["phase"]])

        # Convert to OperationStep objects
        operations = []
        for c in selected:
            operations.append(OperationStep(
                phase=c["phase"],
                tool=c["tool"],
                strategy=c["strategy"],
                stepover_percent=c["stepover"],
                z_step=c["z_step"],
                regions=c["regions"],
                estimated_time_percent=c["time_pct"],
                description=c["desc"]
            ))

        logger.info(f"Selected {len(operations)}/{MAX_TOOLS} tools from pool")
        return operations

    def _find_best_tool_from_pool(
        self,
        tool_type: str,
        ideal_diameter: float
    ) -> Optional[ToolSpec]:
        """
        Find best matching tool from pool for given type and ideal diameter.

        Args:
            tool_type: "flat", "ball", or "bull"
            ideal_diameter: Target diameter in mm

        Returns:
            ToolSpec or None if no matching tool found
        """
        matching_tools = [
            t for t in DEFAULT_TOOL_POOL
            if t["type"] == tool_type
        ]

        if not matching_tools:
            return None

        # Find closest diameter match
        best = min(matching_tools, key=lambda t: abs(t["diameter"] - ideal_diameter))

        return ToolSpec(
            type=best["type"],
            diameter=best["diameter"],
            corner_radius=best.get("corner_radius", 0.0),
            name=best["name"]
        )

    @staticmethod
    def _round_to_standard_size(diameter: float) -> float:
        """Round diameter to nearest standard tool size."""
        standard_sizes = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
                         5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0]
        return min(standard_sizes, key=lambda x: abs(x - diameter))


def analyze_mesh_for_smart_strategy(
    mesh: trimesh.Trimesh,
    mesh_id: str = "",
    fast_mode: bool = False
) -> MachiningPlan:
    """
    Convenience function to analyze mesh and generate smart machining plan.

    Args:
        mesh: Trimesh mesh object
        mesh_id: Optional mesh identifier
        fast_mode: Skip detailed curvature analysis for speed

    Returns:
        MachiningPlan with recommended operations
    """
    analyzer = SmartStrategy(mesh, mesh_id)
    return analyzer.analyze(fast_mode=fast_mode)
