"""
Automatic decision engine for PyCAM3D.

The "brain" that makes intelligent machining decisions:
- Analyzes part geometry
- Recommends optimal settings
- Warns about issues
- Generates complete machining plans
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import math

from .machines import Machine, MachineDatabase, MachineType
from .materials import Material, MaterialDatabase, FeedsSpeedsCalculator, CuttingParameters
from .stock import Stock, StockManager, WorkholdingSetup, WorkholdingType
from .orientation import OrientationOptimizer, AccessibilityAnalysis, Orientation


class MachiningIntent(Enum):
    """User's machining intent."""
    REPRODUCE_EXACT = "reproduce"           # Exact copy
    REPRODUCE_SCALED = "reproduce_scaled"   # Scaled copy
    RELIEF_CARVING = "relief"               # Bas-relief
    FULL_3D_SCULPTURE = "sculpture"         # Full 3D
    CONTOUR_CUT = "contour"                 # 2.5D contour
    POCKET = "pocket"                       # Pocket clearing
    ENGRAVE = "engrave"                     # Surface engraving


class IssueLevel(Enum):
    """Severity of detected issue."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Issue:
    """A detected issue or recommendation."""
    level: IssueLevel
    code: str
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ToolRecommendation:
    """Tool recommendation for an operation."""
    tool_type: str
    diameter: float
    flutes: int
    material: str  # HSS, Carbide, etc.
    reason: str

    def to_dict(self) -> Dict:
        return {
            "type": self.tool_type,
            "diameter": self.diameter,
            "flutes": self.flutes,
            "material": self.material,
            "reason": self.reason,
        }


@dataclass
class OperationPlan:
    """Plan for a single machining operation."""
    name: str
    operation_type: str  # roughing, finishing, contour, etc.
    strategy: str
    tool: ToolRecommendation
    params: CuttingParameters
    estimated_time_min: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.operation_type,
            "strategy": self.strategy,
            "tool": self.tool.to_dict(),
            "params": self.params.to_dict(),
            "estimated_time_min": self.estimated_time_min,
            "notes": self.notes,
        }


@dataclass
class MachiningPlan:
    """Complete machining plan with all operations."""
    operations: List[OperationPlan]
    stock: Stock
    workholding: WorkholdingType
    orientation: Orientation
    accessibility: AccessibilityAnalysis
    issues: List[Issue]
    total_time_min: float
    setup_count: int = 1

    @property
    def has_errors(self) -> bool:
        return any(i.level in [IssueLevel.ERROR, IssueLevel.CRITICAL] for i in self.issues)

    def to_dict(self) -> Dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "stock": self.stock.to_dict(),
            "workholding": self.workholding.value,
            "orientation": self.orientation.to_dict(),
            "accessibility_pct": self.accessibility.accessible_percentage,
            "issues": [{"level": i.level.value, "code": i.code, "message": i.message}
                      for i in self.issues],
            "total_time_min": self.total_time_min,
            "setup_count": self.setup_count,
        }


class DecisionEngine:
    """Automatic machining decision engine."""

    def __init__(self):
        self.machine_db = MachineDatabase()
        self.material_db = MaterialDatabase()

    def analyze_part(self, mesh, intent: MachiningIntent = MachiningIntent.REPRODUCE_EXACT
                    ) -> Dict[str, Any]:
        """Analyze a part and return recommendations.

        Args:
            mesh: Trimesh mesh object
            intent: User's machining intent

        Returns:
            Dictionary with analysis results
        """
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.face_normals)

        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        size = bounds_max - bounds_min

        # Compute mesh stats
        stats = {
            "vertices": len(vertices),
            "faces": len(faces),
            "size_mm": size.tolist(),
            "volume_mm3": float(mesh.volume) if hasattr(mesh, 'volume') else 0,
            "surface_area_mm2": float(mesh.area) if hasattr(mesh, 'area') else 0,
            "is_watertight": bool(mesh.is_watertight),
        }

        # Analyze accessibility
        optimizer = OrientationOptimizer(vertices, faces, normals)
        orientation, accessibility = optimizer.find_optimal_orientation()

        # Classify complexity
        complexity = self._classify_complexity(mesh, size, accessibility)

        return {
            "stats": stats,
            "orientation": orientation.to_dict() if orientation else None,
            "accessibility": {
                "percentage": accessibility.accessible_percentage,
                "undercut_area": accessibility.undercut_area,
                "fully_accessible": accessibility.fully_accessible,
                "critical_undercuts": len(accessibility.critical_undercuts),
            },
            "complexity": complexity,
            "intent": intent.value,
        }

    def _classify_complexity(self, mesh, size: np.ndarray,
                            accessibility: AccessibilityAnalysis) -> str:
        """Classify part complexity."""
        max_dim = max(size)
        aspect_ratio = max_dim / min(size[size > 0]) if min(size[size > 0]) > 0 else 1

        if accessibility.accessible_percentage < 50:
            return "very_complex"  # Significant undercuts
        if accessibility.accessible_percentage < 80:
            return "complex"
        if aspect_ratio > 10:
            return "complex"  # Very thin or elongated
        if len(mesh.faces) > 100000:
            return "complex"  # High detail
        if len(mesh.faces) > 10000:
            return "moderate"
        return "simple"

    def recommend_machine(self, part_size: np.ndarray,
                         material: str = "aluminum-6061") -> List[Tuple[Machine, List[Issue]]]:
        """Recommend suitable machines for part.

        Args:
            part_size: Part dimensions [X, Y, Z] in mm
            material: Material name

        Returns:
            List of (machine, issues) tuples, best first
        """
        mat = self.material_db.get(material)
        results = []

        for machine in self.machine_db.machines.values():
            issues = []

            # Check size fits
            if part_size[0] > machine.envelope.x_travel * 0.9:
                issues.append(Issue(
                    IssueLevel.WARNING,
                    "SIZE_X_TIGHT",
                    f"Part X ({part_size[0]:.0f}mm) is 90%+ of machine capacity",
                ))
            if part_size[0] > machine.envelope.x_travel:
                issues.append(Issue(
                    IssueLevel.ERROR,
                    "SIZE_X_EXCEED",
                    f"Part X ({part_size[0]:.0f}mm) exceeds machine capacity ({machine.envelope.x_travel}mm)",
                ))
                continue

            if part_size[1] > machine.envelope.y_travel * 0.9:
                issues.append(Issue(
                    IssueLevel.WARNING,
                    "SIZE_Y_TIGHT",
                    f"Part Y ({part_size[1]:.0f}mm) is 90%+ of machine capacity",
                ))
            if part_size[1] > machine.envelope.y_travel:
                issues.append(Issue(
                    IssueLevel.ERROR,
                    "SIZE_Y_EXCEED",
                    f"Part Y ({part_size[1]:.0f}mm) exceeds machine capacity ({machine.envelope.y_travel}mm)",
                ))
                continue

            if part_size[2] > machine.envelope.z_travel * 0.9:
                issues.append(Issue(
                    IssueLevel.WARNING,
                    "SIZE_Z_TIGHT",
                    f"Part Z ({part_size[2]:.0f}mm) is 90%+ of machine capacity",
                ))

            # Check material suitability
            if mat and mat.use_coolant and machine.machine_type == MachineType.ROUTER_3AXIS:
                issues.append(Issue(
                    IssueLevel.WARNING,
                    "COOLANT_NEEDED",
                    f"Material {mat.name} typically requires coolant",
                    suggestion="Consider flood coolant or mist system",
                ))

            results.append((machine, issues))

        # Sort by least issues, then by envelope size (smaller = more rigid)
        results.sort(key=lambda x: (
            sum(1 for i in x[1] if i.level == IssueLevel.ERROR),
            sum(1 for i in x[1] if i.level == IssueLevel.WARNING),
            x[0].envelope.x_travel * x[0].envelope.y_travel,
        ))

        return results

    def recommend_tools(self, part_size: np.ndarray, detail_level: float,
                       material_name: str) -> Dict[str, ToolRecommendation]:
        """Recommend tools for roughing and finishing.

        Args:
            part_size: Part dimensions
            detail_level: Smallest feature size in mm
            material_name: Material name

        Returns:
            Dictionary of tool recommendations by operation
        """
        mat = self.material_db.get(material_name)
        min_dim = min(part_size[:2])  # Smallest XY dimension

        tools = {}

        # Roughing tool - larger for efficiency
        rough_dia = min(min_dim * 0.3, 12.0)  # Max 12mm or 30% of part
        rough_dia = max(rough_dia, 6.0)  # Min 6mm
        rough_dia = round(rough_dia * 2) / 2  # Round to 0.5mm

        tools["roughing"] = ToolRecommendation(
            tool_type="flat",
            diameter=rough_dia,
            flutes=2 if mat and mat.chip_load_factor > 0.8 else 3,
            material="Carbide" if mat and mat.category.value.startswith("Steel") else "HSS",
            reason=f"Efficient material removal at {rough_dia}mm diameter",
        )

        # Finishing tool - sized for detail
        finish_dia = min(detail_level * 2, 6.0)  # 2x smallest feature, max 6mm
        finish_dia = max(finish_dia, 1.0)  # Min 1mm
        finish_dia = round(finish_dia * 2) / 2

        tools["finishing"] = ToolRecommendation(
            tool_type="ball",
            diameter=finish_dia,
            flutes=2,
            material="Carbide",
            reason=f"Detail resolution at {finish_dia}mm for {detail_level}mm features",
        )

        # Contour/profile tool if needed
        tools["contour"] = ToolRecommendation(
            tool_type="flat",
            diameter=max(3.0, detail_level),
            flutes=2,
            material="Carbide",
            reason="Profile cutting with good chip evacuation",
        )

        return tools

    def generate_plan(
        self,
        mesh,
        machine_name: str,
        material_name: str,
        intent: MachiningIntent = MachiningIntent.REPRODUCE_EXACT,
        workholding: WorkholdingType = WorkholdingType.CLAMPS,
    ) -> MachiningPlan:
        """Generate complete machining plan.

        Args:
            mesh: Trimesh mesh object
            machine_name: Name of machine from database
            material_name: Name of material from database
            intent: Machining intent
            workholding: Workholding method

        Returns:
            Complete MachiningPlan
        """
        machine = self.machine_db.get(machine_name)
        material = self.material_db.get(material_name)
        issues = []

        if not machine:
            issues.append(Issue(
                IssueLevel.CRITICAL,
                "MACHINE_NOT_FOUND",
                f"Machine '{machine_name}' not found in database",
            ))
        if not material:
            issues.append(Issue(
                IssueLevel.WARNING,
                "MATERIAL_NOT_FOUND",
                f"Material '{material_name}' not found, using defaults",
            ))

        # Analyze part
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.face_normals)

        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        part_size = bounds_max - bounds_min

        # Find optimal orientation
        optimizer = OrientationOptimizer(vertices, faces, normals)
        orientation, accessibility = optimizer.find_optimal_orientation()

        # Handle case where orientation optimization fails
        if accessibility is None:
            from .orientation import AccessibilityAnalyzer
            analyzer = AccessibilityAnalyzer(vertices, faces, normals)
            accessibility = analyzer.analyze()
        if orientation is None:
            orientation = Orientation()

        # Check accessibility
        if accessibility.accessible_percentage < 80:
            issues.append(Issue(
                IssueLevel.WARNING,
                "LOW_ACCESSIBILITY",
                f"Only {accessibility.accessible_percentage:.1f}% of surface is accessible",
                suggestion="Consider multi-setup machining or 4-axis",
            ))
        if accessibility.accessible_percentage < 50:
            issues.append(Issue(
                IssueLevel.ERROR,
                "POOR_ACCESSIBILITY",
                f"Only {accessibility.accessible_percentage:.1f}% accessible - significant undercuts",
                suggestion="Part may require redesign or 5-axis machining",
            ))

        # Create stock
        stock = Stock.from_part_bounds(bounds_min, bounds_max, allowance=5.0,
                                       material=material_name)

        # Check machine envelope
        if machine:
            if stock.length > machine.envelope.x_travel:
                issues.append(Issue(
                    IssueLevel.ERROR,
                    "STOCK_TOO_LARGE_X",
                    f"Stock X ({stock.length:.0f}mm) exceeds machine ({machine.envelope.x_travel}mm)",
                ))
            if stock.width > machine.envelope.y_travel:
                issues.append(Issue(
                    IssueLevel.ERROR,
                    "STOCK_TOO_LARGE_Y",
                    f"Stock Y ({stock.width:.0f}mm) exceeds machine ({machine.envelope.y_travel}mm)",
                ))

        # Recommend tools
        detail_level = 1.0  # Estimate smallest feature
        tool_recs = self.recommend_tools(part_size, detail_level, material_name)

        # Generate operations
        operations = []
        total_time = 0.0

        # Calculate feeds/speeds
        if material:
            calc = FeedsSpeedsCalculator(material)
            max_rpm = machine.spindle.max_rpm if machine else 24000
            max_feed = machine.feeds.max_feed_xy if machine else 5000

            # Roughing operation
            rough_tool = tool_recs["roughing"]
            rough_params = calc.calculate(
                tool_diameter=rough_tool.diameter,
                flutes=rough_tool.flutes,
                roughing=True,
                max_rpm=max_rpm,
                max_feed=max_feed,
            )

            # Estimate roughing time (volume / MRR)
            mrr = rough_params.depth_of_cut * rough_params.stepover * rough_params.feed_rate / 60
            stock_volume = stock.length * stock.width * stock.height
            rough_time = (stock_volume / mrr / 60) if mrr > 0 else 30  # minutes

            operations.append(OperationPlan(
                name="Roughing",
                operation_type="roughing",
                strategy="parallel" if intent != MachiningIntent.POCKET else "pocket",
                tool=rough_tool,
                params=rough_params,
                estimated_time_min=rough_time,
                notes=[f"Remove bulk material, leave {material.get_doc(rough_tool.diameter):.1f}mm stock"],
            ))
            total_time += rough_time

            # Finishing operation
            finish_tool = tool_recs["finishing"]
            finish_params = calc.calculate(
                tool_diameter=finish_tool.diameter,
                flutes=finish_tool.flutes,
                roughing=False,
                max_rpm=max_rpm,
                max_feed=max_feed,
            )

            # Estimate finishing time (surface area / stepover / feed)
            surface_area = float(mesh.area) if hasattr(mesh, 'area') else part_size[0] * part_size[1]
            path_length = surface_area / finish_params.stepover
            finish_time = (path_length / finish_params.feed_rate) if finish_params.feed_rate > 0 else 60

            # Choose strategy based on intent
            if intent == MachiningIntent.RELIEF_CARVING:
                strategy = "parallel"
            elif intent == MachiningIntent.FULL_3D_SCULPTURE:
                strategy = "iso-scallop"
            else:
                strategy = "spiral" if accessibility.accessible_percentage > 90 else "parallel"

            operations.append(OperationPlan(
                name="Finishing",
                operation_type="finishing",
                strategy=strategy,
                tool=finish_tool,
                params=finish_params,
                estimated_time_min=finish_time,
                notes=[f"Final surface quality with {finish_tool.diameter}mm ball nose"],
            ))
            total_time += finish_time

        else:
            # Default operations without material data
            operations.append(OperationPlan(
                name="Roughing",
                operation_type="roughing",
                strategy="parallel",
                tool=tool_recs["roughing"],
                params=CuttingParameters(
                    rpm=12000, feed_rate=1000, depth_of_cut=3.0,
                    stepover=3.0, plunge_rate=300, use_coolant=False,
                    use_air_blast=True, climb_milling=True, chip_load=0.05,
                ),
                estimated_time_min=30.0,
            ))
            operations.append(OperationPlan(
                name="Finishing",
                operation_type="finishing",
                strategy="iso-scallop",
                tool=tool_recs["finishing"],
                params=CuttingParameters(
                    rpm=18000, feed_rate=600, depth_of_cut=0.3,
                    stepover=0.3, plunge_rate=200, use_coolant=False,
                    use_air_blast=True, climb_milling=True, chip_load=0.02,
                ),
                estimated_time_min=60.0,
            ))
            total_time = 90.0

        # Time warnings
        if total_time > 60:
            issues.append(Issue(
                IssueLevel.INFO,
                "LONG_JOB",
                f"Estimated machining time: {total_time:.0f} minutes ({total_time/60:.1f} hours)",
            ))
        if total_time > 480:  # 8 hours
            issues.append(Issue(
                IssueLevel.WARNING,
                "VERY_LONG_JOB",
                f"Job will take {total_time/60:.1f} hours - consider overnight run",
            ))

        return MachiningPlan(
            operations=operations,
            stock=stock,
            workholding=workholding,
            orientation=orientation,
            accessibility=accessibility,
            issues=issues,
            total_time_min=total_time,
            setup_count=1 if accessibility.fully_accessible else 2,
        )

    def quick_recommend(self, mesh, material_name: str = "plywood") -> Dict[str, Any]:
        """Quick one-shot recommendation for simple jobs.

        Args:
            mesh: Trimesh mesh
            material_name: Material name

        Returns:
            Dictionary with quick recommendations
        """
        vertices = np.array(mesh.vertices)
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        part_size = bounds_max - bounds_min

        material = self.material_db.get(material_name)

        # Simple tool selection
        tool_dia = min(max(min(part_size[:2]) * 0.1, 3.0), 6.0)
        tool_dia = round(tool_dia * 2) / 2

        if material:
            calc = FeedsSpeedsCalculator(material)
            params = calc.calculate(tool_diameter=tool_dia, flutes=2, roughing=False)

            return {
                "tool_diameter_mm": tool_dia,
                "tool_type": "ball",
                "rpm": params.rpm,
                "feed_rate_mm_min": params.feed_rate,
                "stepover_mm": params.stepover,
                "depth_of_cut_mm": params.depth_of_cut,
                "strategy": "parallel",
                "coolant": params.use_coolant,
            }

        # Defaults for unknown material
        return {
            "tool_diameter_mm": tool_dia,
            "tool_type": "ball",
            "rpm": 18000,
            "feed_rate_mm_min": 1000,
            "stepover_mm": tool_dia * 0.15,
            "depth_of_cut_mm": tool_dia * 0.3,
            "strategy": "parallel",
            "coolant": False,
        }


# Convenience functions
def auto_plan(mesh, machine: str = "shapeoko-4", material: str = "plywood") -> MachiningPlan:
    """Generate automatic machining plan.

    Args:
        mesh: Trimesh mesh
        machine: Machine name from database
        material: Material name from database

    Returns:
        Complete MachiningPlan
    """
    engine = DecisionEngine()
    return engine.generate_plan(
        mesh=mesh,
        machine_name=machine,
        material_name=material,
    )


def quick_settings(mesh, material: str = "plywood") -> Dict[str, Any]:
    """Get quick cutting settings.

    Args:
        mesh: Trimesh mesh
        material: Material name

    Returns:
        Dictionary of recommended settings
    """
    engine = DecisionEngine()
    return engine.quick_recommend(mesh, material)
