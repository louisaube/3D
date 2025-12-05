"""
Tool Library for PyCAM3D.

Provides:
- Comprehensive tool database with geometries
- Tool materials and coatings
- Tool recommendations by operation
- Tool life estimation
- Tool holder/collet management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
import math


class ToolShape(Enum):
    """Cutter geometry shape."""
    FLAT_ENDMILL = "flat_endmill"      # Standard flat end
    BALL_ENDMILL = "ball_endmill"      # Ball nose / spherical
    BULL_ENDMILL = "bull_endmill"      # Corner radius / bull nose
    CHAMFER = "chamfer"                # V-bit / chamfer mill
    DRILL = "drill"                    # Twist drill
    SPOT_DRILL = "spot_drill"          # Spot/center drill
    THREAD_MILL = "thread_mill"        # Thread milling
    FACE_MILL = "face_mill"            # Face mill / fly cutter
    SLOT_DRILL = "slot_drill"          # 2-flute slot drill
    ROUGHING = "roughing"              # Roughing endmill (corncob)
    TAPERED = "tapered"                # Tapered endmill
    LOLLIPOP = "lollipop"              # Undercutting tool
    DOVETAIL = "dovetail"              # Dovetail cutter
    WOODRUFF = "woodruff"              # Woodruff key cutter
    ENGRAVING = "engraving"            # Fine engraving bit


class ToolMaterial(Enum):
    """Tool substrate material."""
    HSS = "hss"                        # High Speed Steel
    COBALT_HSS = "cobalt_hss"          # Cobalt HSS (M35, M42)
    CARBIDE = "carbide"                # Solid carbide
    CARBIDE_TIPPED = "carbide_tipped"  # Carbide tipped
    CERAMIC = "ceramic"                # Ceramic insert
    CBN = "cbn"                        # Cubic Boron Nitride
    PCD = "pcd"                        # Polycrystalline Diamond


class ToolCoating(Enum):
    """Tool coating type."""
    NONE = "uncoated"
    TIN = "tin"                        # Titanium Nitride (gold)
    TICN = "ticn"                      # Titanium Carbonitride
    TIALN = "tialn"                    # Titanium Aluminum Nitride
    ALTIN = "altin"                    # Aluminum Titanium Nitride
    ZRN = "zrn"                        # Zirconium Nitride
    DLC = "dlc"                        # Diamond-Like Carbon
    DIAMOND = "diamond"                # Diamond coating


class ToolHolderType(Enum):
    """Tool holder / collet type."""
    ER11 = "er11"
    ER16 = "er16"
    ER20 = "er20"
    ER25 = "er25"
    ER32 = "er32"
    ER40 = "er40"
    R8 = "r8"
    CAT40 = "cat40"
    BT30 = "bt30"
    BT40 = "bt40"
    HSK_A63 = "hsk_a63"
    ROUTER_COLLET = "router_collet"


@dataclass
class ToolGeometry:
    """Detailed tool geometry specification."""
    # Core dimensions (mm)
    diameter: float
    flute_length: float              # Cutting length
    overall_length: float
    shank_diameter: float

    # Cutting geometry
    flutes: int = 2
    helix_angle: float = 30.0        # degrees
    rake_angle: float = 10.0         # degrees (positive = sharper)
    clearance_angle: float = 10.0    # degrees

    # Special geometry
    corner_radius: float = 0.0       # For bull nose
    tip_angle: float = 0.0           # For drills/chamfers (full angle)
    taper_angle: float = 0.0         # For tapered tools
    neck_diameter: float = 0.0       # For lollipop/undercut
    neck_length: float = 0.0

    # Chip breaker
    chip_breaker: bool = False
    variable_helix: bool = False

    @property
    def aspect_ratio(self) -> float:
        """Flute length to diameter ratio."""
        return self.flute_length / self.diameter if self.diameter > 0 else 0

    @property
    def effective_diameter(self) -> float:
        """Effective cutting diameter at tip."""
        if self.tip_angle > 0:  # Drill or chamfer
            return self.diameter
        return self.diameter

    def get_corner_profile(self) -> str:
        """Describe the corner cutting profile."""
        if self.corner_radius > 0:
            return f"Bull nose R{self.corner_radius}"
        elif self.tip_angle == 180:
            return "Ball nose"
        elif self.tip_angle > 0:
            return f"Point {self.tip_angle}°"
        return "Square"


@dataclass
class ToolSpec:
    """Complete tool specification."""
    # Identity
    tool_id: str
    name: str
    description: str

    # Classification
    shape: ToolShape
    material: ToolMaterial
    coating: ToolCoating

    # Geometry
    geometry: ToolGeometry

    # Holder
    holder_type: ToolHolderType = ToolHolderType.ER20
    stick_out: float = 25.0          # mm protruding from holder

    # Performance data
    max_rpm: int = 24000
    max_doc: float = 0.0             # Max depth of cut (0 = use defaults)
    max_woc: float = 0.0             # Max width of cut (0 = use defaults)

    # Recommended operations
    recommended_for: List[str] = field(default_factory=list)

    # Tool life
    tool_life_minutes: int = 120     # Estimated life in cutting minutes
    regrind_count: int = 0           # Times reground

    # Cost
    cost_usd: float = 0.0

    # Notes
    notes: str = ""

    @property
    def diameter(self) -> float:
        """Shorthand for geometry diameter."""
        return self.geometry.diameter

    @property
    def flutes(self) -> int:
        """Shorthand for flute count."""
        return self.geometry.flutes

    def get_default_doc(self) -> float:
        """Get recommended depth of cut."""
        if self.max_doc > 0:
            return self.max_doc
        # Conservative defaults based on tool type
        d = self.geometry.diameter
        if self.shape == ToolShape.ROUGHING:
            return d * 1.0  # Aggressive
        elif self.shape == ToolShape.BALL_ENDMILL:
            return d * 0.1  # Light
        elif self.shape in [ToolShape.DRILL, ToolShape.SPOT_DRILL]:
            return d * 2.0  # Peck drilling depth
        return d * 0.5  # Standard endmill

    def get_default_woc(self) -> float:
        """Get recommended width of cut (stepover)."""
        if self.max_woc > 0:
            return self.max_woc
        d = self.geometry.diameter
        if self.shape == ToolShape.ROUGHING:
            return d * 0.5
        elif self.shape == ToolShape.BALL_ENDMILL:
            return d * 0.1  # Fine stepover for surface finish
        return d * 0.4  # Standard

    def get_sfm_multiplier(self) -> float:
        """Get surface feet per minute multiplier based on tool material/coating."""
        base = 1.0

        # Material multiplier
        material_mult = {
            ToolMaterial.HSS: 0.6,
            ToolMaterial.COBALT_HSS: 0.8,
            ToolMaterial.CARBIDE: 1.0,
            ToolMaterial.CARBIDE_TIPPED: 0.9,
            ToolMaterial.CERAMIC: 1.5,
            ToolMaterial.CBN: 2.0,
            ToolMaterial.PCD: 2.5,
        }
        base *= material_mult.get(self.material, 1.0)

        # Coating multiplier
        coating_mult = {
            ToolCoating.NONE: 1.0,
            ToolCoating.TIN: 1.1,
            ToolCoating.TICN: 1.15,
            ToolCoating.TIALN: 1.25,
            ToolCoating.ALTIN: 1.3,
            ToolCoating.ZRN: 1.2,
            ToolCoating.DLC: 1.4,
            ToolCoating.DIAMOND: 2.0,
        }
        base *= coating_mult.get(self.coating, 1.0)

        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "shape": self.shape.value,
            "material": self.material.value,
            "coating": self.coating.value,
            "diameter": self.diameter,
            "flutes": self.flutes,
            "flute_length": self.geometry.flute_length,
            "overall_length": self.geometry.overall_length,
            "shank_diameter": self.geometry.shank_diameter,
            "helix_angle": self.geometry.helix_angle,
            "corner_radius": self.geometry.corner_radius,
            "holder_type": self.holder_type.value,
            "max_rpm": self.max_rpm,
            "recommended_for": self.recommended_for,
        }


class ToolLibrary:
    """Database of tools with search and recommendations."""

    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}
        self._load_default_library()

    def _load_default_library(self):
        """Load default tool library."""
        # === FLAT ENDMILLS ===
        self._add_flat_endmills()

        # === BALL ENDMILLS ===
        self._add_ball_endmills()

        # === BULL NOSE ===
        self._add_bull_nose_endmills()

        # === ROUGHING ===
        self._add_roughing_endmills()

        # === V-BITS / CHAMFER ===
        self._add_vbits()

        # === DRILLS ===
        self._add_drills()

        # === SPECIALTY ===
        self._add_specialty_tools()

    def _add_flat_endmills(self):
        """Add flat endmill collection."""
        # Common sizes: 1, 2, 3, 4, 5, 6, 8, 10, 12mm
        sizes = [
            (1.0, 2, 3.0, 38, 3.175, "Micro"),
            (2.0, 2, 6.0, 38, 3.175, "Small"),
            (3.0, 2, 8.0, 45, 3.175, "Small"),
            (3.175, 2, 10.0, 45, 3.175, "1/8 inch"),  # 1/8"
            (4.0, 3, 12.0, 50, 4.0, "Medium"),
            (5.0, 3, 15.0, 50, 5.0, "Medium"),
            (6.0, 3, 18.0, 57, 6.0, "Standard"),
            (6.35, 4, 19.0, 57, 6.35, "1/4 inch"),  # 1/4"
            (8.0, 4, 20.0, 63, 8.0, "Standard"),
            (10.0, 4, 25.0, 72, 10.0, "Large"),
            (12.0, 4, 30.0, 83, 12.0, "Large"),
            (12.7, 4, 32.0, 83, 12.7, "1/2 inch"),  # 1/2"
        ]

        for d, flutes, fl, ol, sd, label in sizes:
            # Carbide coated
            self.add_tool(ToolSpec(
                tool_id=f"flat_carbide_{d}mm",
                name=f"Flat Endmill {d}mm Carbide",
                description=f"{label} {flutes}-flute carbide flat endmill",
                shape=ToolShape.FLAT_ENDMILL,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=d,
                    flute_length=fl,
                    overall_length=ol,
                    shank_diameter=sd,
                    flutes=flutes,
                    helix_angle=35.0,
                ),
                recommended_for=["roughing", "slotting", "profiling", "pocketing"],
                cost_usd=15.0 + d * 2,
            ))

            # HSS for softer materials
            if d >= 3.0:
                self.add_tool(ToolSpec(
                    tool_id=f"flat_hss_{d}mm",
                    name=f"Flat Endmill {d}mm HSS",
                    description=f"{label} {flutes}-flute HSS flat endmill",
                    shape=ToolShape.FLAT_ENDMILL,
                    material=ToolMaterial.HSS,
                    coating=ToolCoating.TIN,
                    geometry=ToolGeometry(
                        diameter=d,
                        flute_length=fl,
                        overall_length=ol,
                        shank_diameter=sd,
                        flutes=flutes,
                        helix_angle=30.0,
                    ),
                    recommended_for=["aluminum", "wood", "plastic", "general"],
                    cost_usd=8.0 + d,
                ))

    def _add_ball_endmills(self):
        """Add ball endmill collection."""
        sizes = [
            (1.0, 2, 2.0, 38, 3.175),
            (2.0, 2, 4.0, 38, 3.175),
            (3.0, 2, 6.0, 45, 3.175),
            (3.175, 2, 6.35, 45, 3.175),  # 1/8"
            (4.0, 2, 8.0, 50, 4.0),
            (5.0, 2, 10.0, 50, 5.0),
            (6.0, 2, 12.0, 57, 6.0),
            (6.35, 2, 12.7, 57, 6.35),  # 1/4"
            (8.0, 2, 16.0, 63, 8.0),
            (10.0, 2, 20.0, 72, 10.0),
            (12.0, 2, 24.0, 83, 12.0),
        ]

        for d, flutes, fl, ol, sd in sizes:
            self.add_tool(ToolSpec(
                tool_id=f"ball_carbide_{d}mm",
                name=f"Ball Endmill {d}mm Carbide",
                description=f"2-flute carbide ball nose R{d/2}",
                shape=ToolShape.BALL_ENDMILL,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=d,
                    flute_length=fl,
                    overall_length=ol,
                    shank_diameter=sd,
                    flutes=flutes,
                    helix_angle=30.0,
                    tip_angle=180.0,  # Full hemisphere
                ),
                recommended_for=["finishing", "3d_surfacing", "contour", "sculpting"],
                cost_usd=20.0 + d * 2.5,
            ))

    def _add_bull_nose_endmills(self):
        """Add bull nose (corner radius) endmills."""
        configs = [
            (6.0, 0.5, 4),
            (6.0, 1.0, 4),
            (8.0, 0.5, 4),
            (8.0, 1.0, 4),
            (8.0, 2.0, 4),
            (10.0, 1.0, 4),
            (10.0, 2.0, 4),
            (12.0, 1.0, 4),
            (12.0, 2.0, 4),
            (12.0, 3.0, 4),
        ]

        for d, r, flutes in configs:
            self.add_tool(ToolSpec(
                tool_id=f"bull_{d}mm_r{r}",
                name=f"Bull Nose {d}mm R{r}",
                description=f"Corner radius endmill {d}mm with R{r} corner",
                shape=ToolShape.BULL_ENDMILL,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=d,
                    flute_length=d * 2.5,
                    overall_length=d * 6,
                    shank_diameter=d,
                    flutes=flutes,
                    helix_angle=35.0,
                    corner_radius=r,
                ),
                recommended_for=["finishing", "semi_finish", "pocketing"],
                cost_usd=25.0 + d * 2,
            ))

    def _add_roughing_endmills(self):
        """Add roughing (corncob) endmills."""
        sizes = [6.0, 8.0, 10.0, 12.0, 16.0, 20.0]

        for d in sizes:
            self.add_tool(ToolSpec(
                tool_id=f"roughing_{d}mm",
                name=f"Roughing Endmill {d}mm",
                description=f"Corncob roughing endmill for heavy material removal",
                shape=ToolShape.ROUGHING,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=d,
                    flute_length=d * 3,
                    overall_length=d * 6,
                    shank_diameter=d,
                    flutes=4,
                    helix_angle=35.0,
                    chip_breaker=True,
                ),
                max_doc=d * 1.5,
                max_woc=d * 0.6,
                recommended_for=["roughing", "heavy_cutting", "hogging"],
                cost_usd=30.0 + d * 3,
            ))

    def _add_vbits(self):
        """Add V-bits and chamfer mills."""
        angles = [30, 45, 60, 90, 120]

        for angle in angles:
            # Small engraving V-bit
            self.add_tool(ToolSpec(
                tool_id=f"vbit_{angle}deg_small",
                name=f"V-Bit {angle}° Small",
                description=f"{angle}° V-bit for engraving and chamfering",
                shape=ToolShape.CHAMFER,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=6.0,
                    flute_length=10.0,
                    overall_length=50.0,
                    shank_diameter=6.0,
                    flutes=2,
                    tip_angle=float(angle),
                ),
                recommended_for=["engraving", "chamfer", "v_carving", "lettering"],
                cost_usd=15.0,
            ))

            # Larger chamfer mill
            if angle in [45, 60, 90]:
                self.add_tool(ToolSpec(
                    tool_id=f"chamfer_{angle}deg",
                    name=f"Chamfer Mill {angle}°",
                    description=f"{angle}° chamfer mill for edge breaking",
                    shape=ToolShape.CHAMFER,
                    material=ToolMaterial.CARBIDE,
                    coating=ToolCoating.TIALN,
                    geometry=ToolGeometry(
                        diameter=12.0,
                        flute_length=15.0,
                        overall_length=60.0,
                        shank_diameter=12.0,
                        flutes=4,
                        tip_angle=float(angle),
                    ),
                    recommended_for=["chamfer", "deburring", "countersink"],
                    cost_usd=25.0,
                ))

    def _add_drills(self):
        """Add twist drills and spot drills."""
        # Spot drills
        for angle in [90, 120]:
            self.add_tool(ToolSpec(
                tool_id=f"spot_drill_{angle}deg",
                name=f"Spot Drill {angle}°",
                description=f"{angle}° carbide spot drill",
                shape=ToolShape.SPOT_DRILL,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=6.0,
                    flute_length=8.0,
                    overall_length=50.0,
                    shank_diameter=6.0,
                    flutes=2,
                    tip_angle=float(angle),
                ),
                recommended_for=["spotting", "centering", "countersink"],
                cost_usd=20.0,
            ))

        # Twist drills - common sizes
        drill_sizes = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 12.0]

        for d in drill_sizes:
            fl = d * 6  # Standard drill flute length
            ol = fl + 20

            self.add_tool(ToolSpec(
                tool_id=f"drill_carbide_{d}mm",
                name=f"Carbide Drill {d}mm",
                description=f"{d}mm carbide twist drill 118°",
                shape=ToolShape.DRILL,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=d,
                    flute_length=fl,
                    overall_length=ol,
                    shank_diameter=d if d <= 12 else 12.0,
                    flutes=2,
                    tip_angle=118.0,
                    helix_angle=30.0,
                ),
                recommended_for=["drilling", "hole_making"],
                cost_usd=12.0 + d * 1.5,
            ))

    def _add_specialty_tools(self):
        """Add specialty tools."""
        # Tapered endmill for draft angles
        for taper in [1.0, 2.0, 3.0, 5.0]:
            self.add_tool(ToolSpec(
                tool_id=f"tapered_{taper}deg",
                name=f"Tapered Endmill {taper}°",
                description=f"Tapered endmill with {taper}° per side draft",
                shape=ToolShape.TAPERED,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=6.0,
                    flute_length=20.0,
                    overall_length=57.0,
                    shank_diameter=6.0,
                    flutes=2,
                    taper_angle=taper,
                ),
                recommended_for=["mold_making", "draft_walls", "die_work"],
                cost_usd=35.0,
            ))

        # Lollipop / undercut tool
        for d in [4.0, 6.0, 8.0]:
            self.add_tool(ToolSpec(
                tool_id=f"lollipop_{d}mm",
                name=f"Lollipop Cutter {d}mm",
                description=f"Undercutting tool with {d}mm ball on neck",
                shape=ToolShape.LOLLIPOP,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=d,
                    flute_length=d,
                    overall_length=50.0,
                    shank_diameter=6.0,
                    flutes=2,
                    neck_diameter=d * 0.6,
                    neck_length=15.0,
                    tip_angle=180.0,
                ),
                recommended_for=["undercut", "t_slot", "back_machining"],
                cost_usd=45.0,
            ))

        # Thread mills
        pitches = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for pitch in pitches:
            self.add_tool(ToolSpec(
                tool_id=f"thread_mill_m{pitch}",
                name=f"Thread Mill M{pitch} pitch",
                description=f"Single point thread mill for M threads, {pitch}mm pitch",
                shape=ToolShape.THREAD_MILL,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.TIALN,
                geometry=ToolGeometry(
                    diameter=4.0,
                    flute_length=12.0,
                    overall_length=50.0,
                    shank_diameter=6.0,
                    flutes=3,
                ),
                recommended_for=["threading", "internal_thread", "external_thread"],
                cost_usd=55.0,
            ))

        # Engraving tools
        for tip_width in [0.1, 0.2, 0.3, 0.5]:
            self.add_tool(ToolSpec(
                tool_id=f"engraving_{tip_width}mm",
                name=f"Engraving Bit {tip_width}mm tip",
                description=f"Fine engraving tool with {tip_width}mm tip",
                shape=ToolShape.ENGRAVING,
                material=ToolMaterial.CARBIDE,
                coating=ToolCoating.NONE,
                geometry=ToolGeometry(
                    diameter=3.175,
                    flute_length=8.0,
                    overall_length=38.0,
                    shank_diameter=3.175,
                    flutes=1,
                    tip_angle=30.0,
                ),
                recommended_for=["engraving", "fine_detail", "pcb"],
                cost_usd=12.0,
            ))

    def add_tool(self, tool: ToolSpec) -> None:
        """Add a tool to the library."""
        self.tools[tool.tool_id] = tool

    def get_tool(self, tool_id: str) -> Optional[ToolSpec]:
        """Get tool by ID."""
        return self.tools.get(tool_id)

    def find_tools(self,
                   shape: Optional[ToolShape] = None,
                   min_diameter: float = 0,
                   max_diameter: float = float('inf'),
                   material: Optional[ToolMaterial] = None,
                   coating: Optional[ToolCoating] = None,
                   recommended_for: Optional[str] = None) -> List[ToolSpec]:
        """Find tools matching criteria."""
        results = []

        for tool in self.tools.values():
            if shape and tool.shape != shape:
                continue
            if tool.diameter < min_diameter or tool.diameter > max_diameter:
                continue
            if material and tool.material != material:
                continue
            if coating and tool.coating != coating:
                continue
            if recommended_for and recommended_for not in tool.recommended_for:
                continue
            results.append(tool)

        return sorted(results, key=lambda t: t.diameter)

    def find_by_diameter(self, diameter: float, tolerance: float = 0.1) -> List[ToolSpec]:
        """Find tools by exact diameter."""
        return [t for t in self.tools.values()
                if abs(t.diameter - diameter) <= tolerance]

    def recommend_for_operation(self, operation: str,
                                max_diameter: float = float('inf'),
                                material_hardness: str = "medium") -> List[ToolSpec]:
        """Recommend tools for an operation type.

        Args:
            operation: Type of operation (roughing, finishing, drilling, etc.)
            max_diameter: Maximum tool diameter constraint
            material_hardness: soft, medium, hard

        Returns:
            List of recommended tools, sorted by preference
        """
        # Map operations to tool shapes and recommendations
        operation_map = {
            "roughing": [ToolShape.ROUGHING, ToolShape.FLAT_ENDMILL],
            "finishing": [ToolShape.BALL_ENDMILL, ToolShape.BULL_ENDMILL],
            "3d_surfacing": [ToolShape.BALL_ENDMILL],
            "pocketing": [ToolShape.FLAT_ENDMILL, ToolShape.BULL_ENDMILL],
            "slotting": [ToolShape.FLAT_ENDMILL, ToolShape.SLOT_DRILL],
            "profiling": [ToolShape.FLAT_ENDMILL],
            "drilling": [ToolShape.DRILL],
            "spotting": [ToolShape.SPOT_DRILL],
            "chamfer": [ToolShape.CHAMFER],
            "engraving": [ToolShape.ENGRAVING, ToolShape.CHAMFER],
            "threading": [ToolShape.THREAD_MILL],
            "undercut": [ToolShape.LOLLIPOP],
        }

        shapes = operation_map.get(operation, [])
        candidates = []

        for tool in self.tools.values():
            if tool.shape not in shapes:
                continue
            if tool.diameter > max_diameter:
                continue

            # Score based on suitability
            score = 0
            if operation in tool.recommended_for:
                score += 10

            # Prefer carbide for harder materials
            if material_hardness == "hard" and tool.material == ToolMaterial.CARBIDE:
                score += 5
            elif material_hardness == "soft" and tool.material == ToolMaterial.HSS:
                score += 2  # HSS is fine for soft materials

            # Prefer coated tools
            if tool.coating != ToolCoating.NONE:
                score += 2

            candidates.append((score, tool))

        candidates.sort(key=lambda x: (-x[0], x[1].diameter))
        return [t for _, t in candidates]

    def get_tool_set_for_job(self,
                            part_size: Tuple[float, float, float],
                            min_feature: float = 1.0,
                            operations: List[str] = None) -> Dict[str, ToolSpec]:
        """Recommend a complete tool set for a job.

        Args:
            part_size: (length, width, height) of the part
            min_feature: Smallest feature size to machine
            operations: List of required operations

        Returns:
            Dict mapping operation type to recommended tool
        """
        if operations is None:
            operations = ["roughing", "finishing", "drilling", "chamfer"]

        max_tool_diameter = min(part_size[0], part_size[1]) * 0.8

        tool_set = {}

        for op in operations:
            recommendations = self.recommend_for_operation(
                op,
                max_diameter=max_tool_diameter if op != "drilling" else float('inf')
            )
            if recommendations:
                # For finishing, prefer smaller tools
                if op == "finishing":
                    for tool in recommendations:
                        if tool.diameter <= min_feature * 2:
                            tool_set[op] = tool
                            break
                    if op not in tool_set and recommendations:
                        tool_set[op] = recommendations[0]
                else:
                    tool_set[op] = recommendations[0]

        return tool_set

    def list_all(self) -> List[ToolSpec]:
        """List all tools in library."""
        return sorted(self.tools.values(), key=lambda t: (t.shape.value, t.diameter))

    def get_categories(self) -> Dict[str, int]:
        """Get tool count by category."""
        categories = {}
        for tool in self.tools.values():
            cat = tool.shape.value
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def to_dict(self) -> Dict[str, Any]:
        """Export library as dictionary."""
        return {
            "tool_count": len(self.tools),
            "categories": self.get_categories(),
            "tools": [t.to_dict() for t in self.list_all()],
        }


# Global library instance
_default_library: Optional[ToolLibrary] = None


def get_tool_library() -> ToolLibrary:
    """Get the default tool library."""
    global _default_library
    if _default_library is None:
        _default_library = ToolLibrary()
    return _default_library


def get_tool(tool_id: str) -> Optional[ToolSpec]:
    """Get a tool from the default library."""
    return get_tool_library().get_tool(tool_id)


def list_tools(shape: Optional[ToolShape] = None) -> List[ToolSpec]:
    """List tools from the default library."""
    lib = get_tool_library()
    if shape:
        return lib.find_tools(shape=shape)
    return lib.list_all()


def recommend_tools(operation: str, max_diameter: float = float('inf')) -> List[ToolSpec]:
    """Get tool recommendations for an operation."""
    return get_tool_library().recommend_for_operation(operation, max_diameter)
