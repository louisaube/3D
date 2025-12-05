"""
Materials database for PyCAM3D.

Provides:
- Material properties (hardness, machinability)
- Recommended feeds and speeds by material/tool combination
- Automatic calculation of cutting parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple
import math


class MaterialCategory(Enum):
    """Material category."""
    WOOD_SOFT = "Soft Wood"
    WOOD_HARD = "Hard Wood"
    WOOD_PLYWOOD = "Plywood/MDF"
    PLASTIC_SOFT = "Soft Plastic"
    PLASTIC_HARD = "Hard Plastic"
    FOAM = "Foam"
    ALUMINUM = "Aluminum"
    BRASS = "Brass"
    STEEL_MILD = "Mild Steel"
    STEEL_STAINLESS = "Stainless Steel"
    COMPOSITE = "Composite"
    WAX = "Machinable Wax"


@dataclass
class Material:
    """Material with cutting properties."""
    name: str
    category: MaterialCategory
    description: str = ""

    # Cutting characteristics
    surface_speed_min: float = 100.0  # m/min (SFM)
    surface_speed_max: float = 300.0  # m/min
    chip_load_factor: float = 1.0     # Multiplier for chip load

    # Physical properties
    hardness_brinell: Optional[float] = None
    density_g_cm3: Optional[float] = None

    # Recommended settings
    use_coolant: bool = False
    use_air_blast: bool = False
    climb_milling: bool = True

    def get_rpm(self, tool_diameter: float, sfm: Optional[float] = None) -> int:
        """Calculate optimal RPM for given tool diameter.

        RPM = (SFM × 1000) / (π × D)

        Args:
            tool_diameter: Tool diameter in mm
            sfm: Surface speed in m/min (default: mid-range)
        """
        if sfm is None:
            sfm = (self.surface_speed_min + self.surface_speed_max) / 2

        rpm = (sfm * 1000) / (math.pi * tool_diameter)
        return int(rpm)

    def get_feed_rate(self, tool_diameter: float, flutes: int, rpm: int,
                      chip_load: Optional[float] = None) -> float:
        """Calculate feed rate in mm/min.

        Feed = RPM × flutes × chip_load

        Args:
            tool_diameter: Tool diameter in mm
            flutes: Number of flutes
            rpm: Spindle RPM
            chip_load: Chip load per tooth in mm (auto-calculated if None)
        """
        if chip_load is None:
            # Base chip load as fraction of diameter
            chip_load = tool_diameter * 0.01 * self.chip_load_factor

        return rpm * flutes * chip_load

    def get_doc(self, tool_diameter: float, roughing: bool = False) -> float:
        """Get recommended depth of cut.

        Args:
            tool_diameter: Tool diameter in mm
            roughing: True for roughing, False for finishing
        """
        if roughing:
            # Roughing: 50-100% of diameter depending on material
            return tool_diameter * (0.5 if self.category in [
                MaterialCategory.ALUMINUM,
                MaterialCategory.BRASS,
                MaterialCategory.STEEL_MILD,
            ] else 1.0)
        else:
            # Finishing: 5-20% of diameter
            return tool_diameter * 0.1

    def get_stepover(self, tool_diameter: float, roughing: bool = False) -> float:
        """Get recommended stepover.

        Args:
            tool_diameter: Tool diameter in mm
            roughing: True for roughing, False for finishing
        """
        if roughing:
            return tool_diameter * 0.4  # 40% for roughing
        else:
            return tool_diameter * 0.1  # 10% for finishing


# =============================================================================
# MATERIALS DATABASE
# =============================================================================

MATERIALS: Dict[str, Material] = {
    # Soft Woods
    "pine": Material(
        name="Pine",
        category=MaterialCategory.WOOD_SOFT,
        description="Softwood, easy to machine",
        surface_speed_min=300, surface_speed_max=600,
        chip_load_factor=1.2,
        density_g_cm3=0.5,
    ),
    "cedar": Material(
        name="Cedar",
        category=MaterialCategory.WOOD_SOFT,
        description="Soft, aromatic wood",
        surface_speed_min=300, surface_speed_max=600,
        chip_load_factor=1.2,
        density_g_cm3=0.38,
    ),
    "balsa": Material(
        name="Balsa",
        category=MaterialCategory.WOOD_SOFT,
        description="Very soft, lightweight wood",
        surface_speed_min=400, surface_speed_max=800,
        chip_load_factor=1.5,
        density_g_cm3=0.16,
    ),

    # Hard Woods
    "oak": Material(
        name="Oak",
        category=MaterialCategory.WOOD_HARD,
        description="Dense hardwood, classic choice",
        surface_speed_min=200, surface_speed_max=400,
        chip_load_factor=0.8,
        hardness_brinell=3.7,
        density_g_cm3=0.75,
    ),
    "maple": Material(
        name="Maple",
        category=MaterialCategory.WOOD_HARD,
        description="Hard, light-colored wood",
        surface_speed_min=200, surface_speed_max=400,
        chip_load_factor=0.8,
        hardness_brinell=4.0,
        density_g_cm3=0.7,
    ),
    "walnut": Material(
        name="Walnut",
        category=MaterialCategory.WOOD_HARD,
        description="Medium-hard, beautiful grain",
        surface_speed_min=250, surface_speed_max=450,
        chip_load_factor=0.9,
        hardness_brinell=3.4,
        density_g_cm3=0.65,
    ),
    "cherry": Material(
        name="Cherry",
        category=MaterialCategory.WOOD_HARD,
        description="Medium hardness, machines well",
        surface_speed_min=250, surface_speed_max=450,
        chip_load_factor=0.9,
        hardness_brinell=3.0,
        density_g_cm3=0.58,
    ),
    "mahogany": Material(
        name="Mahogany",
        category=MaterialCategory.WOOD_HARD,
        description="Tropical hardwood, excellent machinability",
        surface_speed_min=250, surface_speed_max=450,
        chip_load_factor=1.0,
        hardness_brinell=2.7,
        density_g_cm3=0.54,
    ),

    # Plywood & Engineered
    "plywood": Material(
        name="Plywood",
        category=MaterialCategory.WOOD_PLYWOOD,
        description="Layered wood sheets",
        surface_speed_min=250, surface_speed_max=500,
        chip_load_factor=0.9,
        use_air_blast=True,
    ),
    "mdf": Material(
        name="MDF",
        category=MaterialCategory.WOOD_PLYWOOD,
        description="Medium density fiberboard",
        surface_speed_min=300, surface_speed_max=600,
        chip_load_factor=1.0,
        use_air_blast=True,
        density_g_cm3=0.75,
    ),
    "hdpe-sheet": Material(
        name="HDPE Cutting Board",
        category=MaterialCategory.PLASTIC_SOFT,
        description="HDPE sheet, food safe",
        surface_speed_min=200, surface_speed_max=400,
        chip_load_factor=1.0,
        use_air_blast=True,
    ),

    # Plastics
    "acrylic": Material(
        name="Acrylic (PMMA)",
        category=MaterialCategory.PLASTIC_HARD,
        description="Clear plastic, prone to melting",
        surface_speed_min=100, surface_speed_max=200,
        chip_load_factor=0.6,
        use_air_blast=True,
        climb_milling=False,  # Conventional better for acrylic
        density_g_cm3=1.18,
    ),
    "polycarbonate": Material(
        name="Polycarbonate",
        category=MaterialCategory.PLASTIC_HARD,
        description="Tough clear plastic",
        surface_speed_min=100, surface_speed_max=200,
        chip_load_factor=0.7,
        use_air_blast=True,
        density_g_cm3=1.2,
    ),
    "hdpe": Material(
        name="HDPE",
        category=MaterialCategory.PLASTIC_SOFT,
        description="Soft, easy to machine plastic",
        surface_speed_min=200, surface_speed_max=400,
        chip_load_factor=1.0,
        density_g_cm3=0.95,
    ),
    "delrin": Material(
        name="Delrin (POM)",
        category=MaterialCategory.PLASTIC_HARD,
        description="Acetal, excellent machinability",
        surface_speed_min=150, surface_speed_max=300,
        chip_load_factor=0.9,
        use_air_blast=True,
        density_g_cm3=1.41,
    ),
    "nylon": Material(
        name="Nylon",
        category=MaterialCategory.PLASTIC_HARD,
        description="Tough, abrasion resistant",
        surface_speed_min=100, surface_speed_max=200,
        chip_load_factor=0.7,
        density_g_cm3=1.14,
    ),
    "abs": Material(
        name="ABS",
        category=MaterialCategory.PLASTIC_HARD,
        description="Common 3D printing plastic",
        surface_speed_min=150, surface_speed_max=300,
        chip_load_factor=0.8,
        density_g_cm3=1.04,
    ),
    "pvc-foam": Material(
        name="PVC Foam Board",
        category=MaterialCategory.FOAM,
        description="Sintra/Forex, lightweight",
        surface_speed_min=300, surface_speed_max=600,
        chip_load_factor=1.2,
        density_g_cm3=0.6,
    ),

    # Foam
    "xps-foam": Material(
        name="XPS Foam",
        category=MaterialCategory.FOAM,
        description="Extruded polystyrene insulation",
        surface_speed_min=500, surface_speed_max=1000,
        chip_load_factor=2.0,
        density_g_cm3=0.03,
    ),
    "eps-foam": Material(
        name="EPS Foam",
        category=MaterialCategory.FOAM,
        description="Expanded polystyrene (Styrofoam)",
        surface_speed_min=500, surface_speed_max=1000,
        chip_load_factor=2.0,
        density_g_cm3=0.02,
    ),
    "pu-foam": Material(
        name="Tooling Foam (PU)",
        category=MaterialCategory.FOAM,
        description="High-density modeling foam",
        surface_speed_min=400, surface_speed_max=800,
        chip_load_factor=1.5,
        density_g_cm3=0.4,
    ),

    # Metals
    "aluminum-6061": Material(
        name="Aluminum 6061-T6",
        category=MaterialCategory.ALUMINUM,
        description="Most common machinable aluminum",
        surface_speed_min=150, surface_speed_max=300,
        chip_load_factor=0.5,
        use_coolant=True,
        hardness_brinell=95,
        density_g_cm3=2.7,
    ),
    "aluminum-7075": Material(
        name="Aluminum 7075-T6",
        category=MaterialCategory.ALUMINUM,
        description="High-strength aluminum alloy",
        surface_speed_min=120, surface_speed_max=250,
        chip_load_factor=0.4,
        use_coolant=True,
        hardness_brinell=150,
        density_g_cm3=2.81,
    ),
    "aluminum-cast": Material(
        name="Cast Aluminum",
        category=MaterialCategory.ALUMINUM,
        description="Softer, easier to machine",
        surface_speed_min=200, surface_speed_max=400,
        chip_load_factor=0.6,
        use_coolant=True,
        density_g_cm3=2.7,
    ),
    "brass": Material(
        name="Brass 360",
        category=MaterialCategory.BRASS,
        description="Free-machining brass",
        surface_speed_min=100, surface_speed_max=200,
        chip_load_factor=0.6,
        use_coolant=True,
        hardness_brinell=120,
        density_g_cm3=8.5,
    ),
    "copper": Material(
        name="Copper",
        category=MaterialCategory.BRASS,
        description="Pure copper, gummy to machine",
        surface_speed_min=50, surface_speed_max=100,
        chip_load_factor=0.4,
        use_coolant=True,
        density_g_cm3=8.96,
    ),
    "steel-1018": Material(
        name="Steel 1018",
        category=MaterialCategory.STEEL_MILD,
        description="Low carbon mild steel",
        surface_speed_min=30, surface_speed_max=60,
        chip_load_factor=0.3,
        use_coolant=True,
        hardness_brinell=130,
        density_g_cm3=7.87,
    ),
    "steel-4140": Material(
        name="Steel 4140",
        category=MaterialCategory.STEEL_MILD,
        description="Alloy steel, tougher",
        surface_speed_min=25, surface_speed_max=50,
        chip_load_factor=0.25,
        use_coolant=True,
        hardness_brinell=200,
        density_g_cm3=7.85,
    ),
    "stainless-304": Material(
        name="Stainless 304",
        category=MaterialCategory.STEEL_STAINLESS,
        description="Austenitic stainless, work hardens",
        surface_speed_min=20, surface_speed_max=40,
        chip_load_factor=0.2,
        use_coolant=True,
        hardness_brinell=200,
        density_g_cm3=8.0,
    ),

    # Composites
    "carbon-fiber": Material(
        name="Carbon Fiber (CFRP)",
        category=MaterialCategory.COMPOSITE,
        description="Abrasive, requires diamond tooling",
        surface_speed_min=100, surface_speed_max=200,
        chip_load_factor=0.5,
        use_air_blast=True,
        density_g_cm3=1.6,
    ),
    "fiberglass": Material(
        name="Fiberglass (GFRP)",
        category=MaterialCategory.COMPOSITE,
        description="Glass reinforced plastic",
        surface_speed_min=150, surface_speed_max=300,
        chip_load_factor=0.6,
        use_air_blast=True,
        density_g_cm3=1.8,
    ),

    # Specialty
    "machinable-wax": Material(
        name="Machinable Wax",
        category=MaterialCategory.WAX,
        description="Blue machinable wax for prototyping",
        surface_speed_min=500, surface_speed_max=1000,
        chip_load_factor=1.5,
        density_g_cm3=0.9,
    ),
    "renshape": Material(
        name="Renshape Tooling Board",
        category=MaterialCategory.FOAM,
        description="High-density tooling board",
        surface_speed_min=300, surface_speed_max=600,
        chip_load_factor=1.0,
        use_air_blast=True,
        density_g_cm3=0.7,
    ),
}


@dataclass
class CuttingParameters:
    """Complete cutting parameters for a machining operation."""
    rpm: int
    feed_rate: float  # mm/min
    depth_of_cut: float  # mm
    stepover: float  # mm
    plunge_rate: float  # mm/min
    use_coolant: bool
    use_air_blast: bool
    climb_milling: bool
    chip_load: float  # mm/tooth

    def to_dict(self) -> Dict:
        return {
            "rpm": self.rpm,
            "feed_rate": self.feed_rate,
            "depth_of_cut": self.depth_of_cut,
            "stepover": self.stepover,
            "plunge_rate": self.plunge_rate,
            "use_coolant": self.use_coolant,
            "use_air_blast": self.use_air_blast,
            "climb_milling": self.climb_milling,
            "chip_load": self.chip_load,
        }


class FeedsSpeedsCalculator:
    """Calculate optimal feeds and speeds for material/tool combinations."""

    def __init__(self, material: Material):
        self.material = material

    def calculate(
        self,
        tool_diameter: float,
        flutes: int = 2,
        roughing: bool = False,
        max_rpm: int = 24000,
        max_feed: float = 10000.0,
        conservative: bool = True,
    ) -> CuttingParameters:
        """Calculate complete cutting parameters.

        Args:
            tool_diameter: Tool diameter in mm
            flutes: Number of cutting edges
            roughing: True for roughing, False for finishing
            max_rpm: Machine's maximum RPM
            max_feed: Machine's maximum feed rate
            conservative: Use conservative (safer) values
        """
        # Calculate RPM
        sfm = self.material.surface_speed_min if conservative else (
            (self.material.surface_speed_min + self.material.surface_speed_max) / 2
        )
        rpm = self.material.get_rpm(tool_diameter, sfm)
        rpm = min(rpm, max_rpm)

        # Calculate chip load based on tool size and material
        base_chip_load = tool_diameter * 0.01
        chip_load = base_chip_load * self.material.chip_load_factor
        if conservative:
            chip_load *= 0.7
        if not roughing:
            chip_load *= 0.5  # Light cuts for finishing

        # Calculate feed rate
        feed_rate = rpm * flutes * chip_load
        feed_rate = min(feed_rate, max_feed)

        # Depths
        doc = self.material.get_doc(tool_diameter, roughing)
        stepover = self.material.get_stepover(tool_diameter, roughing)

        # Plunge rate (typically 30-50% of feed rate)
        plunge_rate = feed_rate * 0.3

        return CuttingParameters(
            rpm=int(rpm),
            feed_rate=round(feed_rate, 1),
            depth_of_cut=round(doc, 2),
            stepover=round(stepover, 2),
            plunge_rate=round(plunge_rate, 1),
            use_coolant=self.material.use_coolant,
            use_air_blast=self.material.use_air_blast,
            climb_milling=self.material.climb_milling,
            chip_load=round(chip_load, 4),
        )


class MaterialDatabase:
    """Database of materials with search and filtering."""

    def __init__(self):
        self.materials = MATERIALS.copy()

    def list_all(self) -> List[str]:
        """List all material names."""
        return list(self.materials.keys())

    def get(self, name: str) -> Optional[Material]:
        """Get material by name."""
        return self.materials.get(name)

    def search(self, query: str) -> List[Material]:
        """Search materials by name or description."""
        query = query.lower()
        results = []
        for material in self.materials.values():
            if (query in material.name.lower() or
                query in material.description.lower()):
                results.append(material)
        return results

    def filter_by_category(self, category: MaterialCategory) -> List[Material]:
        """Filter materials by category."""
        return [m for m in self.materials.values() if m.category == category]

    def get_categories(self) -> List[MaterialCategory]:
        """Get list of all categories."""
        return list(MaterialCategory)


# Convenience functions
def get_material(name: str) -> Optional[Material]:
    """Get a material by name from the default database."""
    db = MaterialDatabase()
    return db.get(name)


def list_materials() -> List[str]:
    """List all available material names."""
    db = MaterialDatabase()
    return db.list_all()


def calculate_feeds_speeds(
    material_name: str,
    tool_diameter: float,
    flutes: int = 2,
    roughing: bool = False,
    max_rpm: int = 24000,
) -> Optional[CuttingParameters]:
    """Convenience function to calculate feeds and speeds."""
    material = get_material(material_name)
    if material is None:
        return None
    calc = FeedsSpeedsCalculator(material)
    return calc.calculate(
        tool_diameter=tool_diameter,
        flutes=flutes,
        roughing=roughing,
        max_rpm=max_rpm,
    )
