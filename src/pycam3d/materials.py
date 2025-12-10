"""
Material presets and cutting parameter calculations for CNC machining.

Provides:
- Material database with cutting properties
- Feed/speed calculations based on tool and material
- Optimal DOC (depth of cut) recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import math


class MaterialType(Enum):
    """Common machinable materials."""
    ALUMINUM = "aluminum"
    WOOD_SOFT = "wood_soft"
    WOOD_HARD = "wood_hard"
    PLASTIC_SOFT = "plastic_soft"
    PLASTIC_HARD = "plastic_hard"
    BRASS = "brass"
    STEEL_MILD = "steel_mild"
    FOAM = "foam"
    MDF = "mdf"


@dataclass
class MaterialProperties:
    """Cutting properties for a material."""
    name: str
    type: MaterialType
    sfm_min: float  # Surface feet/min (conservative)
    sfm_max: float  # Surface feet/min (aggressive)
    chipload_min: float  # mm/tooth for 6mm tool
    chipload_max: float
    doc_roughing: float  # % of tool diameter
    doc_finishing: float
    stepover_roughing: float  # % of tool diameter
    stepover_finishing: float
    coolant: str
    color: str


# Material database
MATERIALS: Dict[MaterialType, MaterialProperties] = {
    MaterialType.ALUMINUM: MaterialProperties(
        name="Aluminium", type=MaterialType.ALUMINUM,
        sfm_min=150, sfm_max=300,
        chipload_min=0.05, chipload_max=0.15,
        doc_roughing=50, doc_finishing=10,
        stepover_roughing=40, stepover_finishing=10,
        coolant="Flood/Mist", color="#C0C0C0"
    ),
    MaterialType.WOOD_SOFT: MaterialProperties(
        name="Bois tendre", type=MaterialType.WOOD_SOFT,
        sfm_min=300, sfm_max=600,
        chipload_min=0.1, chipload_max=0.3,
        doc_roughing=100, doc_finishing=20,
        stepover_roughing=50, stepover_finishing=15,
        coolant="Air", color="#DEB887"
    ),
    MaterialType.WOOD_HARD: MaterialProperties(
        name="Bois dur", type=MaterialType.WOOD_HARD,
        sfm_min=200, sfm_max=450,
        chipload_min=0.08, chipload_max=0.2,
        doc_roughing=75, doc_finishing=15,
        stepover_roughing=45, stepover_finishing=12,
        coolant="Air", color="#8B4513"
    ),
    MaterialType.PLASTIC_SOFT: MaterialProperties(
        name="Plastique souple", type=MaterialType.PLASTIC_SOFT,
        sfm_min=200, sfm_max=500,
        chipload_min=0.1, chipload_max=0.25,
        doc_roughing=100, doc_finishing=25,
        stepover_roughing=50, stepover_finishing=15,
        coolant="Air", color="#87CEEB"
    ),
    MaterialType.PLASTIC_HARD: MaterialProperties(
        name="Plastique dur", type=MaterialType.PLASTIC_HARD,
        sfm_min=150, sfm_max=350,
        chipload_min=0.08, chipload_max=0.18,
        doc_roughing=75, doc_finishing=15,
        stepover_roughing=40, stepover_finishing=12,
        coolant="Air/Mist", color="#ADD8E6"
    ),
    MaterialType.BRASS: MaterialProperties(
        name="Laiton", type=MaterialType.BRASS,
        sfm_min=100, sfm_max=200,
        chipload_min=0.05, chipload_max=0.12,
        doc_roughing=40, doc_finishing=10,
        stepover_roughing=35, stepover_finishing=10,
        coolant="Flood", color="#B5A642"
    ),
    MaterialType.STEEL_MILD: MaterialProperties(
        name="Acier doux", type=MaterialType.STEEL_MILD,
        sfm_min=50, sfm_max=120,
        chipload_min=0.03, chipload_max=0.08,
        doc_roughing=30, doc_finishing=5,
        stepover_roughing=30, stepover_finishing=8,
        coolant="Flood", color="#708090"
    ),
    MaterialType.FOAM: MaterialProperties(
        name="Mousse", type=MaterialType.FOAM,
        sfm_min=500, sfm_max=1000,
        chipload_min=0.2, chipload_max=0.5,
        doc_roughing=200, doc_finishing=50,
        stepover_roughing=60, stepover_finishing=20,
        coolant="None", color="#FFFACD"
    ),
    MaterialType.MDF: MaterialProperties(
        name="MDF", type=MaterialType.MDF,
        sfm_min=250, sfm_max=500,
        chipload_min=0.1, chipload_max=0.25,
        doc_roughing=100, doc_finishing=20,
        stepover_roughing=50, stepover_finishing=15,
        coolant="Aspiration", color="#D2691E"
    ),
}


@dataclass
class CuttingParameters:
    """Calculated cutting parameters."""
    spindle_rpm: int
    feed_rate: float  # mm/min
    plunge_rate: float  # mm/min
    depth_of_cut: float  # mm
    stepover: float  # mm
    material: str
    tool_diameter: float

    def to_dict(self) -> dict:
        return {
            "spindle_rpm": self.spindle_rpm,
            "feed_rate": round(self.feed_rate, 0),
            "plunge_rate": round(self.plunge_rate, 0),
            "depth_of_cut": round(self.depth_of_cut, 2),
            "stepover": round(self.stepover, 2),
            "material": self.material,
            "tool_diameter": self.tool_diameter,
        }


def calculate_cutting_params(
    material_type: MaterialType,
    tool_diameter: float,
    tool_flutes: int = 2,
    is_finishing: bool = False,
    max_rpm: int = 24000,
) -> CuttingParameters:
    """Calculate optimal cutting parameters."""
    mat = MATERIALS[material_type]

    # Average surface speed
    sfm = (mat.sfm_min + mat.sfm_max) / 2

    # RPM = (SFM * 1000) / (π * D)
    rpm = int((sfm * 1000) / (math.pi * tool_diameter))
    rpm = min(rpm, max_rpm)
    rpm = max(rpm, 1000)

    # Chipload scaled by tool size
    chipload = (mat.chipload_min + mat.chipload_max) / 2
    chipload *= (tool_diameter / 6.0) ** 0.5
    chipload = max(chipload, 0.01)

    # Feed = RPM * flutes * chipload
    feed_rate = rpm * tool_flutes * chipload
    plunge_rate = feed_rate * 0.4

    # DOC and stepover
    doc_pct = mat.doc_finishing if is_finishing else mat.doc_roughing
    step_pct = mat.stepover_finishing if is_finishing else mat.stepover_roughing

    return CuttingParameters(
        spindle_rpm=rpm,
        feed_rate=feed_rate,
        plunge_rate=plunge_rate,
        depth_of_cut=tool_diameter * doc_pct / 100,
        stepover=tool_diameter * step_pct / 100,
        material=mat.name,
        tool_diameter=tool_diameter,
    )


def get_material_list() -> List[dict]:
    """Get list of materials for UI."""
    return [
        {"type": m.value, "name": p.name, "color": p.color, "coolant": p.coolant}
        for m, p in MATERIALS.items()
    ]


# Aliases for compatibility with __init__.py
Material = MaterialProperties
MaterialDatabase = MATERIALS
MaterialCategory = MaterialType
FeedsSpeedsCalculator = calculate_cutting_params


def get_material(material_type: str) -> MaterialProperties:
    """Get material by type string."""
    return MATERIALS.get(MaterialType(material_type))


def list_materials() -> List[str]:
    """List available material types."""
    return [m.value for m in MaterialType]


def calculate_feeds_speeds(material: str, tool_dia: float, flutes: int = 2) -> dict:
    """Calculate feeds and speeds for material and tool."""
    params = calculate_cutting_params(MaterialType(material), tool_dia, flutes)
    return params.to_dict()
