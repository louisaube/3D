"""
Machine database for PyCAM3D.

Provides:
- Pre-configured CNC machine profiles
- Work envelope limits
- Post-processor configurations
- Spindle and feed rate limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import json
from pathlib import Path


class MachineType(Enum):
    """Type of CNC machine."""
    ROUTER_3AXIS = "3-axis router"
    ROUTER_4AXIS = "4-axis router"
    ROUTER_5AXIS = "5-axis router"
    MILL_3AXIS = "3-axis mill"
    MILL_5AXIS = "5-axis mill"
    LASER = "laser cutter"
    PLASMA = "plasma cutter"


class PostProcessor(Enum):
    """G-code dialect/post-processor."""
    GRBL = "grbl"
    LINUXCNC = "linuxcnc"
    MACH3 = "mach3"
    MACH4 = "mach4"
    FANUC = "fanuc"
    HAAS = "haas"
    MAZAK = "mazak"
    MARLIN = "marlin"
    SMOOTHIE = "smoothie"
    UCCNC = "uccnc"


@dataclass
class WorkEnvelope:
    """Machine work envelope (travel limits)."""
    x_min: float = 0.0
    x_max: float = 300.0
    y_min: float = 0.0
    y_max: float = 300.0
    z_min: float = -80.0
    z_max: float = 0.0

    @property
    def x_travel(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_travel(self) -> float:
        return self.y_max - self.y_min

    @property
    def z_travel(self) -> float:
        return self.z_max - self.z_min

    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if point is within work envelope."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def to_dict(self) -> Dict:
        return {
            "x_min": self.x_min, "x_max": self.x_max,
            "y_min": self.y_min, "y_max": self.y_max,
            "z_min": self.z_min, "z_max": self.z_max,
        }


@dataclass
class SpindleSpec:
    """Spindle specifications."""
    min_rpm: int = 1000
    max_rpm: int = 24000
    power_watts: int = 800
    collet_type: str = "ER11"  # ER11, ER16, ER20, ER32

    def validate_rpm(self, rpm: int) -> int:
        """Clamp RPM to valid range."""
        return max(self.min_rpm, min(self.max_rpm, rpm))


@dataclass
class FeedLimits:
    """Machine feed rate limits."""
    max_feed_xy: float = 5000.0  # mm/min
    max_feed_z: float = 2000.0   # mm/min
    max_rapid: float = 10000.0   # mm/min
    acceleration: float = 500.0  # mm/s²

    def validate_feed(self, feed: float, is_z: bool = False) -> float:
        """Clamp feed rate to valid range."""
        max_feed = self.max_feed_z if is_z else self.max_feed_xy
        return min(max_feed, max(1.0, feed))


@dataclass
class Machine:
    """CNC machine profile."""
    name: str
    manufacturer: str
    model: str
    machine_type: MachineType
    post_processor: PostProcessor
    envelope: WorkEnvelope
    spindle: SpindleSpec
    feeds: FeedLimits
    description: str = ""

    # Post-processor specific settings
    use_line_numbers: bool = False
    line_number_increment: int = 10
    use_canned_cycles: bool = True
    decimal_places: int = 3
    modal_groups: bool = True

    def validate_toolpath(self, points: List[Tuple[float, float, float]]) -> List[str]:
        """Check if toolpath fits within machine envelope."""
        errors = []
        for i, (x, y, z) in enumerate(points):
            if not self.envelope.contains(x, y, z):
                if x < self.envelope.x_min or x > self.envelope.x_max:
                    errors.append(f"Point {i}: X={x:.2f} outside [{self.envelope.x_min}, {self.envelope.x_max}]")
                if y < self.envelope.y_min or y > self.envelope.y_max:
                    errors.append(f"Point {i}: Y={y:.2f} outside [{self.envelope.y_min}, {self.envelope.y_max}]")
                if z < self.envelope.z_min or z > self.envelope.z_max:
                    errors.append(f"Point {i}: Z={z:.2f} outside [{self.envelope.z_min}, {self.envelope.z_max}]")
        return errors

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "machine_type": self.machine_type.value,
            "post_processor": self.post_processor.value,
            "envelope": self.envelope.to_dict(),
            "spindle": {
                "min_rpm": self.spindle.min_rpm,
                "max_rpm": self.spindle.max_rpm,
                "power_watts": self.spindle.power_watts,
                "collet_type": self.spindle.collet_type,
            },
            "feeds": {
                "max_feed_xy": self.feeds.max_feed_xy,
                "max_feed_z": self.feeds.max_feed_z,
                "max_rapid": self.feeds.max_rapid,
            },
        }


# =============================================================================
# PRE-CONFIGURED MACHINE DATABASE
# =============================================================================

MACHINES: Dict[str, Machine] = {
    # Hobby/Desktop CNC Routers
    "shapeoko-4": Machine(
        name="Shapeoko 4",
        manufacturer="Carbide 3D",
        model="Shapeoko 4 XL",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.GRBL,
        envelope=WorkEnvelope(x_max=838, y_max=838, z_min=-75, z_max=0),
        spindle=SpindleSpec(min_rpm=10000, max_rpm=30000, power_watts=500, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=10000, max_feed_z=5000, max_rapid=10000),
        description="Popular hobby CNC router, great for wood and aluminum",
    ),
    "x-carve": Machine(
        name="X-Carve",
        manufacturer="Inventables",
        model="X-Carve Pro",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.GRBL,
        envelope=WorkEnvelope(x_max=610, y_max=610, z_min=-90, z_max=0),
        spindle=SpindleSpec(min_rpm=8000, max_rpm=30000, power_watts=600, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=8000, max_feed_z=3000, max_rapid=8000),
        description="Beginner-friendly CNC router",
    ),
    "onefinity": Machine(
        name="Onefinity Woodworker",
        manufacturer="Onefinity",
        model="Woodworker X-50",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.GRBL,
        envelope=WorkEnvelope(x_max=816, y_max=816, z_min=-114, z_max=0),
        spindle=SpindleSpec(min_rpm=10000, max_rpm=24000, power_watts=2200, collet_type="ER20"),
        feeds=FeedLimits(max_feed_xy=12000, max_feed_z=6000, max_rapid=15000),
        description="Rigid hobbyist CNC with ball screws",
    ),
    "workbee": Machine(
        name="WorkBee",
        manufacturer="Ooznest",
        model="WorkBee 1010",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.GRBL,
        envelope=WorkEnvelope(x_max=1000, y_max=1000, z_min=-120, z_max=0),
        spindle=SpindleSpec(min_rpm=8000, max_rpm=24000, power_watts=2200, collet_type="ER20"),
        feeds=FeedLimits(max_feed_xy=8000, max_feed_z=3000, max_rapid=10000),
        description="Large format V-wheel CNC router",
    ),
    "lowrider-3": Machine(
        name="LowRider 3",
        manufacturer="V1 Engineering",
        model="LowRider 3 CNC",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.MARLIN,
        envelope=WorkEnvelope(x_max=1220, y_max=2440, z_min=-100, z_max=0),
        spindle=SpindleSpec(min_rpm=10000, max_rpm=30000, power_watts=800, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=6000, max_feed_z=2000, max_rapid=8000),
        description="DIY large format CNC for full sheet goods",
    ),
    "mpcnc": Machine(
        name="MPCNC Primo",
        manufacturer="V1 Engineering",
        model="MPCNC Primo",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.MARLIN,
        envelope=WorkEnvelope(x_max=600, y_max=600, z_min=-80, z_max=0),
        spindle=SpindleSpec(min_rpm=10000, max_rpm=30000, power_watts=500, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=4000, max_feed_z=1500, max_rapid=6000),
        description="Affordable DIY CNC with 3D printed parts",
    ),
    "snapmaker": Machine(
        name="Snapmaker 2.0",
        manufacturer="Snapmaker",
        model="Snapmaker 2.0 A350",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.MARLIN,
        envelope=WorkEnvelope(x_max=320, y_max=350, z_min=-330, z_max=0),
        spindle=SpindleSpec(min_rpm=6000, max_rpm=12000, power_watts=50, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=6000, max_feed_z=3000, max_rapid=6000),
        description="Modular 3-in-1 machine (3D print, laser, CNC)",
    ),

    # Semi-Pro CNC Routers
    "avid-pro": Machine(
        name="Avid CNC Pro",
        manufacturer="Avid CNC",
        model="PRO4848",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.MACH4,
        envelope=WorkEnvelope(x_max=1219, y_max=1219, z_min=-152, z_max=0),
        spindle=SpindleSpec(min_rpm=8000, max_rpm=24000, power_watts=2200, collet_type="ER20"),
        feeds=FeedLimits(max_feed_xy=15000, max_feed_z=7500, max_rapid=25000),
        description="Professional-grade CNC router",
    ),
    "stepcraft-d840": Machine(
        name="Stepcraft D.840",
        manufacturer="Stepcraft",
        model="D.840",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.UCCNC,
        envelope=WorkEnvelope(x_max=840, y_max=600, z_min=-140, z_max=0),
        spindle=SpindleSpec(min_rpm=5000, max_rpm=25000, power_watts=1000, collet_type="ER16"),
        feeds=FeedLimits(max_feed_xy=8000, max_feed_z=4000, max_rapid=10000),
        description="German precision desktop CNC",
    ),

    # Industrial Mills
    "tormach-440": Machine(
        name="Tormach 440",
        manufacturer="Tormach",
        model="PCNC 440",
        machine_type=MachineType.MILL_3AXIS,
        post_processor=PostProcessor.LINUXCNC,
        envelope=WorkEnvelope(x_max=254, y_max=159, z_min=-254, z_max=0),
        spindle=SpindleSpec(min_rpm=100, max_rpm=10000, power_watts=750, collet_type="R8"),
        feeds=FeedLimits(max_feed_xy=4000, max_feed_z=2000, max_rapid=5000),
        description="Personal CNC mill for metals",
    ),
    "haas-mini-mill": Machine(
        name="Haas Mini Mill",
        manufacturer="Haas",
        model="Mini Mill",
        machine_type=MachineType.MILL_3AXIS,
        post_processor=PostProcessor.HAAS,
        envelope=WorkEnvelope(x_max=406, y_max=305, z_min=-254, z_max=0),
        spindle=SpindleSpec(min_rpm=100, max_rpm=6000, power_watts=5600, collet_type="BT30"),
        feeds=FeedLimits(max_feed_xy=12700, max_feed_z=12700, max_rapid=15240),
        use_canned_cycles=True,
        description="Entry-level industrial VMC",
    ),

    # Chinese 3020/6040 style routers
    "cnc-3018": Machine(
        name="CNC 3018",
        manufacturer="Generic",
        model="CNC 3018 Pro",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.GRBL,
        envelope=WorkEnvelope(x_max=300, y_max=180, z_min=-45, z_max=0),
        spindle=SpindleSpec(min_rpm=10000, max_rpm=10000, power_watts=60, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=1500, max_feed_z=500, max_rapid=2000),
        description="Budget desktop engraver/light CNC",
    ),
    "cnc-6040": Machine(
        name="CNC 6040",
        manufacturer="Generic",
        model="CNC 6040",
        machine_type=MachineType.ROUTER_3AXIS,
        post_processor=PostProcessor.MACH3,
        envelope=WorkEnvelope(x_max=600, y_max=400, z_min=-80, z_max=0),
        spindle=SpindleSpec(min_rpm=8000, max_rpm=24000, power_watts=1500, collet_type="ER11"),
        feeds=FeedLimits(max_feed_xy=4000, max_feed_z=2000, max_rapid=6000),
        description="Chinese desktop CNC router",
    ),
}


class MachineDatabase:
    """Database of CNC machines with search and filtering."""

    def __init__(self):
        self.machines = MACHINES.copy()
        self._custom_path = Path.home() / ".pycam3d" / "machines.json"
        self._load_custom()

    def _load_custom(self) -> None:
        """Load user-defined machines from config file."""
        if self._custom_path.exists():
            try:
                with open(self._custom_path) as f:
                    custom = json.load(f)
                for key, data in custom.items():
                    self.machines[key] = self._from_dict(data)
            except Exception:
                pass

    def _from_dict(self, data: Dict) -> Machine:
        """Create Machine from dictionary."""
        return Machine(
            name=data["name"],
            manufacturer=data.get("manufacturer", "Custom"),
            model=data.get("model", ""),
            machine_type=MachineType(data.get("machine_type", "3-axis router")),
            post_processor=PostProcessor(data.get("post_processor", "grbl")),
            envelope=WorkEnvelope(**data.get("envelope", {})),
            spindle=SpindleSpec(**data.get("spindle", {})),
            feeds=FeedLimits(**data.get("feeds", {})),
            description=data.get("description", ""),
        )

    def list_all(self) -> List[str]:
        """List all machine names."""
        return list(self.machines.keys())

    def get(self, name: str) -> Optional[Machine]:
        """Get machine by name."""
        return self.machines.get(name)

    def search(self, query: str) -> List[Machine]:
        """Search machines by name, manufacturer, or description."""
        query = query.lower()
        results = []
        for machine in self.machines.values():
            if (query in machine.name.lower() or
                query in machine.manufacturer.lower() or
                query in machine.description.lower()):
                results.append(machine)
        return results

    def filter_by_type(self, machine_type: MachineType) -> List[Machine]:
        """Filter machines by type."""
        return [m for m in self.machines.values() if m.machine_type == machine_type]

    def filter_by_envelope(self, min_x: float, min_y: float, min_z: float) -> List[Machine]:
        """Filter machines that can accommodate given dimensions."""
        return [
            m for m in self.machines.values()
            if (m.envelope.x_travel >= min_x and
                m.envelope.y_travel >= min_y and
                m.envelope.z_travel >= min_z)
        ]

    def add_custom(self, key: str, machine: Machine) -> None:
        """Add a custom machine and save to config."""
        self.machines[key] = machine
        self._save_custom()

    def _save_custom(self) -> None:
        """Save custom machines to config file."""
        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom = {
            k: v.to_dict() for k, v in self.machines.items()
            if k not in MACHINES
        }
        with open(self._custom_path, 'w') as f:
            json.dump(custom, f, indent=2)


# Convenience function
def get_machine(name: str) -> Optional[Machine]:
    """Get a machine by name from the default database."""
    db = MachineDatabase()
    return db.get(name)


def list_machines() -> List[str]:
    """List all available machine names."""
    db = MachineDatabase()
    return db.list_all()
