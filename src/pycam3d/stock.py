"""
Stock and fixture management for PyCAM3D.

Provides:
- Stock definition (rectangular, cylindrical, from mesh)
- Workholding/fixture management
- Collision avoidance zones
- Safe area calculation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import numpy as np
import math


class StockType(Enum):
    """Type of stock material."""
    RECTANGULAR = "rectangular"
    CYLINDRICAL = "cylindrical"
    FROM_MESH = "from_mesh"


class WorkholdingType(Enum):
    """Type of workholding."""
    CLAMPS = "clamps"
    VISE = "vise"
    VACUUM = "vacuum"
    TAPE = "double_sided_tape"
    SCREWS = "screws"
    T_SLOTS = "t_slots"
    FIXTURE_PLATE = "fixture_plate"


@dataclass
class Rectangle:
    """2D rectangle for stock and fixture footprints."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersects(self, other: Rectangle) -> bool:
        return not (self.x_max < other.x_min or
                   self.x_min > other.x_max or
                   self.y_max < other.y_min or
                   self.y_min > other.y_max)


@dataclass
class Stock:
    """Definition of raw stock material."""
    stock_type: StockType
    length: float  # X dimension in mm
    width: float   # Y dimension in mm
    height: float  # Z dimension in mm
    material_name: str = "unknown"

    # Offset from machine origin
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0  # Usually 0 = top of stock

    # For cylindrical stock
    diameter: Optional[float] = None

    # Allowances
    top_allowance: float = 0.0   # Extra material above part
    side_allowance: float = 0.0  # Extra material on sides

    @classmethod
    def rectangular(cls, length: float, width: float, height: float,
                   material: str = "unknown") -> Stock:
        """Create rectangular stock."""
        return cls(
            stock_type=StockType.RECTANGULAR,
            length=length,
            width=width,
            height=height,
            material_name=material,
        )

    @classmethod
    def cylindrical(cls, diameter: float, height: float,
                   material: str = "unknown") -> Stock:
        """Create cylindrical stock."""
        return cls(
            stock_type=StockType.CYLINDRICAL,
            length=diameter,
            width=diameter,
            height=height,
            diameter=diameter,
            material_name=material,
        )

    @classmethod
    def from_part_bounds(cls, bounds_min: np.ndarray, bounds_max: np.ndarray,
                        allowance: float = 2.0, material: str = "unknown") -> Stock:
        """Create stock from part bounding box with allowance."""
        size = bounds_max - bounds_min
        return cls(
            stock_type=StockType.RECTANGULAR,
            length=size[0] + 2 * allowance,
            width=size[1] + 2 * allowance,
            height=size[2] + allowance,  # Top only
            side_allowance=allowance,
            top_allowance=allowance,
            material_name=material,
        )

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get stock bounding box."""
        min_pt = np.array([self.origin_x, self.origin_y, self.origin_z - self.height])
        max_pt = np.array([self.origin_x + self.length,
                          self.origin_y + self.width,
                          self.origin_z])
        return min_pt, max_pt

    @property
    def center(self) -> np.ndarray:
        """Get stock center point."""
        min_pt, max_pt = self.bounds
        return (min_pt + max_pt) / 2

    @property
    def volume(self) -> float:
        """Get stock volume in mm³."""
        if self.stock_type == StockType.CYLINDRICAL:
            return math.pi * (self.diameter / 2) ** 2 * self.height
        return self.length * self.width * self.height

    def to_dict(self) -> Dict:
        return {
            "type": self.stock_type.value,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "material": self.material_name,
            "origin": [self.origin_x, self.origin_y, self.origin_z],
        }


@dataclass
class Clamp:
    """A single clamp or fixture element."""
    name: str
    x: float  # Center X position
    y: float  # Center Y position
    width: float   # Clamp width (X)
    depth: float   # Clamp depth (Y)
    height: float  # Clamp height above stock surface
    clearance: float = 5.0  # Extra clearance around clamp

    @property
    def footprint(self) -> Rectangle:
        """Get clamp footprint with clearance."""
        return Rectangle(
            x_min=self.x - self.width / 2 - self.clearance,
            y_min=self.y - self.depth / 2 - self.clearance,
            x_max=self.x + self.width / 2 + self.clearance,
            y_max=self.y + self.depth / 2 + self.clearance,
        )

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point collides with this clamp."""
        if z < 0:  # Below stock surface, no collision
            return False
        if z > self.height:  # Above clamp height, no collision
            return False
        return self.footprint.contains_point(x, y)


@dataclass
class AvoidanceZone:
    """Zone to avoid during machining (soft limits)."""
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    z_min: float = -1000.0  # Full depth by default
    z_max: float = 1000.0
    reason: str = ""

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if point is in avoidance zone."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)


@dataclass
class WorkholdingSetup:
    """Complete workholding configuration."""
    workholding_type: WorkholdingType
    clamps: List[Clamp] = field(default_factory=list)
    avoidance_zones: List[AvoidanceZone] = field(default_factory=list)
    notes: str = ""

    def add_clamp(self, clamp: Clamp) -> None:
        """Add a clamp to the setup."""
        self.clamps.append(clamp)
        # Auto-create avoidance zone for clamp
        fp = clamp.footprint
        self.avoidance_zones.append(AvoidanceZone(
            name=f"clamp_{clamp.name}",
            x_min=fp.x_min, y_min=fp.y_min,
            x_max=fp.x_max, y_max=fp.y_max,
            z_min=0, z_max=clamp.height + 10,
            reason=f"Clamp: {clamp.name}",
        ))

    def add_corner_clamps(self, stock: Stock, clamp_size: float = 30.0,
                         offset: float = 10.0) -> None:
        """Add clamps at stock corners."""
        corners = [
            ("front_left", offset, offset),
            ("front_right", stock.length - offset, offset),
            ("back_left", offset, stock.width - offset),
            ("back_right", stock.length - offset, stock.width - offset),
        ]
        for name, x, y in corners:
            self.add_clamp(Clamp(
                name=name,
                x=stock.origin_x + x,
                y=stock.origin_y + y,
                width=clamp_size,
                depth=clamp_size,
                height=25.0,
            ))

    def add_side_clamps(self, stock: Stock, clamp_width: float = 40.0) -> None:
        """Add clamps on stock sides."""
        mid_x = stock.origin_x + stock.length / 2
        mid_y = stock.origin_y + stock.width / 2

        sides = [
            ("left", stock.origin_x - clamp_width / 2 - 5, mid_y),
            ("right", stock.origin_x + stock.length + clamp_width / 2 + 5, mid_y),
            ("front", mid_x, stock.origin_y - clamp_width / 2 - 5),
            ("back", mid_x, stock.origin_y + stock.width + clamp_width / 2 + 5),
        ]
        for name, x, y in sides:
            self.add_clamp(Clamp(
                name=name,
                x=x,
                y=y,
                width=clamp_width,
                depth=clamp_width,
                height=25.0,
            ))

    def check_collision(self, x: float, y: float, z: float) -> Optional[str]:
        """Check if point collides with any clamp or avoidance zone.

        Returns:
            Collision reason string, or None if no collision
        """
        for clamp in self.clamps:
            if clamp.contains_point(x, y, z):
                return f"Collision with clamp: {clamp.name}"

        for zone in self.avoidance_zones:
            if zone.contains_point(x, y, z):
                return zone.reason or f"Avoidance zone: {zone.name}"

        return None

    def get_safe_area(self, stock: Stock) -> Rectangle:
        """Get the safe machinable area after excluding clamps."""
        # Start with full stock area
        safe = Rectangle(
            x_min=stock.origin_x,
            y_min=stock.origin_y,
            x_max=stock.origin_x + stock.length,
            y_max=stock.origin_y + stock.width,
        )

        # Shrink based on clamp positions
        for clamp in self.clamps:
            fp = clamp.footprint
            # If clamp overlaps safe area, shrink it
            if fp.intersects(safe):
                # Determine which side to shrink
                if fp.x_min <= safe.x_min:
                    safe.x_min = max(safe.x_min, fp.x_max)
                if fp.x_max >= safe.x_max:
                    safe.x_max = min(safe.x_max, fp.x_min)
                if fp.y_min <= safe.y_min:
                    safe.y_min = max(safe.y_min, fp.y_max)
                if fp.y_max >= safe.y_max:
                    safe.y_max = min(safe.y_max, fp.y_min)

        return safe


class StockManager:
    """Manages stock and workholding for a job."""

    def __init__(self, stock: Stock):
        self.stock = stock
        self.workholding: Optional[WorkholdingSetup] = None
        self.safe_z: float = 10.0  # Retract height

    def set_workholding(self, setup: WorkholdingSetup) -> None:
        """Set workholding configuration."""
        self.workholding = setup

    def auto_workholding(self, method: WorkholdingType = WorkholdingType.CLAMPS) -> WorkholdingSetup:
        """Automatically generate workholding setup."""
        setup = WorkholdingSetup(workholding_type=method)

        if method == WorkholdingType.CLAMPS:
            setup.add_corner_clamps(self.stock)
        elif method == WorkholdingType.VISE:
            # Add vise jaws on left/right
            setup.add_clamp(Clamp(
                name="vise_fixed",
                x=self.stock.origin_x - 20,
                y=self.stock.origin_y + self.stock.width / 2,
                width=40, depth=self.stock.width + 40, height=30,
            ))
            setup.add_clamp(Clamp(
                name="vise_movable",
                x=self.stock.origin_x + self.stock.length + 20,
                y=self.stock.origin_y + self.stock.width / 2,
                width=40, depth=self.stock.width + 40, height=30,
            ))
        elif method in [WorkholdingType.VACUUM, WorkholdingType.TAPE]:
            # No clamps needed, full area accessible
            pass

        self.workholding = setup
        return setup

    def validate_toolpath(self, points: List[Tuple[float, float, float]],
                         tool_diameter: float) -> List[str]:
        """Validate toolpath against stock and workholding.

        Returns:
            List of error/warning messages
        """
        errors = []
        stock_min, stock_max = self.stock.bounds
        tool_radius = tool_diameter / 2

        for i, (x, y, z) in enumerate(points):
            # Check stock bounds
            if x - tool_radius < stock_min[0] or x + tool_radius > stock_max[0]:
                errors.append(f"Point {i}: X={x:.2f} outside stock bounds")
            if y - tool_radius < stock_min[1] or y + tool_radius > stock_max[1]:
                errors.append(f"Point {i}: Y={y:.2f} outside stock bounds")
            if z < stock_min[2]:
                errors.append(f"Point {i}: Z={z:.2f} cuts through stock bottom")

            # Check workholding collisions
            if self.workholding:
                collision = self.workholding.check_collision(x, y, z + self.safe_z)
                if collision:
                    errors.append(f"Point {i}: {collision}")

        return errors

    def get_machinable_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the machinable area bounds."""
        stock_min, stock_max = self.stock.bounds

        if self.workholding:
            safe_rect = self.workholding.get_safe_area(self.stock)
            return (
                np.array([safe_rect.x_min, safe_rect.y_min, stock_min[2]]),
                np.array([safe_rect.x_max, safe_rect.y_max, stock_max[2]]),
            )

        return stock_min, stock_max

    def suggest_stock_size(self, part_bounds_min: np.ndarray,
                          part_bounds_max: np.ndarray,
                          workholding: WorkholdingType) -> Stock:
        """Suggest optimal stock size for part and workholding method."""
        part_size = part_bounds_max - part_bounds_min

        # Base allowance depends on workholding
        allowance = {
            WorkholdingType.CLAMPS: 30.0,  # Need room for clamps
            WorkholdingType.VISE: 20.0,    # Need vise grip area
            WorkholdingType.VACUUM: 5.0,   # Minimal allowance
            WorkholdingType.TAPE: 5.0,
            WorkholdingType.SCREWS: 15.0,  # Room for screw holes
        }.get(workholding, 20.0)

        return Stock.rectangular(
            length=part_size[0] + 2 * allowance,
            width=part_size[1] + 2 * allowance,
            height=part_size[2] + 5,  # 5mm top allowance
            material=self.stock.material_name,
        )

    def to_dict(self) -> Dict:
        result = {
            "stock": self.stock.to_dict(),
            "safe_z": self.safe_z,
        }
        if self.workholding:
            result["workholding"] = {
                "type": self.workholding.workholding_type.value,
                "clamps": [
                    {"name": c.name, "x": c.x, "y": c.y,
                     "width": c.width, "depth": c.depth, "height": c.height}
                    for c in self.workholding.clamps
                ],
            }
        return result
