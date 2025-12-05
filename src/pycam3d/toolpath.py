"""
Toolpath generation module using OpenCAMLib.

This module provides:
- Drop-cutter operations (project tool down to surface)
- Waterline/push-cutter operations (constant Z contours)
- Various tool types (ball, flat, bull nose, conical)
- Toolpath strategies (parallel, spiral, adaptive)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _import_ocl():
    """Import opencamlib with proper error handling."""
    try:
        import opencamlib as ocl
        return ocl
    except ImportError:
        raise ImportError(
            "OpenCAMLib is required. Install with: pip install opencamlib"
        )


class ToolType(Enum):
    """Available tool types."""

    BALL = "ball"  # Ball nose end mill
    FLAT = "flat"  # Flat end mill (cylindrical)
    BULL = "bull"  # Bull nose (toroidal)
    CONE = "cone"  # Conical/tapered
    BALL_CONE = "ball_cone"  # Ball with conical shaft


class Strategy(Enum):
    """Toolpath strategies."""

    PARALLEL_X = "parallel_x"  # Lines parallel to X axis
    PARALLEL_Y = "parallel_y"  # Lines parallel to Y axis
    PARALLEL_45 = "parallel_45"  # Lines at 45 degrees
    ZIGZAG_X = "zigzag_x"  # Alternating direction X
    ZIGZAG_Y = "zigzag_y"  # Alternating direction Y
    WATERLINE = "waterline"  # Constant Z contours
    SPIRAL = "spiral"  # Spiral from outside in


@dataclass
class Tool:
    """
    CNC cutting tool definition.

    Attributes:
        type: Type of tool (ball, flat, bull, cone).
        diameter: Tool diameter in mm.
        length: Tool cutting length in mm.
        corner_radius: Corner radius for bull nose tools.
        angle: Taper angle for conical tools (degrees).
    """

    type: ToolType
    diameter: float
    length: float = 50.0
    corner_radius: float = 0.0
    angle: float = 0.0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.type.value}_{self.diameter}mm"

    @classmethod
    def ball(cls, diameter: float, length: float = 50.0) -> Tool:
        """Create a ball nose end mill."""
        return cls(type=ToolType.BALL, diameter=diameter, length=length)

    @classmethod
    def flat(cls, diameter: float, length: float = 50.0) -> Tool:
        """Create a flat end mill."""
        return cls(type=ToolType.FLAT, diameter=diameter, length=length)

    @classmethod
    def bull(cls, diameter: float, corner_radius: float, length: float = 50.0) -> Tool:
        """Create a bull nose (toroidal) end mill."""
        return cls(
            type=ToolType.BULL, diameter=diameter, corner_radius=corner_radius, length=length
        )

    @classmethod
    def cone(cls, diameter: float, angle: float, length: float = 50.0) -> Tool:
        """Create a conical/tapered tool."""
        return cls(type=ToolType.CONE, diameter=diameter, angle=angle, length=length)


@dataclass
class ToolpathPoint:
    """A single point in a toolpath."""

    x: float
    y: float
    z: float
    feed_rate: Optional[float] = None
    rapid: bool = False

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Toolpath:
    """
    A complete toolpath consisting of multiple points.

    Attributes:
        points: List of toolpath points.
        tool: The tool used for this toolpath.
        strategy: The strategy used to generate this toolpath.
        feed_rate: Default feed rate in mm/min.
        plunge_rate: Plunge feed rate in mm/min.
        safe_z: Safe Z height for rapids.
    """

    points: list[ToolpathPoint] = field(default_factory=list)
    tool: Optional[Tool] = None
    strategy: Optional[Strategy] = None
    feed_rate: float = 1000.0
    plunge_rate: float = 300.0
    safe_z: float = 10.0

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[ToolpathPoint]:
        return iter(self.points)

    def add_point(self, x: float, y: float, z: float, rapid: bool = False) -> None:
        """Add a point to the toolpath."""
        self.points.append(ToolpathPoint(x=x, y=y, z=z, rapid=rapid))

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the toolpath."""
        if not self.points:
            return np.zeros(3), np.zeros(3)

        coords = np.array([p.as_tuple() for p in self.points])
        return coords.min(axis=0), coords.max(axis=0)

    def get_total_length(self) -> float:
        """Calculate total toolpath length."""
        if len(self.points) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.points)):
            p1 = self.points[i - 1].as_array()
            p2 = self.points[i].as_array()
            total += np.linalg.norm(p2 - p1)

        return total

    def optimize(self) -> None:
        """Remove redundant points from toolpath."""
        if len(self.points) < 3:
            return

        optimized = [self.points[0]]
        for i in range(1, len(self.points) - 1):
            p0, p1, p2 = self.points[i - 1], self.points[i], self.points[i + 1]

            # Check if p1 is collinear with p0 and p2
            v1 = p1.as_array() - p0.as_array()
            v2 = p2.as_array() - p1.as_array()

            # Keep point if direction changes significantly
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                if np.dot(v1_norm, v2_norm) < 0.9999:
                    optimized.append(p1)
            else:
                optimized.append(p1)

        optimized.append(self.points[-1])
        removed = len(self.points) - len(optimized)

        if removed > 0:
            logger.info(f"Optimized toolpath: removed {removed} redundant points")
            self.points = optimized


class ToolpathGenerator:
    """
    Generate toolpaths using OpenCAMLib.

    This class provides methods for generating various types of toolpaths
    including drop-cutter (3D surfacing) and waterline (constant Z) operations.
    """

    def __init__(self):
        self._stl_surface = None
        self._bounds: Optional[tuple[np.ndarray, np.ndarray]] = None

    def load_stl(self, filepath: str | Path) -> None:
        """
        Load an STL file for toolpath generation.

        Args:
            filepath: Path to the STL file.
        """
        ocl = _import_ocl()

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"STL file not found: {filepath}")

        logger.info(f"Loading STL surface from {filepath}")

        # Use STLReader to load the STL
        self._stl_surface = ocl.STLSurf()
        reader = ocl.STLReader(str(filepath), self._stl_surface)

        # Get bounds from the STL using trimesh
        import trimesh
        mesh = trimesh.load(filepath)
        self._bounds = (mesh.bounds[0], mesh.bounds[1])

        logger.info(
            f"Loaded STL with bounds: {self._bounds[0]} to {self._bounds[1]}"
        )

    def load_from_mesh(self, mesh) -> None:
        """
        Load from a trimesh.Trimesh object.

        Args:
            mesh: A trimesh.Trimesh object.
        """
        import tempfile

        # Export mesh to temporary STL file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            mesh.export(f.name)
            temp_path = f.name

        self.load_stl(temp_path)

        # Clean up
        Path(temp_path).unlink()

    def _create_ocl_cutter(self, tool: Tool):
        """Create an OpenCAMLib cutter from a Tool definition."""
        ocl = _import_ocl()

        if tool.type == ToolType.BALL:
            return ocl.BallCutter(tool.diameter, tool.length)
        elif tool.type == ToolType.FLAT:
            return ocl.CylCutter(tool.diameter, tool.length)
        elif tool.type == ToolType.BULL:
            return ocl.BullCutter(tool.diameter, tool.corner_radius, tool.length)
        elif tool.type == ToolType.CONE:
            return ocl.ConeCutter(tool.diameter, tool.angle, tool.length)
        else:
            raise ValueError(f"Unsupported tool type: {tool.type}")

    def generate_parallel(
        self,
        tool: Tool,
        stepover: float,
        strategy: Strategy = Strategy.PARALLEL_X,
        margin: float = 0.0,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate parallel toolpaths using drop-cutter.

        Args:
            tool: Tool to use.
            stepover: Distance between passes (typically 10-50% of tool diameter).
            strategy: Direction strategy (PARALLEL_X, PARALLEL_Y, ZIGZAG_X, ZIGZAG_Y).
            margin: Extra margin around the part.
            z_safe: Safe Z height for rapids.

        Returns:
            Generated toolpath.
        """
        ocl = _import_ocl()

        if self._stl_surface is None:
            raise ValueError("No STL surface loaded")

        logger.info(f"Generating parallel toolpath with {tool.name}, stepover={stepover}mm")

        cutter = self._create_ocl_cutter(tool)
        bounds_min, bounds_max = self._bounds

        toolpath = Toolpath(tool=tool, strategy=strategy, safe_z=z_safe)

        # Determine scan direction
        if strategy in (Strategy.PARALLEL_X, Strategy.ZIGZAG_X):
            primary_axis = "y"
            primary_range = np.arange(
                bounds_min[1] - margin, bounds_max[1] + margin + stepover, stepover
            )
            secondary_range = np.linspace(
                bounds_min[0] - margin, bounds_max[0] + margin, 100
            )
        else:
            primary_axis = "x"
            primary_range = np.arange(
                bounds_min[0] - margin, bounds_max[0] + margin + stepover, stepover
            )
            secondary_range = np.linspace(
                bounds_min[1] - margin, bounds_max[1] + margin, 100
            )

        zigzag = strategy in (Strategy.ZIGZAG_X, Strategy.ZIGZAG_Y)
        z_min = bounds_min[2] - 10  # Floor below the part

        # Generate toolpath line by line using PathDropCutter
        for i, primary in enumerate(primary_range):
            # Create path drop cutter for this line
            pdc = ocl.PathDropCutter()
            pdc.setSTL(self._stl_surface)
            pdc.setCutter(cutter)
            pdc.setZ(z_min)
            pdc.setSampling(0.5)  # Sampling along the path

            # Calculate line endpoints
            line_secondary = secondary_range if not (zigzag and i % 2) else secondary_range[::-1]

            if primary_axis == "y":
                start_x, start_y = line_secondary[0], primary
                end_x, end_y = line_secondary[-1], primary
            else:
                start_x, start_y = primary, line_secondary[0]
                end_x, end_y = primary, line_secondary[-1]

            # Create path with single line
            path = ocl.Path()
            start_pt = ocl.Point(start_x, start_y, bounds_max[2] + 50)
            end_pt = ocl.Point(end_x, end_y, bounds_max[2] + 50)
            line = ocl.Line(start_pt, end_pt)
            path.append(line)

            pdc.setPath(path)
            pdc.run()

            # Get results
            cl_points = pdc.getCLPoints()

            # Add rapid to start of line
            if cl_points:
                first_pt = cl_points[0]
                toolpath.add_point(first_pt.x, first_pt.y, z_safe, rapid=True)

                # Add cutting points
                for cl_pt in cl_points:
                    toolpath.add_point(cl_pt.x, cl_pt.y, cl_pt.z)

                # Retract at end of line
                last_pt = cl_points[-1]
                toolpath.add_point(last_pt.x, last_pt.y, z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} toolpath points")

        return toolpath

    def generate_waterline(
        self,
        tool: Tool,
        z_step: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        sampling: float = 0.1,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate waterline (constant Z) toolpaths.

        Args:
            tool: Tool to use.
            z_step: Vertical step between passes.
            z_min: Minimum Z height (defaults to part bottom).
            z_max: Maximum Z height (defaults to part top).
            sampling: Sampling resolution.
            z_safe: Safe Z height for rapids.

        Returns:
            Generated toolpath.
        """
        ocl = _import_ocl()

        if self._stl_surface is None:
            raise ValueError("No STL surface loaded")

        logger.info(f"Generating waterline toolpath with {tool.name}, z_step={z_step}mm")

        cutter = self._create_ocl_cutter(tool)
        bounds_min, bounds_max = self._bounds

        if z_min is None:
            z_min = bounds_min[2]
        if z_max is None:
            z_max = bounds_max[2]

        toolpath = Toolpath(tool=tool, strategy=Strategy.WATERLINE, safe_z=z_safe)

        # Generate waterlines at each Z level
        z_levels = np.arange(z_min + z_step, z_max, z_step)

        for z in z_levels:
            logger.debug(f"Generating waterline at Z={z:.2f}")

            # Create waterline operation
            wl = ocl.Waterline()
            wl.setSTL(self._stl_surface)
            wl.setCutter(cutter)
            wl.setZ(z)
            wl.setSampling(sampling)

            wl.run()

            # Get resulting loops
            loops = wl.getLoops()

            for loop in loops:
                if len(loop) < 2:
                    continue

                # Rapid to start of loop
                first_point = loop[0]
                toolpath.add_point(first_point.x, first_point.y, z_safe, rapid=True)
                toolpath.add_point(first_point.x, first_point.y, z, rapid=False)

                # Follow the loop
                for point in loop[1:]:
                    toolpath.add_point(point.x, point.y, z)

                # Close the loop
                toolpath.add_point(first_point.x, first_point.y, z)

                # Retract
                toolpath.add_point(first_point.x, first_point.y, z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} toolpath points")

        return toolpath

    def generate_roughing(
        self,
        tool: Tool,
        stepover: float,
        z_step: float,
        stock_to_leave: float = 0.5,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate roughing toolpaths (layered parallel passes).

        Args:
            tool: Tool to use.
            stepover: XY stepover between passes.
            z_step: Z step between layers.
            stock_to_leave: Material to leave for finishing pass.
            z_safe: Safe Z height for rapids.

        Returns:
            Generated toolpath.
        """
        ocl = _import_ocl()

        if self._stl_surface is None:
            raise ValueError("No STL surface loaded")

        logger.info(
            f"Generating roughing toolpath with {tool.name}, "
            f"stepover={stepover}mm, z_step={z_step}mm"
        )

        bounds_min, bounds_max = self._bounds

        toolpath = Toolpath(tool=tool, strategy=Strategy.ZIGZAG_X, safe_z=z_safe)

        # Create offset cutter for stock to leave
        rough_tool = Tool(
            type=tool.type,
            diameter=tool.diameter + 2 * stock_to_leave,
            length=tool.length,
            corner_radius=tool.corner_radius + stock_to_leave if tool.corner_radius else 0,
        )
        rough_cutter = self._create_ocl_cutter(rough_tool)

        # Generate Z levels from top to bottom
        z_levels = np.arange(bounds_max[2], bounds_min[2] - z_step, -z_step)

        x_range = np.arange(bounds_min[0] - stepover, bounds_max[0] + stepover * 2, stepover)
        y_samples = 50

        z_min = bounds_min[2] - 10  # Floor below the part

        for layer_idx, z_target in enumerate(z_levels):
            logger.debug(f"Generating roughing layer at Z={z_target:.2f}")

            for i, x in enumerate(x_range):
                y_start = bounds_min[1]
                y_end = bounds_max[1]
                if i % 2 == 1:
                    y_start, y_end = y_end, y_start

                # Create path drop cutter for this line
                pdc = ocl.PathDropCutter()
                pdc.setSTL(self._stl_surface)
                pdc.setCutter(rough_cutter)
                pdc.setZ(z_min)
                pdc.setSampling(stepover / 2)

                # Create path
                path = ocl.Path()
                start_pt = ocl.Point(x, y_start, bounds_max[2] + 50)
                end_pt = ocl.Point(x, y_end, bounds_max[2] + 50)
                line = ocl.Line(start_pt, end_pt)
                path.append(line)

                pdc.setPath(path)
                pdc.run()
                cl_points = pdc.getCLPoints()

                if not cl_points:
                    continue

                # Rapid to start
                first_pt = cl_points[0]
                toolpath.add_point(first_pt.x, first_pt.y, z_safe, rapid=True)

                for cl_pt in cl_points:
                    # Use the higher of drop-cutter result or current layer Z
                    z = max(cl_pt.z, z_target)
                    toolpath.add_point(cl_pt.x, cl_pt.y, z)

                # Retract
                last_pt = cl_points[-1]
                toolpath.add_point(last_pt.x, last_pt.y, z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} toolpath points")

        return toolpath

    def generate_finishing(
        self,
        tool: Tool,
        stepover: float,
        strategy: Strategy = Strategy.PARALLEL_X,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate finishing toolpath (fine surface pass).

        This is essentially a parallel pass with smaller stepover.

        Args:
            tool: Tool to use (typically ball nose).
            stepover: Small stepover for fine finish.
            strategy: Direction strategy.
            z_safe: Safe Z height.

        Returns:
            Generated toolpath.
        """
        logger.info(f"Generating finishing toolpath with {tool.name}")

        return self.generate_parallel(
            tool=tool, stepover=stepover, strategy=strategy, z_safe=z_safe
        )
