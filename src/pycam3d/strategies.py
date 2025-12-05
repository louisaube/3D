"""
Advanced toolpath strategies module.

This module implements state-of-the-art CAM algorithms:
- Iso-Scallop: Adaptive stepover for constant scallop height
- Trochoidal: Constant engagement angle milling
- Spiral: Continuous spiral contour paths
- Adaptive Clearing: Voronoi-based pocketing (via OpenVoronoi)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from pycam3d.curvature import (
    CurvatureAnalyzer,
    CurvatureField,
    compute_scallop_height,
    compute_stepover_for_scallop,
)
from pycam3d.toolpath import Tool, Toolpath, ToolpathPoint, Strategy

logger = logging.getLogger(__name__)


def _import_ocl():
    """Import opencamlib."""
    try:
        import opencamlib as ocl
        return ocl
    except ImportError:
        raise ImportError("OpenCAMLib required: pip install opencamlib")


# =============================================================================
# ISO-SCALLOP STRATEGY
# =============================================================================

class IsoScallopGenerator:
    """
    Generate toolpaths with constant scallop height.

    Unlike fixed stepover, iso-scallop adapts the distance between
    passes based on local surface curvature to maintain uniform
    surface finish quality.

    Benefits:
    - 7-21% shorter toolpath vs fixed stepover
    - Consistent surface finish across varying curvature
    - Optimal material removal
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        stl_surface,  # OpenCAMLib STLSurf
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        self.mesh = mesh
        self.stl_surface = stl_surface
        self.bounds = bounds
        self._curvature_analyzer = CurvatureAnalyzer(mesh)
        self._curvature_field: Optional[CurvatureField] = None

    def generate(
        self,
        tool: Tool,
        target_scallop: float,
        min_stepover: Optional[float] = None,
        max_stepover: Optional[float] = None,
        z_safe: float = 10.0,
        direction: str = "x",  # 'x', 'y', or 'both'
    ) -> Toolpath:
        """
        Generate iso-scallop toolpath.

        Args:
            tool: Cutting tool (should be ball-end for best results).
            target_scallop: Desired scallop height in mm.
            min_stepover: Minimum allowed stepover (default: 5% of diameter).
            max_stepover: Maximum allowed stepover (default: 50% of diameter).
            z_safe: Safe Z height for rapids.
            direction: Primary cutting direction.

        Returns:
            Adaptive toolpath with constant scallop height.
        """
        ocl = _import_ocl()

        logger.info(f"Generating iso-scallop toolpath, target scallop={target_scallop}mm")

        # Defaults
        tool_radius = tool.diameter / 2
        if min_stepover is None:
            min_stepover = tool.diameter * 0.05
        if max_stepover is None:
            max_stepover = tool.diameter * 0.5

        # Compute curvature field
        if self._curvature_field is None:
            self._curvature_field = self._curvature_analyzer.compute_curvature()

        bounds_min, bounds_max = self.bounds
        toolpath = Toolpath(tool=tool, strategy=Strategy.PARALLEL_X, safe_z=z_safe)

        # Create cutter
        cutter = self._create_cutter(tool, ocl)

        # Generate adaptive passes
        if direction in ("x", "both"):
            self._generate_direction(
                toolpath, cutter, ocl,
                primary_axis="y",
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                tool_radius=tool_radius,
                target_scallop=target_scallop,
                min_stepover=min_stepover,
                max_stepover=max_stepover,
                z_safe=z_safe,
            )

        if direction in ("y", "both"):
            self._generate_direction(
                toolpath, cutter, ocl,
                primary_axis="x",
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                tool_radius=tool_radius,
                target_scallop=target_scallop,
                min_stepover=min_stepover,
                max_stepover=max_stepover,
                z_safe=z_safe,
            )

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} iso-scallop points")

        return toolpath

    def _generate_direction(
        self,
        toolpath: Toolpath,
        cutter,
        ocl,
        primary_axis: str,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        tool_radius: float,
        target_scallop: float,
        min_stepover: float,
        max_stepover: float,
        z_safe: float,
    ):
        """Generate passes in one direction with adaptive stepover."""
        z_min = bounds_min[2] - 10

        if primary_axis == "y":
            primary_start, primary_end = bounds_min[1], bounds_max[1]
            secondary_start, secondary_end = bounds_min[0], bounds_max[0]
        else:
            primary_start, primary_end = bounds_min[0], bounds_max[0]
            secondary_start, secondary_end = bounds_min[1], bounds_max[1]

        # Start first pass
        current_primary = primary_start
        pass_num = 0

        while current_primary <= primary_end:
            # Create path for this pass
            pdc = ocl.PathDropCutter()
            pdc.setSTL(self.stl_surface)
            pdc.setCutter(cutter)
            pdc.setZ(z_min)
            pdc.setSampling(0.5)

            path = ocl.Path()
            if primary_axis == "y":
                start_pt = ocl.Point(secondary_start, current_primary, bounds_max[2] + 50)
                end_pt = ocl.Point(secondary_end, current_primary, bounds_max[2] + 50)
            else:
                start_pt = ocl.Point(current_primary, secondary_start, bounds_max[2] + 50)
                end_pt = ocl.Point(current_primary, secondary_end, bounds_max[2] + 50)

            # Alternate direction (zigzag)
            if pass_num % 2 == 1:
                start_pt, end_pt = end_pt, start_pt

            line = ocl.Line(start_pt, end_pt)
            path.append(line)

            pdc.setPath(path)
            pdc.run()
            cl_points = pdc.getCLPoints()

            if cl_points:
                # Add rapid and cutting moves
                first_pt = cl_points[0]
                toolpath.add_point(first_pt.x, first_pt.y, z_safe, rapid=True)

                for cl_pt in cl_points:
                    toolpath.add_point(cl_pt.x, cl_pt.y, cl_pt.z)

                last_pt = cl_points[-1]
                toolpath.add_point(last_pt.x, last_pt.y, z_safe, rapid=True)

                # Compute adaptive stepover for next pass
                # Sample curvature along this pass
                avg_curvature = self._sample_curvature_along_pass(cl_points)
                next_stepover = compute_stepover_for_scallop(
                    tool_radius, target_scallop, avg_curvature
                )
                next_stepover = np.clip(next_stepover, min_stepover, max_stepover)
            else:
                # No contact, use max stepover
                next_stepover = max_stepover

            current_primary += next_stepover
            pass_num += 1

    def _sample_curvature_along_pass(self, cl_points) -> float:
        """Sample average curvature along a toolpath pass."""
        if self._curvature_field is None:
            return 0.0

        curvatures = []
        for pt in cl_points[::10]:  # Sample every 10th point
            point = np.array([pt.x, pt.y, pt.z])
            # Find closest vertex
            distances = np.linalg.norm(
                self._curvature_field.vertices - point, axis=1
            )
            closest_idx = np.argmin(distances)
            k = self._curvature_field.max_curvature[closest_idx]
            curvatures.append(k)

        return np.mean(curvatures) if curvatures else 0.0

    def _create_cutter(self, tool: Tool, ocl):
        """Create OpenCAMLib cutter."""
        from pycam3d.toolpath import ToolType

        if tool.type == ToolType.BALL:
            return ocl.BallCutter(tool.diameter, tool.length)
        elif tool.type == ToolType.FLAT:
            return ocl.CylCutter(tool.diameter, tool.length)
        elif tool.type == ToolType.BULL:
            return ocl.BullCutter(tool.diameter, tool.corner_radius, tool.length)
        else:
            return ocl.BallCutter(tool.diameter, tool.length)


# =============================================================================
# TROCHOIDAL MILLING STRATEGY
# =============================================================================

@dataclass
class TrochoidalParams:
    """Parameters for trochoidal milling."""

    # Circle parameters
    trochoidal_diameter: float  # Diameter of trochoidal circles
    stepover: float  # Forward step per circle

    # Engagement control
    max_engagement_angle: float = 90.0  # Maximum tool engagement (degrees)
    radial_depth: float = 0.5  # Radial depth of cut as fraction of diameter

    # Speed
    helix_angle: float = 2.0  # Helix ramp angle for plunge (degrees)


class TrochoidalGenerator:
    """
    Generate trochoidal (constant engagement) toolpaths.

    Trochoidal milling uses circular/spiral patterns to maintain
    constant tool engagement angle, reducing:
    - Cutting forces and vibration
    - Heat buildup
    - Tool wear

    This enables higher feed rates and longer tool life.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        stl_surface,
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        self.mesh = mesh
        self.stl_surface = stl_surface
        self.bounds = bounds

    def generate_slot(
        self,
        tool: Tool,
        start: Tuple[float, float],
        end: Tuple[float, float],
        z_bottom: float,
        z_top: float,
        params: Optional[TrochoidalParams] = None,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate trochoidal toolpath for a slot.

        Args:
            tool: Cutting tool.
            start: Start point (x, y).
            end: End point (x, y).
            z_bottom: Bottom Z of slot.
            z_top: Top Z (stock surface).
            params: Trochoidal parameters.
            z_safe: Safe Z height.

        Returns:
            Trochoidal toolpath.
        """
        logger.info("Generating trochoidal slot toolpath")

        if params is None:
            params = TrochoidalParams(
                trochoidal_diameter=tool.diameter * 0.8,
                stepover=tool.diameter * 0.1,
            )

        toolpath = Toolpath(tool=tool, safe_z=z_safe)

        # Direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)

        if length < 1e-6:
            return toolpath

        # Unit direction and perpendicular
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # Perpendicular

        # Trochoidal parameters
        radius = params.trochoidal_diameter / 2
        step = params.stepover
        n_circles = int(length / step) + 1

        # Z layers
        z_step = tool.diameter * params.radial_depth
        z_levels = np.arange(z_top, z_bottom - z_step, -z_step)

        # Rapid to start
        toolpath.add_point(start[0], start[1], z_safe, rapid=True)

        for z in z_levels:
            # Generate trochoidal circles along the path
            t = 0.0
            circle_idx = 0

            while t <= length:
                # Center of current circle
                cx = start[0] + t * ux
                cy = start[1] + t * uy

                # Generate circle points
                n_pts = 36  # Points per circle
                for i in range(n_pts + 1):
                    angle = 2 * math.pi * i / n_pts
                    x = cx + radius * math.cos(angle) * px + radius * math.sin(angle) * ux
                    y = cy + radius * math.cos(angle) * py + radius * math.sin(angle) * uy

                    # For first point of first circle at this Z, use helix entry
                    if circle_idx == 0 and i == 0:
                        toolpath.add_point(x, y, z_safe, rapid=True)
                        # Helix down
                        helix_pts = 18
                        z_start = z_safe if z == z_levels[0] else z + z_step
                        for h in range(helix_pts):
                            ha = 2 * math.pi * h / helix_pts
                            hx = cx + radius * 0.5 * math.cos(ha)
                            hy = cy + radius * 0.5 * math.sin(ha)
                            hz = z_start - (z_start - z) * h / helix_pts
                            toolpath.add_point(hx, hy, hz)

                    toolpath.add_point(x, y, z)

                t += step
                circle_idx += 1

            # Retract after each Z level
            if len(toolpath.points) > 0:
                last = toolpath.points[-1]
                toolpath.add_point(last.x, last.y, z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} trochoidal points")

        return toolpath

    def generate_adaptive_2d(
        self,
        tool: Tool,
        contour: np.ndarray,  # Nx2 array of boundary points
        z_bottom: float,
        z_top: float,
        params: Optional[TrochoidalParams] = None,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate adaptive (trochoidal) clearing for a 2D pocket.

        Uses constant engagement angle to clear material efficiently.

        Args:
            tool: Cutting tool.
            contour: Boundary polygon points.
            z_bottom: Bottom Z.
            z_top: Top Z.
            params: Trochoidal parameters.
            z_safe: Safe Z.

        Returns:
            Adaptive clearing toolpath.
        """
        logger.info("Generating adaptive 2D clearing toolpath")

        if params is None:
            params = TrochoidalParams(
                trochoidal_diameter=tool.diameter * 0.8,
                stepover=tool.diameter * 0.15,
                max_engagement_angle=60.0,
            )

        toolpath = Toolpath(tool=tool, safe_z=z_safe)

        # Create offset contours from outside in
        from shapely.geometry import Polygon, LineString
        from shapely.ops import unary_union

        poly = Polygon(contour)
        if not poly.is_valid:
            poly = poly.buffer(0)

        tool_radius = tool.diameter / 2
        offset_step = params.stepover

        # Z levels
        z_step = tool.diameter * params.radial_depth
        z_levels = np.arange(z_top, z_bottom - z_step, -z_step)

        for z in z_levels:
            # Generate offset contours
            offset = tool_radius
            contours = []

            while True:
                offset_poly = poly.buffer(-offset)
                if offset_poly.is_empty:
                    break

                if offset_poly.geom_type == 'Polygon':
                    coords = np.array(offset_poly.exterior.coords)
                    contours.append(coords)
                elif offset_poly.geom_type == 'MultiPolygon':
                    for p in offset_poly.geoms:
                        coords = np.array(p.exterior.coords)
                        contours.append(coords)

                offset += offset_step

            # Generate trochoidal path along each contour
            for contour_pts in reversed(contours):  # Inside out
                self._add_trochoidal_contour(
                    toolpath, contour_pts, z, params, z_safe
                )

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} adaptive points")

        return toolpath

    def _add_trochoidal_contour(
        self,
        toolpath: Toolpath,
        contour: np.ndarray,
        z: float,
        params: TrochoidalParams,
        z_safe: float,
    ):
        """Add trochoidal moves along a contour."""
        if len(contour) < 3:
            return

        radius = params.trochoidal_diameter / 4  # Smaller circles for contour following

        # Move to start
        first = contour[0]
        toolpath.add_point(first[0], first[1], z_safe, rapid=True)
        toolpath.add_point(first[0], first[1], z)

        # Follow contour with trochoidal motion
        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]

            # Direction
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue

            ux, uy = dx / length, dy / length

            # Number of trochoidal steps along this segment
            n_steps = max(1, int(length / params.stepover))

            for j in range(n_steps):
                t = j / n_steps
                cx = p1[0] + t * dx
                cy = p1[1] + t * dy

                # Small circular motion
                for k in range(9):
                    angle = 2 * math.pi * k / 8
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)
                    toolpath.add_point(x, y, z)

        # Return to safe
        last = contour[-1]
        toolpath.add_point(last[0], last[1], z_safe, rapid=True)


# =============================================================================
# SPIRAL CONTOUR STRATEGY
# =============================================================================

class SpiralGenerator:
    """
    Generate continuous spiral toolpaths.

    Spiral paths provide:
    - Continuous cutting (no retracts)
    - Smooth tool engagement
    - Reduced cycle time
    - Better surface finish
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        stl_surface,
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        self.mesh = mesh
        self.stl_surface = stl_surface
        self.bounds = bounds

    def generate_3d_spiral(
        self,
        tool: Tool,
        stepover: float,
        z_safe: float = 10.0,
        direction: str = "outward",  # 'outward' or 'inward'
    ) -> Toolpath:
        """
        Generate 3D spiral surfacing toolpath.

        Creates a continuous spiral from center outward (or inward),
        projecting onto the 3D surface via drop-cutter.

        Args:
            tool: Cutting tool.
            stepover: Radial step between spiral turns.
            z_safe: Safe Z height.
            direction: 'outward' from center or 'inward' from edge.

        Returns:
            Spiral toolpath.
        """
        ocl = _import_ocl()

        logger.info(f"Generating 3D spiral toolpath, stepover={stepover}mm")

        bounds_min, bounds_max = self.bounds
        center_x = (bounds_min[0] + bounds_max[0]) / 2
        center_y = (bounds_min[1] + bounds_max[1]) / 2

        # Maximum radius to cover the part
        max_radius = math.sqrt(
            ((bounds_max[0] - bounds_min[0]) / 2) ** 2 +
            ((bounds_max[1] - bounds_min[1]) / 2) ** 2
        ) + stepover

        toolpath = Toolpath(tool=tool, strategy=Strategy.SPIRAL, safe_z=z_safe)

        # Create cutter
        cutter = self._create_cutter(tool, ocl)

        # Generate spiral points
        # Parametric: r(t) = stepover * t / (2*pi), angle = t
        points_per_turn = 72
        n_turns = int(max_radius / stepover) + 1
        total_points = n_turns * points_per_turn

        spiral_points = []
        for i in range(total_points):
            t = 2 * math.pi * i / points_per_turn
            r = stepover * t / (2 * math.pi)

            if r > max_radius:
                break

            x = center_x + r * math.cos(t)
            y = center_y + r * math.sin(t)
            spiral_points.append((x, y))

        if direction == "inward":
            spiral_points = spiral_points[::-1]

        # Project onto surface using drop-cutter
        z_min = bounds_min[2] - 10

        # Process in batches for efficiency
        batch_size = 100
        all_cl_points = []

        for batch_start in range(0, len(spiral_points), batch_size):
            batch = spiral_points[batch_start:batch_start + batch_size]

            if len(batch) < 2:
                continue

            # Create path from batch
            pdc = ocl.PathDropCutter()
            pdc.setSTL(self.stl_surface)
            pdc.setCutter(cutter)
            pdc.setZ(z_min)
            pdc.setSampling(stepover / 4)

            path = ocl.Path()
            for j in range(len(batch) - 1):
                p1, p2 = batch[j], batch[j + 1]
                start_pt = ocl.Point(p1[0], p1[1], bounds_max[2] + 50)
                end_pt = ocl.Point(p2[0], p2[1], bounds_max[2] + 50)
                line = ocl.Line(start_pt, end_pt)
                path.append(line)

            pdc.setPath(path)
            pdc.run()
            all_cl_points.extend(pdc.getCLPoints())

        # Build toolpath
        if all_cl_points:
            first = all_cl_points[0]
            toolpath.add_point(first.x, first.y, z_safe, rapid=True)

            for pt in all_cl_points:
                toolpath.add_point(pt.x, pt.y, pt.z)

            last = all_cl_points[-1]
            toolpath.add_point(last.x, last.y, z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} spiral points")

        return toolpath

    def generate_contour_spiral(
        self,
        tool: Tool,
        boundary: np.ndarray,  # Nx2 boundary points
        stepover: float,
        z: float,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate spiral from boundary contour inward.

        Uses polygon offsetting to create spiral from edge to center.

        Args:
            tool: Cutting tool.
            boundary: Boundary polygon vertices.
            stepover: Step between offset rings.
            z: Cutting Z height.
            z_safe: Safe Z.

        Returns:
            Contour-following spiral toolpath.
        """
        from shapely.geometry import Polygon
        from shapely.ops import linemerge

        logger.info("Generating contour spiral toolpath")

        toolpath = Toolpath(tool=tool, strategy=Strategy.SPIRAL, safe_z=z_safe)

        poly = Polygon(boundary)
        if not poly.is_valid:
            poly = poly.buffer(0)

        tool_radius = tool.diameter / 2

        # Generate offset contours
        offset = tool_radius
        all_points = []

        while True:
            offset_poly = poly.buffer(-offset)
            if offset_poly.is_empty:
                break

            if offset_poly.geom_type == 'Polygon':
                coords = list(offset_poly.exterior.coords)
                all_points.extend(coords)
            elif offset_poly.geom_type == 'MultiPolygon':
                for p in offset_poly.geoms:
                    coords = list(p.exterior.coords)
                    all_points.extend(coords)

            offset += stepover

        if not all_points:
            return toolpath

        # Connect into continuous spiral
        # Simple approach: just connect sequential points
        first = all_points[0]
        toolpath.add_point(first[0], first[1], z_safe, rapid=True)
        toolpath.add_point(first[0], first[1], z)

        for pt in all_points[1:]:
            toolpath.add_point(pt[0], pt[1], z)

        last = all_points[-1]
        toolpath.add_point(last[0], last[1], z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} contour spiral points")

        return toolpath

    def _create_cutter(self, tool: Tool, ocl):
        """Create OpenCAMLib cutter."""
        from pycam3d.toolpath import ToolType

        if tool.type == ToolType.BALL:
            return ocl.BallCutter(tool.diameter, tool.length)
        elif tool.type == ToolType.FLAT:
            return ocl.CylCutter(tool.diameter, tool.length)
        elif tool.type == ToolType.BULL:
            return ocl.BullCutter(tool.diameter, tool.corner_radius, tool.length)
        else:
            return ocl.BallCutter(tool.diameter, tool.length)


# =============================================================================
# VORONOI-BASED ADAPTIVE POCKETING
# =============================================================================

class VoronoiPocketGenerator:
    """
    Generate toolpaths using Voronoi diagram / medial axis.

    Benefits:
    - Optimal material removal from center
    - Avoids tool overload in corners
    - Enables high-speed machining

    Requires OpenVoronoi library for full functionality.
    Falls back to offset-based approach if not available.
    """

    def __init__(self):
        self._has_openvoronoi = self._check_openvoronoi()

    def _check_openvoronoi(self) -> bool:
        """Check if OpenVoronoi is available."""
        try:
            import openvoronoi as ovd
            return True
        except ImportError:
            logger.warning(
                "OpenVoronoi not available. Using fallback offset method. "
                "Install with: pip install openvoronoi"
            )
            return False

    def generate_medial_axis_pocket(
        self,
        tool: Tool,
        boundary: np.ndarray,
        z_bottom: float,
        z_top: float,
        stepover: float,
        z_safe: float = 10.0,
    ) -> Toolpath:
        """
        Generate pocket toolpath using medial axis strategy.

        Machines from the medial axis (center) outward, ensuring
        constant tool engagement.

        Args:
            tool: Cutting tool.
            boundary: Pocket boundary vertices (Nx2).
            z_bottom: Bottom Z of pocket.
            z_top: Top Z (stock surface).
            stepover: Radial stepover.
            z_safe: Safe Z.

        Returns:
            Medial axis pocketing toolpath.
        """
        logger.info("Generating medial axis pocket toolpath")

        if self._has_openvoronoi:
            return self._generate_voronoi_pocket(
                tool, boundary, z_bottom, z_top, stepover, z_safe
            )
        else:
            return self._generate_offset_pocket(
                tool, boundary, z_bottom, z_top, stepover, z_safe
            )

    def _generate_voronoi_pocket(
        self,
        tool: Tool,
        boundary: np.ndarray,
        z_bottom: float,
        z_top: float,
        stepover: float,
        z_safe: float,
    ) -> Toolpath:
        """Generate pocket using OpenVoronoi medial axis."""
        import openvoronoi as ovd

        toolpath = Toolpath(tool=tool, safe_z=z_safe)

        # Create Voronoi diagram
        vd = ovd.VoronoiDiagram(1, 100)

        # Add polygon vertices
        vertex_ids = []
        for pt in boundary:
            vid = vd.addVertexSite(ovd.Point(pt[0], pt[1]))
            vertex_ids.append(vid)

        # Add line segments
        for i in range(len(vertex_ids)):
            j = (i + 1) % len(vertex_ids)
            vd.addLineSite(vertex_ids[i], vertex_ids[j])

        # Get medial axis
        ma_filter = ovd.MedialAxisFilter(vd.getGraph())
        ma_filter.filter()

        # Extract medial axis edges
        ma_edges = []
        g = vd.getGraph()
        for edge in g.edges:
            if edge.valid:
                src = g.vertices[edge.source]
                tgt = g.vertices[edge.target]
                ma_edges.append(((src.x, src.y), (tgt.x, tgt.y)))

        # Generate toolpath from medial axis outward
        z_step = tool.diameter * 0.3
        z_levels = np.arange(z_top, z_bottom - z_step, -z_step)

        tool_radius = tool.diameter / 2

        for z in z_levels:
            # Start from medial axis
            for edge in ma_edges:
                p1, p2 = edge
                toolpath.add_point(p1[0], p1[1], z_safe, rapid=True)
                toolpath.add_point(p1[0], p1[1], z)
                toolpath.add_point(p2[0], p2[1], z)

            # Then offset passes outward
            # (simplified - full implementation would use offset from MA)

        toolpath.optimize()
        return toolpath

    def _generate_offset_pocket(
        self,
        tool: Tool,
        boundary: np.ndarray,
        z_bottom: float,
        z_top: float,
        stepover: float,
        z_safe: float,
    ) -> Toolpath:
        """Fallback offset-based pocketing."""
        from shapely.geometry import Polygon

        logger.info("Using offset-based pocket (OpenVoronoi fallback)")

        toolpath = Toolpath(tool=tool, safe_z=z_safe)

        poly = Polygon(boundary)
        if not poly.is_valid:
            poly = poly.buffer(0)

        tool_radius = tool.diameter / 2
        z_step = tool.diameter * 0.3
        z_levels = np.arange(z_top, z_bottom - z_step, -z_step)

        for z in z_levels:
            # Generate offset contours from outside in
            offset = tool_radius
            contours = []

            while True:
                offset_poly = poly.buffer(-offset)
                if offset_poly.is_empty:
                    break

                if offset_poly.geom_type == 'Polygon':
                    coords = np.array(offset_poly.exterior.coords)
                    contours.append(coords)
                elif offset_poly.geom_type == 'MultiPolygon':
                    for p in offset_poly.geoms:
                        coords = np.array(p.exterior.coords)
                        contours.append(coords)

                offset += stepover

            # Machine from inside out (reversed contours)
            for contour in reversed(contours):
                if len(contour) < 3:
                    continue

                first = contour[0]
                toolpath.add_point(first[0], first[1], z_safe, rapid=True)
                toolpath.add_point(first[0], first[1], z)

                for pt in contour[1:]:
                    toolpath.add_point(pt[0], pt[1], z)

                # Close contour
                toolpath.add_point(first[0], first[1], z)
                toolpath.add_point(first[0], first[1], z_safe, rapid=True)

        toolpath.optimize()
        logger.info(f"Generated {len(toolpath)} offset pocket points")

        return toolpath
