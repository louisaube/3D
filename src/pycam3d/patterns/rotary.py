"""
Rotary Pattern Generator for 4-Axis Machining.

Port from Fabex get_path_pattern_4_axis() function.
Generates toolpath patterns for rotary/indexing operations.

Strategies:
- PARALLELR: Passes AROUND the rotary axis (circles at positions along axis)
- PARALLEL: Passes ALONG the rotary axis (lines at different angles)
- HELIX: Continuous helical path around the axis
"""

from __future__ import annotations

import numpy as np
from math import pi, floor, ceil
from enum import Enum
from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass

from pycam3d.chunk import CamPathChunk4Axis, AxisRotation


class RotaryStrategy(Enum):
    """4-axis rotary machining strategies."""
    PARALLELR = "parallelr"   # Passes around the rotary axis
    PARALLEL = "parallel"     # Passes along the rotary axis
    HELIX = "helix"          # Helical path
    CROSS = "cross"          # Crossed passes (both directions)


@dataclass
class RotaryBounds:
    """Bounds for rotary machining."""
    axis_min: float   # Min position along rotary axis
    axis_max: float   # Max position along rotary axis
    radius: float     # Outer radius
    radius_end: float # Inner radius (for tapered parts)
    angle_start: float = 0.0      # Start angle in radians
    angle_end: float = 2 * pi     # End angle in radians


class RotaryPatternGenerator:
    """
    Generate toolpath patterns for 4-axis rotary machining.

    Supports machining of cylindrical, conical, and revolution parts
    using rotary table (A-axis) or spindle tilt (B-axis).

    Example:
        generator = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, -50, -50),
            max_bounds=(100, 50, 50),
            distance_between_paths=2.0,
            distance_along_paths=0.5
        )
        chunks = generator.generate_parallel_around(radius=25.0)
    """

    def __init__(
        self,
        rotary_axis: Literal['X', 'Y', 'Z'] = 'X',
        min_bounds: Tuple[float, float, float] = (0, -50, -50),
        max_bounds: Tuple[float, float, float] = (100, 50, 50),
        distance_between_paths: float = 1.0,
        distance_along_paths: float = 0.5,
    ):
        """
        Initialize rotary pattern generator.

        Args:
            rotary_axis: Axis of rotation ('X', 'Y', or 'Z')
            min_bounds: Minimum bounds (x, y, z)
            max_bounds: Maximum bounds (x, y, z)
            distance_between_paths: Stepover between passes
            distance_along_paths: Point spacing along each pass
        """
        self.rotary_axis = rotary_axis
        self.min = np.array(min_bounds)
        self.max = np.array(max_bounds)
        self.path_step = distance_between_paths
        self.along_step = distance_along_paths

        # Setup axis mapping based on rotary axis
        self._setup_axis_mapping()

    def _setup_axis_mapping(self) -> None:
        """
        Configure axis indices based on rotary axis.

        a1 = rotary axis (along the rotation axis)
        a2 = perpendicular axis 1
        a3 = perpendicular axis 2 (radial direction)
        """
        if self.rotary_axis == 'X':
            # X is rotary axis, rotate in YZ plane
            self.a1, self.a2, self.a3 = 0, 1, 2
        elif self.rotary_axis == 'Y':
            # Y is rotary axis, rotate in XZ plane
            self.a1, self.a2, self.a3 = 1, 0, 2
        else:  # Z
            # Z is rotary axis, rotate in XY plane
            self.a1, self.a2, self.a3 = 2, 0, 1

    def _get_axis_letter(self) -> str:
        """Get the G-code axis letter for the rotary axis."""
        if self.rotary_axis == 'X':
            return 'A'
        elif self.rotary_axis == 'Y':
            return 'B'
        else:
            return 'C'

    def _create_rotation(self, angle: float) -> AxisRotation:
        """Create AxisRotation for the current rotary axis."""
        rot = AxisRotation()
        if self.rotary_axis == 'X':
            rot.a = angle
        elif self.rotary_axis == 'Y':
            rot.b = angle
        else:
            rot.c = angle
        return rot

    def _rotate_point(
        self,
        point: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """
        Rotate a point around the rotary axis.

        Args:
            point: 3D point to rotate
            angle: Rotation angle in radians

        Returns:
            Rotated point
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        result = point.copy()

        # Rotate around the appropriate axis
        p2 = point[self.a2]
        p3 = point[self.a3]

        result[self.a2] = p2 * cos_a - p3 * sin_a
        result[self.a3] = p2 * sin_a + p3 * cos_a

        return result

    def generate_parallel_around(
        self,
        radius: float,
        radius_end: Optional[float] = None,
        angle_start: float = 0.0,
        angle_end: float = 2 * pi
    ) -> List[CamPathChunk4Axis]:
        """
        PARALLELR: Generate passes AROUND the rotary axis.

        Creates circular passes at different positions along the rotary axis.
        Ideal for turning cylindrical or conical surfaces.

        Args:
            radius: Outer radius (start of cut)
            radius_end: Inner radius (end of cut), defaults to radius
            angle_start: Starting angle in radians
            angle_end: Ending angle in radians

        Returns:
            List of CamPathChunk4Axis chunks
        """
        if radius_end is None:
            radius_end = radius

        chunks = []
        mradius = max(radius, radius_end)

        # Calculate angular steps based on circumference and step size
        arc_length = mradius * abs(angle_end - angle_start)
        circle_steps = max(4, int(arc_length / self.along_step))
        angle_step = (angle_end - angle_start) / circle_steps

        # Number of passes along the rotary axis
        axis_length = self.max[self.a1] - self.min[self.a1]
        num_passes = max(1, int(axis_length / self.path_step) + 1)

        for pass_idx in range(num_passes):
            chunk = CamPathChunk4Axis(name=f"parallelr_pass_{pass_idx}")

            # Position along rotary axis
            axis_pos = self.min[self.a1] + pass_idx * self.path_step
            axis_pos = min(axis_pos, self.max[self.a1])

            # Interpolate radius for conical parts
            t = pass_idx / max(1, num_passes - 1)
            current_radius = radius + (radius_end - radius) * t

            for step in range(circle_steps + 1):
                angle = angle_start + step * angle_step

                # Create start point (outside, at current_radius)
                start = np.zeros(3)
                start[self.a1] = axis_pos
                start[self.a2] = 0
                start[self.a3] = current_radius

                # Create end point (inside, at radius_end or 0)
                end = np.zeros(3)
                end[self.a1] = axis_pos
                end[self.a2] = 0
                end[self.a3] = 0  # Target center for collision

                # Rotate both points
                rotated_start = self._rotate_point(start, angle)
                rotated_end = self._rotate_point(end, angle)

                # Create rotation for this point
                rotation = self._create_rotation(angle)

                chunk.add_point(
                    point=tuple(rotated_start),
                    startpoint=tuple(rotated_start),
                    endpoint=tuple(rotated_end),
                    rotation=rotation
                )

            chunk.depth = abs(radius - radius_end)
            chunk.closed = abs(angle_end - angle_start - 2*pi) < 0.01
            chunk.layer_index = pass_idx
            chunks.append(chunk)

        return chunks

    def generate_parallel_along(
        self,
        radius: float,
        radius_end: Optional[float] = None,
        angle_start: float = 0.0,
        angle_end: float = 2 * pi
    ) -> List[CamPathChunk4Axis]:
        """
        PARALLEL: Generate passes ALONG the rotary axis.

        Creates linear passes at different angles around the part.
        Ideal for surfacing flanks of cylindrical parts.

        Args:
            radius: Outer radius
            radius_end: Inner radius, defaults to radius
            angle_start: Starting angle in radians
            angle_end: Ending angle in radians

        Returns:
            List of CamPathChunk4Axis chunks
        """
        if radius_end is None:
            radius_end = radius

        chunks = []
        mradius = max(radius, radius_end)

        # Number of angular passes
        arc_length = mradius * abs(angle_end - angle_start)
        num_angles = max(4, int(arc_length / self.path_step) + 1)
        angle_step = (angle_end - angle_start) / num_angles

        # Number of points along axis
        axis_length = self.max[self.a1] - self.min[self.a1]
        num_along = max(2, int(axis_length / self.along_step) + 1)

        reverse = False

        for angle_idx in range(num_angles + 1):
            chunk = CamPathChunk4Axis(name=f"parallel_angle_{angle_idx}")
            angle = angle_start + angle_idx * angle_step

            for along_idx in range(num_along):
                # Position along axis
                if reverse:
                    t = 1.0 - along_idx / max(1, num_along - 1)
                else:
                    t = along_idx / max(1, num_along - 1)

                axis_pos = self.min[self.a1] + t * axis_length

                # Create points
                start = np.zeros(3)
                end = np.zeros(3)

                start[self.a1] = axis_pos
                end[self.a1] = axis_pos
                start[self.a2] = 0
                start[self.a3] = radius
                end[self.a2] = 0
                end[self.a3] = radius_end

                # Rotate
                rotated_start = self._rotate_point(start, angle)
                rotated_end = self._rotate_point(end, angle)

                rotation = self._create_rotation(angle)

                chunk.add_point(
                    point=tuple(rotated_start),
                    startpoint=tuple(rotated_start),
                    endpoint=tuple(rotated_end),
                    rotation=rotation
                )

            chunk.depth = abs(radius - radius_end)
            chunk.layer_index = angle_idx
            chunks.append(chunk)

            # Zigzag pattern
            reverse = not reverse

        return chunks

    def generate_helix(
        self,
        radius: float,
        radius_end: Optional[float] = None,
        angle_start: float = 0.0,
        num_rotations: Optional[int] = None
    ) -> List[CamPathChunk4Axis]:
        """
        HELIX: Generate continuous helical toolpath.

        Creates a single continuous spiral path around the part.
        Ideal for fast roughing of cylindrical parts.

        Args:
            radius: Outer radius
            radius_end: Inner radius
            angle_start: Starting angle
            num_rotations: Number of full rotations (auto-calculated if None)

        Returns:
            List containing single CamPathChunk4Axis
        """
        if radius_end is None:
            radius_end = radius

        chunk = CamPathChunk4Axis(name="helix")
        mradius = max(radius, radius_end)

        # Calculate steps
        circumference = mradius * 2 * pi
        circle_steps = max(4, int(circumference / self.along_step))
        angle_step = 2 * pi / circle_steps

        # Number of full rotations
        axis_length = self.max[self.a1] - self.min[self.a1]
        if num_rotations is None:
            num_rotations = max(1, int(axis_length / self.path_step))

        # Axis increment per angular step
        axis_step = axis_length / (num_rotations * circle_steps)

        current_axis_pos = self.min[self.a1]
        total_steps = num_rotations * circle_steps

        for step in range(total_steps + 1):
            angle = angle_start + step * angle_step

            # Linear interpolation of position along axis
            t = step / total_steps
            axis_pos = self.min[self.a1] + t * axis_length

            # Interpolate radius
            current_radius = radius + (radius_end - radius) * t

            start = np.zeros(3)
            end = np.zeros(3)

            start[self.a1] = axis_pos
            end[self.a1] = axis_pos
            start[self.a2] = 0
            start[self.a3] = current_radius
            end[self.a2] = 0
            end[self.a3] = 0

            # Normalize angle to [0, 2*pi]
            normalized_angle = angle % (2 * pi)

            rotated_start = self._rotate_point(start, normalized_angle)
            rotated_end = self._rotate_point(end, normalized_angle)

            rotation = self._create_rotation(normalized_angle)

            chunk.add_point(
                point=tuple(rotated_start),
                startpoint=tuple(rotated_start),
                endpoint=tuple(rotated_end),
                rotation=rotation
            )

        chunk.depth = abs(radius - radius_end)

        return [chunk]

    def generate_cross(
        self,
        radius: float,
        radius_end: Optional[float] = None
    ) -> List[CamPathChunk4Axis]:
        """
        CROSS: Generate crossed pattern (both parallel directions).

        Combines PARALLEL and PARALLELR for better surface finish.

        Args:
            radius: Outer radius
            radius_end: Inner radius

        Returns:
            List of CamPathChunk4Axis from both strategies
        """
        chunks = []

        # First: passes along the axis
        along_chunks = self.generate_parallel_along(radius, radius_end)
        for chunk in along_chunks:
            chunk.name = f"cross_along_{chunk.layer_index}"
        chunks.extend(along_chunks)

        # Second: passes around the axis
        around_chunks = self.generate_parallel_around(radius, radius_end)
        for chunk in around_chunks:
            chunk.name = f"cross_around_{chunk.layer_index}"
            chunk.layer_index += len(along_chunks)
        chunks.extend(around_chunks)

        return chunks

    def generate(
        self,
        strategy: RotaryStrategy,
        radius: float,
        radius_end: Optional[float] = None,
        **kwargs
    ) -> List[CamPathChunk4Axis]:
        """
        Generate pattern using specified strategy.

        Args:
            strategy: Rotary machining strategy
            radius: Outer radius
            radius_end: Inner radius
            **kwargs: Additional arguments for specific strategies

        Returns:
            List of CamPathChunk4Axis
        """
        if strategy == RotaryStrategy.PARALLELR:
            return self.generate_parallel_around(radius, radius_end, **kwargs)
        elif strategy == RotaryStrategy.PARALLEL:
            return self.generate_parallel_along(radius, radius_end, **kwargs)
        elif strategy == RotaryStrategy.HELIX:
            return self.generate_helix(radius, radius_end, **kwargs)
        elif strategy == RotaryStrategy.CROSS:
            return self.generate_cross(radius, radius_end)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @classmethod
    def from_mesh_bounds(
        cls,
        mesh,
        rotary_axis: Literal['X', 'Y', 'Z'] = 'X',
        distance_between_paths: float = 1.0,
        distance_along_paths: float = 0.5
    ) -> 'RotaryPatternGenerator':
        """
        Create generator from mesh bounds.

        Args:
            mesh: Trimesh mesh object
            rotary_axis: Axis of rotation
            distance_between_paths: Stepover
            distance_along_paths: Point spacing

        Returns:
            Configured RotaryPatternGenerator
        """
        bounds = mesh.bounds
        return cls(
            rotary_axis=rotary_axis,
            min_bounds=tuple(bounds[0]),
            max_bounds=tuple(bounds[1]),
            distance_between_paths=distance_between_paths,
            distance_along_paths=distance_along_paths
        )

    def estimate_radius_from_bounds(self) -> float:
        """
        Estimate radius from bounds assuming centered cylindrical part.

        Returns:
            Estimated radius
        """
        dims = self.max - self.min
        # Take max of the two perpendicular dimensions
        return max(dims[self.a2], dims[self.a3]) / 2
