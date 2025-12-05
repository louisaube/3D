"""
4-Axis Toolpath Chunk Data Structures.

Port from Fabex CamPathChunk for multi-axis machining support.
Stores trajectory points with rotation information for 4/5-axis CNC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
import numpy as np


@dataclass
class AxisRotation:
    """
    Rotation of machine axes for a toolpath point.

    Stores rotations in radians for the A, B, C axes:
    - A: Rotation around X axis (typical rotary table)
    - B: Rotation around Y axis (spindle tilt on LUQUE L1530)
    - C: Rotation around Z axis (rotary table)
    """
    a: float = 0.0  # Rotation around X
    b: float = 0.0  # Rotation around Y (spindle tilt)
    c: float = 0.0  # Rotation around Z

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return as (A, B, C) tuple in radians."""
        return (self.a, self.b, self.c)

    def to_degrees(self) -> 'AxisRotation':
        """Return new AxisRotation with values converted to degrees."""
        return AxisRotation(
            np.degrees(self.a),
            np.degrees(self.b),
            np.degrees(self.c)
        )

    def to_radians(self) -> 'AxisRotation':
        """Return new AxisRotation with values converted to radians."""
        return AxisRotation(
            np.radians(self.a),
            np.radians(self.b),
            np.radians(self.c)
        )

    def copy(self) -> 'AxisRotation':
        """Create a deep copy."""
        return AxisRotation(self.a, self.b, self.c)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AxisRotation):
            return False
        return (abs(self.a - other.a) < 1e-6 and
                abs(self.b - other.b) < 1e-6 and
                abs(self.c - other.c) < 1e-6)

    def interpolate(self, other: 'AxisRotation', t: float) -> 'AxisRotation':
        """Linear interpolation between two rotations."""
        return AxisRotation(
            self.a + (other.a - self.a) * t,
            self.b + (other.b - self.b) * t,
            self.c + (other.c - self.c) * t
        )

    @classmethod
    def from_euler(cls, angles: Tuple[float, float, float], degrees: bool = False) -> 'AxisRotation':
        """Create from Euler angles (A, B, C)."""
        a, b, c = angles
        if degrees:
            a, b, c = np.radians(a), np.radians(b), np.radians(c)
        return cls(a, b, c)


@dataclass
class CamPathChunk4Axis:
    """
    Chunk of toolpath with multi-axis support.

    Stores:
    - points: Final tool contact points on surface
    - startpoints: Ray origin points for collision detection
    - endpoints: Ray target points (surface approach direction)
    - rotations: Spindle orientation at each point

    This is the core data structure for 4/5-axis toolpath generation,
    ported from Fabex's CamPathChunk.
    """

    # Final contact points (tool tip position)
    points: List[Tuple[float, float, float]] = field(default_factory=list)

    # Collision ray start points (safe position)
    startpoints: List[Tuple[float, float, float]] = field(default_factory=list)

    # Collision ray end points (target on/below surface)
    endpoints: List[Tuple[float, float, float]] = field(default_factory=list)

    # Spindle rotation at each point
    rotations: List[AxisRotation] = field(default_factory=list)

    # Depth of this pass
    depth: float = 0.0

    # Whether this chunk forms a closed loop
    closed: bool = False

    # Layer index for multi-pass operations
    layer_index: int = 0

    # Optional name/identifier
    name: str = ""

    def count(self) -> int:
        """Return number of points in chunk."""
        return len(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[Tuple[float, float, float]]:
        return iter(self.points)

    def get_point(self, index: int) -> Tuple[float, float, float]:
        """Get point at index."""
        return self.points[index]

    def get_rotation(self, index: int) -> Optional[AxisRotation]:
        """Get rotation at index, or None if not available."""
        if index < len(self.rotations):
            return self.rotations[index]
        return None

    def add_point(
        self,
        point: Tuple[float, float, float],
        startpoint: Optional[Tuple[float, float, float]] = None,
        endpoint: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[AxisRotation] = None
    ) -> None:
        """
        Add a point with optional metadata.

        Args:
            point: Tool contact point (x, y, z)
            startpoint: Ray start for collision (optional)
            endpoint: Ray target for collision (optional)
            rotation: Spindle rotation (optional)
        """
        self.points.append(point)
        if startpoint is not None:
            self.startpoints.append(startpoint)
        if endpoint is not None:
            self.endpoints.append(endpoint)
        if rotation is not None:
            self.rotations.append(rotation)

    def reverse(self) -> None:
        """Reverse order of all points and metadata."""
        self.points.reverse()
        self.startpoints.reverse()
        self.endpoints.reverse()
        self.rotations.reverse()

    def shift(self, dx: float, dy: float, dz: float) -> None:
        """Translate all points by (dx, dy, dz)."""
        self.points = [(p[0]+dx, p[1]+dy, p[2]+dz) for p in self.points]
        self.startpoints = [(p[0]+dx, p[1]+dy, p[2]+dz) for p in self.startpoints]
        self.endpoints = [(p[0]+dx, p[1]+dy, p[2]+dz) for p in self.endpoints]

    def rotate_around_axis(self, axis: str, angle: float, center: Tuple[float, float, float] = (0, 0, 0)) -> None:
        """
        Rotate all points around an axis.

        Args:
            axis: 'X', 'Y', or 'Z'
            angle: Rotation angle in radians
            center: Center of rotation
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cx, cy, cz = center

        def rotate_point(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
            x, y, z = p[0] - cx, p[1] - cy, p[2] - cz
            if axis == 'X':
                return (x + cx, y * cos_a - z * sin_a + cy, y * sin_a + z * cos_a + cz)
            elif axis == 'Y':
                return (x * cos_a + z * sin_a + cx, y + cy, -x * sin_a + z * cos_a + cz)
            else:  # Z
                return (x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy, z + cz)

        self.points = [rotate_point(p) for p in self.points]
        self.startpoints = [rotate_point(p) for p in self.startpoints]
        self.endpoints = [rotate_point(p) for p in self.endpoints]

    def copy(self) -> 'CamPathChunk4Axis':
        """Create a deep copy of the chunk."""
        return CamPathChunk4Axis(
            points=self.points.copy(),
            startpoints=self.startpoints.copy(),
            endpoints=self.endpoints.copy(),
            rotations=[r.copy() for r in self.rotations],
            depth=self.depth,
            closed=self.closed,
            layer_index=self.layer_index,
            name=self.name
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of points."""
        if not self.points:
            return np.zeros(3), np.zeros(3)
        arr = np.array(self.points)
        return arr.min(axis=0), arr.max(axis=0)

    def get_length(self) -> float:
        """Calculate total path length."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            p1 = np.array(self.points[i-1])
            p2 = np.array(self.points[i])
            total += np.linalg.norm(p2 - p1)
        return total

    def simplify(self, tolerance: float = 0.01) -> None:
        """
        Simplify path using Douglas-Peucker algorithm.
        Removes points that don't significantly change the path.
        """
        if len(self.points) < 3:
            return

        # Use numpy for efficiency
        points = np.array(self.points)
        mask = np.ones(len(points), dtype=bool)

        def rdp_recursive(start: int, end: int):
            if end - start < 2:
                return

            # Find point with max distance from line
            p1, p2 = points[start], points[end]
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)

            if line_len < 1e-10:
                return

            line_unit = line_vec / line_len

            max_dist = 0
            max_idx = start

            for i in range(start + 1, end):
                v = points[i] - p1
                proj = np.dot(v, line_unit)
                proj = max(0, min(proj, line_len))
                closest = p1 + proj * line_unit
                dist = np.linalg.norm(points[i] - closest)

                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            if max_dist > tolerance:
                rdp_recursive(start, max_idx)
                rdp_recursive(max_idx, end)
            else:
                # Mark intermediate points for removal
                for i in range(start + 1, end):
                    mask[i] = False

        rdp_recursive(0, len(points) - 1)

        # Apply mask
        indices = np.where(mask)[0]
        self.points = [self.points[i] for i in indices]
        if self.startpoints:
            self.startpoints = [self.startpoints[i] for i in indices if i < len(self.startpoints)]
        if self.endpoints:
            self.endpoints = [self.endpoints[i] for i in indices if i < len(self.endpoints)]
        if self.rotations:
            self.rotations = [self.rotations[i] for i in indices if i < len(self.rotations)]

    def resample(self, step: float) -> 'CamPathChunk4Axis':
        """
        Resample path with uniform point spacing.

        Args:
            step: Distance between resampled points

        Returns:
            New chunk with resampled points
        """
        if len(self.points) < 2:
            return self.copy()

        new_chunk = CamPathChunk4Axis(
            depth=self.depth,
            closed=self.closed,
            layer_index=self.layer_index,
            name=self.name
        )

        # Add first point
        new_chunk.add_point(
            self.points[0],
            self.startpoints[0] if self.startpoints else None,
            self.endpoints[0] if self.endpoints else None,
            self.rotations[0].copy() if self.rotations else None
        )

        accumulated = 0.0

        for i in range(1, len(self.points)):
            p1 = np.array(self.points[i-1])
            p2 = np.array(self.points[i])
            segment_len = np.linalg.norm(p2 - p1)

            if segment_len < 1e-10:
                continue

            direction = (p2 - p1) / segment_len

            while accumulated + step <= segment_len:
                accumulated += step
                t = accumulated / segment_len

                new_point = tuple(p1 + direction * accumulated)

                # Interpolate rotation if available
                new_rot = None
                if self.rotations and i < len(self.rotations):
                    r1 = self.rotations[i-1]
                    r2 = self.rotations[i]
                    new_rot = r1.interpolate(r2, t)

                new_chunk.add_point(new_point, rotation=new_rot)

            accumulated -= segment_len

        # Add last point
        new_chunk.add_point(
            self.points[-1],
            self.startpoints[-1] if self.startpoints else None,
            self.endpoints[-1] if self.endpoints else None,
            self.rotations[-1].copy() if self.rotations else None
        )

        return new_chunk

    def to_numpy(self) -> np.ndarray:
        """Convert points to numpy array (N x 3)."""
        return np.array(self.points)

    @classmethod
    def from_numpy(cls, points: np.ndarray, **kwargs) -> 'CamPathChunk4Axis':
        """Create chunk from numpy array."""
        chunk = cls(**kwargs)
        chunk.points = [tuple(p) for p in points]
        return chunk

    def merge(self, other: 'CamPathChunk4Axis') -> None:
        """Append another chunk to this one."""
        self.points.extend(other.points)
        self.startpoints.extend(other.startpoints)
        self.endpoints.extend(other.endpoints)
        self.rotations.extend(other.rotations)


def merge_chunks(chunks: List[CamPathChunk4Axis], tolerance: float = 0.1) -> List[CamPathChunk4Axis]:
    """
    Merge consecutive chunks that are close together.

    Args:
        chunks: List of chunks to merge
        tolerance: Max distance between chunk end/start to merge

    Returns:
        List of merged chunks
    """
    if not chunks:
        return []

    result = [chunks[0].copy()]

    for chunk in chunks[1:]:
        if not chunk.points:
            continue

        last = result[-1]
        if not last.points:
            result[-1] = chunk.copy()
            continue

        # Check distance between last point of previous and first of current
        p1 = np.array(last.points[-1])
        p2 = np.array(chunk.points[0])
        dist = np.linalg.norm(p2 - p1)

        if dist <= tolerance and last.layer_index == chunk.layer_index:
            # Merge
            last.merge(chunk)
        else:
            result.append(chunk.copy())

    return result


def sort_chunks_by_distance(chunks: List[CamPathChunk4Axis], start: Tuple[float, float, float] = (0, 0, 0)) -> List[CamPathChunk4Axis]:
    """
    Sort chunks to minimize rapid travel distance.

    Uses a greedy nearest-neighbor approach.

    Args:
        chunks: Chunks to sort
        start: Starting position

    Returns:
        Sorted list of chunks
    """
    if not chunks:
        return []

    remaining = list(range(len(chunks)))
    sorted_indices = []
    current_pos = np.array(start)

    while remaining:
        # Find nearest chunk
        min_dist = float('inf')
        min_idx = 0

        for i, chunk_idx in enumerate(remaining):
            chunk = chunks[chunk_idx]
            if not chunk.points:
                continue

            # Check both start and end of chunk
            start_dist = np.linalg.norm(np.array(chunk.points[0]) - current_pos)
            end_dist = np.linalg.norm(np.array(chunk.points[-1]) - current_pos)

            dist = min(start_dist, end_dist)

            if dist < min_dist:
                min_dist = dist
                min_idx = i
                # Reverse if end is closer
                if end_dist < start_dist:
                    chunks[chunk_idx].reverse()

        chunk_idx = remaining.pop(min_idx)
        sorted_indices.append(chunk_idx)

        if chunks[chunk_idx].points:
            current_pos = np.array(chunks[chunk_idx].points[-1])

    return [chunks[i] for i in sorted_indices]
