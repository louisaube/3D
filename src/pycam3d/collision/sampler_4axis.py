"""
4-Axis Surface Sampler.

Port from Fabex sample_chunks_n_axis() and get_sample_bullet_n_axis().
Uses Trimesh ray casting instead of Blender's Bullet physics.

This module projects toolpath patterns onto mesh surfaces
with proper tool orientation for multi-axis machining.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import logging

try:
    import trimesh
    from scipy.spatial.transform import Rotation as R
except ImportError as e:
    raise ImportError(
        f"Required packages missing: {e}. "
        "Install with: pip install trimesh scipy"
    )

from pycam3d.chunk import CamPathChunk4Axis, AxisRotation
from pycam3d.toolpath import Tool, ToolType

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Result of surface sampling operation."""
    chunks: List[CamPathChunk4Axis] = field(default_factory=list)
    total_points: int = 0
    sampled_points: int = 0
    missed_points: int = 0
    sampling_time: float = 0.0


class NAxisSampler:
    """
    Surface sampler for multi-axis CNC machining.

    Projects toolpath patterns onto mesh surfaces using ray casting.
    Handles tool geometry compensation and rotation.

    Replaces Fabex's get_sample_bullet_n_axis() which used Blender's
    built-in Bullet physics engine (rigidbody_world.convex_sweep_test).

    Example:
        sampler = NAxisSampler(mesh, tool)
        result = sampler.sample_chunks(pattern_chunks, layers)
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        tool: Tool,
        use_tool_geometry: bool = True
    ):
        """
        Initialize surface sampler.

        Args:
            mesh: Trimesh mesh of the workpiece
            tool: Tool definition
            use_tool_geometry: Whether to compensate for tool geometry
        """
        self.mesh = mesh
        self.tool = tool
        self.use_tool_geometry = use_tool_geometry

        # Create ray casting accelerator
        self._ray_caster = None
        self._init_ray_caster()

        # Tool geometry for compensation
        self._tool_radius = tool.diameter / 2
        self._tool_corner_radius = getattr(tool, 'corner_radius', 0.0)

        logger.info(
            f"NAxisSampler initialized: mesh with {len(mesh.vertices)} vertices, "
            f"tool {tool.type.value} D{tool.diameter}"
        )

    def _init_ray_caster(self) -> None:
        """Initialize trimesh ray casting."""
        # Trimesh's ray module handles acceleration automatically
        # For large meshes, it uses an RTree or similar structure
        self._ray_caster = self.mesh.ray

    def _get_tool_compensation(
        self,
        direction: np.ndarray,
        rotation: AxisRotation
    ) -> np.ndarray:
        """
        Calculate tool compensation vector.

        The compensation accounts for the difference between:
        - Tool tip position (where we want the G-code point)
        - Tool center (where ray casting detects collision)

        Args:
            direction: Ray direction (normalized)
            rotation: Tool rotation

        Returns:
            Compensation vector to add to hit point
        """
        if not self.use_tool_geometry:
            return np.zeros(3)

        # Base compensation along tool axis
        comp = np.zeros(3)

        if self.tool.type == ToolType.BALL:
            # Ball nose: compensate by radius in ray direction
            comp = direction * self._tool_radius

        elif self.tool.type == ToolType.FLAT:
            # Flat end: compensate along tool axis (Z typically)
            # Tool axis is rotated by the spindle rotation
            tool_axis = self._rotate_vector(np.array([0, 0, 1]), rotation)
            comp = tool_axis * self._tool_radius

        elif self.tool.type == ToolType.BULL:
            # Bull nose: combination of radius and corner radius
            tool_axis = self._rotate_vector(np.array([0, 0, 1]), rotation)
            # Simplified: use corner radius in direction, rest along axis
            comp = direction * self._tool_corner_radius

        return comp

    def _rotate_vector(self, vec: np.ndarray, rotation: AxisRotation) -> np.ndarray:
        """
        Apply Euler rotation to a vector.

        Args:
            vec: 3D vector to rotate
            rotation: Euler angles (A, B, C)

        Returns:
            Rotated vector
        """
        rot = R.from_euler('xyz', [rotation.a, rotation.b, rotation.c])
        return rot.apply(vec)

    def sample_point(
        self,
        startpoint: np.ndarray,
        endpoint: np.ndarray,
        rotation: AxisRotation,
    ) -> Optional[np.ndarray]:
        """
        Sample a single point on the mesh surface.

        Projects a ray from startpoint toward endpoint and finds
        the intersection with the mesh surface.

        Args:
            startpoint: Ray origin (safe position above surface)
            endpoint: Ray target (on or below surface)
            rotation: Tool orientation

        Returns:
            Contact point adjusted for tool geometry, or None if no hit
        """
        # Calculate ray direction
        direction = endpoint - startpoint
        distance = np.linalg.norm(direction)

        if distance < 1e-10:
            return None

        direction_normalized = direction / distance

        # Perform ray cast
        ray_origins = np.array([startpoint])
        ray_directions = np.array([direction_normalized])

        locations, index_ray, index_tri = self._ray_caster.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) == 0:
            return None

        # Find closest intersection
        distances = np.linalg.norm(locations - startpoint, axis=1)
        closest_idx = np.argmin(distances)
        hit_point = locations[closest_idx]

        # Apply tool compensation
        compensation = self._get_tool_compensation(direction_normalized, rotation)
        adjusted_point = hit_point + compensation

        return adjusted_point

    def sample_point_multi_ray(
        self,
        startpoint: np.ndarray,
        endpoint: np.ndarray,
        rotation: AxisRotation,
        num_rays: int = 5
    ) -> Optional[np.ndarray]:
        """
        Sample using multiple rays for better accuracy with complex tool geometry.

        Casts multiple rays in a cone pattern to simulate tool shape.

        Args:
            startpoint: Ray origin
            endpoint: Ray target
            rotation: Tool orientation
            num_rays: Number of rays to cast

        Returns:
            Best contact point, or None
        """
        direction = endpoint - startpoint
        distance = np.linalg.norm(direction)

        if distance < 1e-10:
            return None

        direction_normalized = direction / distance

        # Generate cone of rays around main direction
        ray_origins = []
        ray_directions = []

        # Central ray
        ray_origins.append(startpoint)
        ray_directions.append(direction_normalized)

        # Peripheral rays (if tool has radius)
        if num_rays > 1 and self._tool_radius > 0:
            # Create perpendicular vectors
            perp1 = np.cross(direction_normalized, [1, 0, 0])
            if np.linalg.norm(perp1) < 0.1:
                perp1 = np.cross(direction_normalized, [0, 1, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction_normalized, perp1)

            for i in range(num_rays - 1):
                angle = 2 * np.pi * i / (num_rays - 1)
                offset = (perp1 * np.cos(angle) + perp2 * np.sin(angle)) * self._tool_radius * 0.5
                ray_origins.append(startpoint + offset)
                ray_directions.append(direction_normalized)

        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)

        # Cast all rays
        locations, index_ray, index_tri = self._ray_caster.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) == 0:
            return None

        # Use the closest hit from any ray
        all_distances = []
        for i, loc in enumerate(locations):
            origin_idx = index_ray[i]
            dist = np.linalg.norm(loc - ray_origins[origin_idx])
            all_distances.append((dist, loc, origin_idx))

        all_distances.sort(key=lambda x: x[0])
        best_hit = all_distances[0][1]

        # Apply compensation
        compensation = self._get_tool_compensation(direction_normalized, rotation)
        adjusted_point = best_hit + compensation

        return adjusted_point

    def sample_chunks(
        self,
        pattern_chunks: List[CamPathChunk4Axis],
        layers: Optional[List[Tuple[float, float]]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SamplingResult:
        """
        Sample all pattern chunks onto the mesh surface.

        Port of Fabex sample_chunks_n_axis().

        Args:
            pattern_chunks: Chunks from pattern generator
            layers: List of (start_depth, end_depth) for multi-pass
            progress_callback: Optional callback(percent) for progress

        Returns:
            SamplingResult with sampled chunks
        """
        import time
        start_time = time.time()

        result = SamplingResult()

        # Default to single layer if not specified
        if layers is None or len(layers) == 0:
            layers = [(0.0, 1000.0)]  # Large range to capture everything

        # Create output chunks for each layer
        layer_chunks = [CamPathChunk4Axis(layer_index=i) for i in range(len(layers))]

        # Count total points for progress
        total_points = sum(len(chunk.startpoints) for chunk in pattern_chunks)
        processed = 0

        logger.info(f"Sampling {total_points} points across {len(pattern_chunks)} chunks")

        for chunk_idx, pattern_chunk in enumerate(pattern_chunks):
            if not pattern_chunk.startpoints:
                continue

            for i in range(len(pattern_chunk.startpoints)):
                if progress_callback and processed % 100 == 0:
                    progress_callback(processed / total_points * 100)

                startpoint = np.array(pattern_chunk.startpoints[i])
                endpoint = np.array(pattern_chunk.endpoints[i])
                rotation = pattern_chunk.rotations[i] if i < len(pattern_chunk.rotations) else AxisRotation()

                # Sample this point
                sampled_point = self.sample_point(startpoint, endpoint, rotation)

                if sampled_point is not None:
                    result.sampled_points += 1

                    # Calculate depth (distance from start)
                    depth = np.linalg.norm(sampled_point - startpoint)

                    # Find appropriate layer
                    for layer_idx, (layer_start, layer_end) in enumerate(layers):
                        if layer_start <= depth <= layer_end:
                            layer_chunks[layer_idx].add_point(
                                point=tuple(sampled_point),
                                startpoint=tuple(startpoint),
                                endpoint=tuple(endpoint),
                                rotation=rotation
                            )
                            break
                else:
                    result.missed_points += 1
                    # Use startpoint as fallback (air cut)
                    layer_chunks[0].add_point(
                        point=tuple(startpoint),
                        startpoint=tuple(startpoint),
                        endpoint=tuple(endpoint),
                        rotation=rotation
                    )

                processed += 1
                result.total_points += 1

        # Collect non-empty chunks
        for chunk in layer_chunks:
            if chunk.count() > 0:
                result.chunks.append(chunk)

        result.sampling_time = time.time() - start_time

        logger.info(
            f"Sampling complete: {result.sampled_points}/{result.total_points} points "
            f"sampled in {result.sampling_time:.2f}s"
        )

        if progress_callback:
            progress_callback(100)

        return result

    def sample_chunks_parallel(
        self,
        pattern_chunks: List[CamPathChunk4Axis],
        layers: Optional[List[Tuple[float, float]]] = None,
        num_workers: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SamplingResult:
        """
        Sample chunks using parallel processing.

        For large meshes/toolpaths, this can significantly speed up sampling.

        Args:
            pattern_chunks: Chunks from pattern generator
            layers: Layer definitions
            num_workers: Number of parallel workers
            progress_callback: Progress callback

        Returns:
            SamplingResult
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except ImportError:
            logger.warning("concurrent.futures not available, falling back to sequential")
            return self.sample_chunks(pattern_chunks, layers, progress_callback)

        import time
        start_time = time.time()

        result = SamplingResult()

        if layers is None:
            layers = [(0.0, 1000.0)]

        # Flatten all points for parallel processing
        all_points = []
        for chunk_idx, chunk in enumerate(pattern_chunks):
            for i in range(len(chunk.startpoints)):
                all_points.append((
                    chunk_idx, i,
                    np.array(chunk.startpoints[i]),
                    np.array(chunk.endpoints[i]),
                    chunk.rotations[i] if i < len(chunk.rotations) else AxisRotation()
                ))

        total_points = len(all_points)
        processed = [0]  # Use list for closure

        def sample_one(point_data):
            chunk_idx, point_idx, start, end, rot = point_data
            sampled = self.sample_point(start, end, rot)
            return (chunk_idx, point_idx, start, end, rot, sampled)

        # Process in parallel
        layer_chunks = [CamPathChunk4Axis(layer_index=i) for i in range(len(layers))]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(sample_one, p): p for p in all_points}

            for future in as_completed(futures):
                chunk_idx, point_idx, start, end, rot, sampled = future.result()

                if sampled is not None:
                    result.sampled_points += 1
                    depth = np.linalg.norm(sampled - start)

                    for layer_idx, (layer_start, layer_end) in enumerate(layers):
                        if layer_start <= depth <= layer_end:
                            layer_chunks[layer_idx].add_point(
                                point=tuple(sampled),
                                startpoint=tuple(start),
                                endpoint=tuple(end),
                                rotation=rot
                            )
                            break
                else:
                    result.missed_points += 1
                    layer_chunks[0].add_point(
                        point=tuple(start),
                        startpoint=tuple(start),
                        endpoint=tuple(end),
                        rotation=rot
                    )

                processed[0] += 1
                result.total_points += 1

                if progress_callback and processed[0] % 100 == 0:
                    progress_callback(processed[0] / total_points * 100)

        for chunk in layer_chunks:
            if chunk.count() > 0:
                result.chunks.append(chunk)

        result.sampling_time = time.time() - start_time

        if progress_callback:
            progress_callback(100)

        return result


class AdaptiveSampler(NAxisSampler):
    """
    Adaptive surface sampler that adjusts density based on curvature.

    Areas with high curvature get denser sampling for better accuracy.
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        tool: Tool,
        min_step: float = 0.1,
        max_step: float = 2.0,
        curvature_threshold: float = 0.1
    ):
        """
        Initialize adaptive sampler.

        Args:
            mesh: Workpiece mesh
            tool: Tool definition
            min_step: Minimum step size in high-curvature areas
            max_step: Maximum step size in flat areas
            curvature_threshold: Curvature value to start adapting
        """
        super().__init__(mesh, tool)
        self.min_step = min_step
        self.max_step = max_step
        self.curvature_threshold = curvature_threshold

        # Precompute vertex curvatures
        self._compute_curvature()

    def _compute_curvature(self) -> None:
        """Compute mean curvature at mesh vertices."""
        try:
            # Use trimesh's discrete mean curvature
            self._vertex_curvature = trimesh.curvature.discrete_mean_curvature_measure(
                self.mesh,
                self.mesh.vertices,
                radius=self.max_step
            )
        except Exception as e:
            logger.warning(f"Could not compute curvature: {e}, using uniform sampling")
            self._vertex_curvature = np.zeros(len(self.mesh.vertices))

    def get_adaptive_step(self, point: np.ndarray) -> float:
        """
        Get adaptive step size based on local curvature.

        Args:
            point: 3D point to query

        Returns:
            Recommended step size
        """
        # Find nearest vertex
        distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
        nearest_idx = np.argmin(distances)

        curvature = abs(self._vertex_curvature[nearest_idx])

        if curvature > self.curvature_threshold:
            # High curvature: use small step
            t = min(1.0, curvature / self.curvature_threshold)
            return self.min_step + (self.max_step - self.min_step) * (1 - t)
        else:
            return self.max_step
