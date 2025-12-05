"""
Mesh processing module using Trimesh and Open3D.

This module handles:
- Loading various 3D file formats (STL, PLY, OBJ, GLTF, point clouds)
- Point cloud to mesh reconstruction
- Mesh cleaning and repair
- Mesh transformations and analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class ReconstructionMethod(Enum):
    """Methods for reconstructing mesh from point cloud."""

    POISSON = "poisson"
    BALL_PIVOTING = "ball_pivoting"
    ALPHA_SHAPE = "alpha_shape"


@dataclass
class MeshStats:
    """Statistics about a mesh."""

    vertex_count: int
    face_count: int
    is_watertight: bool
    is_winding_consistent: bool
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    volume: Optional[float]
    surface_area: float

    def __str__(self) -> str:
        volume_str = f"{self.volume:.2f}" if self.volume is not None else "N/A"
        return (
            f"MeshStats(\n"
            f"  vertices: {self.vertex_count:,}\n"
            f"  faces: {self.face_count:,}\n"
            f"  watertight: {self.is_watertight}\n"
            f"  winding_consistent: {self.is_winding_consistent}\n"
            f"  bounds: {self.bounds_min} -> {self.bounds_max}\n"
            f"  volume: {volume_str}\n"
            f"  surface_area: {self.surface_area:.2f}\n"
            f")"
        )


class MeshProcessor:
    """
    Processor for 3D mesh data using Trimesh and Open3D.

    This class provides methods for loading, cleaning, repairing,
    and transforming 3D mesh data from various sources.
    """

    def __init__(self):
        self._mesh: Optional[trimesh.Trimesh] = None
        self._original_mesh: Optional[trimesh.Trimesh] = None

    @property
    def mesh(self) -> Optional[trimesh.Trimesh]:
        """Get the current mesh."""
        return self._mesh

    @property
    def has_mesh(self) -> bool:
        """Check if a mesh is loaded."""
        return self._mesh is not None

    def load_mesh(self, filepath: str | Path) -> trimesh.Trimesh:
        """
        Load a mesh from file.

        Supports STL, PLY, OBJ, GLTF/GLB, OFF, and other formats
        supported by trimesh.

        Args:
            filepath: Path to the mesh file.

        Returns:
            The loaded mesh.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        logger.info(f"Loading mesh from {filepath}")

        loaded = trimesh.load(filepath)

        # Handle scene vs single mesh
        if isinstance(loaded, trimesh.Scene):
            # Combine all meshes in the scene
            meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No valid meshes found in scene")
            self._mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            self._mesh = loaded
        else:
            raise ValueError(f"Unsupported geometry type: {type(loaded)}")

        self._original_mesh = self._mesh.copy()
        logger.info(f"Loaded mesh with {len(self._mesh.vertices)} vertices, {len(self._mesh.faces)} faces")

        return self._mesh

    def load_point_cloud(
        self,
        filepath: str | Path,
        reconstruction_method: ReconstructionMethod = ReconstructionMethod.POISSON,
        depth: int = 9,
        radii: Optional[list[float]] = None,
    ) -> trimesh.Trimesh:
        """
        Load a point cloud and reconstruct it as a mesh.

        Uses Open3D for point cloud processing and mesh reconstruction.

        Args:
            filepath: Path to the point cloud file (PLY, PCD, XYZ, etc.).
            reconstruction_method: Method to use for reconstruction.
            depth: Depth parameter for Poisson reconstruction (default 9).
            radii: Radii for ball pivoting (if None, auto-calculated).

        Returns:
            The reconstructed mesh.
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is required for point cloud processing. Install with: pip install open3d")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Point cloud file not found: {filepath}")

        logger.info(f"Loading point cloud from {filepath}")
        pcd = o3d.io.read_point_cloud(str(filepath))

        if len(pcd.points) == 0:
            raise ValueError("Point cloud is empty")

        logger.info(f"Loaded {len(pcd.points)} points")

        # Estimate normals if not present
        if not pcd.has_normals():
            logger.info("Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Reconstruct mesh
        logger.info(f"Reconstructing mesh using {reconstruction_method.value}...")

        if reconstruction_method == ReconstructionMethod.POISSON:
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            # Remove low-density vertices (cleanup)
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        elif reconstruction_method == ReconstructionMethod.BALL_PIVOTING:
            if radii is None:
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

        elif reconstruction_method == ReconstructionMethod.ALPHA_SHAPE:
            alpha = 0.03  # Default alpha value
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        else:
            raise ValueError(f"Unknown reconstruction method: {reconstruction_method}")

        # Convert Open3D mesh to Trimesh
        self._mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh_o3d.vertices),
            faces=np.asarray(mesh_o3d.triangles),
            vertex_normals=np.asarray(mesh_o3d.vertex_normals) if mesh_o3d.has_vertex_normals() else None,
        )

        self._original_mesh = self._mesh.copy()
        logger.info(f"Reconstructed mesh with {len(self._mesh.vertices)} vertices, {len(self._mesh.faces)} faces")

        return self._mesh

    def get_stats(self) -> MeshStats:
        """Get statistics about the current mesh."""
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        volume = None
        if self._mesh.is_watertight:
            try:
                volume = self._mesh.volume
            except Exception:
                pass

        return MeshStats(
            vertex_count=len(self._mesh.vertices),
            face_count=len(self._mesh.faces),
            is_watertight=self._mesh.is_watertight,
            is_winding_consistent=self._mesh.is_winding_consistent,
            bounds_min=self._mesh.bounds[0],
            bounds_max=self._mesh.bounds[1],
            volume=volume,
            surface_area=self._mesh.area,
        )

    def fix_normals(self) -> None:
        """Fix mesh normals to be consistent and outward-facing."""
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        logger.info("Fixing normals...")
        self._mesh.fix_normals()

    def fill_holes(self) -> int:
        """
        Fill holes in the mesh.

        Returns:
            Number of holes filled.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        logger.info("Filling holes...")
        initial_faces = len(self._mesh.faces)
        self._mesh.fill_holes()
        new_faces = len(self._mesh.faces) - initial_faces

        if new_faces > 0:
            logger.info(f"Added {new_faces} faces to fill holes")
        return new_faces

    def remove_duplicates(self) -> Tuple[int, int]:
        """
        Remove duplicate vertices and faces.

        Returns:
            Tuple of (vertices_removed, faces_removed).
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        logger.info("Removing duplicates...")
        initial_verts = len(self._mesh.vertices)
        initial_faces = len(self._mesh.faces)

        # Merge duplicate vertices
        self._mesh.merge_vertices()

        # Remove degenerate faces (using nondegenerate_faces mask)
        if hasattr(self._mesh, 'nondegenerate_faces'):
            valid_faces = self._mesh.nondegenerate_faces()
            if not np.all(valid_faces):
                self._mesh.update_faces(valid_faces)

        # Remove unreferenced vertices
        self._mesh.remove_unreferenced_vertices()

        verts_removed = initial_verts - len(self._mesh.vertices)
        faces_removed = initial_faces - len(self._mesh.faces)

        logger.info(f"Removed {verts_removed} vertices, {faces_removed} faces")
        return verts_removed, faces_removed

    def smooth(self, iterations: int = 1, lamb: float = 0.5) -> None:
        """
        Apply Laplacian smoothing to the mesh.

        Args:
            iterations: Number of smoothing iterations.
            lamb: Smoothing factor (0-1).
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        logger.info(f"Smoothing mesh ({iterations} iterations, lambda={lamb})...")
        self._mesh = trimesh.smoothing.filter_laplacian(
            self._mesh, iterations=iterations, lamb=lamb
        )

    def simplify(self, target_faces: int) -> None:
        """
        Simplify the mesh to a target number of faces.

        Args:
            target_faces: Target number of faces after simplification.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        if target_faces >= len(self._mesh.faces):
            logger.warning("Target faces >= current faces, skipping simplification")
            return

        logger.info(f"Simplifying mesh from {len(self._mesh.faces)} to {target_faces} faces...")
        self._mesh = self._mesh.simplify_quadric_decimation(target_faces)

    def scale(self, factor: float) -> None:
        """Scale the mesh uniformly."""
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        self._mesh.apply_scale(factor)

    def translate(self, offset: np.ndarray | list) -> None:
        """Translate the mesh."""
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        self._mesh.apply_translation(np.array(offset))

    def center(self) -> np.ndarray:
        """
        Center the mesh at the origin.

        Returns:
            The translation applied.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        centroid = self._mesh.centroid
        self._mesh.apply_translation(-centroid)
        return -centroid

    def place_on_bed(self) -> float:
        """
        Move the mesh so its lowest point is at Z=0.

        Returns:
            The Z translation applied.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        z_min = self._mesh.bounds[0][2]
        self._mesh.apply_translation([0, 0, -z_min])
        return -z_min

    def auto_orient(self) -> None:
        """Automatically orient the mesh for optimal machining."""
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        # Use trimesh's principal axes to find optimal orientation
        transform = self._mesh.principal_inertia_transform
        self._mesh.apply_transform(transform)
        self.place_on_bed()

    def repair(self) -> dict:
        """
        Perform comprehensive mesh repair.

        Returns:
            Dictionary with repair statistics.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        logger.info("Starting comprehensive mesh repair...")
        stats = {
            "duplicates_removed": self.remove_duplicates(),
            "holes_filled": self.fill_holes(),
            "normals_fixed": True,
        }

        self.fix_normals()
        logger.info("Mesh repair complete")

        return stats

    def reset(self) -> None:
        """Reset mesh to original loaded state."""
        if self._original_mesh is None:
            raise ValueError("No original mesh to reset to")

        self._mesh = self._original_mesh.copy()
        logger.info("Mesh reset to original state")

    def export(self, filepath: str | Path, file_type: Optional[str] = None) -> None:
        """
        Export the mesh to a file.

        Args:
            filepath: Output file path.
            file_type: Output format (inferred from extension if not provided).
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting mesh to {filepath}")
        self._mesh.export(filepath, file_type=file_type)

    def get_cross_section(self, z_height: float) -> list:
        """
        Get a 2D cross-section of the mesh at a given Z height.

        Args:
            z_height: The Z coordinate for the slice.

        Returns:
            List of Path2D objects representing the cross-section contours.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        section = self._mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])

        if section is None:
            return []

        return section.to_planar()[0]

    def slice_layers(self, layer_height: float) -> list:
        """
        Slice the mesh into multiple layers.

        Args:
            layer_height: Height of each layer.

        Returns:
            List of (z_height, paths) tuples.
        """
        if self._mesh is None:
            raise ValueError("No mesh loaded")

        z_min, z_max = self._mesh.bounds[0][2], self._mesh.bounds[1][2]
        z_heights = np.arange(z_min + layer_height / 2, z_max, layer_height)

        layers = []
        for z in z_heights:
            paths = self.get_cross_section(z)
            if paths:
                layers.append((z, paths))

        return layers
