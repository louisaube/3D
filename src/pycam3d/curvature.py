"""
Curvature analysis module for adaptive toolpath generation.

This module provides:
- Local curvature estimation on mesh surfaces
- Scallop height calculation based on tool geometry and curvature
- Adaptive stepover computation for iso-scallop machining
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class CurvatureField:
    """
    Curvature information for a mesh surface.

    Attributes:
        vertices: Vertex positions (N, 3)
        principal_curvatures_1: First principal curvature at each vertex
        principal_curvatures_2: Second principal curvature at each vertex
        principal_directions_1: First principal direction at each vertex (N, 3)
        principal_directions_2: Second principal direction at each vertex (N, 3)
        gaussian: Gaussian curvature (k1 * k2)
        mean: Mean curvature ((k1 + k2) / 2)
    """

    vertices: np.ndarray
    principal_curvatures_1: np.ndarray
    principal_curvatures_2: np.ndarray
    principal_directions_1: np.ndarray
    principal_directions_2: np.ndarray
    gaussian: np.ndarray
    mean: np.ndarray

    @property
    def max_curvature(self) -> np.ndarray:
        """Maximum absolute curvature at each vertex."""
        return np.maximum(
            np.abs(self.principal_curvatures_1),
            np.abs(self.principal_curvatures_2)
        )

    @property
    def min_curvature(self) -> np.ndarray:
        """Minimum absolute curvature at each vertex."""
        return np.minimum(
            np.abs(self.principal_curvatures_1),
            np.abs(self.principal_curvatures_2)
        )


class CurvatureAnalyzer:
    """
    Analyze surface curvature for adaptive machining.

    Uses discrete differential geometry to estimate principal curvatures
    and directions on mesh surfaces.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize with a mesh.

        Args:
            mesh: Input trimesh mesh.
        """
        self.mesh = mesh
        self._curvature_field: Optional[CurvatureField] = None

    def compute_curvature(self) -> CurvatureField:
        """
        Compute principal curvatures using discrete Laplacian method.

        Returns:
            CurvatureField with curvature information.
        """
        logger.info("Computing surface curvature...")

        vertices = self.mesh.vertices
        faces = self.mesh.faces
        n_vertices = len(vertices)

        # Compute vertex normals
        vertex_normals = self.mesh.vertex_normals

        # Compute mean curvature using cotangent Laplacian
        # H = (1/2A) * sum(cot(alpha) + cot(beta)) * (vi - vj)
        mean_curvature = self._compute_mean_curvature_laplacian()

        # Estimate principal curvatures from mean curvature
        # For surfaces, we use shape operator eigenvalues
        # Simplified: k1 ≈ H + sqrt(H² - K), k2 ≈ H - sqrt(H² - K)
        # We'll estimate Gaussian curvature from angle defect

        gaussian_curvature = self._compute_gaussian_curvature_angle_defect()

        # Compute principal curvatures from mean and Gaussian
        discriminant = mean_curvature**2 - gaussian_curvature
        discriminant = np.maximum(discriminant, 0)  # Numerical stability
        sqrt_disc = np.sqrt(discriminant)

        k1 = mean_curvature + sqrt_disc
        k2 = mean_curvature - sqrt_disc

        # Estimate principal directions (simplified - use normal plane basis)
        # For accurate directions, need shape operator eigenvectors
        d1, d2 = self._compute_principal_directions_simple(vertex_normals)

        self._curvature_field = CurvatureField(
            vertices=vertices,
            principal_curvatures_1=k1,
            principal_curvatures_2=k2,
            principal_directions_1=d1,
            principal_directions_2=d2,
            gaussian=gaussian_curvature,
            mean=mean_curvature,
        )

        logger.info(
            f"Curvature computed: mean range [{mean_curvature.min():.4f}, {mean_curvature.max():.4f}]"
        )

        return self._curvature_field

    def _compute_mean_curvature_laplacian(self) -> np.ndarray:
        """Compute mean curvature using cotangent Laplacian."""
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        n_vertices = len(vertices)

        # Build cotangent weights
        mean_curvature = np.zeros(n_vertices)
        area = np.zeros(n_vertices)

        for face in faces:
            i, j, k = face
            vi, vj, vk = vertices[i], vertices[j], vertices[k]

            # Edge vectors
            eij = vj - vi
            ejk = vk - vj
            eki = vi - vk

            # Cotangent weights
            cot_i = self._cotangent(eij, -eki)
            cot_j = self._cotangent(ejk, -eij)
            cot_k = self._cotangent(eki, -ejk)

            # Triangle area (for normalization)
            tri_area = 0.5 * np.linalg.norm(np.cross(eij, -eki))
            area[[i, j, k]] += tri_area / 3

            # Laplacian contribution
            mean_curvature[i] += cot_j * np.linalg.norm(eki) + cot_k * np.linalg.norm(eij)
            mean_curvature[j] += cot_k * np.linalg.norm(eij) + cot_i * np.linalg.norm(ejk)
            mean_curvature[k] += cot_i * np.linalg.norm(ejk) + cot_j * np.linalg.norm(eki)

        # Normalize by area
        area = np.maximum(area, 1e-10)  # Avoid division by zero
        mean_curvature = mean_curvature / (4 * area)

        return mean_curvature

    def _compute_gaussian_curvature_angle_defect(self) -> np.ndarray:
        """Compute Gaussian curvature using angle defect method."""
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        n_vertices = len(vertices)

        # Angle sum at each vertex
        angle_sum = np.zeros(n_vertices)
        area = np.zeros(n_vertices)

        for face in faces:
            i, j, k = face
            vi, vj, vk = vertices[i], vertices[j], vertices[k]

            # Compute angles at each vertex
            angle_i = self._angle_between(vj - vi, vk - vi)
            angle_j = self._angle_between(vi - vj, vk - vj)
            angle_k = self._angle_between(vi - vk, vj - vk)

            angle_sum[i] += angle_i
            angle_sum[j] += angle_j
            angle_sum[k] += angle_k

            # Mixed area (Voronoi area approximation)
            tri_area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
            area[[i, j, k]] += tri_area / 3

        # Gaussian curvature = (2π - angle_sum) / area
        area = np.maximum(area, 1e-10)
        gaussian = (2 * np.pi - angle_sum) / area

        return gaussian

    def _compute_principal_directions_simple(
        self, normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute approximate principal directions.

        Uses the tangent plane basis as approximation.
        For accurate results, need full shape operator computation.
        """
        n_vertices = len(normals)

        # Create orthonormal basis in tangent plane
        d1 = np.zeros((n_vertices, 3))
        d2 = np.zeros((n_vertices, 3))

        for i, n in enumerate(normals):
            # Find a vector not parallel to normal
            if abs(n[0]) < 0.9:
                t = np.array([1, 0, 0])
            else:
                t = np.array([0, 1, 0])

            # Gram-Schmidt
            d1[i] = t - np.dot(t, n) * n
            d1[i] /= np.linalg.norm(d1[i]) + 1e-10

            d2[i] = np.cross(n, d1[i])
            d2[i] /= np.linalg.norm(d2[i]) + 1e-10

        return d1, d2

    @staticmethod
    def _cotangent(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cotangent of angle between two vectors."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        sin_angle = np.sqrt(1 - cos_angle**2) + 1e-10
        return cos_angle / sin_angle

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors in radians."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle)

    def get_curvature_at_point(self, point: np.ndarray) -> Tuple[float, float]:
        """
        Get interpolated curvature at a point.

        Args:
            point: 3D point on or near the surface.

        Returns:
            Tuple of (k1, k2) principal curvatures.
        """
        if self._curvature_field is None:
            self.compute_curvature()

        # Find closest vertex
        distances = np.linalg.norm(self._curvature_field.vertices - point, axis=1)
        closest_idx = np.argmin(distances)

        return (
            self._curvature_field.principal_curvatures_1[closest_idx],
            self._curvature_field.principal_curvatures_2[closest_idx],
        )


def compute_scallop_height(
    tool_radius: float,
    stepover: float,
    surface_curvature: float = 0.0,
) -> float:
    """
    Compute scallop height for ball-end mill.

    The scallop height depends on:
    - Tool radius (R)
    - Stepover distance (s)
    - Local surface curvature (κ)

    For a flat surface: h = R - sqrt(R² - (s/2)²)
    For curved surface: effective radius changes

    Args:
        tool_radius: Ball-end mill radius.
        stepover: Distance between adjacent passes.
        surface_curvature: Local surface curvature (positive = convex).

    Returns:
        Scallop height in same units as inputs.
    """
    # Effective radius considering surface curvature
    # R_eff = R / (1 - R * κ) for convex, R / (1 + R * |κ|) for concave
    if surface_curvature > 0:  # Convex
        r_eff = tool_radius / (1 - tool_radius * surface_curvature + 1e-10)
    else:  # Flat or concave
        r_eff = tool_radius / (1 + tool_radius * abs(surface_curvature) + 1e-10)

    # Clamp effective radius
    r_eff = max(r_eff, tool_radius * 0.1)
    r_eff = min(r_eff, tool_radius * 10)

    # Scallop height formula
    half_step = stepover / 2
    if half_step >= r_eff:
        return r_eff  # Maximum possible

    scallop = r_eff - np.sqrt(max(0, r_eff**2 - half_step**2))
    return scallop


def compute_stepover_for_scallop(
    tool_radius: float,
    target_scallop: float,
    surface_curvature: float = 0.0,
) -> float:
    """
    Compute stepover to achieve target scallop height.

    Inverse of compute_scallop_height.

    Args:
        tool_radius: Ball-end mill radius.
        target_scallop: Desired scallop height.
        surface_curvature: Local surface curvature.

    Returns:
        Required stepover distance.
    """
    # Effective radius
    if surface_curvature > 0:
        r_eff = tool_radius / (1 - tool_radius * surface_curvature + 1e-10)
    else:
        r_eff = tool_radius / (1 + tool_radius * abs(surface_curvature) + 1e-10)

    r_eff = max(r_eff, tool_radius * 0.1)
    r_eff = min(r_eff, tool_radius * 10)

    # Clamp scallop
    h = min(target_scallop, r_eff * 0.9)
    h = max(h, 0.001)

    # Solve: h = r_eff - sqrt(r_eff² - (s/2)²)
    # => (s/2)² = r_eff² - (r_eff - h)²
    # => s = 2 * sqrt(2*r_eff*h - h²)

    discriminant = 2 * r_eff * h - h**2
    if discriminant <= 0:
        return tool_radius * 0.1  # Minimum stepover

    stepover = 2 * np.sqrt(discriminant)

    # Clamp to reasonable range
    stepover = max(stepover, tool_radius * 0.05)
    stepover = min(stepover, tool_radius * 1.5)

    return stepover
