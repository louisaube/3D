"""
Automatic part orientation and accessibility analysis for PyCAM3D.

Provides:
- Automatic optimal part orientation
- Undercut detection
- Accessibility analysis for 3-axis machining
- Multi-setup planning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import numpy as np
import math


class MachiningAxis(Enum):
    """Primary machining axis direction."""
    Z_DOWN = "z_down"   # Standard 3-axis, tool points down
    Z_UP = "z_up"       # Flip part
    X_PLUS = "x_plus"   # 4th axis or multi-setup
    X_MINUS = "x_minus"
    Y_PLUS = "y_plus"
    Y_MINUS = "y_minus"


@dataclass
class Orientation:
    """Part orientation for machining."""
    rotation_x: float = 0.0  # Rotation around X axis (degrees)
    rotation_y: float = 0.0  # Rotation around Y axis (degrees)
    rotation_z: float = 0.0  # Rotation around Z axis (degrees)
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def get_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        rx = np.radians(self.rotation_x)
        ry = np.radians(self.rotation_y)
        rz = np.radians(self.rotation_z)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def apply_to_mesh(self, vertices: np.ndarray) -> np.ndarray:
        """Apply orientation to mesh vertices."""
        R = self.get_rotation_matrix()
        rotated = vertices @ R.T
        return rotated + self.translation

    def to_dict(self) -> Dict:
        return {
            "rotation_x": self.rotation_x,
            "rotation_y": self.rotation_y,
            "rotation_z": self.rotation_z,
            "translation": self.translation.tolist(),
        }


@dataclass
class AccessibilityRegion:
    """Region classified by machining accessibility."""
    accessible: bool
    face_indices: np.ndarray
    normal_direction: np.ndarray
    area: float
    description: str


@dataclass
class AccessibilityAnalysis:
    """Results of accessibility analysis."""
    total_surface_area: float
    accessible_area: float
    accessible_percentage: float
    undercut_area: float
    regions: List[AccessibilityRegion]
    critical_undercuts: List[Dict]  # List of problematic undercut locations

    @property
    def fully_accessible(self) -> bool:
        return self.accessible_percentage >= 99.0


class AccessibilityAnalyzer:
    """Analyze mesh accessibility for 3-axis machining."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        """
        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)
            normals: Face normals (M, 3)
        """
        self.vertices = vertices
        self.faces = faces
        self.normals = normals

    def analyze(self, tool_axis: np.ndarray = np.array([0, 0, 1]),
                max_angle: float = 90.0) -> AccessibilityAnalysis:
        """Analyze accessibility from a tool direction.

        Args:
            tool_axis: Tool approach direction (normalized)
            max_angle: Maximum angle from vertical that's accessible (degrees)

        Returns:
            AccessibilityAnalysis with detailed results
        """
        # Normalize tool axis
        tool_axis = tool_axis / np.linalg.norm(tool_axis)

        # Calculate angle between each face normal and tool axis
        # Accessible if normal points somewhat toward tool (angle < max_angle)
        dot_products = np.dot(self.normals, tool_axis)
        angles = np.degrees(np.arccos(np.clip(dot_products, -1, 1)))

        # Face is accessible if angle to tool is less than max_angle
        accessible_mask = angles < max_angle

        # Calculate face areas
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)

        total_area = np.sum(face_areas)
        accessible_area = np.sum(face_areas[accessible_mask])
        undercut_area = total_area - accessible_area

        # Find critical undercuts (large areas that are inaccessible)
        critical_undercuts = []
        undercut_indices = np.where(~accessible_mask)[0]
        for idx in undercut_indices:
            if face_areas[idx] > total_area * 0.01:  # More than 1% of total
                center = self.vertices[self.faces[idx]].mean(axis=0)
                critical_undercuts.append({
                    "face_index": int(idx),
                    "center": center.tolist(),
                    "area": float(face_areas[idx]),
                    "normal": self.normals[idx].tolist(),
                    "angle": float(angles[idx]),
                })

        # Group into regions
        regions = []
        if np.any(accessible_mask):
            regions.append(AccessibilityRegion(
                accessible=True,
                face_indices=np.where(accessible_mask)[0],
                normal_direction=tool_axis,
                area=float(accessible_area),
                description="Accessible from tool direction",
            ))
        if np.any(~accessible_mask):
            regions.append(AccessibilityRegion(
                accessible=False,
                face_indices=np.where(~accessible_mask)[0],
                normal_direction=-tool_axis,
                area=float(undercut_area),
                description="Undercut - not accessible",
            ))

        return AccessibilityAnalysis(
            total_surface_area=float(total_area),
            accessible_area=float(accessible_area),
            accessible_percentage=float(accessible_area / total_area * 100) if total_area > 0 else 0,
            undercut_area=float(undercut_area),
            regions=regions,
            critical_undercuts=sorted(critical_undercuts, key=lambda x: -x["area"]),
        )


class OrientationOptimizer:
    """Find optimal part orientation for machining."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.analyzer = AccessibilityAnalyzer(vertices, faces, normals)

    def find_optimal_orientation(self, single_setup: bool = True,
                                 prefer_flat_base: bool = True) -> Tuple[Orientation, AccessibilityAnalysis]:
        """Find the best orientation for machining.

        Args:
            single_setup: Only consider orientations achievable in one setup
            prefer_flat_base: Prefer orientations with a flat base for stable fixturing

        Returns:
            Tuple of (best_orientation, analysis)
        """
        # Standard orientations to try
        orientations_to_try = [
            Orientation(0, 0, 0),     # Original
            Orientation(180, 0, 0),   # Flip X
            Orientation(0, 180, 0),   # Flip Y
            Orientation(90, 0, 0),    # Rotate 90 X
            Orientation(-90, 0, 0),   # Rotate -90 X
            Orientation(0, 90, 0),    # Rotate 90 Y
            Orientation(0, -90, 0),   # Rotate -90 Y
            Orientation(0, 0, 90),    # Rotate 90 Z
        ]

        best_orientation = orientations_to_try[0]  # Default to original orientation
        best_analysis = self.analyzer.analyze()
        best_score = -float('inf')

        for orient in orientations_to_try:
            # Apply orientation to normals
            R = orient.get_rotation_matrix()
            rotated_normals = self.normals @ R.T
            rotated_vertices = self.vertices @ R.T

            # Analyze accessibility
            temp_analyzer = AccessibilityAnalyzer(
                rotated_vertices, self.faces, rotated_normals
            )
            analysis = temp_analyzer.analyze()

            # Calculate score
            score = self._calculate_orientation_score(
                orient, analysis, rotated_vertices, prefer_flat_base
            )

            if score > best_score:
                best_score = score
                best_orientation = orient
                best_analysis = analysis

        # Center and place on bed
        if best_orientation:
            R = best_orientation.get_rotation_matrix()
            rotated_vertices = self.vertices @ R.T
            bounds_min = rotated_vertices.min(axis=0)
            bounds_max = rotated_vertices.max(axis=0)

            # Center XY, place Z on bed
            best_orientation.translation = np.array([
                -(bounds_min[0] + bounds_max[0]) / 2,
                -(bounds_min[1] + bounds_max[1]) / 2,
                -bounds_min[2],
            ])

        return best_orientation, best_analysis

    def _calculate_orientation_score(self, orientation: Orientation,
                                     analysis: AccessibilityAnalysis,
                                     vertices: np.ndarray,
                                     prefer_flat_base: bool) -> float:
        """Calculate score for an orientation.

        Higher score = better orientation
        """
        score = 0.0

        # Primary: accessibility percentage (0-100 points)
        score += analysis.accessible_percentage

        # Penalty for critical undercuts (-50 points per significant undercut)
        score -= len(analysis.critical_undercuts) * 50

        # Bonus for flat base
        if prefer_flat_base:
            # Check if bottom is relatively flat
            bottom_z = vertices[:, 2].min()
            bottom_vertices = vertices[vertices[:, 2] < bottom_z + 1.0]
            if len(bottom_vertices) > 10:
                z_variance = np.var(bottom_vertices[:, 2])
                if z_variance < 0.1:
                    score += 20  # Flat base bonus

        # Slight preference for no rotation (simpler setup)
        if orientation.rotation_x == 0 and orientation.rotation_y == 0:
            score += 5

        return score

    def suggest_multi_setup(self, max_setups: int = 3) -> List[Tuple[Orientation, AccessibilityAnalysis]]:
        """Suggest multiple setups to achieve full coverage.

        Returns:
            List of (orientation, analysis) tuples for each setup
        """
        setups = []
        remaining_faces = set(range(len(self.faces)))

        # Primary setup - top down
        orient1, analysis1 = self.find_optimal_orientation()
        setups.append((orient1, analysis1))

        # Remove accessible faces
        for region in analysis1.regions:
            if region.accessible:
                remaining_faces -= set(region.face_indices.tolist())

        if len(remaining_faces) == 0 or len(setups) >= max_setups:
            return setups

        # Second setup - flip 180 (bottom access)
        orient2 = Orientation(180, 0, 0)
        R = orient2.get_rotation_matrix()
        rotated_normals = self.normals @ R.T
        rotated_vertices = self.vertices @ R.T

        temp_analyzer = AccessibilityAnalyzer(
            rotated_vertices, self.faces, rotated_normals
        )
        analysis2 = temp_analyzer.analyze()
        setups.append((orient2, analysis2))

        # Update remaining
        for region in analysis2.regions:
            if region.accessible:
                remaining_faces -= set(region.face_indices.tolist())

        if len(remaining_faces) == 0 or len(setups) >= max_setups:
            return setups

        # Third setup - side access if needed
        orient3 = Orientation(90, 0, 0)
        R = orient3.get_rotation_matrix()
        rotated_normals = self.normals @ R.T
        rotated_vertices = self.vertices @ R.T

        temp_analyzer = AccessibilityAnalyzer(
            rotated_vertices, self.faces, rotated_normals
        )
        analysis3 = temp_analyzer.analyze()
        setups.append((orient3, analysis3))

        return setups


def analyze_mesh_accessibility(mesh) -> AccessibilityAnalysis:
    """Convenience function to analyze mesh accessibility.

    Args:
        mesh: Trimesh mesh object

    Returns:
        AccessibilityAnalysis
    """
    analyzer = AccessibilityAnalyzer(
        vertices=np.array(mesh.vertices),
        faces=np.array(mesh.faces),
        normals=np.array(mesh.face_normals),
    )
    return analyzer.analyze()


def find_optimal_orientation(mesh) -> Tuple[Orientation, AccessibilityAnalysis]:
    """Convenience function to find optimal orientation.

    Args:
        mesh: Trimesh mesh object

    Returns:
        Tuple of (orientation, analysis)
    """
    optimizer = OrientationOptimizer(
        vertices=np.array(mesh.vertices),
        faces=np.array(mesh.faces),
        normals=np.array(mesh.face_normals),
    )
    return optimizer.find_optimal_orientation()
