"""Tests for mesh processing module."""

import numpy as np
import pytest
import trimesh

from pycam3d.mesh import MeshProcessor, MeshStats


@pytest.fixture
def sample_mesh():
    """Create a simple cube mesh for testing."""
    return trimesh.creation.box(extents=[10, 10, 10])


@pytest.fixture
def processor_with_mesh(sample_mesh, tmp_path):
    """Create a MeshProcessor with a loaded mesh."""
    # Save mesh to temp file
    mesh_path = tmp_path / "test_cube.stl"
    sample_mesh.export(mesh_path)

    # Load into processor
    processor = MeshProcessor()
    processor.load_mesh(mesh_path)
    return processor


class TestMeshProcessor:
    """Tests for MeshProcessor class."""

    def test_init(self):
        """Test processor initialization."""
        processor = MeshProcessor()
        assert processor.mesh is None
        assert not processor.has_mesh

    def test_load_mesh(self, tmp_path):
        """Test loading a mesh file."""
        # Create and save a mesh
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        mesh_path = tmp_path / "test.stl"
        mesh.export(mesh_path)

        processor = MeshProcessor()
        loaded = processor.load_mesh(mesh_path)

        assert processor.has_mesh
        assert loaded is not None
        assert len(loaded.vertices) > 0
        assert len(loaded.faces) > 0

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises error."""
        processor = MeshProcessor()
        with pytest.raises(FileNotFoundError):
            processor.load_mesh("/nonexistent/path.stl")

    def test_get_stats(self, processor_with_mesh):
        """Test getting mesh statistics."""
        stats = processor_with_mesh.get_stats()

        assert isinstance(stats, MeshStats)
        assert stats.vertex_count == 8  # Cube has 8 vertices
        assert stats.face_count == 12  # Cube has 12 triangles
        assert stats.is_watertight
        assert stats.surface_area > 0

    def test_fix_normals(self, processor_with_mesh):
        """Test fixing mesh normals."""
        # Should not raise
        processor_with_mesh.fix_normals()
        stats = processor_with_mesh.get_stats()
        assert stats.is_winding_consistent

    def test_center(self, processor_with_mesh):
        """Test centering mesh."""
        processor_with_mesh.translate([50, 50, 50])  # Move away from origin
        offset = processor_with_mesh.center()

        # Centroid should now be at origin
        centroid = processor_with_mesh.mesh.centroid
        assert np.allclose(centroid, [0, 0, 0], atol=1e-6)

    def test_place_on_bed(self, processor_with_mesh):
        """Test placing mesh on bed (Z=0)."""
        processor_with_mesh.translate([0, 0, 50])  # Move up
        processor_with_mesh.place_on_bed()

        z_min = processor_with_mesh.mesh.bounds[0][2]
        assert np.isclose(z_min, 0, atol=1e-6)

    def test_scale(self, processor_with_mesh):
        """Test scaling mesh."""
        original_bounds = processor_with_mesh.mesh.bounds.copy()
        processor_with_mesh.scale(2.0)

        new_bounds = processor_with_mesh.mesh.bounds
        expected_size = (original_bounds[1] - original_bounds[0]) * 2
        actual_size = new_bounds[1] - new_bounds[0]

        assert np.allclose(actual_size, expected_size, atol=1e-6)

    def test_export(self, processor_with_mesh, tmp_path):
        """Test exporting mesh."""
        output_path = tmp_path / "output.stl"
        processor_with_mesh.export(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Reload and verify
        reloaded = trimesh.load(output_path)
        assert len(reloaded.vertices) == len(processor_with_mesh.mesh.vertices)

    def test_reset(self, processor_with_mesh):
        """Test resetting to original mesh."""
        original_verts = len(processor_with_mesh.mesh.vertices)

        # Modify mesh
        processor_with_mesh.scale(2.0)
        processor_with_mesh.translate([100, 100, 100])

        # Reset
        processor_with_mesh.reset()

        assert len(processor_with_mesh.mesh.vertices) == original_verts

    def test_repair(self, processor_with_mesh):
        """Test mesh repair."""
        stats = processor_with_mesh.repair()

        assert "duplicates_removed" in stats
        assert "holes_filled" in stats
        assert "normals_fixed" in stats

    def test_get_cross_section(self, processor_with_mesh):
        """Test getting cross section."""
        processor_with_mesh.center()
        processor_with_mesh.place_on_bed()

        # Get cross section at middle of cube
        paths = processor_with_mesh.get_cross_section(z_height=5.0)

        # Should get a square cross section
        assert paths is not None


class TestMeshStats:
    """Tests for MeshStats dataclass."""

    def test_str(self):
        """Test string representation."""
        stats = MeshStats(
            vertex_count=100,
            face_count=200,
            is_watertight=True,
            is_winding_consistent=True,
            bounds_min=np.array([0, 0, 0]),
            bounds_max=np.array([10, 10, 10]),
            volume=1000.0,
            surface_area=600.0,
        )

        s = str(stats)
        assert "100" in s
        assert "200" in s
        assert "1000" in s
