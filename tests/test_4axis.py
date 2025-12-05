"""
Tests for 4-axis machining module.

Tests the complete pipeline from pattern generation
through surface sampling to G-code output.
"""

import pytest
import numpy as np
from math import pi

# Test imports
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from pycam3d.chunk import AxisRotation, CamPathChunk4Axis, merge_chunks, sort_chunks_by_distance
from pycam3d.patterns.rotary import RotaryPatternGenerator, RotaryStrategy
from pycam3d.toolpath import Tool, ToolType


class TestAxisRotation:
    """Test AxisRotation dataclass."""

    def test_create_default(self):
        """Test default rotation is zero."""
        rot = AxisRotation()
        assert rot.a == 0.0
        assert rot.b == 0.0
        assert rot.c == 0.0

    def test_create_with_values(self):
        """Test creating rotation with values."""
        rot = AxisRotation(a=0.5, b=1.0, c=1.5)
        assert rot.a == 0.5
        assert rot.b == 1.0
        assert rot.c == 1.5

    def test_to_tuple(self):
        """Test conversion to tuple."""
        rot = AxisRotation(a=1.0, b=2.0, c=3.0)
        assert rot.to_tuple() == (1.0, 2.0, 3.0)

    def test_to_degrees(self):
        """Test conversion to degrees."""
        rot = AxisRotation(a=pi, b=pi/2, c=pi/4)
        deg = rot.to_degrees()
        assert abs(deg.a - 180.0) < 0.01
        assert abs(deg.b - 90.0) < 0.01
        assert abs(deg.c - 45.0) < 0.01

    def test_copy(self):
        """Test deep copy."""
        rot1 = AxisRotation(a=1.0, b=2.0, c=3.0)
        rot2 = rot1.copy()
        rot2.a = 5.0
        assert rot1.a == 1.0  # Original unchanged

    def test_equality(self):
        """Test equality comparison."""
        rot1 = AxisRotation(a=1.0, b=2.0, c=3.0)
        rot2 = AxisRotation(a=1.0, b=2.0, c=3.0)
        rot3 = AxisRotation(a=1.0, b=2.0, c=4.0)
        assert rot1 == rot2
        assert rot1 != rot3

    def test_interpolate(self):
        """Test linear interpolation."""
        rot1 = AxisRotation(a=0.0, b=0.0, c=0.0)
        rot2 = AxisRotation(a=1.0, b=2.0, c=4.0)
        mid = rot1.interpolate(rot2, 0.5)
        assert abs(mid.a - 0.5) < 0.001
        assert abs(mid.b - 1.0) < 0.001
        assert abs(mid.c - 2.0) < 0.001

    def test_from_euler(self):
        """Test creation from Euler angles."""
        rot = AxisRotation.from_euler((90, 45, 30), degrees=True)
        assert abs(rot.a - np.radians(90)) < 0.001
        assert abs(rot.b - np.radians(45)) < 0.001
        assert abs(rot.c - np.radians(30)) < 0.001


class TestCamPathChunk4Axis:
    """Test CamPathChunk4Axis dataclass."""

    def test_create_empty(self):
        """Test creating empty chunk."""
        chunk = CamPathChunk4Axis()
        assert chunk.count() == 0
        assert len(chunk) == 0

    def test_add_point_simple(self):
        """Test adding a simple point."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((1.0, 2.0, 3.0))
        assert chunk.count() == 1
        assert chunk.get_point(0) == (1.0, 2.0, 3.0)

    def test_add_point_with_rotation(self):
        """Test adding point with rotation."""
        chunk = CamPathChunk4Axis()
        rot = AxisRotation(a=0.5)
        chunk.add_point(
            point=(1.0, 2.0, 3.0),
            startpoint=(1.0, 2.0, 10.0),
            endpoint=(1.0, 2.0, 0.0),
            rotation=rot
        )
        assert chunk.count() == 1
        assert len(chunk.rotations) == 1
        assert chunk.get_rotation(0) == rot

    def test_reverse(self):
        """Test reversing chunk."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((0, 0, 0))
        chunk.add_point((1, 1, 1))
        chunk.add_point((2, 2, 2))
        chunk.reverse()
        assert chunk.get_point(0) == (2, 2, 2)
        assert chunk.get_point(2) == (0, 0, 0)

    def test_shift(self):
        """Test translating chunk."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((0, 0, 0), startpoint=(0, 0, 5), endpoint=(0, 0, -5))
        chunk.shift(10, 20, 30)
        assert chunk.get_point(0) == (10, 20, 30)
        assert chunk.startpoints[0] == (10, 20, 35)

    def test_copy(self):
        """Test deep copy of chunk."""
        chunk1 = CamPathChunk4Axis(name="test")
        chunk1.add_point((1, 2, 3), rotation=AxisRotation(a=0.5))
        chunk2 = chunk1.copy()
        chunk2.points[0] = (5, 5, 5)
        assert chunk1.get_point(0) == (1, 2, 3)  # Original unchanged

    def test_get_bounds(self):
        """Test bounding box calculation."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((0, 0, 0))
        chunk.add_point((10, 5, 3))
        chunk.add_point((-5, 10, -2))
        min_b, max_b = chunk.get_bounds()
        np.testing.assert_array_almost_equal(min_b, [-5, 0, -2])
        np.testing.assert_array_almost_equal(max_b, [10, 10, 3])

    def test_get_length(self):
        """Test path length calculation."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((0, 0, 0))
        chunk.add_point((10, 0, 0))
        chunk.add_point((10, 10, 0))
        length = chunk.get_length()
        assert abs(length - 20.0) < 0.001

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((1, 2, 3))
        chunk.add_point((4, 5, 6))
        arr = chunk.to_numpy()
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr[0], [1, 2, 3])

    def test_from_numpy(self):
        """Test creation from numpy array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        chunk = CamPathChunk4Axis.from_numpy(arr, name="test")
        assert chunk.count() == 2
        assert chunk.name == "test"

    def test_iterate(self):
        """Test iterating over chunk points."""
        chunk = CamPathChunk4Axis()
        chunk.add_point((0, 0, 0))
        chunk.add_point((1, 1, 1))
        points = list(chunk)
        assert len(points) == 2

    def test_merge(self):
        """Test merging two chunks."""
        chunk1 = CamPathChunk4Axis()
        chunk1.add_point((0, 0, 0))
        chunk2 = CamPathChunk4Axis()
        chunk2.add_point((1, 1, 1))
        chunk1.merge(chunk2)
        assert chunk1.count() == 2


class TestMergeChunks:
    """Test chunk merging utilities."""

    def test_merge_close_chunks(self):
        """Test merging chunks that are close together."""
        chunk1 = CamPathChunk4Axis()
        chunk1.add_point((0, 0, 0))
        chunk1.add_point((1, 0, 0))

        chunk2 = CamPathChunk4Axis()
        chunk2.add_point((1.05, 0, 0))  # Close to end of chunk1
        chunk2.add_point((2, 0, 0))

        merged = merge_chunks([chunk1, chunk2], tolerance=0.1)
        assert len(merged) == 1
        assert merged[0].count() == 4

    def test_dont_merge_distant_chunks(self):
        """Test that distant chunks are not merged."""
        chunk1 = CamPathChunk4Axis()
        chunk1.add_point((0, 0, 0))

        chunk2 = CamPathChunk4Axis()
        chunk2.add_point((10, 0, 0))  # Far from chunk1

        merged = merge_chunks([chunk1, chunk2], tolerance=0.1)
        assert len(merged) == 2


class TestSortChunksByDistance:
    """Test chunk sorting for minimal travel."""

    def test_sort_nearest_first(self):
        """Test that chunks are sorted by distance."""
        chunk1 = CamPathChunk4Axis()
        chunk1.add_point((10, 0, 0))

        chunk2 = CamPathChunk4Axis()
        chunk2.add_point((1, 0, 0))  # Closer to origin

        chunk3 = CamPathChunk4Axis()
        chunk3.add_point((5, 0, 0))

        sorted_chunks = sort_chunks_by_distance([chunk1, chunk2, chunk3], start=(0, 0, 0))

        # chunk2 should be first (closest to origin)
        assert sorted_chunks[0].get_point(0) == (1, 0, 0)


class TestRotaryPatternGenerator:
    """Test rotary pattern generation."""

    def test_create_generator(self):
        """Test creating a pattern generator."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, -50, -50),
            max_bounds=(100, 50, 50),
            distance_between_paths=5.0,
            distance_along_paths=1.0
        )
        assert gen.rotary_axis == 'X'
        assert gen.path_step == 5.0

    def test_generate_parallel_around(self):
        """Test PARALLELR pattern generation."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(50, 0, 0),
            distance_between_paths=10.0,
            distance_along_paths=1.0
        )
        chunks = gen.generate_parallel_around(radius=25.0)

        # Should have multiple chunks (one per axial position)
        assert len(chunks) > 0

        # Each chunk should have rotations
        for chunk in chunks:
            assert len(chunk.rotations) == chunk.count()

    def test_generate_parallel_along(self):
        """Test PARALLEL pattern generation."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(50, 0, 0),
            distance_between_paths=10.0,
            distance_along_paths=5.0
        )
        chunks = gen.generate_parallel_along(radius=25.0)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.count() > 0

    def test_generate_helix(self):
        """Test HELIX pattern generation."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(50, 0, 0),
            distance_between_paths=5.0,
            distance_along_paths=1.0
        )
        chunks = gen.generate_helix(radius=25.0)

        # Helix should produce single chunk
        assert len(chunks) == 1
        assert chunks[0].count() > 10  # Should have many points

    def test_generate_cross(self):
        """Test CROSS pattern generation."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(50, 0, 0),
            distance_between_paths=10.0,
            distance_along_paths=5.0
        )
        chunks = gen.generate_cross(radius=25.0)

        # Cross should produce chunks from both directions
        assert len(chunks) > 0

    def test_generate_with_strategy_enum(self):
        """Test using RotaryStrategy enum."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(50, 0, 0),
            distance_between_paths=10.0,
            distance_along_paths=5.0
        )
        chunks = gen.generate(RotaryStrategy.PARALLELR, radius=25.0)
        assert len(chunks) > 0

    def test_y_axis_rotation(self):
        """Test rotation around Y axis."""
        gen = RotaryPatternGenerator(
            rotary_axis='Y',
            min_bounds=(0, 0, 0),
            max_bounds=(0, 50, 0),
            distance_between_paths=10.0,
            distance_along_paths=5.0
        )
        chunks = gen.generate_parallel_around(radius=25.0)

        # Should have B rotations
        for chunk in chunks:
            if chunk.rotations:
                rot = chunk.rotations[0]
                # A and C should be zero, B should vary
                assert rot.a == 0.0
                assert rot.c == 0.0

    def test_conical_part(self):
        """Test pattern for conical part (different start/end radii)."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(100, 0, 0),
            distance_between_paths=10.0,
            distance_along_paths=5.0
        )
        chunks = gen.generate_parallel_around(radius=30.0, radius_end=15.0)

        # Check that depth is set
        assert chunks[0].depth == 15.0  # abs(30 - 15)

    def test_partial_angle(self):
        """Test partial rotation (not full 360)."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, 0, 0),
            max_bounds=(50, 0, 0),
            distance_between_paths=10.0,
            distance_along_paths=5.0
        )
        chunks = gen.generate_parallel_around(
            radius=25.0,
            angle_start=0.0,
            angle_end=pi  # Only 180 degrees
        )

        # Check that chunks are not marked as closed
        assert not chunks[0].closed

    def test_estimate_radius(self):
        """Test radius estimation from bounds."""
        gen = RotaryPatternGenerator(
            rotary_axis='X',
            min_bounds=(0, -25, -25),
            max_bounds=(100, 25, 25),
        )
        radius = gen.estimate_radius_from_bounds()
        assert radius == 25.0


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not available")
class TestNAxisSampler:
    """Test surface sampling (requires trimesh)."""

    @pytest.fixture
    def cylinder_mesh(self):
        """Create a simple cylinder mesh for testing."""
        return trimesh.creation.cylinder(radius=25, height=100)

    @pytest.fixture
    def simple_tool(self):
        """Create a simple ball tool."""
        return Tool.ball(diameter=6.0)

    def test_create_sampler(self, cylinder_mesh, simple_tool):
        """Test creating a sampler."""
        from pycam3d.collision.sampler_4axis import NAxisSampler

        sampler = NAxisSampler(cylinder_mesh, simple_tool)
        assert sampler.mesh is cylinder_mesh
        assert sampler.tool is simple_tool

    def test_sample_single_point(self, cylinder_mesh, simple_tool):
        """Test sampling a single point."""
        from pycam3d.collision.sampler_4axis import NAxisSampler

        sampler = NAxisSampler(cylinder_mesh, simple_tool)

        # Ray from outside pointing toward center
        start = np.array([0, 0, 50])  # Outside
        end = np.array([0, 0, 0])     # Center
        rot = AxisRotation()

        result = sampler.sample_point(start, end, rot)

        # Should hit the cylinder surface
        assert result is not None
        # Should be around radius 25 from center
        dist_from_axis = np.sqrt(result[0]**2 + result[1]**2)
        assert abs(dist_from_axis - 25) < 5  # Within tolerance

    def test_sample_chunks(self, cylinder_mesh, simple_tool):
        """Test sampling multiple chunks."""
        from pycam3d.collision.sampler_4axis import NAxisSampler

        sampler = NAxisSampler(cylinder_mesh, simple_tool)

        # Create a simple pattern chunk
        chunk = CamPathChunk4Axis()
        for i in range(5):
            z = i * 10
            chunk.add_point(
                point=(50, 0, z),
                startpoint=(50, 0, z),
                endpoint=(0, 0, z),
                rotation=AxisRotation()
            )

        result = sampler.sample_chunks([chunk])

        assert result.total_points == 5
        assert result.sampled_points > 0
        assert len(result.chunks) > 0


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not available")
class TestGCodeGenerator4Axis:
    """Test 4-axis G-code generation."""

    def test_create_generator(self):
        """Test creating 4-axis G-code generator."""
        from pycam3d.gcode_multiaxis import GCodeGenerator4Axis, MachineType4Axis

        gen = GCodeGenerator4Axis(machine_type=MachineType4Axis.LUQUE_L1530)
        assert gen.machine_type == MachineType4Axis.LUQUE_L1530
        assert gen.config_4axis.rotary_axis_letter == 'B'

    def test_generate_simple_gcode(self):
        """Test generating simple 4-axis G-code."""
        from pycam3d.gcode_multiaxis import GCodeGenerator4Axis, MachineType4Axis

        gen = GCodeGenerator4Axis(machine_type=MachineType4Axis.GENERIC_4AXIS)

        # Create simple chunk with rotation
        chunk = CamPathChunk4Axis(name="test")
        chunk.add_point((0, 0, 0), rotation=AxisRotation(a=0))
        chunk.add_point((10, 0, 0), rotation=AxisRotation(a=pi/4))
        chunk.add_point((20, 0, 0), rotation=AxisRotation(a=pi/2))

        gcode = gen.generate_from_chunks_4axis(
            [chunk],
            feed_rate=1000,
            spindle_speed=12000
        )

        # Check basic structure
        assert "G21" in gcode  # Metric
        assert "G90" in gcode  # Absolute
        assert "S12000" in gcode  # Spindle speed
        assert "A" in gcode  # Has A-axis commands
        assert "M30" in gcode  # Program end

    def test_luque_l1530_config(self):
        """Test LUQUE L1530 specific configuration."""
        from pycam3d.gcode_multiaxis import GCodeGenerator4Axis, MachineType4Axis

        gen = GCodeGenerator4Axis(machine_type=MachineType4Axis.LUQUE_L1530)

        chunk = CamPathChunk4Axis()
        chunk.add_point((0, 0, 0), rotation=AxisRotation(b=0))
        chunk.add_point((10, 0, 0), rotation=AxisRotation(b=pi/6))

        gcode = gen.generate_from_chunks_4axis([chunk], feed_rate=1000)

        # LUQUE uses B-axis
        assert "B" in gcode
        assert "luque_l1530" in gcode.lower()

    def test_indexed_gcode(self):
        """Test indexed (non-simultaneous) G-code."""
        from pycam3d.gcode_multiaxis import GCodeGenerator4Axis, MachineType4Axis

        gen = GCodeGenerator4Axis(machine_type=MachineType4Axis.ROTARY_TABLE_A)

        # Two chunks at different angles
        chunk1 = CamPathChunk4Axis()
        chunk1.add_point((0, 0, 0))
        chunk1.add_point((10, 0, 0))

        chunk2 = CamPathChunk4Axis()
        chunk2.add_point((0, 0, 0))
        chunk2.add_point((10, 0, 0))

        gcode = gen.generate_indexed([chunk1, chunk2], [0.0, 90.0])

        # Should have indexed positioning comments
        assert "Index Position" in gcode


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not available")
class TestPipeline4Axis:
    """Test complete 4-axis pipeline."""

    @pytest.fixture
    def cylinder_mesh_file(self, tmp_path):
        """Create a cylinder mesh file for testing."""
        mesh = trimesh.creation.cylinder(radius=25, height=100)
        filepath = tmp_path / "cylinder.stl"
        mesh.export(filepath)
        return filepath

    def test_create_pipeline(self, cylinder_mesh_file):
        """Test creating pipeline from mesh file."""
        from pycam3d.pipeline_4axis import Pipeline4Axis
        from pycam3d.gcode_multiaxis import MachineType4Axis

        tool = Tool.ball(diameter=6.0)
        pipeline = Pipeline4Axis(
            str(cylinder_mesh_file),
            tool,
            machine_type=MachineType4Axis.LUQUE_L1530
        )

        assert pipeline.tool == tool

    def test_generate_parallelr(self, cylinder_mesh_file):
        """Test generating PARALLELR toolpath."""
        from pycam3d.pipeline_4axis import Pipeline4Axis

        tool = Tool.ball(diameter=6.0)
        pipeline = Pipeline4Axis(str(cylinder_mesh_file), tool)

        result = pipeline.generate(
            strategy='PARALLELR',
            rotary_axis='Z',  # Cylinder axis
            stepover=5.0,
            feed_rate=1000,
            spindle_speed=12000
        )

        assert result.gcode != ""
        assert result.total_points > 0
        assert len(result.chunks) > 0

    def test_generate_helix(self, cylinder_mesh_file):
        """Test generating HELIX toolpath."""
        from pycam3d.pipeline_4axis import Pipeline4Axis

        tool = Tool.ball(diameter=6.0)
        pipeline = Pipeline4Axis(str(cylinder_mesh_file), tool)

        result = pipeline.generate(
            strategy='HELIX',
            rotary_axis='Z',
            stepover=5.0,
            feed_rate=1000
        )

        assert result.gcode != ""
        assert result.total_points > 0

    def test_save_gcode(self, cylinder_mesh_file, tmp_path):
        """Test saving G-code to file."""
        from pycam3d.pipeline_4axis import Pipeline4Axis

        tool = Tool.ball(diameter=6.0)
        pipeline = Pipeline4Axis(str(cylinder_mesh_file), tool)

        result = pipeline.generate(
            strategy='PARALLELR',
            rotary_axis='Z',
            stepover=10.0,
            feed_rate=1000
        )

        output_path = tmp_path / "output.nc"
        pipeline.save_gcode(result, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "G21" in content


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not available")
class TestQuick4Axis:
    """Test quick_4axis convenience function."""

    def test_quick_4axis(self, tmp_path):
        """Test quick 4-axis generation."""
        from pycam3d.pipeline_4axis import quick_4axis

        # Create test mesh
        mesh = trimesh.creation.cylinder(radius=25, height=100)
        mesh_path = tmp_path / "test.stl"
        mesh.export(mesh_path)

        output_path = tmp_path / "output.nc"

        result = quick_4axis(
            str(mesh_path),
            str(output_path),
            tool_diameter=6.0,
            tool_type='ball',
            strategy='PARALLELR',
            machine='GENERIC_4AXIS',
            stepover=10.0,
            feed_rate=1000
        )

        assert output_path.exists()
        assert result.total_points > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
