"""Tests for G-code generation module."""

import pytest

from pycam3d.gcode import (
    GCodeWriter,
    GCodeProgram,
    MachineConfig,
    MachineType,
    Units,
    simulate_gcode,
)
from pycam3d.toolpath import Toolpath, ToolpathPoint, Tool


@pytest.fixture
def simple_toolpath():
    """Create a simple toolpath for testing."""
    tool = Tool.ball(diameter=6.0)
    tp = Toolpath(tool=tool, safe_z=10.0, feed_rate=1000.0)

    # Rapid to start
    tp.add_point(0, 0, 10, rapid=True)
    tp.add_point(0, 0, 0, rapid=False)  # Plunge

    # Cut a square
    tp.add_point(10, 0, 0)
    tp.add_point(10, 10, 0)
    tp.add_point(0, 10, 0)
    tp.add_point(0, 0, 0)

    # Retract
    tp.add_point(0, 0, 10, rapid=True)

    return tp


class TestGCodeWriter:
    """Tests for GCodeWriter class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        writer = GCodeWriter()
        assert writer.config.units == Units.MM
        assert writer.config.machine_type == MachineType.GENERIC

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = MachineConfig(
            machine_type=MachineType.GRBL,
            units=Units.INCH,
            max_feed_rate=3000.0,
        )
        writer = GCodeWriter(config)

        assert writer.config.machine_type == MachineType.GRBL
        assert writer.config.units == Units.INCH

    def test_generate_basic(self, simple_toolpath):
        """Test basic G-code generation."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath)

        assert isinstance(gcode, GCodeProgram)
        assert len(gcode) > 0

        # Check for essential G-codes
        text = str(gcode)
        assert "G21" in text  # MM mode
        assert "G90" in text  # Absolute positioning
        assert "M3" in text  # Spindle on
        assert "M5" in text  # Spindle off
        assert "G0" in text  # Rapid move
        assert "G1" in text  # Linear move

    def test_generate_with_feed_rate(self, simple_toolpath):
        """Test feed rate is included in G-code."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath)

        text = str(gcode)
        assert "F1000" in text  # Feed rate from toolpath

    def test_generate_with_spindle(self, simple_toolpath):
        """Test spindle speed is included."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath, spindle_rpm=15000)

        text = str(gcode)
        assert "S15000" in text

    def test_generate_with_coolant(self, simple_toolpath):
        """Test coolant codes."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath, coolant=True)

        text = str(gcode)
        assert "M8" in text  # Coolant on
        assert "M9" in text  # Coolant off

    def test_generate_grbl(self, simple_toolpath):
        """Test GRBL-specific output."""
        config = MachineConfig(machine_type=MachineType.GRBL)
        writer = GCodeWriter(config)
        gcode = writer.generate(simple_toolpath)

        text = str(gcode)
        assert "$H" in text  # GRBL homing

    def test_generate_linuxcnc(self, simple_toolpath):
        """Test LinuxCNC-specific output."""
        config = MachineConfig(machine_type=MachineType.LINUXCNC)
        writer = GCodeWriter(config)
        gcode = writer.generate(simple_toolpath)

        text = str(gcode)
        assert "G64" in text  # Path blending

    def test_generate_with_line_numbers(self, simple_toolpath):
        """Test line number generation."""
        config = MachineConfig(line_numbers=True)
        writer = GCodeWriter(config)
        gcode = writer.generate(simple_toolpath)

        text = str(gcode)
        assert "N10" in text
        assert "N20" in text

    def test_generate_multi(self, simple_toolpath):
        """Test multi-toolpath generation."""
        roughing = Toolpath(tool=Tool.flat(10), feed_rate=1500)
        roughing.add_point(0, 0, 10, rapid=True)
        roughing.add_point(0, 0, 0)
        roughing.add_point(50, 50, 0)

        writer = GCodeWriter()
        gcode = writer.generate_multi([roughing, simple_toolpath])

        text = str(gcode)
        assert "TOOLPATH 1" in text
        assert "TOOLPATH 2" in text
        assert "M6" in text  # Tool change


class TestGCodeProgram:
    """Tests for GCodeProgram class."""

    def test_save(self, simple_toolpath, tmp_path):
        """Test saving G-code to file."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath)

        output = tmp_path / "test.nc"
        gcode.save(output)

        assert output.exists()
        content = output.read_text()
        assert "G21" in content

    def test_str(self):
        """Test string conversion."""
        program = GCodeProgram(lines=["G21", "G90", "G0 X0 Y0"])
        text = str(program)

        assert "G21" in text
        assert "G90" in text

    def test_len(self):
        """Test length."""
        program = GCodeProgram(lines=["G21", "G90", "G0 X0 Y0"])
        assert len(program) == 3


class TestSimulateGcode:
    """Tests for G-code simulation."""

    def test_simulate_basic(self, simple_toolpath):
        """Test basic simulation."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath)

        stats = simulate_gcode(gcode)

        assert "rapid_distance" in stats
        assert "cut_distance" in stats
        assert "estimated_time_min" in stats
        assert stats["cut_distance"] > 0

    def test_simulate_bounds(self, simple_toolpath):
        """Test bounds calculation in simulation."""
        writer = GCodeWriter()
        gcode = writer.generate(simple_toolpath)

        stats = simulate_gcode(gcode)

        # Our toolpath goes from 0,0 to 10,10
        assert stats["min_x"] <= 0
        assert stats["max_x"] >= 10
        assert stats["min_y"] <= 0
        assert stats["max_y"] >= 10


class TestMachineConfig:
    """Tests for MachineConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = MachineConfig()

        assert config.machine_type == MachineType.GENERIC
        assert config.units == Units.MM
        assert config.max_feed_rate == 5000.0
        assert config.safe_z == 10.0
        assert config.decimal_places == 3
