"""
G-code generation module.

This module handles:
- Converting toolpaths to G-code
- Machine-specific post-processors
- G-code optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, TextIO

from pycam3d.toolpath import Toolpath, ToolpathPoint

logger = logging.getLogger(__name__)


class MachineType(Enum):
    """Supported machine types / dialects."""

    GENERIC = "generic"  # Standard RS274/NGC
    LINUXCNC = "linuxcnc"
    GRBL = "grbl"
    MACH3 = "mach3"
    FANUC = "fanuc"
    HAAS = "haas"


class Units(Enum):
    """Machine units."""

    MM = "mm"
    INCH = "inch"


@dataclass
class MachineConfig:
    """
    Machine configuration for G-code generation.

    Attributes:
        machine_type: Type of machine / G-code dialect.
        units: Units (mm or inch).
        max_feed_rate: Maximum feed rate.
        max_spindle_rpm: Maximum spindle RPM.
        safe_z: Default safe Z height.
        home_position: Home position (X, Y, Z).
        decimal_places: Number of decimal places for coordinates.
        line_numbers: Whether to include line numbers.
        line_number_increment: Increment for line numbers.
    """

    machine_type: MachineType = MachineType.GENERIC
    units: Units = Units.MM
    max_feed_rate: float = 5000.0
    max_spindle_rpm: int = 24000
    safe_z: float = 10.0
    home_position: tuple[float, float, float] = (0.0, 0.0, 50.0)
    decimal_places: int = 3
    line_numbers: bool = False
    line_number_increment: int = 10


@dataclass
class GCodeProgram:
    """
    A complete G-code program.

    Attributes:
        lines: List of G-code lines.
        config: Machine configuration.
        program_name: Optional program name/number.
        comments: Header comments.
    """

    lines: list[str] = field(default_factory=list)
    config: MachineConfig = field(default_factory=MachineConfig)
    program_name: str = "PYCAM3D"
    comments: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def save(self, filepath: str | Path) -> None:
        """Save G-code to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(str(self))
            f.write("\n")

        logger.info(f"Saved G-code to {filepath} ({len(self.lines)} lines)")


class GCodeWriter:
    """
    G-code writer with support for multiple machine types.

    This class converts toolpaths to G-code with proper formatting,
    safety moves, and machine-specific dialect handling.
    """

    def __init__(self, config: Optional[MachineConfig] = None):
        """
        Initialize G-code writer.

        Args:
            config: Machine configuration (uses defaults if not provided).
        """
        self.config = config or MachineConfig()
        self._current_x: Optional[float] = None
        self._current_y: Optional[float] = None
        self._current_z: Optional[float] = None
        self._current_feed: Optional[float] = None
        self._line_number: int = 10

    def _format_coord(self, value: float) -> str:
        """Format a coordinate value."""
        return f"{value:.{self.config.decimal_places}f}"

    def _format_line(self, code: str, comment: str = "") -> str:
        """Format a G-code line with optional line number and comment."""
        parts = []

        if self.config.line_numbers:
            parts.append(f"N{self._line_number}")
            self._line_number += self.config.line_number_increment

        parts.append(code)

        if comment:
            parts.append(f"({comment})")

        return " ".join(parts)

    def _g0(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> str:
        """Generate rapid move (G0)."""
        parts = ["G0"]

        if x is not None and x != self._current_x:
            parts.append(f"X{self._format_coord(x)}")
            self._current_x = x
        if y is not None and y != self._current_y:
            parts.append(f"Y{self._format_coord(y)}")
            self._current_y = y
        if z is not None and z != self._current_z:
            parts.append(f"Z{self._format_coord(z)}")
            self._current_z = z

        if len(parts) == 1:
            return ""  # No movement needed

        return " ".join(parts)

    def _g1(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        feed: Optional[float] = None,
    ) -> str:
        """Generate linear move (G1)."""
        parts = ["G1"]

        if x is not None and x != self._current_x:
            parts.append(f"X{self._format_coord(x)}")
            self._current_x = x
        if y is not None and y != self._current_y:
            parts.append(f"Y{self._format_coord(y)}")
            self._current_y = y
        if z is not None and z != self._current_z:
            parts.append(f"Z{self._format_coord(z)}")
            self._current_z = z

        if feed is not None and feed != self._current_feed:
            parts.append(f"F{feed:.0f}")
            self._current_feed = feed

        if len(parts) == 1:
            return ""  # No movement needed

        return " ".join(parts)

    def _generate_header(self, toolpath: Toolpath, program_name: str) -> list[str]:
        """Generate G-code header."""
        lines = []

        # Program number (Fanuc/Haas style)
        if self.config.machine_type in (MachineType.FANUC, MachineType.HAAS):
            lines.append(f"O{program_name[:4].upper()}")
        else:
            lines.append(f"({program_name})")

        # Header comments
        lines.append(f"(Generated by PyCAM3D)")
        lines.append(f"(Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        if toolpath.tool:
            lines.append(f"(Tool: {toolpath.tool.name})")
            lines.append(f"(Tool Diameter: {toolpath.tool.diameter} mm)")
        if toolpath.strategy:
            lines.append(f"(Strategy: {toolpath.strategy.value})")
        lines.append("")

        # Machine setup
        if self.config.units == Units.MM:
            lines.append(self._format_line("G21", "Units: mm"))
        else:
            lines.append(self._format_line("G20", "Units: inches"))

        lines.append(self._format_line("G90", "Absolute positioning"))
        lines.append(self._format_line("G17", "XY plane"))

        # Machine-specific initialization
        if self.config.machine_type == MachineType.GRBL:
            lines.append("$H")  # Homing cycle
        elif self.config.machine_type == MachineType.LINUXCNC:
            lines.append("G64 P0.01")  # Path blending with tolerance

        lines.append("")

        return lines

    def _generate_footer(self) -> list[str]:
        """Generate G-code footer."""
        lines = []
        lines.append("")

        # Retract to safe Z
        lines.append(self._format_line(f"G0 Z{self._format_coord(self.config.safe_z)}", "Retract"))

        # Return to home
        home_x, home_y, home_z = self.config.home_position
        lines.append(self._format_line(f"G0 X{self._format_coord(home_x)} Y{self._format_coord(home_y)}", "Return home"))

        # Spindle off
        lines.append(self._format_line("M5", "Spindle off"))

        # Program end
        if self.config.machine_type in (MachineType.FANUC, MachineType.HAAS):
            lines.append("M30")
        else:
            lines.append(self._format_line("M2", "Program end"))

        lines.append("%")

        return lines

    def generate(
        self,
        toolpath: Toolpath,
        program_name: str = "PYCAM3D",
        spindle_rpm: int = 10000,
        coolant: bool = False,
    ) -> GCodeProgram:
        """
        Generate G-code from a toolpath.

        Args:
            toolpath: Toolpath to convert.
            program_name: Program name/number.
            spindle_rpm: Spindle speed in RPM.
            coolant: Whether to enable coolant.

        Returns:
            Generated G-code program.
        """
        logger.info(f"Generating G-code for {len(toolpath)} points")

        # Reset state
        self._current_x = None
        self._current_y = None
        self._current_z = None
        self._current_feed = None
        self._line_number = 10

        program = GCodeProgram(config=self.config, program_name=program_name)
        program.lines.append("%")

        # Header
        program.lines.extend(self._generate_header(toolpath, program_name))

        # Spindle on
        spindle_rpm = min(spindle_rpm, self.config.max_spindle_rpm)
        program.lines.append(self._format_line(f"S{spindle_rpm} M3", "Spindle on CW"))

        # Coolant
        if coolant:
            program.lines.append(self._format_line("M8", "Coolant on"))

        program.lines.append("")

        # Initial safe position
        safe_z_line = self._g0(z=toolpath.safe_z)
        if safe_z_line:
            program.lines.append(self._format_line(safe_z_line, "Safe Z"))

        # Generate toolpath moves
        for point in toolpath.points:
            if point.rapid:
                line = self._g0(x=point.x, y=point.y, z=point.z)
            else:
                feed = point.feed_rate or toolpath.feed_rate
                feed = min(feed, self.config.max_feed_rate)
                line = self._g1(x=point.x, y=point.y, z=point.z, feed=feed)

            if line:
                program.lines.append(line)

        # Coolant off
        if coolant:
            program.lines.append("")
            program.lines.append(self._format_line("M9", "Coolant off"))

        # Footer
        program.lines.extend(self._generate_footer())

        logger.info(f"Generated {len(program.lines)} G-code lines")

        return program

    def generate_multi(
        self,
        toolpaths: list[Toolpath],
        program_name: str = "PYCAM3D",
        spindle_rpm: int = 10000,
        coolant: bool = False,
        tool_change: bool = True,
    ) -> GCodeProgram:
        """
        Generate G-code from multiple toolpaths (e.g., roughing + finishing).

        Args:
            toolpaths: List of toolpaths.
            program_name: Program name/number.
            spindle_rpm: Spindle speed in RPM.
            coolant: Whether to enable coolant.
            tool_change: Whether to insert tool change between toolpaths.

        Returns:
            Combined G-code program.
        """
        if not toolpaths:
            raise ValueError("No toolpaths provided")

        logger.info(f"Generating G-code for {len(toolpaths)} toolpaths")

        # Reset state
        self._current_x = None
        self._current_y = None
        self._current_z = None
        self._current_feed = None
        self._line_number = 10

        program = GCodeProgram(config=self.config, program_name=program_name)
        program.lines.append("%")

        # Header
        program.lines.extend(self._generate_header(toolpaths[0], program_name))

        for idx, toolpath in enumerate(toolpaths):
            program.lines.append("")
            program.lines.append(f"(===== TOOLPATH {idx + 1} =====)")

            if toolpath.tool:
                program.lines.append(f"(Tool: {toolpath.tool.name})")

            # Tool change
            if tool_change and idx > 0:
                program.lines.append(self._format_line("M5", "Spindle off"))
                program.lines.append(self._format_line(f"G0 Z{self._format_coord(self.config.safe_z)}", "Retract"))
                program.lines.append(self._format_line("M6", "Tool change"))
                program.lines.append(self._format_line(f"S{spindle_rpm} M3", "Spindle on"))
                if coolant:
                    program.lines.append(self._format_line("M8", "Coolant on"))
            elif idx == 0:
                # First toolpath - spindle on
                spindle_rpm = min(spindle_rpm, self.config.max_spindle_rpm)
                program.lines.append(self._format_line(f"S{spindle_rpm} M3", "Spindle on CW"))
                if coolant:
                    program.lines.append(self._format_line("M8", "Coolant on"))

            program.lines.append("")

            # Safe Z
            safe_z_line = self._g0(z=toolpath.safe_z)
            if safe_z_line:
                program.lines.append(safe_z_line)

            # Toolpath moves
            for point in toolpath.points:
                if point.rapid:
                    line = self._g0(x=point.x, y=point.y, z=point.z)
                else:
                    feed = point.feed_rate or toolpath.feed_rate
                    feed = min(feed, self.config.max_feed_rate)
                    line = self._g1(x=point.x, y=point.y, z=point.z, feed=feed)

                if line:
                    program.lines.append(line)

        # Coolant off
        if coolant:
            program.lines.append("")
            program.lines.append(self._format_line("M9", "Coolant off"))

        # Footer
        program.lines.extend(self._generate_footer())

        logger.info(f"Generated {len(program.lines)} G-code lines")

        return program


def simulate_gcode(gcode: GCodeProgram) -> dict:
    """
    Simple G-code simulation for validation.

    Returns estimated machining statistics.

    Args:
        gcode: G-code program to simulate.

    Returns:
        Dictionary with simulation results.
    """
    import re

    stats = {
        "rapid_distance": 0.0,
        "cut_distance": 0.0,
        "estimated_time_min": 0.0,
        "min_x": float("inf"),
        "max_x": float("-inf"),
        "min_y": float("inf"),
        "max_y": float("-inf"),
        "min_z": float("inf"),
        "max_z": float("-inf"),
    }

    current_x, current_y, current_z = 0.0, 0.0, 0.0
    current_feed = 1000.0
    rapid_feed = 5000.0

    coord_pattern = re.compile(r"([XYZF])(-?\d+\.?\d*)")

    for line in gcode.lines:
        line = line.split("(")[0].strip()  # Remove comments
        if not line:
            continue

        is_rapid = "G0" in line or "G00" in line
        is_cut = "G1" in line or "G01" in line

        if not (is_rapid or is_cut):
            continue

        # Parse coordinates
        new_x, new_y, new_z = current_x, current_y, current_z

        for match in coord_pattern.finditer(line):
            axis, value = match.groups()
            value = float(value)

            if axis == "X":
                new_x = value
            elif axis == "Y":
                new_y = value
            elif axis == "Z":
                new_z = value
            elif axis == "F":
                current_feed = value

        # Calculate distance
        dist = ((new_x - current_x) ** 2 + (new_y - current_y) ** 2 + (new_z - current_z) ** 2) ** 0.5

        if is_rapid:
            stats["rapid_distance"] += dist
            stats["estimated_time_min"] += dist / rapid_feed
        else:
            stats["cut_distance"] += dist
            stats["estimated_time_min"] += dist / current_feed

        # Update bounds
        stats["min_x"] = min(stats["min_x"], new_x)
        stats["max_x"] = max(stats["max_x"], new_x)
        stats["min_y"] = min(stats["min_y"], new_y)
        stats["max_y"] = max(stats["max_y"], new_y)
        stats["min_z"] = min(stats["min_z"], new_z)
        stats["max_z"] = max(stats["max_z"], new_z)

        current_x, current_y, current_z = new_x, new_y, new_z

    # Handle inf values
    for key in ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]:
        if stats[key] in (float("inf"), float("-inf")):
            stats[key] = 0.0

    return stats
