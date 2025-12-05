"""
Multi-Axis G-code Generator.

Extension of GCodeWriter for 4/5-axis machining support.
Handles rotary axis output (A, B, C) and machine-specific configurations.

Supports:
- LUQUE L1530 (B-axis spindle tilt)
- Generic 4-axis rotary table (A or C axis)
- Simultaneous 4-axis interpolation
- Indexed (positional) 4-axis
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

from pycam3d.chunk import CamPathChunk4Axis, AxisRotation
from pycam3d.gcode import GCodeWriter, GCodeProgram, MachineConfig


class MachineType4Axis(Enum):
    """4-axis machine configurations."""
    GENERIC_4AXIS = "generic_4axis"
    LUQUE_L1530 = "luque_l1530"       # B-axis spindle tilt
    ROTARY_TABLE_A = "rotary_table_a"  # A-axis rotary table
    ROTARY_TABLE_C = "rotary_table_c"  # C-axis rotary table
    TRUNNION = "trunnion"              # Trunnion-style A+C
    HAAS_4AXIS = "haas_4axis"          # HAAS with rotary


@dataclass
class MachineConfig4Axis(MachineConfig):
    """
    Configuration for 4-axis machines.

    Extends base MachineConfig with rotary axis settings.
    """
    # Rotary axis configuration
    rotary_axis_letter: str = 'A'      # A, B, or C
    rotary_axis_type: str = 'TABLE'    # TABLE or HEAD (spindle tilt)

    # Rotation limits
    min_rotation: float = -360.0       # degrees
    max_rotation: float = 360.0        # degrees

    # Motion capabilities
    supports_simultaneous: bool = True  # True = 4-axis interp, False = indexed
    rotation_speed: float = 10.0        # degrees/second

    # G-code format options
    rotation_in_degrees: bool = True
    rotation_decimal_places: int = 3
    use_incremental_rotation: bool = False  # G91 for rotation

    # Post-processor specific
    rotation_direction: int = 1         # 1 = CCW positive, -1 = CW positive
    wrap_rotation: bool = True          # Wrap angles to [-180, 180]
    home_rotation: float = 0.0          # Home position for rotary axis


def get_machine_config_4axis(machine_type: MachineType4Axis) -> MachineConfig4Axis:
    """
    Get predefined machine configuration.

    Args:
        machine_type: Type of 4-axis machine

    Returns:
        Configured MachineConfig4Axis
    """
    configs = {
        MachineType4Axis.GENERIC_4AXIS: MachineConfig4Axis(
            rotary_axis_letter='A',
            rotary_axis_type='TABLE',
            supports_simultaneous=True,
            max_feed_rate=5000,
            max_spindle_rpm=24000,
        ),
        MachineType4Axis.LUQUE_L1530: MachineConfig4Axis(
            rotary_axis_letter='B',
            rotary_axis_type='HEAD',
            supports_simultaneous=True,
            max_feed_rate=3000,
            max_spindle_rpm=18000,
            min_rotation=-120.0,
            max_rotation=120.0,
            rotation_speed=5.0,
        ),
        MachineType4Axis.ROTARY_TABLE_A: MachineConfig4Axis(
            rotary_axis_letter='A',
            rotary_axis_type='TABLE',
            supports_simultaneous=False,  # Indexed only
            max_feed_rate=5000,
            rotation_speed=15.0,
        ),
        MachineType4Axis.ROTARY_TABLE_C: MachineConfig4Axis(
            rotary_axis_letter='C',
            rotary_axis_type='TABLE',
            supports_simultaneous=True,
            max_feed_rate=5000,
            rotation_speed=20.0,
        ),
        MachineType4Axis.TRUNNION: MachineConfig4Axis(
            rotary_axis_letter='A',  # Primary, C is secondary
            rotary_axis_type='TABLE',
            supports_simultaneous=True,
            max_feed_rate=4000,
            min_rotation=-30.0,
            max_rotation=120.0,
        ),
        MachineType4Axis.HAAS_4AXIS: MachineConfig4Axis(
            rotary_axis_letter='A',
            rotary_axis_type='TABLE',
            supports_simultaneous=True,
            max_feed_rate=5080,  # 200 ipm
            max_spindle_rpm=12000,
            rotation_speed=100.0,  # Fast indexing
        ),
    }
    return configs.get(machine_type, configs[MachineType4Axis.GENERIC_4AXIS])


class GCodeGenerator4Axis(GCodeWriter):
    """
    G-code generator with 4-axis support.

    Extends GCodeWriter to handle rotary axes and
    multi-axis interpolation or indexed positioning.

    Example:
        generator = GCodeGenerator4Axis(
            machine_type=MachineType4Axis.LUQUE_L1530
        )
        gcode = generator.generate_from_chunks_4axis(
            chunks,
            feed_rate=1000,
            spindle_speed=12000
        )
    """

    def __init__(
        self,
        machine_type: MachineType4Axis = MachineType4Axis.GENERIC_4AXIS,
        config: Optional[MachineConfig4Axis] = None,
    ):
        """
        Initialize 4-axis G-code generator.

        Args:
            machine_type: Predefined machine type
            config: Custom configuration (overrides machine_type)
        """
        if config is None:
            config = get_machine_config_4axis(machine_type)

        super().__init__(config)
        self.config_4axis: MachineConfig4Axis = config
        self.machine_type = machine_type

        # State tracking
        self._current_rotation: Optional[float] = None

    def _format_rotation(self, rotation: AxisRotation) -> str:
        """
        Format rotation value for G-code output.

        Args:
            rotation: AxisRotation with A, B, C values

        Returns:
            Formatted string like "A45.000" or "B-30.500"
        """
        # Get the appropriate angle based on axis letter
        letter = self.config_4axis.rotary_axis_letter
        if letter == 'A':
            angle = rotation.a
        elif letter == 'B':
            angle = rotation.b
        else:  # C
            angle = rotation.c

        # Convert to degrees if needed
        if self.config_4axis.rotation_in_degrees:
            angle = np.degrees(angle)

        # Apply direction multiplier
        angle *= self.config_4axis.rotation_direction

        # Wrap angle if configured
        if self.config_4axis.wrap_rotation:
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360

        # Clamp to limits
        angle = max(self.config_4axis.min_rotation,
                   min(self.config_4axis.max_rotation, angle))

        decimals = self.config_4axis.rotation_decimal_places
        return f"{letter}{angle:.{decimals}f}"

    def _format_coord_4axis(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        rotation: Optional[AxisRotation] = None,
        feed: Optional[float] = None
    ) -> str:
        """
        Format a 4-axis coordinate string.

        Args:
            x, y, z: Linear coordinates
            rotation: Rotary axis value
            feed: Feed rate

        Returns:
            Formatted coordinate string
        """
        parts = []

        if x is not None:
            parts.append(f"X{x:.{self.config.decimal_places}f}")
        if y is not None:
            parts.append(f"Y{y:.{self.config.decimal_places}f}")
        if z is not None:
            parts.append(f"Z{z:.{self.config.decimal_places}f}")
        if rotation is not None:
            parts.append(self._format_rotation(rotation))
        if feed is not None:
            parts.append(f"F{feed:.0f}")

        return " ".join(parts)

    def _generate_header_4axis(
        self,
        program_name: str,
        spindle_speed: int
    ) -> List[str]:
        """Generate 4-axis program header."""
        lines = []

        lines.append(f"({program_name})")
        lines.append(f"(Generated by PyCAM3D - 4-Axis)")
        lines.append(f"(Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append(f"(Machine: {self.machine_type.value})")
        lines.append(f"(Rotary Axis: {self.config_4axis.rotary_axis_letter})")
        lines.append("")

        # Units and mode
        lines.append("G21 (Metric mm)")
        lines.append("G90 (Absolute positioning)")
        lines.append("G17 (XY plane)")

        # Machine-specific initialization
        if self.machine_type == MachineType4Axis.LUQUE_L1530:
            lines.append("G40 (Cancel cutter comp)")
            lines.append("G49 (Cancel tool length comp)")
            lines.append("G80 (Cancel canned cycles)")
        elif self.machine_type == MachineType4Axis.HAAS_4AXIS:
            lines.append("G40 G49 G80 (Cancel compensations)")
            lines.append("G28 G91 Z0. (Z home)")
            lines.append("G90")

        lines.append("")

        # Spindle
        lines.append(f"S{spindle_speed} M3 (Spindle on CW)")
        lines.append("G4 P2 (Dwell for spindle)")
        lines.append("")

        return lines

    def _generate_footer_4axis(self) -> List[str]:
        """Generate 4-axis program footer."""
        lines = []
        lines.append("")
        lines.append("M5 (Spindle stop)")

        # Return rotary to home
        home_rot = self.config_4axis.home_rotation
        letter = self.config_4axis.rotary_axis_letter
        lines.append(f"G0 {letter}{home_rot:.3f} (Home rotation)")

        # Return to safe position
        lines.append(f"G0 Z{self.config.safe_z:.3f} (Safe Z)")
        lines.append("G0 X0 Y0 (Home XY)")

        lines.append("")
        lines.append("M30 (Program end)")
        lines.append("%")

        return lines

    def generate_from_chunks_4axis(
        self,
        chunks: List[CamPathChunk4Axis],
        feed_rate: float = 1000,
        spindle_speed: int = 12000,
        plunge_rate: float = 300,
        safe_z: float = 10.0,
        program_name: str = "4AXIS"
    ) -> str:
        """
        Generate G-code from 4-axis chunks.

        Args:
            chunks: List of CamPathChunk4Axis
            feed_rate: Cutting feed rate mm/min
            spindle_speed: Spindle RPM
            plunge_rate: Plunge feed rate mm/min
            safe_z: Safe Z height for rapids
            program_name: G-code program name

        Returns:
            Complete G-code as string
        """
        lines = []
        lines.append("%")

        # Header
        lines.extend(self._generate_header_4axis(program_name, spindle_speed))

        # Initial safe position
        lines.append(f"G0 Z{safe_z:.3f} (Initial safe Z)")

        last_rotation: Optional[AxisRotation] = None
        total_points = sum(chunk.count() for chunk in chunks)
        point_count = 0

        for chunk_idx, chunk in enumerate(chunks):
            if chunk.count() == 0:
                continue

            lines.append("")
            lines.append(f"(=== Chunk {chunk_idx + 1}: {chunk.name or 'unnamed'} ===)")
            lines.append(f"(Points: {chunk.count()}, Layer: {chunk.layer_index})")

            # Get first point
            first_point = chunk.get_point(0)
            first_rotation = chunk.get_rotation(0)

            # Retract to safe Z
            lines.append(f"G0 Z{safe_z:.3f}")

            # Move rotation if needed (and different from current)
            if first_rotation and first_rotation != last_rotation:
                if self.config_4axis.supports_simultaneous:
                    # Can move rotation with XY
                    rot_str = self._format_rotation(first_rotation)
                    lines.append(f"G0 X{first_point[0]:.4f} Y{first_point[1]:.4f} {rot_str}")
                else:
                    # Indexed: must rotate first, then move
                    rot_str = self._format_rotation(first_rotation)
                    lines.append(f"G0 {rot_str} (Index rotation)")
                    lines.append(f"G0 X{first_point[0]:.4f} Y{first_point[1]:.4f}")
                last_rotation = first_rotation
            else:
                # Just move XY
                lines.append(f"G0 X{first_point[0]:.4f} Y{first_point[1]:.4f}")

            # Plunge to first point
            lines.append(f"G1 Z{first_point[2]:.4f} F{plunge_rate}")

            # Process remaining points
            for i in range(1, chunk.count()):
                point = chunk.get_point(i)
                rotation = chunk.get_rotation(i)

                x, y, z = point

                # Build command
                cmd_parts = ["G1"]
                cmd_parts.append(f"X{x:.4f}")
                cmd_parts.append(f"Y{y:.4f}")
                cmd_parts.append(f"Z{z:.4f}")

                # Handle rotation
                if rotation and rotation != last_rotation:
                    if self.config_4axis.supports_simultaneous:
                        # Add rotation to same move (true 4-axis)
                        cmd_parts.append(self._format_rotation(rotation))
                        cmd_parts.append(f"F{feed_rate}")
                        lines.append(" ".join(cmd_parts))
                    else:
                        # Indexed: retract, rotate, approach, continue
                        lines.append(f"G0 Z{safe_z:.3f} (Retract for index)")
                        lines.append(f"G0 {self._format_rotation(rotation)}")
                        lines.append(f"G0 X{x:.4f} Y{y:.4f}")
                        lines.append(f"G1 Z{z:.4f} F{plunge_rate}")
                    last_rotation = rotation
                else:
                    cmd_parts.append(f"F{feed_rate}")
                    lines.append(" ".join(cmd_parts))

                point_count += 1

            # Retract after chunk
            if chunk.count() > 0:
                lines.append(f"G0 Z{safe_z:.3f} (End chunk retract)")

        # Footer
        lines.extend(self._generate_footer_4axis())

        return "\n".join(lines)

    def generate_indexed(
        self,
        chunks: List[CamPathChunk4Axis],
        angles: List[float],
        feed_rate: float = 1000,
        spindle_speed: int = 12000,
        safe_z: float = 10.0
    ) -> str:
        """
        Generate indexed (positional) 4-axis G-code.

        For machines that can only position the rotary axis,
        not interpolate during cutting.

        Args:
            chunks: Chunks organized by angle
            angles: List of rotary positions (degrees)
            feed_rate: Cutting feed rate
            spindle_speed: Spindle RPM
            safe_z: Safe Z height

        Returns:
            G-code string
        """
        lines = []
        lines.append("%")
        lines.extend(self._generate_header_4axis("INDEXED", spindle_speed))

        letter = self.config_4axis.rotary_axis_letter

        for idx, (chunk, angle) in enumerate(zip(chunks, angles)):
            if chunk.count() == 0:
                continue

            lines.append("")
            lines.append(f"(=== Index Position {idx + 1}: {angle:.1f} deg ===)")

            # Safe retract
            lines.append(f"G0 Z{safe_z:.3f}")

            # Index to angle
            lines.append(f"G0 {letter}{angle:.3f}")

            # Process chunk as 3-axis
            for i, point in enumerate(chunk.points):
                x, y, z = point

                if i == 0:
                    # Rapid to first point XY
                    lines.append(f"G0 X{x:.4f} Y{y:.4f}")
                    # Plunge
                    lines.append(f"G1 Z{z:.4f} F300")
                else:
                    lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f} F{feed_rate}")

            # Retract
            lines.append(f"G0 Z{safe_z:.3f}")

        lines.extend(self._generate_footer_4axis())

        return "\n".join(lines)

    def estimate_time(
        self,
        chunks: List[CamPathChunk4Axis],
        feed_rate: float,
        rapid_rate: float = 5000
    ) -> float:
        """
        Estimate machining time for 4-axis toolpath.

        Args:
            chunks: Toolpath chunks
            feed_rate: Cutting feed rate
            rapid_rate: Rapid traverse rate

        Returns:
            Estimated time in minutes
        """
        total_time = 0.0

        for chunk in chunks:
            # Cutting time
            cutting_length = chunk.get_length()
            total_time += cutting_length / feed_rate

            # Rotation time (if indexed)
            if not self.config_4axis.supports_simultaneous:
                for i in range(1, len(chunk.rotations)):
                    r1 = chunk.rotations[i-1]
                    r2 = chunk.rotations[i]

                    # Get angle difference
                    if self.config_4axis.rotary_axis_letter == 'A':
                        delta = abs(r2.a - r1.a)
                    elif self.config_4axis.rotary_axis_letter == 'B':
                        delta = abs(r2.b - r1.b)
                    else:
                        delta = abs(r2.c - r1.c)

                    # Convert to degrees and calculate time
                    delta_deg = np.degrees(delta)
                    total_time += delta_deg / self.config_4axis.rotation_speed / 60

        return total_time


def gcode_4axis_from_chunks(
    chunks: List[CamPathChunk4Axis],
    machine: MachineType4Axis = MachineType4Axis.LUQUE_L1530,
    **kwargs
) -> str:
    """
    Convenience function to generate 4-axis G-code.

    Args:
        chunks: Toolpath chunks
        machine: Machine type
        **kwargs: Arguments passed to generate_from_chunks_4axis

    Returns:
        G-code string
    """
    generator = GCodeGenerator4Axis(machine_type=machine)
    return generator.generate_from_chunks_4axis(chunks, **kwargs)
