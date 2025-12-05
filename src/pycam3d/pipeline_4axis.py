"""
4-Axis Machining Pipeline.

Complete pipeline for 4-axis CNC toolpath generation:
1. Pattern generation (PARALLEL, PARALLELR, HELIX)
2. Surface sampling
3. G-code generation

This integrates all 4-axis components into a simple API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple, Callable, Union
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("trimesh is required. Install with: pip install trimesh")

from pycam3d.chunk import CamPathChunk4Axis, AxisRotation, sort_chunks_by_distance
from pycam3d.patterns.rotary import RotaryPatternGenerator, RotaryStrategy
from pycam3d.collision.sampler_4axis import NAxisSampler, SamplingResult
from pycam3d.gcode_multiaxis import (
    GCodeGenerator4Axis,
    MachineType4Axis,
    MachineConfig4Axis,
    get_machine_config_4axis
)
from pycam3d.toolpath import Tool, ToolType

logger = logging.getLogger(__name__)


@dataclass
class Pipeline4AxisResult:
    """Result of 4-axis pipeline execution."""
    gcode: str = ""
    chunks: List[CamPathChunk4Axis] = field(default_factory=list)
    sampling_result: Optional[SamplingResult] = None

    # Statistics
    total_points: int = 0
    estimated_time_min: float = 0.0
    path_length_mm: float = 0.0

    # Bounds
    bounds_min: Tuple[float, float, float] = (0, 0, 0)
    bounds_max: Tuple[float, float, float] = (0, 0, 0)


class Pipeline4Axis:
    """
    Complete pipeline for 4-axis CNC machining.

    Integrates pattern generation, surface sampling, and G-code output
    into a single easy-to-use interface.

    Example:
        # Load mesh and create tool
        mesh = trimesh.load("part.stl")
        tool = Tool.ball(diameter=6.0)

        # Create and run pipeline
        pipeline = Pipeline4Axis(mesh, tool)
        result = pipeline.generate(
            strategy='PARALLELR',
            rotary_axis='X',
            stepover=2.0,
            feed_rate=1000
        )

        # Save G-code
        with open("output.nc", "w") as f:
            f.write(result.gcode)
    """

    def __init__(
        self,
        mesh: Union[trimesh.Trimesh, str, Path],
        tool: Tool,
        machine_type: MachineType4Axis = MachineType4Axis.LUQUE_L1530,
        machine_config: Optional[MachineConfig4Axis] = None
    ):
        """
        Initialize 4-axis pipeline.

        Args:
            mesh: Trimesh object or path to mesh file
            tool: Tool definition
            machine_type: Type of 4-axis machine
            machine_config: Custom machine configuration (optional)
        """
        # Load mesh if path provided
        if isinstance(mesh, (str, Path)):
            logger.info(f"Loading mesh from {mesh}")
            self.mesh = trimesh.load(mesh)
        else:
            self.mesh = mesh

        self.tool = tool
        self.machine_type = machine_type
        self.machine_config = machine_config or get_machine_config_4axis(machine_type)

        # Compute mesh properties
        self.bounds = self.mesh.bounds
        self.bounds_min = self.bounds[0]
        self.bounds_max = self.bounds[1]
        self.center = (self.bounds_min + self.bounds_max) / 2
        self.dimensions = self.bounds_max - self.bounds_min

        # Estimate radius for rotary operations
        self._compute_rotary_properties()

        logger.info(
            f"Pipeline4Axis initialized: "
            f"mesh {len(self.mesh.vertices)} verts, "
            f"tool {tool.type.value} D{tool.diameter}, "
            f"machine {machine_type.value}"
        )

    def _compute_rotary_properties(self) -> None:
        """Compute properties needed for rotary machining."""
        # Default radius from bounding box
        # Assumes part is centered on rotary axis
        self.radius = max(self.dimensions[1], self.dimensions[2]) / 2
        self.radius_end = self.radius  # Can be overridden for conical parts

        # Axis length (along rotary axis)
        self.axis_length = self.dimensions[0]  # Default X-axis

    def set_conical(self, radius_start: float, radius_end: float) -> None:
        """
        Configure for conical/tapered parts.

        Args:
            radius_start: Radius at start of rotary axis
            radius_end: Radius at end of rotary axis
        """
        self.radius = radius_start
        self.radius_end = radius_end

    def generate(
        self,
        strategy: Literal['PARALLELR', 'PARALLEL', 'HELIX', 'CROSS'] = 'PARALLELR',
        rotary_axis: Literal['X', 'Y', 'Z'] = 'X',
        stepover: float = 1.0,
        stepdown: float = 2.0,
        feed_rate: float = 1000,
        plunge_rate: float = 300,
        spindle_speed: int = 12000,
        num_layers: int = 1,
        safe_z: float = 10.0,
        angle_start: float = 0.0,
        angle_end: float = 360.0,
        optimize_path: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Pipeline4AxisResult:
        """
        Generate complete 4-axis G-code.

        Args:
            strategy: Machining strategy
                - PARALLELR: Passes around the rotary axis (circular)
                - PARALLEL: Passes along the rotary axis (linear)
                - HELIX: Continuous helical path
                - CROSS: Both parallel directions
            rotary_axis: Axis of rotation ('X', 'Y', 'Z')
            stepover: Distance between passes (mm)
            stepdown: Depth per layer (mm)
            feed_rate: Cutting feed rate (mm/min)
            plunge_rate: Plunge feed rate (mm/min)
            spindle_speed: Spindle speed (RPM)
            num_layers: Number of depth layers
            safe_z: Safe Z height for rapids
            angle_start: Starting angle (degrees)
            angle_end: Ending angle (degrees)
            optimize_path: Whether to optimize chunk order
            progress_callback: Optional callback(percent, message)

        Returns:
            Pipeline4AxisResult with G-code and statistics
        """
        result = Pipeline4AxisResult()

        def progress(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logger.debug(f"Progress {pct:.0f}%: {msg}")

        progress(0, "Starting 4-axis pipeline")

        # Convert strategy string to enum
        strategy_enum = RotaryStrategy[strategy]

        # Convert angles to radians
        angle_start_rad = np.radians(angle_start)
        angle_end_rad = np.radians(angle_end)

        # ========== PHASE 1: Pattern Generation ==========
        progress(5, "Generating toolpath pattern")

        pattern_gen = RotaryPatternGenerator(
            rotary_axis=rotary_axis,
            min_bounds=tuple(self.bounds_min),
            max_bounds=tuple(self.bounds_max),
            distance_between_paths=stepover,
            distance_along_paths=self.tool.diameter * 0.3  # 30% of tool for sampling
        )

        if strategy_enum == RotaryStrategy.PARALLELR:
            pattern_chunks = pattern_gen.generate_parallel_around(
                self.radius, self.radius_end,
                angle_start=angle_start_rad,
                angle_end=angle_end_rad
            )
        elif strategy_enum == RotaryStrategy.PARALLEL:
            pattern_chunks = pattern_gen.generate_parallel_along(
                self.radius, self.radius_end,
                angle_start=angle_start_rad,
                angle_end=angle_end_rad
            )
        elif strategy_enum == RotaryStrategy.HELIX:
            pattern_chunks = pattern_gen.generate_helix(
                self.radius, self.radius_end,
                angle_start=angle_start_rad
            )
        elif strategy_enum == RotaryStrategy.CROSS:
            pattern_chunks = pattern_gen.generate_cross(
                self.radius, self.radius_end
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Generated {len(pattern_chunks)} pattern chunks")
        progress(25, f"Pattern generated: {len(pattern_chunks)} chunks")

        # ========== PHASE 2: Surface Sampling ==========
        progress(30, "Sampling surface")

        # Calculate layers for depth of cut
        total_depth = abs(self.radius - self.radius_end) if self.radius != self.radius_end else stepdown
        if total_depth < 0.001:
            total_depth = stepdown

        layers = []
        current = 0.0
        for i in range(num_layers):
            layer_end = min(current + stepdown, total_depth)
            layers.append((current, layer_end))
            current = layer_end
            if current >= total_depth:
                break

        logger.info(f"Using {len(layers)} depth layers")

        # Create sampler and sample
        sampler = NAxisSampler(self.mesh, self.tool)

        def sampling_progress(pct):
            progress(30 + pct * 0.5, "Sampling surface")

        sampling_result = sampler.sample_chunks(
            pattern_chunks,
            layers,
            progress_callback=sampling_progress
        )

        result.sampling_result = sampling_result
        result.chunks = sampling_result.chunks

        logger.info(
            f"Sampling complete: {sampling_result.sampled_points}/{sampling_result.total_points} "
            f"points in {sampling_result.sampling_time:.2f}s"
        )
        progress(80, f"Sampled {sampling_result.sampled_points} points")

        # ========== PHASE 3: Path Optimization ==========
        if optimize_path and len(result.chunks) > 1:
            progress(82, "Optimizing path order")
            result.chunks = sort_chunks_by_distance(result.chunks)

        # ========== PHASE 4: G-code Generation ==========
        progress(85, "Generating G-code")

        gcode_gen = GCodeGenerator4Axis(
            machine_type=self.machine_type,
            config=self.machine_config
        )

        result.gcode = gcode_gen.generate_from_chunks_4axis(
            result.chunks,
            feed_rate=feed_rate,
            spindle_speed=spindle_speed,
            plunge_rate=plunge_rate,
            safe_z=safe_z,
            program_name=f"4AXIS_{strategy}"
        )

        # ========== PHASE 5: Statistics ==========
        progress(95, "Calculating statistics")

        result.total_points = sum(c.count() for c in result.chunks)
        result.path_length_mm = sum(c.get_length() for c in result.chunks)
        result.estimated_time_min = gcode_gen.estimate_time(
            result.chunks, feed_rate
        )

        # Calculate bounds from sampled points
        all_points = []
        for chunk in result.chunks:
            all_points.extend(chunk.points)

        if all_points:
            points_arr = np.array(all_points)
            result.bounds_min = tuple(points_arr.min(axis=0))
            result.bounds_max = tuple(points_arr.max(axis=0))

        logger.info(
            f"Pipeline complete: {result.total_points} points, "
            f"{result.path_length_mm:.1f}mm path, "
            f"~{result.estimated_time_min:.1f} min"
        )
        progress(100, "Complete!")

        return result

    def generate_roughing(
        self,
        rotary_axis: Literal['X', 'Y', 'Z'] = 'X',
        stepover: float = None,
        stepdown: float = None,
        stock_to_leave: float = 0.5,
        **kwargs
    ) -> Pipeline4AxisResult:
        """
        Generate roughing toolpath.

        Uses aggressive parameters for material removal.

        Args:
            rotary_axis: Rotation axis
            stepover: Stepover (default: 50% of tool diameter)
            stepdown: Depth per pass (default: tool diameter)
            stock_to_leave: Material to leave for finishing
            **kwargs: Additional arguments for generate()

        Returns:
            Pipeline4AxisResult
        """
        if stepover is None:
            stepover = self.tool.diameter * 0.5
        if stepdown is None:
            stepdown = self.tool.diameter

        # Temporarily increase radius for stock to leave
        original_radius = self.radius
        self.radius += stock_to_leave

        result = self.generate(
            strategy='HELIX',  # Fast roughing
            rotary_axis=rotary_axis,
            stepover=stepover,
            stepdown=stepdown,
            **kwargs
        )

        self.radius = original_radius
        return result

    def generate_finishing(
        self,
        rotary_axis: Literal['X', 'Y', 'Z'] = 'X',
        stepover: float = None,
        **kwargs
    ) -> Pipeline4AxisResult:
        """
        Generate finishing toolpath.

        Uses fine parameters for surface quality.

        Args:
            rotary_axis: Rotation axis
            stepover: Stepover (default: 10% of tool diameter for ball)
            **kwargs: Additional arguments for generate()

        Returns:
            Pipeline4AxisResult
        """
        if stepover is None:
            if self.tool.type == ToolType.BALL:
                stepover = self.tool.diameter * 0.1  # Fine stepover for ball nose
            else:
                stepover = self.tool.diameter * 0.2

        return self.generate(
            strategy='PARALLELR',  # Best surface finish
            rotary_axis=rotary_axis,
            stepover=stepover,
            stepdown=1000,  # Single pass
            num_layers=1,
            **kwargs
        )

    def generate_both(
        self,
        rotary_axis: Literal['X', 'Y', 'Z'] = 'X',
        roughing_tool: Optional[Tool] = None,
        finishing_tool: Optional[Tool] = None,
        **kwargs
    ) -> Tuple[Pipeline4AxisResult, Pipeline4AxisResult]:
        """
        Generate both roughing and finishing toolpaths.

        Args:
            rotary_axis: Rotation axis
            roughing_tool: Tool for roughing (default: current tool)
            finishing_tool: Tool for finishing (default: current tool)
            **kwargs: Additional arguments

        Returns:
            Tuple of (roughing_result, finishing_result)
        """
        # Roughing
        if roughing_tool:
            original_tool = self.tool
            self.tool = roughing_tool

        roughing = self.generate_roughing(rotary_axis=rotary_axis, **kwargs)

        if roughing_tool:
            self.tool = original_tool

        # Finishing
        if finishing_tool:
            original_tool = self.tool
            self.tool = finishing_tool

        finishing = self.generate_finishing(rotary_axis=rotary_axis, **kwargs)

        if finishing_tool:
            self.tool = original_tool

        return roughing, finishing

    def save_gcode(
        self,
        result: Pipeline4AxisResult,
        filepath: Union[str, Path],
        include_stats: bool = True
    ) -> None:
        """
        Save G-code to file.

        Args:
            result: Pipeline result
            filepath: Output file path
            include_stats: Include statistics as comments
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            if include_stats:
                f.write(f"(Statistics:)\n")
                f.write(f"(  Points: {result.total_points})\n")
                f.write(f"(  Path length: {result.path_length_mm:.1f} mm)\n")
                f.write(f"(  Est. time: {result.estimated_time_min:.1f} min)\n")
                f.write(f"(  Bounds: {result.bounds_min} to {result.bounds_max})\n")
                f.write("\n")

            f.write(result.gcode)

        logger.info(f"Saved G-code to {filepath}")


def quick_4axis(
    mesh_path: Union[str, Path],
    output_path: Union[str, Path],
    tool_diameter: float = 6.0,
    tool_type: str = 'ball',
    strategy: str = 'PARALLELR',
    machine: str = 'LUQUE_L1530',
    **kwargs
) -> Pipeline4AxisResult:
    """
    Quick function for simple 4-axis toolpath generation.

    Args:
        mesh_path: Path to input mesh
        output_path: Path for output G-code
        tool_diameter: Tool diameter in mm
        tool_type: 'ball', 'flat', or 'bull'
        strategy: 'PARALLELR', 'PARALLEL', 'HELIX', or 'CROSS'
        machine: Machine type name
        **kwargs: Additional arguments for generate()

    Returns:
        Pipeline4AxisResult

    Example:
        result = quick_4axis(
            "part.stl",
            "output.nc",
            tool_diameter=6.0,
            strategy='PARALLELR',
            feed_rate=1000
        )
    """
    # Create tool
    if tool_type == 'ball':
        tool = Tool.ball(diameter=tool_diameter)
    elif tool_type == 'flat':
        tool = Tool.flat(diameter=tool_diameter)
    elif tool_type == 'bull':
        tool = Tool.bull(diameter=tool_diameter, corner_radius=tool_diameter * 0.1)
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")

    # Get machine type
    machine_type = MachineType4Axis[machine.upper()]

    # Create pipeline and generate
    pipeline = Pipeline4Axis(mesh_path, tool, machine_type)
    result = pipeline.generate(strategy=strategy, **kwargs)

    # Save
    pipeline.save_gcode(result, output_path)

    return result
