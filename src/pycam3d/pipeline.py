"""
CAM Pipeline - Complete workflow orchestration.

This module provides a high-level interface that combines:
- Mesh loading and processing (Trimesh + Open3D)
- Toolpath generation (OpenCAMLib)
- G-code output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pycam3d.mesh import MeshProcessor, ReconstructionMethod
from pycam3d.toolpath import Tool, ToolpathGenerator, Toolpath, Strategy
from pycam3d.gcode import GCodeWriter, GCodeProgram, MachineConfig

logger = logging.getLogger(__name__)


@dataclass
class CAMJob:
    """
    Definition of a CAM job.

    Attributes:
        roughing_tool: Tool for roughing pass.
        finishing_tool: Tool for finishing pass.
        roughing_stepover: XY stepover for roughing (% of tool diameter).
        roughing_z_step: Z step for roughing.
        finishing_stepover: XY stepover for finishing (% of tool diameter).
        stock_to_leave: Material to leave after roughing.
        feed_rate: Cutting feed rate (mm/min).
        plunge_rate: Plunge feed rate (mm/min).
        spindle_rpm: Spindle speed (RPM).
        safe_z: Safe Z height for rapids.
    """

    roughing_tool: Optional[Tool] = None
    finishing_tool: Optional[Tool] = None
    roughing_stepover: float = 0.5  # 50% of tool diameter
    roughing_z_step: float = 2.0
    finishing_stepover: float = 0.1  # 10% of tool diameter
    stock_to_leave: float = 0.5
    feed_rate: float = 1000.0
    plunge_rate: float = 300.0
    spindle_rpm: int = 10000
    safe_z: float = 10.0
    coolant: bool = False

    @classmethod
    def default_3d_surfacing(cls) -> CAMJob:
        """Create a default job for 3D surface machining."""
        return cls(
            roughing_tool=Tool.flat(diameter=10.0),
            finishing_tool=Tool.ball(diameter=6.0),
            roughing_stepover=0.4,
            roughing_z_step=3.0,
            finishing_stepover=0.15,
            stock_to_leave=0.3,
            feed_rate=1200.0,
            plunge_rate=400.0,
            spindle_rpm=12000,
        )

    @classmethod
    def default_finishing_only(cls, tool_diameter: float = 6.0) -> CAMJob:
        """Create a job for finishing pass only (no roughing)."""
        return cls(
            roughing_tool=None,
            finishing_tool=Tool.ball(diameter=tool_diameter),
            finishing_stepover=0.1,
            feed_rate=1000.0,
            spindle_rpm=15000,
        )


@dataclass
class CAMResult:
    """
    Result of a CAM pipeline run.

    Attributes:
        roughing_toolpath: Roughing toolpath (if generated).
        finishing_toolpath: Finishing toolpath.
        gcode: Generated G-code program.
        stats: Processing statistics.
    """

    roughing_toolpath: Optional[Toolpath] = None
    finishing_toolpath: Optional[Toolpath] = None
    gcode: Optional[GCodeProgram] = None
    stats: dict = field(default_factory=dict)


class CAMPipeline:
    """
    Complete CAM pipeline from 3D scan to G-code.

    This class orchestrates the full workflow:
    1. Load input (mesh or point cloud)
    2. Clean and repair mesh
    3. Generate toolpaths
    4. Output G-code

    Example:
        ```python
        pipeline = CAMPipeline()
        pipeline.load_mesh("model.stl")
        pipeline.prepare_mesh()

        job = CAMJob.default_3d_surfacing()
        result = pipeline.run(job)

        result.gcode.save("output.nc")
        ```
    """

    def __init__(self, machine_config: Optional[MachineConfig] = None):
        """
        Initialize the CAM pipeline.

        Args:
            machine_config: Machine configuration for G-code generation.
        """
        self.mesh_processor = MeshProcessor()
        self.toolpath_generator = ToolpathGenerator()
        self.gcode_writer = GCodeWriter(machine_config)
        self._temp_stl_path: Optional[Path] = None

    def load_mesh(self, filepath: str | Path) -> None:
        """
        Load a mesh file (STL, OBJ, PLY, etc.).

        Args:
            filepath: Path to the mesh file.
        """
        logger.info(f"Loading mesh from {filepath}")
        self.mesh_processor.load_mesh(filepath)

    def load_point_cloud(
        self,
        filepath: str | Path,
        method: ReconstructionMethod = ReconstructionMethod.POISSON,
        depth: int = 9,
    ) -> None:
        """
        Load a point cloud and reconstruct as mesh.

        Args:
            filepath: Path to the point cloud file.
            method: Reconstruction method.
            depth: Depth for Poisson reconstruction.
        """
        logger.info(f"Loading point cloud from {filepath}")
        self.mesh_processor.load_point_cloud(filepath, method, depth)

    def prepare_mesh(
        self,
        repair: bool = True,
        center: bool = True,
        place_on_bed: bool = True,
        simplify_to: Optional[int] = None,
        smooth_iterations: int = 0,
    ) -> dict:
        """
        Prepare the mesh for machining.

        Args:
            repair: Whether to repair the mesh.
            center: Whether to center the mesh XY.
            place_on_bed: Whether to place mesh bottom at Z=0.
            simplify_to: Target face count for simplification (None = no simplification).
            smooth_iterations: Number of smoothing iterations (0 = no smoothing).

        Returns:
            Dictionary with preparation statistics.
        """
        if not self.mesh_processor.has_mesh:
            raise ValueError("No mesh loaded")

        stats = {}
        logger.info("Preparing mesh for machining...")

        # Repair
        if repair:
            repair_stats = self.mesh_processor.repair()
            stats["repair"] = repair_stats

        # Simplify
        if simplify_to is not None:
            initial_faces = len(self.mesh_processor.mesh.faces)
            self.mesh_processor.simplify(simplify_to)
            stats["simplification"] = {
                "initial_faces": initial_faces,
                "final_faces": len(self.mesh_processor.mesh.faces),
            }

        # Smooth
        if smooth_iterations > 0:
            self.mesh_processor.smooth(iterations=smooth_iterations)
            stats["smoothing"] = {"iterations": smooth_iterations}

        # Transform
        if center:
            offset = self.mesh_processor.center()
            # Keep Z position, only center XY
            self.mesh_processor.translate([0, 0, -offset[2]])
            stats["centered"] = True

        if place_on_bed:
            z_offset = self.mesh_processor.place_on_bed()
            stats["z_offset"] = z_offset

        # Get final stats
        stats["mesh_stats"] = str(self.mesh_processor.get_stats())

        logger.info("Mesh preparation complete")
        return stats

    def _prepare_for_toolpath(self) -> Path:
        """Export mesh to temp STL for toolpath generation."""
        import tempfile

        if not self.mesh_processor.has_mesh:
            raise ValueError("No mesh loaded")

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            self._temp_stl_path = Path(f.name)

        self.mesh_processor.export(self._temp_stl_path)
        self.toolpath_generator.load_stl(self._temp_stl_path)

        return self._temp_stl_path

    def run(
        self,
        job: CAMJob,
        program_name: str = "PYCAM3D",
        strategy: Strategy = Strategy.ZIGZAG_X,
    ) -> CAMResult:
        """
        Run the complete CAM pipeline.

        Args:
            job: CAM job definition.
            program_name: Name for the G-code program.
            strategy: Toolpath strategy.

        Returns:
            CAMResult with toolpaths and G-code.
        """
        result = CAMResult()
        result.stats["job"] = {
            "roughing_tool": job.roughing_tool.name if job.roughing_tool else None,
            "finishing_tool": job.finishing_tool.name if job.finishing_tool else None,
        }

        # Prepare mesh for toolpath generation
        logger.info("Preparing mesh for toolpath generation...")
        self._prepare_for_toolpath()

        toolpaths = []

        # Generate roughing toolpath
        if job.roughing_tool:
            logger.info("Generating roughing toolpath...")
            stepover = job.roughing_tool.diameter * job.roughing_stepover

            roughing_tp = self.toolpath_generator.generate_roughing(
                tool=job.roughing_tool,
                stepover=stepover,
                z_step=job.roughing_z_step,
                stock_to_leave=job.stock_to_leave,
                z_safe=job.safe_z,
            )
            roughing_tp.feed_rate = job.feed_rate
            roughing_tp.plunge_rate = job.plunge_rate

            result.roughing_toolpath = roughing_tp
            toolpaths.append(roughing_tp)

            result.stats["roughing"] = {
                "points": len(roughing_tp),
                "length_mm": roughing_tp.get_total_length(),
            }

        # Generate finishing toolpath
        if job.finishing_tool:
            logger.info("Generating finishing toolpath...")
            stepover = job.finishing_tool.diameter * job.finishing_stepover

            finishing_tp = self.toolpath_generator.generate_finishing(
                tool=job.finishing_tool,
                stepover=stepover,
                strategy=strategy,
                z_safe=job.safe_z,
            )
            finishing_tp.feed_rate = job.feed_rate
            finishing_tp.plunge_rate = job.plunge_rate

            result.finishing_toolpath = finishing_tp
            toolpaths.append(finishing_tp)

            result.stats["finishing"] = {
                "points": len(finishing_tp),
                "length_mm": finishing_tp.get_total_length(),
            }

        # Generate G-code
        if toolpaths:
            logger.info("Generating G-code...")

            if len(toolpaths) == 1:
                result.gcode = self.gcode_writer.generate(
                    toolpaths[0],
                    program_name=program_name,
                    spindle_rpm=job.spindle_rpm,
                    coolant=job.coolant,
                )
            else:
                result.gcode = self.gcode_writer.generate_multi(
                    toolpaths,
                    program_name=program_name,
                    spindle_rpm=job.spindle_rpm,
                    coolant=job.coolant,
                )

            result.stats["gcode_lines"] = len(result.gcode)

        # Cleanup temp file
        if self._temp_stl_path and self._temp_stl_path.exists():
            self._temp_stl_path.unlink()
            self._temp_stl_path = None

        logger.info("Pipeline complete")
        return result

    def generate_waterline(
        self,
        tool: Tool,
        z_step: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        spindle_rpm: int = 10000,
        feed_rate: float = 1000.0,
        program_name: str = "WATERLINE",
    ) -> CAMResult:
        """
        Generate waterline (constant Z) toolpaths.

        Args:
            tool: Cutting tool.
            z_step: Step between Z levels.
            z_min: Minimum Z (defaults to part bottom).
            z_max: Maximum Z (defaults to part top).
            spindle_rpm: Spindle speed.
            feed_rate: Feed rate.
            program_name: G-code program name.

        Returns:
            CAMResult with waterline toolpath and G-code.
        """
        result = CAMResult()

        # Prepare mesh
        self._prepare_for_toolpath()

        # Generate waterline
        logger.info("Generating waterline toolpath...")
        waterline_tp = self.toolpath_generator.generate_waterline(
            tool=tool,
            z_step=z_step,
            z_min=z_min,
            z_max=z_max,
        )
        waterline_tp.feed_rate = feed_rate

        result.finishing_toolpath = waterline_tp

        # Generate G-code
        result.gcode = self.gcode_writer.generate(
            waterline_tp,
            program_name=program_name,
            spindle_rpm=spindle_rpm,
        )

        result.stats = {
            "points": len(waterline_tp),
            "length_mm": waterline_tp.get_total_length(),
            "gcode_lines": len(result.gcode),
        }

        # Cleanup
        if self._temp_stl_path and self._temp_stl_path.exists():
            self._temp_stl_path.unlink()
            self._temp_stl_path = None

        return result

    def export_mesh(self, filepath: str | Path) -> None:
        """Export the processed mesh."""
        self.mesh_processor.export(filepath)

    def get_mesh_stats(self) -> dict:
        """Get statistics about the loaded mesh."""
        return self.mesh_processor.get_stats().__dict__
