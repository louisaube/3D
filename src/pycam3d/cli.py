"""
Command-line interface for PyCAM3D.

Provides commands for:
- Processing 3D meshes
- Generating toolpaths
- Creating G-code
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="pycam3d")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """
    PyCAM3D - CAM software for 3D toolpath generation.

    Integrates OpenCAMLib, Trimesh, and Open3D for complete
    scan-to-gcode workflow.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output G-code file")
@click.option("--tool-diameter", "-d", type=float, default=6.0, help="Tool diameter in mm")
@click.option("--tool-type", "-t", type=click.Choice(["ball", "flat", "bull"]), default="ball")
@click.option("--stepover", "-s", type=float, default=0.15, help="Stepover as fraction of tool diameter")
@click.option("--feed-rate", "-f", type=float, default=1000.0, help="Feed rate in mm/min")
@click.option("--spindle-rpm", "-r", type=int, default=12000, help="Spindle RPM")
@click.option("--roughing/--no-roughing", default=False, help="Include roughing pass")
@click.option("--roughing-tool", type=float, default=10.0, help="Roughing tool diameter")
@click.option("--z-step", type=float, default=2.0, help="Z step for roughing")
@click.option("--stock-to-leave", type=float, default=0.3, help="Stock to leave after roughing")
@click.option("--safe-z", type=float, default=10.0, help="Safe Z height")
@click.option("--repair/--no-repair", default=True, help="Repair mesh before processing")
@click.pass_context
def process(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    tool_diameter: float,
    tool_type: str,
    stepover: float,
    feed_rate: float,
    spindle_rpm: int,
    roughing: bool,
    roughing_tool: float,
    z_step: float,
    stock_to_leave: float,
    safe_z: float,
    repair: bool,
) -> None:
    """
    Process a 3D model and generate G-code.

    INPUT_FILE can be an STL, OBJ, PLY, or other mesh format.
    """
    from pycam3d.pipeline import CAMPipeline, CAMJob
    from pycam3d.toolpath import Tool

    input_path = Path(input_file)
    if output is None:
        output = str(input_path.with_suffix(".nc"))

    console.print(Panel(f"[bold blue]PyCAM3D[/bold blue]\nProcessing: {input_path.name}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize pipeline
        pipeline = CAMPipeline()

        # Load mesh
        task = progress.add_task("Loading mesh...", total=None)
        try:
            pipeline.load_mesh(input_path)
        except Exception as e:
            console.print(f"[red]Error loading mesh: {e}[/red]")
            sys.exit(1)
        progress.update(task, completed=True, description="Mesh loaded")

        # Prepare mesh
        task = progress.add_task("Preparing mesh...", total=None)
        stats = pipeline.prepare_mesh(repair=repair, center=True, place_on_bed=True)
        progress.update(task, completed=True, description="Mesh prepared")

        # Display mesh info
        mesh_stats = pipeline.get_mesh_stats()
        table = Table(title="Mesh Statistics")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Vertices", f"{mesh_stats['vertex_count']:,}")
        table.add_row("Faces", f"{mesh_stats['face_count']:,}")
        table.add_row("Watertight", str(mesh_stats["is_watertight"]))
        table.add_row("Surface Area", f"{mesh_stats['surface_area']:.2f} mm²")
        if mesh_stats.get("volume"):
            table.add_row("Volume", f"{mesh_stats['volume']:.2f} mm³")
        console.print(table)

        # Create job
        if tool_type == "ball":
            finishing_tool = Tool.ball(diameter=tool_diameter)
        elif tool_type == "flat":
            finishing_tool = Tool.flat(diameter=tool_diameter)
        else:
            finishing_tool = Tool.bull(diameter=tool_diameter, corner_radius=tool_diameter * 0.1)

        job = CAMJob(
            roughing_tool=Tool.flat(diameter=roughing_tool) if roughing else None,
            finishing_tool=finishing_tool,
            roughing_stepover=0.4,
            roughing_z_step=z_step,
            finishing_stepover=stepover,
            stock_to_leave=stock_to_leave,
            feed_rate=feed_rate,
            spindle_rpm=spindle_rpm,
            safe_z=safe_z,
        )

        # Run pipeline
        task = progress.add_task("Generating toolpaths...", total=None)
        try:
            result = pipeline.run(job)
        except ImportError as e:
            console.print(f"[red]Missing dependency: {e}[/red]")
            console.print("[yellow]Install OpenCAMLib with: pip install opencamlib[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error generating toolpaths: {e}[/red]")
            sys.exit(1)
        progress.update(task, completed=True, description="Toolpaths generated")

        # Save G-code
        task = progress.add_task("Saving G-code...", total=None)
        result.gcode.save(output)
        progress.update(task, completed=True, description="G-code saved")

    # Display results
    result_table = Table(title="Generation Results")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")

    if result.roughing_toolpath:
        result_table.add_row("Roughing Points", f"{len(result.roughing_toolpath):,}")
        result_table.add_row("Roughing Length", f"{result.roughing_toolpath.get_total_length():.1f} mm")

    if result.finishing_toolpath:
        result_table.add_row("Finishing Points", f"{len(result.finishing_toolpath):,}")
        result_table.add_row("Finishing Length", f"{result.finishing_toolpath.get_total_length():.1f} mm")

    result_table.add_row("G-code Lines", f"{len(result.gcode):,}")

    console.print(result_table)
    console.print(f"\n[green]G-code saved to: {output}[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output G-code file")
@click.option("--tool-diameter", "-d", type=float, default=6.0, help="Tool diameter in mm")
@click.option("--z-step", "-z", type=float, default=1.0, help="Z step between waterlines")
@click.option("--feed-rate", "-f", type=float, default=1000.0, help="Feed rate in mm/min")
@click.option("--spindle-rpm", "-r", type=int, default=12000, help="Spindle RPM")
@click.pass_context
def waterline(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    tool_diameter: float,
    z_step: float,
    feed_rate: float,
    spindle_rpm: int,
) -> None:
    """
    Generate waterline (constant Z) toolpaths.

    Useful for steep walls and vertical surfaces.
    """
    from pycam3d.pipeline import CAMPipeline
    from pycam3d.toolpath import Tool

    input_path = Path(input_file)
    if output is None:
        output = str(input_path.with_suffix(".waterline.nc"))

    console.print(Panel(f"[bold blue]PyCAM3D Waterline[/bold blue]\nProcessing: {input_path.name}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        pipeline = CAMPipeline()

        task = progress.add_task("Loading mesh...", total=None)
        pipeline.load_mesh(input_path)
        progress.update(task, completed=True)

        task = progress.add_task("Preparing mesh...", total=None)
        pipeline.prepare_mesh(repair=True)
        progress.update(task, completed=True)

        task = progress.add_task("Generating waterlines...", total=None)
        tool = Tool.ball(diameter=tool_diameter)
        result = pipeline.generate_waterline(
            tool=tool,
            z_step=z_step,
            feed_rate=feed_rate,
            spindle_rpm=spindle_rpm,
        )
        progress.update(task, completed=True)

        task = progress.add_task("Saving G-code...", total=None)
        result.gcode.save(output)
        progress.update(task, completed=True)

    console.print(f"\n[green]Waterline G-code saved to: {output}[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output mesh file")
@click.option("--repair/--no-repair", default=True, help="Repair mesh")
@click.option("--simplify", type=int, default=None, help="Simplify to N faces")
@click.option("--smooth", type=int, default=0, help="Smoothing iterations")
@click.option("--center/--no-center", default=True, help="Center mesh")
@click.pass_context
def prepare(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    repair: bool,
    simplify: int | None,
    smooth: int,
    center: bool,
) -> None:
    """
    Prepare a mesh for machining (repair, simplify, etc.).

    Outputs a cleaned STL ready for toolpath generation.
    """
    from pycam3d.mesh import MeshProcessor

    input_path = Path(input_file)
    if output is None:
        output = str(input_path.with_stem(input_path.stem + "_prepared").with_suffix(".stl"))

    console.print(Panel(f"[bold blue]Mesh Preparation[/bold blue]\nInput: {input_path.name}"))

    processor = MeshProcessor()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading mesh...", total=None)
        processor.load_mesh(input_path)
        progress.update(task, completed=True)

        initial_stats = processor.get_stats()

        if repair:
            task = progress.add_task("Repairing mesh...", total=None)
            processor.repair()
            progress.update(task, completed=True)

        if simplify:
            task = progress.add_task(f"Simplifying to {simplify} faces...", total=None)
            processor.simplify(simplify)
            progress.update(task, completed=True)

        if smooth > 0:
            task = progress.add_task(f"Smoothing ({smooth} iterations)...", total=None)
            processor.smooth(iterations=smooth)
            progress.update(task, completed=True)

        if center:
            task = progress.add_task("Centering mesh...", total=None)
            processor.center()
            processor.place_on_bed()
            progress.update(task, completed=True)

        task = progress.add_task("Saving mesh...", total=None)
        processor.export(output)
        progress.update(task, completed=True)

    final_stats = processor.get_stats()

    table = Table(title="Mesh Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Before", style="yellow")
    table.add_column("After", style="green")
    table.add_row("Vertices", f"{initial_stats.vertex_count:,}", f"{final_stats.vertex_count:,}")
    table.add_row("Faces", f"{initial_stats.face_count:,}", f"{final_stats.face_count:,}")
    table.add_row("Watertight", str(initial_stats.is_watertight), str(final_stats.is_watertight))

    console.print(table)
    console.print(f"\n[green]Prepared mesh saved to: {output}[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def info(ctx: click.Context, input_file: str) -> None:
    """
    Display information about a mesh file.
    """
    from pycam3d.mesh import MeshProcessor

    input_path = Path(input_file)

    console.print(Panel(f"[bold blue]Mesh Info[/bold blue]\nFile: {input_path.name}"))

    processor = MeshProcessor()
    processor.load_mesh(input_path)
    stats = processor.get_stats()

    table = Table(title="Mesh Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Vertices", f"{stats.vertex_count:,}")
    table.add_row("Faces", f"{stats.face_count:,}")
    table.add_row("Watertight", str(stats.is_watertight))
    table.add_row("Winding Consistent", str(stats.is_winding_consistent))
    table.add_row("Surface Area", f"{stats.surface_area:.2f} mm²")
    if stats.volume:
        table.add_row("Volume", f"{stats.volume:.2f} mm³")

    table.add_row("Min Bounds", f"({stats.bounds_min[0]:.2f}, {stats.bounds_min[1]:.2f}, {stats.bounds_min[2]:.2f})")
    table.add_row("Max Bounds", f"({stats.bounds_max[0]:.2f}, {stats.bounds_max[1]:.2f}, {stats.bounds_max[2]:.2f})")

    size = stats.bounds_max - stats.bounds_min
    table.add_row("Size (X x Y x Z)", f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm")

    console.print(table)


@main.command()
@click.argument("gcode_file", type=click.Path(exists=True))
@click.pass_context
def simulate(ctx: click.Context, gcode_file: str) -> None:
    """
    Simulate G-code and display statistics.
    """
    from pycam3d.gcode import GCodeProgram, simulate_gcode

    gcode_path = Path(gcode_file)

    console.print(Panel(f"[bold blue]G-code Simulation[/bold blue]\nFile: {gcode_path.name}"))

    with open(gcode_path) as f:
        lines = f.readlines()

    program = GCodeProgram(lines=[line.strip() for line in lines])
    stats = simulate_gcode(program)

    table = Table(title="Simulation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Rapid Distance", f"{stats['rapid_distance']:.1f} mm")
    table.add_row("Cut Distance", f"{stats['cut_distance']:.1f} mm")
    table.add_row("Total Distance", f"{stats['rapid_distance'] + stats['cut_distance']:.1f} mm")
    table.add_row("Estimated Time", f"{stats['estimated_time_min']:.1f} minutes")
    table.add_row("X Range", f"{stats['min_x']:.2f} to {stats['max_x']:.2f} mm")
    table.add_row("Y Range", f"{stats['min_y']:.2f} to {stats['max_y']:.2f} mm")
    table.add_row("Z Range", f"{stats['min_z']:.2f} to {stats['max_z']:.2f} mm")

    console.print(table)


# =============================================================================
# ADVANCED STRATEGIES
# =============================================================================

@main.command(name="iso-scallop")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output G-code file")
@click.option("--tool-diameter", "-d", type=float, default=6.0, help="Ball tool diameter in mm")
@click.option("--scallop", "-s", type=float, default=0.02, help="Target scallop height in mm")
@click.option("--min-stepover", type=float, default=None, help="Minimum stepover (default: 5% diameter)")
@click.option("--max-stepover", type=float, default=None, help="Maximum stepover (default: 50% diameter)")
@click.option("--feed-rate", "-f", type=float, default=1000.0, help="Feed rate in mm/min")
@click.option("--spindle-rpm", "-r", type=int, default=12000, help="Spindle RPM")
@click.option("--safe-z", type=float, default=10.0, help="Safe Z height")
@click.option("--direction", type=click.Choice(["x", "y", "both"]), default="x", help="Cutting direction")
@click.pass_context
def iso_scallop(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    tool_diameter: float,
    scallop: float,
    min_stepover: float | None,
    max_stepover: float | None,
    feed_rate: float,
    spindle_rpm: int,
    safe_z: float,
    direction: str,
) -> None:
    """
    Generate iso-scallop toolpath with adaptive stepover.

    Uses curvature analysis to maintain constant scallop height,
    resulting in 7-21% shorter toolpaths vs fixed stepover.
    Best with ball-end mills for 3D surface finishing.
    """
    import numpy as np
    import trimesh
    import opencamlib as ocl

    from pycam3d.toolpath import Tool
    from pycam3d.strategies import IsoScallopGenerator
    from pycam3d.gcode import GCodeWriter

    input_path = Path(input_file)
    if output is None:
        output = str(input_path.with_suffix(".iso-scallop.nc"))

    console.print(Panel(f"[bold blue]Iso-Scallop Strategy[/bold blue]\nProcessing: {input_path.name}\nTarget scallop: {scallop}mm"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading mesh...", total=None)
        mesh = trimesh.load_mesh(input_path)
        progress.update(task, completed=True)

        task = progress.add_task("Loading for OpenCAMLib...", total=None)
        stl_surf = ocl.STLSurf()
        ocl.STLReader(str(input_path), stl_surf)
        progress.update(task, completed=True)

        bounds_min = np.array([mesh.vertices[:, i].min() for i in range(3)])
        bounds_max = np.array([mesh.vertices[:, i].max() for i in range(3)])
        bounds = (bounds_min, bounds_max)

        task = progress.add_task("Computing curvature and generating toolpath...", total=None)
        generator = IsoScallopGenerator(mesh, stl_surf, bounds)
        tool = Tool.ball(diameter=tool_diameter)

        toolpath = generator.generate(
            tool=tool,
            target_scallop=scallop,
            min_stepover=min_stepover,
            max_stepover=max_stepover,
            z_safe=safe_z,
            direction=direction,
        )
        toolpath.feed_rate = feed_rate
        progress.update(task, completed=True)

        task = progress.add_task("Generating G-code...", total=None)
        writer = GCodeWriter()
        gcode = writer.generate(toolpath, spindle_rpm=spindle_rpm)
        gcode.save(output)
        progress.update(task, completed=True)

    table = Table(title="Iso-Scallop Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Toolpath Points", f"{len(toolpath):,}")
    table.add_row("Total Length", f"{toolpath.get_total_length():.1f} mm")
    table.add_row("G-code Lines", f"{len(gcode):,}")

    console.print(table)
    console.print(f"\n[green]Iso-scallop G-code saved to: {output}[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output G-code file")
@click.option("--tool-diameter", "-d", type=float, default=6.0, help="Ball tool diameter in mm")
@click.option("--stepover", "-s", type=float, default=2.0, help="Radial stepover in mm")
@click.option("--feed-rate", "-f", type=float, default=1000.0, help="Feed rate in mm/min")
@click.option("--spindle-rpm", "-r", type=int, default=12000, help="Spindle RPM")
@click.option("--safe-z", type=float, default=10.0, help="Safe Z height")
@click.option("--direction", type=click.Choice(["outward", "inward"]), default="outward", help="Spiral direction")
@click.pass_context
def spiral(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    tool_diameter: float,
    stepover: float,
    feed_rate: float,
    spindle_rpm: int,
    safe_z: float,
    direction: str,
) -> None:
    """
    Generate continuous spiral toolpath for 3D surfacing.

    Creates smooth spiral motion from center outward (or inward),
    minimizing retracts for faster cycle time and better finish.
    """
    import numpy as np
    import trimesh
    import opencamlib as ocl

    from pycam3d.toolpath import Tool
    from pycam3d.strategies import SpiralGenerator
    from pycam3d.gcode import GCodeWriter

    input_path = Path(input_file)
    if output is None:
        output = str(input_path.with_suffix(".spiral.nc"))

    console.print(Panel(f"[bold blue]Spiral Strategy[/bold blue]\nProcessing: {input_path.name}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading mesh...", total=None)
        mesh = trimesh.load_mesh(input_path)
        progress.update(task, completed=True)

        task = progress.add_task("Loading for OpenCAMLib...", total=None)
        stl_surf = ocl.STLSurf()
        ocl.STLReader(str(input_path), stl_surf)
        progress.update(task, completed=True)

        bounds_min = np.array([mesh.vertices[:, i].min() for i in range(3)])
        bounds_max = np.array([mesh.vertices[:, i].max() for i in range(3)])
        bounds = (bounds_min, bounds_max)

        task = progress.add_task("Generating spiral toolpath...", total=None)
        generator = SpiralGenerator(mesh, stl_surf, bounds)
        tool = Tool.ball(diameter=tool_diameter)

        toolpath = generator.generate_3d_spiral(
            tool=tool,
            stepover=stepover,
            z_safe=safe_z,
            direction=direction,
        )
        toolpath.feed_rate = feed_rate
        progress.update(task, completed=True)

        task = progress.add_task("Generating G-code...", total=None)
        writer = GCodeWriter()
        gcode = writer.generate(toolpath, spindle_rpm=spindle_rpm)
        gcode.save(output)
        progress.update(task, completed=True)

    table = Table(title="Spiral Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Toolpath Points", f"{len(toolpath):,}")
    table.add_row("Total Length", f"{toolpath.get_total_length():.1f} mm")
    table.add_row("G-code Lines", f"{len(gcode):,}")

    console.print(table)
    console.print(f"\n[green]Spiral G-code saved to: {output}[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output G-code file")
@click.option("--tool-diameter", "-d", type=float, default=10.0, help="Tool diameter in mm")
@click.option("--start-x", type=float, required=True, help="Slot start X coordinate")
@click.option("--start-y", type=float, required=True, help="Slot start Y coordinate")
@click.option("--end-x", type=float, required=True, help="Slot end X coordinate")
@click.option("--end-y", type=float, required=True, help="Slot end Y coordinate")
@click.option("--z-bottom", type=float, required=True, help="Bottom Z of slot")
@click.option("--z-top", type=float, default=0.0, help="Top Z (stock surface)")
@click.option("--troch-diameter", type=float, default=None, help="Trochoidal circle diameter (default: 80% tool)")
@click.option("--troch-stepover", type=float, default=None, help="Forward step per circle (default: 10% tool)")
@click.option("--feed-rate", "-f", type=float, default=1500.0, help="Feed rate in mm/min")
@click.option("--spindle-rpm", "-r", type=int, default=12000, help="Spindle RPM")
@click.option("--safe-z", type=float, default=10.0, help="Safe Z height")
@click.pass_context
def trochoidal(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    tool_diameter: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    z_bottom: float,
    z_top: float,
    troch_diameter: float | None,
    troch_stepover: float | None,
    feed_rate: float,
    spindle_rpm: int,
    safe_z: float,
) -> None:
    """
    Generate trochoidal milling toolpath for slot cutting.

    Uses circular motion patterns to maintain constant tool engagement,
    reducing cutting forces, vibration, and heat buildup.
    Ideal for deep slots and hard materials.
    """
    import numpy as np
    import trimesh
    import opencamlib as ocl

    from pycam3d.toolpath import Tool
    from pycam3d.strategies import TrochoidalGenerator, TrochoidalParams
    from pycam3d.gcode import GCodeWriter

    input_path = Path(input_file)
    if output is None:
        output = str(input_path.with_suffix(".trochoidal.nc"))

    console.print(Panel(f"[bold blue]Trochoidal Milling[/bold blue]\nSlot: ({start_x}, {start_y}) to ({end_x}, {end_y})\nDepth: {z_top} to {z_bottom}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading mesh...", total=None)
        mesh = trimesh.load_mesh(input_path)
        progress.update(task, completed=True)

        task = progress.add_task("Loading for OpenCAMLib...", total=None)
        stl_surf = ocl.STLSurf()
        ocl.STLReader(str(input_path), stl_surf)
        progress.update(task, completed=True)

        bounds_min = np.array([mesh.vertices[:, i].min() for i in range(3)])
        bounds_max = np.array([mesh.vertices[:, i].max() for i in range(3)])
        bounds = (bounds_min, bounds_max)

        task = progress.add_task("Generating trochoidal toolpath...", total=None)
        generator = TrochoidalGenerator(mesh, stl_surf, bounds)
        tool = Tool.flat(diameter=tool_diameter)

        params = TrochoidalParams(
            trochoidal_diameter=troch_diameter or tool_diameter * 0.8,
            stepover=troch_stepover or tool_diameter * 0.1,
        )

        toolpath = generator.generate_slot(
            tool=tool,
            start=(start_x, start_y),
            end=(end_x, end_y),
            z_bottom=z_bottom,
            z_top=z_top,
            params=params,
            z_safe=safe_z,
        )
        toolpath.feed_rate = feed_rate
        progress.update(task, completed=True)

        task = progress.add_task("Generating G-code...", total=None)
        writer = GCodeWriter()
        gcode = writer.generate(toolpath, spindle_rpm=spindle_rpm)
        gcode.save(output)
        progress.update(task, completed=True)

    table = Table(title="Trochoidal Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Toolpath Points", f"{len(toolpath):,}")
    table.add_row("Total Length", f"{toolpath.get_total_length():.1f} mm")
    table.add_row("G-code Lines", f"{len(gcode):,}")

    console.print(table)
    console.print(f"\n[green]Trochoidal G-code saved to: {output}[/green]")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def wizard(ctx: click.Context, input_file: str) -> None:
    """
    Interactive wizard for toolpath generation.

    Guides you through the complete workflow with simple questions:
    mesh analysis, tool selection, strategy choice, and G-code output.
    """
    import questionary
    from pycam3d.mesh import MeshProcessor
    from pycam3d.pipeline import CAMPipeline, CAMJob
    from pycam3d.toolpath import Tool

    input_path = Path(input_file)

    console.print(Panel.fit(
        "[bold blue]PyCAM3D Wizard[/bold blue]\n"
        "Interactive toolpath generation",
        border_style="blue"
    ))

    # Load and analyze mesh
    console.print("\n[cyan]Step 1: Analyzing mesh...[/cyan]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading mesh...", total=None)
        processor = MeshProcessor()
        processor.load_mesh(input_path)
        stats = processor.get_stats()
        progress.update(task, completed=True, description="Mesh loaded")

    # Display mesh info
    table = Table(title="Mesh Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Vertices", f"{stats.vertex_count:,}")
    table.add_row("Faces", f"{stats.face_count:,}")
    table.add_row("Watertight", "Yes ✓" if stats.is_watertight else "No ✗")
    size = stats.bounds_max - stats.bounds_min
    table.add_row("Size", f"{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm")
    console.print(table)

    # Repair mesh if needed
    if not stats.is_watertight:
        repair = questionary.confirm(
            "Mesh is not watertight. Repair it?",
            default=True
        ).ask()
        if repair:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Repairing mesh...", total=None)
                processor.repair()
                progress.update(task, completed=True)
            console.print("[green]Mesh repaired![/green]")

    # Strategy selection
    console.print("\n[cyan]Step 2: Select machining strategy[/cyan]")
    strategy = questionary.select(
        "Choose a strategy:",
        choices=[
            questionary.Choice("Iso-Scallop (Adaptive stepover for best finish)", value="iso-scallop"),
            questionary.Choice("Spiral (Continuous path, minimal retracts)", value="spiral"),
            questionary.Choice("Parallel Lines (Simple raster)", value="parallel"),
            questionary.Choice("Waterline (Constant Z for steep walls)", value="waterline"),
            questionary.Choice("Trochoidal (For slots and pockets)", value="trochoidal"),
        ]
    ).ask()

    if not strategy:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Tool selection
    console.print("\n[cyan]Step 3: Configure tool[/cyan]")
    tool_type = questionary.select(
        "Tool type:",
        choices=[
            questionary.Choice("Ball End Mill (Best for 3D surfaces)", value="ball"),
            questionary.Choice("Flat End Mill (For pockets and floors)", value="flat"),
            questionary.Choice("Bull Nose (Compromise between ball and flat)", value="bull"),
        ]
    ).ask()

    tool_diameter = questionary.text(
        "Tool diameter (mm):",
        default="6.0",
        validate=lambda x: x.replace(".", "").isdigit()
    ).ask()
    tool_diameter = float(tool_diameter)

    # Machining parameters
    console.print("\n[cyan]Step 4: Machining parameters[/cyan]")

    stepover_pct = questionary.text(
        "Stepover (% of tool diameter):",
        default="15",
        validate=lambda x: x.isdigit() and 1 <= int(x) <= 50
    ).ask()
    stepover = float(stepover_pct) / 100

    feed_rate = questionary.text(
        "Feed rate (mm/min):",
        default="1000",
        validate=lambda x: x.isdigit()
    ).ask()
    feed_rate = float(feed_rate)

    spindle_rpm = questionary.text(
        "Spindle speed (RPM):",
        default="12000",
        validate=lambda x: x.isdigit()
    ).ask()
    spindle_rpm = int(spindle_rpm)

    # Output file
    console.print("\n[cyan]Step 5: Output[/cyan]")
    default_output = str(input_path.with_suffix(f".{strategy}.nc"))
    output_file = questionary.text(
        "Output G-code file:",
        default=default_output
    ).ask()

    # Summary
    console.print("\n")
    summary = Table(title="Configuration Summary", show_header=False)
    summary.add_column("Setting", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Strategy", strategy.replace("-", " ").title())
    summary.add_row("Tool", f"{tool_type.title()} {tool_diameter}mm")
    summary.add_row("Stepover", f"{stepover_pct}%")
    summary.add_row("Feed Rate", f"{feed_rate} mm/min")
    summary.add_row("Spindle", f"{spindle_rpm} RPM")
    summary.add_row("Output", output_file)
    console.print(summary)

    proceed = questionary.confirm("Generate toolpath?", default=True).ask()
    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Generate toolpath
    console.print("\n[cyan]Generating toolpath...[/cyan]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Initializing...", total=None)

        # Create tool
        if tool_type == "ball":
            tool = Tool.ball(diameter=tool_diameter)
        elif tool_type == "flat":
            tool = Tool.flat(diameter=tool_diameter)
        else:
            tool = Tool.bull(diameter=tool_diameter, corner_radius=tool_diameter * 0.1)

        progress.update(task, description="Loading pipeline...")
        pipeline = CAMPipeline()
        pipeline.load_mesh(input_path)
        pipeline.prepare_mesh(repair=True, center=True, place_on_bed=True)

        progress.update(task, description="Creating job...")
        job = CAMJob(
            finishing_tool=tool,
            finishing_stepover=stepover,
            feed_rate=feed_rate,
            spindle_rpm=spindle_rpm,
            safe_z=10.0,
        )

        progress.update(task, description="Generating toolpath...")
        result = pipeline.run(job)

        progress.update(task, description="Saving G-code...")
        result.gcode.save(output_file)
        progress.update(task, completed=True, description="Complete!")

    # Results
    console.print("\n")
    result_table = Table(title="Results")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")

    if result.finishing_toolpath:
        result_table.add_row("Toolpath Points", f"{len(result.finishing_toolpath):,}")
        result_table.add_row("Toolpath Length", f"{result.finishing_toolpath.get_total_length():.1f} mm")
        time_min = result.finishing_toolpath.get_total_length() / feed_rate
        result_table.add_row("Estimated Time", f"{time_min:.1f} min")

    result_table.add_row("G-code Lines", f"{len(result.gcode):,}")
    console.print(result_table)

    console.print(f"\n[bold green]G-code saved to: {output_file}[/bold green]")

    # Offer to visualize
    visualize = questionary.confirm("Open in web viewer?", default=False).ask()
    if visualize:
        console.print("[cyan]Starting web server...[/cyan]")
        from pycam3d.web import run_server
        run_server()


@main.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind to")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """
    Start the web interface with 3D visualization.

    Opens a browser-based interface for uploading models,
    configuring toolpaths, and visualizing results in 3D.
    """
    from pycam3d.web import run_server

    console.print(Panel.fit(
        "[bold blue]PyCAM3D Web Interface[/bold blue]\n"
        f"Starting server at [cyan]http://{host}:{port}[/cyan]\n\n"
        "[dim]Press Ctrl+C to stop[/dim]",
        border_style="blue"
    ))

    run_server(host=host, port=port)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def curvature(ctx: click.Context, input_file: str) -> None:
    """
    Analyze surface curvature of a mesh.

    Computes principal curvatures, Gaussian and mean curvature
    for adaptive machining strategies.
    """
    import trimesh
    from pycam3d.curvature import CurvatureAnalyzer

    input_path = Path(input_file)

    console.print(Panel(f"[bold blue]Curvature Analysis[/bold blue]\nFile: {input_path.name}"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading mesh...", total=None)
        mesh = trimesh.load_mesh(input_path)
        progress.update(task, completed=True)

        task = progress.add_task("Computing curvature...", total=None)
        analyzer = CurvatureAnalyzer(mesh)
        field = analyzer.compute_curvature()
        progress.update(task, completed=True)

    table = Table(title="Curvature Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="green")
    table.add_column("Mean", style="blue")

    table.add_row(
        "Principal K1",
        f"{field.principal_curvatures_1.min():.4f}",
        f"{field.principal_curvatures_1.max():.4f}",
        f"{field.principal_curvatures_1.mean():.4f}",
    )
    table.add_row(
        "Principal K2",
        f"{field.principal_curvatures_2.min():.4f}",
        f"{field.principal_curvatures_2.max():.4f}",
        f"{field.principal_curvatures_2.mean():.4f}",
    )
    table.add_row(
        "Gaussian",
        f"{field.gaussian.min():.4f}",
        f"{field.gaussian.max():.4f}",
        f"{field.gaussian.mean():.4f}",
    )
    table.add_row(
        "Mean",
        f"{field.mean.min():.4f}",
        f"{field.mean.max():.4f}",
        f"{field.mean.mean():.4f}",
    )
    table.add_row(
        "Max Curvature",
        f"{field.max_curvature.min():.4f}",
        f"{field.max_curvature.max():.4f}",
        f"{field.max_curvature.mean():.4f}",
    )

    console.print(table)

    # Recommendations
    avg_max_k = field.max_curvature.mean()
    if avg_max_k > 0.1:
        console.print("\n[yellow]High curvature detected. Recommend iso-scallop strategy.[/yellow]")
    else:
        console.print("\n[green]Relatively flat surface. Standard parallel strategy should work well.[/green]")


if __name__ == "__main__":
    main()
