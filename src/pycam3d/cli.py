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


if __name__ == "__main__":
    main()
