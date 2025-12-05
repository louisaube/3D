#!/usr/bin/env python3
"""
Advanced Toolpath Strategies Example.

This example demonstrates different toolpath strategies:
- Parallel X/Y
- Zigzag (bidirectional)
- Waterline (constant Z)
- Combined roughing + finishing

Usage:
    python advanced_strategies.py model.stl
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pycam3d import CAMPipeline, Tool
from pycam3d.toolpath import Strategy
from pycam3d.gcode import GCodeWriter, MachineConfig, MachineType


def generate_parallel_toolpath(pipeline: CAMPipeline, input_file: Path):
    """Generate parallel X toolpath."""
    print("\n=== Parallel X Strategy ===")

    pipeline.load_mesh(input_file)
    pipeline.prepare_mesh()

    # Manual toolpath generation for more control
    pipeline._prepare_for_toolpath()

    tool = Tool.ball(diameter=6.0)
    toolpath = pipeline.toolpath_generator.generate_parallel(
        tool=tool,
        stepover=0.9,  # 0.9mm stepover
        strategy=Strategy.PARALLEL_X,
    )

    writer = GCodeWriter()
    gcode = writer.generate(toolpath, program_name="PARALLEL_X")

    output = input_file.with_stem(input_file.stem + "_parallel_x").with_suffix(".nc")
    gcode.save(output)
    print(f"  Saved to: {output}")
    print(f"  Points: {len(toolpath)}, Length: {toolpath.get_total_length():.0f}mm")


def generate_zigzag_toolpath(pipeline: CAMPipeline, input_file: Path):
    """Generate zigzag (bidirectional) toolpath."""
    print("\n=== Zigzag Strategy ===")

    pipeline.load_mesh(input_file)
    pipeline.prepare_mesh()
    pipeline._prepare_for_toolpath()

    tool = Tool.ball(diameter=6.0)
    toolpath = pipeline.toolpath_generator.generate_parallel(
        tool=tool,
        stepover=0.9,
        strategy=Strategy.ZIGZAG_Y,  # Zigzag along Y
    )

    writer = GCodeWriter()
    gcode = writer.generate(toolpath, program_name="ZIGZAG_Y")

    output = input_file.with_stem(input_file.stem + "_zigzag").with_suffix(".nc")
    gcode.save(output)
    print(f"  Saved to: {output}")
    print(f"  Points: {len(toolpath)}, Length: {toolpath.get_total_length():.0f}mm")


def generate_waterline_toolpath(pipeline: CAMPipeline, input_file: Path):
    """Generate waterline (constant Z) toolpath."""
    print("\n=== Waterline Strategy ===")

    pipeline.load_mesh(input_file)
    pipeline.prepare_mesh()

    tool = Tool.ball(diameter=6.0)
    result = pipeline.generate_waterline(
        tool=tool,
        z_step=1.0,  # 1mm between Z levels
        feed_rate=800.0,
    )

    output = input_file.with_stem(input_file.stem + "_waterline").with_suffix(".nc")
    result.gcode.save(output)
    print(f"  Saved to: {output}")
    print(f"  Points: {len(result.finishing_toolpath)}")


def generate_for_grbl(pipeline: CAMPipeline, input_file: Path):
    """Generate G-code optimized for GRBL controllers."""
    print("\n=== GRBL-Optimized Output ===")

    pipeline.load_mesh(input_file)
    pipeline.prepare_mesh()
    pipeline._prepare_for_toolpath()

    tool = Tool.ball(diameter=6.0)
    toolpath = pipeline.toolpath_generator.generate_parallel(
        tool=tool,
        stepover=0.9,
        strategy=Strategy.ZIGZAG_X,
    )

    # Configure for GRBL
    config = MachineConfig(
        machine_type=MachineType.GRBL,
        max_feed_rate=3000.0,  # GRBL typically limited
        max_spindle_rpm=24000,
        safe_z=5.0,
        decimal_places=3,
    )

    writer = GCodeWriter(config)
    gcode = writer.generate(toolpath, program_name="GRBL", spindle_rpm=15000)

    output = input_file.with_stem(input_file.stem + "_grbl").with_suffix(".nc")
    gcode.save(output)
    print(f"  Saved to: {output}")
    print(f"  Machine type: GRBL")


def main():
    if len(sys.argv) < 2:
        print("Usage: python advanced_strategies.py <model.stl>")
        print("\nThis will generate multiple G-code files with different strategies:")
        print("  - *_parallel_x.nc  : Parallel lines along X")
        print("  - *_zigzag.nc      : Bidirectional zigzag")
        print("  - *_waterline.nc   : Constant Z contours")
        print("  - *_grbl.nc        : Optimized for GRBL")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    print(f"Input: {input_file}")

    pipeline = CAMPipeline()

    try:
        generate_parallel_toolpath(pipeline, input_file)
        generate_zigzag_toolpath(pipeline, input_file)
        generate_waterline_toolpath(pipeline, input_file)
        generate_for_grbl(pipeline, input_file)
    except ImportError as e:
        print(f"\nError: {e}")
        print("Make sure OpenCAMLib is installed: pip install opencamlib")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("All strategies generated successfully!")


if __name__ == "__main__":
    main()
