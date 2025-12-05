#!/usr/bin/env python3
"""
Basic PyCAM3D Workflow Example.

This example demonstrates the complete workflow:
1. Load a 3D mesh (STL)
2. Clean and prepare the mesh
3. Generate roughing and finishing toolpaths
4. Export G-code

Usage:
    python basic_workflow.py model.stl output.nc
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pycam3d import CAMPipeline, CAMJob, Tool


def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python basic_workflow.py <input.stl> [output.nc]")
        print("\nThis example will:")
        print("  1. Load and repair the mesh")
        print("  2. Generate roughing toolpath (10mm flat endmill)")
        print("  3. Generate finishing toolpath (6mm ball nose)")
        print("  4. Output G-code")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_suffix(".nc")

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Initialize pipeline
    pipeline = CAMPipeline()

    # Step 1: Load mesh
    print("Loading mesh...")
    pipeline.load_mesh(input_file)

    # Step 2: Prepare mesh
    print("Preparing mesh (repair, center, orient)...")
    stats = pipeline.prepare_mesh(
        repair=True,
        center=True,
        place_on_bed=True,
    )

    mesh_stats = pipeline.get_mesh_stats()
    print(f"  Vertices: {mesh_stats['vertex_count']:,}")
    print(f"  Faces: {mesh_stats['face_count']:,}")
    print(f"  Watertight: {mesh_stats['is_watertight']}")
    print()

    # Step 3: Define the CAM job
    job = CAMJob(
        # Roughing with 10mm flat endmill
        roughing_tool=Tool.flat(diameter=10.0),
        roughing_stepover=0.4,  # 40% stepover = 4mm
        roughing_z_step=3.0,  # 3mm depth per pass
        stock_to_leave=0.3,  # Leave 0.3mm for finishing

        # Finishing with 6mm ball nose
        finishing_tool=Tool.ball(diameter=6.0),
        finishing_stepover=0.15,  # 15% stepover = 0.9mm for fine finish

        # Speeds and feeds
        feed_rate=1200.0,  # mm/min
        plunge_rate=400.0,  # mm/min
        spindle_rpm=12000,
        safe_z=10.0,
    )

    # Step 4: Run the pipeline
    print("Generating toolpaths...")
    result = pipeline.run(job, program_name="EXAMPLE")

    print(f"  Roughing: {len(result.roughing_toolpath)} points, {result.roughing_toolpath.get_total_length():.0f}mm")
    print(f"  Finishing: {len(result.finishing_toolpath)} points, {result.finishing_toolpath.get_total_length():.0f}mm")
    print()

    # Step 5: Save G-code
    print(f"Saving G-code to {output_file}...")
    result.gcode.save(output_file)
    print(f"  Generated {len(result.gcode)} lines of G-code")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
