#!/usr/bin/env python3
"""
Point Cloud to G-code Workflow Example.

This example demonstrates processing 3D scan data:
1. Load a point cloud (PLY, PCD, XYZ)
2. Reconstruct mesh using Poisson reconstruction
3. Clean and prepare the mesh
4. Generate toolpaths
5. Export G-code

Usage:
    python point_cloud_workflow.py scan.ply output.nc
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pycam3d import CAMPipeline, CAMJob, Tool
from pycam3d.mesh import ReconstructionMethod


def main():
    if len(sys.argv) < 2:
        print("Usage: python point_cloud_workflow.py <scan.ply> [output.nc]")
        print("\nSupported formats: PLY, PCD, XYZ, PTS")
        print("\nReconstruction methods:")
        print("  - Poisson (default): Best for dense, uniform point clouds")
        print("  - Ball Pivoting: Good for point clouds with varying density")
        print("  - Alpha Shape: Fast but may have holes")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_suffix(".nc")

    print(f"Point Cloud: {input_file}")
    print(f"Output: {output_file}")
    print()

    # Initialize pipeline
    pipeline = CAMPipeline()

    # Step 1: Load point cloud and reconstruct mesh
    print("Loading point cloud and reconstructing mesh...")
    print("  (This may take a while for large point clouds)")

    pipeline.load_point_cloud(
        input_file,
        method=ReconstructionMethod.POISSON,
        depth=9,  # Higher = more detail but slower
    )

    # Step 2: Prepare mesh
    print("Preparing mesh...")
    stats = pipeline.prepare_mesh(
        repair=True,
        center=True,
        place_on_bed=True,
        smooth_iterations=2,  # Light smoothing for scan noise
    )

    mesh_stats = pipeline.get_mesh_stats()
    print(f"  Vertices: {mesh_stats['vertex_count']:,}")
    print(f"  Faces: {mesh_stats['face_count']:,}")
    print()

    # Optional: Save the reconstructed mesh
    reconstructed_path = input_file.with_stem(input_file.stem + "_reconstructed").with_suffix(".stl")
    print(f"Saving reconstructed mesh to {reconstructed_path}...")
    pipeline.export_mesh(reconstructed_path)

    # Step 3: Generate toolpaths
    job = CAMJob.default_3d_surfacing()

    print("Generating toolpaths...")
    result = pipeline.run(job)

    # Step 4: Save G-code
    print(f"Saving G-code to {output_file}...")
    result.gcode.save(output_file)

    print()
    print("Done!")
    print(f"  Reconstructed mesh: {reconstructed_path}")
    print(f"  G-code: {output_file}")


if __name__ == "__main__":
    main()
