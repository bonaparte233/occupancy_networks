#!/usr/bin/env python3
"""
Command Line Interface for Voxel2Mesh
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from voxel2mesh import Voxel2Mesh
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're running from the voxel2mesh directory")
    sys.exit(1)


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert voxel data to 3D meshes using marching cubes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use data structure (recommended)
  python cli.py my_voxel.npy --verbose
  
  # Specify exact paths
  python cli.py input.binvox output.off --verbose
  
  # Custom threshold
  python cli.py my_voxel.npy --threshold 0.3 --verbose
        """,
    )

    parser.add_argument(
        "input", type=str, help="Input voxel file (.binvox, .npy, .npz) - can be relative to data/input or absolute path"
    )

    parser.add_argument(
        "--output", type=str, default=None, help="Output mesh file (.off, .ply, .obj, etc.) - if not specified, uses data/output structure"
    )
    
    parser.add_argument(
        "--use-data-structure", action="store_true", help="Use data/input and data/output folder structure"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for marching cubes (default: 0.5)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        print("Voxel2Mesh - Convert voxel data to 3D meshes")
        print("=" * 50)

    # Determine if using data structure or traditional approach
    use_data_structure = args.use_data_structure or args.output is None
    
    if not use_data_structure:
        # Traditional approach - check input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist")
            sys.exit(1)

        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize converter
        if args.verbose:
            print("Initializing Voxel2Mesh...")

        converter = Voxel2Mesh(threshold=args.threshold)

        # Convert voxels to mesh
        if use_data_structure:
            # Use new data structure interface
            if args.verbose:
                print(f"Processing '{args.input}' using data structure...")
            
            result = converter.process_voxel_file(args.input, create_visualizations=True)
            mesh = result['mesh']
            
            print("Successfully converted voxels to mesh!")
            print(f"Output directory: {result['output_dir']}")
            print(f"Mesh file: {result['mesh_file']}")
            if 'visualization_paths' in result:
                print("Visualizations created:")
                for vis_type, path in result['visualization_paths'].items():
                    print(f"  - {vis_type}: {path}")
        else:
            # Traditional approach
            if args.verbose:
                print(f"Converting '{args.input}' to mesh...")

            mesh = converter.convert_voxels_to_mesh(args.input, args.output)

            print("Successfully converted voxels to mesh!")
            print(f"Output saved to: {args.output}")

        print("Mesh statistics:")
        print(f"  - Vertices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.faces)}")

        if hasattr(mesh, "is_watertight"):
            print(f"  - Watertight: {mesh.is_watertight}")

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
