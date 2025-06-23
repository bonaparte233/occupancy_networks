"""
Voxel2Mesh: A simplified tool for converting voxel data to 3D meshes
using pre-trained Occupancy Networks.

Extracted and modernized from the original Occupancy Networks project:
https://github.com/autonomousvision/occupancy_networks
"""

__version__ = "1.0.0"
__author__ = "Extracted from Occupancy Networks"

try:
    from .voxel2mesh import Voxel2Mesh
    from .evaluation import MeshEvaluator
    from .visualization import (
        visualize_voxels,
        visualize_mesh,
        visualize_comparison,
        create_output_dir,
    )
except ImportError:
    # Fallback for direct script execution
    from voxel2mesh import Voxel2Mesh
    from evaluation import MeshEvaluator
    from visualization import (
        visualize_voxels,
        visualize_mesh,
        visualize_comparison,
        create_output_dir,
    )

__all__ = [
    "Voxel2Mesh",
    "MeshEvaluator",
    "visualize_voxels",
    "visualize_mesh",
    "visualize_comparison",
    "create_output_dir",
]
