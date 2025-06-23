"""
Voxel2Mesh: Convert voxel data to 3D meshes using marching cubes

This module provides a simple and reliable interface for converting voxel data to meshes
using the marching cubes algorithm.
"""

import os
import sys
import numpy as np
import trimesh
from pathlib import Path
from skimage import measure

# Add utils to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")
sys.path.insert(0, utils_dir)

try:
    from .utils import load_voxel_data, save_mesh
except ImportError:
    from utils import load_voxel_data, save_mesh


class Voxel2Mesh:
    """Simple and reliable voxel to mesh converter using marching cubes."""
    
    def __init__(self, threshold=0.5):
        """Initialize the converter.
        
        Args:
            threshold (float): Threshold for marching cubes (default: 0.5)
        """
        self.threshold = threshold
        print("Initialized Voxel2Mesh with marching cubes algorithm")
    
    def convert_voxels_to_mesh(self, voxel_input, output_path=None):
        """Convert voxel data to mesh.
        
        Args:
            voxel_input (str or ndarray): Path to voxel file or voxel array
            output_path (str): Path to save output mesh (optional)
            
        Returns:
            mesh (trimesh.Trimesh): Generated mesh
        """
        # Load voxel data
        if isinstance(voxel_input, str):
            voxels, metadata = load_voxel_data(voxel_input)
            print(f"Loaded voxels from {voxel_input}: shape {voxels.shape}")
        else:
            voxels = np.asarray(voxel_input, dtype=np.float32)
            metadata = {}
            print(f"Using provided voxel array: shape {voxels.shape}")
        
        # Generate mesh using marching cubes
        print("Generating mesh using marching cubes...")
        mesh = self._voxels_to_mesh(voxels)
        
        print(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Save mesh if output path is provided
        if output_path is not None:
            save_mesh(mesh, output_path)
        
        return mesh
    
    def _voxels_to_mesh(self, voxels):
        """Convert voxels to mesh using marching cubes.
        
        Args:
            voxels (np.ndarray): Voxel data
            
        Returns:
            trimesh.Trimesh: Generated mesh
        """
        # Ensure voxels are in the right format
        voxels = np.asarray(voxels, dtype=np.float32)
        
        print(f"Voxel shape: {voxels.shape}, occupied: {voxels.sum()}")
        
        # Pad voxels to avoid boundary issues
        voxels_padded = np.pad(voxels, 1, mode='constant', constant_values=0)
        
        try:
            # Use marching cubes
            vertices, faces, normals, values = measure.marching_cubes(
                voxels_padded, level=self.threshold, spacing=(1.0, 1.0, 1.0)
            )
            
            # Adjust vertices to account for padding and normalize to [-0.5, 0.5] space
            vertices -= 1  # Remove padding offset
            vertices = vertices / np.array(voxels.shape) - 0.5  # Normalize to [-0.5, 0.5]
            
            # Create mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            
            # Basic mesh processing
            if len(mesh.vertices) > 0:
                # Remove duplicate vertices
                mesh.merge_vertices()
                # Fix normals
                mesh.fix_normals()
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Marching cubes failed: {e}")
            return trimesh.Trimesh()
    
    def process_voxel_file(self, input_file_path, create_visualizations=True):
        """Process a single voxel file and save all outputs to data/output folder.
        
        Args:
            input_file_path (str): Path to input voxel file (relative to data/input or absolute path)
            create_visualizations (bool): Whether to create visualization images
            
        Returns:
            dict: Results containing mesh, evaluation stats, and output paths
        """
        # Handle relative paths
        if not os.path.isabs(input_file_path):
            # If relative path, assume it's in data/input
            full_input_path = os.path.join("data", "input", input_file_path)
        else:
            full_input_path = input_file_path
        
        if not os.path.exists(full_input_path):
            raise FileNotFoundError(f"Input file not found: {full_input_path}")
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(full_input_path))[0]
        
        # Create output directory
        output_dir = os.path.join("data", "output", base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing: {full_input_path}")
        print(f"Output directory: {output_dir}")
        
        # Load voxel data
        voxels, metadata = load_voxel_data(full_input_path)
        print(f"Loaded voxels: shape {voxels.shape}, occupied: {voxels.sum()}")
        
        # Generate mesh
        mesh_output_path = os.path.join(output_dir, f"{base_name}_mesh.off")
        mesh = self.convert_voxels_to_mesh(voxels, mesh_output_path)
        
        # Get mesh statistics
        try:
            from .evaluation import MeshEvaluator
            evaluator = MeshEvaluator()
            stats = evaluator.get_mesh_stats(mesh)
        except ImportError:
            from evaluation import MeshEvaluator
            evaluator = MeshEvaluator()
            stats = evaluator.get_mesh_stats(mesh)
        
        print("Mesh Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # Create visualizations if requested
        visualization_paths = {}
        if create_visualizations:
            print("Creating visualizations...")
            
            try:
                from .visualization import visualize_voxels, visualize_mesh, visualize_comparison
            except ImportError:
                from visualization import visualize_voxels, visualize_mesh, visualize_comparison
            
            # Voxel visualization
            voxel_vis_path = os.path.join(output_dir, f"{base_name}_voxels.png")
            visualize_voxels(voxels, voxel_vis_path, f"Input Voxels - {base_name}")
            visualization_paths["voxels"] = voxel_vis_path
            
            # Mesh visualization
            mesh_vis_path = os.path.join(output_dir, f"{base_name}_mesh.png")
            visualize_mesh(mesh, mesh_vis_path, f"Generated Mesh - {base_name}")
            visualization_paths["mesh"] = mesh_vis_path
            
            # Comparison visualization
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            visualize_comparison(voxels, mesh, comparison_path, f"Voxel to Mesh - {base_name}")
            visualization_paths["comparison"] = comparison_path
        
        # Save metadata and results
        import json
        results = {
            "input_file": full_input_path,
            "output_directory": output_dir,
            "mesh_file": mesh_output_path,
            "voxel_shape": list(voxels.shape),
            "voxel_occupied": float(voxels.sum()),
            "mesh_stats": stats,
            "visualization_paths": visualization_paths,
        }
        
        results_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        print(f"All outputs saved in: {output_dir}")
        
        return {
            "mesh": mesh,
            "stats": stats,
            "output_dir": output_dir,
            "mesh_file": mesh_output_path,
            "visualization_paths": visualization_paths,
            "results_file": results_path,
        }
    
    def __call__(self, voxel_input, output_path=None):
        """Convenience method for conversion."""
        return self.convert_voxels_to_mesh(voxel_input, output_path)
