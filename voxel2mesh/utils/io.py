"""
Input/Output utilities for Voxel2Mesh
"""

import os
import numpy as np
import trimesh
from .binvox_rw import read_as_3d_array


def load_voxel_data(input_path):
    """
    Load voxel data from various formats.

    Args:
        input_path (str): Path to voxel file (.binvox) or numpy array file (.npy, .npz)

    Returns:
        voxels (ndarray): 3D boolean or float array representing voxel occupancy
        metadata (dict): Additional metadata (scale, translate, etc.)
    """
    metadata = {}

    if input_path.endswith(".binvox"):
        # Load binvox file
        with open(input_path, "rb") as f:
            voxel_model = read_as_3d_array(f)

        voxels = voxel_model.data.astype(np.float32)
        metadata = {
            "dims": voxel_model.dims,
            "translate": voxel_model.translate,
            "scale": voxel_model.scale,
            "axis_order": voxel_model.axis_order,
        }

    elif input_path.endswith(".npy"):
        # Load numpy array
        voxels = np.load(input_path)
        if voxels.dtype == bool:
            voxels = voxels.astype(np.float32)
        metadata = {"dims": list(voxels.shape)}

    elif input_path.endswith(".npz"):
        # Load compressed numpy array
        data = np.load(input_path)
        if "voxels" in data:
            voxels = data["voxels"]
        elif "data" in data:
            voxels = data["data"]
        else:
            # Use the first array found
            key = list(data.keys())[0]
            voxels = data[key]

        if voxels.dtype == bool:
            voxels = voxels.astype(np.float32)
        metadata = {"dims": list(voxels.shape)}

        # Load additional metadata if available
        for key in ["dims", "translate", "scale", "axis_order"]:
            if key in data:
                metadata[key] = data[key]

    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    # Ensure voxels are 3D
    if voxels.ndim != 3:
        raise ValueError(f"Voxel data must be 3D, got {voxels.ndim}D")

    return voxels, metadata


def save_mesh(mesh, output_path, file_format=None):
    """
    Save mesh to file.

    Args:
        mesh (trimesh.Trimesh): Mesh to save
        output_path (str): Output file path
        file_format (str): File format ('off', 'ply', 'obj', etc.).
                          If None, inferred from file extension.
    """
    if file_format is None:
        file_format = os.path.splitext(output_path)[1][1:]  # Remove the dot

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    try:
        mesh.export(output_path, file_type=file_format)
        print(f"Mesh saved to: {output_path}")
    except Exception as e:
        print(f"Error saving mesh: {e}")
        raise


def load_mesh(input_path):
    """
    Load mesh from file.

    Args:
        input_path (str): Path to mesh file

    Returns:
        mesh (trimesh.Trimesh): Loaded mesh
    """
    try:
        mesh = trimesh.load(input_path, process=False)
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        raise


def export_pointcloud(vertices, out_file, as_text=True):
    """
    Export point cloud to PLY file.

    Args:
        vertices (ndarray): Point coordinates (N, 3)
        out_file (str): Output file path
        as_text (bool): Whether to save as text format
    """
    try:
        from plyfile import PlyElement, PlyData

        assert vertices.shape[1] == 3
        vertices = vertices.astype(np.float32)
        vertices = np.ascontiguousarray(vertices)
        vector_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        vertices = vertices.view(dtype=vector_dtype).flatten()
        plyel = PlyElement.describe(vertices, "vertex")
        plydata = PlyData([plyel], text=as_text)
        plydata.write(out_file)
        print(f"Point cloud saved to: {out_file}")

    except ImportError:
        print("Warning: plyfile not available. Saving as numpy array.")
        np.save(out_file.replace(".ply", ".npy"), vertices)


def create_voxel_mesh_for_visualization(voxels, threshold=0.5):
    """
    Create a simple mesh representation of voxels for visualization.

    Args:
        voxels (ndarray): 3D voxel array
        threshold (float): Threshold for occupied voxels

    Returns:
        mesh (trimesh.Trimesh): Voxel mesh
    """
    # Find occupied voxels
    occupied = voxels > threshold
    coords = np.where(occupied)

    if len(coords[0]) == 0:
        # Return empty mesh if no occupied voxels
        return trimesh.Trimesh()

    # Create unit cubes for each occupied voxel
    vertices_list = []
    faces_list = []

    # Unit cube vertices (relative to voxel center)
    cube_vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )

    # Unit cube faces
    cube_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 7, 6],
            [4, 6, 5],  # top
            [0, 4, 5],
            [0, 5, 1],  # front
            [2, 6, 7],
            [2, 7, 3],  # back
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ]
    )

    vertex_offset = 0
    for i, j, k in zip(coords[0], coords[1], coords[2]):
        # Translate cube to voxel position
        voxel_vertices = cube_vertices + np.array([i, j, k])
        vertices_list.append(voxel_vertices)

        # Add faces with proper vertex indexing
        voxel_faces = cube_faces + vertex_offset
        faces_list.append(voxel_faces)

        vertex_offset += 8

    if vertices_list:
        all_vertices = np.vstack(vertices_list)
        all_faces = np.vstack(faces_list)
        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    else:
        return trimesh.Trimesh()
