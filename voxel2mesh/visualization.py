"""
Visualization utilities for Voxel2Mesh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_voxels(voxels, output_path=None, title="Voxel Data", show=False):
    """Visualize 3D voxel data.

    Args:
        voxels (np.ndarray): 3D voxel array
        output_path (str): Path to save the visualization
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    voxels = np.asarray(voxels)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # For visualization, transpose to match expected orientation
    voxels_vis = voxels.transpose(2, 0, 1)

    # Create voxel plot
    ax.voxels(voxels_vis > 0.5, edgecolor="k", alpha=0.7, facecolors="red")

    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    ax.set_title(title)
    ax.view_init(elev=30, azim=45)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Voxel visualization saved to: {output_path}")

    if show:
        plt.show()

    if not show:
        plt.close(fig)


def visualize_mesh(
    mesh,
    output_path=None,
    title="Generated Mesh",
    show=False,
    color="lightblue",
    alpha=0.8,
):
    """Visualize 3D mesh.

    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
        output_path (str): Path to save the visualization
        title (str): Plot title
        show (bool): Whether to display the plot
        color (str): Mesh color
        alpha (float): Mesh transparency
    """
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print("Warning: Empty mesh, cannot visualize")
        return

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot mesh
    vertices = mesh.vertices
    faces = mesh.faces

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=color,
        alpha=alpha,
        edgecolor="none",
    )

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                vertices[:, 0].max() - vertices[:, 0].min(),
                vertices[:, 1].max() - vertices[:, 1].min(),
                vertices[:, 2].max() - vertices[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}\n{len(vertices)} vertices, {len(faces)} faces")
    ax.view_init(elev=30, azim=45)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Mesh visualization saved to: {output_path}")

    if show:
        plt.show()

    if not show:
        plt.close(fig)


def visualize_comparison(
    voxels, mesh, output_path=None, title="Voxel to Mesh Conversion", show=False
):
    """Visualize voxels and generated mesh side by side.

    Args:
        voxels (np.ndarray): Input voxel data
        mesh (trimesh.Trimesh): Generated mesh
        output_path (str): Path to save the visualization
        title (str): Plot title
        show (bool): Whether to display the plot
    """
    fig = plt.figure(figsize=(16, 8))

    # Plot voxels
    ax1 = fig.add_subplot(121, projection="3d")
    voxels_vis = voxels.transpose(2, 0, 1)
    ax1.voxels(voxels_vis > 0.5, edgecolor="k", alpha=0.7, facecolors="red")
    ax1.set_title("Input Voxels")
    ax1.set_xlabel("Z")
    ax1.set_ylabel("X")
    ax1.set_zlabel("Y")
    ax1.view_init(elev=30, azim=45)

    # Plot mesh
    ax2 = fig.add_subplot(122, projection="3d")

    if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
        vertices = mesh.vertices
        faces = mesh.faces

        ax2.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            color="lightblue",
            alpha=0.8,
            edgecolor="none",
        )

        # Set equal aspect ratio
        max_range = (
            np.array(
                [
                    vertices[:, 0].max() - vertices[:, 0].min(),
                    vertices[:, 1].max() - vertices[:, 1].min(),
                    vertices[:, 2].max() - vertices[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)

        ax2.set_title(f"Generated Mesh\n{len(vertices)} vertices, {len(faces)} faces")
    else:
        ax2.text(
            0.5,
            0.5,
            0.5,
            "Empty Mesh",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Generated Mesh (Empty)")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=30, azim=45)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison visualization saved to: {output_path}")

    if show:
        plt.show()

    if not show:
        plt.close(fig)


def create_output_dir(base_name=None):
    """Create output directory in data/output folder."""
    if base_name is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"run_{timestamp}"

    output_dir = os.path.join("data", "output", base_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
