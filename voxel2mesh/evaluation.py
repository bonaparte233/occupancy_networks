"""
Mesh evaluation metrics for Voxel2Mesh
Based on the original Occupancy Networks evaluation
"""

import numpy as np
import trimesh
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_MESH_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer_l1": 6,
    "chamfer_l2": 6,
    "f_score": 0.0,
}


class MeshEvaluator:
    """Mesh evaluation class based on Occupancy Networks."""

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh_pred, mesh_gt=None, pointcloud_gt=None):
        """Evaluate a predicted mesh against ground truth.

        Args:
            mesh_pred (trimesh.Trimesh): Predicted mesh
            mesh_gt (trimesh.Trimesh): Ground truth mesh (optional)
            pointcloud_gt (np.ndarray): Ground truth point cloud (optional)

        Returns:
            dict: Evaluation metrics
        """
        if (
            mesh_pred is None
            or len(mesh_pred.vertices) == 0
            or len(mesh_pred.faces) == 0
        ):
            logger.warning("Empty mesh detected!")
            return EMPTY_MESH_DICT.copy()

        # Sample points from predicted mesh
        try:
            pointcloud_pred, face_indices = mesh_pred.sample(
                self.n_points, return_index=True
            )
            pointcloud_pred = pointcloud_pred.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to sample from mesh: {e}")
            return EMPTY_MESH_DICT.copy()

        # If ground truth mesh is provided, sample from it
        if mesh_gt is not None and len(mesh_gt.vertices) > 0 and len(mesh_gt.faces) > 0:
            try:
                pointcloud_gt, _ = mesh_gt.sample(self.n_points, return_index=True)
                pointcloud_gt = pointcloud_gt.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to sample from ground truth mesh: {e}")
                pointcloud_gt = None

        # Evaluate point cloud metrics
        if pointcloud_gt is not None:
            return self.eval_pointcloud(pointcloud_pred, pointcloud_gt)
        else:
            # Return basic mesh statistics
            return self.get_mesh_stats(mesh_pred)

    def eval_pointcloud(self, pointcloud_pred, pointcloud_gt):
        """Evaluate point cloud metrics.

        Args:
            pointcloud_pred (np.ndarray): Predicted point cloud (N, 3)
            pointcloud_gt (np.ndarray): Ground truth point cloud (M, 3)

        Returns:
            dict: Evaluation metrics
        """
        if pointcloud_pred.shape[0] == 0 or pointcloud_gt.shape[0] == 0:
            logger.warning("Empty point cloud!")
            return EMPTY_MESH_DICT.copy()

        pointcloud_pred = np.asarray(pointcloud_pred)
        pointcloud_gt = np.asarray(pointcloud_gt)

        # Completeness: distance from GT to prediction
        completeness = self.distance_p2p(pointcloud_gt, pointcloud_pred)
        completeness2 = completeness**2

        completeness_mean = completeness.mean()
        completeness2_mean = completeness2.mean()

        # Accuracy: distance from prediction to GT
        accuracy = self.distance_p2p(pointcloud_pred, pointcloud_gt)
        accuracy2 = accuracy**2

        accuracy_mean = accuracy.mean()
        accuracy2_mean = accuracy2.mean()

        # Chamfer distance
        chamfer_l1 = 0.5 * (completeness_mean + accuracy_mean)
        chamfer_l2 = 0.5 * (completeness2_mean + accuracy2_mean)

        # F-Score
        f_score = self.compute_f_score(accuracy, completeness, threshold=0.01)

        return {
            "completeness": completeness_mean,
            "accuracy": accuracy_mean,
            "completeness2": completeness2_mean,
            "accuracy2": accuracy2_mean,
            "chamfer_l1": chamfer_l1,
            "chamfer_l2": chamfer_l2,
            "f_score": f_score,
        }

    def distance_p2p(self, points_src, points_tgt):
        """Compute point-to-point distances.

        Args:
            points_src (np.ndarray): Source points
            points_tgt (np.ndarray): Target points

        Returns:
            np.ndarray: Distances
        """
        # Compute pairwise distances and find minimum for each source point
        distances = cdist(points_src, points_tgt)
        min_distances = np.min(distances, axis=1)
        return min_distances

    def compute_f_score(self, accuracy, completeness, threshold=0.01):
        """Compute F-score at given threshold.

        Args:
            accuracy (np.ndarray): Accuracy distances
            completeness (np.ndarray): Completeness distances
            threshold (float): Distance threshold

        Returns:
            float: F-score
        """
        precision = np.mean(accuracy < threshold)
        recall = np.mean(completeness < threshold)

        if precision + recall == 0:
            return 0.0

        f_score = 2 * precision * recall / (precision + recall)
        return f_score

    def get_mesh_stats(self, mesh):
        """Get basic mesh statistics.

        Args:
            mesh (trimesh.Trimesh): Input mesh

        Returns:
            dict: Basic mesh statistics
        """
        stats = {
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.faces),
            "volume": 0.0,
            "surface_area": 0.0,
            "is_watertight": False,
        }

        try:
            if mesh.is_volume:
                stats["volume"] = mesh.volume
            stats["surface_area"] = mesh.area
            stats["is_watertight"] = mesh.is_watertight
        except Exception as e:
            logger.warning(f"Failed to compute mesh statistics: {e}")

        return stats
