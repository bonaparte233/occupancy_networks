import os
import numpy as np
import torch
try:
    import binvox_rw
except ImportError:
    print("Warning: binvox_rw not available. Install with: pip install binvox-rw")
    binvox_rw = None


class Field(object):
    """Base field class."""
    
    def load(self, model_path, idx, category):
        """Load data point.
        
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        raise NotImplementedError

    def check_complete(self, files):
        """Check if field is complete.
        
        Args:
            files (list): list of files
        """
        raise NotImplementedError


class IndexField(Field):
    """Index field class."""
    
    def load(self, model_path, idx, category):
        """Load index.
        
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        return idx

    def check_complete(self, files):
        """Check if field is complete."""
        return True


class CategoryField(Field):
    """Category field class."""
    
    def load(self, model_path, idx, category):
        """Load category.
        
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        return category

    def check_complete(self, files):
        """Check if field is complete."""
        return True


class VoxelsField(Field):
    """Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    """
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        """Load the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Voxel file not found: {file_path}")

        if binvox_rw is None:
            raise ImportError("binvox_rw is required for loading voxel data")

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        """Check if field is complete."""
        return self.file_name in files


class PointsField(Field):
    """Points field class.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether to apply transforms
        unpackbits (bool): whether to unpack bits
    """
    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, idx, category):
        """Load the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Points file not found: {file_path}")

        points_dict = np.load(file_path)

        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        """Check if field is complete."""
        return self.file_name in files


class MeshField(Field):
    """Mesh field class.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    """
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        """Load the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mesh file not found: {file_path}")

        mesh = trimesh.load(file_path, process=False)

        if self.transform is not None:
            mesh = self.transform(mesh)

        return mesh

    def check_complete(self, files):
        """Check if field is complete."""
        return self.file_name in files


class PointCloudField(Field):
    """Point cloud field class.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether to apply transforms
    """
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        """Load the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        file_path = os.path.join(model_path, self.file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        """Check if field is complete."""
        return self.file_name in files
