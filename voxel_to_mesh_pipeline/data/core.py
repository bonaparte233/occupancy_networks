import os
import logging
import yaml
import numpy as np
import torch
from torch.utils import data


logger = logging.getLogger(__name__)


def collate_remove_none(batch):
    """Collate function that removes None values."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    """Worker initialization function."""
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


class Shapes3dDataset(data.Dataset):
    """3D Shapes dataset class.
    
    Args:
        dataset_folder (str): path to dataset folder
        fields (dict): dictionary of fields
        split (str): dataset split
        categories (list): list of categories
        no_except (bool): whether to ignore exceptions
        transform (callable): transform function
    """

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None):
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories if 
                         os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {c: {'idx': i, 'name': c} 
                           for i, c in enumerate(categories)}

        # Get all models
        models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            if split is not None:
                split_file = os.path.join(subpath, split + '.lst')
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        models_c = f.read().split('\n')
                else:
                    models_c = [d for d in os.listdir(subpath) 
                              if os.path.isdir(os.path.join(subpath, d))]
            else:
                models_c = [d for d in os.listdir(subpath) 
                          if os.path.isdir(os.path.join(subpath, d))]

            models_c = [{'category': c, 'model': m} for m in models_c if m != '']
            models.extend(models_c)

        self.models = models

    def __len__(self):
        """Return length of dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """Return an item of the dataset.

        Args:
            idx (int): ID of data point
        """
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, c_idx)
            except Exception as e:
                if self.no_except:
                    logger.warning(
                        'Error occurred when loading field %s of model %s: %s'
                        % (field_name, model, str(e))
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        """Get model dictionary for given index."""
        return self.models[idx]

    def test_model_complete(self, category, model):
        """Test if model is complete."""
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warning('Field "%s" is incomplete: model %s (%s)'
                             % (field_name, model, category))
                return False

        return True


class VoxelDataset(data.Dataset):
    """Simple voxel dataset for single file processing.
    
    Args:
        voxel_file (str): path to voxel file
        transform (callable): transform function
    """
    
    def __init__(self, voxel_file, transform=None):
        self.voxel_file = voxel_file
        self.transform = transform
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        """Load voxel data."""
        if idx != 0:
            raise IndexError("VoxelDataset only has one item")
            
        try:
            import binvox_rw
        except ImportError:
            raise ImportError("binvox_rw is required. Install with: pip install binvox-rw")
            
        with open(self.voxel_file, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)
        
        data = {
            'inputs': torch.from_numpy(voxels),
            'idx': 0
        }
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
