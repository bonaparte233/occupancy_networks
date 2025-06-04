import yaml
import os
import torch
import torch.distributions as dist
from models.encoder import encoder_dict
from models.decoder import decoder_dict
from models.onet import OccupancyNetwork
from models.generation import Generator3D
from data import (
    Shapes3dDataset, VoxelDataset, IndexField, CategoryField, 
    VoxelsField, PointsField, collate_remove_none, worker_init_fn
)


def load_config(path, default_path=None):
    """Load config file.
    
    Args:
        path (str): path to config file
        default_path (str): path to default config file
    """
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def get_model(cfg, device=None, dataset=None, **kwargs):
    """Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    """
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    decoder = decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )

    if z_dim != 0:
        # For now, we don't support latent encoders in this simplified version
        encoder_latent = None
    else:
        encoder_latent = None

    if encoder == 'idx':
        encoder = torch.nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim,
            **encoder_kwargs
        )
    else:
        encoder = None

    p0_z = get_prior_z(cfg, device)
    model = OccupancyNetwork(
        decoder, encoder, encoder_latent, p0_z, device=device
    )

    return model


def get_prior_z(cfg, device, **kwargs):
    """Return prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_generator(model, cfg, device, **kwargs):
    """Return the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    preprocessor = get_preprocessor(cfg, device=device)

    generator = Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_preprocessor(cfg, device=None, **kwargs):
    """Return the preprocessor.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    preprocessor = cfg['preprocessor']['type']
    if preprocessor is None:
        preprocessor = None
    else:
        # For now, we don't support preprocessors in this simplified version
        preprocessor = None

    return preprocessor


def get_dataset(mode, cfg, return_idx=False, return_category=False):
    """Return the dataset.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
        return_idx (bool): whether to include index in data
        return_category (bool): whether to include category in data
    """
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }
    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        fields = get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = IndexField()

        if return_category:
            fields['category'] = CategoryField()

        dataset = Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
        )
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}')

    return dataset


def get_inputs_field(mode, cfg):
    """Return the inputs field.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    input_type = cfg['data']['input_type']
    
    if input_type is None:
        inputs_field = None
    elif input_type == 'voxels':
        inputs_field = VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field


def get_data_fields(mode, cfg):
    """Return the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    fields = {}
    
    if mode in ('val', 'test'):
        fields['points_iou'] = PointsField(
            cfg['data']['points_iou_file'],
            unpackbits=cfg['data']['points_unpackbits'],
        )

    return fields


class CheckpointIO(object):
    """CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path to checkpoint directory
        model (nn.Module): model
        optimizer (optimizer): pytorch optimizer
    """

    def __init__(self, checkpoint_dir='.', model=None, optimizer=None, **kwargs):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.kwargs = kwargs

    def save(self, filename, **kwargs):
        """Save the current state.

        Args:
            filename (str): name of checkpoint file
        """
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        outdict = kwargs
        if self.model is not None:
            outdict['model'] = self.model.state_dict()
        if self.optimizer is not None:
            outdict['optimizer'] = self.optimizer.state_dict()

        torch.save(outdict, os.path.join(self.checkpoint_dir, filename))

    def load(self, filename, **kwargs):
        """Load a checkpoint.

        Args:
            filename (str): name of checkpoint file
        """
        if filename.startswith('http'):
            # Download from URL
            import urllib.request
            model_dir = os.path.join(self.checkpoint_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, os.path.basename(filename))
            
            if not os.path.exists(model_file):
                print(f'Downloading model from {filename}...')
                urllib.request.urlretrieve(filename, model_file)
                print(f'Model downloaded to {model_file}')
            
            filename = model_file

        if not os.path.exists(filename):
            raise FileNotFoundError(f'Checkpoint file not found: {filename}')

        print(f'Loading checkpoint from {filename}')
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location='cpu')

        scalars = self.parse_state_dict(checkpoint, **kwargs)
        return scalars

    def parse_state_dict(self, checkpoint, **kwargs):
        """Parse state_dict of model and return scalars.

        Args:
            checkpoint (dict): checkpoint dictionary
        """
        for key, value in self.kwargs.items():
            if key in checkpoint:
                value.load_state_dict(checkpoint[key])

        if self.model is not None and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        scalars = {k: v for k, v in checkpoint.items()
                  if k not in ('model', 'optimizer')}
        return scalars
