import torch
import numpy as np


def make_3d_grid(bb_min, bb_max, shape):
    """Make a 3D grid.
    
    Args:
        bb_min (tuple): minimum values for each axis
        bb_max (tuple): maximum values for each axis  
        shape (tuple): shape of the grid
        
    Returns:
        torch.Tensor: 3D grid points
    """
    size = shape[0] * shape[1] * shape[2]
    
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])
    
    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    
    p = torch.stack([pxs, pys, pzs], dim=1)
    
    return p


def coordinate2index(x, reso, coord_type='2d'):
    """Convert coordinates to indices.
    
    Args:
        x (torch.Tensor): coordinates
        reso (int): resolution
        coord_type (str): coordinate type
        
    Returns:
        torch.Tensor: indices
    """
    x = (x * reso).long()
    if coord_type == '2d':
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d':
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_coordinate(p, padding=0.1, plane='xz'):
    """Normalize coordinates to [-1, 1].
    
    Args:
        p (torch.Tensor): coordinates
        padding (float): padding value
        plane (str): plane type
        
    Returns:
        torch.Tensor: normalized coordinates
    """
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane == 'xy':
        xy = p[:, :, [0, 1]]
    elif plane == 'yz':
        xy = p[:, :, [1, 2]]
    else:
        xy = p
        
    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)
    
    # f there are outliers out of the range
    xy_new[xy_new < 0] = 0
    xy_new[xy_new > 1] = 1
    return xy_new


def normalize_3d_coordinate(p, padding=0.1):
    """Normalize 3D coordinates to [-1, 1].
    
    Args:
        p (torch.Tensor): 3D coordinates
        padding (float): padding value
        
    Returns:
        torch.Tensor: normalized coordinates
    """
    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    p_nor[p_nor < 0] = 0
    p_nor[p_nor > 1] = 1
    return p_nor


def map2local(p, c_plane, unit_size, unet_kwargs):
    """Map points to local coordinate system.
    
    Args:
        p (torch.Tensor): points
        c_plane (torch.Tensor): plane features
        unit_size (float): unit size
        unet_kwargs (dict): UNet arguments
        
    Returns:
        torch.Tensor: local coordinates
    """
    xy = normalize_coordinate(p, plane=unet_kwargs['plane'])
    index = coordinate2index(xy, c_plane.shape[-1], coord_type='2d')
    
    # scatter plane features from points
    fea_plane = c_plane.view(c_plane.shape[0], c_plane.shape[1], -1)
    
    # aggregate information from plane
    c = fea_plane.gather(dim=2, index=index.expand(-1, fea_plane.shape[1], -1))
    
    return c


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.), subsample_to=None):
    """Arrange pixels in a grid.
    
    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of image coordinates
        subsample_to (int): subsample to this number of pixels
        
    Returns:
        torch.Tensor: pixel coordinates
    """
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    
    # Arrange pixel coordinates
    pixel_locations = torch.meshgrid(torch.linspace(image_range[0], image_range[1], h),
                                   torch.linspace(image_range[0], image_range[1], w))
    pixel_locations = torch.stack([pixel_locations[1], pixel_locations[0]], dim=-1).view(-1, 2)
    pixel_locations = pixel_locations.unsqueeze(0).repeat(batch_size, 1, 1)
    
    if subsample_to is not None and subsample_to < n_points:
        idx = np.random.choice(pixel_locations.shape[1], size=(subsample_to,), replace=False)
        pixel_locations = pixel_locations[:, idx]
    
    return pixel_locations


def to_pytorch(tensor, return_type=False):
    """Convert numpy array to pytorch tensor.
    
    Args:
        tensor: input tensor
        return_type (bool): whether to return type info
        
    Returns:
        torch.Tensor: pytorch tensor
    """
    is_numpy = isinstance(tensor, np.ndarray)
    if is_numpy:
        tensor = torch.from_numpy(tensor)
    if return_type:
        return tensor, is_numpy
    return tensor


def get_mask(tensor):
    """Get mask for valid values.
    
    Args:
        tensor (torch.Tensor): input tensor
        
    Returns:
        torch.Tensor: mask
    """
    return (tensor != -1).any(dim=-1)
