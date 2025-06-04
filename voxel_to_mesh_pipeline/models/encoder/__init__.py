from .voxels import VoxelEncoder, CoordVoxelEncoder

encoder_dict = {
    'voxel_simple': VoxelEncoder,
    'voxel_coord': CoordVoxelEncoder,
}
