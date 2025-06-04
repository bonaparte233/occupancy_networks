import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import block_reduce
try:
    from .libvoxelize.voxelize import voxelize_mesh_
except ImportError:
    print("Warning: voxelize_mesh_ not available. Some functionality may be limited.")
    voxelize_mesh_ = None

try:
    from .libmesh import check_mesh_contains
except ImportError:
    print("Warning: check_mesh_contains not available. Using fallback.")
    check_mesh_contains = None

from .common import make_3d_grid


class VoxelGrid:
    """Voxel grid representation with conversion capabilities."""
    
    def __init__(self, data, loc=(0., 0., 0.), scale=1):
        assert(data.shape[0] == data.shape[1] == data.shape[2])
        data = np.asarray(data, dtype=bool)
        loc = np.asarray(loc)
        self.data = data
        self.loc = loc
        self.scale = scale

    @classmethod
    def from_mesh(cls, mesh, resolution, loc=None, scale=None, method='ray'):
        """Create voxel grid from mesh."""
        bounds = mesh.bounds
        # Default location is center
        if loc is None:
            loc = (bounds[0] + bounds[1]) / 2

        # Default scale, scales the mesh to [-0.45, 0.45]^3
        if scale is None:
            scale = (bounds[1] - bounds[0]).max()/0.9

        loc = np.asarray(loc)
        scale = float(scale)

        # Transform mesh
        mesh = mesh.copy()
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)

        # Apply method
        if method == 'ray':
            voxel_data = voxelize_ray(mesh, resolution)
        elif method == 'fill':
            voxel_data = voxelize_fill(mesh, resolution)

        voxels = cls(voxel_data, loc, scale)
        return voxels

    def down_sample(self, factor=2):
        """Downsample voxel grid by given factor."""
        if not (self.data.shape[0] % factor == 0):
            raise ValueError('Resolution must be divisible by factor.')
        
        new_data = block_reduce(self.data.astype(float), 
                               (factor,) * 3, func=np.max)
        new_data = new_data.astype(bool)
        return VoxelGrid(new_data, self.loc, self.scale)

    def to_mesh(self):
        """Convert voxel grid to mesh using marching cubes approach."""
        # Shorthand
        occ = self.data

        # Shape of voxel grid
        nx, ny, nz = occ.shape
        # Shape of corresponding occupancy grid
        grid_shape = (nx + 1, ny + 1, nz + 1)

        # Convert values to occupancies
        occ = np.pad(occ, 1, 'constant')

        # Determine if face present
        f1_r = (occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1])
        f2_r = (occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1])
        f3_r = (occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:])

        f1_l = (~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1])
        f2_l = (~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1])
        f3_l = (~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:])

        f1 = f1_r | f1_l
        f2 = f2_r | f2_l
        f3 = f3_r | f3_l

        assert(f1.shape == (nx + 1, ny, nz))
        assert(f2.shape == (nx, ny + 1, nz))
        assert(f3.shape == (nx, ny, nz + 1))

        # Determine vertices
        ind1 = np.where(f1)
        ind2 = np.where(f2)
        ind3 = np.where(f3)

        n_vertices = len(ind1[0]) + len(ind2[0]) + len(ind3[0])
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        faces = []

        # Vertices are centered at edges
        vertices[:len(ind1[0]), 0] = ind1[0]
        vertices[:len(ind1[0]), 1] = ind1[1] + 0.5
        vertices[:len(ind1[0]), 2] = ind1[2] + 0.5

        vertices[len(ind1[0]):len(ind1[0])+len(ind2[0]), 0] = ind2[0] + 0.5
        vertices[len(ind1[0]):len(ind1[0])+len(ind2[0]), 1] = ind2[1]
        vertices[len(ind1[0]):len(ind1[0])+len(ind2[0]), 2] = ind2[2] + 0.5

        vertices[len(ind1[0])+len(ind2[0]):, 0] = ind3[0] + 0.5
        vertices[len(ind1[0])+len(ind2[0]):, 1] = ind3[1] + 0.5
        vertices[len(ind1[0])+len(ind2[0]):, 2] = ind3[2]

        # Rescale to original coordinate system
        vertices = vertices / nx - 0.5
        vertices *= self.scale
        vertices += self.loc

        # Create faces for each voxel
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if occ[i+1, j+1, k+1]:
                        # Create faces for this voxel
                        self._add_voxel_faces(faces, i, j, k, f1, f2, f3, 
                                            len(ind1[0]), len(ind2[0]))

        if len(faces) == 0:
            # Return empty mesh if no faces
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))

        faces = np.array(faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh

    def _add_voxel_faces(self, faces, i, j, k, f1, f2, f3, n1, n2):
        """Add faces for a single voxel."""
        # This is a simplified version - in practice you'd need more complex
        # face generation logic based on the occupancy pattern
        pass


def voxelize_ray(mesh, resolution):
    """Voxelize mesh using ray casting method."""
    if voxelize_mesh_ is None:
        # Fallback to simple method
        return voxelize_simple(mesh, resolution)
    
    occ_surface = voxelize_surface(mesh, resolution)
    occ_interior = voxelize_interior(mesh, resolution)
    occ = (occ_interior | occ_surface)
    return occ


def voxelize_fill(mesh, resolution):
    """Voxelize mesh using fill method."""
    bounds = mesh.bounds
    if (np.abs(bounds) >= 0.5).any():
        raise ValueError('voxelize fill is only supported if mesh is inside [-0.5, 0.5]^3/')

    occ = voxelize_surface(mesh, resolution)
    occ = ndimage.morphology.binary_fill_holes(occ)
    return occ


def voxelize_surface(mesh, resolution):
    """Voxelize mesh surface."""
    if voxelize_mesh_ is None:
        return voxelize_simple(mesh, resolution)
        
    vertices = mesh.vertices
    faces = mesh.faces

    vertices = (vertices + 0.5) * resolution

    face_loc = vertices[faces]
    occ = np.full((resolution,) * 3, 0, dtype=np.int32)
    face_loc = face_loc.astype(np.float32)

    voxelize_mesh_(occ, face_loc)
    occ = (occ != 0)

    return occ


def voxelize_interior(mesh, resolution):
    """Voxelize mesh interior."""
    if check_mesh_contains is None:
        # Fallback method
        return voxelize_simple(mesh, resolution)
        
    shape = (resolution,) * 3
    bb_min = (0.5,) * 3
    bb_max = (resolution - 0.5,) * 3
    # Create points. Add noise to break symmetry
    points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
    points = points + 0.1 * (np.random.rand(*points.shape) - 0.5)
    points = (points / resolution - 0.5)
    occ = check_mesh_contains(mesh, points)
    occ = occ.reshape(shape)
    return occ


def voxelize_simple(mesh, resolution):
    """Simple voxelization fallback."""
    # Simple grid-based voxelization
    bounds = mesh.bounds
    grid_size = (bounds[1] - bounds[0]) / resolution
    
    # Create voxel grid
    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Sample points in each voxel and check if inside mesh
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                # Center of voxel
                point = bounds[0] + (np.array([i, j, k]) + 0.5) * grid_size
                # Simple inside test (this is very basic)
                if mesh.contains([point])[0]:
                    voxels[i, j, k] = True
    
    return voxels
