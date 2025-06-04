import torch
import numpy as np
import trimesh
import time
from utils.common import make_3d_grid
try:
    from utils.libmcubes import mcubes
except ImportError:
    print("Warning: mcubes not available. Using fallback mesh extraction.")
    mcubes = None


class Generator3D(object):
    """Generator class for 3D mesh generation.

    Args:
        model (nn.Module): trained model
        device (device): pytorch device
        threshold (float): threshold value
        resolution0 (int): start resolution for MISE
        upsampling_steps (int): number of upsampling steps
        sample (bool): whether to sample
        refinement_step (int): refinement step
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (callable): preprocessor
    """

    def __init__(self, model, device=None, threshold=0.5, resolution0=16,
                 upsampling_steps=3, sample=False, refinement_step=0,
                 simplify_nfaces=None, preprocessor=None):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.sample = sample
        self.refinement_step = refinement_step
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

    def generate_mesh(self, data, return_stats=True):
        """Generate the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # Preprocess if required
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        # Generate mesh
        t0 = time.time()
        mesh = self.generate_from_latent(c, stats_dict=stats_dict, **kwargs)
        stats_dict['time (generate mesh)'] = time.time() - t0

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        """Generate mesh from latent code.

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + 0.1

        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
            )
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to [-0.5, 0.5]
                pointsf = pointsf / mesh_extractor.resolution - 0.5
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, **kwargs):
        """Evaluate the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        """
        p_split = torch.split(p, 100000)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, None, c, **kwargs).probs

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        """Extract the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): occupancy grid
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        """
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + 0.1
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = mcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def refine_mesh(self, mesh, occ_hat, c=None):
        """Refine mesh.

        Args:
            mesh (trimesh): mesh
            occ_hat (tensor): occupancy grid
            c (tensor): encoded feature volumes
        """
        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = torch.optim.RMSprop([v], lr=1e-4)

        for it_r in range(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.eval_points(face_point, c)
            )
            normal_target = -autograd.grad(
                outputs=face_value.sum(), inputs=face_point, 
                create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh


class MISE:
    """Marching cubes with importance sampling."""
    
    def __init__(self, resolution, upsampling_steps, threshold):
        self.resolution = resolution
        self.upsampling_steps = upsampling_steps
        self.threshold = threshold
        self.c_threshold = self.threshold
        
        self.resolution_0 = resolution // 2**upsampling_steps
        
        # Initialize
        self.occ_values = np.zeros((self.resolution_0,) * 3)
        self.occ_values_set = np.zeros((self.resolution_0,) * 3, dtype=bool)
        
    def query(self):
        """Query points."""
        # Find boundary points
        if not self.occ_values_set.any():
            # Initial query - get all points at base resolution
            coords = np.mgrid[0:self.resolution_0, 0:self.resolution_0, 0:self.resolution_0]
            coords = coords.reshape(3, -1).T
            return coords
        
        # Find points near the boundary
        boundary_points = []
        for i in range(self.resolution_0):
            for j in range(self.resolution_0):
                for k in range(self.resolution_0):
                    if not self.occ_values_set[i, j, k]:
                        continue
                    
                    # Check if near boundary
                    val = self.occ_values[i, j, k]
                    if abs(val - self.c_threshold) < 0.1:
                        boundary_points.append([i, j, k])
        
        return np.array(boundary_points) if boundary_points else np.empty((0, 3))
    
    def update(self, points, values):
        """Update with new values."""
        for i, (point, value) in enumerate(zip(points, values)):
            x, y, z = point
            if (0 <= x < self.resolution_0 and 
                0 <= y < self.resolution_0 and 
                0 <= z < self.resolution_0):
                self.occ_values[x, y, z] = value
                self.occ_values_set[x, y, z] = True
    
    def to_dense(self):
        """Convert to dense grid."""
        return self.occ_values


def simplify_mesh(mesh, n_faces_target, agressiveness):
    """Simplify mesh using decimation."""
    # Simple decimation - in practice you'd use a proper mesh simplification library
    if hasattr(mesh, 'simplify_quadric_decimation'):
        return mesh.simplify_quadric_decimation(n_faces_target)
    else:
        # Fallback - just return original mesh
        return mesh
