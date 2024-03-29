from firedrake import *
from firedrake.petsc import PETSc
import phmg
from phmg import *
import numpy as np


class ThreeDimPinchedChannelFlowProblem(StokesProblem):

    mesh_dir   = os.path.dirname(phmg.meshes.__file__) 
    mesh_path  = f'{mesh_dir}/meshes/3D/pinched_channel/pinched_channel_h_%s.msh'
    mesh_range = range(0, 12)

    def __init__(self, mesh_id, quadrilateral=False):
        super().__init__()
        self.baseN = mesh_id
        self.quadrilateral = quadrilateral
        assert not quadrilateral, 'only triangular meshes.'
        assert mesh_id in self.mesh_range, f'mesh_id outside of allowable {str(self.mesh.range)}'

    def mesh(self, distribution_parameters):
        base = Mesh(self.mesh_path % (str(self.baseN).zfill(2)))
        return base

    def bcs(self, Z):
        mesh = Z.mesh()
        dim  = mesh.geometric_dimension()

        other_val = tuple([0] * dim)
        inflow_val = [1]+[0]*(dim-1)

        inflow_id = (0,)
        other_id  = (2,)
        # side 1 = outflow = neumann bc

        bcs = [ DirichletBC(Z.sub(0), other_val, other_id),
                DirichletBC(Z.sub(0), inflow_val, inflow_id),
              ]
        return bcs

    def has_nullspace(self): return True

    def char_length(self): return 200.0


