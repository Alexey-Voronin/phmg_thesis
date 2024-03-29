from firedrake import *
from firedrake.petsc import PETSc
from phmg import *
import phmg.meshes
import numpy as np
import os

class ThreeDimBackwardFacingStepProblem(StokesProblem):

    mesh_dir   = os.path.dirname(phmg.meshes.__file__)
    mesh_path  = f'{mesh_dir}/meshes/3D/bfs/coarse%s.msh'
    mesh_range = range(0, 9)

    def __init__(self, mesh_id, quadrilateral=False):
        super().__init__()
        self.baseN = mesh_id
        self.quadrilateral = quadrilateral
        assert not quadrilateral, 'only triangular meshes.'
        assert mesh_id in self.mesh_range, f'mesh_id outside of allowable {str(self.mesh.range)}'

    def mesh(self, distribution_parameters):
        base = Mesh(self.mesh_path % (str(self.baseN).zfill(2)))
        return base

    @staticmethod
    def poiseuille_flow(domain):
        (x, y, z) = SpatialCoordinate(domain)
        return as_vector([16*(2-y)*(y-1)*z*(1-z)*(y>1), 0, 0])

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.poiseuille_flow(Z.mesh()), 1),
               DirichletBC(Z.sub(0), Constant((0., 0., 0)), 3)]
        return bcs

    def has_nullspace(self):
        return False

    def relaxation_direction(self): return "0+:1-"

