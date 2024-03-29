from firedrake import *
from firedrake.petsc import PETSc
from phmg import *
import numpy as np


class TwoDimLidDrivenCavityProblem(StokesProblem):

    def __init__(self, baseN, diagonal=None, quadrilateral=False):
        super().__init__()
        self.baseN = baseN
        self.quadrilateral = quadrilateral

        if quadrilateral:
            assert diagonal is None, "Invalid diagonal arg for quadrilateral mesh."
        elif diagonal is None:
            diagonal = "left"

        self.diagonal = diagonal


    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1,
                     originX=-1, originY=-1,
                     quadrilateral=self.quadrilateral,
                     distribution_parameters=distribution_parameters,
                     diagonal=self.diagonal)
        return base

    def bcs(self, Z):
        mesh = Z.mesh()
        dim  = mesh.geometric_dimension()
        x, y = SpatialCoordinate(mesh)

        fx   = self.driver(mesh)
        bcs  = [DirichletBC(Z.sub(0), as_vector([fx]+[0] * (dim - 1)), (4,)),
                 DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3)),
               ]
        return bcs

    def has_nullspace(self): return True

    def driver(self, domain):
        (x, y) = SpatialCoordinate(domain)
        fxn    = (x+1)*(1-x)

        return fxn

    def char_length(self): return 2.0

