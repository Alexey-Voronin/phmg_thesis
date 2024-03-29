from firedrake import *
from firedrake.petsc import PETSc
from phmg import *
import numpy as np


class ThreeDimLidDrivenCavityProblem(StokesProblem):
    def __init__(self, baseN, hexahedral):
        super().__init__()
        self.baseN = baseN
        self.quadrilateral = hexahedral

    def mesh(self, distribution_parameters):
        base = BoxMesh(self.baseN, self.baseN, 
                       self.baseN, 
                       2, 2, 2,
                       hexahedral=self.quadrilateral,
                        distribution_parameters=distribution_parameters,
                       )
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(0), Constant((0., 0., 0.)), [1, 2, 3, 5, 6])]
        return bcs

    def has_nullspace(self): return True

    def driver(self, domain):
        (x, y, z) = SpatialCoordinate(domain)
        driver = as_vector([x*x*(2-x)*(2-x)*z*z*(2-z)*(2-z)*(0.25*y*y), 0, 0])
        return driver

    def char_length(self): return 2.0

    def relaxation_direction(self): return "0+:1-"

