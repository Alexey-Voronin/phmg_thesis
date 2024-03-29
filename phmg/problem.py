from firedrake import *
from .meshes.alfi_bary import BaryMeshHierarchy
from .meshes.mixed_hierarchy import UniBaryMeshHierarchy
from abc import ABC, abstractmethod


class StokesProblem(ABC):
    """abstract class to be implemented by specific problem configuration.
    Roughly adopted from alfi."""

    @abstractmethod
    def mesh(self, distribution_parameters):
        raise NotImplementedError

    def mesh_hierarchy(self, hierarchy_type, nref, callbacks, distribution_parameters):
        baseMesh = self.mesh(distribution_parameters)

        if hierarchy_type == "uniform":
            mh = MeshHierarchy(
                baseMesh,
                nref,
                reorder=True,
                callbacks=callbacks,
                distribution_parameters=distribution_parameters,
            )
        elif hierarchy_type == "bary":
            mh = BaryMeshHierarchy(
                baseMesh,
                nref,
                callbacks=callbacks,
                reorder=True,
                distribution_parameters=distribution_parameters,
            )
        elif hierarchy_type == "uniformbary":
            # uniform refinement of barycenric mesh
            bmesh = Mesh(
                bary(baseMesh._topology_dm),
                distribution_parameters={"partition": False},
            )
            mh = MeshHierarchy(
                bmesh,
                nref,
                reorder=True,
                callbacks=callbacks,
                distribution_parameters=distribution_parameters,
            )
        elif hierarchy_type == "mixed":
            mh = UniBaryMeshHierarchy(
                baseMesh,
                nref,
                callbacks=callbacks,
                reorder=True,
                distribution_parameters=distribution_parameters,
            )
        else:
            raise NotImplementedError(
                "Only know bary, uniformbary, uniform, mixed, r&mixed for the hierarchy."
            )
        return mh

    @abstractmethod
    def bcs(self, Z):
        raise NotImplementedError

    @abstractmethod
    def has_nullspace(self):
        raise NotImplementedError

    def nullspace(self, Z):
        if self.has_nullspace():
            MVSB = MixedVectorSpaceBasis
            return MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=Z.comm)])
        else:
            return None

    def mesh_size(self, u, domain_type):
        mesh = u.ufl_domain()
        if domain_type == "facet":
            dim = u.ufl_domain().topological_dimension()
            return FacetArea(mesh) if dim == 2 else FacetArea(mesh) ** 0.5
        elif domain_type == "cell":
            return CellSize(mesh)

    def char_velocity(self):
        return 1.0

    def char_length(self):
        return 1.0

    def rhs(self, Z):
        return None

    def relaxation_direction(self):
        raise Excepotion("Not implemented")
