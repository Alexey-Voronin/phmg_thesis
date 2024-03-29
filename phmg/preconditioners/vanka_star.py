from firedrake import *
from firedrake.petsc import PETSc

distribution_parameters = {
    "partition": True,
    "overlap_type": (DistributedMeshOverlapType.VERTEX, 2),
}

import petsc4py

petsc4py.PETSc.Sys.popErrorHandler()

import numpy as np


def order_points(mesh_dm, points, ordering_type, prefix):
    """Order a the points (topological entities) of a patch based on the adjacency graph of the mesh.
    :arg mesh_dm: the `mesh.topology_dm`
    :arg points: array with point indices forming the patch
    :arg ordering_type: a `PETSc.Mat.OrderingType`
    :arg prefix: the prefix associated with additional ordering options

    :returns: the permuted array of points
    """

    if ordering_type == "natural":
        return points
    subgraph = [
        numpy.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1]
        for p in points
    ]
    ia = numpy.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
    ja = numpy.concatenate(subgraph).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ(
        (len(points),) * 2,
        csr=(ia, ja, numpy.ones(ja.shape, PETSc.RealType)),
        comm=PETSc.COMM_SELF,
    )
    A.setOptionsPrefix(prefix)
    rperm, _ = A.getOrdering(ordering_type)
    A.destroy()

    return points[rperm.getIndices()]


class ASMVankaStarPC(ASMPatchPC):
    """Patch-based PC using closure of star of mesh entities implemented as an
    :class:`ASMPatchPC`.

    ASMVankaStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the closure of the star of the mesh entity
    specified by `pc_vanka_construct_dim` (or codim).

    This version includes the star of the "exclude_subspaces" in the patch
    """

    _prefix = "pc_vankastar_"

    @PETSc.Log.EventDecorator("ASMVankaStarGetPatches")
    def get_patches(self, V):
        mesh = V._mesh
        mesh_dm = mesh.topology_dm
        if mesh.layers:
            warning("applying ASMVankaPC on an extruded mesh")

        # Obtain the topological entities to use to construct the stars
        depth = PETSc.Options().getInt(self.prefix + "construct_dim", default=-1)
        height = PETSc.Options().getInt(self.prefix + "construct_codim", default=-1)
        if (depth == -1 and height == -1) or (depth != -1 and height != -1):
            raise ValueError(
                f"Must set exactly one of {self.prefix}construct_dim or {self.prefix}construct_codim"
            )

        exclude_subspaces = [
            int(subspace)
            for subspace in PETSc.Options()
            .getString(self.prefix + "exclude_subspaces", default="-1")
            .split(",")
        ]
        ordering = PETSc.Options().getString(
            self.prefix + "mat_ordering_type", default="natural"
        )
        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = []
        for i, W in enumerate(V):
            V_local_ises_indices.append(V.dof_dset.local_ises[i].indices)

        # Build index sets for the patches
        ises = []
        if depth != -1:
            (start, end) = mesh_dm.getDepthStratum(depth)
        else:
            (start, end) = mesh_dm.getHeightStratum(height)

        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                continue

            # Create point list from mesh DM
            star, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
            pt_array_star = order_points(mesh_dm, star, ordering, self.prefix)

            pt_array_vanka = set()
            for pt in star.tolist():
                closure, _ = mesh_dm.getTransitiveClosure(pt, useCone=True)
                pt_array_vanka.update(closure.tolist())

            pt_array_vanka = order_points(
                mesh_dm, pt_array_vanka, ordering, self.prefix
            )
            # Get DoF indices for patch
            indices = []
            for i, W in enumerate(V):
                section = W.dm.getDefaultSection()
                if i in exclude_subspaces:
                    loop_list = pt_array_star
                else:
                    loop_list = pt_array_vanka
                for p in loop_list:
                    dof = section.getDof(p)
                    if dof <= 0:
                        continue
                    off = section.getOffset(p)
                    # Local indices within W
                    W_indices = slice(off * W.value_size, W.value_size * (off + dof))
                    indices.extend(V_local_ises_indices[i][W_indices])
            iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            ises.append(iset)

        return ises
