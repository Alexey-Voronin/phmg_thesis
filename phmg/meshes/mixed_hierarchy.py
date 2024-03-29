from firedrake import *
import firedrake
from pyop2.datatypes import IntType
from mpi4py import MPI
from fractions import Fraction
from firedrake.cython import mgimpl as impl
from firedrake.cython.dmcommon import FACE_SETS_LABEL
from firedrake.cython.mgimpl import get_entity_renumbering
from firedrake.petsc import *

import numpy as np

from .alfi_bary import bary


def UniBaryMeshHierarchy(
    mesh,
    refinement_levels,
    refinements_per_level=1,
    reorder=None,
    distribution_parameters=None,
    callbacks=None,
    mesh_builder=firedrake.Mesh,
):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh.

    :arg mesh: the coarse :func:`~.Mesh` to refine
    :arg refinement_levels: the number of levels of refinement
    :arg refinements_per_level: the number of refinements for each
        level in the hierarchy.
    :arg distribution_parameters: options controlling mesh
        distribution, see :func:`~.Mesh` for details.  If ``None``,
        use the same distribution parameters as were used to
        distribute the coarse mesh, otherwise, these options override
        the default.
    :arg reorder: optional flag indicating whether to reorder the
         refined meshes.
    :arg callbacks: A 2-tuple of callbacks to call before and
        after refinement of the DM.  The before callback receives
        the DM to be refined (and the current level), the after
        callback receives the refined DM (and the current level).
    :arg mesh_builder: Function to turn a DM into a ``Mesh``. Used by pyadjoint.
    """
    cdm = mesh.topology_dm
    cdm.setRefinementUniform(True)
    dms = []
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError(
            "Cannot refine parallel overlapped meshes "
            "(make sure the MeshHierarchy is built immediately after the Mesh)"
        )
    parameters = {}
    if distribution_parameters is not None:
        parameters.update(distribution_parameters)
    else:
        parameters.update(mesh._distribution_parameters)

    parameters["partition"] = False
    distribution_parameters = parameters

    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None

    for i in range((refinement_levels - 1) * refinements_per_level):
        if i % refinements_per_level == 0:
            before(cdm, i)
        rdm = cdm.refine()
        if i % refinements_per_level == 0:
            after(rdm, i)
        rdm.removeLabel("pyop2_core")
        rdm.removeLabel("pyop2_owned")
        rdm.removeLabel("pyop2_ghost")

        dms.append(rdm)
        cdm = rdm
        # Fix up coords if refining embedded circle or sphere
        if hasattr(mesh, "_radius"):
            # FIXME, really we need some CAD-like representation
            # of the boundary we're trying to conform to.  This
            # doesn't DTRT really for cubed sphere meshes (the
            # refined meshes are no longer gnonomic).
            coords = cdm.getCoordinatesLocal().array.reshape(
                -1, mesh.geometric_dimension()
            )
            scale = mesh._radius / np.linalg.norm(coords, axis=1).reshape(-1, 1)
            coords *= scale

    meshes = [mesh] + [
        firedrake.Mesh(
            dm,
            dim=mesh.ufl_cell().geometric_dimension(),
            distribution_parameters=distribution_parameters,
            comm=mesh.comm,
            reorder=reorder,
        )
        for dm in dms
    ]

    meshes += [
        firedrake.Mesh(
            bary(meshes[-1].topology_dm),
            dim=mesh.ufl_cell().geometric_dimension(),
            distribution_parameters=distribution_parameters,
            comm=mesh.comm,
            reorder=reorder,
        )
    ]

    lgmaps = []
    for i, m in enumerate(meshes):
        no = impl.create_lgmap(m.topology_dm)
        m.init()
        o = impl.create_lgmap(m.topology_dm)
        m.topology_dm.setRefineLevel(i)
        lgmaps.append((no, o))

    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(
        zip(meshes[:-1], meshes[1:]), zip(lgmaps[:-1], lgmaps[1:])
    ):
        if len(coarse_to_fine_cells) < len(lgmaps) - 2:
            # print('hi uniform')
            c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        else:
            # print('hi bary')
            c2f, f2c = coarse_to_fine_cells2(coarse, fine, clgmaps, flgmaps)

        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    coarse_to_fine_cells = dict(
        (Fraction(i, refinements_per_level), c2f)
        for i, c2f in enumerate(coarse_to_fine_cells)
    )
    fine_to_coarse_cells = dict(
        (Fraction(i, refinements_per_level), f2c)
        for i, f2c in enumerate(fine_to_coarse_cells)
    )

    return HierarchyBase(
        meshes,
        coarse_to_fine_cells,
        fine_to_coarse_cells,
        refinements_per_level,
        nested=True,
    )


def coarse_to_fine_cells2(mc, mf, clgmaps, flgmaps):
    """Return a map from (renumbered) cells in a coarse mesh to those
    in a refined fine mesh.

    :arg mc: the coarse mesh to create the map from.
    :arg mf: the fine mesh to map to.
    :arg clgmaps: coarse lgmaps (non-overlapped and overlapped)
    :arg flgmaps: fine lgmaps (non-overlapped and overlapped)
    :returns: Two arrays, one mapping coarse to fine cells, the second fine to coarse cells.

    Change Log: needed to modify nref - # of cells a single cell gets refined into.
    """
    # cdef:
    #     PETSc.DM cdm, fdm
    #     PetscInt cStart, cEnd, c, val, dim, nref, ncoarse
    #     PetscInt i, ccell, fcell, nfine
    #     np.ndarray[PetscInt, ndim=2, mode="c"] coarse_to_fine
    #     np.ndarray[PetscInt, ndim=2, mode="c"] fine_to_coarse
    #     np.ndarray[PetscInt, ndim=1, mode="c"] co2n, fn2o, idx

    cdm = mc.topology_dm
    fdm = mf.topology_dm
    dim = cdm.getDimension()
    nref = dim + 1  # 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size

    # print(f'dim={dim}, nref={nref}, ncoarse={ncoarse}, nfine={nfine}')

    co2n, _ = get_entity_renumbering(cdm, mc._cell_numbering, "cell")
    _, fn2o = get_entity_renumbering(fdm, mf._cell_numbering, "cell")

    # print('co2n:\n', co2n)
    # print('fn2o:\n', fn2o)

    coarse_to_fine = np.full((ncoarse, nref), -1, dtype=PETSc.IntType)
    fine_to_coarse = np.full((nfine, 1), -1, dtype=PETSc.IntType)
    # Walk owned fine cells:
    cStart, cEnd = 0, nfine

    if mc.comm.size > 1:
        cno, co = clgmaps
        fno, fo = flgmaps
        # Compute global numbers of original cell numbers
        fo.apply(fn2o, result=fn2o)
        # Compute local numbers of original cells on non-overlapped mesh
        fn2o = fno.applyInverse(fn2o, PETSc.LGMap.MapMode.MASK)
        # Need to permute k of co2n so it maps from non-overlapped
        # cells to new cells (these may have changed k).  Need to
        # map all known cells through.
        idx = np.arange(mc.cell_set.total_size, dtype=PETSc.IntType)
        # LocalToGlobal
        co.apply(idx, result=idx)
        # GlobalToLocal
        # Drop values that did not exist on non-overlapped mesh
        idx = cno.applyInverse(idx, PETSc.LGMap.MapMode.DROP)
        co2n = co2n[idx]

    for c in range(cStart, cEnd):
        # get original (overlapped) cell number
        fcell = fn2o[c]
        # The owned cells should map into non-overlapped cell numbers
        # (due to parallel growth strategy)
        assert 0 <= fcell < cEnd

        # Find original coarse cell (fcell / nref) and then map
        # forward to renumbered coarse cell (again non-overlapped
        # cells should map into owned coarse cells)
        ccell = co2n[fcell // nref]
        assert 0 <= ccell < ncoarse
        fine_to_coarse[c, 0] = ccell
        for i in range(nref):
            if coarse_to_fine[ccell, i] == -1:
                coarse_to_fine[ccell, i] = c
                break

    return coarse_to_fine, fine_to_coarse


if __name__ == "__main__":
    from firedrake import *
    import matplotlib.pyplot as plt

    for ref in range(1, 4):
        cmesh = UnitSquareMesh(2, 2, quadrilateral=False)
        mh = UniBaryMeshHierarchy(cmesh, ref)

        #####################################################################
        # Visualize
        nplots = len(mh)
        fig, axs = plt.subplots(1, nplots, figsize=(5.5 * nplots, 5))
        for i, m in enumerate(mh):
            triplot(m, axes=axs[i])
        plt.show()

        #####################################################################
        # Test in context of phMG (this will break if grid-maps
        # (coarse_to_fine_cells) were not setup correctly.
        class MixedPMG(PMGPC):
            def coarsen_element(self, ele):
                lst = [PMGPC.coarsen_element(self, sub) for sub in ele.sub_elements()]
                ME = MixedElement(lst)
                # print(ele,'--->\n',ME)
                return ME

        order = 8
        mesh = mh[-1]
        V = FunctionSpace(mesh, "CG", order)
        Z = MixedFunctionSpace([V, V])

        z = Function(Z)
        E = 0.5 * inner(grad(z), grad(z)) * dx - inner(Constant((1, 1)), z) * dx
        F = derivative(E, z, TestFunction(Z))

        bcs = [
            DirichletBC(Z.sub(0), 0, "on_boundary"),
            DirichletBC(Z.sub(1), 0, "on_boundary"),
        ]

        relax = {
            "ksp_type": "chebyshev",
            "pc_type": "jacobi",
            "ksp_max_it": 3,
        }

        pmg = {
            "pc_mg_type": "multiplicative",
            "mg_levels": relax,
            "mg_coarse_ksp_type": "richardson",
            "mg_coarse_ksp_max_it": 1,
        }

        gmg = {
            "pc_type": "mg",
            # relaxation
            "mg_levels": relax,
            # coarse-grid
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "python",
            "mg_coarse_pc_python_type": "firedrake.AssembledPC",
            "mg_coarse_assembled_pc_type": "svd",
        }

        solver_params = {
            "snes_monitor": None,
            "snes_type": "ksponly",
            "ksp_type": "fgmres",
            "ksp_monitor": None,
            # PMG
            "mat_type": "aij",
            "pc_type": "python",
            "pc_python_type": __name__ + ".MixedPMG",
        }

        solver_params.update({"pmg_" + k: v for k, v in pmg.items()})
        solver_params.update({"pmg_mg_coarse_" + k: v for k, v in gmg.items()})

        problem = NonlinearVariationalProblem(F, z, bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_params)
        solver.solve()
