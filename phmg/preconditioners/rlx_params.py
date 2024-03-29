from .vanka_star import ASMVankaStarPC


def pc_vanka_params(
    pdegree,
    quadrilateral,
    ksp_it,
    ksp_type="chebyshev",
    ksp_chebyshev_esteig=None,
    dim=2
):
    # PCPATCH Paper: https://arxiv.org/pdf/1912.08516.pdf
    # Vanka params from Scott's PR:
    # https://github.com/firedrakeproject/firedrake/blob/9d5b1826e3df13dc1dacc2954fd148b868134c2a/tests/regression/test_star_pc.py

    patch_solver = lambda dim: {
        # setup
        "pc_python_type": "firedrake.PatchPC",
        "patch_pc_patch_save_operators": True,
        "patch_pc_patch_partition_of_unity": False,
        "patch_pc_patch_sub_mat_type": "seqdense",
        # Topological decomposition
        "patch_pc_patch_construct_dim": dim,
        "patch_pc_patch_construct_type": "vanka",
        "patch_pc_patch_local_type": "additive",
        # this excludes from the patch pressure pdegrees of freedom (in the subspace indexed by 1)
        # other than that at the vertex around which the patch is built,
        # ensuring that each patch contains exactly one pressure pdegree of freedom.
        "patch_pc_patch_exclude_subspaces": "1",
        # solve
        "patch_sub_ksp_type": "preonly",
        "patch_sub_pc_type": "lu",
        "patch_sub_pc_factor_shift_type": "nonzero",
    }

    params = {
        "ksp_type": ksp_type,
        "ksp_convergence_test": "skip",
        "ksp_max_it": ksp_it,
    }

    if ksp_type == "chebyshev":
        lb = 0.5**dim
        ksp_chebyshev_esteig = (
            f"0,{lb},0,1.1" if ksp_chebyshev_esteig is None else ksp_chebyshev_esteig
        )
        params["ksp_chebyshev_esteig"] = ksp_chebyshev_esteig

    if pdegree == 1:
        params.update({"pc_type": "python"})
        params.update(patch_solver(0))  # vertex
    elif pdegree == 2 and not quadrilateral:
        # quad meshes have face DoFs, so go to else.
        params.update(
            {
                "pc_type": "composite",
                "pc_composite_type": "additive",
                "pc_composite_pcs": "python,python",
                "sub_0": patch_solver(0),  # vertex
                "sub_1": patch_solver(1),  # edge
            }
        )
    else:
        params = {
            "pc_type": "composite",
            "pc_composite_type": "additive",
            "pc_composite_pcs": "python,python,python",
            "sub_0": patch_solver(0),  # vertex
            "sub_1": patch_solver(1),  # edge
            "sub_2": patch_solver(2),  # face
        }

    return params


def asm_vanka_star_pc_params(
    pdegree,
    quadrilateral,
    ksp_it,
    entity,
    ksp_type="chebyshev",
    ksp_chebyshev_esteig=None,
    dim=2,
):
    """
    Scott M, sent it via slack.
    """

    patch_solver = lambda dim: {
        "pc_python_type": __name__ + ".ASMVankaStarPC",
        "pc_vankastar_construct_dim": dim,
        "pc_vankastar_exclude_subspaces": "1",
        "pc_vankastar_sub_sub_pc_factor_shift_type": "nonzero",
    }

    params = {
        "ksp_type": ksp_type,
        "ksp_convergence_test": "skip",
        "ksp_max_it": ksp_it,
        "pc_type": "python",
    }

    if ksp_type == "chebyshev":
        lb = 0.5**dim
        ksp_chebyshev_esteig = (
            f"0,{lb},0,1.1" if ksp_chebyshev_esteig is None else ksp_chebyshev_esteig
        )
        params["ksp_chebyshev_esteig"] = ksp_chebyshev_esteig
        # params["ksp_chebyshev_kind"] = "OPT_FOURTH"
        # params["ksp_chebyshev_esteig_noisy"] = True # default value True
        # params["ksp_chebyshev_esteig_steps"] = 40   # default value 10
        # params["ksp_view"] = None
    if type(entity) is int:
        entity = (entity,)

    # Same as in pc_vanka we want to support composite patches.
    if len(entity) == 1:
        params["pc_type"] = "python"
        params.update(patch_solver(entity[0]))
    else:
        params["pc_type"] = "composite"
        for i, e in enumerate(entity):
            params[f"sub_{i}"] = patch_solver(e)

    return params


def asm_vanka_pc_params(
    pdegree,
    quadrilateral,
    ksp_it,
    ksp_type="chebyshev",
    ksp_chebyshev_esteig=None,
    dim=2
):
    patch_solver = lambda dim: {
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMVankaPC",
        "pc_vanka_construct_dim": dim,
        "pc_vanka_exclude_subspaces": "1",
        "pc_vanka_sub_sub_pc_factor_shift_type": "nonzero",
    }

    params = {
        "ksp_type": ksp_type,
        "ksp_convergence_test": "skip",
        "ksp_max_it": ksp_it,
    }

    if ksp_type == "chebyshev":
        lb = 0.5**dim
        ksp_chebyshev_esteig = (
            f"0,{lb},0,1.1" if ksp_chebyshev_esteig is None else ksp_chebyshev_esteig
        )
        params["ksp_chebyshev_esteig"] = ksp_chebyshev_esteig

    if pdegree == 1:
        params.update({"pc_type": "python"})
        params.update(patch_solver(0))  # vertex
    elif pdegree == 2 and not quadrilateral:
        # quad meshes have face DoFs, so go to else.
        params.update(
            {
                "pc_type": "composite",
                "pc_composite_type": "additive",
                "pc_composite_pcs": "python,python",
                "sub_0": patch_solver(0),  # vertex
                "sub_1": patch_solver(1),  # edge
            }
        )
    else:
        params = {
            "pc_type": "composite",
            "pc_composite_type": "additive",
            "pc_composite_pcs": "python,python,python",
            "sub_0": patch_solver(0),  # vertex
            "sub_1": patch_solver(1),  # edge
            "sub_2": patch_solver(2),  # face
        }

    return params
