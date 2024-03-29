from abc import ABC, abstractmethod
from firedrake import *
from firedrake.petsc import *
import numpy as np

from mpi4py import MPI
from pyop2.mpi import COMM_WORLD

from functools import partial
import pprint
import sys
from datetime import datetime


class StokesSolver(ABC):
    """Abstract until function_space is implemented."""

    @PETSc.Log.EventDecorator("StokesSolver:init")
    def __init__(
        self,
        # problem
        problem,
        k=2,
        nref=0,
        hierarchy="bary",
        # solver
        k_coarsening_schedule=None,  # None = hMG
        solver_type="phmg",
        # hMG
        gmg_iters=1,
        h_patch=None,
        h_patch_composition="additive",
        h_smoothing=None,
        h_patch_entity=None,
        h_distinct_smoothup=False,
        h_patch_ksp_type="chebyshev",
        h_patch_ksp_chebyshev_esteig=None,
        h_bary_th=None,
        # pMG
        p_patch=None,
        p_patch_composition="additive",
        p_smoothing=None,
        p_patch_entity=None,
        p_distinct_smoothup=False,
        p_patch_ksp_type="chebyshev",
        p_patch_ksp_chebyshev_esteig=None,
        # other
        hierarchy_callback=None,
        rebalance_vertices=False,
        high_accuracy=True,
        warmup_run=True,
        log_file=None,
    ):
        if solver_type in ["lu", "none"]:
            assert k_coarsening_schedule is None, (
                "Coarsening schedule is not used in  %s solver." % solver_type
            )
        elif solver_type == "hmg":
            assert (
                p_smoothing is None
            ), "For hmg you only need to specify h_* parameters."
            assert k_coarsening_schedule is None, (
                "Coarsening schedule is not used in  %s solver." % solver_type
            )
        elif solver_type == "afbf":
            assert k_coarsening_schedule in {"direct", "gradual", "none", None}, (
                "Invalid coarsening schedule %s for phMG" % k_coarsening_schedule
            )

        elif solver_type == "phmg":
            for patch, entity, ksp_type, mg_type in zip(
                [p_patch, h_patch],
                [p_patch_entity, h_patch_entity],
                [p_patch_ksp_type, h_patch_ksp_type],
                ["p", "h"],
            ):
                assert patch in {"patch_pc", "asm_vanka_pc", "asm_star_pc", None}, (
                    "Invalid h_patch type %s" % patch
                )
                assert ksp_type in {
                    "chebyshev",
                    "fgmres",
                }, f"Invalid {mg_type}_patch_ksp_type"

                if patch == "asm_star_pc":
                    assert (
                        entity is not None
                    ), f"Invalid {mg_type}_patch_entity for asm_star_pc"

            assert k_coarsening_schedule in {"direct", "gradual", "gradual_sv", None}, (
                "Invalid coarsening schedule %s for phMG" % k_coarsening_schedule
            )
        else:
            raise Exception("Invalid solver type %s" % solver_type)

        assert hierarchy in {"uniform", "bary", "uniformbary", "mixed"}, (
            "Invalid mesh hierarchy type %s" % hierarchy
        )

        # problem
        self._problem = problem
        self.hierarchy = hierarchy
        self.nref = nref
        self.k = k
        # solver
        self.k_coarsening_schedule = k_coarsening_schedule
        self.solver_type = solver_type
        # hMG
        self.gmg_iters = gmg_iters
        self.h_patch = h_patch
        self.h_patch_composition = h_patch_composition
        self.h_smoothing = h_smoothing
        self.h_patch_entity = h_patch_entity
        self.h_distinct_smoothup = h_distinct_smoothup
        self.h_patch_ksp_type = h_patch_ksp_type
        self.h_patch_ksp_chebyshev_esteig = h_patch_ksp_chebyshev_esteig
        self.h_bary_th = h_bary_th
        # pMG
        self.p_patch = p_patch
        self.p_patch_composition = p_patch_composition
        self.p_smoothing = p_smoothing
        self.p_patch_entity = p_patch_entity
        self.p_patch_ksp_type = p_patch_ksp_type
        self.p_distinct_smoothup = p_distinct_smoothup
        self.p_patch_ksp_chebyshev_esteig = p_patch_ksp_chebyshev_esteig
        # data collection
        self.warmup_run = warmup_run
        self.high_accuracy = high_accuracy
        self.log_file = log_file
        self._petsc_profiler_dict = {}
        self.log_file = "results.log"

        def rebalance(dm, i):
            if rebalance_vertices:
                try:
                    dm.rebalanceSharedPoints(useInitialGuess=False, parallel=False)
                except:
                    warning(
                        "Vertex rebalancing in serial from scratch failed on level %i"
                        % i
                    )
                try:
                    dm.rebalanceSharedPoints(useInitialGuess=True, parallel=True)
                except:
                    warning(
                        "Vertex rebalancing from initial guess failed on level %i" % i
                    )

        def before(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i + 1)

        def after(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i + 2)
            rebalance(dm, i)

        mh = problem.mesh_hierarchy(
            hierarchy, nref, (before, after), self.distribution_parameters()
        )

        if hierarchy_callback is not None:
            mh = hierarchy_callback(mh)
        self.parallel = mh[0].comm.size > 1
        self.tdim = mh[0].topological_dimension()
        self.mh = mh

        nu = Constant(1.0)
        self.nu = nu
        self.char_L = problem.char_length()
        self.char_U = problem.char_velocity()
        self.nu.assign(self.char_L * self.char_U)

        """
        Re = self.char_L*self.char_U / nu
        self.Re = Re
        self.advect = Constant(0)
        """

        mesh = mh[-1]

        self.mesh = mesh
        Z = self.function_space(mesh, k)
        self.Z = Z

        # compute volume for zero-mean pressure
        ztmp = Function(Z, name="tmp")
        (_, p) = ztmp.subfunctions
        p.assign(1)
        self.area = assemble(p * dx)

        comm = mesh.mpi_comm()
        Zdim = self.Z.dim()
        size = comm.size
        Vdim = self.Z.sub(0).dim()
        z = Function(Z, name="Solution")
        z.subfunctions[0].rename("Velocity")
        z.subfunctions[1].rename("Pressure")
        self.z = z
        (u, p) = split(z)
        (v, q) = split(TestFunction(Z))

        bcs = problem.bcs(Z)
        nsp = problem.nullspace(Z)
        if nsp is not None and solver_type == "lu":
            """Pin the pressure because LU sometimes fails for the saddle
            point problem with a nullspace"""
            bcs.append(DirichletBC(Z.sub(1), Constant(0), None))
            if Z.mesh().comm.rank == 0:
                bcs[-1].nodes = np.asarray([0])
            else:
                bcs[-1].nodes = np.asarray([], dtype=np.int64)
            self.nsp = None
        else:
            self.nsp = nsp

        rhs = problem.rhs(Z)
        F = self.residual()
        if rhs is not None:
            F -= inner(rhs[0], v) * dx + inner(rhs[1], q) * dx

        self.problem = NonlinearVariationalProblem(F, z, bcs=bcs)

        self.message(
            f"{self._name} Problem has been set up", "sanity_check.log", stdio=False
        )

        ########################################
        # Solver
        self.message(f"Setting up the solver..", "sanity_check.log", stdio=False)
        # solver object
        self.solver_params = self.get_parameters()
        # dump solver parameters and respective ksp solvers info to a log-file
        self.message(
            pprint.pformat(self.solver_params), "sanity_check.log", stdio=False
        )
        self.solver_params.update({"ksp_view": "ascii:ksp.log"})
        self.appctx = {
            "nu": self.nu,
            "gamma": 0,  # alfi.solver.DGMassInv needs gamma
            "grid_sizes": [],  # firedrake.preconditioners.ASMPatchPC
            "patch_sizes": [],  # firedrake.preconditioners.ASMPatchPC
        }
        self.solver = NonlinearVariationalSolver(
            self.problem,
            solver_parameters=self.solver_params,
            nullspace=self.nsp,
            appctx=self.appctx,
        )

    @abstractmethod
    def function_space(self, mesh, k):
        raise NotImplementedError

    def residual(self):
        u, p = split(self.z)
        v, q = TestFunctions(self.Z)
        F = (
            self.nu * inner(2 * sym(grad(u)), grad(v)) * dx
            - p * div(v) * dx
            - div(u) * q * dx
        )
        return F

    def _solve(self, name="", petsc_prof=False):
        self.z.assign(0)

        COMM_WORLD.Barrier()  # sync before timings
        start = datetime.now()
        try:
            with PETSc.Log.Stage(f"{name} Solve: k={self.k} nref={self.nref}"):
                self.solver.solve()
                if petsc_prof:
                    self._petsc_profiler_dict[name] = self._get_petsc_timers()
        except Exception as e:
            if COMM_WORLD.rank == 0:
                import traceback

                tb_str = traceback.format_exc()
                detailed_message = f"{name} solve failed due to: {str(e)}. Please check the input conditions and solver configurations.\nTraceback: {tb_str}"
                self.message(detailed_message, self.log_file, stdio=True)
            else:
                pass
        stop = datetime.now()
        COMM_WORLD.Barrier()

        self.message(f"{name} run complete.", "sanity_check.log", stdio=False)
        return (stop - start).total_seconds()

    def _get_petsc_timers(self, setup=False):
        """
        Composed with the help of
        https://github.com/wence-/composable-solvers/blob/f37763bedf04fb5f6efefb538ebff4b01ec42030/poisson- weak-scale.py#L51
        alfi, and Scott M.'ssuggestions.
        """
        solver = self.solver
        comm = self.Z.mesh().comm

        if not setup:
            events = [
                "SNESSolve",
                "KSPSolve",
                "ASMPatchPCApply",
                "PCSetUp",
                "PCApply",
                "KSPSolve_FS_0",
                "KSPSolve_FS_Low",
                "KSPSolve_FS_Schu",
                #
                "MatMult",
                "MatSolve",
                "MatMultAdd",  # prolong
                "MatMultTranspose",  # restrict
                "MatResidual",  # relays the calls to MatMult
                "VecMDot",  # Chebyshev?
                "VecScatterBegin",
                "VecScatterEnd",
                "MatSolve",  # LU solve in patches and coarse-grid
            ]
            patch_name = "PatchPC" if self.h_patch == "patch_pc" else "ASMPatch"
            rlx_events = [
                f"{patch_name}_size={size}" for size in self.appctx["grid_sizes"]
            ]
            """ The last set of events provided level-wise relaxation cost and
            requires the following changes additions to firedrake.preconditioners.asm.py

            initialize()
                self.prob_size = P.size[0]
                self.is_built = False
                self.appctx = self.get_appctx(pc)
                self.appctx['grid_sizes'] = self.appctx.get('grid_sizes', []) + [self.prob_size]
                self.appctx["patch_sizes"] = self.appctx.get('patch_sizes', []) + [i.size for i in ises]
            apply()
                if not self.is_built:
                    self.is_built = True
                    with PETSc.Log.Event(f"ASMPatch0_size={self.prob_size}"):
                        self.asmpc.apply(x, y)
                else:
                    with PETSc.Log.Event(f"ASMPatch_size={self.prob_size}"):
                        self.asmpc.apply(x, y)

            """
        else:
            events = ["StokesSolver:init"]

        perf = dict((e, PETSc.Log.Event(e).getPerfInfo()) for e in events)
        perf_reduced = {}
        for k, v in perf.items():
            for kk, vv in v.items():
                if kk in ["count", "time", "flops"]:
                    perf_reduced[f"{k}({kk})"] = (
                        comm.allreduce(vv, op=MPI.SUM) / comm.size
                    )

        rlx_dict = {}
        if not setup:
            perf_rlx = {}
            perf = dict((e, PETSc.Log.Event(e).getPerfInfo()) for e in rlx_events)
            for k, v in perf.items():
                for kk, vv in v.items():
                    if kk in ["count", "time", "flops"]:
                        perf_rlx[f"{k}({kk})"] = (
                            comm.allreduce(vv, op=MPI.SUM) / comm.size
                        )
            # Process to extract and sort data
            sizes = {}
            for key, value in perf_rlx.items():
                # Extract size and type (time or flops)
                parts = key.split("=")
                size = int(parts[1].split("(")[0])
                type_ = key.split("(")[1].split(")")[0]

                # Organize data by size and then by type
                if size not in sizes:
                    sizes[size] = {}
                sizes[size][type_] = value

            # Prepare the final dictionary with sorted sizes
            sorted_sizes = sorted(sizes.keys())
            rlx_dict = {
                f"{patch_name}(size)": sorted_sizes,
                f"{patch_name}(count)": [sizes[size]["count"] for size in sorted_sizes],
                f"{patch_name}(times)": [sizes[size]["time"] for size in sorted_sizes],
                f"{patch_name}(flops)": [sizes[size]["flops"] for size in sorted_sizes],
            }

        patch_size_dict = {}
        if not setup:
            if len(self.appctx["patch_sizes"]) > 0:
                patch_size_dict[f"PatchSizes"] = []
                # for each level
                for loc_ps in self.appctx["patch_sizes"]:
                    unique_values, counts = np.unique(loc_ps, return_counts=True)
                    frequency_map = dict(zip(unique_values, counts))

                    gk = comm.allgather(list(unique_values))
                    gv = comm.allgather(list(counts))

                    merged_dict = {}
                    for rank_keys, rank_values in zip(gk, gv):
                        for key, value in zip(rank_keys, rank_values):
                            if key in merged_dict:
                                merged_dict[key] += value
                            else:
                                merged_dict[key] = value
                    patch_size_dict[f"PatchSizes"].append(merged_dict)

        return {**perf_reduced, **rlx_dict, **patch_size_dict}

    def solve(self, petsc_prof=False):

        # stash setup timings
        setup_times = self._get_petsc_timers(setup=True)
        # Collect timings:
        # Start with warm-up solve followed by another solve
        t0 = 0.0
        if self.warmup_run:
            t0 = self._solve(name="Warm-up", petsc_prof=petsc_prof)
        self.solver.snes.ksp.setConvergenceHistory()
        t1 = self._solve(name="Warm", petsc_prof=petsc_prof)

        # Hardcode that pressure integral is zero
        if self.nsp is not None:
            (u, p) = self.z.subfunctions
            pintegral = assemble(p * dx)
            p.assign(p - Constant(pintegral / self.area))

        # Combine solve diagnostic results into a single dict,
        # which will be used for pada dataframe assembly.
        ksp_history = self.solver.snes.ksp.getConvergenceHistory()
        linear_its = self.solver.snes.getLinearSolveIterations()
        """
        """
        info_dict = {}  # save results here
        info_dict = {
            "order": self.k,
            "ref": self.nref,
            "ncells": self.mesh.num_cells(),
            "udofs": self.Z.sub(0).dim(),
            "pdofs": self.Z.sub(1).dim(),
            "ndofs": self.Z.sub(0).dim() + self.Z.sub(1).dim(),
            "linear_iter": linear_its,
            "overall_time(s)": t0 + t1,
            "warm_time(s)": t1,
            "resids": list(ksp_history),
            "nu": self.nu.values()[0],
            "solver_params": self.solver_params,
        }

        if petsc_prof:
            info_dict.update(setup_times)
            for name, prof_dict in self._petsc_profiler_dict.items():
                info_dict.update({f"{k}:{name}": v for k, v in prof_dict.items()})
            viewer = PETSc.Viewer().createASCII("petsc_profiler.log", "w")
            PETSc.Log.view(viewer)

        ndofs = info_dict["ndofs"]
        try:
            # if solve fails during setup this will cause issues
            rresid = ksp_history[-1] / ksp_history[0]
        except:
            rresid = [-1, -1]
        msg = f"order={self.k:2d} nref={self.nref:2d} baseN(quad={self._problem.quadrilateral})={self._problem.baseN} | ndofs={ndofs:8d} | rel_resid[{linear_its:2d}]={rresid:1.2e} | overall_time(s)={t0+t1:1.3e} warm_time(s)={t1:1.3e}"
        self.message(msg, self.log_file, stdio=True)
        self.message(msg, "sanity_check.log", stdio=False)

        return info_dict

    def plot(self, z):
        u, p = self.z.subfunctions
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # 1 row, 2 columns

        u, p = self.z.subfunctions
        # ubar = tricontourf(u, axes=axs[0])
        ubar = streamplot(u, axes=axs[0])
        pbar = tricontourf(p, axes=axs[1])

        for ax, contour, ttl in zip(axs, [ubar, pbar], ["$u$", "$p$"]):
            ax.axis("equal")
            ax.axis("off")
        fig.colorbar(contour, ax=ax)
        plt.show()

    def get_parameters(self):
        outer_base = {
            'ksp_monitor' : None,
            "petscpartitioner_type" : "parmetis",
            "snes_type": "ksponly",
            "snes_max_it": 1,
            "snes_convergence_test": "skip",
            "snes_lag_jacobian": -2,  # -2 = recompute in next newtons step but never again
            "snes_lag_preconditioner": -2,
            "ksp_type": "fgmres",
            "ksp_max_it": 80,
            # "ksp_norm" : 'unpreconditioned',
            # deal with non-convergence in post-processing
            "ksp_error_if_not_converged": False,
        }
        if self.high_accuracy:
            tolerances = {
                "ksp_rtol": 1.0e-11,
                "ksp_atol": 1.0e-15,
                "snes_rtol": 1.0e-10,
                "snes_atol": 1.0e-10,
                "snes_stol": 1.0e-10,
            }
        else:
            if self.tdim == 2:
                tolerances = {
                    "ksp_rtol": 1.0e-9,
                    "ksp_atol": 1.0e-10,
                    "snes_rtol": 1.0e-9,
                    "snes_atol": 1.0e-8,
                    "snes_stol": 1.0e-6,
                }
            else:
                tolerances = {
                    "ksp_rtol": 1.0e-8,
                    "ksp_atol": 1.0e-8,
                    "snes_rtol": 1.0e-8,
                    "snes_atol": 1.0e-8,
                    "snes_stol": 1.0e-6,
                }

        outer_base = {**outer_base, **tolerances}

        return outer_base

    def _get_relaxation_parameters(
        self,
        name,
        patch_type,
        smooth_iters,
        patch_composition,
        entity,
        ksp_type="chebyshev",
        ksp_chebyshev_esteig=None,
    ):
        """Set up a relaxation function based on patch type, composition, and entity."""
        if patch_composition != "additive":
            raise ValueError(
                "Only additive patches implemented. Modify rlx_params function for more."
            )

        from phmg.preconditioners.rlx_params import (
            pc_vanka_params,
            asm_vanka_star_pc_params,
            asm_vanka_pc_params,
        )

        rlx_param_fxn = {
            "patch_pc": pc_vanka_params,
            "asm_vanka_pc": asm_vanka_pc_params,
            "asm_star_pc": asm_vanka_star_pc_params,
        }[patch_type]

        rlx_param_fxn_wrapped = partial(
            rlx_param_fxn,
            pdegree=(self.k - 1),
            quadrilateral=self._problem.quadrilateral,
            ksp_it=smooth_iters,
            ksp_chebyshev_esteig=ksp_chebyshev_esteig,
            ksp_type=ksp_type,
            dim=self.tdim,
        )

        if patch_type == "asm_star_pc":
            rlx_param_fxn_wrapped = partial(rlx_param_fxn_wrapped, entity=entity)

        return rlx_param_fxn_wrapped()

    def message(self, msg, file_name, stdio=True):
        if COMM_WORLD.Get_rank() == 0:
            if stdio:
                print(msg)
            if file_name is not None:
                with open(file_name, "a") as file:
                    file.write(msg + "\n")


class ScottVogeliusSolver(StokesSolver):
    _name = "Scott Vogelius"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hierarchy in [
            "bary",
            "mixed",
            "r&mixed",
        ], "Invalid mesh type for Scott-Vogelius discretization. Pick bary."

    def function_space(self, mesh, k):
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k - 1)
        V = FunctionSpace(mesh, eleu)
        Q = FunctionSpace(mesh, elep)
        return MixedFunctionSpace([V, Q])

    def get_parameters(self):
        """Combine the relaxation parameters with p&h coarsening classes,
        as well as the convergence criteria.
        """
        convergence_criteria = super().get_parameters()

        outer_solver = {}
        if self.solver_type == "lu":
            outer_solver = {
                "mat_type": "aij",
                "ksp_max_it": 1,
                "ksp_convergence_test": "skip",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        elif self.solver_type == "none":
            outer_solver = {
                "mat_type": "matfree",
                "ksp_max_it": 1,
                "ksp_convergence_test": "skip",
                "pc_type": "none",
            }
        elif self.solver_type == "afbf":
            p_coarsening = {
                "gradual": "phmg.Gradual_Coarsening_Pk",
                "direct": "phmg.Direct_Coarsening_Pk",
                "none": "",
            }[self.k_coarsening_schedule]

            def get_rlx_params(patch):
                if patch == "asm_star_pc":
                    rlx_params = {
                        "pc_python_type": "firedrake.ASMStarPC",
                        "pc_type": "python",
                        "pc_star_construct_dim": 0,
                        "pc_star": {
                            "patch_precompute_element_tensors": True,
                            "patch_save_operators": True,
                            "patch_statistics": False,
                            "patch_sub_mat_type": "seqaij",
                            "patch_symmetrise_sweep": False,
                        },
                    }
                elif patch == "patch_pc":
                    rlx_params = {
                        "patch_pc_patch_construct_dim": 0,
                        "patch_pc_patch_construct_type": "star",
                        "patch_pc_patch_local_type": "additive",
                        "patch_pc_patch_partition_of_unity": False,
                        "patch_pc_patch_precompute_element_tensors": True,
                        "patch_pc_patch_save_operators": True,
                        "patch_pc_patch_statistics": False,
                        "patch_pc_patch_sub_mat_type": "seqaij",
                        "patch_pc_patch_symmetrise_sweep": False,
                        "patch_sub_ksp_type": "preonly",
                        "patch_sub_pc_factor_mat_solver_type": "petsc",
                        "patch_sub_pc_type": "lu",
                        "pc_python_type": "firedrake.PatchPC",
                    }
                else:
                    raise Exception(f"{self.p_patch} relaxation not defined")

                return rlx_params

            import alfi

            if self.k_coarsening_schedule == "none":
                # hmg
                afbf_params = {
                    "pc_type": "fieldsplit",
                    "mat_type": "nest",
                    "pc_fieldsplit_schur_factorization_type": "full",
                    "pc_fieldsplit_schur_precondition": "user",
                    "pc_fieldsplit_type": "schur",
                    "fieldsplit_0": {
                        "ksp_type": "preonly",
                        "pc_type": "mg",
                        "pc_cycle_type": "V",
                        "pc_mg_type": "multiplicative",
                        "mat_type": "aij",
                        "mg_levels": {
                            "ksp_chebyshev_esteig": "0,0.25,0,1.1",
                            "ksp_convergence_test": "skip",
                            "ksp_max_it": str(self.h_smoothing),
                            "ksp_norm_type": "unpreconditioned",
                            "ksp_type": "chebyshev",
                            **get_rlx_params(self.h_patch),
                            "pc_type": "python",
                        },
                        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                        "mg_coarse_pc_type": "python",
                        "mg_coarse_assembled": {
                            "mat_type": "aij",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                        },
                    },
                    "fieldsplit_1": {
                        "ksp_type": "preonly",
                        "pc_python_type": "alfi.solver.DGMassInv",
                        "pc_type": "python",
                    },
                }

                if self.h_distinct_smoothup:
                    afbf_params["fieldsplit_0"].update(
                        {
                            "pc_mg_distinct_smoothup": "true",
                            "mg_levels_up": {
                                "ksp_type": "richardson",
                                "ksp_max_iter": 0,
                                "ksp_richardson_scale": 0,
                                "pc_type": "none",
                            },
                        }
                    )

            else:
                afbf_params = {
                    "pc_type": "fieldsplit",
                    "mat_type": "nest",
                    "pc_fieldsplit_schur_factorization_type": "full",
                    "pc_fieldsplit_schur_precondition": "user",
                    "pc_fieldsplit_type": "schur",
                    "fieldsplit_0": {
                        "ksp_convergence_test": "skip",
                        "ksp_max_it": 1,
                        "ksp_norm_type": "unpreconditioned",
                        "ksp_richardson_self_scale": False,
                        "ksp_type": "richardson",
                        "pc_python_type": p_coarsening,
                        "pc_type": "python",
                        "pmg_mg_coarse": {
                            "degree": 2,
                            "mg_coarse_assembled": {
                                "mat_type": "aij",
                                "pc_factor_mat_solver_type": "mumps",
                                "pc_type": "lu",
                            },
                            "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                            "mg_coarse_pc_type": "python",
                            "mg_levels": {
                                "ksp_chebyshev_esteig": "0,0.25,0,1.1",
                                "ksp_convergence_test": "skip",
                                "ksp_max_it": str(self.h_smoothing),
                                "ksp_norm_type": "unpreconditioned",
                                "ksp_type": "chebyshev",
                                **get_rlx_params(self.h_patch),
                                "pc_type": "python",
                            },
                            "pc_cycle_type": "V",
                            # do not turn on if you care to see accurate PCPatch
                            # timings. The counter are off on coarse-level when
                            # this options is used
                            # "pc_mg_log": None,
                            "pc_mg_type": "multiplicative",
                            "pc_mg_multiplicative_cycles": self.gmg_iters,
                            "pc_type": "mg",
                        },
                        "pmg_mg_levels": {
                            "ksp_chebyshev_esteig": "0,0.25,0,1.1",
                            "ksp_convergence_test": "skip",
                            "ksp_max_it": str(self.p_smoothing),
                            "ksp_norm_type": "unpreconditioned",
                            "ksp_type": "chebyshev",
                            **get_rlx_params(self.p_patch),
                            "pc_type": "python",
                        },
                    },
                    "fieldsplit_1": {
                        "ksp_type": "preonly",
                        "pc_python_type": "alfi.solver.DGMassInv",
                        "pc_type": "python",
                    },
                }

                if self.p_distinct_smoothup:
                    afbf_params["fieldsplit_0"].update(
                        {
                            "pc_mg_distinct_smoothup": "true",
                            "mg_levels_up": {
                                "ksp_type": "richardson",
                                "ksp_max_iter": 0,
                                "ksp_richardson_scale": 0,
                                "pc_type": "none",
                            },
                        }
                    )
                if self.h_distinct_smoothup:
                    afbf_params["fieldsplit_0"]["pmg_mg_coarse"].update(
                        {
                            "pc_mg_distinct_smoothup": "true",
                            "mg_levels_up": {
                                "ksp_type": "richardson",
                                "ksp_max_iter": 0,
                                "ksp_richardson_scale": 0,
                                "pc_type": "none",
                            },
                        }
                    )

            outer_solver.update(afbf_params)
        elif self.solver_type == "phmg":
            # type of relaxation used dictates whether the system
            # needs to be assembled.
            if self.p_patch in ["asm_vanka_pc", "asm_star_pc"]:
                outer_solver["mat_type"] = "aij"
            else:
                outer_solver["mat_type"] = "matfree"

            p_coarsening = {
                "gradual": "phmg.Gradual_Coarsening_SVk_THkhat_TH2",
                "gradual_sv": "phmg.Gradual_Coarsening_SVk_SVkhat_TH2",
                "direct": "phmg.Direct_Coarsening_SVk_TH2",
            }[self.k_coarsening_schedule]
            # Relaxation parameter getters
            rlx_param = {
                "p": self._get_relaxation_parameters(
                    "p",
                    self.p_patch,
                    self.p_smoothing,
                    self.p_patch_composition,
                    self.p_patch_entity,
                    self.p_patch_ksp_type,
                    self.p_patch_ksp_chebyshev_esteig,
                ),
                "h": self._get_relaxation_parameters(
                    "h",
                    self.h_patch,
                    self.h_smoothing,
                    self.h_patch_composition,
                    self.h_patch_entity,
                    self.h_patch_ksp_type,
                    self.h_patch_ksp_chebyshev_esteig,
                ),
            }
            # pMG hierarchy
            pmg = {
                "pc_mg_type": "multiplicative",
                "mg_levels": rlx_param["p"],
            }
            if self.p_distinct_smoothup:
                pmg.update(
                    {
                        "pc_mg_distinct_smoothup": "true",
                        "mg_levels_up": {
                            "ksp_type": "richardson",
                            "ksp_max_iter": 0,
                            "ksp_richardson_scale": 0,
                            "pc_type": "none",
                        },
                    }
                )
            # hMG hierarchy
            gmg = {
                "ksp_type": "preonly",
                "ksp_convergence_test": "skip",
                "ksp_norm": "none",
                "pc_type": "mg",
                "pc_mg_multiplicative_cycles": self.gmg_iters,
                "mg_levels": rlx_param["h"],
                "mg_coarse": {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            }

            if self.h_distinct_smoothup:
                gmg.update(
                    {
                        "pc_mg_distinct_smoothup": "true",
                        "mg_levels_up": {
                            "ksp_type": "richardson",
                            "ksp_max_iter": 0,
                            "ksp_richardson_scale": 0,
                            "pc_type": "none",
                        },
                    }
                )

            assert self.nref + 1 == len(
                self.mh
            ), "Double check the hierarchy. It should contain an extra bary mesh"
            from copy import deepcopy

            tmp = deepcopy(rlx_param["h"])
            if self.h_bary_th is not None:
                tmp["ksp_chebyshev_esteig"] = self.h_bary_th
                gmg.update({f"mg_levels_{self.nref}": tmp})

            outer_solver.update(
                {
                    "pc_type": "python",
                    "pc_python_type": p_coarsening,
                    "ksp_view": None,
                    "pmg": pmg,
                    "pmg_mg_coarse": gmg,
                }
            )
        else:
            raise Exception("Only LU and pMG solvers are currently supported")

        return {**convergence_criteria, **outer_solver}

    def configure_patch_solver(self, opts):
        patchlu3d = "mkl_pardiso" if self.use_mkl else "umfpack"
        patchlu2d = "petsc"
        opts["patch_pc_patch_sub_mat_type"] = "seqaij"
        opts["patch_sub_pc_factor_mat_solver_type"] = (
            patchlu3d if self.tdim > 2 else patchlu2d
        )

    def distribution_parameters(self):
        return (
            {}
            if COMM_WORLD.size == 1
            else {
                "partition": True,
                "overlap_type": (DistributedMeshOverlapType.VERTEX, 3),
            }
        )


class TaylorHoodSolver(StokesSolver):
    _name = "Taylor Hood"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def function_space(self, mesh, k):
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Lagrange", mesh.ufl_cell(), k - 1)
        V = FunctionSpace(mesh, eleu)
        Q = FunctionSpace(mesh, elep)
        return MixedFunctionSpace([V, Q])

    def get_parameters(self):
        """Combine the relaxation parameters with p&h coarsening classes,
        as well as the convergence criteria.
        """
        convergence_criteria = super().get_parameters()

        outer_solver = {}
        if self.solver_type == "lu":
            outer_solver = {
                "mat_type": "aij",
                "ksp_max_it": 1,
                "ksp_convergence_test": "skip",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        elif self.solver_type == "none":
            outer_solver = {
                "mat_type": "matfree",
                "ksp_max_it": 1,
                "ksp_convergence_test": "skip",
                "pc_type": "none",
            }
        elif self.solver_type == "phmg":
            # type of relaxation used dictates whether the system
            # needs to be assembled.
            if self.p_patch in ["asm_vanka_pc", "asm_star_pc"]:
                outer_solver["mat_type"] = "aij"
            else:
                outer_solver["mat_type"] = "matfree"

            # p-coarsening schedule
            from phmg.preconditioners.pmg import (
                Direct_Coarsening_THk_TH2,
                Gradual_Coarsening_THk_TH2,
            )

            p_coarsening = {
                "gradual": "phmg.Gradual_Coarsening_THk_TH2",
                "direct": "phmg.Direct_Coarsening_THk_TH2",
            }[self.k_coarsening_schedule]
            # p_coarsenign schedules map everything to P2/P1 or Q2/Q1 discretizations.
            assert (
                self.k > 2
            ), "for pMG the fine-grid needs to be of higher-order than 2."
            # Relaxation parameter getters
            rlx_param = {
                "p": self._get_relaxation_parameters(
                    "p",
                    self.p_patch,
                    self.p_smoothing,
                    self.p_patch_composition,
                    self.p_patch_entity,
                    self.p_patch_ksp_type,
                    self.p_patch_ksp_chebyshev_esteig,
                ),
                "h": self._get_relaxation_parameters(
                    "h",
                    self.h_patch,
                    self.h_smoothing,
                    self.h_patch_composition,
                    self.h_patch_entity,
                    self.h_patch_ksp_type,
                    self.h_patch_ksp_chebyshev_esteig,
                ),
            }
            # pMG hierarchy
            pmg = {
                "pc_mg_type": "multiplicative",
                "mg_levels": rlx_param["p"],
            }
            if self.p_distinct_smoothup:
                pmg.update(
                    {
                        "pc_mg_distinct_smoothup": "true",
                        "mg_levels_up": {
                            "ksp_type": "richardson",
                            "ksp_max_iter": 0,
                            "ksp_richardson_scale": 0,
                            "pc_type": "none",
                        },
                    }
                )
            # hMG hierarchy
            gmg = {
                "ksp_type": "preonly",
                "ksp_convergence_test": "skip",
                "ksp_norm": "none",
                "pc_type": "mg",
                "pc_mg_multiplicative_cycles": self.gmg_iters,
                "mg_levels": rlx_param["h"],
                "mg_coarse": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled": {
                        # "pc_type": "svd"
                        # lu has been known to fail on singular-matrices
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",
                    },
                },
            }
            if self.h_distinct_smoothup:
                gmg.update(
                    {
                        "pc_mg_distinct_smoothup": "true",
                        "mg_levels_up": {
                            "ksp_type": "richardson",
                            "ksp_max_iter": 0,
                            "ksp_richardson_scale": 0,
                            "pc_type": "none",
                        },
                    }
                )

            outer_solver.update(
                {
                    "pc_type": "python",
                    "pc_python_type": p_coarsening,
                    "pmg": pmg,
                    "pmg_mg_coarse": gmg,
                }
            )
        elif self.solver_type == "hmg":
            # type of relaxation used dictates whether the system
            # needs to be assembled.
            if self.h_patch in ["asm_vanka_pc", "asm_star_pc"]:
                outer_solver["mat_type"] = "aij"
            else:
                outer_solver["mat_type"] = "matfree"

            # Relaxation parameter getters
            rlx_param = {
                "h": self._get_relaxation_parameters(
                    "h",
                    self.h_patch,
                    self.h_smoothing,
                    self.h_patch_composition,
                    self.h_patch_entity,
                    self.h_patch_ksp_type,
                    self.h_patch_ksp_chebyshev_esteig,
                ),
            }
            # hMG hierarchy
            size = self.mesh.mpi_comm().size
            if True: #size == 1:
                mg_coarse_assembled = {
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }
            else:
                # causes stagnating convergence for high-discretization orders
                if size > 32:
                    telescope_factor = round(size / 32.0)
                else:
                    telescope_factor = 1
                mg_coarse_assembled = {
                    "pc_type": "telescope",
                    "pc_telescope_reduction_factor": telescope_factor,
                    "pc_telescope_subcomm_type": "contiguous",
                    "telescope_pc_type": "lu",
                    "telescope_pc_factor_mat_solver_type": "superlu_dist",
                }
            gmg = {
                "pc_type": "mg",
                "pc_mg_multiplicative_cycles": self.gmg_iters,
                "mg_levels": rlx_param["h"],
                "mg_coarse": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "mat_type": "aij",
                    "assembled": mg_coarse_assembled,
                },
            }
            outer_solver.update(gmg)
        else:
            raise Exception("Only LU and pMG solvers are currently supported")

        return {**convergence_criteria, **outer_solver}

    def configure_patch_solver(self, opts):
        patchlu3d = "mkl_pardiso" if self.use_mkl else "umfpack"
        patchlu2d = "petsc"
        opts["patch_pc_patch_sub_mat_type"] = "seqaij"
        opts["patch_sub_pc_factor_mat_solver_type"] = (
            patchlu3d if self.tdim > 2 else patchlu2d
        )

    def distribution_parameters(self):
        return (
            {}
            if COMM_WORLD.size == 1
            else {
                "partition": True,
                "overlap_type": (
                    DistributedMeshOverlapType.VERTEX,
                    2 if self.solver_type == "hmg" else 3,
                ),
            }
        )
