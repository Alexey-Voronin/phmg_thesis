from phmg.solver import TaylorHoodSolver, ScottVogeliusSolver
from pyop2.mpi import COMM_WORLD
from firedrake.petsc import PETSc
from firedrake import *

import os
import shutil
import gc

from time import time


def str2bool(value):
    # get_default_parser helper function
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def int_or_int_list(value):
    # get_default_parser helper function
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If it fails, assume it's a list of integers separated by commas
        return [int(i) for i in value.split(",")]


def get_default_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    ###########################################################
    # Problem object arguments
    ###########################################################
    # mesh arguments
    parser.add_argument("--quadrilateral", type=str2bool, default=True)
    parser.add_argument(
        "--diagonal", type=str, default=None, choices=["left", "right", "crossed"]
    )
    ###########################################################
    # Solver object atguments
    ###########################################################
    # Solver mesh arguments
    parser.add_argument("--nref", type=int, default=None)
    parser.add_argument("--baseN", type=int, default=8)
    parser.add_argument(
        "--mh",
        type=str,
        default="uniform",
        choices=["uniform", "bary", "uniformbary", "mixed"],
    )
    # discretization
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--discretization", type=str, default="th", choices=["th", "sv"]
    )

    # solver parameters
    parser.add_argument(
        "--k-coarsening-schedule",
        type=str,
        default=None,
        choices=["none", "direct", "gradual", "gradual_sv"],
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="lu",
        choices=["none", "lu", "hmg", "phmg", "afbf"],
    )

    parser.add_argument(f"--gmg-iters", type=int, default=1)

    for mg_type in ["h", "p"]:
        parser.add_argument(
            f"--{mg_type}-patch",
            type=str,
            default=None,
            choices=["patch_pc", "asm_vanka_pc", "asm_star_pc"],
        )
        parser.add_argument(
            f"--{mg_type}-patch-composition",
            type=str,
            default="additive",
            choices=["additive", "multiplicative"],
        )
        parser.add_argument(f"--{mg_type}-smoothing", type=int, default=None)

        # allow for int or list of ints input
        # e.g., `--h-patch-entity 0 1 2`
        parser.add_argument(
            f"--{mg_type}-patch-entity", type=int_or_int_list, nargs="+", default=None
        )
        parser.add_argument(
            f"--{mg_type}-patch-ksp-type",
            type=str,
            default="chebyshev",
            choices=["chebyshev", "fgmres"],
        )
        parser.add_argument(
            f"--{mg_type}-distinct-smoothup",
            type=str2bool,
            default=False,
        )
        parser.add_argument(
            f"--{mg_type}-patch-ksp-chebyshev-esteig", type=str, default=None
        )
        parser.add_argument(f"--{mg_type}-bary-th", type=str, default=None)

    parser.add_argument(
        "--high-accuracy", dest="high_accuracy", default=True, action="store_true"
    )

    # data collection
    parser.add_argument("--nref-start", type=int, default=None)
    parser.add_argument("--nref-end", type=int, default=None)
    parser.add_argument("--k-start", type=int, default=None)
    parser.add_argument("--k-end", type=int, default=None)
    parser.add_argument("--petsc-timings", type=str2bool, default=False)
    parser.add_argument("--warmup-run", type=str2bool, default=True)
    parser.add_argument("--log-file", type=str, default=None)

    # parser.add_argument("--clear", dest="clear", default=False,
    #                    action="store_true")
    parser.add_argument(
        "--paraview", dest="paraview", default=False, action="store_true"
    )
    # parallel options
    parser.add_argument(
        "--rebalance", dest="rebalance", default=False, action="store_true"
    )

    return parser


def get_solver(problem, args, hierarchy_callback=None):
    solver_t = {"th": TaylorHoodSolver, "sv": ScottVogeliusSolver}[args.discretization]
    solver = solver_t(
        # problem
        problem,
        k=args.k,
        nref=args.nref,
        hierarchy=args.mh,
        # solver
        k_coarsening_schedule=args.k_coarsening_schedule,
        solver_type=args.solver_type,
        # hMG
        gmg_iters=args.gmg_iters,
        h_patch=args.h_patch,
        h_patch_composition=args.h_patch_composition,
        h_smoothing=args.h_smoothing,
        h_patch_entity=args.h_patch_entity,
        h_distinct_smoothup=args.h_distinct_smoothup,
        h_patch_ksp_type=args.h_patch_ksp_type,
        h_patch_ksp_chebyshev_esteig=args.h_patch_ksp_chebyshev_esteig,
        h_bary_th=args.h_bary_th,
        # pMG
        p_patch=args.p_patch,
        p_patch_composition=args.p_patch_composition,
        p_smoothing=args.p_smoothing,
        p_patch_entity=args.p_patch_entity,
        p_distinct_smoothup=args.p_distinct_smoothup,
        p_patch_ksp_type=args.p_patch_ksp_type,
        p_patch_ksp_chebyshev_esteig=args.p_patch_ksp_chebyshev_esteig,
        # other
        rebalance_vertices=args.rebalance,
        warmup_run=args.warmup_run,
        high_accuracy=args.high_accuracy,
        hierarchy_callback=hierarchy_callback,
        log_file=args.log_file,
    )
    return solver

def collect_data(problem_fn, args):
    tic = time()
    # Obtain mesh refinement ranges
    nref_start = args.nref_start
    nref_end = args.nref_end
    if args.nref is not None:
        assert (
            args.nref_start is None and args.nref_end is None
        ), "Specify either --nref or (--nref_start, --nref_end), not both."
        nref_start = args.nref_start
        nref_end = nref_start + 1
        args.nref_start = args.nref_end = None
    else:
        assert (
            args.nref_start is not None and args.nref_end is not None
        ), "Specify either --nref or (--nref_start, --nref_end), not both."

    # Obtain discretization order refinement ranges
    k_start = args.k_start
    k_end = args.k_end
    if args.k is not None:
        assert (
            args.k_start is None and args.k_end is None
        ), "Specify either --k or (--k-start, --k-end), not both."
        k_start = args.k_start
        k_end = k_start + 1
        args.k_start = args.k_end = None
    else:
        args.nref_start is not None and args.nref_end is not None, "Specify either --nref or (--nref_start, --nref_end), not both."

    if args.petsc_timings:
        PETSc.Log.begin()

    # Collect data
    results = []
    for k in range(k_start, k_end)[::-1]:
        for nref in range(nref_start, nref_end):
            # update parameters (kludge but works)
            args.k = k
            args.nref = nref

            solver = get_solver(problem_fn(k), args)
            dd = solver.solve(petsc_prof=args.petsc_timings)
            results.append(dd)

            solver.solver.snes.destroy()
            del solver._problem
            del solver.problem
            del solver.solver
            del solver

    total_time = time() - tic
    if COMM_WORLD.Get_rank() == 0:
        from phmg.solver import StokesSolver

        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        msg = f"Overall run-time: {hours}h {minutes}min {seconds}s"
        StokesSolver.message("", msg, "results.log", stdio=True)

    gc.collect()

    return results
