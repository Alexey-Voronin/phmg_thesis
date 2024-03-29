from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import COMM_WORLD
from phmg import *
import numpy as np
import pandas as pd
from phmg.driver import collect_data
import sys,os
sys.path.append(os.path.abspath("../../"))
from params import get_params
sys.path.append(os.path.abspath("../../../problems"))
from iterators import ldc3d_iterator

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        k = sys.argv[1]
        r = sys.argv[2]
        p_sweeps = sys.argv[3]
        h_sweeps = sys.argv[4]
        ncycles  = sys.argv[5]
        directory = f'r{r}_k{k.zfill(2)}_V{ncycles}_p{p_sweeps}_h{h_sweeps}'

        if COMM_WORLD.rank == 0:
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)
    else:
        raise Exception("Provide number of relaxation sweeps.")

    input_args = get_params(k, r, p_sweeps, h_sweeps, ncycles)
    parser  = get_default_parser()
    args    = parser.parse_args(input_args)

    problem_fn = lambda k : ldc3d_iterator(k, args.quadrilateral, ref=r)
    results = collect_data(problem_fn, args)

    df = pd.DataFrame(results)
    if COMM_WORLD.rank == 0:
        print(df)
        df.to_pickle('results.pkl')
