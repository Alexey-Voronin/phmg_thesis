from firedrake import *
from firedrake.petsc import PETSc
from phmg import *
import numpy as np
import pandas as pd
from phmg.driver import collect_data
import sys,os
sys.path.append(os.path.abspath("../../"))
from params import get_params
sys.path.append(os.path.abspath("../../../"))
from iterators import ldc2d_iterator

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        p_sweeps = sys.argv[1]
        h_sweeps = sys.argv[2]
        h_cycles = sys.argv[3]
        directory = f"V{h_cycles}_p{p_sweeps}_h{h_sweeps}"

        if COMM_WORLD.rank == 0:
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)
    else:
        raise Exception("Provide number of relaxation sweeps.")

    input_args = get_params(p_sweeps, h_sweeps, h_cycles)
    parser  = get_default_parser()
    args    = parser.parse_args(input_args)

    problem_fn = lambda k : ldc2d_iterator(k, args.diagonal,
                                            args.quadrilateral,
                                           solver='afbf')
    results = collect_data(problem_fn, args)

    df = pd.DataFrame(results)
    if COMM_WORLD.rank == 0:
        print(df)
        df.to_pickle('results.pkl')
