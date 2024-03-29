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
from iterators import bfs2d_iterator

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        h_sweeps = sys.argv[1]
        directory = f'V_h{h_sweeps}'

        if not os.path.exists(directory):
            os.makedirs(directory)

        os.chdir(directory)
    else:
        raise Exception("Provide number of relaxation sweeps.")

    input_args = get_params(h_sweeps, mesh_type="unstructured")
    parser  = get_default_parser()
    args    = parser.parse_args(input_args)

    problem_fn = lambda k : bfs2d_iterator(k, ref_mesh=True)
    results = collect_data(problem_fn, args)

    df = pd.DataFrame(results)
    if COMM_WORLD.rank == 0:
        print(df)
        df.to_pickle('results.pkl')
