# Data

## Collection

- Data is organized by clusters: e.g., `data_koelsch` and `data_delta`.
- Includes Scott-Vogelius (SV) and Taylor-Hood (TH) discretization.
- Directories `sv` and `th` contain subdirectories for each solver/preconditioner combination.

```
th (discretization)
└── hmg (preconditioner type)
    ├── ldc2d (problem type)
    │   ├── parallel_p8 (parallel run with 8 MPI tasks)
    │   │   ├── V2 (V(2,2)-cycle)
    │   │   │   ├── ksp.log (PETSc's KSP log)
    │   │   │   ├── petsc_profiler.log (PETSc's stage profileri output)
    │   │   │   ├── results.log (summary of convergence and timing results)
    │   │   │   ├── results.pkl (pd.DataFram containing more detailed data) 
    │   │   │   └── sanity_check.log (solver parameter dictionary fed to PETSc)
    │   │   ├── V4
    │   │   │   ├── ksp.log
    │   │   │   ├── petsc_profiler.log
    │   │   │   ├── results.log
    │   │   │   ├── results.pkl
    │   │   │   └── sanity_check.log
    │   │   └── run.sh
    │   ├── run.sh
    │   ├── serial
    │   │   └── run.sh
    │   └── solver.py
    ├── params.py (preconditioner parameters shared by all solvers in ldc2d)
    └── run.sh
```

## [Data Visualization](plot_data)

This section contains Jupyter notebooks and scripts for data processing.

**Note**: Run-times are measured using "KSPSolve(time):Warm", a more accurate metric that excludes various setup costs, rather than warm-timings in `results.log`.

