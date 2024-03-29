
def get_params(h_sweeps, mesh_type="structured"):
    return [# problem params
                  '--baseN', '100',
                  '--quadrilateral',  'false',
                  '--mh',             'mixed',
                  '--discretization', 'sv',
                  '--nref-start', '1',
                  '--nref-end', '4',
                  '--k-start', '3',
                  '--k-end', '11',
                  # solver params
                  '--k-coarsening-schedule', 'none', # hmg
                  '--solver-type', 'afbf',
                  #hMG relaxation
                  '--h-patch', 'patch_pc',
                  '--h-patch-composition', 'additive',
                  '--h-smoothing', str(h_sweeps),
                  # other
                  '--warmup-run',  'true',
                  '--petsc-timings',  'true',
                  '--log-file', 'output.log'
                  ]
