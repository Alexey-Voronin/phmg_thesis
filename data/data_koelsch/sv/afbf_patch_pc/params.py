
def get_params(p_sweeps, h_sweeps, h_cycles, mesh_type="structured"):
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
                  '--k-coarsening-schedule', 'direct',
                  '--solver-type', 'afbf',
                  #pMG relaxation
                  '--p-patch', 'patch_pc',
                  '--p-patch-composition', 'additive',
                  '--p-smoothing', str(p_sweeps),
                  #hMG relaxation
                  '--gmg-iters', str(h_cycles),
                  '--h-patch', 'patch_pc',
                  '--h-patch-composition', 'additive',
                  '--h-smoothing', str(h_sweeps),
                  # other
                  '--warmup-run', 'true',
                  '--petsc-timings', 'true',
                  '--log-file', 'output.log'
                  ]
