
def get_params(p_sweeps, h_sweeps, hcycles):
    return [# problem params
                  '--baseN', '100',
                  '--quadrilateral',  'false',
                  '--mh',             'uniform',
                  '--discretization', 'th',
                  '--nref-start', '1',
                  '--nref-end', '4',
                  '--k-start', '3',
                  '--k-end', '11',
                  # solver params
                  '--k-coarsening-schedule', 'gradual',
                  '--solver-type', 'phmg',
                  #pMG relaxation
                  '--p-patch', 'asm_star_pc',
                  '--p-patch-composition', 'additive',
                  '--p-smoothing', str(p_sweeps),
                  '--p-patch-entity', '0',
                  #hMG relaxation
                  '--gmg-iters', str(hcycles),
                  '--h-patch', 'asm_star_pc',
                  '--h-patch-composition', 'additive',
                  '--h-smoothing', str(h_sweeps),
                  '--h-patch-entity', '0',
                  # other
                  '--warmup-run', 'true',
                  '--petsc-timings', 'true',
                  '--log-file', 'output.log'
                  ]
