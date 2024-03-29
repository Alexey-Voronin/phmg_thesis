
def get_params(k, r, p_sweeps, h_sweeps, hcycles):
    k = int(k)
    r = int(r)
    return [# problem params
                  '--baseN', '100',
                  '--quadrilateral',  'false',
                  '--mh',             'mixed',
                  '--discretization', 'sv',
                  '--nref-start', str(r),
                  '--nref-end', str(r+1),
                  '--k-start', str(k),
                  '--k-end', str(k+1),
                  # solver params
                  '--k-coarsening-schedule', 'gradual',
                  '--solver-type', 'phmg',
                  #pMG relaxation
                  '--p-patch', 'asm_star_pc',
                  '--p-patch-composition', 'additive',
                  '--p-smoothing', str(p_sweeps),
                  '--p-patch-entity', '0',
                  '--p-patch-ksp-type', 'chebyshev',
                  '--p-patch-ksp-chebyshev-esteig', "0,0.125,0,1.1",
                  #hMG relaxation
                  '--gmg-iters', str(hcycles),
                  '--h-patch', 'asm_star_pc',
                  '--h-patch-composition', 'additive',
                  '--h-smoothing', str(h_sweeps),
                  '--h-patch-entity', '0',
                  '--h-patch-ksp-type', 'chebyshev',
                  '--h-patch-ksp-chebyshev-esteig', "0,0.125,0,1.1",
                  # other
                  '--warmup-run', 'true',
                  '--petsc-timings', 'true',
                  '--log-file', 'output.log'
                  ]
