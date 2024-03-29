
def get_params(k, r, p_sweeps, h_sweeps, h_cycles, mesh_type="structured"):
    r = int(r)
    k = int(k)
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
                  '--k-coarsening-schedule', 'direct',
                  '--solver-type', 'afbf',
                  #pMG relaxation
                  '--p-patch', 'patch_pc',
                  '--p-patch-composition', 'additive',
                  '--p-smoothing', str(p_sweeps),
                  '--p-patch-ksp-type', 'chebyshev',
                  '--p-patch-ksp-chebyshev-esteig', "0,0.125,0,1.1",
                  #hMG relaxation
                  '--gmg-iters', str(h_cycles),
                  '--h-patch', 'patch_pc',
                  '--h-patch-composition', 'additive',
                  '--h-smoothing', str(h_sweeps),
                  '--h-patch-ksp-type', 'chebyshev',
                  '--h-patch-ksp-chebyshev-esteig', "0,0.125,0,1.1",
                  # other
                  '--warmup-run', 'true',
                  '--petsc-timings', 'true',
                  '--log-file', 'output.log'
                  ]
