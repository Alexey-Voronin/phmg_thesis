
def get_params(k, r, h_sweeps, mesh_type="structured"):
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
                  '--k-coarsening-schedule', 'none', # hmg
                  '--solver-type', 'afbf',
                  #hMG relaxation
                  '--h-patch', 'asm_star_pc',
                  '--h-patch-composition', 'additive',
                  '--h-smoothing', str(h_sweeps),
                  '--h-patch-ksp-type', 'chebyshev', 
                  '--h-patch-ksp-chebyshev-esteig', "0,0.125,0,1.1", 
                  # other
                  '--warmup-run',  'true',
                  '--petsc-timings',  'true',
                  '--log-file', 'output.log'
                  ]
