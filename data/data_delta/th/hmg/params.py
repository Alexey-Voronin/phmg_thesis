
def get_params(k, r, h_sweeps, dim=2):
    r = int(r)
    k = int(k)
    return  [# problem params
              '--baseN', '1',
              '--quadrilateral',  'false',
              '--mh',             'uniform',
              '--discretization', 'th',
              '--nref-start', str(r),
              '--nref-end', str(r+1), 
              '--k-start', str(k),
              '--k-end', str(k+1),
              # solver params
              '--solver-type', 'hmg',
              #hMG relaxation
              '--h-patch', 'asm_star_pc',
              '--h-patch-composition', 'additive',
              '--h-smoothing', str(h_sweeps),
              '--h-patch-entity', '0',
              '--h-patch-ksp-type', 'chebyshev', # 'fgmres',
              '--h-patch-ksp-chebyshev-esteig', "0,0.125,0,1.1", 
              # other
              '--warmup-run',  'true',
              '--petsc-timings',  'true',
              '--log-file', 'output.log'
              ]
