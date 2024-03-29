
def get_params(h_sweeps):
    return  [# problem params
              '--baseN', '1',
              '--quadrilateral',  'false',
              '--mh',             'uniform',
              '--discretization', 'th',
              '--nref-start', '2',
              '--nref-end', '3',
              '--k-start', '4',
              '--k-end', '5',
              # solver params
              '--solver-type', 'hmg',
              #hMG relaxation
              '--h-patch', 'asm_star_pc',
              '--h-patch-composition', 'additive',
              '--h-smoothing', str(h_sweeps),
              '--h-patch-entity', '0',
              # other
              '--warmup-run', 'true',
              '--petsc-timings', 'false', # 'true',
              '--log-file', 'output.log'
              ]
