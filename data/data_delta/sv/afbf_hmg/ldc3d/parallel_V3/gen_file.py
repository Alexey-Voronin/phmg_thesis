import subprocess


gen_file = lambda N, t, k, r : \
rf"""#!/bin/bash
#SBATCH --mem=246G 
#SBATCH --nodes={N}
#SBATCH --ntasks-per-node={t}
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcfx-delta-cpu
#SBATCH --job-name=th_new/hmg_ldc3d
#SBATCH --time=2:00:00      # hh:mm:ss for the job
#SBATCH --constraint='scratch'
#SBATCH --propagate=STACK

export OMP_NUM_THREADS=1
module purge

module load gcc/11.4.0
module load openmpi/4.1.6 
source /u/avoronin1/firedrake_metis/firedrake/bin/activate

export GCC_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku
export OMPI_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74
export LIBSTDC=/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/lib64

export PATH=${{GCC_PATH}}/bin:${{OMPI_PATH}}/bin:${{PATH}}
export LD_LIBRARY_PATH=${{LIBSTDC}}:${{GCC_PATH}}/lib:${{OMPI_PATH}}/lib:$LD_LIBRARY_PATH

export FI_CXI_RX_MATCH_MODE=software 

srun python ../solver.py {k} {r} 3 
"""

with open("job_map.log", 'a') as log_file_handle:

    for r, (nodes, tasks) in zip(range(2,3),         # refinements
                                [#(6, 12), 
                                    (32, 6)]): # (nodes, tasks per node)
        for k in range(3, 11):
            file_path = f'r{r}_k{str(k).zfill(2)}_V3.slurm'
            gen_file_content = gen_file(nodes, tasks, k, r)
            with open(file_path, 'w') as file:
                file.write(gen_file_content)

            result = subprocess.run(['sbatch', file_path], capture_output=True, text=True)
            print(f'{file_path} --> {result.stdout.split()[-1]}', file=log_file_handle)
            #print(f'{file_path} --> 0', file=log_file_handle)
