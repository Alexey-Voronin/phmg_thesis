hp_sweeps=(2 3 4)

for h in "${hp_sweeps[@]}"; do
    nice -20 mpiexec -n 8 python ../solver.py $h
done
