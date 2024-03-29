echo `pwd`
for h in 2 3; do
    echo "V_h${h}"
    nice -20 mpiexec -n 8 python ../solver.py $h
done
