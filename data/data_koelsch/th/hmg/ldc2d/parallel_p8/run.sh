echo `pwd`
for h in 2 3; do
    nice -20 mpiexec -n 8 python ../solver.py $h
    echo "V_h${h}"
done
