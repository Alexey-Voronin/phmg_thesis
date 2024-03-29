hcycles=(1 2 3)
hp_sweeps=("2 2" "2 3" "3 3")

for cycle in "${hcycles[@]}"; do
	for hp in "${hp_sweeps[@]}"; do
	    read -r h p <<< "$hp"
	    nice -20 mpiexec -n 8 python ../solver.py $p $h $cycle
	done
done
