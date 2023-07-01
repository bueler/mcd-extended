# generate weak-scaling results for SIA Example 8.4 in paper

OPTS="-bumps -mcoarse 20 -fascd_cycle_type full -fascd_monitor -fascd_rtol 2.0e-4 -fascd_atol 1.0e-8 -quad"
time mpiexec -n  1 --map-by core --bind-to hwthread python3 sia.py $OPTS -levs 7
time mpiexec -n  4 --map-by core --bind-to hwthread python3 sia.py $OPTS -levs 8
time mpiexec -n 16 --map-by core --bind-to hwthread python3 sia.py $OPTS -levs 9
