# generate weak-scaling results for SIA Example 8.4 in paper
# uncomment later lines if you are on a big machine
# siaweak.txt is result from this script; see archer2/ for large machine logs

MPI="mpiexec -map-by numa -bind-to core"
OPTS="-bumps -mcoarse 20 -fascd_cycle_type full -fascd_monitor -fascd_rtol 1.0e-4 -fascd_atol 1.0e-8 -quad"
time $MPI -n  1 python3 sia.py $OPTS -levs 6
time $MPI -n  4 python3 sia.py $OPTS -levs 7
time $MPI -n 16 python3 sia.py $OPTS -levs 8
#time $MPI -n 64 python3 sia.py $OPTS -levs 9
#time $MPI -n 256 python3 sia.py $OPTS -levs 10
#time $MPI -n 1024 python3 sia.py $OPTS -levs 11
