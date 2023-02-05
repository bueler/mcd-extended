#!/bin/bash

# run a sequence of FAS Bratu solves to get scaling data, e.g.:
#   ./run.sh >& rawFD.txt

# remember to do in parent directory:
#export PETSC_ARCH=linux-c-opt;  make bratu

# compare Firedrake Poisson with P1 elements on 2049^2 grid using p4pdes/python/ch13/fish.py:
# $ time mpiexec -n 4 --map-by core --bind-to hwthread ./fish.py -refine 10 -k 1 -s_ksp_converged_reason -s_ksp_rtol 1.0e-10 -s_pc_type mg
#  Linear s_ solve converged due to CONVERGED_RTOL iterations 10
#done on 2049 x 2049 grid with P_1 elements:
#  error |u-uexact|_inf = 3.785e-09, |u-uexact|_h = 2.088e-09
#real 84.97
# = 20.2 microseconds per degree of freedom

NP=4
DEPTHLIST="7 8 9 10 11"
DISC="-lb_fd"   # versus "-lb_fem"

echo "NP=$NP"
echo "DEPTHLIST=\"$DEPTHLIST\""
echo "DISC=$DISC"
echo

for DEPTH in $DEPTHLIST; do
    echo "DEPTH=$DEPTH"
    time mpiexec -n $NP --map-by core --bind-to hwthread ../bratu $DISC -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_levels_snes_norm_schedule none -fas_coarse_snes_type ngs -fas_coarse_snes_max_it 1 -fas_coarse_snes_ngs_sweeps 4 -fas_coarse_snes_ngs_max_it 1 -da_refine $DEPTH
    echo
done
