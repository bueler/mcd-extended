# mcd-extended/c/

This directory contains a sequence of PETSc C codes on 2D structured meshes (DMDA)
which build toward NMCD functionality.  This document gives the sequence of codes
and their capabilities or performance, especially relative to the previous
in the sequence.  See the individual codes themselves for more information.

## bratu.c

This is a copy, with clean-up, of `c/ch7/solns/bratu2D.c` in
https://github.com/bueler/p4pdes.  It solves the Liouville-Bratu problem on a
square using a structured-grid DMDA and either finite differences or
Q1 finite elements.

With finite differences, two matrix-free, FAS full cycles with
NGS smoothing suffices to give 10-digit accuracy on a problem with 270 million
degrees of freedom, at only 802 flops per degree of freedom, and 2.4 processor-
microseconds per degree of freedom:

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fd -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 12
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
    flops = 2.152e+11,  residual calls = 855,  NGS calls = 428
    done on 16385 x 16385 grid:   error |u-uexact|_inf = 7.761e-11
    real 32.66

This uses 70% of the 128 Gb memory of my Thelio massive machine.  Note the
tight -snes_rtol tolerance, that NGS is used for the coarse solve, and that
20 cores out of available 40 are used.  Also, of course, an optimized PETSc
build is used.  This is extraordinary performance.

The Q1 finite element implementation is slower.  It is built starting from the
basic quadrature and FE tools from `c/ch9/phelm.c` in `p4pdes`; here they are
in `q1fem.{h|c}`.  The following FAS+NGS run gets comparable error to above on
a grid with 4 times fewer degrees of freedom (67 million), but it costs 62000
flops per degree of freedom.  This is almost 100 times more work per degree
of freedom.  The reason is that NGS is much more expensive per degree of
freedom.

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 11
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 4.173e+12,  residual calls = 1479,  NGS calls = 753
    done on 8193 x 8193 grid:   error |u-uexact|_inf = 8.366e-11
    real 207.49

Here is the FAS+NGS Q1 FEM run on the same grid as the highest-resolution FD run above:

    timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 12
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 1.629e+13,  residual calls = 1733,  NGS calls = 868
    done on 16385 x 16385 grid:   error |u-uexact|_inf = 2.091e-11
    real 824.33

This did about 61000 flops per degree of freedom, illustrating optimal
complexity, and about 61 processor-microseconds per degree of freedom.

The following comparison of Newton-Krylov-multigrid runs, which do not use NGS,
shows that in the above FD-versus-FEM comparison it is the Q1 FEM NGS that
causes most of the performance hit.  This compares different resolutions but
generating nearly the same accuracy; Q1 FEM is more accurate in L^inf:

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fd -lb_exact -da_grid_x 5 -da_grid_y 5 -snes_converged_reason -lb_counts -ksp_converged_reason -pc_type mg -snes_rtol 1.0e-12 -da_refine 11
      Linear solve converged due to CONVERGED_RTOL iterations 5
      Linear solve converged due to CONVERGED_RTOL iterations 4
      Linear solve converged due to CONVERGED_RTOL iterations 6
      Linear solve converged due to CONVERGED_RTOL iterations 8
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 5.087e+11,  residual calls = 293,  NGS calls = 0
    done on 8193 x 8193 grid:   error |u-uexact|_inf = 3.104e-10
    real 91.04
    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -lb_exact -da_grid_x 5 -da_grid_y 5 -snes_converged_reason -lb_counts -ksp_converged_reason -pc_type mg -snes_rtol 1.0e-12 -da_refine 10
      Linear solve converged due to CONVERGED_RTOL iterations 5
      Linear solve converged due to CONVERGED_RTOL iterations 3
      Linear solve converged due to CONVERGED_RTOL iterations 6
      Linear solve converged due to CONVERGED_RTOL iterations 8
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 3.444e+11,  residual calls = 445,  NGS calls = 0
    done on 4097 x 4097 grid:   error |u-uexact|_inf = 3.346e-10
    real 53.79

The number of quadrature points is important in the performance of
`bratu.c -lb_fem`.  With `-lb_quadpts 1` there seem to be convergence
difficulties.  With `-lb_quadpts 2` and `-lb_quadpts 3` the error norms
and convergence results are nearly the same, but since the 9/4 increase
in the number of 2D quadrature points is indeed reflected by a slowdown
by a factor of about 2.

See the regression tests in `makefile` for additional comparisons between FD
and FEM results.


## obstaclesl.c

This solves a classical obstacle problem, specifically the problem solved in
Chapter 12 of Bueler (2021), _PETSc for Partial Differential Equations_, but
here by a Q1 finite element method using basic quadrature and FE tools from
`q1fem.h`.  Option `-ob_pngs` gives an inefficient single-level solution by
sweeps of projected, nonlinear Gauss-Seidel.  Solution by SNESVI type
`vinewtonrsls` is also implemented, which is actually more efficient.  Compare:

    $ ./obstaclesl -ob_counts -snes_converged_reason -ob_pngs -da_refine 4
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 330
    flops = 1.409e+09,  residual calls = 331,  PNGS calls = 330
    done on 33 x 33 grid:   error |u-uexact|_inf = 5.781e-03
    $ ./obstaclesl -ob_counts -snes_converged_reason -ksp_converged_reason -da_refine 4
      Linear solve converged due to CONVERGED_RTOL iterations 10
      Linear solve converged due to CONVERGED_RTOL iterations 12
      Linear solve converged due to CONVERGED_RTOL iterations 12
      Linear solve converged due to CONVERGED_RTOL iterations 12
      Linear solve converged due to CONVERGED_RTOL iterations 12
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
    flops = 2.922e+07,  residual calls = 56,  PNGS calls = 0
    done on 33 x 33 grid:   error |u-uexact|_inf = 5.781e-03

## nmcd.c

FIXME do the thing
