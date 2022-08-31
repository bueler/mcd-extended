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

With finite differences, a single matrix-free, FAS full cycle with
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
in `q1fem.{h|c}`.  The following run gets comparable error to above on a
grid with 4 times fewer degrees of freedom (67 million), but it costs 62000
flops per degree of freedom.  This is almost 100 times more work per degree
of freedom.  The reason is that NGS is much more expensive per degree of
freedom.

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 11
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 4.173e+12,  residual calls = 1479,  NGS calls = 753
    done on 8193 x 8193 grid:   error |u-uexact|_inf = 8.366e-11
    real 207.49

Here is the Q1 FEM run on the same grid as the highest-resolution FD run above:

    timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 12
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 1.629e+13,  residual calls = 1733,  NGS calls = 868
    done on 16385 x 16385 grid:   error |u-uexact|_inf = 2.091e-11
    real 824.33

This did about 61000 flops per degree of freedom, illustrating optimal
complexity, and about 61 processor-microseconds per degree of freedom.

See the regression tests in `makefile` for additional comparisons between FD
and FEM results.

FIXME performance comparison between Newton-Krylov-multigrid and
FAS+NGS multigrid.

Using a Newton-Krylov-multigrid approach, with default `-lb_quadpts 2` setting,
`bratu.c` does 5 times as many flops, is about 4 times slower, but generates
smaller inf-norm error, than `bratufd.c`.  Compare:

    $ timer ./bratu -da_refine 9 -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type newtonls -snes_fd_color -ksp_type cg -pc_type mg
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
    flops = 2.493e+10,  residual calls = 301,  NGS calls = 0
    done on 1025 x 1025 grid:   error |u-uexact|_inf = 5.355e-09
    real 23.59
    $ timer ./bratufd -da_refine 9 -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type newtonls -snes_fd_color -ksp_type cg -pc_type mg
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
    flops = 4.362e+09,  residual calls = 181,  NGS calls = 0
    done on 1025 x 1025 grid:   error |u-uexact|_inf = 1.986e-08
    real 5.62

The number of quadrature points matters critically in the performance of
`bratu.c`.

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

## dcsl.c

FIXME rewrite of above to use defect constraint
