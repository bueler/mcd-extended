# mcd-extended/c/

This directory contains a sequence of PETSc C codes on structured meshes (DMDA)
which build toward NMCD functionality.  This document gives the sequence of codes
and their capabilities or performance, especially relative to the previous
in the sequence.  See the individual codes themselves for more information.

## bratufd.c

This is a copy, with cleaning, of `c/ch7/solns/bratu2D.c` in
https://github.com/bueler/p4pdes.  It solves the Liouville-Bratu problem on a
square using finite differences and DMDA.

It is the starting point showing that a single matrix-free, FAS full cycle with
NGS smoothing suffices to give 10-digit accuracy on a problem with 270 million
degrees of freedom, at only 352 flops per degree of freedom.  (It uses 66% of
the 128 Gb memory of my Thelio massive machine.  Note 20 cores out of available
40.  Of course, use an optimized PETSc build.)  Note NGS is also used for the
coarse solve.  This is extraordinary performance.

        $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratufd -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 12
        Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
        flops = 9.448e+10,  residual calls = 416,  NGS calls = 208
        done on 16385 x 16385 grid:   error |u-uexact|_inf = 4.038e-10
        real 21.70

See the code comments for convergence comparison, and performance comparison
between Newton-Krylov-multigrid and FAS+NGS multigrid.

## bratu.c

This solves the same problem as `bratufd.c`, but now by a Q1 finite element
method.  Some basic quadrature and FE tools from `c/ch9/phelm.c` in
`p4pdes` are here in `q1fem.h`.

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
`bratu.c`.  It does more flops per quadrature point than `bratufd.c` does
per grid point, and (by default) there are 4 quadrature points per element.

FAS+NGS multigrid works like `bratufd` but with the same kind of loss of performance:

FIXME run on thelio with mpiexec -n 20 and -da_refine 11 or 12

        $ timer mpiexec -n 2 --map-by core --bind-to hwthread ./bratufd -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 7
        Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
        flops = 2.724e+08,  residual calls = 325,  NGS calls = 183
        done on 513 x 513 grid:   error |u-uexact|_inf = 7.947e-08
        real 4.61


## obstaclesl.c

This solves a classical obstacle problem, specifically the problem solved in Chapter 12 of Bueler (2021), _PETSc for Partial Differential Equations_, by a Q1 finite element method using basic quadrature and FE tools from `q1fem.h`.  This is an inefficient single-level solution by sweeps of projected, nonlinear Gauss-Seidel.

FIXME show runs
