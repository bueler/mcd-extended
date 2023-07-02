# mcd-extended/c/

> **Note**  
> This C+DMDA+2D _attempt_ to implement "NMCD" (old name) is completely
> deprecated and replaced by the Python+Firedrake+DMPlex+any-dimension
> implementation of FASCD in https://bitbucket.org/pefarrell/fascd/src/master/.

This directory contains a sequence of PETSc C codes on 2D structured meshes (DMDA)
which build toward NMCD functionality.  This document gives the sequence of codes
and their capabilities or performance, especially relative to the previous
in the sequence.  See the individual codes themselves for more information.

## bratu.c

This is a copy, with clean-up, of `c/ch7/solns/bratu2D.c` in
https://github.com/bueler/p4pdes.  It solves the Liouville-Bratu problem on a
square using a structured-grid DMDA and either finite differences or
Q1 finite elements.

With finite differences, just two matrix-free, FAS full cycles with
NGS smoothing suffices to give 10-digit accuracy on a problem with 270 million
degrees of freedom:

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fd -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_levels_snes_norm_schedule none -fas_coarse_snes_type ngs -fas_coarse_snes_max_it 1 -fas_coarse_snes_ngs_sweeps 4 -fas_coarse_snes_ngs_max_it 1 -da_refine 12
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
    flops = 1.671e+11,  exps = 7.484e+09,  residual calls = 531,  NGS calls = 350
    done on 16385 x 16385 grid:   error |u-uexact|_inf = 7.759e-11
    real 29.51

This uses 70% of the 128 Gb memory of my Thelio massive machine.  Note the
tight -snes_rtol tolerance, that NGS is used for the coarse solve, and that
20 cores out of available 40 are used.  Also, of course, an optimized PETSc
build is used.  Expenses per degree of freedom are 622 flops, 28 exponentials,
and 2.2 processor-microseconds.  This is extraordinary performance.

Alternative settings, using `nrichardson` for the coarse-level SNES, work
essentially the same way:

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread  ./bratu -lb_fd -lb_exact -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_coarse_snes_type nrichardson -fas_coarse_snes_linesearch_type basic -fas_coarse_npc_snes_type ngs -fas_coarse_npc_snes_ngs_sweeps 2 -fas_coarse_npc_snes_ngs_max_it 4 -snes_monitor_short -da_refine 12 -da_grid_x 5 -da_grid_y 5 -snes_rtol 1.0e-12

The Q1 finite element implementation is slower.  (The basic quadrature and FE
tools are in `q1fem.{h|c}`.)  The following FAS+NGS run gets comparable error to
above on a grid with 4 times fewer degrees of freedom (67 million).  Expenses
per degree of freedom: 41000 flops, 674 exponentials, 34 processor-microseconds.
This is about 65 times more flops per degree of freedom, though only 16 times
more time.

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_levels_snes_norm_schedule none -fas_coarse_snes_type ngs -fas_coarse_snes_max_it 1 -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_ngs_max_it 4 -da_refine 11
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 2.780e+12,  exps = 4.512e+10,  residual calls = 918,  NGS calls = 609
    done on 8193 x 8193 grid:   error |u-uexact|_inf = 8.366e-11
    real 113.31

Clearly FEM NGS is much more expensive per degree of freedom.  One can get a bit
better performance by switching to a nonlinear Jacobi smoother, which can only
do one iteration per point.  Expenses per degree of freedom: 36000 flops,
359 exponentials, 33 processor-microseconds.

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-12 -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_levels_snes_norm_schedule none -fas_coarse_snes_type ngs -fas_coarse_snes_max_it 1 -fas_coarse_snes_ngs_sweeps 4 -fas_coarse_snes_ngs_max_it 1 -da_refine 11 -lb_njac
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 6
    flops = 2.386e+12,  exps = 2.411e+10,  residual calls = 1382,  NGS calls = 919
    done on 8193 x 8193 grid:   error |u-uexact|_inf = 8.366e-11
    real 111.81

One can also use FAS with nonlinear Richardson, thereby avoiding calling NGS.
owever, this requires tuning, here through linesearch and its initial damping
parameter `-fas_levels_snes_linesearch_damping`.  For clarity we solve the
coarse problem exactly with Newton, and we do 4 `nrichardson` iterations as the
smoother.  Note these are FAS V-cycles; full cycles seem not to work for reasons
not clear to me:

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_converged_reason -lb_counts -snes_type fas -fas_levels_snes_type nrichardson -fas_levels_snes_linesearch_type l2 -fas_levels_snes_linesearch_damping 1.0 -fas_levels_snes_max_it 4 -fas_coarse_snes_type newtonls -da_refine 10 -fas_coarse_snes_converged_reason
                          Nonlinear fas_coarse_ solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
                          Nonlinear fas_coarse_ solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
                          Nonlinear fas_coarse_ solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
                          Nonlinear fas_coarse_ solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
                          Nonlinear fas_coarse_ solve converged due to CONVERGED_SNORM_RELATIVE iterations 1
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
    flops = 8.607e+11,  exps = 1.139e+10,  residual calls = 1411,  NGS calls = 0
    done on 4097 x 4097 grid:   error |u-uexact|_inf = 3.608e-08
    real 38.33

Now consider Newton-Krylov-multigrid runs, which do not use NGS.  These use
 `-snes_fd_color`, the default, and the FEM method has a 9 point stencil
versus the 5 point stencil of FD, so FEM already does more residual calls
on a given mesh.

The following comparison shows that in the above FD-versus-FEM comparison for
FAS-multigrid it is the mostly the NGS part of the Q1 FEM solver that causes
the performance hit.

    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fd -lb_exact -da_grid_x 5 -da_grid_y 5 -snes_converged_reason -lb_counts -ksp_converged_reason -pc_type mg -snes_rtol 1.0e-12 -da_refine 10
      Linear solve converged due to CONVERGED_RTOL iterations 6
      Linear solve converged due to CONVERGED_RTOL iterations 4
      Linear solve converged due to CONVERGED_RTOL iterations 6
      Linear solve converged due to CONVERGED_RTOL iterations 8
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 1.301e+11,  exps = 6.203e+08,  residual calls = 269,  NGS calls = 0
    done on 4097 x 4097 grid:   error |u-uexact|_inf = 1.242e-09
    real 22.01
    $ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu -lb_fem -lb_exact -da_grid_x 5 -da_grid_y 5 -snes_converged_reason -lb_counts -ksp_converged_reason -pc_type mg -snes_rtol 1.0e-12 -da_refine 10
      Linear solve converged due to CONVERGED_RTOL iterations 5
      Linear solve converged due to CONVERGED_RTOL iterations 3
      Linear solve converged due to CONVERGED_RTOL iterations 6
      Linear solve converged due to CONVERGED_RTOL iterations 8
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
    flops = 4.629e+11,  exps = 3.924e+09,  residual calls = 445,  NGS calls = 0
    done on 4097 x 4097 grid:   error |u-uexact|_inf = 3.346e-10
    real 38.88

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
`q1fem.h`.  Default solution by is by SNESVI type `vinewtonrsls`.

One can solve the linear step equations using GMG preconditioning.  However,
the combination does not scale.  This single-grid method can only
advance/retreat the margin (i.e. change the active set) one cell at a time.
In the runs

    $ ./obstaclesl -ob_counts -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine LEV

the number of linear iterations in the Newton solves is steady at 3 or 4, but
the number of SNES iterations starts to double with each refinement.  For
LEV=4,5,6,7,8 the SNES iterations are 5,9,13,24,46, respectively.

Option `-ob_pngs` gives an inefficient single-level solution by sweeps of
projected, nonlinear Gauss-Seidel.  This also does not scale.


## nmcd.c

FIXME do the thing
