# notes

WARNING:  These notes are far out of date, but might contain something future useful.

## achievements

  * `examples/drape.py`: all four 2D Laplacian problems (3 obstacle, one PDE) tested show FASCD pure V-cycle flying colors, with nearly mesh-independent iterations when memory-limited:

        for PROB in ball spiral elasto fish; do timer mpiexec -n 16 --map-by core --bind-to hwthread python3 drape.py -prob $PROB -levs 11; done

    on thelio with 134GB ram, these use half the memory
    [also: uncomment asserts in nested iteration to confirm adjustments needed for admissibility]
    [also: F-cycles and nested iteration do great ... but V-cycles are already great]

  * `examples/blplap.py`: following 2D p-Laplacian PDE (unconstrained) FASCD nested iteration works great; 1 or 2 V-cycles do the job in most cases:

        for PP in 1.5 2 2.5 3 4; do for L in 2 3 4 5 6; do python3 blplap.py -levs $L -p $PP; done; done

  * `examples/plap1d.py`: for 1D p-Laplacian obstacle problem, FASCD F-cycle is clearly optimal for p=1.5 (fast diffusion) and p=4 (degenerate diffusion); one V-cycle does the job in all cases!:

        for P in 1.5 4; do for L in 4 6 8 10 12; do timer python3 plap1d.py -fascd_cycle_type full -p $P -levs $L; done; done

  * `examples/pollutant.py`: for 2D steady advection-diffusion bi-obstacle problem, convection-dominated, FASCD F-cycle is optimal in runs up to memory-limited; here with P1 but same conclusion with Q1:

        for L in 1 2 3 4 5 6 7 8 9; do mpiexec -n 4 --map-by core --bind-to hwthread python3 pollutant.py -dim 2 -levs $L; done

  * `examples/pollutant.py`: for 3D steady advection-diffusion bi-obstacle problem is similar; here is Q1 but basically same with P1 (except P1 uses more time for mesh hierarchy generation and more peak memory):

        for L in 1 2 3 4 5; do mpiexec -n 16 --map-by core --bind-to hwthread python3 pollutant.py -dim 3 -q1 -levs $L; done

  * `examples/sia.py`: for 2D flat bed shallow ice approximation obstacle problem with FASCD F-cycles, get evidence of optimality (and very good V-cycle convergence rates):

        for L in 1 2 3 4 5 6 7 8 9; do timer mpiexec -n 16 --map-by core --bind-to hwthread python3 sia.py -fascd_monitor -levs $L; done

  * `examples/sia.py`: for 2D *bumpy* bed shallow ice approximation obstacle problem with FASCD F-cycles, get evidence of optimality (and very good V-cycle convergence rates):

        for L in 1 2 3 4 5 6 7 8 9; do timer mpiexec -n 16 --map-by core --bind-to hwthread python3 sia.py -fascd_monitor -levs $L -bumps; done

## questions

  * in the coarse correction, should the R operator ever be conjugated with a power or something?
  * in `operators.py`, do `op2.Max` and `op2.Min` act as no ops in serial? if so, do they do the right thing, or they do anything at all, in parallel?
