static char help[] =
"Solve nonlinear Liouville-Bratu equation by finite differences\n"
"in 2D on a structured-grid.  Option prefix lb_.  Solves\n"
"  - nabla^2 u - lambda e^u = 0\n"
"on the unit square [0,1]x[0,1] subject to zero Dirichlet boundary conditions.\n"
"Critical value occurs about at lambda = 6.808.  Optional exact solution by\n"
"Liouville (1853) for case lambda=1.0.\n\n";

/*
incredible performance: FAS+NGS, matrix-free, full-cycles, NGS for coarse solve
400 million unknowns in 22 seconds, 10-digit accuracy
uses 66% of 128 Gb memory of my Thelio massive machine; note 20 cores out of 40
note only 352 flops per degree of freedom!

$ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratufd -da_grid_x 5 -da_grid_y 5 -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 12
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
flops = 9.448e+10,  residual calls = 416,  NGS calls = 208
done on 16385 x 16385 grid:   error |u-uexact|_inf = 4.038e-10
real 21.70
*/

/* excellent evidence of convergence and optimality in Liouville exact solution case (use opt PETSc build):
NEWTON+MG with FD Jacobian matrix is fast
$ for LEV in 6 7 8 9 10; do timer ./bratufd -da_refine $LEV -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type newtonls -snes_fd_color -ksp_type cg -pc_type mg; done
FAS+NGS matrix-free full cycles are much faster
$ for LEV in 6 7 8 9 10; do timer ./bratufd -da_refine $LEV -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4; done
*/

#include <petsc.h>

typedef struct {
  // Dirichlet boundary condition g(x,y)
  PetscReal (*g_bdry)(PetscReal x, PetscReal y, void *ctx);
  PetscReal lambda;
  PetscBool exact;
  int       residualcount, ngscount;
} BratuCtx;

static PetscReal g_zero(PetscReal x, PetscReal y, void *ctx) {
    return 0.0;
}

static PetscReal g_liouville(PetscReal x, PetscReal y, void *ctx) {
    PetscReal r2 = (x + 1.0) * (x + 1.0) + (y + 1.0) * (y + 1.0),
              qq = r2 * r2 + 1.0,
              omega = r2 / (qq * qq);
    return PetscLogReal(32.0 * omega);
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, Vec, BratuCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal **,
                                        PetscReal**, BratuCtx*);
extern PetscErrorCode NonlinearGS(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    DM             da, da_after;
    SNES           snes;
    Vec            u, uexact;
    BratuCtx       bctx;
    DMDALocalInfo  info;
    PetscBool      showcounts = PETSC_FALSE;
    PetscLogDouble lflops, flops;
    PetscReal      errinf;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    bctx.g_bdry = &g_zero;
    bctx.lambda = 1.0;
    bctx.exact = PETSC_FALSE;
    bctx.residualcount = 0;
    bctx.ngscount = 0;
    PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","Liouville-Bratu equation solver options","");
    PetscCall(PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratufd.c",bctx.lambda,&(bctx.lambda),NULL));
    PetscCall(PetscOptionsBool("-exact","use case of Liouville exact solution",
                            "bratufd.c",bctx.exact,&(bctx.exact),NULL));
    PetscCall(PetscOptionsBool("-showcounts","print counts for calls to call-back functions",
                            "bratufd.c",showcounts,&showcounts,NULL));
    PetscOptionsEnd();
    if (bctx.exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_SELF,1,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        bctx.g_bdry = &g_liouville;
    }

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&bctx));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&bctx));
    PetscCall(SNESSetNGS(snes,NonlinearGS,&bctx));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(DMGetGlobalVector(da,&u));
    PetscCall(VecSet(u,0.0));  // initialize to zero
    PetscCall(SNESSolve(snes,NULL,u));
    PetscCall(DMRestoreGlobalVector(da,&u));
    PetscCall(DMDestroy(&da));

    if (showcounts) {
        PetscCall(PetscGetFlops(&lflops));
        PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)snes)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"flops = %.3e,  residual calls = %d,  NGS calls = %d\n",
                           flops,bctx.residualcount,bctx.ngscount));
    }

    PetscCall(SNESGetDM(snes,&da_after));
    PetscCall(DMDAGetLocalInfo(da_after,&info));
    if (bctx.exact) {
        PetscCall(SNESGetSolution(snes,&u));  // SNES owns u; we do not destroy it
        PetscCall(DMCreateGlobalVector(da_after,&uexact));
        PetscCall(FormUExact(&info,uexact,&bctx));
        PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
        PetscCall(VecDestroy(&uexact));
        PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                info.mx,info.my,errinf));
    } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",info.mx,info.my));
    }

    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormUExact(DMDALocalInfo *info, Vec u, BratuCtx* user) {
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    if (user->g_bdry != &g_liouville) {
        SETERRQ(PETSC_COMM_SELF,1,"exact solution only implemented for g_liouville() boundary conditions\n");
    }
    if (user->lambda != 1.0) {
        SETERRQ(PETSC_COMM_SELF,2,"Liouville exact solution only implemented for lambda = 1.0\n");
    }
    hx = 1.0 / (PetscReal)(info->mx - 1);
    hy = 1.0 / (PetscReal)(info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = i * hx;
            au[j][i] = user->g_bdry(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}


PetscBool NodeOnBdry(DMDALocalInfo *info, PetscInt i, PetscInt j) {
    return (((i == 0) || (i == info->mx-1) || (j == 0) || (j == info->my-1)));
}


// compute F(u), the residual of the discretized PDE on the given grid
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, BratuCtx *user) {
    PetscInt   i, j;
    const PetscInt nn = 0, ss = 1, ee = 2, ww = 3;
    PetscReal  hx, hy, darea, hxhy, hyhx, uu[4];

    hx = 1.0 / (PetscReal)(info->mx - 1);
    hy = 1.0 / (PetscReal)(info->my - 1);
    darea = hx * hy;
    hxhy = hx / hy;
    hyhx = hy / hx;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (j==0 || i==0 || i==info->mx-1 || j==info->my-1) {
                FF[j][i] = au[j][i] - user->g_bdry(i*hx,j*hy,user);
            } else {
                uu[nn] = NodeOnBdry(info,i,j+1) ? user->g_bdry(i*hx,(j+1)*hy,user) : au[j+1][i];
                uu[ss] = NodeOnBdry(info,i,j-1) ? user->g_bdry(i*hx,(j-1)*hy,user) : au[j-1][i];
                uu[ee] = NodeOnBdry(info,i+1,j) ? user->g_bdry((i+1)*hx,j*hy,user) : au[j][i+1];
                uu[ww] = NodeOnBdry(info,i-1,j) ? user->g_bdry((i-1)*hx,j*hy,user) : au[j][i-1];
                FF[j][i] =   hyhx * (2.0 * au[j][i] - uu[ww] - uu[ee])
                           + hxhy * (2.0 * au[j][i] - uu[ss] - uu[nn])
                           - darea * user->lambda * PetscExpScalar(au[j][i]);
            }
        }
    }
    PetscCall(PetscLogFlops(12.0 * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

// do nonlinear Gauss-Seidel (processor-block) sweeps on
//     F(u) = b
PetscErrorCode NonlinearGS(SNES snes, Vec u, Vec b, void *ctx) {
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    PetscReal      atol, rtol, stol, hx, hy, darea, hxhy, hyhx, x, y,
                   **au, **ab, bij, uu, phi0, phi, dphidu, s;
    DM             da;
    DMDALocalInfo  info;
    Vec            uloc;
    BratuCtx*      user = (BratuCtx*)ctx;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));

    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);
    darea = hx * hy;
    hxhy = hx / hy;
    hyhx = hy / hx;

    PetscCall(DMGetLocalVector(da,&uloc));
    for (l=0; l<sweeps; l++) {
        PetscCall(DMGlobalToLocalBegin(da,u,INSERT_VALUES,uloc));
        PetscCall(DMGlobalToLocalEnd(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(da,uloc,&au));
        if (b) {
            PetscCall(DMDAVecGetArrayRead(da,b,&ab));
        }
        for (j = info.ys; j < info.ys + info.ym; j++) {
            y = j * hy;
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (j==0 || i==0 || i==info.mx-1 || j==info.my-1) {
                    x = i * hx;
                    au[j][i] = user->g_bdry(x,y,user);
                } else {
                    if (b)
                        bij = ab[j][i];
                    else
                        bij = 0.0;
                    // do pointwise Newton iterations on scalar function
                    //   phi(u) =   hyhx * (2 u - au[j][i-1] - au[j][i+1])
                    //            + hxhy * (2 u - au[j-1][i] - au[j+1][i])
                    //            - darea * lambda * e^u - bij
                    uu = au[j][i];
                    for (k = 0; k < maxits; k++) {
                        phi =   hyhx * (2.0 * uu - au[j][i-1] - au[j][i+1])
                              + hxhy * (2.0 * uu - au[j-1][i] - au[j+1][i])
                              - darea * user->lambda * PetscExpScalar(uu) - bij;
                        if (k == 0)
                             phi0 = phi;
                        dphidu = 2.0 * (hyhx + hxhy)
                                 - darea * user->lambda * PetscExpScalar(uu);
                        s = - phi / dphidu;     // Newton step
                        uu += s;
                        totalits++;
                        if (   atol > PetscAbsReal(phi)
                            || rtol*PetscAbsReal(phi0) > PetscAbsReal(phi)
                            || stol*PetscAbsReal(uu) > PetscAbsReal(s)    ) {
                            break;
                        }
                    }
                    au[j][i] = uu;
                }
            }
        }
        PetscCall(DMDAVecRestoreArray(da,uloc,&au));
        PetscCall(DMLocalToGlobalBegin(da,uloc,INSERT_VALUES,u));
        PetscCall(DMLocalToGlobalEnd(da,uloc,INSERT_VALUES,u));
    }
    PetscCall(DMRestoreLocalVector(da,&uloc));
    if (b) {
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));
    }
    PetscCall(PetscLogFlops(21.0 * totalits));
    (user->ngscount)++;
    return 0;
}
