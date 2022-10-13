static char help[] =
"Solve classical obstacle problem by nonlinear multilevel constraint\n"
"decomposition (NMCD) method using Q1 finite elements in 2D square (-2,2)^2\n"
"on a structured-grid (DMDA).  Problem is\n"
"  - nabla^2 u = f(x,y),  u >= psi(x,y),\n"
"subject to Dirichlet boundary conditions u=g.  Smoother and coarse-level\n"
"solver is projected, nonlinear Gauss-Seidel (PNGS) sweeps.  Option prefix\n"
"nm_.  Compare obstaclesl.c.\n\n";

#include <petsc.h>
#include "src/utilities.h"
#include "src/q1fem.h"
#include "src/q1transfers.h"
#include "src/ldc.h"

typedef struct {
  // obstacle gamma_lower(x,y)
  PetscReal (*gamma_lower)(PetscReal x, PetscReal y, void *ctx);
  // right-hand side f(x,y)
  PetscReal (*f_rhs)(PetscReal x, PetscReal y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscReal (*g_bdry)(PetscReal x, PetscReal y, void *ctx);
  PetscInt  maxits, quadpts, sweeps,
            residualcount, ngscount;
} ObsCtx;

typedef struct {
  PetscInt  _level;  // =0 in single-level usage; otherwise 0 is coarsest
  LDC       ldc;     // object which holds/manages constraints at each level
  DM        dmda;    // DMDA (structured grid) for this level
  Vec       g,       // iterate on the level
            ell,     // right-hand side linear functional on level
            y,       // downward correction on level
            z;       // upward correction on level
} Level;

// z = gamma_lower(x,y) is the hemispherical obstacle, but made C^1 with "skirt" at r=r0
PetscReal gamma_lower(PetscReal x, PetscReal y, void *ctx) {
    const PetscReal  r = PetscSqrtReal(x * x + y * y),
                     r0 = 0.9,
                     psi0 = PetscSqrtReal(1.0 - r0 * r0),
                     dpsi0 = - r0 / psi0;
    if (r <= r0) {
        return PetscSqrtReal(1.0 - r * r);
    } else {
        return psi0 + dpsi0 * (r - r0);
    }
}

/*  This exact solution solves a 1D radial free-boundary problem for the
Laplace equation, on r >= 0, with hemispherical obstacle
gamma_lower(r).  The Laplace equation applies where u(r) > gamma_lower(r),
    u''(r) + r^-1 u'(r) = 0
with boundary conditions including free b.c.s at an unknown location r = a:
    u(a) = gamma_lower(a),  u'(a) = gamma_lower'(a),  u(2) = 0
The solution is  u(r) = - A log(r) + B   on  r > a.  The boundary conditions
can then be reduced to a root-finding problem for a:
    a^2 (log(2) - log(a)) = 1 - a^2
The solution is a = 0.697965148223374 (giving residual 1.5e-15).  Then
A = a^2*(1-a^2)^(-0.5) and B = A*log(2) are as given below in the code.  */
PetscReal u_exact(PetscReal x, PetscReal y, void *ctx) {
    const PetscReal afree = 0.697965148223374,
                    A     = 0.680259411891719,
                    B     = 0.471519893402112;
    PetscReal       r = PetscSqrtReal(x * x + y * y);
    return (r <= afree) ? gamma_lower(x,y,ctx)  // active set; on the obstacle
                        : - A * PetscLogReal(r) + B; // solves laplace eqn
}

static PetscReal fg_zero(PetscReal x, PetscReal y, void *ctx) {
    return 0.0;
}

extern PetscErrorCode AssertBoundaryAdmissible(DM da, ObsCtx*);
extern PetscErrorCode FormExact(DM da, Vec, ObsCtx*);
extern PetscErrorCode ApplyOperatorF(DM, Vec, Vec, ObsCtx*);
extern PetscErrorCode ProjectedNGS(LDC*, PetscBool, Vec, Vec, ObsCtx*);

int main(int argc,char **argv) {
    Level          *levs;
    Vec            gamlow, w, tmp, uexact, F;
    ObsCtx         ctx;
    DMDALocalInfo  finfo;
    PetscInt       totlevs=2, cycles=1, jtop, j;
    PetscBool      ldcinfo = PETSC_FALSE, counts = PETSC_FALSE, view = PETSC_FALSE,
                   admis;
    PetscLogDouble lflops, flops;
    PetscReal      errinf;
    char           viewname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    ctx.gamma_lower = &gamma_lower;
    ctx.f_rhs = &fg_zero;
    ctx.g_bdry = &u_exact;
    ctx.maxits = 2;
    ctx.quadpts = 2;
    ctx.sweeps = 1;
    ctx.residualcount = 0;
    ctx.ngscount = 0;
    PetscOptionsBegin(PETSC_COMM_WORLD,"nm_","obstacle problem NMCD solver options","");
    PetscCall(PetscOptionsBool("-counts","print counts for calls to call-back functions",
                            "nmcd.c",counts,&counts,NULL));
    PetscCall(PetscOptionsInt("-cycles","maximum number of NMCD V-cycles",
                            "nmcd.c",cycles,&cycles,NULL));
    PetscCall(PetscOptionsBool("-info","print info on LDC (MCD) actions",
                            "nmcd.c",ldcinfo,&ldcinfo,NULL));
    PetscCall(PetscOptionsInt("-levels","total number NMCD levels (>= 1)",
                            "nmcd.c",totlevs,&totlevs,NULL));
    PetscCall(PetscOptionsInt("-maxits","number of Newton iterations at each point",
                            "nmcd.c",ctx.maxits,&(ctx.maxits),NULL));
    // WARNING: coarse problems are badly solved with -nm_quadpts 1
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only)",
                            "nmcd.c",ctx.quadpts,&(ctx.quadpts),NULL));
    PetscCall(PetscOptionsInt("-sweeps","number of PNGS sweeps in smoother",  // FIXME separate for coarse solver
                            "nmcd.c",ctx.sweeps,&(ctx.sweeps),NULL));
    PetscCall(PetscOptionsString("-view","custom view of solution,residual,cr in ascii_matlab",
                            "nmcd",viewname,viewname,sizeof(viewname),&view));
    PetscOptionsEnd();

    // options consistency checking
    if (totlevs < 1) {
        SETERRQ(PETSC_COMM_SELF,1,"totlevs >= 1 required");
    }
    if (ctx.quadpts < 1 || ctx.quadpts > 3) {
        SETERRQ(PETSC_COMM_SELF,2,"quadrature points n=1,2,3 only");
    }

    // allocate Level stack
    PetscCall(PetscMalloc1(totlevs,&levs));

    // create DMDA for coarsest level: 3x3 grid on on Omega = (-2,2)x(-2,2)
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(levs[0].dmda)));
    PetscCall(DMSetFromOptions(levs[0].dmda));  // allows -da_grid_x mx -da_grid_y my etc.
    PetscCall(DMDASetInterpolationType(levs[0].dmda,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(levs[0].dmda,2,2,2));
    PetscCall(DMSetUp(levs[0].dmda));  // must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(levs[0].dmda,-2.0,2.0,-2.0,2.0,0.0,0.0));

    // create Level stack: create coarsest, then refine jtop times;
    // each Level has an LDC; each level has allocated g,ell,y,z Vecs
    for (j=0; j<totlevs; j++) {
        levs[j]._level = j;
        if (j == 0) {
            PetscCall(LDCCreateCoarsest(ldcinfo,levs[0].dmda,&(levs[0].ldc)));
        } else {
            PetscCall(LDCRefine(&(levs[j-1].ldc),&(levs[j].ldc)));
            levs[j].dmda = levs[j].ldc.dal;
        }
        PetscCall(DMSetApplicationContext(levs[j].dmda,&ctx));
        PetscCall(DMCreateGlobalVector(levs[j].dmda,&(levs[j].g)));
        PetscCall(VecDuplicate(levs[j].g,&(levs[j].ell)));
        PetscCall(VecDuplicate(levs[j].g,&(levs[j].y)));
        PetscCall(VecDuplicate(levs[j].g,&(levs[j].z)));
    }
    jtop = totlevs - 1;

    // check admissibility of the finest-level boundary condition;
    // checks ctx->g_bdry() >= ctx->gamma_lower() along boundary
    PetscCall(AssertBoundaryAdmissible(levs[jtop].dmda,&ctx));

    // generate finest-level obstacle gamlow as Vec
    PetscCall(DMCreateGlobalVector(levs[jtop].dmda,&gamlow));
    PetscCall(LDCVecFromFormula(levs[jtop].ldc,gamma_lower,gamlow,&ctx));

    // create initial iterate w on finest level
    PetscCall(DMCreateGlobalVector(levs[jtop].dmda,&w));
    PetscCall(FormExact(levs[jtop].dmda,w,&ctx));  // FIXME initializing with exact solution
    //PetscCall(VecSet(w,0.0));

    // check admissibility of initial iterate u on finest level
    PetscCall(VecLessThanOrEqual(levs[jtop].dmda,gamlow,w,&admis));
    if (!admis) {
        SETERRQ(PETSC_COMM_SELF,3,"initial iterate is not admissible\n");
    }

    // complete set-up of LDC stack for initial iterate u
    PetscCall(LDCFinestUpDCsFromVecs(w,NULL,gamlow,&(levs[jtop].ldc)));
    PetscCall(LDCGenerateDCsVCycle(&(levs[jtop].ldc)));

    // solve the problem
    PetscCall(VecPrintRange(w,"initial iterate w",""));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"*NOT YET* SOLVING PROBLEM BY %d V-CYCLES\n",cycles));

    // one NMCD V-cycle with one smoother iteration at each j>0 level
    PetscCall(VecCopy(w,levs[jtop].g));
    PetscCall(VecSet(levs[jtop].ell,0.0));
    // downward direction
    for (j=jtop; j>0; j--) {
        PetscCall(VecSet(levs[j].y,0.0));
        // FIXME modify NGS to take g,y separately (not u together)
        //PetscCall(ProjectedNGS(&(levs[j].ldc),PETSC_FALSE,levs[j].ell,
        //                       levs[j].g,levs[j].y,&ctx));
        PetscCall(VecDuplicate(levs[j].g,&tmp));
        PetscCall(VecWAXPY(tmp,1.0,levs[j].g,levs[j].y));
        PetscCall(Q1Inject(levs[j].dmda,levs[j-1].dmda,tmp,&(levs[j-1].g)));
        PetscCall(VecDestroy(&tmp));
        // FIXME construct ell on j-1
        //PetscCall(ApplyOperatorF(levs[j].dmda,u,F,&ctx));
        //PetscCall(VecWAXPY(levs[j-1].ell,ldc[j-1].
    }
    // coarse solve
    // FIXME
    // upward direction
    for (j=1; j<jtop; j++) {
        // FIXME
    }

    // FIXME not needed for run
    // compute and report residual for initial iterate
    PetscCall(DMCreateGlobalVector(levs[jtop].dmda,&F));
    PetscCall(ApplyOperatorF(levs[jtop].dmda,w,F,&ctx));
    PetscCall(VecPrintRange(F,"residual F[u]",""));
    PetscCall(VecDestroy(&F));

    if (counts) {
        // note calls to FormResidualOrCRLocal() and ProjectedNGS() are
        // collective but flops are per-process, so we need a reduction
        PetscCall(PetscGetFlops(&lflops));
        PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,
                                PETSC_COMM_WORLD));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "flops = %.3e,  residual calls = %d,  PNGS calls = %d\n",
                              flops,ctx.residualcount,ctx.ngscount));
    }

#if 0
    if (view) {
        Vec          F;
        PetscReal    **au, **aF;
        PetscViewer  file;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "viewing solution,residual,cr into ascii_matlab file %s\n",viewname));
        PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&file));
        PetscCall(PetscViewerSetType(file,PETSCVIEWERASCII));
        PetscCall(PetscViewerFileSetMode(file,FILE_MODE_WRITE));
        PetscCall(PetscViewerFileSetName(file,viewname));
        PetscCall(PetscViewerPushFormat(file,PETSC_VIEWER_ASCII_MATLAB));
        PetscCall(PetscObjectSetName((PetscObject)u,"u"));
        PetscCall(VecView(u,file));
        PetscCall(DMDAVecGetArray(da, u, &au));
        PetscCall(DMCreateGlobalVector(da,&F));
        PetscCall(DMDAVecGetArray(da, F, &aF));
        ctx.pngs = PETSC_FALSE;  // we want to view plain F, *then* CR
        PetscCall(FormResidualOrCRLocal(&info,au,aF,&ctx));
        PetscCall(DMDAVecRestoreArray(da, F, &aF));
        PetscCall(PetscObjectSetName((PetscObject)F,"F"));
        PetscCall(VecView(F,file));
        PetscCall(DMDAVecGetArray(da, F, &aF));
        PetscCall(CRLocal(&info,au,aF,aF,&ctx));
        PetscCall(DMDAVecRestoreArray(da, F, &aF));
        PetscCall(DMDAVecRestoreArray(da, u, &au));
        PetscCall(PetscObjectSetName((PetscObject)F,"Fhat"));
        PetscCall(VecView(F,file));
        PetscCall(VecDestroy(&F));
        PetscCall(PetscViewerPopFormat(file));
        PetscCall(PetscViewerDestroy(&file));
    }
#endif

    // report on numerical error
    PetscCall(DMCreateGlobalVector(levs[jtop].dmda,&uexact));
    PetscCall(FormExact(levs[jtop].dmda,uexact,&ctx));
    PetscCall(VecAXPY(w,-1.0,uexact));    // u <- u + (-1.0) uexact
    PetscCall(VecNorm(w,NORM_INFINITY,&errinf));
    PetscCall(DMDAGetLocalInfo(levs[jtop].dmda,&finfo));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                          finfo.mx,finfo.my,errinf));

    // destroy it all
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&gamlow));
    PetscCall(VecDestroy(&w));
    for (j=0; j<totlevs; j++) {
        PetscCall(VecDestroy(&(levs[j].g)));
        PetscCall(VecDestroy(&(levs[j].ell)));
        PetscCall(VecDestroy(&(levs[j].y)));
        PetscCall(VecDestroy(&(levs[j].z)));
        PetscCall(LDCDestroy(&(levs[j].ldc))); // destroys levs[j].dmda
    }
    PetscCall(PetscFree(levs));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode AssertBoundaryAdmissible(DM da, ObsCtx* user) {
    DMDALocalInfo  info;
    PetscInt       i, j;
    PetscReal      hx, hy, x, y;
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1);
    hy = 4.0 / (PetscReal)(info.my - 1);
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = -2.0 + j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = -2.0 + i * hx;
            if (user->g_bdry(x,y,user) < user->gamma_lower(x,y,user)) {
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                    "ERROR: g(x,y) >= gamma(x,y) fails at x=%.6e,y=%.6e\n",x,y));
                SETERRQ(PETSC_COMM_SELF,1,"assertion fails: boundary values are above obstacle");
            }
        }
    }
    return 0;
}

PetscErrorCode FormExact(DM da, Vec u, ObsCtx* user) {
    DMDALocalInfo  info;
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **au;
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1);
    hy = 4.0 / (PetscReal)(info.my - 1);
    PetscCall(DMDAVecGetArray(da, u, &au));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = -2.0 + j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = -2.0 + i * hx;
            au[j][i] = u_exact(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, u, &au));
    return 0;
}

PetscBool _NodeOnBdry(DMDALocalInfo info, PetscInt i, PetscInt j) {
    return (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1);
}

// compute complementarity residual Fhat from conventional residual F:
//     Fhat_ij = F_ij         if u_ij > gamma_lower(x_i,y_j)
//               min{F_ij,0}  if u_ij <= gamma_lower(x_i,y_j)
// note arrays aF and aFhat can be the same, giving in-place calculation
// note this op is idempotent if u,gamma_lower unchanged:  CRLocal(CRLocal()) = CRLocal()
PetscErrorCode _CRLocal(DMDALocalInfo *info, PetscReal **au, PetscReal **aF,
                        PetscReal **aFhat, ObsCtx *user) {
    const PetscReal hx = 4.0 / (PetscReal)(info->mx - 1),
                    hy = 4.0 / (PetscReal)(info->my - 1);
    PetscInt        i, j;
    PetscReal       x, y;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = -2.0 + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (!_NodeOnBdry(*info,i,j)) {
                x = -2.0 + i * hx;
                if (au[j][i] > user->gamma_lower(x,y,user))
                     aFhat[j][i] = aF[j][i];
                else
                     aFhat[j][i] = PetscMin(aF[j][i],0.0);
            }
        }
    }
    return 0;
}

// FLOPS: 2 + (16 + 5 + 7) = 30
PetscReal _IntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                        const PetscReal uu[4], const PetscReal ff[4],
                        PetscInt r, PetscInt s, ObsCtx *user) {
    const Q1GradRef  du    = Q1DEval(uu,r,s),
                     dchiL = Q1dchi[L][r][s];
    return Q1GradInnerProd(du,dchiL)
           - Q1Eval(ff,r,s) * Q1chi[L][r][s];
}

// compute F_ij, the discretized nonlinear operator:
//     F(u)[v] = int_Omega grad u . grad v - f v
// giving the vector of nodal values (residuals w/o ell)
//     F_ij = F(u)[psi_ij]
// where i,j is a node and psi_ij is the hat function
// at boundary nodes we have
//     F_ij = u_ij - g(x_i,y_j)
PetscErrorCode ApplyOperatorF(DM da, Vec u, Vec F, ObsCtx *user) {
    DMDALocalInfo    info;
    const Q1Quad1D   q = Q1gausslegendre[user->quadpts-1];
    const PetscInt   li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
    PetscInt         i, j, l, PP, QQ, r, s;
    const PetscReal  **au;
    PetscReal        hx, hy, detj, **FF, x, y, uu[4], ff[4];

    // grid info and arrays
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1),
    hy = 4.0 / (PetscReal)(info.my - 1),
    detj = 0.25 * hx * hy;
    PetscCall(DMDAVecGetArrayRead(da, u, &au));
    PetscCall(DMDAVecGetArray(da, F, &FF));

    // set up Q1 FEM tools for this grid
    PetscCall(Q1Setup(user->quadpts,da,-2.0,2.0,-2.0,2.0));

    // clear residuals (because we sum over elements)
    // and assign F for Dirichlet nodes
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = -2.0 + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = -2.0 + i * hx;
            FF[j][i] = (_NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(x,y,user)
                                               : 0.0;
        }
    }

    // Sum over elements, with upper-right node indices i,j, to compute F for
    // interior nodes.  Note we own elements down or left of owned nodes, but
    // in parallel the integral also needs to include non-owned (halo) elements
    // up or right of owned nodes.  See diagram in Chapter 9 of Bueler (2021)
    // for indexing and element ownership.
    for (j = info.ys; j <= info.ys + info.ym; j++) {
        if ((j == 0) || (j > info.my-1))  // does element actually exist?
            continue;
        y = -2.0 + j * hy;
        for (i = info.xs; i <= info.xs + info.xm; i++) {
            if ((i == 0) || (i > info.mx-1))  // does element actually exist?
                continue;
            x = -2.0 + i * hx;
            // this element, down or left of node i,j, is adjacent to an owned
            // and interior node; so get values of rhs f at corners of element
            ff[0] = user->f_rhs(x,y,user);
            ff[1] = user->f_rhs(x-hx,y,user);
            ff[2] = user->f_rhs(x-hx,y-hy,user);
            ff[3] = user->f_rhs(x,y-hy,user);
            // get values of iterate u at corners of element, using Dirichlet
            // value if known (to generate symmetric Jacobian)
            uu[0] = _NodeOnBdry(info,i,  j)
                    ? user->g_bdry(x,y,user)       : au[j][i];
            uu[1] = _NodeOnBdry(info,i-1,j)
                    ? user->g_bdry(x-hx,y,user)    : au[j][i-1];
            uu[2] = _NodeOnBdry(info,i-1,j-1)
                    ? user->g_bdry(x-hx,y-hy,user) : au[j-1][i-1];
            uu[3] = _NodeOnBdry(info,i,  j-1)
                    ? user->g_bdry(x,y-hy,user)    : au[j-1][i];
            // loop over corners of element i,j; l is local (elementwise)
            // index of the corner and PP,QQ are global indices of same corner
            for (l = 0; l < 4; l++) {
                PP = i + li[l];
                QQ = j + lj[l];
                // only update residual if we own node and it is not boundary
                if (PP >= info.xs && PP < info.xs + info.xm
                    && QQ >= info.ys && QQ < info.ys + info.ym
                    && !_NodeOnBdry(info,PP,QQ)) {
                    // loop over quadrature points to contribute to residual
                    // for this l corner of this i,j element
                    for (r = 0; r < q.n; r++) {
                        for (s = 0; s < q.n; s++) {
                            FF[QQ][PP] += detj * q.w[r] * q.w[s]
                                          * _IntegrandRef(hx,hy,l,uu,ff,r,s,user);
                        }
                    }
                }
            }
        }
    }

    PetscCall(DMDAVecRestoreArrayRead(da, u, &au));
    PetscCall(DMDAVecRestoreArray(da, F, &FF));

    // FLOPS: only count flops per quadrature point in residual computations:
    //   4 + 30 = 34
    // note q.n^2 quadrature points per element
    PetscCall(PetscLogFlops(34.0 * q.n * q.n * info.xm * info.ym));
    (user->residualcount)++;
    return 0;
}

// FIXME want complementarity residual too?

// for owned, interior nodes i,j, we define the pointwise residual corresponding
// to the hat function psi_ij:
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
//            = int_Omega (grad u + c grad psi_ij) . grad psi_ij - f psi_ij
//              - b_ij
// note b_ij is outside the integral; also note
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij
// note the integral is over four elements, each with four quadrature
// points "+" (in the default -lb_quadpts 2 case), which we traverse in the
// order 0,1,2,3 given:
//     j+1  *-------*-------*
//          | +   + | +   + |
//          |   1   |   0   |
//          | +   + | +   + |
//     j    *-------*-------*
//          | +   + | +   + |
//          |   2   |   3   |
//          | +   + | +   + |
//     j-1  *-------*-------*
//         i-1      i      i+1
// and each of the four elements has nodes with local (reference element)
// indices:
//        1 *-------* 0
//          |       |
//          |       |
//          |       |
//        2 *-------* 3

// evaluate integrand of rho(c) at a point xi,eta in the reference element,
// for the hat function at corner L (i.e. chi_L = psi_ij from caller)
// FLOPS: 2 + (16 + 5 + 4 + 7 + 5) = 39
PetscErrorCode _rhoIntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                 PetscReal c, const PetscReal uu[4], const PetscReal ff[4],
                 PetscInt r, PetscInt s,
                 PetscReal *rho, PetscReal *drhodc, ObsCtx *user) {
    const Q1GradRef du    = Q1DEval(uu,r,s),
                    dchiL = Q1dchi[L][r][s];
    if (rho)
        *rho = Q1GradInnerProd(Q1GradAXPY(c,dchiL,du),dchiL)
               - Q1Eval(ff,r,s) * Q1chi[L][r][s];
    if (drhodc)
        *drhodc = Q1GradInnerProd(dchiL,dchiL);
    return 0;
}

// for owned, interior node i,j, evaluate rho(c) and rho'(c)
PetscErrorCode _rhoFcn(DMDALocalInfo *info, PetscInt i, PetscInt j,
                       PetscReal c, PetscReal **au,
                       PetscReal *rho, PetscReal *drhodc, ObsCtx *user) {
    // i+oi[k],j+oj[k] gives index of upper-right node of element k
    // ll[k] gives local (ref. element) index of the i,j node on the k element
    const PetscInt  oi[4] = {1, 0, 0, 1},
                    oj[4] = {1, 1, 0, 0},
                    ll[4] = {2, 3, 0, 1};
    const PetscReal hx = 4.0 / (PetscReal)(info->mx - 1),
                    hy = 4.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    const Q1Quad1D  q = Q1gausslegendre[user->quadpts-1];
    PetscInt  k, ii, jj, r, s;
    PetscReal x, y, uu[4], ff[4], prho, pdrhodc, tmp;

    *rho = 0.0;
    if (drhodc)
        *drhodc = 0.0;
    // loop around 4 elements adjacent to global index node i,j
    for (k=0; k < 4; k++) {
        // global index of this element
        ii = i + oi[k];
        jj = j + oj[k];
        // field values for f, b, and u on this element
        x = -2.0 + ii * hx;
        y = -2.0 + jj * hy;
        ff[0] = user->f_rhs(x,y,user);
        ff[1] = user->f_rhs(x-hx,y,user);
        ff[2] = user->f_rhs(x-hx,y-hy,user);
        ff[3] = user->f_rhs(x,y-hy,user);
        uu[0] = _NodeOnBdry(*info,ii,  jj)   ?
                user->g_bdry(x,y,user)       : au[jj][ii];
        uu[1] = _NodeOnBdry(*info,ii-1,jj)   ?
                user->g_bdry(x-hx,y,user)    : au[jj][ii-1];
        uu[2] = _NodeOnBdry(*info,ii-1,jj-1) ?
                user->g_bdry(x-hx,y-hy,user) : au[jj-1][ii-1];
        uu[3] = _NodeOnBdry(*info,ii,  jj-1) ?
                user->g_bdry(x,y-hy,user)    : au[jj-1][ii];
        // loop over quadrature points in this element, summing to get rho
        for (r = 0; r < q.n; r++) {
            for (s = 0; s < q.n; s++) {
                // ll[k] is local (elementwise) index of the corner (= i,j)
                _rhoIntegrandRef(hx,hy,ll[k],c,uu,ff,r,s,&prho,&pdrhodc,user);
                tmp = detj * q.w[r] * q.w[s];
                *rho += tmp * prho;
                if (drhodc)
                    *drhodc += tmp * pdrhodc;
            }
        }
    }
    // FLOPS per quadrature point in the four elements: 6 + 39 = 45
    PetscCall(PetscLogFlops(45.0 * q.n * q.n * 4.0));
    return 0;
}

// do projected nonlinear Gauss-Seidel (processor-block) sweeps on equation
//     F(u)[psi_ij] = ell_ij   for all nodes i,j
// where psi_ij is the hat function,
//     F(u)[v] = int_Omega grad u . grad v - f v,
// ell is a nodal field (linear functional)
// for each interior node i,j we define
//     rho(c) = F(u + c psi_ij)[psi_ij] - ell_ij
// and do Newton iterations
//     c <-- c - rho(c) / rho'(c)
// followed by the bi-obstacle projection,
//     c <-- min{c, upper_ij - u_ij}
//     c <-- max{c, lower_ij - u_ij}
// note
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij
// also note that for boundary nodes we simply set
//     u_ij = g(x_i,y_j)
PetscErrorCode ProjectedNGS(LDC *ldc, PetscBool updcs, Vec ell, Vec u,
                            ObsCtx *user) {
    PetscInt        i, j, k, totalits=0, l;
    const PetscReal **aell, **aupper, **alower;
    PetscBool       haveupper = PETSC_FALSE, havelower = PETSC_FALSE;
    PetscReal       x, y, hx, hy, c, rho, drhodc,
                    **au;
    DMDALocalInfo   info;
    Vec             uloc;

    PetscCall(Q1Setup(user->quadpts,ldc->dal,-2.0,2.0,-2.0,2.0));

    // for Dirichlet nodes assign boundary value once; assumes g >= gamma_lower
    PetscCall(DMDAGetLocalInfo(ldc->dal,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1);
    hy = 4.0 / (PetscReal)(info.my - 1);
    PetscCall(DMDAVecGetArray(ldc->dal,u,&au));
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = -2.0 + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = -2.0 + i * hx;
            if (_NodeOnBdry(info,i,j))
                au[j][i] = user->g_bdry(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(ldc->dal,u,&au));

    // set-up for bi-obstacles
    if (updcs) {
        if (ldc->chiupp) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->chiupp,&aupper));
            haveupper = PETSC_TRUE;
        }
        if (ldc->chilow) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->chilow,&alower));
            havelower = PETSC_TRUE;
        }
    } else {
        if (ldc->phiupp) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->phiupp,&aupper));
            haveupper = PETSC_TRUE;
        }
        if (ldc->philow) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->philow,&alower));
            havelower = PETSC_TRUE;
        }
    }

    if (ell)
        PetscCall(DMDAVecGetArrayRead(ldc->dal,ell,&aell));
    // need local vector for stencil width in parallel
    PetscCall(DMGetLocalVector(ldc->dal,&uloc));

    // NGS sweeps over interior nodes
    for (l=0; l<user->sweeps; l++) {
        // update ghosts
        PetscCall(DMGlobalToLocal(ldc->dal,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(ldc->dal,uloc,&au));
        for (j = info.ys; j < info.ys + info.ym; j++) {
            y = -2.0 + j * hy;
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (_NodeOnBdry(info,i,j))
                    continue;
                x = -2.0 + i * hx;
                // i,j is owned interior node; do projected Newton iterations
                c = 0.0;
                for (k = 0; k < user->maxits; k++) {
                    // evaluate rho(c) and rho'(c) for current c
                    PetscCall(_rhoFcn(&info,i,j,c,au,&rho,&drhodc,user));
                    if (ell)
                        rho -= aell[j][i];
                    c = c - rho / drhodc;  // Newton step
                    // bi-obstacle projection
                    if (haveupper)
                        c = PetscMin(c, aupper[j][i] - au[j][i]);
                    else if (havelower)
                        c = PetscMax(c, alower[j][i] - au[j][i]);
                    totalits++;
                }
                au[j][i] += c;
            }
        }
        PetscCall(DMDAVecRestoreArray(ldc->dal,uloc,&au));
        PetscCall(DMLocalToGlobal(ldc->dal,uloc,INSERT_VALUES,u));
    }

    PetscCall(DMRestoreLocalVector(ldc->dal,&uloc));
    if (updcs) {
        if (ldc->chiupp)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->chiupp,&aupper));
        if (ldc->chilow)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->chilow,&alower));
    } else {
        if (ldc->phiupp)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->phiupp,&aupper));
        if (ldc->philow)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->philow,&alower));
    }
    if (ell)
        PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ell,&aell));

    // add flops for Newton iteration arithmetic; note rhoFcn() already counts flops
    PetscCall(PetscLogFlops(8 * totalits));
    (user->ngscount)++;
    return 0;
}
