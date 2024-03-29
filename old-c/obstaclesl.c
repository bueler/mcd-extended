static char help[] =
"Solve classical obstacle problem by Q1 finite elements in 2D square (-2,2)^2\n"
"using a structured-grid (DMDA):\n"
"  - nabla^2 u = f(x,y),  u >= psi(x,y)\n"
"subject to Dirichlet boundary conditions u=g.  Implements projected, nonlinear\n"
"Gauss-Seidel (PNGS) sweeps, a non-scalable single-level method.  Also allows\n"
"SNESVI solutions.  Option prefix ob_.  The same problem, with exact\n"
"solution, is solved as in Chapter 12 of Bueler (2021), 'PETSc for Partial\n"
"Differential Equations', SIAM Press.\n\n";

#include <petsc.h>
#include "src/q1fem.h"

typedef struct {
  // obstacle gamma_lower(x,y)
  PetscReal (*gamma_lower)(PetscReal x, PetscReal y, void *ctx);
  // right-hand side f(x,y)
  PetscReal (*f_rhs)(PetscReal x, PetscReal y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscReal (*g_bdry)(PetscReal x, PetscReal y, void *ctx);
  // are we in PNGS solver mode where we need to report complementarity residual
  PetscBool pngs;
  PetscInt  residualcount, ngscount, quadpts;
} ObsCtx;

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

// z = bump(x,y) is zero along boundary of Omega=(-2,2)^2 and reaches maximum
// of bump(0,0) = 1.0 in center
PetscReal bump(PetscReal x, PetscReal y, void *ctx) {
    return (x + 2.0) * (2.0 - x) * (y + 2.0) * (2.0 - y) / 16.0;
}

extern PetscErrorCode AssertBoundaryAdmissible(DMDALocalInfo*, ObsCtx*);
extern PetscErrorCode FormExact(PetscReal (*)(PetscReal,PetscReal,void*),
                                DMDALocalInfo*, Vec, ObsCtx*);
extern PetscErrorCode AddBump(PetscReal, DMDALocalInfo*, Vec, ObsCtx*);
extern PetscErrorCode FormBounds(SNES, Vec, Vec);
extern PetscBool NodeOnBdry(DMDALocalInfo*, PetscInt, PetscInt);
extern PetscErrorCode CRLocal(DMDALocalInfo*, PetscReal **, PetscReal **,
                              PetscReal**, ObsCtx*);
extern PetscErrorCode FormResidualOrCRLocal(DMDALocalInfo*, PetscReal **,
                                            PetscReal**, ObsCtx*);
extern PetscErrorCode ProjectedNGS(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes, npc;
    SNESLineSearch ls;
    KSP            ksp;
    Vec            u, uexact;
    ObsCtx         ctx;
    DMDALocalInfo  info;
    PetscBool      counts = PETSC_FALSE, view = PETSC_FALSE,
                   initialzero = PETSC_FALSE;
    PetscLogDouble lflops, flops;
    PetscReal      bumpsize  = 1.0, errinf;
    char           viewname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    ctx.gamma_lower = &gamma_lower;
    ctx.f_rhs = &fg_zero;
    ctx.g_bdry = &u_exact;
    ctx.pngs = PETSC_FALSE;
    ctx.residualcount = 0;
    ctx.ngscount = 0;
    ctx.quadpts = 2;
    PetscOptionsBegin(PETSC_COMM_WORLD,"ob_","obstacle problem solver options","");
    // WARNING: coarse problems are badly solved with -ob_quadpts 1, so avoid in MG
    PetscCall(PetscOptionsBool("-counts","print counts for calls to call-back functions",
                            "obstaclesl.c",counts,&counts,NULL));
    PetscCall(PetscOptionsReal("-initialbump","initialize with admissible: exact solution plus bump of this size",
                            "obstaclesl.c",bumpsize,&(bumpsize),NULL));
    PetscCall(PetscOptionsBool("-initialzero","initial iterate is inadmissible u=0",
                            "obstaclesl.c",initialzero,&initialzero,NULL));
    PetscCall(PetscOptionsBool("-pngs","only do sweeps of projected nonlinear Gauss-Seidel",
                            "obstaclesl.c",ctx.pngs,&(ctx.pngs),NULL));
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only)",
                            "obstaclesl.c",ctx.quadpts,&(ctx.quadpts),NULL));
    PetscCall(PetscOptionsString("-view","custom view of solution,residual,cr in ascii_matlab",
                            "obstaclesl",viewname,viewname,sizeof(viewname),&view));
    PetscOptionsEnd();

    // options consistency checking
    if (ctx.quadpts < 1 || ctx.quadpts > 3) {
        SETERRQ(PETSC_COMM_SELF,3,"quadrature points n=1,2,3 only");
    }

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&ctx));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,0.0,1.0));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(AssertBoundaryAdmissible(&info,&ctx));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetApplicationContext(snes,&ctx));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormResidualOrCRLocal,&ctx));
    if (ctx.pngs) {
        // defaults equivalent to
        //   -snes_type nrichardson -npc_snes_type ngs -snes_linesearch_type basic
        PetscCall(SNESSetType(snes,SNESNRICHARDSON));
        PetscCall(SNESSetNGS(snes,ProjectedNGS,&ctx));
        PetscCall(SNESGetNPC(snes,&npc));
        PetscCall(SNESSetType(npc,SNESNGS));
        // no need to call SNESSetNGS() on npc ... it is passed down
        PetscCall(SNESGetLineSearch(snes,&ls));
        PetscCall(SNESLineSearchSetType(ls,SNESLINESEARCHBASIC));
    } else {
        // SNESVI solver, defaults to reduced-space (RS) type, with provided
        // bounds, and CG solution of step equations:
        //   -snes_type vinewtonrsls -ksp_type cg
        PetscCall(SNESSetType(snes,SNESVINEWTONRSLS));
        PetscCall(SNESVISetComputeVariableBounds(snes,&FormBounds));
        PetscCall(SNESGetKSP(snes,&ksp));
        PetscCall(KSPSetType(ksp,KSPCG));
    }
    PetscCall(SNESSetFromOptions(snes));

    // set initial iterate
    PetscCall(DMCreateGlobalVector(da,&u));
    if (initialzero)
        PetscCall(VecSet(u,0.0));
    else {
        PetscCall(FormExact(u_exact,&info,u,&ctx));
        PetscCall(AddBump(bumpsize,&info,u,&ctx));
    }

    // solve the problem
    PetscCall(SNESSolve(snes,NULL,u));
    PetscCall(VecDestroy(&u));
    PetscCall(DMDestroy(&da));

    if (counts) {
        // note calls to FormResidualOrCRLocal() and ProjectedNGS() are
        // collective but flops are per-process, so we need a reduction
        PetscCall(PetscGetFlops(&lflops));
        PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,
                                PetscObjectComm((PetscObject)snes)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "flops = %.3e,  residual calls = %d,  PNGS calls = %d\n",
                              flops,ctx.residualcount,ctx.ngscount));
    }

    // get post-solution stuff
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(SNESGetSolution(snes,&u));  // SNES owns u; we do not destroy it

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
        PetscCall(DMDAVecGetArrayRead(da, u, &au));
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
        PetscCall(DMDAVecRestoreArrayRead(da, u, &au));
        PetscCall(PetscObjectSetName((PetscObject)F,"Fhat"));
        PetscCall(VecView(F,file));
        PetscCall(VecDestroy(&F));
        PetscCall(PetscViewerPopFormat(file));
        PetscCall(PetscViewerDestroy(&file));
    }

    // report on numerical error
    PetscCall(DMCreateGlobalVector(da,&uexact));
    PetscCall(FormExact(u_exact,&info,uexact,&ctx));
    PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                          info.mx,info.my,errinf));

    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode AssertBoundaryAdmissible(DMDALocalInfo *info, ObsCtx* user) {
    PetscInt     i, j;
    PetscReal    hx, hy, x, y;
    hx = 4.0 / (PetscReal)(info->mx - 1);
    hy = 4.0 / (PetscReal)(info->my - 1);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -2.0 + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
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

PetscErrorCode FormExact(PetscReal (*ufcn)(PetscReal,PetscReal,void*),
                         DMDALocalInfo *info, Vec u, ObsCtx* user) {
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    hx = 4.0 / (PetscReal)(info->mx - 1);
    hy = 4.0 / (PetscReal)(info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -2.0 + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -2.0 + i * hx;
            au[j][i] = (*ufcn)(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}

PetscErrorCode AddBump(PetscReal bumpsize, DMDALocalInfo *info, Vec u, ObsCtx* user) {
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    hx = 4.0 / (PetscReal)(info->mx - 1);
    hy = 4.0 / (PetscReal)(info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -2.0 + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -2.0 + i * hx;
            au[j][i] += bumpsize * bump(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}

// tell SNESVI we want  gamma_lower <= u < +infinity;  not used when doing pNGS sweeps
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
    DM             da;
    DMDALocalInfo  info;
    PetscInt       i, j;
    PetscReal      **aXl, dx, dy, x, y;
    void           *ctx;
    ObsCtx         *user;
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(SNESGetApplicationContext(snes,&ctx));
    user = (ObsCtx*)ctx;
    PetscCall(DMDAGetLocalInfo(da,&info));
    dx = 4.0 / (PetscReal)(info.mx-1);
    dy = 4.0 / (PetscReal)(info.my-1);
    PetscCall(DMDAVecGetArray(da, Xl, &aXl));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = -2.0 + j * dy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = -2.0 + i * dx;
            aXl[j][i] = gamma_lower(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, Xl, &aXl));
    PetscCall(VecSet(Xu,PETSC_INFINITY));
    return 0;
}

PetscBool NodeOnBdry(DMDALocalInfo *info, PetscInt i, PetscInt j) {
    return ((i == 0 || i == info->mx-1 || j == 0 || j == info->my-1));
}

// compute complementarity residual Fhat from conventional residual F:
//     Fhat_ij = F_ij         if u_ij > gamma_lower(x_i,y_j)
//               min{F_ij,0}  if u_ij <= gamma_lower(x_i,y_j)
// note arrays aF and aFhat can be the same, giving in-place calculation
// note this op is idempotent if u,gamma_lower unchanged:  CRLocal(CRLocal()) = CRLocal()
PetscErrorCode CRLocal(DMDALocalInfo *info, PetscReal **au, PetscReal **aF,
                       PetscReal **aFhat, ObsCtx *user) {
    const PetscReal hx = 4.0 / (PetscReal)(info->mx - 1),
                    hy = 4.0 / (PetscReal)(info->my - 1);
    PetscInt        i, j;
    PetscReal       x, y;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = -2.0 + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (!NodeOnBdry(info,i,j)) {
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
PetscReal IntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                       const PetscReal uu[4], const PetscReal ff[4],
                       PetscInt r, PetscInt s, ObsCtx *user) {
    const Q1GradRef  du    = Q1DEval(uu,r,s),
                     dchiL = Q1dchi[L][r][s];
    return Q1GradInnerProd(du,dchiL)
           - Q1Eval(ff,r,s) * Q1chi[L][r][s];
}

// compute F_ij, the nodal residual of the discretized nonlinear operator,
// or the complementarity residual if user->pngs is true
// the residual is
//     F(u)[v] = int_Omega grad u . grad v - f v
// giving the vector of nodal residuals
//     F_ij = F(u)[psi_ij]
// where i,j is a node and psi_ij is the hat function; at boundary nodes
// we have
//     F_ij = u_ij - g(x_i,y_j)
PetscErrorCode FormResidualOrCRLocal(DMDALocalInfo *info, PetscReal **au,
                                     PetscReal **FF, ObsCtx *user) {
    const Q1Quad1D  q = Q1gausslegendre[user->quadpts-1];
    const PetscInt  li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
    const PetscReal hx = 4.0 / (PetscReal)(info->mx - 1),
                    hy = 4.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    PetscInt   i, j, l, PP, QQ, r, s;
    PetscReal  x, y, uu[4], ff[4];

    // set up Q1 FEM tools for this grid
    PetscCall(Q1Setup(user->quadpts,info->da,-2.0,2.0,-2.0,2.0));

    // clear residuals (because we sum over elements)
    // and assign F for Dirichlet nodes
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = -2.0 + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = -2.0 + i * hx;
            FF[j][i] = (NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(x,y,user)
                                              : 0.0;
        }
    }

    // Sum over elements, with upper-right node indices i,j, to compute F for
    // interior nodes.  Note we own elements down or left of owned nodes, but
    // in parallel the integral also needs to include non-owned (halo) elements
    // up or right of owned nodes.  See diagram in Chapter 9 of Bueler (2021)
    // for indexing and element ownership.
    for (j = info->ys; j <= info->ys + info->ym; j++) {
        if ((j == 0) || (j > info->my-1))  // does element actually exist?
            continue;
        y = -2.0 + j * hy;
        for (i = info->xs; i <= info->xs + info->xm; i++) {
            if ((i == 0) || (i > info->mx-1))  // does element actually exist?
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
            uu[0] = NodeOnBdry(info,i,  j)
                    ? user->g_bdry(x,y,user)       : au[j][i];
            uu[1] = NodeOnBdry(info,i-1,j)
                    ? user->g_bdry(x-hx,y,user)    : au[j][i-1];
            uu[2] = NodeOnBdry(info,i-1,j-1)
                    ? user->g_bdry(x-hx,y-hy,user) : au[j-1][i-1];
            uu[3] = NodeOnBdry(info,i,  j-1)
                    ? user->g_bdry(x,y-hy,user)    : au[j-1][i];
            // loop over corners of element i,j; l is local (elementwise)
            // index of the corner and PP,QQ are global indices of same corner
            for (l = 0; l < 4; l++) {
                PP = i + li[l];
                QQ = j + lj[l];
                // only update residual if we own node and it is not boundary
                if (PP >= info->xs && PP < info->xs + info->xm
                    && QQ >= info->ys && QQ < info->ys + info->ym
                    && !NodeOnBdry(info,PP,QQ)) {
                    // loop over quadrature points to contribute to residual
                    // for this l corner of this i,j element
                    for (r = 0; r < q.n; r++) {
                        for (s = 0; s < q.n; s++) {
                            FF[QQ][PP] += detj * q.w[r] * q.w[s]
                                          * IntegrandRef(hx,hy,l,uu,ff,r,s,user);
                        }
                    }
                }
            }
        }
    }

    // in PNGS mode we report the complementarity residual instead
    if (user->pngs)
        PetscCall(CRLocal(info,au,FF,FF,user));

    // FLOPS: only count flops per quadrature point in residual computations:
    //   4 + 30 = 34
    // note q.n^2 quadrature points per element
    PetscCall(PetscLogFlops(34.0 * q.n * q.n * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

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
PetscErrorCode rhoIntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
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
PetscErrorCode rhoFcn(DMDALocalInfo *info, PetscInt i, PetscInt j,
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
        uu[0] = NodeOnBdry(info,ii,  jj)   ?
                user->g_bdry(x,y,user)       : au[jj][ii];
        uu[1] = NodeOnBdry(info,ii-1,jj)   ?
                user->g_bdry(x-hx,y,user)    : au[jj][ii-1];
        uu[2] = NodeOnBdry(info,ii-1,jj-1) ?
                user->g_bdry(x-hx,y-hy,user) : au[jj-1][ii-1];
        uu[3] = NodeOnBdry(info,ii,  jj-1) ?
                user->g_bdry(x,y-hy,user)    : au[jj-1][ii];
        // loop over quadrature points in this element, summing to get rho
        for (r = 0; r < q.n; r++) {
            for (s = 0; s < q.n; s++) {
                // ll[k] is local (elementwise) index of the corner (= i,j)
                rhoIntegrandRef(hx,hy,ll[k],c,uu,ff,r,s,&prho,&pdrhodc,user);
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
//     F(u)[psi_ij] = b_ij   for all nodes i,j
// where psi_ij is the hat function,
//     F(u)[v] = int_Omega grad u . grad v - f v,
// b is a nodal field provided by the call-back,
// and the projection onto the obstacle is nodal:
//     u_ij <-- max{u_ij, gamma_lower_ij}
// for each interior node i,j we define
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
// and do Newton iterations
//     c <-- c - rho(c) / rho'(c)
// followed by the projection,
//     c <-- max{c, gamma_lower_ij - u_ij}
// note
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij
// also note that for boundary nodes we simply set
//     u_ij = g(x_i,y_j)
PetscErrorCode ProjectedNGS(SNES snes, Vec u, Vec b, void *ctx) {
    ObsCtx*        user = (ObsCtx*)ctx;
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    const PetscReal **ab;
    PetscReal      x, y, atol, rtol, stol, hx, hy, **au,
                   c, rho, rho0, drhodc, s, cold, glij;
    DM             da;
    DMDALocalInfo  info;
    Vec            uloc;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits));
    //PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    //    "in PNGS: sweepw=%.d, atol=%.3e, rtol=%.3e, stol=%.3e, maxits=%d\n",
    //    sweeps,atol,rtol,stol,maxits));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(Q1Setup(user->quadpts,da,-2.0,2.0,-2.0,2.0));

    // for Dirichlet nodes assign boundary value once; assumes g >= gamma_lower
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1);
    hy = 4.0 / (PetscReal)(info.my - 1);
    PetscCall(DMDAVecGetArray(da,u,&au));
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = -2.0 + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = -2.0 + i * hx;
            if (NodeOnBdry(&info,i,j))
                au[j][i] = user->g_bdry(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(da,u,&au));

    if (b)
        PetscCall(DMDAVecGetArrayRead(da,b,&ab));
    // need local vector for stencil width in parallel
    PetscCall(DMGetLocalVector(da,&uloc));

    // NGS sweeps over interior nodes
    for (l=0; l<sweeps; l++) {
        // update ghosts
        PetscCall(DMGlobalToLocal(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(da,uloc,&au));
        for (j = info.ys; j < info.ys + info.ym; j++) {
            y = -2.0 + j * hy;
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (NodeOnBdry(&info,i,j))
                    continue;
                x = -2.0 + i * hx;
                // i,j is owned interior node; do projected Newton iterations
                c = 0.0;
                for (k = 0; k < maxits; k++) {
                    // evaluate rho(c) and rho'(c) for current c
                    PetscCall(rhoFcn(&info,i,j,c,au,&rho,&drhodc,user));
                    if (b)
                        rho -= ab[j][i];
                    if (k == 0)
                        rho0 = rho;
                    s = - rho / drhodc;  // Newton step
                    cold = c;
                    c += s;
                    // do projection
                    glij = user->gamma_lower(x,y,user);
                    c = PetscMax(c, glij - au[j][i]);
                    // redefine s as magnitude of actual step taken
                    s = PetscAbsReal(c - cold);
                    // recompute rho as complementarity residual
                    PetscCall(rhoFcn(&info,i,j,c,au,&rho,NULL,user));
                    if (au[j][i] + c <= glij)  // if i,j now active,
                        rho = PetscMin(rho, 0.0);  // only punish negative F values
                    totalits++;
                    if (   atol > PetscAbsReal(rho)
                        || rtol*PetscAbsReal(rho0) > PetscAbsReal(rho)
                        || stol*PetscAbsReal(c) > PetscAbsReal(s)    ) {
                        break;
                    }
                }
                au[j][i] += c;
            }
        }
        PetscCall(DMDAVecRestoreArray(da,uloc,&au));
        PetscCall(DMLocalToGlobal(da,uloc,INSERT_VALUES,u));
    }

    PetscCall(DMRestoreLocalVector(da,&uloc));
    if (b)
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));

    // add flops for Newton iteration arithmetic; note rhoFcn() already counts flops
    PetscCall(PetscLogFlops(8 * totalits));
    (user->ngscount)++;
    return 0;
}
