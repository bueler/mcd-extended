static char help[] =
"Solve classical obstacle problem by nonlinear multilevel constraint\n"
"decomposition (NMCD) method using Q1 finite elements in 2D square (-2,2)^2\n"
"on a structured-grid (DMDA).  Problem is\n"
"  - nabla^2 u = f(x,y),  u >= psi(x,y),\n"
"subject to Dirichlet boundary conditions u=g.  Smoother and coarse-level\n"
"solver is projected, nonlinear Gauss-Seidel (PNGS) sweeps.  Option prefix\n"
"nm_.  Compare obstaclesl.c.\n\n";

#include <petsc.h>
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

extern PetscErrorCode AssertBoundaryAdmissible(DMDALocalInfo*, ObsCtx*);
extern PetscErrorCode FormExact(DMDALocalInfo*, PetscReal (*)(PetscReal,PetscReal,void*),
                                Vec, ObsCtx*);
extern PetscErrorCode FormResidualOrCRLocal(DMDALocalInfo*, PetscReal **,
                                            PetscReal**, ObsCtx*);
extern PetscErrorCode ProjectedNGS(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    LDC            *ldc;
    Vec            gamlow, u, uexact;
    ObsCtx         ctx;
    DMDALocalInfo  finfo;
    PetscInt       levels, l, maxit;
    PetscBool      ldcinfo = PETSC_FALSE, counts = PETSC_FALSE, view = PETSC_FALSE,
                   admis;
    PetscLogDouble lflops, flops;
    PetscReal      errinf;
    char           viewname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    ctx.gamma_lower = &gamma_lower;
    ctx.f_rhs = &fg_zero;
    ctx.g_bdry = &u_exact;
    ctx.residualcount = 0;
    ctx.ngscount = 0;
    ctx.quadpts = 2;
    levels = 2;
    maxit = 1;
    PetscOptionsBegin(PETSC_COMM_WORLD,"nm_","obstacle problem NMCD solver options","");
    PetscCall(PetscOptionsBool("-counts","print counts for calls to call-back functions",
                            "nmcd.c",counts,&counts,NULL));
    PetscCall(PetscOptionsBool("-info","print info on LDC (MCD) actions",
                            "nmcd.c",ldcinfo,&ldcinfo,NULL));
    PetscCall(PetscOptionsInt("-levels","number NMCD levels (>= 1)",
                            "nmcd.c",levels,&levels,NULL));
    PetscCall(PetscOptionsInt("-max_it","maximum number of NMCD V-cycles",
                            "nmcd.c",maxit,&maxit,NULL));
    // WARNING: coarse problems are badly solved with -nm_quadpts 1
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only)",
                            "nmcd.c",ctx.quadpts,&(ctx.quadpts),NULL));
    PetscCall(PetscOptionsString("-view","custom view of solution,residual,cr in ascii_matlab",
                            "nmcd",viewname,viewname,sizeof(viewname),&view));
    PetscOptionsEnd();

    // options consistency checking
    if (levels < 1) {
        SETERRQ(PETSC_COMM_SELF,1,"levels >= 1 required");
    }
    if (ctx.quadpts < 1 || ctx.quadpts > 3) {
        SETERRQ(PETSC_COMM_SELF,2,"quadrature points n=1,2,3 only");
    }

    // create LDC stack: create coarsest, then refine levels-1 times
    PetscCall(PetscMalloc1(levels,&ldc));
    PetscCall(LDCCreateCoarsest(ldcinfo,3,3,-2.0,2.0,-2.0,2.0,&(ldc[0])));
    for (l=1; l<levels; l++)
        PetscCall(LDCRefine(&(ldc[l-1]),&(ldc[l])));
    for (l=0; l<levels; l++)
        PetscCall(DMSetApplicationContext(ldc[l].dal,&ctx));
    PetscCall(DMDAGetLocalInfo(ldc[levels-1].dal,&finfo));

    // check finest-level boundary admissiblity
    PetscCall(AssertBoundaryAdmissible(&finfo,&ctx));

    // generate finest-level obstacle as Vec
    PetscCall(DMCreateGlobalVector(ldc[levels-1].dal,&gamlow));
    PetscCall(LDCVecFromFormula(ldc[levels-1],gamma_lower,gamlow,&ctx));

    // initial iterate
    PetscCall(DMCreateGlobalVector(ldc[levels-1].dal,&u));
    PetscCall(FormExact(&finfo,u_exact,u,&ctx));  // FIXME initializing with exact solution
    //PetscCall(VecSet(u,0.0));
    PetscCall(LDCVecLessThanOrEqual(ldc[levels-1],gamlow,u,&admis));
    if (!admis) {
        SETERRQ(PETSC_COMM_SELF,3,"initial iterate u is not admissible\n");
    }

    // complete set-up of LDC stack for initial iterate
    PetscCall(LDCFinestUpDCsFromVecs(u,NULL,gamlow,&(ldc[levels-1])));
    PetscCall(LDCGenerateDCsVCycle(&(ldc[levels-1])));

    // solve the problem
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"*NOT YET* SOLVING PROBLEM BY %d V-CYCLES\n",maxit));
    //FIXME replace SNESSolve() with V-cycle action

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
    PetscCall(DMCreateGlobalVector(ldc[levels-1].dal,&uexact));
    PetscCall(FormExact(&finfo,u_exact,uexact,&ctx));
    PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
    PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                          finfo.mx,finfo.my,errinf));

    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&gamlow));
    PetscCall(VecDestroy(&u));
    for (l=levels-1; l>=0; l--)
        PetscCall(LDCDestroy(&(ldc[l])));
    PetscCall(PetscFree(ldc));
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

PetscErrorCode FormExact(DMDALocalInfo *info, PetscReal (*ufcn)(PetscReal,PetscReal,void*),
                         Vec u, ObsCtx* user) {
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

PetscBool _NodeOnBdry(DMDALocalInfo *info, PetscInt i, PetscInt j) {
    return (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1);
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
            if (!_NodeOnBdry(info,i,j)) {
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
            FF[j][i] = (_NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(x,y,user)
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
                if (PP >= info->xs && PP < info->xs + info->xm
                    && QQ >= info->ys && QQ < info->ys + info->ym
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

    // FIXME in PNGS mode we report the complementarity residual instead
//    if (user->pngs)
//        PetscCall(CRLocal(info,au,FF,FF,user));

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
        uu[0] = _NodeOnBdry(info,ii,  jj)   ?
                user->g_bdry(x,y,user)       : au[jj][ii];
        uu[1] = _NodeOnBdry(info,ii-1,jj)   ?
                user->g_bdry(x-hx,y,user)    : au[jj][ii-1];
        uu[2] = _NodeOnBdry(info,ii-1,jj-1) ?
                user->g_bdry(x-hx,y-hy,user) : au[jj-1][ii-1];
        uu[3] = _NodeOnBdry(info,ii,  jj-1) ?
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
            if (_NodeOnBdry(&info,i,j))
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
                if (_NodeOnBdry(&info,i,j))
                    continue;
                x = -2.0 + i * hx;
                // i,j is owned interior node; do projected Newton iterations
                c = 0.0;
                for (k = 0; k < maxits; k++) {
                    // evaluate rho(c) and rho'(c) for current c
                    PetscCall(_rhoFcn(&info,i,j,c,au,&rho,&drhodc,user));
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
                    PetscCall(_rhoFcn(&info,i,j,c,au,&rho,NULL,user));
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
