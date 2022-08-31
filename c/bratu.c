static char help[] =
"Solve nonlinear Liouville-Bratu equation\n"
"  - nabla^2 u - lambda e^u = f(x,y)\n"
"on the 2D square (0,1)^2, subject to Dirichlet boundary conditions u=g.\n"
"Grid is structured and equally-spaced.  Option prefix is lb_.\n"
"Choose either method:\n"
"  * finite differences (-lb_fd)\n"
"  * Q1 finite elements (-lb_fem)\n"
"For f=0 and g=0 the critical value occurs about at lambda = 6.808.\n"
"Optional exact solutions are from Liouville (1853) (for lambda=1.0;\n"
"-lb_exact), or from method of manufactured solutions (arbitrary lambda;\n"
"-lb_mms).\n\n";

#include <petsc.h>
#include "src/q1fem.h"

typedef struct {
  // right-hand side f(x,y)
  PetscReal (*f_rhs)(PetscReal x, PetscReal y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscReal (*g_bdry)(PetscReal x, PetscReal y, void *ctx);
  PetscReal lambda;
  PetscInt  residualcount, ngscount, quadpts;
} BratuCtx;

static PetscReal fg_zero(PetscReal x, PetscReal y, void *ctx) {
    return 0.0;
}

static PetscReal u_mms(PetscReal x, PetscReal y, void *ctx) {
    PetscReal xx = x * x,  yy = y * y;
    return (xx - xx * xx) * (yy * yy - yy);
}

static PetscReal f_mms(PetscReal x, PetscReal y, void *ctx) {
    BratuCtx *user = (BratuCtx*)ctx;
    PetscReal xx = x * x,  yy = y * y;
    return 2.0 * (1.0 - 6.0 * xx) * yy * (1.0 - yy)
           + 2.0 * (1.0 - 6.0 * yy) * xx * (1.0 - xx)
           - user->lambda * PetscExpScalar(u_mms(x,y,ctx));
}

static PetscReal g_liouville(PetscReal x, PetscReal y, void *ctx) {
    PetscReal r2 = (x + 1.0) * (x + 1.0) + (y + 1.0) * (y + 1.0),
              qq = r2 * r2 + 1.0,
              omega = r2 / (qq * qq);
    return PetscLogReal(32.0 * omega);
}

extern PetscErrorCode FormExact(PetscReal (*)(PetscReal,PetscReal,void*),
                                DMDALocalInfo*, Vec, BratuCtx*);
extern PetscBool NodeOnBdry(DMDALocalInfo*, PetscInt, PetscInt);

extern PetscErrorCode FormFunctionLocalFD(DMDALocalInfo*, PetscReal **,
                                          PetscReal**, BratuCtx*);
extern PetscErrorCode FormFunctionLocalFEM(DMDALocalInfo*, PetscReal **,
                                           PetscReal**, BratuCtx*);

extern PetscErrorCode NonlinearGSFD(SNES, Vec, Vec, void*);
extern PetscErrorCode NonlinearGSFEM(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes;
    Vec            u, uexact;
    BratuCtx       bctx;
    DMDALocalInfo  info;
    PetscBool      exact = PETSC_FALSE, mms = PETSC_FALSE, counts = PETSC_FALSE,
                   fd = PETSC_FALSE, fem = PETSC_FALSE;
    PetscLogDouble lflops, flops;
    PetscReal      errinf;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    bctx.f_rhs = &fg_zero;
    bctx.g_bdry = &fg_zero;
    bctx.lambda = 1.0;
    bctx.residualcount = 0;
    bctx.ngscount = 0;
    bctx.quadpts = 2;
    PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","Liouville-Bratu equation solver options","");
    PetscCall(PetscOptionsBool("-exact","use Liouville exact solution",
                            "bratu.c",exact,&exact,NULL));
    PetscCall(PetscOptionsBool("-fd","use finite differences",
                            "bratu.c",fd,&fd,NULL));
    PetscCall(PetscOptionsBool("-fem","use Q1 finite elements",
                            "bratu.c",fem,&fem,NULL));
    PetscCall(PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratu.c",bctx.lambda,&(bctx.lambda),NULL));
    PetscCall(PetscOptionsBool("-mms","use MMS exact solution",
                            "bratu.c",mms,&mms,NULL));
    // WARNING: coarse problems are badly solved with -lb_quadpts 1, so avoid in MG
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only; for Q1 FEM case only)",
                            "bratu.c",bctx.quadpts,&(bctx.quadpts),NULL));
    PetscCall(PetscOptionsBool("-counts","print counts for calls to call-back functions",
                            "bratu.c",counts,&counts,NULL));
    PetscOptionsEnd();

    // options consistency checking
    if (fd && fem) {
        SETERRQ(PETSC_COMM_SELF,1,"invalid option combination -lb_fd -lb_fem; choose one\n");
    }
    if (!fd && !fem) {
        SETERRQ(PETSC_COMM_SELF,2,"invalid option combination; choose either -lb_fd or -lb_fem\n");
    }
    if (exact && mms) {
        SETERRQ(PETSC_COMM_SELF,3,"invalid option combination -lb_exact -lb_mms; choose one or neither\n");
    }
    if (exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_SELF,4,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        bctx.g_bdry = &g_liouville;  // and zero for f_rhs()
    }
    if (mms)
        bctx.f_rhs = &f_mms;  // and zero for g_bdry()
    if ((bctx.quadpts < 1) || (bctx.quadpts > 3)) {
        SETERRQ(PETSC_COMM_SELF,5,"quadrature points n=1,2,3 only");
    }

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           fd ? DMDA_STENCIL_STAR : DMDA_STENCIL_BOX,
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&bctx));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetApplicationContext(snes,&bctx));
    PetscCall(SNESSetDM(snes,da));
    if (fd) {
        PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
                   (DMDASNESFunction)FormFunctionLocalFD,&bctx));
        PetscCall(SNESSetNGS(snes,NonlinearGSFD,&bctx));
    } else {
        PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
                   (DMDASNESFunction)FormFunctionLocalFEM,&bctx));
        PetscCall(SNESSetNGS(snes,NonlinearGSFEM,&bctx));
    }
    PetscCall(SNESSetFromOptions(snes));

    // solve the problem
    PetscCall(DMGetGlobalVector(da,&u));
    PetscCall(VecSet(u,0.0));  // initialize to zero
    PetscCall(SNESSolve(snes,NULL,u));
    PetscCall(DMRestoreGlobalVector(da,&u));
    PetscCall(DMDestroy(&da));

    if (counts) {
        PetscCall(PetscGetFlops(&lflops));
        PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,
                                PetscObjectComm((PetscObject)snes)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "flops = %.3e,  residual calls = %d,  NGS calls = %d\n",
                              flops,bctx.residualcount,bctx.ngscount));
    }

    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    if (exact || mms) {
        PetscCall(SNESGetSolution(snes,&u));  // SNES owns u; we do not destroy it
        PetscCall(DMCreateGlobalVector(da,&uexact));
        if (exact)
            PetscCall(FormExact(g_liouville,&info,uexact,&bctx));
        else
            PetscCall(FormExact(u_mms,&info,uexact,&bctx));
        PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
        PetscCall(VecDestroy(&uexact));
        PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                              info.mx,info.my,errinf));
    } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",
                              info.mx,info.my));
    }

    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormExact(PetscReal (*ufcn)(PetscReal,PetscReal,void*),
                         DMDALocalInfo *info, Vec u, BratuCtx* user) {
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    hx = 1.0 / (PetscReal)(info->mx - 1);
    hy = 1.0 / (PetscReal)(info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = i * hx;
            au[j][i] = (*ufcn)(x,y,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}

PetscBool NodeOnBdry(DMDALocalInfo *info, PetscInt i, PetscInt j) {
    return (((i == 0) || (i == info->mx-1) || (j == 0) || (j == info->my-1)));
}

// compute F(u), the residual of the discretized PDE on the given grid,
// using finite differences
PetscErrorCode FormFunctionLocalFD(DMDALocalInfo *info, PetscReal **au,
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
                           - darea * (user->lambda * PetscExpScalar(au[j][i])
                                      + user->f_rhs(i*hx,j*hy,user));
            }
        }
    }
    PetscCall(PetscLogFlops(14.0 * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

// using Q1 finite elements, evaluate the integrand of the residual on
// the reference element
// FLOPS: 5 + (16 + 7 + 5 + 7) = 40
PetscReal IntegrandRef(PetscInt L, const PetscReal uu[4], const PetscReal ff[4],
                       PetscInt r, PetscInt s, BratuCtx *user) {
    const Q1GradRef  du    = Q1DEval(uu,r,s),
                     dchiL = Q1dchi[L][r][s];
    const PetscReal  tmp = user->lambda * PetscExpScalar(Q1Eval(uu,r,s));
    return Q1GradInnerProd(du,dchiL)
           - (tmp + Q1Eval(ff,r,s)) * Q1chi[L][r][s];
}

// using Q1 finite elements, compute F(u), the residual of the discretized
// PDE on the given grid,
//     F(u)[v] = int_Omega grad u . grad v - lambda e^u v - f v
// this method computes the vector
//     F_ij = F(u)[psi_ij]
// where i,j is a node and psi_ij is the hat function there
// note that at boundary nodes we have
//     F_ij = u_ij - g(x_i,y_j)
// where g(x,y) is the boundary value
PetscErrorCode FormFunctionLocalFEM(DMDALocalInfo *info, PetscReal **au,
                                    PetscReal **FF, BratuCtx *user) {
    const Q1Quad1D  q = Q1gausslegendre[user->quadpts-1];
    const PetscInt  li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
    const PetscReal hx = 1.0 / (PetscReal)(info->mx - 1),
                    hy = 1.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    PetscInt   i, j, l, PP, QQ, r, s;
    PetscReal  uu[4], ff[4];

    // set up Q1 FEM tools for this grid
    PetscCall(Q1SetupForGrid(user->quadpts,hx,hy));

    // clear residuals (because we sum over elements)
    // and assign F for Dirichlet nodes
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            FF[j][i] = (NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(i*hx,j*hy,user)
                                              : 0.0;

    // sum over elements to compute F for interior nodes
    // we own elements down or left of owned nodes, but in parallel the integral
    // needs to include elements up or right of owned nodes (i.e. halo elements)
    for (j = info->ys; j <= info->ys + info->ym; j++) {
        if ((j == 0) || (j > info->my-1))
            continue;
        for (i = info->xs; i <= info->xs + info->xm; i++) {
            if ((i == 0) || (i > info->mx-1))
                continue;
            // this element, down-or-left of node i,j, is adjacent to an owned
            // and interior node
            // values of rhs f at corners of element
            ff[0] = user->f_rhs(i*hx,j*hy,user);
            ff[1] = user->f_rhs((i-1)*hx,j*hy,user);
            ff[2] = user->f_rhs((i-1)*hx,(j-1)*hy,user);
            ff[3] = user->f_rhs(i*hx,(j-1)*hy,user);
            // values of iterate u at corners of element, using Dirichlet
            // value if known (for symmetry of Jacobian)
            uu[0] = NodeOnBdry(info,i,  j)
                    ? user->g_bdry(i*hx,j*hy,user)         : au[j][i];
            uu[1] = NodeOnBdry(info,i-1,j)
                    ? user->g_bdry((i-1)*hx,j*hy,user)     : au[j][i-1];
            uu[2] = NodeOnBdry(info,i-1,j-1)
                    ? user->g_bdry((i-1)*hx,(j-1)*hy,user) : au[j-1][i-1];
            uu[3] = NodeOnBdry(info,i,  j-1)
                    ? user->g_bdry(i*hx,(j-1)*hy,user)     : au[j-1][i];
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
                                          * IntegrandRef(l,uu,ff,r,s,user);
                        }
                    }
                }
            }
        }
    }
    // FLOPS only counting quadrature-point residual computations:
    //     4 + 40 = 44 flops per quadrature point
    //     q.n^2 quadrature points per element
    PetscCall(PetscLogFlops(44.0 * q.n * q.n * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

// using finite differences, do nonlinear Gauss-Seidel (processor-block)
// sweeps on equation
//     F(u) = b
PetscErrorCode NonlinearGSFD(SNES snes, Vec u, Vec b, void *ctx) {
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    PetscReal      atol, rtol, stol, hx, hy, darea, hxhy, hyhx,
                   **au, **ab, bij, uu, tmp, phi0, phi, dphidu, s;
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

    if (b) {
        PetscCall(DMDAVecGetArrayRead(da,b,&ab));
    }
    PetscCall(DMGetLocalVector(da,&uloc));
    for (l=0; l<sweeps; l++) {
        PetscCall(DMGlobalToLocal(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(da,uloc,&au));
        for (j = info.ys; j < info.ys + info.ym; j++) {
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (j==0 || i==0 || i==info.mx-1 || j==info.my-1) {
                    au[j][i] = user->g_bdry(i*hx,j*hy,user);
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
                        tmp = user->lambda * PetscExpScalar(uu);
                        phi =   hyhx * (2.0 * uu - au[j][i-1] - au[j][i+1])
                              + hxhy * (2.0 * uu - au[j-1][i] - au[j+1][i])
                              - darea * (tmp + user->f_rhs(i*hx,j*hy,user))
                              - bij;
                        if (k == 0)
                             phi0 = phi;
                        dphidu = 2.0 * (hyhx + hxhy) - darea * tmp;
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
        PetscCall(DMLocalToGlobal(da,uloc,INSERT_VALUES,u));
    }
    PetscCall(DMRestoreLocalVector(da,&uloc));
    if (b) {
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));
    }
    PetscCall(PetscLogFlops(22.0 * totalits));
    (user->ngscount)++;
    return 0;
}

// using Q1 finite elements, for owned, interior nodes i,j, we define the
// pointwise residual corresponding to the hat function psi_ij:
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
//            = int_Omega (grad u + c grad psi_ij) . grad psi_ij
//                        - (lambda exp(u + c psi_ij) + f) psi_ij
//              - b_ij
// note b_ij is outside the integral
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
// FLOPS:  9 + (16 + 7 + 5 + 4 + 7 + 5) = 53
PetscErrorCode rhoIntegrandRef(PetscInt L,
                 PetscReal c, const PetscReal uu[4], const PetscReal ff[4],
                 PetscInt r, PetscInt s,
                 PetscReal *rho, PetscReal *drhodc, BratuCtx *user) {
    const Q1GradRef du     = Q1DEval(uu,r,s),
                    dchiL  = Q1dchi[L][r][s];
    const PetscReal chiL   = Q1chi[L][r][s],
                    ushift = Q1Eval(uu,r,s) + c * chiL,
                    phiL   = user->lambda * PetscExpScalar(ushift);
    *rho = Q1GradInnerProd(Q1GradAXPY(c,dchiL,du),dchiL)
           - (phiL + Q1Eval(ff,r,s)) * chiL;
    *drhodc = Q1GradInnerProd(dchiL,dchiL) - phiL * chiL;
    return 0;

}

// using Q1 finite elements, for owned, interior nodes i,j, evaluate
// rho(c) and
//   rho'(c) = int_Omega grad psi_ij . grad psi_ij
//                       - lambda e^(u + c psi_ij) psi_ij
// FLOPS: (155 + 6) * q.n * q.n
PetscErrorCode rhoFcn(DMDALocalInfo *info, PetscInt i, PetscInt j,
                      PetscReal c, PetscReal **au,
                      PetscReal *rho, PetscReal *drhodc, BratuCtx *user) {
    // i+oi[k],j+oj[k] gives index of upper-right node of element k
    // ll[k] gives local (ref. element) index of the i,j node on the k element
    const PetscInt  oi[4] = {1, 0, 0, 1},
                    oj[4] = {1, 1, 0, 0},
                    ll[4] = {2, 3, 0, 1};
    const PetscReal hx = 1.0 / (PetscReal)(info->mx - 1),
                    hy = 1.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    const Q1Quad1D  q = Q1gausslegendre[user->quadpts-1];
    PetscInt  k, ii, jj, r, s;
    PetscReal uu[4], ff[4], prho, pdrhodc, tmp;

    *rho = 0.0;
    *drhodc = 0.0;
    // loop around 4 elements adjacent to global index node i,j
    for (k=0; k < 4; k++) {
        // global index of this element
        ii = i + oi[k];
        jj = j + oj[k];
        // field values for f, b, and u on this element
        ff[0] = user->f_rhs(ii*hx,jj*hy,user);
        ff[1] = user->f_rhs((ii-1)*hx,jj*hy,user);
        ff[2] = user->f_rhs((ii-1)*hx,(jj-1)*hy,user);
        ff[3] = user->f_rhs(ii*hx,(jj-1)*hy,user);
        uu[0] = NodeOnBdry(info,ii,  jj)   ?
                user->g_bdry(ii*hx,jj*hy,user)         : au[jj][ii];
        uu[1] = NodeOnBdry(info,ii-1,jj)   ?
                user->g_bdry((ii-1)*hx,jj*hy,user)     : au[jj][ii-1];
        uu[2] = NodeOnBdry(info,ii-1,jj-1) ?
                user->g_bdry((ii-1)*hx,(jj-1)*hy,user) : au[jj-1][ii-1];
        uu[3] = NodeOnBdry(info,ii,  jj-1) ?
                user->g_bdry(ii*hx,(jj-1)*hy,user)     : au[jj-1][ii];
        // loop over quadrature points in this element, summing to get rho
        for (r = 0; r < q.n; r++) {
            for (s = 0; s < q.n; s++) {
                // ll[k] is local (elementwise) index of the corner (= i,j)
                rhoIntegrandRef(ll[k],c,uu,ff,r,s,&prho,&pdrhodc,user);
                tmp = detj * q.w[r] * q.w[s];
                *rho += tmp * prho;
                *drhodc += tmp * pdrhodc;
            }
        }
    }
    // work per quadrature point:
    //   FLOPS = 6 + 53 = 59
    PetscCall(PetscLogFlops(59.0 * q.n * q.n));
    return 0;
}

// using Q1 finite elements, do nonlinear Gauss-Seidel (processor-block)
// sweeps on equation
//     F(u)[psi_ij] = b_ij   for all nodes i,j
// where psi_ij is the hat function and
//     F(u)[v] = int_Omega grad u . grad v - (lambda e^u + f) v
// and b is a nodal field provided by the call-back
// for each interior node i,j we define
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
// and do Newton iterations
//     c <-- c + rho(c) / rho'(c)
// note
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij
//                         - lambda e^(u + c psi_ij) psi_ij
// for boundary nodes we set
//     u_ij = g(x_i,y_j)
// and we do not iterate
PetscErrorCode NonlinearGSFEM(SNES snes, Vec u, Vec b, void *ctx) {
    BratuCtx*      user = (BratuCtx*)ctx;
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    PetscReal      atol, rtol, stol, hx, hy, **au, **ab,
                   c, rho, rho0, drhodc, s;
    DM             da;
    DMDALocalInfo  info;
    Vec            uloc;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits));
    PetscCall(SNESGetDM(snes,&da));

    // set up Q1 FEM tools for this grid
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);
    PetscCall(Q1SetupForGrid(user->quadpts,hx,hy));

    // for Dirichlet nodes assign boundary value once
    PetscCall(DMDAVecGetArray(da,u,&au));
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++)
            if (NodeOnBdry(&info,i,j))
                au[j][i] = user->g_bdry(i*hx,j*hy,user);
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
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (!NodeOnBdry(&info,i,j)) { // i,j is an owned interior node
                    // do pointwise Newton iterations
                    c = 0.0;
                    for (k = 0; k < maxits; k++) {
                        // evaluate rho(c) and rho'(c) for current c
                        PetscCall(rhoFcn(&info,i,j,c,au,&rho,&drhodc,user));
                        if (b)
                            rho -= ab[j][i];
                        if (k == 0)
                            rho0 = rho;
                        s = - rho / drhodc;  // Newton step
                        c += s;
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
        }
        PetscCall(DMDAVecRestoreArray(da,uloc,&au));
        PetscCall(DMLocalToGlobal(da,uloc,INSERT_VALUES,u));
    }

    PetscCall(DMRestoreLocalVector(da,&uloc));
    if (b)
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));

    // add flops for Newton iteration arithmetic; note rhoFcn() already counts flops
    PetscCall(PetscLogFlops(6 * totalits));
    (user->ngscount)++;
    return 0;
}
