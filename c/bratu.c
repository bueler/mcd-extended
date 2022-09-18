static char help[] =
"Solve nonlinear Liouville-Bratu, a semilinear elliptic equation\n"
"  - nabla^2 u - R(u) = f(x,y)\n"
"on the 2D square (0,1)^2, subject to Dirichlet boundary conditions u=g,\n"
"with reaction term R(u) = lambda e^u.  Grid is structured and equally-spaced.\n"
"Option prefix is lb_.  Choose either discretization method:\n"
"  * finite differences (-lb_fd)\n"
"  * Q1 finite elements (-lb_fem)\n"
"For f=0 and g=0 the critical value occurs about at lambda = 6.808.\n"
"Optional exact solutions are from Liouville (1853) (for lambda=1.0;\n"
"-lb_exact), or from method of manufactured solutions (arbitrary lambda;\n"
"-lb_mms).\n\n";

#include <petsc.h>
#include "src/q1fem.h"

typedef struct {
  // nonlinear reaction R(u)
  PetscReal       (*R_fcn)(PetscReal u, void *ctx);
  // nonlinear reaction derivative dRdu(u)
  PetscReal       (*dRdu_fcn)(PetscReal u, void *ctx);
  // right-hand side f(x,y)
  PetscReal       (*f_rhs)(PetscReal x, PetscReal y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscReal       (*g_bdry)(PetscReal x, PetscReal y, void *ctx);
  PetscReal       lambda,         // parameter in Bratu PDE
                  njacalpha;      // alpha in nonlinear Jacobi (ignored in NGS)
  PetscInt        residualcount,  // count of FormFunctionLocalX() calls
                  ngscount,       // count of N{GS|Jac}X() calls
                  quadpts;        // number of quadrature points in each dim in FEM
  PetscLogDouble  expcount;       // same as PetscLogFlops(); exact integers to 2^53=10^16
} BratuCtx;

static PetscReal R_bratu(PetscReal u, void *ctx) {
    BratuCtx *user = (BratuCtx*)ctx;
    user->expcount += 1.0;
    return user->lambda * PetscExpScalar(u);
}

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
           - user->R_fcn(u_mms(x,y,ctx),user);
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

extern PetscErrorCode NGSFD(SNES, Vec, Vec, void*);
extern PetscErrorCode NJacFD(SNES, Vec, Vec, void*);
extern PetscErrorCode NGSFEM(SNES, Vec, Vec, void*);
extern PetscErrorCode NJacFEM(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes;
    Vec            u, uexact;
    BratuCtx       bctx;
    DMDALocalInfo  info;
    PetscBool      exact = PETSC_FALSE, mms = PETSC_FALSE, counts = PETSC_FALSE,
                   fd = PETSC_FALSE, fem = PETSC_FALSE, njac = PETSC_FALSE;
    PetscLogDouble lflops, flops;
    PetscReal      errinf;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    bctx.R_fcn = &R_bratu;
    bctx.dRdu_fcn = &R_bratu;
    bctx.f_rhs = &fg_zero;
    bctx.g_bdry = &fg_zero;
    bctx.lambda = 1.0;
    bctx.njacalpha = 0.8;    // standard choice for Jacobi as smoother in 2D
    bctx.expcount = 0;
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
    PetscCall(PetscOptionsBool("-njac","use nonlinear Jacobi iteration as the NGS (smoother)",
                            "bratu.c",njac,&njac,NULL));
    PetscCall(PetscOptionsReal("-njac_alpha","weight used in nonlinear Jacobi iteration (smoother)",
                            "bratu.c",bctx.njacalpha,&(bctx.njacalpha),NULL));
    // WARNING: coarse problems are badly solved with -lb_quadpts 1, so avoid in MG
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only; for Q1 FEM case only)",
                            "bratu.c",bctx.quadpts,&(bctx.quadpts),NULL));
    PetscCall(PetscOptionsBool("-counts","print counts for calls to call-back functions",
                            "bratu.c",counts,&counts,NULL));
    PetscOptionsEnd();

    // options consistency checking
    if (fd && fem) {
        SETERRQ(PETSC_COMM_SELF,1,"invalid option combination -lb_fd -lb_fem; choose one!\n");
    }
    if (!fd && !fem) {
        SETERRQ(PETSC_COMM_SELF,2,"invalid option combination; choose at least one of -lb_fd or -lb_fem\n");
    }
    if (exact && mms) {
        SETERRQ(PETSC_COMM_SELF,3,"invalid option combination -lb_exact -lb_mms; choose one or neither\n");
    }
    if (exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_SELF,4,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        bctx.g_bdry = &g_liouville;  // and keep zero for f_rhs()
    }
    if (mms)
        bctx.f_rhs = &f_mms;  // and keep zero for g_bdry()
    if (bctx.quadpts < 1 || bctx.quadpts > 3) {
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
        PetscCall(SNESSetNGS(snes,njac ? NJacFD : NGSFD,&bctx));
    } else {
        PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
                   (DMDASNESFunction)FormFunctionLocalFEM,&bctx));
        PetscCall(SNESSetNGS(snes,njac ? NJacFEM : NGSFEM,&bctx));
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
                              "flops = %.3e,  exps = %.3e,  residual calls = %d,  NGS calls = %d\n",
                              flops,(PetscReal)(bctx.expcount),bctx.residualcount,bctx.ngscount));
    }

    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid",
                          info.mx,info.my));
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
                              ":   error |u-uexact|_inf = %.3e\n",errinf));
    } else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));

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
    return (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1);
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
                           - darea * (user->R_fcn(au[j][i],user)
                                      + user->f_rhs(i*hx,j*hy,user));
            }
        }
    }
    // estimated FLOPS
    PetscCall(PetscLogFlops(14.0 * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

// using Q1 finite elements, evaluate the integrand of the residual on
// the reference element; L for vertex, r,s for quadrature point
// FLOPS: 3 + (5) = 8
PetscReal IntegrandRef(PetscInt L, PetscInt r, PetscInt s,
                       const PetscReal Ru, const Q1GradRef du,
                       const PetscReal f, BratuCtx *user) {
    return Q1GradInnerProd(du,Q1dchi[L][r][s]) - (Ru + f) * Q1chi[L][r][s];
}

// using Q1 finite elements, compute F(u), the residual of the discretized
// PDE on the given grid:
//     F(u)[v] = int_Omega grad u . grad v - R(u) v - f v
// this method computes the Vec local pointer FF:
//     F_ij = F(u)[psi_ij]
// where i,j is a node and psi_ij is the corresponding hat function;
// at boundary nodes:
//     F_ij = u_ij - g(x_i,y_j)
PetscErrorCode FormFunctionLocalFEM(DMDALocalInfo *info, PetscReal **au,
                                    PetscReal **FF, BratuCtx *user) {
    const Q1Quad1D  q = Q1gausslegendre[user->quadpts-1];
    const PetscInt  li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
    const PetscReal hx = 1.0 / (PetscReal)(info->mx - 1),
                    hy = 1.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    PetscInt   i, j, l, PP, QQ, r, s;
    PetscReal  uu[4], ff[4], Rurs, frs, crs;
    Q1GradRef  durs;

    // set up Q1 FEM tools for this grid
    PetscCall(Q1Setup(user->quadpts,info->da,0.0,1.0,0.0,1.0));

    // clear residuals (because we sum over elements)
    // and assign F for Dirichlet nodes
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            FF[j][i] = (NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(i*hx,j*hy,user)
                                              : 0.0;

    // sum over elements to compute F for interior nodes
    // we own elements down or left of owned nodes, but in parallel the integral
    // needs to include elements up or right of owned nodes (halo elements)
    for (j = info->ys; j <= info->ys + info->ym; j++) {
        if (j == 0 || j > info->my-1)
            continue;
        for (i = info->xs; i <= info->xs + info->xm; i++) {
            if (i == 0 || i > info->mx-1)
                continue;
            // this element, down-and-left of node i,j, is adjacent to an owned
            // and interior node
            // values of rhs f at corners of element
            ff[0] = user->f_rhs(i*hx,j*hy,user);
            ff[1] = user->f_rhs((i-1)*hx,j*hy,user);
            ff[2] = user->f_rhs((i-1)*hx,(j-1)*hy,user);
            ff[3] = user->f_rhs(i*hx,(j-1)*hy,user);
            // values of iterate u at corners of this element, using Dirichlet
            // value for symmetry of Jacobian
            uu[0] = NodeOnBdry(info,i,  j)
                    ? user->g_bdry(i*hx,j*hy,user)         : au[j][i];
            uu[1] = NodeOnBdry(info,i-1,j)
                    ? user->g_bdry((i-1)*hx,j*hy,user)     : au[j][i-1];
            uu[2] = NodeOnBdry(info,i-1,j-1)
                    ? user->g_bdry((i-1)*hx,(j-1)*hy,user) : au[j-1][i-1];
            uu[3] = NodeOnBdry(info,i,  j-1)
                    ? user->g_bdry(i*hx,(j-1)*hy,user)     : au[j-1][i];
            // loop over quadrature points of this element
            for (r = 0; r < q.n; r++) {
                for (s = 0; s < q.n; s++) {
                    // compute quantities that depend on r,s but not l (= test function)
                    Rurs = user->R_fcn(Q1Eval(uu,r,s),user);
                    durs = Q1DEval(uu,r,s);
                    frs = Q1Eval(ff,r,s);
                    crs = detj * q.w[r] * q.w[s];
                    // loop over corners of element; l is local (elementwise)
                    // index of the corner, i.e. test function, and PP,QQ are
                    // global indices of same corner
                    for (l = 0; l < 4; l++) {
                        PP = i + li[l];
                        QQ = j + lj[l];
                        // only update residual if we own node and it is not boundary
                        if (PP >= info->xs && PP < info->xs + info->xm
                                && QQ >= info->ys && QQ < info->ys + info->ym
                                && !NodeOnBdry(info,PP,QQ))
                            FF[QQ][PP] += crs * IntegrandRef(l,r,s,Rurs,durs,frs,user);
                    }
                }
            }
        }
    }
    // estimated FLOPS: only counting quadrature-point float computations:
    //     q.n^2 quadrature points per element
    //     2 + 7 + 16 + 7 + 2 + 4 * (2 + 8) = 74 flops per quadrature point
    PetscCall(PetscLogFlops(74.0 * q.n * q.n * (info->xm + 1.0) * (info->ym + 1.0)));
    (user->residualcount)++;
    return 0;
}

// next method is private and called by NGSFD() and NJacFD()
// using finite differences, do processor-block sweeps on equation F(u)=b:
//     njac = PETSC_TRUE:   nonlinear Jacobi
//     njac = PETSC_FALSE:  nonlinear Gauss-Seidel
PetscErrorCode _SmootherFD(PetscBool njac, SNES snes, Vec u, Vec b, void *ctx) {
    PetscInt       i, j, k, m, sweeps, maxits;
    PetscLogDouble totalits = 0.0;
    PetscReal      atol, rtol, stol, hx, hy, darea, hxhy, hyhx,
                   **au, **aunew, **ab, bij, uu, Ru, dRdu, phi0, phi, dphidu, s;
    DM             da;
    DMDALocalInfo  info;
    Vec            uloc, unew;
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
    for (m=0; m<sweeps; m++) {
        PetscCall(DMGlobalToLocal(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(da,uloc,&au));
        if (njac) { // nonlinear Jacobi
            // uloc and au are old values with stencil width;
            // Vec unew is created; pointer aunew is different from au
            PetscCall(DMGetGlobalVector(da,&unew));
            PetscCall(DMDAVecGetArray(da,unew,&aunew));
        } else { // nonlinear Gauss-Seidel
            // uloc and au are current and updating values with stencil width;
            // Vec unew is not created; pointer aunew is same as au
            aunew = au;
        }
        for (j = info.ys; j < info.ys + info.ym; j++) {
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (NodeOnBdry(&info,i,j)) {
                    aunew[j][i] = user->g_bdry(i*hx,j*hy,user);
                } else {
                    if (b)
                        bij = ab[j][i];
                    else
                        bij = 0.0;
                    // do pointwise Newton iterations:
                    //   u_k+1 = u_k - phi(u_k) / phi'(u_k)
                    uu = au[j][i];
                    for (k = 0; k < maxits; k++) {
                        Ru = user->R_fcn(uu,user);
                        dRdu = Ru;  // NOTE: this is special optimization for Bratu;
                                    //       generally apply user->dRdu_fcn()
                        phi =   hyhx * (2.0 * uu - au[j][i-1] - au[j][i+1])
                              + hxhy * (2.0 * uu - au[j-1][i] - au[j+1][i])
                              - darea * (Ru + user->f_rhs(i*hx,j*hy,user))
                              - bij;
                        if (k == 0)
                             phi0 = phi;
                        dphidu = 2.0 * (hyhx + hxhy) - darea * dRdu;
                        s = - phi / dphidu;     // Newton step
                        uu += s;
                        totalits += 1.0;
                        if (                        atol > PetscAbsReal(phi)
                            || rtol * PetscAbsReal(phi0) > PetscAbsReal(phi)
                            ||   stol * PetscAbsReal(uu) > PetscAbsReal(s)  ) {
                            break;
                        }
                    }
                    if (njac)
                        aunew[j][i] = (1.0 - user->njacalpha) * au[j][i] + user->njacalpha * uu;
                    else
                        aunew[j][i] = uu; // NGS does not use njacalpha
                }
            }
        }
        PetscCall(DMDAVecRestoreArray(da,uloc,&au));
        if (njac) {
            PetscCall(DMDAVecRestoreArray(da,unew,&aunew));
            PetscCall(VecCopy(unew,u));
            PetscCall(DMRestoreGlobalVector(da,&unew));
        } else {
            PetscCall(DMLocalToGlobal(da,uloc,INSERT_VALUES,u));
        }
    }
    PetscCall(DMRestoreLocalVector(da,&uloc));
    if (b) {
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));
    }
    PetscCall(PetscLogFlops(22.0 * totalits));
    (user->ngscount)++;
    return 0;
}

PetscErrorCode NGSFD(SNES snes, Vec u, Vec b, void *ctx) {
    PetscCall(_SmootherFD(PETSC_FALSE, snes, u, b, ctx));
    return 0;
}

PetscErrorCode NJacFD(SNES snes, Vec u, Vec b, void *ctx) {
    PetscCall(_SmootherFD(PETSC_TRUE, snes, u, b, ctx));
    return 0;
}

// using Q1 finite elements, for owned, interior nodes i,j, we define a
// pointwise residual corresponding to hat function psi_ij perturbations:
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
//            = int_Omega (grad u + c grad psi_ij) . grad psi_ij
//                        - (R(u + c psi_ij) + f) psi_ij
//              - b_ij
// and
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij - dRdu(u + c psi_ij) psi_ij^2
// note b_ij is outside the integral
// the integral is over four elements, each with four quadrature points "+"
// (in the default -lb_quadpts 2 case), which we traverse in the order 0,1,2,3:
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
// each of the four elements has nodes with local (reference element) indices:
//              1 *---* 0
//                |   |
//              2 *---* 3

// evaluate integrand of rho(c), drhodc(c) at a point xi,eta in the
// reference element, for the hat function at corner L (i.e. chi_L = psi_ij
// from caller), returning rho and drhodc
// FLOPS:  9 + (16 + 7 + 5 + 4 + 7 + 5) = 53
PetscErrorCode rhoIntegrandRef(PetscInt L, PetscInt r, PetscInt s,
                 PetscReal c, const PetscReal uu[4], const PetscReal ff[4],
                 PetscReal *rho, PetscReal *drhodc, BratuCtx *user) {
    const Q1GradRef du     = Q1DEval(uu,r,s),
                    dchiL  = Q1dchi[L][r][s];
    const PetscReal chiL   = Q1chi[L][r][s],
                    ushift = Q1Eval(uu,r,s) + c * chiL,
                    RuL    = user->R_fcn(ushift,user);
    *rho = Q1GradInnerProd(Q1GradAXPY(c,dchiL,du),dchiL)
           - (RuL + Q1Eval(ff,r,s)) * chiL;
    // NOTE: next line uses special optimization for Bratu
    //       generally apply user->dRdu_fcn()
    *drhodc = Q1GradInnerProd(dchiL,dchiL) - RuL * chiL * chiL;
    return 0;
}

// using Q1 finite elements, for owned, interior nodes i,j, evaluate
// rho(c) and rho'(c); we loop over 4 node-adjacent elements, but then only
// evaluate once at each quadrature points inside the element
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
    PetscInt        k, ii, jj, r, s;
    PetscReal       uu[4], ff[4], prho, pdrhodc, cc;

    *rho = 0.0;
    *drhodc = 0.0;
    // loop over 4 elements adjacent to global index node i,j
    for (k=0; k < 4; k++) {
        // ii,jj = global index of upper-right corner of this element
        ii = i + oi[k];
        jj = j + oj[k];
        // field values for f and u on this element
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
                rhoIntegrandRef(ll[k],r,s,c,uu,ff,&prho,&pdrhodc,user);
                cc = detj * q.w[r] * q.w[s];
                *rho += cc * prho;
                *drhodc += cc * pdrhodc;
            }
        }
    }
    // estimated FLOPS per quadrature point in the 4 elements: 6 + 53 = 59
    PetscCall(PetscLogFlops(59.0 * 4.0 * q.n * q.n));
    return 0;
}

// the next two functions use Q1 finite elements to do parallel
// (processor-block) nonlinear Gauss-Seidel sweeps, either as a solver
// (e.g. -snes_type nrichardson) or as a smoother (e.g. -snes_type fas)
// we do sweeps by updating u pointwise on equation
//     F(u)[psi_ij] = b_ij   for all nodes i,j
// where psi_ij is the hat function and
//     F(u)[v] = int_Omega grad u . grad v - (R(u) + f) v
// and b is a nodal field provided by the call-back
// for each interior node i,j we define the pointwise residual with
// respect to a pointwise perturbation:
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
// and do Newton iterations
//     c_k+1 = c_k + rho(c_k) / rho'(c_k)
// and then
//     u_ij <-- u_ij + c
// for boundary nodes we instead set
//     u_ij = g(x_i,y_j)

// nonlinear Gauss-Seidel: do maxits pointwise Newton iterations,
// and check tolerances
PetscErrorCode NGSFEM(SNES snes, Vec u, Vec b, void *ctx) {
    BratuCtx*      user = (BratuCtx*)ctx;
    PetscInt       i, j, k, m, sweeps, maxits;
    PetscLogDouble totalits = 0.0;
    PetscReal      atol, rtol, stol, hx, hy, **au, **ab,
                   c, rho, rho0, drhodc, s;
    DM             da;
    DMDALocalInfo  info;
    Vec            uloc;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(Q1Setup(user->quadpts,da,0.0,1.0,0.0,1.0));

    // for Dirichlet nodes assign boundary value once
    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);
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
    for (m=0; m<sweeps; m++) {
        // update ghosts
        PetscCall(DMGlobalToLocal(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(da,uloc,&au));
        for (j = info.ys; j < info.ys + info.ym; j++) {
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (!NodeOnBdry(&info,i,j)) { // i,j is an owned interior node
                    // do pointwise Newton iterations:
                    //   c_k+1 = c_k - rho(c_k) / rho'(c_k)
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
                        totalits += 1.0;
                        if (                        atol > PetscAbsReal(rho)
                            || rtol * PetscAbsReal(rho0) > PetscAbsReal(rho)
                            ||    stol * PetscAbsReal(c) > PetscAbsReal(s)  ) {
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
    PetscCall(PetscLogFlops(6.0 * totalits));
    (user->ngscount)++;
    return 0;
}

// evaluate integrand of rho(0), drhodc(0) at a quadrature point r,s in the
// reference element, for the hat function at corner L (i.e. chi_L = psi_ij
// from caller), returning rho and drhodc
// FLOPS:  5 + 3 + 5 + 3 = 16
PetscErrorCode rhoIntegrandRef_NJac(PetscInt L, PetscInt r, PetscInt s,
                   const PetscReal Ru, const Q1GradRef du, const PetscReal f,
                   PetscReal *rho, PetscReal *drhodc, BratuCtx *user) {
    const Q1GradRef dchiL  = Q1dchi[L][r][s];
    const PetscReal chiL   = Q1chi[L][r][s];
    *rho = Q1GradInnerProd(du,dchiL) - (Ru + f) * chiL;
    // NOTE: next line uses special optimization for Bratu
    //       generally apply user->dRdu_fcn()
    *drhodc = Q1GradInnerProd(dchiL,dchiL) - Ru * chiL * chiL;
    return 0;
}
// nonlinear weighted Jacobi: do a single pointwise Newton iteration
// (require maxits=1, and ignores tolerances)
// we compute rho and drhodc across all nodes using element-wise assembly
// (just like FormFunctionLocalFEM()) before doing the nodal-wise Newton step
PetscErrorCode NJacFEM(SNES snes, Vec u, Vec b, void *ctx) {
    BratuCtx*      user = (BratuCtx*)ctx;
    const PetscInt li[4] = {0,-1,-1,0}, lj[4] = {0,0,-1,-1};
    PetscInt       i, j, l, m, r, s, PP, QQ, sweeps, maxits;
    PetscReal      hx, hy, detj, Rurs, frs, crs, prho, pdrhodc,
                   uu[4], ff[4], **au, **arho, **adrhodc;
    const PetscReal **ab, **auloc;
    Q1GradRef      durs;
    DM             da;
    DMDALocalInfo  info;
    Vec            uloc, rho, drhodc;
    Q1Quad1D       q;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,NULL,NULL,NULL,&maxits));
    if (maxits != 1)
        SETERRQ(PETSC_COMM_SELF,1,"FEM nonlinear Jacobi only allows maxits==1\n");
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(Q1Setup(user->quadpts,da,0.0,1.0,0.0,1.0));
    q = Q1gausslegendre[user->quadpts-1];
    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);
    detj = 0.25 * hx * hy;

    // for Dirichlet nodes assign boundary value once
    PetscCall(DMDAVecGetArray(da,u,&au));
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++)
            if (NodeOnBdry(&info,i,j))
                au[j][i] = user->g_bdry(i*hx,j*hy,user);
    PetscCall(DMDAVecRestoreArray(da,u,&au));

    // need local vector for stencil width in parallel
    PetscCall(DMGetLocalVector(da,&uloc));
    // need global vectors for for element-wise assembly of pointwise
    // residuals, and derivatives thereof, which are integrals
    PetscCall(DMGetGlobalVector(da,&rho));
    PetscCall(DMGetGlobalVector(da,&drhodc));
    if (b)
        PetscCall(DMDAVecGetArrayRead(da,b,&ab));

    // sweeps over interior nodes
    for (m=0; m<sweeps; m++) {
        // update u values, with ghosts, for next sweep
        PetscCall(DMGlobalToLocal(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArrayRead(da,uloc,&auloc));

        // compute Vec rho with pointwise residual, and Vec drhodc with
        // derivative-of-pointwise-residual, from element-wise assembly
        PetscCall(VecSet(rho,0.0));
        PetscCall(VecSet(drhodc,0.0));
        PetscCall(DMDAVecGetArray(da,rho,&arho));
        PetscCall(DMDAVecGetArray(da,drhodc,&adrhodc));
        // sum over *elements* to compute arrays with rho(0), drhodc(0) values,
        // for interior nodes; we own elements down or left of owned nodes;
        // in parallel the integral needs to include elements up or right of
        // owned nodes (halo elements)
        for (j = info.ys; j <= info.ys + info.ym; j++) {
            if (j == 0 || j > info.my-1)
                continue;
            for (i = info.xs; i <= info.xs + info.xm; i++) {
                if (i == 0 || i > info.mx-1)
                    continue;
                // this element, down-and-left of node i,j, is adjacent to an
                // owned and interior node
                // get values of rhs f at corners of element
                ff[0] = user->f_rhs(i*hx,j*hy,user);
                ff[1] = user->f_rhs((i-1)*hx,j*hy,user);
                ff[2] = user->f_rhs((i-1)*hx,(j-1)*hy,user);
                ff[3] = user->f_rhs(i*hx,(j-1)*hy,user);
                // values of iterate u at corners of this element, using Dirichlet
                // value for symmetry of Jacobian
                uu[0] = NodeOnBdry(&info,i,  j)
                        ? user->g_bdry(i*hx,j*hy,user)         : auloc[j][i];
                uu[1] = NodeOnBdry(&info,i-1,j)
                        ? user->g_bdry((i-1)*hx,j*hy,user)     : auloc[j][i-1];
                uu[2] = NodeOnBdry(&info,i-1,j-1)
                        ? user->g_bdry((i-1)*hx,(j-1)*hy,user) : auloc[j-1][i-1];
                uu[3] = NodeOnBdry(&info,i,  j-1)
                        ? user->g_bdry(i*hx,(j-1)*hy,user)     : auloc[j-1][i];
                // loop over quadrature points of this element
                for (r = 0; r < q.n; r++) {
                    for (s = 0; s < q.n; s++) {
                        Rurs = user->R_fcn(Q1Eval(uu,r,s),user);
                        durs = Q1DEval(uu,r,s);
                        frs = Q1Eval(ff,r,s);
                        crs = detj * q.w[r] * q.w[s];
                        // loop over corners of element; l is local (elementwise)
                        // index of the corner, i.e. test function, and PP,QQ are
                        // global indices of same corner
                        for (l = 0; l < 4; l++) {
                            PP = i + li[l];
                            QQ = j + lj[l];
                            // only compute rho if we own node and it is not boundary
                            if (PP >= info.xs && PP < info.xs + info.xm
                                    && QQ >= info.ys && QQ < info.ys + info.ym
                                    && !NodeOnBdry(&info,PP,QQ)) {
                                rhoIntegrandRef_NJac(l,r,s,Rurs,durs,frs,&prho,&pdrhodc,user);
                                arho[QQ][PP] += crs * prho;
                                adrhodc[QQ][PP] += crs * pdrhodc;
                            }
                        }
                    }
                }
            } // for i
        } // for j
        PetscCall(DMDAVecRestoreArrayRead(da,uloc,&auloc));
        // estimated FLOPS: only counting quadrature-point float computations:
        //     q.n^2 quadrature points per element
        //     7 + 16 + 7 + 2 + 4 * (16 + 4) = 112.0 flops per quadrature point
        PetscCall(PetscLogFlops(112.0 * q.n * q.n * (info.xm + 1) * (info.ym + 1)));

        // loop over nodes to take the single Newton step
        PetscCall(DMDAVecGetArray(da,u,&au));  // will update u
        for (j = info.ys; j < info.ys + info.ym; j++) {
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (!NodeOnBdry(&info,i,j)) { // i,j is an owned interior node
                    // do single (unconditional) pointwise Newton iteration
                    // from c_0 = 0:
                    //   c_1 = c_0 - rho(c_0) / rho'(c_0) = - rho(0) / rho'(0)
                    if (b)
                        arho[j][i] -= ab[j][i];
                    au[j][i] -= arho[j][i] / adrhodc[j][i];  // Newton step
                }
            }
        }
        PetscCall(DMDAVecRestoreArray(da,u,&au));
        PetscCall(DMDAVecRestoreArray(da,rho,&arho));
        PetscCall(DMDAVecRestoreArray(da,drhodc,&adrhodc));
        PetscCall(PetscLogFlops(2.0 * info.xm * info.ym));
    } // for m (sweeps)

    if (b) {
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));
    }
    PetscCall(DMRestoreGlobalVector(da,&rho));
    PetscCall(DMRestoreGlobalVector(da,&drhodc));
    PetscCall(DMRestoreLocalVector(da,&uloc));

    (user->ngscount)++;
    return 0;
}
