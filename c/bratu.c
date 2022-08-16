static char help[] =
"Solve nonlinear Liouville-Bratu equation by Q1 finite elements\n"
"in 2D square (0,1)^2 using a structured-grid.  Option prefix lb_.  Solves\n"
"  - nabla^2 u - lambda e^u = f(x,y)\n"
"subject to Dirichlet boundary conditions u=g.  For f=0 and g=0 the critical\n"
"value occurs about at lambda = 6.808.  Optional exact solutions are by\n"
"Liouville (1853) (for lambda=1.0; -lb_exact) and by method of manufactured\n"
"solutions (arbitrary lambda; -lb_mms).\n\n";

#include <petsc.h>
#include "q1fem.h"

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
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal **,
                                        PetscReal**, BratuCtx*);
extern PetscErrorCode NonlinearGS(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes;
    Vec            u, uexact;
    BratuCtx       bctx;
    DMDALocalInfo  info;
    PetscBool      exact = PETSC_FALSE, mms = PETSC_FALSE, showcounts = PETSC_FALSE;
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
    PetscCall(PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratu.c",bctx.lambda,&(bctx.lambda),NULL));
    PetscCall(PetscOptionsBool("-exact","use Liouville exact solution",
                            "bratu.c",exact,&exact,NULL));
    PetscCall(PetscOptionsBool("-mms","use MMS exact solution",
                            "bratu.c",mms,&mms,NULL));
    // WARNING: coarse problems are badly solved with -lb_quadpts 1, so avoid in MG
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only)",
                            "bratu.c",bctx.quadpts,&(bctx.quadpts),NULL));
    PetscCall(PetscOptionsBool("-showcounts","print counts for calls to call-back functions",
                            "bratu.c",showcounts,&showcounts,NULL));
    PetscOptionsEnd();

    // options consistency checking
    if (exact && mms) {
        SETERRQ(PETSC_COMM_SELF,1,"invalid option combination -lb_exact -lb_mms\n");
    }
    if (exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_SELF,2,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        bctx.g_bdry = &g_liouville;  // and zero for f_rhs()
    }
    if (mms)
        bctx.f_rhs = &f_mms;  // and zero for g_bdry()
    if ((bctx.quadpts < 1) || (bctx.quadpts > 3)) {
        SETERRQ(PETSC_COMM_SELF,3,"quadrature points n=1,2,3 only");
    }

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,  // contrast with bratufd
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&bctx));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetApplicationContext(snes,&bctx));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&bctx));
    PetscCall(SNESSetNGS(snes,NonlinearGS,&bctx));
    PetscCall(SNESSetFromOptions(snes));

    // solve the problem
    PetscCall(DMGetGlobalVector(da,&u));
    PetscCall(VecSet(u,0.0));  // initialize to zero
    PetscCall(SNESSolve(snes,NULL,u));
    PetscCall(DMRestoreGlobalVector(da,&u));
    PetscCall(DMDestroy(&da));

    if (showcounts) {
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

// FLOPS: 5 + (48 + 8 + 31 + 9 + 31 + 6) = 138
PetscReal IntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                       const PetscReal uu[4], const PetscReal ff[4],
                       PetscReal xi, PetscReal eta, BratuCtx *user) {
    const gradRef    du    = deval(uu,xi,eta),
                     dchiL = dchi(L,xi,eta);
    const PetscReal  tmp = user->lambda * PetscExpScalar(eval(uu,xi,eta));
    return GradInnerProd(hx,hy,du,dchiL)
           - (tmp + eval(ff,xi,eta)) * chi(L,xi,eta);
}

PetscBool NodeOnBdry(DMDALocalInfo *info, PetscInt i, PetscInt j) {
    return (((i == 0) || (i == info->mx-1) || (j == 0) || (j == info->my-1)));
}

// compute F(u), the residual of the discretized PDE on the given grid:
//     F(u)[v] = int_Omega grad u . grad v - lambda e^u v - f v
// this method computes the vector
//     F_ij = F(u)[psi_ij]
// where i,j is a node and psi_ij is the hat function there
// note that at boundary nodes we have
//     F_ij = u_ij - g(x_i,y_j)
// where g(x,y) is the boundary value
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, BratuCtx *user) {
    const Quad1D    q = gausslegendre[user->quadpts-1];
    const PetscInt  li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
    const PetscReal hx = 1.0 / (PetscReal)(info->mx - 1),
                    hy = 1.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    PetscInt   i, j, l, PP, QQ, r, s;
    PetscReal  uu[4], ff[4];

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
                                          * IntegrandRef(hx,hy,l,uu,ff,
                                                         q.xi[r],q.xi[s],user);
                        }
                    }
                }
            }
        }
    }
    // FLOPS only counting quadrature-point residual computations:
    //     4 + 138 = 142 flops per quadrature point
    //     q.n^2 quadrature points per element
    PetscCall(PetscLogFlops(142.0 * q.n * q.n * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

// for owned, interior nodes i,j, we define the pointwise residual corresponding
// to the hat function psi_ij:
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
// FLOPS:  9 + (48 + 8 + 6 + 31 + 9 + 4 + 31 + 9) = 155
PetscErrorCode rhoIntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                 PetscReal c, const PetscReal uu[4], const PetscReal ff[4],
                 PetscReal xi, PetscReal eta,
                 PetscReal *rho, PetscReal *drhodc, BratuCtx *user) {
    const gradRef du    = deval(uu,xi,eta),
                  dchiL = dchi(L,xi,eta);
    const PetscReal chiL   = chi(L,xi,eta),
                    ushift = eval(uu,xi,eta) + c * chiL,
                    phiL   = user->lambda * PetscExpScalar(ushift);
    *rho = GradInnerProd(hx,hy,gradRefAXPY(c,dchiL,du),dchiL)
           - (phiL + eval(ff,xi,eta)) * chiL;
    *drhodc = GradInnerProd(hx,hy,dchiL,dchiL) - phiL * chiL;
    return 0;

}

// for owned, interior nodes i,j, evaluate rho(c) and
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
    const Quad1D    q = gausslegendre[user->quadpts-1];
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
                rhoIntegrandRef(hx,hy,ll[k],c,uu,ff,q.xi[r],q.xi[s],&prho,&pdrhodc,user);
                tmp = detj * q.w[r] * q.w[s];
                *rho += tmp * prho;
                *drhodc += tmp * pdrhodc;
            }
        }
    }
    PetscCall(PetscLogFlops(161.0 * q.n * q.n));
    return 0;
}

// do nonlinear Gauss-Seidel (processor-block) sweeps on equation
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
PetscErrorCode NonlinearGS(SNES snes, Vec u, Vec b, void *ctx) {
    BratuCtx*      user = (BratuCtx*)ctx;
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    PetscReal      atol, rtol, stol, hx, hy, **au, **ab,
                   c, rho, rho0, drhodc, s;
    DM             da;
    DMDALocalInfo  info;
    Vec            myb, uloc;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);

    // for Dirichlet nodes assign boundary value once
    PetscCall(DMDAVecGetArray(da,u,&au));
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++)
            if (NodeOnBdry(&info,i,j))
                au[j][i] = user->g_bdry(i*hx,j*hy,user);
    PetscCall(DMDAVecRestoreArray(da,u,&au));

    // if b is not defined, create it and set it to zero
    if (b) {
        myb = b;
    } else {
        PetscCall(DMGetGlobalVector(da,&myb));
        PetscCall(VecSet(myb,0.0));
    }
    PetscCall(DMDAVecGetArrayRead(da,myb,&ab));

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
    PetscCall(DMDAVecRestoreArrayRead(da,myb,&ab));
    if (!b) {
        PetscCall(DMRestoreGlobalVector(da,&myb));
    }

    // add flops for Newton iteration arithmetic; note rhoFcn() already counts flops
    PetscCall(PetscLogFlops(6 * totalits));
    (user->ngscount)++;
    return 0;
}
