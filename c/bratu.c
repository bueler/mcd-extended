static char help[] =
"Solve nonlinear Liouville-Bratu equation by Q1 finite elements\n"
"in 2D on a structured-grid.  Option prefix lb_.  Solves\n"
"  - nabla^2 u - lambda e^u = 0\n"
"on the unit square [0,1]x[0,1] subject to zero Dirichlet boundary conditions.\n"
"Critical value occurs about at lambda = 6.808.  Optional exact solution by\n"
"Liouville (1853) for case lambda=1.0.\n\n";

#include <petsc.h>
#include "q1fem.h"

typedef struct {
  PetscReal (*g_bdry)(PetscReal x, PetscReal y, void *ctx);  // Dirichlet b.c.
  PetscReal lambda;
  PetscBool exact;
  PetscInt  residualcount, ngscount, quadpts;
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
    bctx.quadpts = 2;
    PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","Liouville-Bratu equation solver options","");
    PetscCall(PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratu.c",bctx.lambda,&(bctx.lambda),NULL));
    PetscCall(PetscOptionsBool("-exact","use case of Liouville exact solution",
                            "bratu.c",bctx.exact,&(bctx.exact),NULL));
    // WARNING: coarse problems are badly solved with -lb_quadpts 1, so avoid in MG
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only)",
                            "bratu.c",bctx.quadpts,&(bctx.quadpts),NULL));
    PetscCall(PetscOptionsBool("-showcounts","print counts for calls to call-back functions",
                            "bratu.c",showcounts,&showcounts,NULL));
    PetscOptionsEnd();

    // check option validity
    if (bctx.exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_SELF,1,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        bctx.g_bdry = &g_liouville;
    }
    if ((bctx.quadpts < 1) || (bctx.quadpts > 3)) {
        SETERRQ(PETSC_COMM_SELF,2,"quadrature points n=1,2,3 only");
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
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",
                              info.mx,info.my));
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

// FLOPS: 4 + (48 + 8 + 9 + 31 + 6) = 106
PetscReal IntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                       const PetscReal uu[4],
                       PetscReal xi, PetscReal eta, BratuCtx *user) {
    const gradRef    du    = deval(uu,xi,eta),
                     dchiL = dchi(L,xi,eta);
    return GradInnerProd(hx,hy,du,dchiL)
           - user->lambda * PetscExpScalar(eval(uu,xi,eta)) * chi(L,xi,eta);
}

static PetscBool NodeOnBdry(DMDALocalInfo *info, PetscInt i, PetscInt j) {
    return (((i == 0) || (i == info->mx-1) || (j == 0) || (j == info->my-1)));
}

// compute F(u), the residual of the discretized PDE on the given grid:
//     F(u)[v] = int_Omega grad u . grad v - lambda e^u v
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
    PetscReal  uu[4];

    // clear residuals because we sum over elements; for Dirichlet nodes assign
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            FF[j][i] = (NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(i*hx,j*hy,user)
                                              : 0.0;

    // sum over elements; we own elements down or left of owned nodes,
    // but in parallel the integral needs to include elements up or right
    // of owned nodes (i.e. halo elements)
    for (j = info->ys; j <= info->ys + info->ym; j++) {
        if ((j == 0) || (j > info->my-1))
            continue;
        for (i = info->xs; i <= info->xs + info->xm; i++) {
            if ((i == 0) || (i > info->mx-1))
                continue;
            // values of iterate u at corners of element, using Dirichlet
            // value if known (for symmetry)
            uu[0] = NodeOnBdry(info,i,  j)   ? user->g_bdry(i*hx,j*hy,user)         : au[j][i];
            uu[1] = NodeOnBdry(info,i-1,j)   ? user->g_bdry((i-1)*hx,j*hy,user)     : au[j][i-1];
            uu[2] = NodeOnBdry(info,i-1,j-1) ? user->g_bdry((i-1)*hx,(j-1)*hy,user) : au[j-1][i-1];
            uu[3] = NodeOnBdry(info,i,  j-1) ? user->g_bdry(i*hx,(j-1)*hy,user)     : au[j-1][i];
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
                                         * IntegrandRef(hx,hy,l,uu,
                                                        q.xi[r],q.xi[s],user);
                        }
                    }
                }
            }
        }
    }
    // only count quadrature-point residual computations:
    //     4 + 106 = 110 flops per quadrature point
    //     q.n^2 quadrature points per element
    PetscCall(PetscLogFlops(110.0 * q.n * q.n * info->xm * info->ym));
    (user->residualcount)++;
    return 0;
}

// evaluate rho(c) at a point xi,eta in the reference element,
// for the hat function at corner L
// (i.e. chi_L = psi_ij from caller)
PetscReal rhoRef(PetscReal hx, PetscReal hy, PetscInt L,
                 PetscReal c, const PetscReal uu[4], const PetscReal bb[4],
                 PetscReal xi, PetscReal eta, BratuCtx *user) {
    const gradRef du    = deval(uu,xi,eta),
                  dchiL = dchi(L,xi,eta);
    return GradInnerProd(hx,hy,gradRefAXPY(c,dchiL,du),dchiL)
           - (user->lambda * PetscExpScalar(eval(uu,xi,eta) + c * chi(L,xi,eta))
              + eval(bb,xi,eta)) * chi(L,xi,eta);
}

// for owned, interior nodes i,j, evaluate the pointwise residual corresponding
// to the hat function psi_ij:
//     rho(c) = F(u + c psi_ij)[psi_ij] - ell_b[psi_ij]
//            = int_Omega (grad u + c grad psi_ij) . grad psi_ij
//                        - (lambda exp(u + c psi_ij) + b) psi_ij
// note this is value an integral over four elements, each with four quadrature
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
PetscErrorCode rhoFcn(DMDALocalInfo *info, PetscInt i, PetscInt j,
                      PetscReal c, PetscReal **au, PetscReal **ab,
                      PetscReal *rho, BratuCtx *user) {
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
    PetscReal uu[4], bb[4];

    *rho = 0.0;
    // loop around 4 elements adjacent to global index node i,j
    for (k=0; k < 4; k++) {
        // global index of adjacent element
        ii = i + oi[k];
        jj = j + oj[k];
        // field values for b and u on adjacent element
        bb[0] = ab[jj][ii];
        bb[1] = ab[jj][ii-1];
        bb[2] = ab[jj-1][ii-1];
        bb[3] = ab[jj-1][ii];
        uu[0] = NodeOnBdry(info,ii,  jj)   ?
                user->g_bdry(ii*hx,jj*hy,user)         : au[jj][ii];
        uu[1] = NodeOnBdry(info,ii-1,jj)   ?
                user->g_bdry((ii-1)*hx,jj*hy,user)     : au[jj][ii-1];
        uu[2] = NodeOnBdry(info,ii-1,jj-1) ?
                user->g_bdry((ii-1)*hx,(jj-1)*hy,user) : au[jj-1][ii-1];
        uu[3] = NodeOnBdry(info,ii,  jj-1) ?
                user->g_bdry(ii*hx,(jj-1)*hy,user)     : au[jj-1][ii];
        // loop over quadrature points in adjacent element, summing to get rho
        for (r = 0; r < q.n; r++) {
            for (s = 0; s < q.n; s++) {
                // ll[k] is local (elementwise) index of the corner (= i,j)
                *rho += detj * q.w[r] * q.w[s]
                        * rhoRef(hx,hy,ll[k],c,uu,bb,q.xi[r],q.xi[s],user);
            }
        }
    }
    return 0;
}

// FIXME also need
// PetscReal drhodcRef(...)

// FIXME
PetscErrorCode drhodcFcn(DMDALocalInfo *info, PetscInt i, PetscInt j,
                         PetscReal c, PetscReal **au, PetscReal **ab,
                         PetscReal *drhodc, BratuCtx *user) {
    *drhodc = 0.0;  // BOGUS
    return 0;
}

// do nonlinear Gauss-Seidel (processor-block) sweeps on equation
//     F(u)[v] = ell_b[v]    for all v
// where
//     F(u)[v] = int_Omega grad u . grad v - lambda e^u v
//     ell_b[v] = int_Omega b v
// and b is a field provided by the call-back
// for each interior node i,j we define
//     rho(c) = F(u + c psi_ij)[psi_ij] - ell_b[psi_ij]
// where psi_ij is the hat function, and do Newton iterations
//     c <-- c + rho(c) / rho'(c)
// note
//     rho'(c) = FIXME
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
                        rhoFcn(&info,i,j,c,au,ab,&rho,user);
                        SETERRQ(PETSC_COMM_SELF,1,"NOT YET IMPLEMENTED\n");
                        drhodcFcn(&info,i,j,c,au,ab,&drhodc,user); // NOT IMPLEMENTED
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
        PetscCall(DMDAVecRestoreArray(da,uloc,&au));
        PetscCall(DMLocalToGlobal(da,uloc,INSERT_VALUES,u));
        }
    }

    PetscCall(DMRestoreLocalVector(da,&uloc));
    PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));
    if (!b) {
        PetscCall(DMRestoreGlobalVector(da,&myb));
    }

    // FIXME PetscCall(PetscLogFlops(21.0 * totalits));
    (user->ngscount)++;
    return 0;
}
