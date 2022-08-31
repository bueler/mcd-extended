#include <petsc.h>
#include "q1fem.h"

static const PetscReal xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
                       etaL[4] = { 1.0,  1.0, -1.0, -1.0};

PetscReal Q1FEM_IP_CX = NAN,
          Q1FEM_IP_CY = NAN;

PetscReal chiFormula(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

gradRef dchiFormula(PetscInt L, PetscReal xi, PetscReal eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

PetscErrorCode q1setup(PetscInt quadpts, PetscReal hx, PetscReal hy) {
    const Quad1D  q = gausslegendre[quadpts-1];
    PetscInt l, r, s;
    Q1FEM_IP_CX = 4.0 / (hx * hx);
    Q1FEM_IP_CY = 4.0 / (hy * hy);
    for (l = 0; l < 4; l++)
        for (r = 0; r < q.n; r++)
            for (s = 0; s < q.n; s++) {
                chi[l][r][s] = chiFormula(l,q.xi[r],q.xi[s]);
                dchi[l][r][s] = dchiFormula(l,q.xi[r],q.xi[s]);
            }
    return 0;
}


PetscReal eval(const PetscReal v[4], PetscInt r, PetscInt s) {
    return   v[0] * chi[0][r][s] + v[1] * chi[1][r][s]
           + v[2] * chi[2][r][s] + v[3] * chi[3][r][s];
}

gradRef deval(const PetscReal v[4], PetscInt r, PetscInt s) {
    gradRef   sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = dchi[L][r][s];
        sum.xi += v[L] * tmp.xi;
        sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

gradRef gradRefAXPY(PetscReal a, gradRef X, gradRef Y) {
    const gradRef result = {a * X.xi  + Y.xi,
                            a * X.eta + Y.eta};
    return result;
}

PetscReal GradInnerProd(gradRef du, gradRef dv) {
    return Q1FEM_IP_CX * du.xi * dv.xi + Q1FEM_IP_CX * du.eta * dv.eta;
}

PetscReal GradPow(gradRef du, PetscReal P, PetscReal eps) {
    return PetscPowScalar(GradInnerProd(du,du) + eps*eps, P/2.0);
}
