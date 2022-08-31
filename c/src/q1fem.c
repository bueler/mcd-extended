#include <petsc.h>
#include "q1fem.h"

static const PetscReal _Q1xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
                       _Q1etaL[4] = { 1.0,  1.0, -1.0, -1.0};

PetscReal Q1_IP_CX = NAN,
          Q1_IP_CY = NAN;

PetscReal _Q1chiFormula(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + _Q1xiL[L] * xi) * (1.0 + _Q1etaL[L] * eta);
}

Q1GradRef _Q1dchiFormula(PetscInt L, PetscReal xi, PetscReal eta) {
    const Q1GradRef result = {0.25 * _Q1xiL[L]  * (1.0 + _Q1etaL[L] * eta),
                            0.25 * _Q1etaL[L] * (1.0 + _Q1xiL[L]  * xi)};
    return result;
}

PetscErrorCode Q1Setup(PetscInt quadpts, PetscReal hx, PetscReal hy) {
    const Q1Quad1D q = Q1gausslegendre[quadpts-1];
    PetscInt l, r, s;
    Q1_IP_CX = 4.0 / (hx * hx);
    Q1_IP_CY = 4.0 / (hy * hy);
    for (l = 0; l < 4; l++)
        for (r = 0; r < q.n; r++)
            for (s = 0; s < q.n; s++) {
                Q1chi[l][r][s] = _Q1chiFormula(l,q.xi[r],q.xi[s]);
                Q1dchi[l][r][s] = _Q1dchiFormula(l,q.xi[r],q.xi[s]);
            }
    return 0;
}

PetscReal Q1Eval(const PetscReal v[4], PetscInt r, PetscInt s) {
    return   v[0] * Q1chi[0][r][s] + v[1] * Q1chi[1][r][s]
           + v[2] * Q1chi[2][r][s] + v[3] * Q1chi[3][r][s];
}

Q1GradRef Q1DEval(const PetscReal v[4], PetscInt r, PetscInt s) {
    Q1GradRef sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = Q1dchi[L][r][s];
        sum.xi += v[L] * tmp.xi;
        sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

Q1GradRef Q1GradAXPY(PetscReal a, Q1GradRef X, Q1GradRef Y) {
    const Q1GradRef result = {a * X.xi  + Y.xi,
                              a * X.eta + Y.eta};
    return result;
}

PetscReal Q1GradInnerProd(Q1GradRef du, Q1GradRef dv) {
    return Q1_IP_CX * du.xi * dv.xi + Q1_IP_CX * du.eta * dv.eta;
}

PetscReal Q1GradPow(Q1GradRef du, PetscReal p, PetscReal eps) {
    return PetscPowScalar(Q1GradInnerProd(du,du) + eps*eps, p/2.0);
}
