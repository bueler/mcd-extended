#include <petsc.h>
#include "q1fem.h"

PetscReal xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
          etaL[4] = { 1.0,  1.0, -1.0, -1.0};

PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    return   v[0] * chi(0,xi,eta) + v[1] * chi(1,xi,eta)
           + v[2] * chi(2,xi,eta) + v[3] * chi(3,xi,eta);
}

gradRef gradRefAXPY(PetscReal a, gradRef X, gradRef Y) {
    const gradRef result = {a * X.xi  + Y.xi,
                            a * X.eta + Y.eta};
    return result;
}

gradRef dchi(PetscInt L, PetscReal xi, PetscReal eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

gradRef deval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    gradRef   sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi += v[L] * tmp.xi;  sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

PetscReal GradInnerProd(PetscReal hx, PetscReal hy, gradRef du, gradRef dv) {
    const PetscReal cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi * dv.xi + cy * du.eta * dv.eta;
}

PetscReal GradPow(PetscReal hx, PetscReal hy,
                  gradRef du, PetscReal P, PetscReal eps) {
    return PetscPowScalar(GradInnerProd(hx,hy,du,du) + eps*eps, P/2.0);
}
