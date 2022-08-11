// Tools for Q1 finite elements in two dimensions.  Here xi,eta
// denote coordinates on [-1,1]^2 reference element.  Node numbering is
//   1 *---* 0
//     |   |
//   2 *---* 3
// on reference element.  (E.g. L=0 is (1,1), ..., L=3 is (1,-1).)
// Reference element hat functions are denoted chi_L(xi,eta),
// with gradient (in reference coordinates) returned by dchi().
// For documentation see Chapter 9 of Bueler, "PETSc for Partial
// Differential Equations", SIAM Press 2021, and c/ch9/phelm.c at
//   https://github.com/bueler/p4pdes

#ifndef Q1FEM_H_
#define Q1FEM_H_

static PetscReal xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
                 etaL[4] = { 1.0,  1.0, -1.0, -1.0};

static PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

// evaluate v(xi,eta) on reference element using local node numbering
static PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    return   v[0] * chi(0,xi,eta) + v[1] * chi(1,xi,eta)
           + v[2] * chi(2,xi,eta) + v[3] * chi(3,xi,eta);
}

typedef struct {
    PetscReal  xi, eta;
} gradRef;

static gradRef dchi(PetscInt L, PetscReal xi, PetscReal eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

// evaluate partial derivs of v(xi,eta) on reference element
static gradRef deval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    gradRef   sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi += v[L] * tmp.xi;  sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

static PetscReal GradInnerProd(PetscReal hx, PetscReal hy,
                               gradRef du, gradRef dv) {
    const PetscReal cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi * dv.xi + cy * du.eta * dv.eta;
}

static PetscReal GradPow(PetscReal hx, PetscReal hy,
                         gradRef du, PetscReal P, PetscReal eps) {
    return PetscPowScalar(GradInnerProd(hx,hy,du,du) + eps*eps, P/2.0);
}

#endif
