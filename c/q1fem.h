// Tools for Q1 finite elements in two dimensions, including quadrature.
// Here xi,eta denote coordinates on [-1,1]^2 reference element.  Node
// numbering is
//   1 *---* 0
//     |   |
//   2 *---* 3
// on reference element.  (E.g. L=0 is (1,1), ..., L=3 is (1,-1).)
// Reference element hat functions are denoted chi_L(xi,eta),
// with gradient (in reference coordinates) returned by dchi().
//
// At bottom, one-dimensional Gauss-Legendre quadrature for interval [-1,1],
// of degree 1,2,3.
//
// For documentation see Chapter 9 and Interlude of Bueler, "PETSc for Partial
// Differential Equations", SIAM Press 2021, and c/ch9/phelm.c at
//   https://github.com/bueler/p4pdes

#ifndef Q1FEM_H_
#define Q1FEM_H_

// FLOPS: 6
PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta);

// evaluate v(xi,eta) on reference element using local node numbering
// FLOPS: 7 + 4 * chi = 31
PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta);

typedef struct {
    PetscReal  xi, eta;
} gradRef;

// FLOPS: 4
gradRef gradRefAXPY(PetscReal a, gradRef X, gradRef Y);

// FLOPS: 8
gradRef dchi(PetscInt L, PetscReal xi, PetscReal eta);

// evaluate partial derivs of v(xi,eta) on reference element
// FLOPS: 4 * (8 + 4) = 48
gradRef deval(const PetscReal v[4], PetscReal xi, PetscReal eta);

// FLOPS: 9
PetscReal GradInnerProd(PetscReal hx, PetscReal hy, gradRef du, gradRef dv);

// FLOPS: 13
PetscReal GradPow(PetscReal hx, PetscReal hy, gradRef du, PetscReal P, PetscReal eps);

#define MAXPTS 3

typedef struct {
    PetscInt   n;          // number of quadrature points for this rule
    PetscReal  xi[MAXPTS], // locations in [-1,1]
               w[MAXPTS];  // weights (sum to 2)
} Quad1D;

static const Quad1D gausslegendre[3]
    = {  {1,
          {0.0,                NAN,               NAN},
          {2.0,                NAN,               NAN}},
         {2,
          {-0.577350269189626, 0.577350269189626, NAN},
          {1.0,                1.0,               NAN}},
         {3,
          {-0.774596669241483, 0.0,               0.774596669241483},
          {0.555555555555556,  0.888888888888889, 0.555555555555556}} };

#endif
