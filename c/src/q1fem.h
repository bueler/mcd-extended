// Tools for Q1 finite elements in two dimensions, including quadrature.
//
// Starts with one-dimensional Gauss-Legendre quadrature for interval [-1,1],
// for 1,2,3 quadrature points.  Namely, type Q1Quad1D and array Q1gausslegendre[].
//
// Node numbering on reference element [-1,1]^2 is
//   1 *---* 0
//     |   |
//   2 *---* 3
// on reference element.  (E.g. L=0 is (1,1), ..., L=3 is (1,-1).)
// Reference element hat functions are
//    chi_L(xi,eta) = Q1chi[L][r][s]
// where xi,eta denote coordinates on reference element and quadature points
// are xi=xi[r],eta=xi[s].  Similarly, the gradient of chi_L(xi,et), in
// reference coordinates, is in the array Q1dchi[L][r][s].
//
// For documentation see Chapter 9 and Interlude of Bueler, "PETSc for Partial
// Differential Equations", SIAM Press 2021, and c/ch9/phelm.c at
//   https://github.com/bueler/p4pdes
// but not optimizations have been applied here.

#ifndef Q1FEM_H_
#define Q1FEM_H_

typedef struct {
    PetscInt   n;     // number of quadrature points for this rule (=1,2,3)
    PetscReal  xi[3], // locations in [-1,1]
               w[3];  // weights (sum to 2)
} Q1Quad1D;

static const Q1Quad1D Q1gausslegendre[3]
    = {  {1,
          {0.0,                NAN,               NAN},
          {2.0,                NAN,               NAN}},
         {2,
          {-0.577350269189626, 0.577350269189626, NAN},
          {1.0,                1.0,               NAN}},
         {3,
          {-0.774596669241483, 0.0,               0.774596669241483},
          {0.555555555555556,  0.888888888888889, 0.555555555555556}} };

typedef struct {
    PetscReal  xi, eta;
} Q1GradRef;

// following are global, NOT static
PetscReal Q1chi[4][3][3];   // Q1chi[L][r][s]
Q1GradRef Q1dchi[4][3][3];  // Q1dchi[L][r][s]

PetscErrorCode Q1Setup(PetscInt quadpts);

PetscErrorCode Q1SetupForGrid(PetscReal hx, PetscReal hy);

// evaluate v(xi,eta) at xi=xi[r],eta=xi[s] on reference element using
// local node numbering
// FLOPS: 7
PetscReal Q1Eval(const PetscReal v[4], PetscInt r, PetscInt s);

// evaluate partial derivs of v(xi,eta) on reference element
// FLOPS: 16
Q1GradRef Q1DEval(const PetscReal v[4], PetscInt r, PetscInt s);

// FLOPS: 4
Q1GradRef Q1GradAXPY(PetscReal a, Q1GradRef X, Q1GradRef Y);

// FLOPS: 5
PetscReal Q1GradInnerProd(Q1GradRef du, Q1GradRef dv);

// FLOPS: 9
PetscReal Q1GradPow(Q1GradRef du, PetscReal p, PetscReal eps);

#endif
