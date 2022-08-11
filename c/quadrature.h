// One-dimensional Gauss-Legendre quadrature for interval [-1,1],
// of degree 1,2,3.

#ifndef QUADRATURE_H_
#define QUADRATURE_H_

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
