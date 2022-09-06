// These utilities use Vec and 2D DMDA concepts but not defect constraint
// (see ldc.h|c) or Q1 (see q1fem.h|c and q1transfers.h|c) concepts.

#ifndef UTILITIES_H_
#define UTILITIES_H_

// print extended-reals min and max of a Vec; string infcase says what
// to print if X==NULL
PetscErrorCode VecPrintRange(Vec X, const char *name, const char *infcase);

// set flg=PETSC_TRUE if  u <= v  everywhere, otherwise flg=PETSC_FALSE
// extended reals rule:  if u=NULL (-infty) or v=NULL (+infty) then flg=PETSC_TRUE
PetscErrorCode VecLessThanOrEqual(DM da, Vec u, Vec v, PetscBool *flg);

// compute complementarity residual Fhat from ordinary residual F, for bi-lateral constraints:
//     Fhat_ij = F_ij         if  Lower_ij < u_ij < Upper_ij   (inactive constraints)
//               min{F_ij,0}  if  u_ij <= Lower_ij
//               max{F_ij,0}  if  u_ij >= Upper_ij
// reference: page 24 of https://pages.cs.wisc.edu/~ferris/cs635/complementarity.pdf
// FIXME consider an epsilon for active determination
PetscErrorCode CRFromResidual(DM da, Vec Upper, Vec Lower, Vec u, Vec F, Vec Fhat);

#endif  // #ifndef UTILITIES_H_
