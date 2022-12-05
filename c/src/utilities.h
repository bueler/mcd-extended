// These utilities use Vec, 2D DMDA, and multilevel concepts.

#ifndef UTILITIES_H_
#define UTILITIES_H_

// print extended-reals min and max of a Vec; string infcase says what
// to print if X==NULL
PetscErrorCode VecPrintRange(Vec X, const char *name, const char *infcase, PetscBool newline);

// print the name, level j, and range of values, indenting jtop-j levels
PetscErrorCode IndentPrintRange(Vec v, const char* name, PetscInt jtop, PetscInt j);

// set flg=PETSC_TRUE if  u <= v  everywhere, otherwise flg=PETSC_FALSE
// extended reals rule:  if u=NULL (-infty) or v=NULL (+infty) then flg=PETSC_TRUE
PetscErrorCode VecLessThanOrEqual(DM da, Vec u, Vec v, PetscBool *flg);

// evaluate ufcn(x,y,ctx) on DMDA rectangle to fill u
PetscErrorCode VecFromFormula(DM da, PetscReal (*ufcn)(PetscReal,PetscReal,void*),
                              Vec u, void *ctx);

// compute complementarity residual Fhat from ordinary residual F, for bi-lateral constraints:
//     Fhat_ij = F_ij         if  Lower_ij < u_ij < Upper_ij   (inactive constraints)
//               min{F_ij,0}  if  u_ij <= Lower_ij
//               max{F_ij,0}  if  u_ij >= Upper_ij
// avoids tests if Upper=NULL (+infty) or Lower=NULL (-infty)
// reference: page 24 of https://pages.cs.wisc.edu/~ferris/cs635/complementarity.pdf
// FIXME consider an epsilon for active determination
PetscErrorCode CRFromResidual(DM da, Vec Upper, Vec Lower, Vec u, Vec F, Vec Fhat);

#endif  // #ifndef UTILITIES_H_
