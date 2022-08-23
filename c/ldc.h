// LDC = Level Defect Constraints
//
// Tools for managing defect constraints, specifically box constraints, at each
// level of a mesh hierarchy.  Contains a structured grid, and upper and lower
// obstacles, for each level.
//
// Canonical usage for single-level with nontrivial upper and lower constraints:
//     [create and set up DM da]
//     LDCCreate(k,da,&ldc);
//     DMCreateGlobalVector(ldc.dal,&(ldc.gamupp)));
//     [set values in Vec ldc.gamupp from a formula]
//     DMCreateGlobalVector(ldc.dal,&(ldc.gamlow)));
//     [set values in Vec ldc.gamlow from a formula]
//     [start a box-constrained solver, and get an iterate w]
//     LDCUpDefects(ldc,w);
//     [continue with the solver]
//     LDCDestroy(&ldc);
//
// FIXME for multilevel use, do monotone restriction in here

#ifndef LDC_H_
#define LDC_H_

// notes:
//   * DM dal must be allocated at creation
//   * obstacle values from extended real line [-\infty,+\infty]
//   * PETSC_INFINITY and PETSC_NINFINITY are valid entries
//   * special case: if an obstacle is identically PETSC_INFINITY or
//     PETSC_NINFINITY then it is unallocated and NULL
typedef struct {
  DM            dal;             // DMDA (structured grid) for this level
  Vec           gamupp, gamlow,  // original obstacle; only non-NULL on finest-level
                chiupp, chilow,  // down defect constraints: chi* = w - gamma*
                phiupp, philow;  // up defect constraints
// private
  PetscInt      level;           // =0 in single-level usage; otherwise 0 is coarsest
  DMDALocalInfo dalinfo;
} LDC;

PetscErrorCode LDCCreate(PetscInt level, DM da, LDC *ldc);

PetscErrorCode LDCDestroy(LDC *ldc);

PetscErrorCode LDCUpDefects(LDC ldc, Vec w);

#endif  // #ifndef LDC_H_
