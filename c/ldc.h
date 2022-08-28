// LDC = Level Defect Constraints
//
// Tools for managing defect constraints, specifically box constraints, at each
// level of a mesh hierarchy.  Contains a structured grid, and upper and lower
// obstacles, for each level.
//
// Canonical two-level usage with nontrivial upper and lower obstacles;
// note ldc[0] is coarse, ldc[1] is fine:
//     LDC ldc[2];
//     [  create and set up coarse DMDA  ]
//     LDCCreate(0,coarseda,&(ldc[0]));
//     LDCRefine(ldc[0],&(ldc[1]));
//     DMCreateGlobalVector(ldc[1].dal,&(ldc[1].gamupp)));
//     [  set Vec ldc[1].gamupp  ]
//     DMCreateGlobalVector(ldc[1].dal,&(ldc[1].gamlow)));
//     [  set Vec ldc[1].gamlow  ]
//     [  start a box-constrained solver, and get a fine-level iterate w  ]
//     LDCUpDefectsFromObstacles(w,&(ldc[1]));
//     LDCUpDefectsMonotoneRestrict(ldc[1],&(ldc[0]));
//     LDCDownDefects(&(ldc[0]),&(ldc[1]));
//     LDCDownDefects(NULL,&(ldc[0]));
//     [  continue with the solver  ]
//     LDCDestroy(&(ldc[1]));
//     LDCDestroy(&(ldc[0]));
//
// Canonical usage for single-level with nontrivial upper and lower obstacles:
//     LDC ldc;
//     [  create and set up DMDA  ]
//     LDCCreate(0,da,&ldc);
//     DMCreateGlobalVector(ldc.dal,&(ldc.gamupp)));
//     [  set Vec ldc.gamupp  ]
//     DMCreateGlobalVector(ldc.dal,&(ldc.gamlow)));
//     [  set Vec ldc.gamlow  ]
//     [  start a box-constrained solver, and get an iterate w  ]
//     LDCUpDefectsFromObstacles(w,&ldc);
//     LDCDownDefects(NULL,&ldc);
//     [  continue with the solver  ]
//     LDCDestroy(&ldc);

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
  PetscBool     printinfo;
} LDC;

PetscErrorCode LDCCreate(PetscInt level, DM da, LDC *ldc);

PetscErrorCode LDCDestroy(LDC *ldc);

PetscErrorCode LDCTogglePrintInfo(LDC *ldc);

PetscErrorCode LDCReportRanges(LDC ldc);

PetscErrorCode LDCRefine(LDC coarse, LDC *fine);

PetscErrorCode LDCUpDefectsFromObstacles(Vec w, LDC *ldc);

//FIXME implement PetscErrorCode LDCUpDefectsMonotoneRestrict(LDC fine, LDC *coarse);

PetscErrorCode LDCDownDefects(LDC *coarse, LDC *fine); // modifies fine but not coarse

#endif  // #ifndef LDC_H_
