// Level Defect Constraints (LDC) are box constraints for a mesh hierarchy
// based on subtracting a fine-level admissible iterate.  The intended use
// is that there is one LDC object per level during each multilevel V-cycle
// of a multilevel constraint decomposition (MCD) method.  Each LDC contains
// 4 DCs (defect constraints), namely upper and lower DCs for the up and
// down directions in a V-cycle.
//
// The theory of nonlinear MCD methods using defect constraints at each level
// is in paper/mcd2.pdf, namely
//   E. Bueler (2022), "Multilevel constraint decomposition methods for
//   nonlinear variational inequalities," in preparation.
//
// LDC is a struct which contains
//   * a structured grid (DM dal, a DMDA)
//   * upper and lower up DCs (Vec chiupp, chilow)
//   * upper and lower down DCs (Vec phiupp, philow)
// for each level.
//
// After creation, at the finest level one uses the original upper/lower
// obstacles (e.g. Vec gamupp, gamlow) to generate the finest-level up
// DCs.  Then monotone restriction (see src/q1transfers.h|c) is used to
// generate up DCs at coarser levels.  Then subtraction is used to generate
// down DCs.
//
// The following gives canonical V-cycle usage with nontrivial upper and
// lower finest-level obstacles defined by Vecs vgamupp, vgamlow.  Note
// ldc[0] is coarse while ldc[N] is finest:
//
//     LDC       ldc[N+1];
//     PetscInt  k;
//     LDCCreateCoarsest(PETSC_TRUE,0,mx,my,xmin,xmax,ymin,ymax,&(ldc[0]));
//     for (k=0; k<N; k++)
//         LDCRefine(ldc[k],&(ldc[k+1]));
//     [  start a box-constrained solver on the finest level, and get a
//        fine-level iterate w  ]
//     LDCFinestUpDCsFromVecs(w,vgamupp,vgamlow,&(ldc[N]));
//     LDCGenerateDCsVCycle(&(ldc[N]));
//     [  continue with the solver  ]
//     for (k=0; k<N+1; k++)
//         LDCDestroy(&(ldc[k]));
//
// Alternatively one can use formulas to define the finest-level up defect
// constraints:
//     LDCFinestUpDCsFromFormulas(w,fgamupp,fgamlow,&(ldc[N]));


#ifndef LDC_H_
#define LDC_H_

// notes:
//   * DM dal is for a 2D rectangle with uniform coordinates, box stencil,
//     and no (extra) boundary allocations
//   * defect constraint values from extended real line [-\infty,+\infty]
//   * PETSC_INFINITY and PETSC_NINFINITY are valid entries
//   * special case: if defect constraint is identically PETSC_INFINITY or
//     PETSC_NINFINITY then it is unallocated and NULL
typedef struct {
  DM            dal;             // DMDA (structured grid) for this level
  Vec           chiupp, chilow,  // up defect constraints: chiX = w - gammaX
                phiupp, philow;  // down defect constraints
// private
  PetscInt      _level;           // =0 in single-level usage; otherwise 0 is coarsest
  PetscBool     _printinfo;
  PetscReal     _xmin, _xmax, _ymin, _ymax;
  void*         _coarser;         // cast to LDC*; this is NULL at coarsest level
                                  // (level=0) or points to next-coarser LDC
} LDC;

PetscErrorCode LDCCreateCoarsest(PetscBool verbose, PetscInt mx, PetscInt my,
                                 PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax,
                                 LDC *ldc);

PetscErrorCode LDCDestroy(LDC *ldc);

PetscErrorCode LDCRefine(LDC *coarse, LDC *fine);

PetscErrorCode LDCFinestUpDCsFromVecs(Vec w, Vec gamupp, Vec gamlow, LDC *ldc);

PetscErrorCode LDCFinestUpDCsFromFormulas(Vec w,
                   PetscReal (*fgamupp)(PetscReal,PetscReal),
                   PetscReal (*fgamlow)(PetscReal,PetscReal),
                   LDC *ldc);

PetscErrorCode LDCGenerateDCsVCycle(LDC *finest);

// utilities:

// create a Vec from a formula
PetscErrorCode LDCVecFromFormula(LDC ldc,PetscReal (*ufcn)(PetscReal,PetscReal),
                                 Vec u);

// for each of 4 DCs, report min and max
PetscErrorCode LDCReportDCRanges(LDC ldc);

// return flg=PETSC_TRUE if  ldc.philow <= y <= ldc.phiupp,  otherwise flg=PETSC_FALSE
PetscErrorCode LDCCheckAdmissibleDownDefect(LDC ldc, Vec y, PetscBool *flg);

// return flg=PETSC_TRUE if  ldc.chilow <= z <= ldc.chiupp,  otherwise flg=PETSC_FALSE
PetscErrorCode LDCCheckAdmissibleUpDefect(LDC ldc, Vec z, PetscBool *flg);

#endif  // #ifndef LDC_H_
