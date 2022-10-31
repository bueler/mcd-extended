// Level Defect Constraints (LDC) are box constraints for a mesh hierarchy
// based on subtracting a fine-level admissible iterate.  The intended use
// is that there is one LDC object per level during each multilevel V-cycle
// of a multilevel constraint decomposition (MCD) method.  Each LDC object
// contains 4 LDCs, namely above and below lDCs for the up and
// down directions in a V-cycle.
//
// The theory of nonlinear MCD methods using defect constraints at each level
// is in paper/mcd2.pdf, namely
//   E. Bueler (2022), "Multilevel constraint decomposition methods for
//   nonlinear variational inequalities," in preparation.
//
// LDC is a struct which contains
//   * a structured grid (DM dal, a DMDA)
//   * above and below up LDCs (Vec chiupp, chilow)
//   * above and below down LDCs (Vec phiupp, philow)
// for each level.
//
// After creation, at the finest level one uses the original upper/lower
// obstacles (e.g. Vec gamupp, gamlow) to generate the finest-level up
// DCs.  Then monotone restriction (see src/q1transfers.h|c) is used to
// generate up LDCs at coarser levels.  Then subtraction is used to generate
// down LDCs.
//
// Support for extended real line [-infty,+infty] values is *incomplete*.
// Any LDC Vec is allowed to be NULL in which case it is interpreted as
// +infty or -infty accordingly, but if an LDC Vec is not null then it is
// assumed that *all* values of it are finite.
//
// The following gives canonical V-cycle usage with nontrivial upper and
// lower finest-level obstacles defined by Vecs vgamupp, vgamlow.  Note
// ldc[0] is coarse while ldc[N] is finest:
//
//     LDC       ldc[N+1];
//     DMDA      cdmda
//     PetscInt  k;
//     [  use DMDACreate2d() to create cdmda, and then fully configure it  ]
//     LDCCreateCoarsest(PETSC_TRUE,cdmda,&(ldc[0]));
//     for (k=0; k<N; k++)
//         LDCRefine(ldc[k],&(ldc[k+1]));
//     [  a box-constrained solver on finest level returns fine-level iterate w  ]
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
  void*         _coarser;         // cast to LDC*; this is NULL at coarsest level
                                  // (level=0) or points to next-coarser LDC
} LDC;

// create LDC for coarsest level from user's coarsest DMDA; that is,
// the user must set up ldc->dal before calling this
PetscErrorCode LDCCreateCoarsest(PetscBool verbose, LDC *ldc);

PetscErrorCode LDCDestroy(LDC *ldc);

// refine LDC *coarse to create LDC *fine; this creates a DMDA for *fine
PetscErrorCode LDCRefine(LDC *coarse, LDC *fine);

// create up DCs on finest level using original constraints gamupp, gamlow
// and current iterate w
PetscErrorCode LDCFinestUpDCsFromVecs(Vec w, Vec gamupp, Vec gamlow, LDC *ldc);

// same, but use formulas for gamupp, gamlow
PetscErrorCode LDCFinestUpDCsFromFormulas(Vec w,
                   PetscReal (*fgamupp)(PetscReal,PetscReal,void*),
                   PetscReal (*fgamlow)(PetscReal,PetscReal,void*),
                   LDC *ldc, void *ctx);

// after finest up DCs are created, generate up and down DCs on all levels
PetscErrorCode LDCGenerateDCsVCycle(LDC *finest);

// compute complementarity residual Fhat from ordinary residual F and
// defect z, for up DCs
PetscErrorCode LDCUpDCsCRFromResidual(LDC *ldc, Vec z, Vec F, Vec Fhat);

// compute complementarity residual Fhat from ordinary residual F and
// defect y, for down DCs
PetscErrorCode LDCDownDCsCRFromResidual(LDC *ldc, Vec y, Vec F, Vec Fhat);

// create a Vec from a formula
PetscErrorCode LDCVecFromFormula(LDC ldc, PetscReal (*ufcn)(PetscReal,PetscReal,void*),
                                 Vec u, void *ctx);

// for each pair of DCs, check lower <= 0 <= upper, and report min and max
PetscErrorCode LDCCheckDCRanges(LDC ldc);

// return flg=PETSC_TRUE if  ldc.philow <= y <= ldc.phiupp,  otherwise flg=PETSC_FALSE
PetscErrorCode LDCCheckAdmissibleDownDefect(LDC ldc, Vec y, PetscBool *flg);

// return flg=PETSC_TRUE if  ldc.chilow <= z <= ldc.chiupp,  otherwise flg=PETSC_FALSE
PetscErrorCode LDCCheckAdmissibleUpDefect(LDC ldc, Vec z, PetscBool *flg);

#endif  // #ifndef LDC_H_
