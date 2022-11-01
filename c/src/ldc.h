// Level Defect Constraints (LDC) are box constraints for a mesh hierarchy
// based on "defect constraints", i.e. subtracting a fine-level admissible
// iterate from some original box constraints.  The intended use
// is that there is one LDC object per level during each multilevel V-cycle
// of a multilevel constraint decomposition (MCD) method.  Each LDC object
// contains 4 LDCs, namely above and below LDCs for the up and
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
// The following gives canonical V-cycle usage.  Note ldc[0] is the
// coarsest level while ldc[N] is the finest level.  We do maxiters V-cycles.
// We start with
//   * upper and lower finest-level obstacles (Vecs) gamupp, gamlow
//   * an admissible finest-level iterate w:  gamlow <= w <= gamupp
//
//     LDC       ldc[N+1];
//     [  use DMDACreate2d() to create ldc[0].dal, and fully configure it  ]
//     LDCCreateCoarsest(PETSC_TRUE,&(ldc[0]));
//     for (j=0; j<N; j++) {
//         LDCRefine(ldc[j],&(ldc[j+1]));
//     }
//     [  create admissible finest-level initial iterate w  ]
//     for (iter=0; iter<maxiters; iter++) {
//         LDCSetFinestUpDCs(w,gamupp,gamlow,&(ldc[N]));
//         for (j=N; j>=1; j--) {
//             LDCSetLevel(&(ldc[j]))
//             [  do level j solver actions including down-smoother  ]
//         }
//         [  do level 0 (coarsest level) solver  ]
//         for (j=1; j<=N; j++) {
//             [  do level j solver actions including up-smoother  ]
//         }
//         [  compute updated finest-level iterate w  ]
//     }
//     for (j=0; j<=N; j++) {
//         LDCDestroy(&(ldc[j]));
//     }

#ifndef LDC_H_
#define LDC_H_

// notes:
//   * DM dal is for a 2D rectangle with uniform coordinates, box stencil,
//     and no (extra) boundary allocations
//   * the user sets up the DM dal for the coarsest level
//   * defect constraint values from extended real line [-\infty,+\infty]
//   * FIXME want to allow PETSC_INFINITY and PETSC_NINFINITY as valid entries
//   * special case: a defect constraint which is unallocated and NULL
//                   is interpreted as identically PETSC_INFINITY or
//                   PETSC_NINFINITY, respectively
typedef struct {
  DM            dal;             // DMDA (structured grid) for this level
  Vec           chiupp, chilow,  // up defect constraints: chiX = w - gammaX
                phiupp, philow;  // down defect constraints
// private
  PetscInt      _level;           // =0 in single-level usage; otherwise 0 is coarsest
  PetscBool     _printinfo;
  void*         _coarser;         // cast to LDC*; this is NULL at coarsest level
                                  // (level=0), or it points to next-coarser LDC
} LDC;

// create LDC for coarsest level from user's coarsest DMDA; that is,
// the user must set up ldc->dal before calling this
PetscErrorCode LDCCreateCoarsest(PetscBool verbose, LDC *ldc);

PetscErrorCode LDCDestroy(LDC *ldc);

// refine LDC *coarse to create LDC *fine; this creates a DMDA for *fine
PetscErrorCode LDCRefine(LDC *coarse, LDC *fine);

// create up DCs chiupp,chilow on finest level using original constraints
// gamupp, gamlow and the current finest-level iterate w
PetscErrorCode LDCSetFinestUpDCs(Vec w, Vec gamupp, Vec gamlow, LDC *ldc);

// assuming LDC fine has chiupp, chilow set,
//     * set up chiupp, chilow on fine->_coarser using monotone restriction
//     * set up phiupp, philow on fine using subtraction
PetscErrorCode LDCSetLevel(LDC *fine);

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
