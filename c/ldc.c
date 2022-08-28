#include <petsc.h>
#include "ldc.h"

PetscErrorCode LDCCreate(PetscInt level, DM da, LDC *ldc) {
    ldc->level = level;
    ldc->printinfo = PETSC_FALSE;
    if (da) {
        ldc->dal = da;
        PetscCall(DMDAGetLocalInfo(da,&(ldc->dalinfo)));
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"LDC error: allocate DMDA before calling LDCCreate()");
    }
    ldc->gamupp = NULL;
    ldc->gamlow = NULL;
    ldc->chiupp = NULL;
    ldc->chilow = NULL;
    ldc->phiupp = NULL;
    ldc->philow = NULL;
    return 0;
}

PetscErrorCode LDCDestroy(LDC *ldc) {
    if (ldc->printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "LDC info: destroying LDC with level %d\n",ldc->level));
    if (ldc->gamupp)
        PetscCall(VecDestroy(&(ldc->gamupp)));
    if (ldc->gamlow)
        PetscCall(VecDestroy(&(ldc->gamlow)));
    if (ldc->chiupp)
        PetscCall(VecDestroy(&(ldc->chiupp)));
    if (ldc->chilow)
        PetscCall(VecDestroy(&(ldc->chilow)));
    if (ldc->phiupp)
        PetscCall(VecDestroy(&(ldc->phiupp)));
    if (ldc->philow)
        PetscCall(VecDestroy(&(ldc->philow)));
    if (ldc->dal)
        PetscCall(DMDestroy(&(ldc->dal)));
    return 0;
}

PetscErrorCode LDCRefine(LDC coarse, LDC *fine) {
    if (!(coarse.dal)) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC error: allocate coarse DMDA before calling LDCRefine()");
    }
    fine->level = coarse.level + 1;
    if (coarse.printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "LDC info: refining coarse LDC with level %d to generate fine LDC with level %d\n",
        coarse.level,fine->level));
    fine->printinfo = PETSC_FALSE;
    PetscCall(DMRefine(coarse.dal,PETSC_COMM_WORLD,&(fine->dal)));
    PetscCall(DMDAGetLocalInfo(fine->dal,&(fine->dalinfo)));
    fine->gamupp = NULL;
    fine->gamlow = NULL;
    fine->chiupp = NULL;
    fine->chilow = NULL;
    fine->phiupp = NULL;
    fine->philow = NULL;
    return 0;
}

PetscErrorCode LDCTogglePrintInfo(LDC *ldc) {
    ldc->printinfo = !(ldc->printinfo);
    return 0;
}

PetscErrorCode LDCUpDefects(Vec w, LDC *ldc) {
    if (ldc->chiupp) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC error: chiupp already created");
    }
    if (ldc->chilow) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC error: chilow already created");
    }
    if (ldc->gamupp) {
        if (ldc->printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating chiupp and setting chiupp=gamupp-w at level %d\n",ldc->level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chiupp)));
        PetscCall(VecWAXPY(ldc->chiupp,-1.0,w,ldc->gamupp));  // chiupp = gamupp - w
    } else
        if (ldc->printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: chiupp=NULL because gamupp=+infty\n"));
    if (ldc->gamlow) {
        if (ldc->printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating chilow and setting chilow=gamlow-w at level %d\n",ldc->level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chilow)));
        PetscCall(VecWAXPY(ldc->chilow,-1.0,w,ldc->gamlow));  // chilow = gamlow - w
    } else
        if (ldc->printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: chilow=NULL because gamlow=-infty\n"));
    return 0;
}
