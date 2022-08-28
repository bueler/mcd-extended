#include <petsc.h>
#include "ldc.h"

PetscErrorCode LDCCreate(PetscInt level, DM da, LDC *ldc) {
    ldc->level = level;
    ldc->printinfo = PETSC_FALSE;
    if (da) {
        ldc->dal = da;
        PetscCall(DMDAGetLocalInfo(da,&(ldc->dalinfo)));
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: allocate DMDA before calling LDCCreate()");
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
        "LDC info: destroying LDC at level %d\n",ldc->level));
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
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: allocate coarse DMDA before calling LDCRefine()");
    }
    fine->level = coarse.level + 1;
    if (coarse.printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "LDC info: refining coarse LDC at level %d to generate fine LDC at level %d\n",
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

PetscErrorCode _PrintVecRange(Vec X, const char *name, const char *infcase) {
    PetscReal vmin, vmax;
    if (X) {
        PetscCall(VecMin(X,NULL,&vmin));
        PetscCall(VecMax(X,NULL,&vmax));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  %9.6f <= |%s| <= %9.6f\n",
                              vmin,name,vmax));
    } else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  [ %s=NULL is %s ]\n",
                              name,infcase));
    return 0;
}

PetscErrorCode LDCReportRanges(LDC ldc) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"defect constraint ranges at level %d:\n",
                          ldc.level));
    PetscCall(_PrintVecRange(ldc.gamupp,"gamupp","+infty"));
    PetscCall(_PrintVecRange(ldc.gamlow,"gamlow","-infty"));
    PetscCall(_PrintVecRange(ldc.chiupp,"chiupp","+infty"));
    PetscCall(_PrintVecRange(ldc.chilow,"chilow","-infty"));
    PetscCall(_PrintVecRange(ldc.phiupp,"phiupp","+infty"));
    PetscCall(_PrintVecRange(ldc.philow,"philow","-infty"));
    return 0;
}

PetscErrorCode LDCUpDefectsFromObstacles(Vec w, LDC *ldc) {
    if (ldc->chiupp) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: chiupp already created");
    }
    if (ldc->chilow) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: chilow already created");
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

PetscErrorCode LDCDownDefects(LDC *coarse, LDC *fine) {
    // generate phiupp
    if (!coarse) {
        if (fine->chiupp) {
            if (fine->printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: creating phiupp and setting phiupp=chiupp (coarsest case) at level %d\n",
                fine->level));
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
            PetscCall(VecCopy(fine->chiupp,fine->phiupp));
        } else {
            if (fine->printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: phiupp=NULL at level %d\n",fine->level));
        }
    } else if (coarse->chiupp && fine->chiupp) {
        if (fine->printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating phiupp and setting phiupp=chiupp-chiupp_coarse at level %d\n",
            fine->level));
        PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
        PetscCall(VecWAXPY(fine->phiupp,-1.0,coarse->chiupp,fine->chiupp));  // phiupp = chiupp - chiupp_coarse
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: unanticipated case");
    }
    // generate philow
    if (!coarse) {
        if (fine->chilow) {
            if (fine->printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: creating philow and setting philow=chilow (coarsest case) at level %d\n",
                fine->level));
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
            PetscCall(VecCopy(fine->chilow,fine->philow));
        } else {
            if (fine->printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: philow=NULL at level %d\n",fine->level));
        }
    } else if (coarse->chilow && fine->chilow) {
        if (fine->printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating philow and setting phiupp=chilow-chilow_coarse at level %d\n",
            fine->level));
        PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
        PetscCall(VecWAXPY(fine->philow,-1.0,coarse->chilow,fine->chilow));  // philow = chilow - chilow_coarse
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: unanticipated case");
    }
    return 0;
}
