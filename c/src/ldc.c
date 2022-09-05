#include <petsc.h>
#include "q1transfers.h"
#include "ldc.h"

PetscErrorCode LDCCreateCoarsest(PetscBool verbose, PetscInt mx, PetscInt my,
                                 PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax,
                                 LDC *ldc) {
    ldc->_level = 0;
    ldc->_printinfo = verbose;
    if (ldc->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  LDC info: creating LDC at level %d based on %d x %d grid DMDA\n",
        ldc->_level,mx,my));
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           mx,my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(ldc->dal)));
    // make defaults explicit:
    PetscCall(DMDASetInterpolationType(ldc->dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(ldc->dal,2,2,2));
    PetscCall(DMSetUp(ldc->dal));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(ldc->dal,xmin,xmax,ymin,ymax,0.0,0.0));
    ldc->_xmin = xmin;
    ldc->_xmax = xmax;
    ldc->_ymin = ymin;
    ldc->_ymax = ymax;
    ldc->chiupp = NULL;
    ldc->chilow = NULL;
    ldc->phiupp = NULL;
    ldc->philow = NULL;
    ldc->_coarser = NULL;
    return 0;
}

PetscErrorCode LDCDestroy(LDC *ldc) {
    if (ldc->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  LDC info: destroying LDC at level %d\n",ldc->_level));
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

PetscErrorCode LDCRefine(LDC *coarse, LDC *fine) {
    if (!(coarse->dal)) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: allocate coarse DMDA before calling LDCRefine()");
    }
    fine->_level = coarse->_level + 1;
    if (coarse->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  LDC info: refining level %d LDC to generate finer LDC at level %d\n",
        coarse->_level,fine->_level));
    fine->_printinfo = coarse->_printinfo;
    PetscCall(DMRefine(coarse->dal,PETSC_COMM_WORLD,&(fine->dal)));
    PetscCall(DMDASetInterpolationType(fine->dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(fine->dal,2,2,2));
    fine->_xmin = coarse->_xmin;
    fine->_xmax = coarse->_xmax;
    fine->_ymin = coarse->_ymin;
    fine->_ymax = coarse->_ymax;
    PetscCall(DMDASetUniformCoordinates(fine->dal,
                  fine->_xmin,fine->_xmax,fine->_ymin,fine->_ymax,0.0,0.0));
    fine->chiupp = NULL;
    fine->chilow = NULL;
    fine->phiupp = NULL;
    fine->philow = NULL;
    fine->_coarser = (void*)(coarse);
    return 0;
}

PetscErrorCode _VecPrintRange(Vec X, const char *name, const char *infcase) {
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

PetscErrorCode LDCReportDCRanges(LDC ldc) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"defect constraint ranges at level %d:\n",
                          ldc._level));
    PetscCall(_VecPrintRange(ldc.chiupp,"chiupp","+infty"));
    PetscCall(_VecPrintRange(ldc.chilow,"chilow","-infty"));
    PetscCall(_VecPrintRange(ldc.phiupp,"phiupp","+infty"));
    PetscCall(_VecPrintRange(ldc.philow,"philow","-infty"));
    return 0;
}

PetscErrorCode LDCFinestUpDCsFromVecs(Vec w, Vec vgamupp, Vec vgamlow, LDC *ldc) {
    if (ldc->chiupp) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: chiupp already created");
    }
    if (ldc->chilow) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: chilow already created");
    }
    if (vgamupp) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating chiupp and setting chiupp=gamupp-w at level %d\n",
            ldc->_level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chiupp)));
        PetscCall(VecWAXPY(ldc->chiupp,-1.0,w,vgamupp));  // chiupp = gamupp - w
    } else
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chiupp=NULL because gamupp=NULL is +infty at level %d\n",
            ldc->_level));
    if (vgamlow) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating chilow and setting chilow=gamlow-w at level %d\n",
            ldc->_level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chilow)));
        PetscCall(VecWAXPY(ldc->chilow,-1.0,w,vgamlow));  // chilow = gamlow - w
    } else
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chilow=NULL because gamlow=NULL is -infty at level %d\n",
            ldc->_level));
    return 0;
}

PetscErrorCode LDCVecFromFormula(LDC ldc,PetscReal (*ufcn)(PetscReal,PetscReal),
                                 Vec u) {
    PetscInt      i, j;
    PetscReal     hx, hy, x, y, **au;
    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(ldc.dal,&info));
    hx = (ldc._xmax - ldc._xmin) / (PetscReal)(info.mx - 1);
    hy = (ldc._ymax - ldc._ymin) / (PetscReal)(info.my - 1);
    PetscCall(DMDAVecGetArray(info.da, u, &au));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = ldc._ymin + j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = ldc._xmin + i * hx;
            au[j][i] = (*ufcn)(x,y);
        }
    }
    PetscCall(DMDAVecRestoreArray(info.da, u, &au));
    return 0;
}

PetscErrorCode LDCFinestUpDCsFromFormulas(Vec w,
                   PetscReal (*fgamupp)(PetscReal,PetscReal),
                   PetscReal (*fgamlow)(PetscReal,PetscReal),
                   LDC *ldc) {
    Vec vgamupp = NULL, vgamlow = NULL;
    if (fgamupp) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: using formula gamupp at level %d\n",
            ldc->_level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&vgamupp));
        PetscCall(LDCVecFromFormula(*ldc,fgamupp,vgamupp));
    } else
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: formula gamupp=NULL at level %d\n",
            ldc->_level));
    if (fgamlow) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: using formula gamlow at level %d\n",
            ldc->_level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&vgamlow));
        PetscCall(LDCVecFromFormula(*ldc,fgamlow,vgamlow));
    } else
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: formula gamlow=NULL at level %d\n",
            ldc->_level));
    PetscCall(LDCFinestUpDCsFromVecs(w,vgamupp,vgamlow,ldc));
    if (vgamupp)
        PetscCall(VecDestroy(&vgamupp));
    if (vgamlow)
        PetscCall(VecDestroy(&vgamlow));
    return 0;
}

PetscErrorCode _LDCUpDCsMonotoneRestrict(LDC fine, LDC *coarse) {
    if (!coarse) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: coarse not created");
    }
    if (coarse->chiupp) {
        SETERRQ(PETSC_COMM_SELF,2,"LDC ERROR: coarse chiupp already created");
    }
    if (coarse->chilow) {
        SETERRQ(PETSC_COMM_SELF,3,"LDC ERROR: coarse chilow already created");
    }
    if (fine.chiupp) {
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: setting chiupp at level %d using monotone restriction from level %d\n",
            coarse->_level,fine._level));
        PetscCall(DMCreateGlobalVector(coarse->dal,&(coarse->chiupp)));
        PetscCall(Q1MonotoneRestrict(Q1_MIN,fine.dal,coarse->dal,
                                     fine.chiupp,&(coarse->chiupp)));
    } else
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chiupp=NULL is +infty at level %d\n",
            coarse->_level));
    if (fine.chilow) {
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: setting chilow at level %d using monotone restriction from level %d\n",
            coarse->_level,fine._level));
        PetscCall(DMCreateGlobalVector(coarse->dal,&(coarse->chilow)));
        PetscCall(Q1MonotoneRestrict(Q1_MAX,fine.dal,coarse->dal,
                                     fine.chilow,&(coarse->chilow)));
    } else
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chilow=NULL is -infty at level %d\n",
            coarse->_level));
    return 0;
}

PetscErrorCode _LDCDownDCs(LDC *coarse, LDC *fine) {
    if (fine->phiupp) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: phiupp already created");
    }
    if (fine->philow) {
        SETERRQ(PETSC_COMM_SELF,2,"LDC ERROR: philow already created");
    }
    // generate phiupp
    if (!coarse) {
        if (fine->chiupp) {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: creating and setting phiupp=chiupp (coarsest case) at level %d\n",
                fine->_level));
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
            PetscCall(VecCopy(fine->chiupp,fine->phiupp));
        } else {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: phiupp=NULL is +infty at level %d\n",fine->_level));
        }
    } else if (coarse->chiupp && fine->chiupp) {
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating and setting phiupp=chiupp-chiupp_coarse at level %d\n",
            fine->_level));
        PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
        // phiupp = chiupp - P(chiupp_coarse)
        PetscCall(Q1Interpolate(coarse->dal,fine->dal,coarse->chiupp,&(fine->phiupp)));
        PetscCall(VecAYPX(fine->phiupp,-1.0,fine->chiupp));
    } else
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: phiupp=NULL is -infty at level %d\n",fine->_level));
    // generate philow
    if (!coarse) {
        if (fine->chilow) {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: creating and setting philow=chilow (coarsest case) at level %d\n",
                fine->_level));
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
            PetscCall(VecCopy(fine->chilow,fine->philow));
        } else {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: philow=NULL at level %d\n",fine->_level));
        }
    } else if (coarse->chilow && fine->chilow) {
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating and setting philow=chilow-chilow_coarse at level %d\n",
            fine->_level));
        PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
        // philow = chilow - P(chilow_coarse)
        PetscCall(Q1Interpolate(coarse->dal,fine->dal,coarse->chilow,&(fine->philow)));
        PetscCall(VecAYPX(fine->philow,-1.0,fine->chilow));
    } else
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: philow=NULL at level %d\n",fine->_level));
    return 0;
}

PetscErrorCode LDCGenerateDCsVCycle(LDC *finest) {
    LDC *ldc = finest,
        *coarse;
    while (ldc->_coarser) {
        coarse = (LDC*)(ldc->_coarser);
        PetscCall(_LDCUpDCsMonotoneRestrict(*ldc,coarse));
        PetscCall(_LDCDownDCs(coarse,ldc));
        ldc = coarse;
    }
    PetscCall(_LDCDownDCs(NULL,ldc));
    return 0;
}

// set flg=PETSC_TRUE if  u <= v  everywhere, otherwise flg=PETSC_FALSE
// extended reals rule:  if u=NULL (-infty) or v=NULL (+infty) then flg=PETSC_TRUE
PetscErrorCode _LDCVecLessThanOrEqual(LDC ldc, Vec u, Vec v, PetscBool *flg) {
    PetscInt      i, j;
    PetscReal     **au, **av;
    DMDALocalInfo info;
    if ((!u) || (!v)) {
        *flg = PETSC_TRUE;
        return 0;
    }
    PetscCall(DMDAGetLocalInfo(ldc.dal,&info));
    PetscCall(DMDAVecGetArray(info.da, u, &au));
    PetscCall(DMDAVecGetArray(info.da, v, &av));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            if (au[j][i] > av[j][i]) {
                PetscCall(DMDAVecRestoreArray(info.da, u, &au));
                PetscCall(DMDAVecRestoreArray(info.da, v, &av));
                *flg = PETSC_FALSE;
                return 0;
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(info.da, u, &au));
    PetscCall(DMDAVecRestoreArray(info.da, v, &av));
    *flg = PETSC_TRUE;
    return 0;
}

PetscErrorCode LDCCheckAdmissibleDownDefect(LDC ldc, Vec y, PetscBool *flg) {
    PetscCall(_LDCVecLessThanOrEqual(ldc,ldc.philow,y,flg));
    if (!(*flg))
        return 0;
    PetscCall(_LDCVecLessThanOrEqual(ldc,y,ldc.phiupp,flg));
    return 0;
}

PetscErrorCode LDCCheckAdmissibleUpDefect(LDC ldc, Vec z, PetscBool *flg) {
    PetscCall(_LDCVecLessThanOrEqual(ldc,ldc.chilow,z,flg));
    if (!(*flg))
        return 0;
    PetscCall(_LDCVecLessThanOrEqual(ldc,z,ldc.chiupp,flg));
    return 0;
}
