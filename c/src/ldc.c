#include <petsc.h>
#include "utilities.h"
#include "q1transfers.h"
#include "ldc.h"

PetscErrorCode LDCCreateCoarsest(PetscBool verbose, LDC *ldc) {
    DMDALocalInfo  info;
    PetscReal      xymin[2], xymax[2];
    ldc->_level = 0;
    ldc->_printinfo = verbose;
    if (ldc->_printinfo) {
        PetscCall(DMDAGetLocalInfo(ldc->dal,&info));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating LDC at level %d based on provided %d x %d coarse DMDA\n",
        ldc->_level,info.mx,info.my));
        PetscCall(DMGetBoundingBox(ldc->dal,xymin,xymax));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "            with bounding box [%10.6f,%10.6f,%10.6f,%10.6f]\n",
            xymin[0],xymax[0],xymin[1],xymax[1]));
    }
    PetscCall(DMDASetInterpolationType(ldc->dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(ldc->dal,2,2,2));
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
    if (ldc->dal) {
        PetscCall(DMDestroy(&(ldc->dal)));
        ldc->dal = NULL;
    }
    return 0;
}

PetscErrorCode LDCRefine(LDC *coarse, LDC *fine) {
    DMDALocalInfo  info;
    PetscReal      xymin[2], xymax[2];
    if (!(coarse->dal)) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: allocate coarse DMDA before calling LDCRefine()");
    }
    if (coarse->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  LDC info: refining level %d LDC",coarse->_level));
    fine->_printinfo = coarse->_printinfo;
    PetscCall(DMRefine(coarse->dal,PETSC_COMM_WORLD,&(fine->dal)));
    fine->_level = coarse->_level + 1;
    PetscCall(DMDAGetLocalInfo(fine->dal,&info));
    if (coarse->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        ", yielding LDC with %d x %d DMDA at level %d\n",
        info.mx,info.my,fine->_level));
    PetscCall(DMDASetInterpolationType(fine->dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(fine->dal,2,2,2));
    PetscCall(DMGetBoundingBox(coarse->dal,xymin,xymax));
    PetscCall(DMDASetUniformCoordinates(fine->dal,
              xymin[0],xymax[0],xymin[1],xymax[1],0.0,0.0));
    fine->chiupp = NULL;
    fine->chilow = NULL;
    fine->phiupp = NULL;
    fine->philow = NULL;
    fine->_coarser = (void*)(coarse);
    return 0;
}

PetscErrorCode LDCCheckDCRanges(LDC ldc) {
    PetscReal clmax = PETSC_NINFINITY, cumin = PETSC_INFINITY,
              plmax = PETSC_NINFINITY, pumin = PETSC_INFINITY;
    PetscInt  retval = 0;
    if (ldc.chilow)
        PetscCall(VecMin(ldc.chilow,NULL,&clmax));
    if (ldc.chiupp)
        PetscCall(VecMin(ldc.chiupp,NULL,&cumin));
    if (ldc.philow)
        PetscCall(VecMin(ldc.philow,NULL,&plmax));
    if (ldc.phiupp)
        PetscCall(VecMin(ldc.phiupp,NULL,&pumin));
    if ((clmax <= 0.0) && (0.0 <= cumin) && (plmax <= 0.0) && (0.0 <= pumin))
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"zero bracket checks PASS (level %d):\n",
                              ldc._level));
    else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"zero bracket checks FAIL (level %d):\n",
                              ldc._level));
        retval = 1;
    }
    PetscCall(VecPrintRange(ldc.chilow,"chilow","-infty",PETSC_FALSE));
    PetscCall(VecPrintRange(ldc.chiupp,"chiupp","+infty",PETSC_TRUE));
    PetscCall(VecPrintRange(ldc.philow,"philow","-infty",PETSC_FALSE));
    PetscCall(VecPrintRange(ldc.phiupp,"phiupp","+infty",PETSC_TRUE));
    return retval;
}

PetscErrorCode LDCSetFinestUpDCs(Vec w, Vec vgamupp, Vec vgamlow, LDC *ldc) {
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
            "  LDC info: chiupp=NULL is +infty because gamupp=NULL at level %d\n",
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
            "  LDC info: chilow=NULL is -infty because gamlow=NULL at level %d\n",
            ldc->_level));
    return 0;
}

PetscErrorCode LDCVecFromFormula(LDC ldc, PetscReal (*ufcn)(PetscReal,PetscReal,void*),
                                 Vec u, void *ctx) {
    PetscInt      i, j;
    PetscReal     hx, hy, x, y, **au, xymin[2], xymax[2];
    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(ldc.dal,&info));
    PetscCall(DMGetBoundingBox(ldc.dal,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (PetscReal)(info.mx - 1);
    hy = (xymax[1] - xymin[1]) / (PetscReal)(info.my - 1);
    PetscCall(DMDAVecGetArray(info.da, u, &au));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = xymin[1] + j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = xymin[0] + i * hx;
            au[j][i] = (*ufcn)(x,y,ctx);
        }
    }
    PetscCall(DMDAVecRestoreArray(info.da, u, &au));
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
                "  LDC info: phiupp=NULL is +infty (coarsest case) at level %d\n",fine->_level));
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
            "  LDC info: phiupp=NULL is +infty at level %d\n",fine->_level));
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
                "  LDC info: philow=NULL is -infty (coarsest case) at level %d\n",fine->_level));
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
            "  LDC info: philow=NULL is -infty at level %d\n",fine->_level));
    return 0;
}

PetscErrorCode LDCGenerateDCsVCycle(LDC *finest) {
    LDC *ldc = finest,
        *coarse;
    if (finest->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  LDC info: generating V-cycle from finest level %d\n",finest->_level));
    while (ldc->_coarser) {
        coarse = (LDC*)(ldc->_coarser);
        PetscCall(_LDCUpDCsMonotoneRestrict(*ldc,coarse));
        PetscCall(_LDCDownDCs(coarse,ldc));
        ldc = coarse;
    }
    PetscCall(_LDCDownDCs(NULL,ldc));
    if (finest->_printinfo) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "  LDC info: calling LDCCheckDCRanges() on all levels\n"));
        ldc = finest;
        while (PETSC_TRUE) {
            PetscCall(LDCCheckDCRanges(*ldc));
            if (!ldc->_coarser)
                break;
            ldc = (LDC*)(ldc->_coarser);
        }
    }
    return 0;
}

PetscErrorCode LDCUpDCsCRFromResidual(LDC *ldc, Vec z, Vec F, Vec Fhat) {
    PetscCall(CRFromResidual(ldc->dal,ldc->chiupp,ldc->chilow,z,F,Fhat));
    return 0;
}

PetscErrorCode LDCDownDCsCRFromResidual(LDC *ldc, Vec y, Vec F, Vec Fhat) {
    PetscCall(CRFromResidual(ldc->dal,ldc->phiupp,ldc->philow,y,F,Fhat));
    return 0;
}

PetscErrorCode LDCCheckAdmissibleDownDefect(LDC ldc, Vec y, PetscBool *flg) {
    PetscCall(VecLessThanOrEqual(ldc.dal,ldc.philow,y,flg));
    if (!(*flg))
        return 0;
    PetscCall(VecLessThanOrEqual(ldc.dal,y,ldc.phiupp,flg));
    return 0;
}

PetscErrorCode LDCCheckAdmissibleUpDefect(LDC ldc, Vec z, PetscBool *flg) {
    PetscCall(VecLessThanOrEqual(ldc.dal,ldc.chilow,z,flg));
    if (!(*flg))
        return 0;
    PetscCall(VecLessThanOrEqual(ldc.dal,z,ldc.chiupp,flg));
    return 0;
}
