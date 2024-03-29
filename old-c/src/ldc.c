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
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: allocate DM coarse->dal before calling LDCRefine()");
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
    fine->_coarser = coarse;
    return 0;
}

PetscErrorCode LDCCheckDCRanges(LDC ldc, PetscBool *zerobracket) {
    PetscReal clmax = PETSC_NINFINITY, cumin = PETSC_INFINITY,
              plmax = PETSC_NINFINITY, pumin = PETSC_INFINITY;
    if (ldc.chilow)
        PetscCall(VecMin(ldc.chilow,NULL,&clmax));
    if (ldc.chiupp)
        PetscCall(VecMin(ldc.chiupp,NULL,&cumin));
    if (ldc._level > 0) {
        if (ldc.philow)
            PetscCall(VecMin(ldc.philow,NULL,&plmax));
        if (ldc.phiupp)
            PetscCall(VecMin(ldc.phiupp,NULL,&pumin));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"           "));
    PetscCall(VecPrintRange(ldc.chilow,"chilow","-infty",PETSC_FALSE));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    "));
    PetscCall(VecPrintRange(ldc.chiupp,"chiupp","+infty",PETSC_TRUE));
    if (ldc._level > 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"           "));
        PetscCall(VecPrintRange(ldc.philow,"philow","-infty",PETSC_FALSE));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    "));
        PetscCall(VecPrintRange(ldc.phiupp,"phiupp","+infty",PETSC_TRUE));
    }
    *zerobracket = ((clmax <= 0.0) && (0.0 <= cumin) && (plmax <= 0.0) && (0.0 <= pumin));
    return 0;
}

PetscErrorCode LDCSetFinestUpDCs(Vec w, Vec vgamupp, Vec vgamlow, LDC *ldc) {
    if (vgamupp) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating chiupp and setting chiupp=gamupp-w at level %d\n",
            ldc->_level));
        if (!ldc->chiupp)
            PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chiupp)));
        PetscCall(VecWAXPY(ldc->chiupp,-1.0,w,vgamupp));  // chiupp = gamupp - w
    } else {
        if (ldc->chiupp) {
            SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: conflicting state for vgamupp versus ldc->chiupp");
        }
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chiupp=NULL is +infty because gamupp=NULL at level %d\n",
            ldc->_level));
    }
    if (vgamlow) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating chilow and setting chilow=gamlow-w at level %d\n",
            ldc->_level));
        if (!ldc->chilow)
            PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chilow)));
        PetscCall(VecWAXPY(ldc->chilow,-1.0,w,vgamlow));  // chilow = gamlow - w
    } else {
        if (ldc->chilow) {
            SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: conflicting state for vgamlow versus ldc->chilow");
        }
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chilow=NULL is -infty because gamlow=NULL at level %d\n",
            ldc->_level));
    }
    return 0;
}

PetscErrorCode _LDCUpDCsMonotoneRestrict(LDC fine, LDC *coarse) {
    if (!coarse) {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: coarse not created");
    }
    if (fine.chiupp) {
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: setting chiupp at level %d using monotone restriction from level %d\n",
            coarse->_level,fine._level));
        if (!coarse->chiupp)
            PetscCall(DMCreateGlobalVector(coarse->dal,&(coarse->chiupp)));
        PetscCall(Q1MonotoneRestrict(MONOTONE_MIN,fine.dal,coarse->dal,
                                     fine.chiupp,&(coarse->chiupp)));
    } else {
        if (coarse->chiupp) {
            SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: conflicting state for coarse->chiupp");
        }
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chiupp=NULL is +infty at level %d\n",
            coarse->_level));
    }
    if (fine.chilow) {
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: setting chilow at level %d using monotone restriction from level %d\n",
            coarse->_level,fine._level));
        if (!coarse->chilow)
            PetscCall(DMCreateGlobalVector(coarse->dal,&(coarse->chilow)));
        PetscCall(Q1MonotoneRestrict(MONOTONE_MAX,fine.dal,coarse->dal,
                                     fine.chilow,&(coarse->chilow)));
    } else {
        if (coarse->chilow) {
            SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: conflicting state for coarse->chilow");
        }
        if (coarse->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: chilow=NULL is -infty at level %d\n",
            coarse->_level));
    }
    return 0;
}

PetscErrorCode _LDCDownDCs(LDC *coarse, LDC *fine) {
    // generate phiupp
    if (!coarse) {
        if (fine->chiupp) {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: creating and setting phiupp=chiupp (coarsest case) at level %d\n",
                fine->_level));
            if (!fine->phiupp)
                PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
            PetscCall(VecCopy(fine->chiupp,fine->phiupp));
        } else {
            if (fine->phiupp) {
                SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: conflicting state for fine->phiupp");
            }
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: phiupp=NULL is +infty (coarsest case) at level %d\n",fine->_level));
        }
    } else if (coarse->chiupp && fine->chiupp) {
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating and setting phiupp=chiupp-chiupp_coarse at level %d\n",
            fine->_level));
        if (!fine->phiupp)
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
        // phiupp = chiupp - P(chiupp_coarse)
        PetscCall(Q1Interpolate(coarse->dal,fine->dal,coarse->chiupp,&(fine->phiupp)));
        PetscCall(VecAYPX(fine->phiupp,-1.0,fine->chiupp));
    } else {
        if (fine->phiupp) {
            SETERRQ(PETSC_COMM_SELF,2,"LDC ERROR: conflicting state for fine->phiupp");
        }
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: phiupp=NULL is +infty at level %d\n",fine->_level));
    }
    // generate philow
    if (!coarse) {
        if (fine->chilow) {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: creating and setting philow=chilow (coarsest case) at level %d\n",
                fine->_level));
            if (!fine->philow)
                PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
            PetscCall(VecCopy(fine->chilow,fine->philow));
        } else {
            if (fine->philow) {
                SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: conflicting state for fine->philow");
            }
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "  LDC info: philow=NULL is -infty (coarsest case) at level %d\n",fine->_level));
        }
    } else if (coarse->chilow && fine->chilow) {
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: creating and setting philow=chilow-chilow_coarse at level %d\n",
            fine->_level));
        if (!fine->philow)
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
        // philow = chilow - P(chilow_coarse)
        PetscCall(Q1Interpolate(coarse->dal,fine->dal,coarse->chilow,&(fine->philow)));
        PetscCall(VecAYPX(fine->philow,-1.0,fine->chilow));
    } else {
        if (fine->philow) {
            SETERRQ(PETSC_COMM_SELF,2,"LDC ERROR: conflicting state for fine->philow");
        }
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: philow=NULL is -infty at level %d\n",fine->_level));
    }
    return 0;
}

PetscErrorCode LDCSetLevel(LDC *fine) {
    LDC       *coarse = NULL;
    PetscBool bracket;
    if (fine->_coarser) {
        coarse = (LDC*)(fine->_coarser);
        // generate chiupp, chilow on level fine
        PetscCall(_LDCUpDCsMonotoneRestrict(*fine,coarse));
        // generate phiupp, philow on level fine
        PetscCall(_LDCDownDCs(coarse,fine));
    } else
        // generate chiupp, chilow on coarsest level
        PetscCall(_LDCDownDCs(NULL,fine));
    if (fine->_printinfo) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: DC ranges on level %d:\n",fine->_level));
        PetscCall(LDCCheckDCRanges(*fine,&bracket));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: zero bracketed checks %s on level %d\n",
            bracket ? "PASS" : "FALL", fine->_level));
    }
    if ((coarse) && (!coarse->_coarser) && (coarse->_printinfo)) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: DC ranges on level %d:\n",coarse->_level));
        PetscCall(LDCCheckDCRanges(*coarse,&bracket));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "  LDC info: zero bracketed checks %s on level %d\n",
            bracket ? "PASS" : "FALL", coarse->_level));
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
