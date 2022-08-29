#include <petsc.h>
#include "ldc.h"

PetscErrorCode LDCCreate(PetscBool verbose, PetscInt level,
                         PetscInt mx, PetscInt my,
                         PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax,
                         LDC *ldc) {
    ldc->_level = level;
    ldc->_printinfo = verbose;
    if (ldc->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "LDC info: creating LDC at level %d based on %d x %d grid DMDA\n",
        ldc->_level,mx,my));
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           mx,my,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(ldc->dal)));
    // make defaults explicit:
    PetscCall(DMDASetInterpolationType(ldc->dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(ldc->dal,2,2,2));
    PetscCall(DMSetUp(ldc->dal));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(ldc->dal,xmin,xmax,ymin,ymax,0.0,0.0));
    ldc->gamupp = NULL;
    ldc->gamlow = NULL;
    ldc->chiupp = NULL;
    ldc->chilow = NULL;
    ldc->phiupp = NULL;
    ldc->philow = NULL;
    return 0;
}

PetscErrorCode LDCDestroy(LDC *ldc) {
    if (ldc->_printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "LDC info: destroying LDC at level %d\n",ldc->_level));
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
    fine->_level = coarse._level + 1;
    if (coarse._printinfo)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "LDC info: refining coarse LDC at level %d to generate fine LDC at level %d\n",
        coarse._level,fine->_level));
    fine->_printinfo = coarse._printinfo;
    PetscCall(DMRefine(coarse.dal,PETSC_COMM_WORLD,&(fine->dal)));
    PetscCall(DMDASetInterpolationType(fine->dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(fine->dal,2,2,2));
    fine->gamupp = NULL;
    fine->gamlow = NULL;
    fine->chiupp = NULL;
    fine->chilow = NULL;
    fine->phiupp = NULL;
    fine->philow = NULL;
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
                          ldc._level));
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
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating chiupp and setting chiupp=gamupp-w at level %d\n",
            ldc->_level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chiupp)));
        PetscCall(VecWAXPY(ldc->chiupp,-1.0,w,ldc->gamupp));  // chiupp = gamupp - w
    } else
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: chiupp=NULL because gamupp=+infty\n"));
    if (ldc->gamlow) {
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating chilow and setting chilow=gamlow-w at level %d\n",
            ldc->_level));
        PetscCall(DMCreateGlobalVector(ldc->dal,&(ldc->chilow)));
        PetscCall(VecWAXPY(ldc->chilow,-1.0,w,ldc->gamlow));  // chilow = gamlow - w
    } else
        if (ldc->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: chilow=NULL because gamlow=-infty\n"));
    return 0;
}

PetscErrorCode LDCQ1InterpolateVec(LDC coarse, LDC fine, Vec vcoarse, Vec *vfine) {
    Mat Ainterp;
    PetscCall(DMCreateInterpolation(coarse.dal, fine.dal, &Ainterp, NULL));
    PetscCall(MatInterpolate(Ainterp,vcoarse,*vfine));
    PetscCall(MatDestroy(&Ainterp));
    return 0;
}

PetscErrorCode LDCQ1RestrictVec(LDC fine, LDC coarse, Vec vfine, Vec *vcoarse) {
    Mat Ainterp;
    Vec vscale;
    PetscCall(DMCreateInterpolation(coarse.dal, fine.dal, &Ainterp, &vscale));
    PetscCall(MatRestrict(Ainterp,vfine,*vcoarse));
    PetscCall(VecPointwiseMult(*vcoarse,vscale,*vcoarse));
    PetscCall(VecDestroy(&vscale));
    PetscCall(MatDestroy(&Ainterp));
    return 0;
}

PetscErrorCode LDCQ1InjectVec(LDC fine, LDC coarse, Vec vfine, Vec *vcoarse) {
    Mat Ainject;
    PetscCall(DMCreateInjection(coarse.dal, fine.dal, &Ainject));
    PetscCall(MatRestrict(Ainject,vfine,*vcoarse));
    PetscCall(MatDestroy(&Ainject));
    return 0;
}

PetscBool _NodeOnBdry(DM da, PetscInt i, PetscInt j) {
    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(da,&info));
    return (((i == 0) || (i == info.mx-1) || (j == 0) || (j == info.my-1)));
}

typedef enum {
    MAX, MIN
} OptimumType;

typedef enum {
    INTERIOR, E, N, S, W, NE, NW, SW, SE
} DirectionType;

// find optimum values of au[][] at points +,o in 9-, 6-, or 4-point
// neighborhood of o, according to dir:
//   INTERIOR    E        N        S        W     NE       NW     SW   SE
//    + + +      + +    + + +             + +     + +    + +
//    + o +      o +    + o +    + o +    + o     o +    + o    + o    o +
//    + + +      + +             + + +    + +                   + +    + +
// (note * has indices i,j)
PetscReal _OptNeighbors(OptimumType opt, DirectionType dir,
                        PetscReal **au, PetscInt i, PetscInt j) {
    PetscInt        p, q;
    const PetscInt  ps[9] = { -1,  0, -1, -1, -1,  0, -1, -1,  0},
                    pe[9] = {  1,  1,  1,  1,  0,  1,  0,  0,  1},
                    qs[9] = { -1, -1,  0, -1, -1,  0,  0, -1, -1},
                    qe[9] = {  1,  1,  1,  0,  1,  1,  1,  0,  0};
    PetscReal       x = au[j][i];
    for (q=qs[(int)dir]; q<=qe[(int)dir]; q++) {
        for (p=ps[(int)dir]; p<=pe[(int)dir]; p++) {
            if (opt == MAX)
                x = PetscMax(x, au[j+q][i+p]);
            else
                x = PetscMin(x, au[j+q][i+p]);
        }
    }
    return x;
}

// assumes both Vecs are already created
PetscErrorCode _MonotoneRestrict(OptimumType opt, DM dac, DM daf,
                                 Vec vfine, Vec vcoarse) {
    DMDALocalInfo cinfo;
    Vec           vfineloc;
    PetscInt      ic, jc;
    PetscReal     **ac, **af;
    PetscCall(DMDAGetLocalInfo(dac,&cinfo));
    PetscCall(DMGetLocalVector(daf,&vfineloc));
    PetscCall(DMGlobalToLocal(daf,vfine,INSERT_VALUES,vfineloc));
    PetscCall(DMDAVecGetArray(dac,vcoarse,&ac));
    PetscCall(DMDAVecGetArray(daf,vfineloc,&af));
    for (jc=cinfo.ys; jc<cinfo.ys+cinfo.ym; jc++) {
        for (ic=cinfo.xs; ic<cinfo.xs+cinfo.xm; ic++) {
            if (!_NodeOnBdry(dac,ic,jc)) {
                ac[jc][ic] = _OptNeighbors(opt,INTERIOR,af,2*ic,2*jc);
                continue;
            }
            // special cases for boundary nodes
            if (ic == 0) {
                // along left side of domain
                if (jc == 0)
                    ac[jc][ic] = _OptNeighbors(opt,NE,af,2*ic,2*jc);
                else if (jc == cinfo.my-1)
                    ac[jc][ic] = _OptNeighbors(opt,SE,af,2*ic,2*jc);
                else
                    ac[jc][ic] = _OptNeighbors(opt,E,af,2*ic,2*jc);
            } else if (ic == cinfo.mx-1) {
                // along right side of domain
                if (jc == 0)
                    ac[jc][ic] = _OptNeighbors(opt,NW,af,2*ic,2*jc);
                else if (jc == cinfo.my-1)
                    ac[jc][ic] = _OptNeighbors(opt,SW,af,2*ic,2*jc);
                else
                    ac[jc][ic] = _OptNeighbors(opt,W,af,2*ic,2*jc);
            } else {
                // along bottom or top sides of domain, not at corners
                if (jc == 0)
                    ac[jc][ic] = _OptNeighbors(opt,N,af,2*ic,2*jc);
                else
                    ac[jc][ic] = _OptNeighbors(opt,S,af,2*ic,2*jc);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(daf,vfineloc,&af));
    PetscCall(DMDAVecRestoreArray(dac,vcoarse,&ac));
    PetscCall(DMRestoreLocalVector(daf,&vfineloc));
    return 0;
}

PetscErrorCode LDCUpDefectsMonotoneRestrict(LDC fine, LDC *coarse) {
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
        PetscCall(DMCreateGlobalVector(coarse->dal,&(coarse->chiupp)));
        PetscCall(_MonotoneRestrict(MIN,coarse->dal,fine.dal,fine.chiupp,coarse->chiupp));
    }
    if (fine.chilow) {
        PetscCall(DMCreateGlobalVector(coarse->dal,&(coarse->chilow)));
        PetscCall(_MonotoneRestrict(MAX,coarse->dal,fine.dal,fine.chilow,coarse->chilow));
    }
    return 0;
}

PetscErrorCode LDCDownDefects(LDC *coarse, LDC *fine) {
    // generate phiupp
    if (!coarse) {
        if (fine->chiupp) {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: creating phiupp and setting phiupp=chiupp (coarsest case) at level %d\n",
                fine->_level));
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
            PetscCall(VecCopy(fine->chiupp,fine->phiupp));
        } else {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: phiupp=NULL at level %d\n",fine->_level));
        }
    } else if (coarse->chiupp && fine->chiupp) {
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating phiupp and setting phiupp=chiupp-chiupp_coarse at level %d\n",
            fine->_level));
        PetscCall(DMCreateGlobalVector(fine->dal,&(fine->phiupp)));
        PetscCall(VecWAXPY(fine->phiupp,-1.0,coarse->chiupp,fine->chiupp));  // phiupp = chiupp - chiupp_coarse
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: unanticipated case");
    }
    // generate philow
    if (!coarse) {
        if (fine->chilow) {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: creating philow and setting philow=chilow (coarsest case) at level %d\n",
                fine->_level));
            PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
            PetscCall(VecCopy(fine->chilow,fine->philow));
        } else {
            if (fine->_printinfo)
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "LDC info: philow=NULL at level %d\n",fine->_level));
        }
    } else if (coarse->chilow && fine->chilow) {
        if (fine->_printinfo)
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "LDC info: creating philow and setting phiupp=chilow-chilow_coarse at level %d\n",
            fine->_level));
        PetscCall(DMCreateGlobalVector(fine->dal,&(fine->philow)));
        PetscCall(VecWAXPY(fine->philow,-1.0,coarse->chilow,fine->chilow));  // philow = chilow - chilow_coarse
    } else {
        SETERRQ(PETSC_COMM_SELF,1,"LDC ERROR: unanticipated case");
    }
    return 0;
}
