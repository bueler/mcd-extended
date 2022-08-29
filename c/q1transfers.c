#include <petsc.h>
#include "q1transfers.h"

PetscErrorCode Q1Interpolate(DM dac, DM daf, Vec vcoarse, Vec *vfine) {
    Mat Ainterp;
    PetscCall(DMCreateInterpolation(dac,daf,&Ainterp,NULL));
    PetscCall(MatInterpolate(Ainterp,vcoarse,*vfine));
    PetscCall(MatDestroy(&Ainterp));
    return 0;
}

PetscErrorCode Q1Restrict(DM daf, DM dac, Vec vfine, Vec *vcoarse) {
    Mat Ainterp;
    Vec vscale;
    PetscCall(DMCreateInterpolation(dac, daf,&Ainterp,&vscale));
    PetscCall(MatRestrict(Ainterp,vfine,*vcoarse));
    PetscCall(VecPointwiseMult(*vcoarse,vscale,*vcoarse));
    PetscCall(VecDestroy(&vscale));
    PetscCall(MatDestroy(&Ainterp));
    return 0;
}

PetscErrorCode Q1Inject(DM daf, DM dac, Vec vfine, Vec *vcoarse) {
    Mat Ainject;
    PetscCall(DMCreateInjection(dac,daf,&Ainject));
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
    INTERIOR, E, N, S, W, NE, NW, SW, SE
} _DirectionType;

// find optimum values of au[][] at points +,o in 9-, 6-, or 4-point
// neighborhood of o, according to dir:
//   INTERIOR    E        N        S        W     NE       NW     SW   SE
//    + + +      + +    + + +             + +     + +    + +
//    + o +      o +    + o +    + o +    + o     o +    + o    + o    o +
//    + + +      + +             + + +    + +                   + +    + +
// (note * has indices i,j)
PetscReal _OptNeighbors(Q1MonotoneType opt, _DirectionType dir,
                        PetscReal **au, PetscInt i, PetscInt j) {
    PetscInt        p, q;
    const PetscInt  ps[9] = { -1,  0, -1, -1, -1,  0, -1, -1,  0},
                    pe[9] = {  1,  1,  1,  1,  0,  1,  0,  0,  1},
                    qs[9] = { -1, -1,  0, -1, -1,  0,  0, -1, -1},
                    qe[9] = {  1,  1,  1,  0,  1,  1,  1,  0,  0};
    PetscReal       x = au[j][i];
    for (q=qs[(int)dir]; q<=qe[(int)dir]; q++) {
        for (p=ps[(int)dir]; p<=pe[(int)dir]; p++) {
            if (opt == Q1_MAX)
                x = PetscMax(x, au[j+q][i+p]);
            else
                x = PetscMin(x, au[j+q][i+p]);
        }
    }
    return x;
}

PetscErrorCode Q1MonotoneRestrict(Q1MonotoneType opt, DM daf, DM dac,
                                  Vec vfine, Vec *vcoarse) {
    DMDALocalInfo cinfo;
    Vec           vfineloc;
    PetscInt      ic, jc;
    PetscReal     **ac, **af;
    PetscCall(DMDAGetLocalInfo(dac,&cinfo));
    PetscCall(DMGetLocalVector(daf,&vfineloc));
    PetscCall(DMGlobalToLocal(daf,vfine,INSERT_VALUES,vfineloc));
    PetscCall(DMDAVecGetArray(dac,*vcoarse,&ac));
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
    PetscCall(DMDAVecRestoreArray(dac,*vcoarse,&ac));
    PetscCall(DMRestoreLocalVector(daf,&vfineloc));
    return 0;
}