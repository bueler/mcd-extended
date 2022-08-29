static char help[] =
"Test a two-level LDC stack.\n\n";

// FIXME: generate and test Mats for interpolation and restriction and injection
// FIXME: implement and test monotone restriction

#include <petsc.h>
#include "ldc.h"


// z = gamma_lower(x,y) is the hemispherical obstacle, but made C^1 with "skirt" at r=r0
PetscReal gamma_lower(PetscReal x, PetscReal y) {
    const PetscReal  r = PetscSqrtReal(x * x + y * y),
                     r0 = 0.9,
                     psi0 = PetscSqrtReal(1.0 - r0 * r0),
                     dpsi0 = - r0 / psi0;
    if (r <= r0) {
        return PetscSqrtReal(1.0 - r * r);
    } else {
        return psi0 + dpsi0 * (r - r0);
    }
}

extern PetscErrorCode VecViewMatlabStdout(Vec);
extern PetscErrorCode FormVecFromFormula(PetscReal (*)(PetscReal,PetscReal),
                                         DMDALocalInfo*, Vec);

int main(int argc,char **argv) {
    DMDALocalInfo  info;
    Vec            w;
    LDC            ldc[2];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // create LDC stack
    PetscCall(LDCCreate(PETSC_TRUE,0,3,3,-2.0,2.0,-2.0,2.0,&(ldc[0])));
    PetscCall(LDCRefine(ldc[0], &(ldc[1])));

    // view DMDA at each level
    PetscCall(PetscOptionsSetValue(NULL, "-dm_view", ""));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[0].dal),"ldc[0].dal"));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[1].dal),"ldc[1].dal"));
    PetscCall(DMViewFromOptions(ldc[0].dal, NULL, "-dm_view"));
    PetscCall(DMViewFromOptions(ldc[1].dal, NULL, "-dm_view"));

    // gamma_lower obstacle on fine level
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"at level 1: creating gamlow at level 1 from formula\n"));
    PetscCall(DMDAGetLocalInfo(ldc[1].dal,&info));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&(ldc[1].gamlow)));
    PetscCall(FormVecFromFormula(gamma_lower,&info,ldc[1].gamlow));

#if 0
    // test Q1 restriction, injection, and interpolation on temporary vecs
    Vec vcoarseFW, vcoarseINJ, vfine;
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&vcoarseFW));
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&vcoarseINJ));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&vfine));
    PetscCall(LDCQ1RestrictVec(ldc[1],ldc[0],ldc[1].gamlow,&vcoarseFW));
    PetscCall(LDCQ1InjectVec(ldc[1],ldc[0],ldc[1].gamlow,&vcoarseINJ));
    PetscCall(LDCQ1InterpolateVec(ldc[0],ldc[1],vcoarseFW,&vfine));
    PetscCall(VecViewMatlabStdout(ldc[1].gamlow));
    PetscCall(VecViewMatlabStdout(vcoarseFW));
    PetscCall(VecViewMatlabStdout(vcoarseINJ));
    PetscCall(VecViewMatlabStdout(vfine));
    PetscCall(VecDestroy(&vfine));
    PetscCall(VecDestroy(&vcoarseINJ));
    PetscCall(VecDestroy(&vcoarseFW));
#endif

    // zero iterate w generates up defect constraints (FIXME only on fine level so far)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"at level 1: using iterate w=1.000000\n"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&w));
    PetscCall(VecSet(w,1.0));
    PetscCall(LDCUpDefectsFromObstacles(w,&(ldc[1])));
    // FIXME monotone restrict for chiupp,chilow on ldc[0]
    //PetscCall(LDCUpDefectsMonotoneRestrict(ldc[1],&(ldc[0])));

    // generate down defects
    // FIXME untested
    //PetscCall(LDCDownDefects(&(ldc[0]),&(ldc[1])));
    //PetscCall(LDCDownDefects(NULL,&(ldc[0])));

    // ranges on Vecs on each level
    PetscCall(LDCReportRanges(ldc[0]));
    PetscCall(LDCReportRanges(ldc[1]));

    // destroy
    PetscCall(VecDestroy(&w));
    PetscCall(LDCDestroy(&(ldc[1])));
    PetscCall(LDCDestroy(&(ldc[0])));
    PetscCall(PetscFinalize());
    return 0;
}


PetscErrorCode VecViewMatlabStdout(Vec v) {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
    return 0;
}

PetscErrorCode FormVecFromFormula(PetscReal (*ufcn)(PetscReal,PetscReal),
                                  DMDALocalInfo *info, Vec u) {
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    hx = 4.0 / (PetscReal)(info->mx - 1);
    hy = 4.0 / (PetscReal)(info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -2.0 + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -2.0 + i * hx;
            au[j][i] = (*ufcn)(x,y);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}
