static char help[] =
"Test a two-level LDC stack.\n\n";

// FIXME check all results!

#include <petsc.h>
#include "src/q1transfers.h"
#include "src/ldc.h"

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

    // lower obstacle on fine level
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"at level 1: creating gamlow at level 1 from formula\n"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&(ldc[1].gamlow)));
    PetscCall(DMDAGetLocalInfo(ldc[1].dal,&info));
    PetscCall(FormVecFromFormula(gamma_lower,&info,ldc[1].gamlow));

#if 0
    // test Q1 restriction, injection, and interpolation on temporary vecs
    Vec vcoarseFW, vcoarseINJ, vfine;
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&vcoarseFW));
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&vcoarseINJ));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&vfine));
    PetscCall(Q1Restrict(ldc[1].dal,ldc[0].dal,ldc[1].gamlow,&vcoarseFW));
    PetscCall(Q1Inject(ldc[1].dal,ldc[0].dal,ldc[1].gamlow,&vcoarseINJ));
    PetscCall(Q1Interpolate(ldc[0].dal,ldc[1].dal,vcoarseFW,&vfine));
    PetscCall(VecViewMatlabStdout(ldc[1].gamlow));
    PetscCall(VecViewMatlabStdout(vcoarseFW));
    PetscCall(VecViewMatlabStdout(vcoarseINJ));
    PetscCall(VecViewMatlabStdout(vfine));
    PetscCall(VecDestroy(&vfine));
    PetscCall(VecDestroy(&vcoarseINJ));
    PetscCall(VecDestroy(&vcoarseFW));
#endif

    // iterate w=1 gives up defect constraint on fine level
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"at level 1: using admissible iterate w=1\n"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&w));
    PetscCall(VecSet(w,1.0));
    PetscCall(LDCUpDefectConstraintsFromObstacles(w,&(ldc[1])));
    PetscCall(LDCUpDefectConstraintsMonotoneRestrict(ldc[1],&(ldc[0])));

    // generate down defects
    PetscCall(LDCDownDefectConstraints(&(ldc[0]),&(ldc[1])));
    PetscCall(LDCDownDefectConstraints(NULL,&(ldc[0])));

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
