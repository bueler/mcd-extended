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
    Vec            w;
    LDC            ldc[2];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // create LDC stack
    PetscCall(LDCCreateCoarsest(PETSC_TRUE,3,3,-2.0,2.0,-2.0,2.0,&(ldc[0])));
    PetscCall(LDCRefine(&(ldc[0]),&(ldc[1])));

    // view DMDA at each level
    PetscCall(PetscObjectSetName((PetscObject)(ldc[0].dal),"ldc[0].dal"));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[1].dal),"ldc[1].dal"));
    PetscCall(DMView(ldc[0].dal, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMView(ldc[1].dal, PETSC_VIEWER_STDOUT_WORLD));

    // iterate w=1 gives up defect constraint on finest level
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"using iterate w=1 to generate finest-level up defect constraints\n"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&w));
    PetscCall(VecSet(w,1.0));
    PetscCall(LDCFinestUpDefectConstraintsFromFormulas(w,NULL,&gamma_lower,&(ldc[1])));
    PetscCall(VecDestroy(&w));

    // generate up and down defect constraints for both levels
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"generating defect constraints for V-cycle\n"));
    PetscCall(LDCGenerateDefectConstraintsVCycle(&(ldc[1])));

    // ranges on Vecs on each level
    PetscCall(LDCReportRanges(ldc[0]));
    PetscCall(LDCReportRanges(ldc[1]));

    // destroy
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
