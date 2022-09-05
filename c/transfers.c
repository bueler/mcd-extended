static char help[] =
"Test Q1 transfer operators on a two-level LDC stack.  The region is the\n"
"square [0,1]^2, the fine level is a 5x5 grid and the coarse is a 3x3 grid.\n"
"The test here only addresses the DMDAs at the two level.  (Defect\n"
"constraints are not tested; compare ldctwolevel.c.)\n\n";

#include <petsc.h>
#include "src/q1transfers.h"
#include "src/ldc.h"

// z = gamma_lower(x,y) has tight bounds  0 <= z <= 1
PetscReal gamma_lower(PetscReal x, PetscReal y) {
    return 16.0 * x * (1.0 - x) * y * (1.0 - y);
}

extern PetscErrorCode VecViewMatlabStdout(Vec);

int main(int argc,char **argv) {
    LDC            ldc[2];
    DMDALocalInfo  info;
    Vec            vgamlow, vcoarseFW, vcoarseINJ, vfine;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // create LDC stack
    PetscCall(LDCCreateCoarsest(PETSC_TRUE,3,3,0.0,1.0,0.0,1.0,&(ldc[0])));
    PetscCall(LDCRefine(&(ldc[0]),&(ldc[1])));

    // create and view a Vec on the fine level
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&vgamlow));
    PetscCall(PetscObjectSetName((PetscObject)(vgamlow),"gamlow"));
    PetscCall(DMDAGetLocalInfo(ldc[1].dal,&info));
    //PetscCall(FormVecFromFormula(gamma_lower,&info,vgamlow));
    PetscCall(LDCVecFromFormula(ldc[1],gamma_lower,vgamlow));
    PetscCall(VecViewMatlabStdout(vgamlow));

    // test Q1 restriction, injection, and interpolation on temporary vecs
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&vcoarseFW));
    PetscCall(PetscObjectSetName((PetscObject)(vcoarseFW),"coarseFW"));
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&vcoarseINJ));
    PetscCall(PetscObjectSetName((PetscObject)(vcoarseINJ),"coarseINJ"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&vfine));
    PetscCall(PetscObjectSetName((PetscObject)(vfine),"fine"));
    PetscCall(Q1Restrict(ldc[1].dal,ldc[0].dal,vgamlow,&vcoarseFW));
    PetscCall(Q1Inject(ldc[1].dal,ldc[0].dal,vgamlow,&vcoarseINJ));
    PetscCall(Q1Interpolate(ldc[0].dal,ldc[1].dal,vcoarseFW,&vfine));
    PetscCall(VecViewMatlabStdout(vcoarseFW));
    PetscCall(VecViewMatlabStdout(vcoarseINJ));
    PetscCall(VecViewMatlabStdout(vfine));

    // destroy
    PetscCall(VecDestroy(&vfine));
    PetscCall(VecDestroy(&vcoarseINJ));
    PetscCall(VecDestroy(&vcoarseFW));
    PetscCall(VecDestroy(&vgamlow));
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
