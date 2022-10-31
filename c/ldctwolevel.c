static char help[] =
"Test a two-level LDC stack.  The below LDCs are nontrivial.  With -unilateral\n"
"the above LDCs are +infty, otherwise they are finite.  The region is the\n"
"square [0,1]^2, the fine level is a 5x5 grid and the coarse is a 3x3 grid.\n"
"On the fine-level, the original lower obstacle has range 0 <= gamlow <= 1,\n"
"the original above obstacle is gamupp=1.0, and the iterate w=2 is constant.\n\n";

#include <petsc.h>
#include "src/q1transfers.h"
#include "src/ldc.h"

// z = gamma_lower(x,y) has tight bounds  0 <= z <= 1
PetscReal gamma_lower(PetscReal x, PetscReal y, void *ctx) {
    return 16.0 * x * (1.0 - x) * y * (1.0 - y);
}

// z = gamma_upper(x,y) = 3
PetscReal gamma_upper(PetscReal x, PetscReal y, void *ctx) {
    return 3.0;
}

extern PetscErrorCode VecViewMatlabStdout(Vec);

int main(int argc,char **argv) {
    Vec            gamlow, gamupp, w, v;
    PetscBool      unilateral, admis;
    LDC            ldc[2];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    PetscOptionsBegin(PETSC_COMM_WORLD,"","ldctwolevel options","");
    PetscCall(PetscOptionsBool("-unilateral","set original above obstacle to +infty",
                               "ldctwolevel.c",unilateral,&unilateral,NULL));
    PetscOptionsEnd();

    // create DMDA for coarsest level: 3x3 grid on on Omega = (0,1)x(0,1)
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(ldc[0].dal)));
    PetscCall(DMSetFromOptions(ldc[0].dal));  // allows -da_grid_x mx -da_grid_y my etc.
    PetscCall(DMDASetInterpolationType(ldc[0].dal,DMDA_Q1));
    PetscCall(DMDASetRefinementFactor(ldc[0].dal,2,2,2));
    PetscCall(DMSetUp(ldc[0].dal));  // must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(ldc[0].dal,0.0,1.0,0.0,1.0,0.0,0.0));

    // create LDC stack
    PetscCall(LDCCreateCoarsest(PETSC_TRUE,&(ldc[0])));
    PetscCall(LDCRefine(&(ldc[0]),&(ldc[1])));

    // view DMDA at each level
    PetscCall(PetscObjectSetName((PetscObject)(ldc[0].dal),"ldc[0].dal"));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[1].dal),"ldc[1].dal"));
    PetscCall(DMView(ldc[0].dal, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMView(ldc[1].dal, PETSC_VIEWER_STDOUT_WORLD));

    // iterate w=2 gives up LDCs on finest level
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"using iterate w=2 to generate finest-level up defect constraints\n"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&w));
    PetscCall(VecSet(w,2.0));
    PetscCall(DMGetGlobalVector(ldc[1].dal,&gamlow));
    PetscCall(LDCVecFromFormula(ldc[1],gamma_lower,gamlow,NULL));
    if (unilateral) {
        gamupp = NULL;
    } else {
        PetscCall(DMGetGlobalVector(ldc[1].dal,&gamupp));
        PetscCall(LDCVecFromFormula(ldc[1],gamma_upper,gamupp,NULL));
    }
    PetscCall(LDCSetFinestUpDCs(w,gamupp,gamlow,&(ldc[1])));
    PetscCall(DMRestoreGlobalVector(ldc[1].dal,&gamlow));
    if (!unilateral) {
        PetscCall(DMRestoreGlobalVector(ldc[1].dal,&gamupp));
    }
    PetscCall(VecDestroy(&w));

    // generate up and down defect constraints for both levels
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"generating defect constraints for V-cycle\n"));
    PetscCall(LDCGenerateDCsVCycle(&(ldc[1])));

#if 0
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"printing lower defect constraints in Matlab format\n"));
    // print ldc[0].chilow and ldc[1].chilow to check that monotone restriction
    // did the right thing; in Matlab use "reshape(chilow0,3,3)" etc
    PetscCall(PetscObjectSetName((PetscObject)(ldc[0].chilow),"chilow0"));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[1].chilow),"chilow1"));
    PetscCall(VecViewMatlabStdout(ldc[0].chilow));
    PetscCall(VecViewMatlabStdout(ldc[1].chilow));
    // print ldc[0].philow and ldc[1].philow to check subtraction etc.
    PetscCall(PetscObjectSetName((PetscObject)(ldc[0].philow),"philow0"));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[1].philow),"philow1"));
    PetscCall(VecViewMatlabStdout(ldc[0].philow));
    PetscCall(VecViewMatlabStdout(ldc[1].philow));
#endif

    // check admissibility of different constants z0 on level 0 and y1 and z1 on level 1
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"checking admissibility of constant vectors:\n"));
    PetscCall(DMCreateGlobalVector(ldc[0].dal,&v));
    PetscCall(VecSet(v,0.0));
    PetscCall(LDCCheckAdmissibleUpDefect(ldc[0],v,&admis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    z0=0     admissible up   defect? %d\n",(int)admis));
    PetscCall(VecSet(v,-1.25));
    PetscCall(LDCCheckAdmissibleUpDefect(ldc[0],v,&admis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    z0=-1.25 admissible up   defect? %d\n",(int)admis));
    PetscCall(VecDestroy(&v));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&v));
    PetscCall(VecSet(v,0.0));
    PetscCall(LDCCheckAdmissibleDownDefect(ldc[1],v,&admis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    y1=0     admissible down defect? %d\n",(int)admis));
    PetscCall(VecSet(v,-1.00));
    PetscCall(LDCCheckAdmissibleDownDefect(ldc[1],v,&admis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    y1=-1.00 admissible down defect? %d\n",(int)admis));
    PetscCall(VecSet(v,-1.00));
    PetscCall(LDCCheckAdmissibleUpDefect(ldc[1],v,&admis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    z1=-1.00 admissible up   defect? %d\n",(int)admis));
    PetscCall(VecSet(v,-1.25));
    PetscCall(LDCCheckAdmissibleUpDefect(ldc[1],v,&admis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    z1=-1.25 admissible up   defect? %d\n",(int)admis));
    PetscCall(VecDestroy(&v));

    // destroy
    PetscCall(LDCDestroy(&(ldc[1])));
    PetscCall(LDCDestroy(&(ldc[0])));  // destroys cdmda
    PetscCall(PetscFinalize());
    return 0;
}


PetscErrorCode VecViewMatlabStdout(Vec v) {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
    return 0;
}
