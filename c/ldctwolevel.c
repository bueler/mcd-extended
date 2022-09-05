static char help[] =
"Test a two-level LDC stack.  Here the upper DCs are +infty, but the\n"
"lower DCs are nontrivial.  The region is the square [0,1]^2, the\n"
"fine level is a 5x5 grid and the coarse is a 3x3 grid.  The fine-level\n"
"original lower obstacle has range 0 <= gamlow <= 1, and the iterate\n"
"w=2 is constant.\n\n";

#include <petsc.h>
#include "src/q1transfers.h"
#include "src/ldc.h"

// z = gamma_lower(x,y) has tight bounds  0 <= z <= 1
PetscReal gamma_lower(PetscReal x, PetscReal y) {
    return 16.0 * x * (1.0 - x) * y * (1.0 - y);
}

extern PetscErrorCode VecViewMatlabStdout(Vec);
extern PetscErrorCode FormVecFromFormula(PetscReal (*)(PetscReal,PetscReal),
                                         DMDALocalInfo*, Vec);

int main(int argc,char **argv) {
    Vec            w, v;
    PetscBool      admis;
    LDC            ldc[2];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // create LDC stack
    PetscCall(LDCCreateCoarsest(PETSC_TRUE,3,3,0.0,1.0,0.0,1.0,&(ldc[0])));
    PetscCall(LDCRefine(&(ldc[0]),&(ldc[1])));

    // view DMDA at each level
    PetscCall(PetscObjectSetName((PetscObject)(ldc[0].dal),"ldc[0].dal"));
    PetscCall(PetscObjectSetName((PetscObject)(ldc[1].dal),"ldc[1].dal"));
    PetscCall(DMView(ldc[0].dal, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMView(ldc[1].dal, PETSC_VIEWER_STDOUT_WORLD));

    // iterate w=1 gives up defect constraint on finest level
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"using iterate w=2 to generate finest-level up defect constraints\n"));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&w));
    PetscCall(VecSet(w,2.0));
    PetscCall(LDCFinestUpDCsFromFormulas(w,NULL,&gamma_lower,&(ldc[1])));
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

    // ranges on Vecs on each level
    PetscCall(LDCReportDCRanges(ldc[0]));
    PetscCall(LDCReportDCRanges(ldc[1]));

    // check admissibility of constant z on level 0 and y and z on level 1
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
