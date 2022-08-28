static char help[] =
"Test a two-level LDC stack.\n\n";

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

extern PetscErrorCode FormVecFromFormula(PetscReal (*)(PetscReal,PetscReal),
                                         DMDALocalInfo*, Vec);

int main(int argc,char **argv) {
    DM             coarseda;
    DMDALocalInfo  info;
    LDC            ldc[2];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // create coarse DMDA
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&coarseda));
    PetscCall(DMSetFromOptions(coarseda));
    PetscCall(DMSetUp(coarseda));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(coarseda,-2.0,2.0,-2.0,2.0,0.0,1.0));

    // create LDC stack
    PetscCall(LDCCreate(0,coarseda,&(ldc[0])));
    PetscCall(LDCRefine(ldc[0], &(ldc[1])));

    // gamma_lower obstacle on fine level
    PetscCall(DMDAGetLocalInfo(ldc[1].dal,&info));
    PetscCall(DMCreateGlobalVector(ldc[1].dal,&(ldc[1].gamlow)));
    PetscCall(FormVecFromFormula(gamma_lower,&info,ldc[1].gamlow));

    // view stuff
    PetscCall(PetscOptionsSetValue(NULL, "-dm_view", ""));
    PetscCall(PetscOptionsSetValue(NULL, "-vec_view", "::ascii_matlab"));
    PetscCall(DMViewFromOptions(ldc[0].dal, NULL, "-dm_view"));
    PetscCall(DMViewFromOptions(ldc[1].dal, NULL, "-dm_view"));
    PetscCall(VecViewFromOptions(ldc[1].gamlow, NULL, "-vec_view"));

    // destroy
    PetscCall(LDCDestroy(&(ldc[1])));
    PetscCall(LDCDestroy(&(ldc[0])));
    PetscCall(PetscFinalize());
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
