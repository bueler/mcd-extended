#include <petsc.h>

PetscErrorCode VecPrintRange(Vec X, const char *name, const char *infcase) {
    PetscReal vmin, vmax;
    if (X) {
        PetscCall(VecMin(X,NULL,&vmin));
        PetscCall(VecMax(X,NULL,&vmax));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  %9.6f <= %s <= %9.6f\n",
                              vmin,name,vmax));
    } else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  [ %s=NULL is %s ]\n",
                              name,infcase));
    return 0;
}

PetscErrorCode VecLessThanOrEqual(DM da, Vec u, Vec v, PetscBool *flg) {
    Vec       w;
    PetscReal wmin;
    if (u == NULL || v == NULL)
        *flg = PETSC_TRUE;
    else {
        PetscCall(DMGetGlobalVector(da,&w));
        PetscCall(VecWAXPY(w,-1.0,u,v));   // w = v - u  (should be nonnegative)
        PetscCall(VecMin(w,NULL,&wmin));
        PetscCall(DMRestoreGlobalVector(da,&w));
        *flg = (wmin >= 0.0);
    }
    return 0;
}

PetscErrorCode CRFromResidual(DM da, Vec Upper, Vec Lower, Vec u, Vec F, Vec Fhat) {
    DMDALocalInfo    info;
    PetscInt         i, j;
    const PetscReal  **au, **aUpper, **aLower, **aF;
    PetscReal        **aFhat;
    PetscCall(DMDAGetLocalInfo(da,&info));
    if (Upper)
        PetscCall(DMDAVecGetArrayRead(da, Upper, &aUpper));
    if (Lower)
        PetscCall(DMDAVecGetArrayRead(da, Lower, &aLower));
    PetscCall(DMDAVecGetArrayRead(da, u, &au));
    PetscCall(DMDAVecGetArrayRead(da, F, &aF));
    PetscCall(DMDAVecGetArray(da, Fhat, &aFhat));
    for (j = info.ys; j < info.ys + info.ym; j++) {
        for (i = info.xs; i < info.xs + info.xm; i++) {
            if (Upper && au[j][i] >= aUpper[j][i])       // active upper constraint
                aFhat[j][i] = PetscMax(aF[j][i],0.0);
            else if (Lower && au[j][i] <= aLower[j][i])  // active lower constraint
                aFhat[j][i] = PetscMin(aF[j][i],0.0);
            else
                aFhat[j][i] = aF[j][i];                  // constraints inactive
        }
    }
    if (Upper)
        PetscCall(DMDAVecRestoreArrayRead(da, Upper, &aUpper));
    if (Lower)
        PetscCall(DMDAVecRestoreArrayRead(da, Lower, &aLower));
    PetscCall(DMDAVecRestoreArrayRead(da, u, &au));
    PetscCall(DMDAVecRestoreArrayRead(da, F, &aF));
    PetscCall(DMDAVecRestoreArray(da, Fhat, &aFhat));
    return 0;
}
