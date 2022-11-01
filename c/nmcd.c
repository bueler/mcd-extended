static char help[] =
"Solve classical, unilateral obstacle problem by nonlinear multilevel constraint\n"
"decomposition (NMCD) method using Q1 finite elements in 2D square (-2,2)^2\n"
"on a structured-grid (DMDA):\n"
"  - nabla^2 u = f(x,y),  u >= psi(x,y),\n"
"subject to Dirichlet boundary conditions u=g.  Optional problem (-nm_bratu)\n"
"is unconstrained Bratu equation  - nabla^2 u - e^u = 0,  with Liouville\n"
"exact solution, on square (0,1)^2.\n"
"Smoother and coarse-level solver are both projected, nonlinear Gauss-Seidel\n"
"(PNGS) sweeps.  Option prefix nm_.  Compare obstaclesl.c and bratu.c.\n\n";

// FIXME add unconstrained bratu case, and compare to FAS solves
// (w/o line search) from bratu.c

// FIXME possible ways to describe possible current issues:
//   * the iterate w is only slowly falling toward the obstacle
//   * the corrections are always negative
// see stdout and foo.m from
//   $ ./nmcd -nm_monitor -nm_monitor_ranges -nm_monitor_vcycles -nm_view foo.m -nm_levels 7 -nm_cycles 1 -nm_bumpsize 1.0

#include <petsc.h>
#include "src/utilities.h"
#include "src/q1fem.h"
#include "src/q1transfers.h"
#include "src/ldc.h"

typedef struct {
  // obstacle gamma_lower(x,y)
  PetscReal (*gamma_lower)(PetscReal x, PetscReal y, void *ctx);
  // right-hand side f(x,y)
  PetscReal (*f_rhs)(PetscReal x, PetscReal y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscReal (*g_bdry)(PetscReal x, PetscReal y, void *ctx);
  PetscInt  quadpts,     // number n of quadrature points (= 1,2,3 only)
            maxits,      // in PNGS, number of Newton iterations at each point
            sweeps,      // in PNGS, number of sweeps over the grid
            residualcount, ngscount; // for performance reporting
} ObsCtx;

typedef struct {
  PetscInt  _level;  // 0 is coarsest level (is 0 in single-level usage)
  LDC       ldc;     // object which holds/manages constraints at each level;
                     // ldc.dal is DMDA for the level
  Vec       g,       // iterate on the level
            ell,     // right-hand side linear functional on level
            y,       // downward correction on level; NULL on level 0
            z;       // upward correction on level
} Level;

// z = gamma_lower(x,y) is the hemispherical obstacle, but made C^1 using
// "skirt" at r=r0; on square (-2,2)^2
PetscReal gamma_lower(PetscReal x, PetscReal y, void *ctx) {
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

/*  This exact solution solves a 1D radial free-boundary problem for the
Laplace equation, on r >= 0, with hemispherical obstacle
gamma_lower(r).  The Laplace equation applies where u(r) > gamma_lower(r),
    u''(r) + r^-1 u'(r) = 0
with boundary conditions including free b.c.s at an unknown location r = a:
    u(a) = gamma_lower(a),  u'(a) = gamma_lower'(a),  u(2) = 0
The solution is  u(r) = - A log(r) + B   on  r > a.  The boundary conditions
can then be reduced to a root-finding problem for a:
    a^2 (log(2) - log(a)) = 1 - a^2
The solution is a = 0.697965148223374 (giving residual 1.5e-15).  Then
A = a^2*(1-a^2)^(-0.5) and B = A*log(2) are given below in the code.  */
PetscReal uexact_obstacle(PetscReal x, PetscReal y, void *ctx) {
    const PetscReal afree = 0.697965148223374,
                    A     = 0.680259411891719,
                    B     = 0.471519893402112,
                    r = PetscSqrtReal(x * x + y * y);
    return (r <= afree) ? gamma_lower(x,y,ctx)  // active set; on the obstacle
                        : - A * PetscLogReal(r) + B; // solves laplace eqn
}

// exact solution to bratu problem with lambda = 1; on square (0,1)^2
static PetscReal uexact_liouville(PetscReal x, PetscReal y, void *ctx) {
    PetscReal r2 = (x + 1.0) * (x + 1.0) + (y + 1.0) * (y + 1.0),
              qq = r2 * r2 + 1.0,
              omega = r2 / (qq * qq);
    return PetscLogReal(32.0 * omega);
}

static PetscReal zero_fcn(PetscReal x, PetscReal y, void *ctx) {
    return 0.0;
}

// z = bump1_fcn(x,y) is zero along boundary of Omega=(0,1)^2 and reaches
// maximum of bump1_fcn(0.5,0.5) = 1.0 in center
PetscReal bump1_fcn(PetscReal x, PetscReal y, void *ctx) {
    return 16.0 * x * (1.0 - x) * y * (1.0 - y);
}

// z = bump4_fcn(x,y) is zero along boundary of Omega=(-2,2)^2 and reaches
// maximum of bump4_fcn(0,0) = 1.0 in center
PetscReal bump4_fcn(PetscReal x, PetscReal y, void *ctx) {
    return (x + 2.0) * (2.0 - x) * (y + 2.0) * (2.0 - y) / 16.0;
}

extern PetscErrorCode AssertBoundaryValuesAdmissible(DM, ObsCtx*);
extern PetscErrorCode MonitorCRNorm(DM, Vec, Vec, Vec, PetscInt, ObsCtx*);
extern PetscErrorCode MonitorRanges(DM, Vec, PetscInt, ObsCtx*);
extern PetscErrorCode ApplyOperatorF(DM, Vec, Vec, ObsCtx*);
extern PetscErrorCode ProjectedNGS(LDC*, PetscBool, Vec, Vec, Vec, ObsCtx*);

int main(int argc,char **argv) {
    ObsCtx         ctx;
    Level          *levs;
    Vec            gamlow, w, uexact, F, Fhat, gplusy, tmpfine, tmpcoarse;
    DMDALocalInfo  finfo;
    PetscInt       totlevs=2, cycles=1, csweeps=1, jtop, viter, j, k;
    PetscBool      bratu = PETSC_FALSE, counts = PETSC_FALSE,
                   ldcinfo = PETSC_FALSE, monitor = PETSC_FALSE,
                   monitorranges = PETSC_FALSE, monitorvcycles = PETSC_FALSE,
                   view = PETSC_FALSE, admis;
    PetscLogDouble lflops, flops;
    PetscReal      bumpsize=0.0, errinf;
    char           viewname[PETSC_MAX_PATH_LEN];

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    ctx.gamma_lower = &gamma_lower;
    ctx.f_rhs = &zero_fcn;
    ctx.g_bdry = &uexact_obstacle;
    ctx.maxits = 1;
    ctx.quadpts = 2;
    ctx.sweeps = 1;
    ctx.residualcount = 0;
    ctx.ngscount = 0;
    PetscOptionsBegin(PETSC_COMM_WORLD,"nm_","NMCD solver options","");
    PetscCall(PetscOptionsBool("-bratu","solve unconstrained Bratu equation problem (vs default linear obstacle problem)",
                            "nmcd.c",bratu,&bratu,NULL));
    PetscCall(PetscOptionsReal("-bumpsize","initialization from exact solution plus this much bump",
                            "nmcd.c",bumpsize,&bumpsize,NULL));
    PetscCall(PetscOptionsBool("-counts","print counts for calls to call-back functions",
                            "nmcd.c",counts,&counts,NULL));
    PetscCall(PetscOptionsInt("-csweeps","number of PNGS sweeps used for coarse solver",
                            "nmcd.c",csweeps,&csweeps,NULL));
    PetscCall(PetscOptionsInt("-cycles","maximum number of NMCD V-cycles",
                            "nmcd.c",cycles,&cycles,NULL));
    PetscCall(PetscOptionsBool("-ldcinfo","print info on LDC (MCD) actions",
                            "nmcd.c",ldcinfo,&ldcinfo,NULL));
    PetscCall(PetscOptionsInt("-levels","total number NMCD levels (>= 1)",
                            "nmcd.c",totlevs,&totlevs,NULL));
    PetscCall(PetscOptionsInt("-maxits","in PNGS, number of Newton iterations at each point",
                            "nmcd.c",ctx.maxits,&(ctx.maxits),NULL));
    PetscCall(PetscOptionsBool("-monitor","print CR residual norm for each V-cycle",
                            "nmcd.c",monitor,&monitor,NULL));
    PetscCall(PetscOptionsBool("-monitor_ranges","print iterate and (raw) residual ranges",
                            "nmcd.c",monitorranges,&monitorranges,NULL));
    PetscCall(PetscOptionsBool("-monitor_vcycles","print info on CR residual norm and V cycles",
                            "nmcd.c",monitorvcycles,&monitorvcycles,NULL));
    // WARNING: coarse problems are badly solved with -nm_quadpts 1
    PetscCall(PetscOptionsInt("-quadpts","number n of quadrature points (= 1,2,3 only)",
                            "nmcd.c",ctx.quadpts,&(ctx.quadpts),NULL));
    PetscCall(PetscOptionsInt("-sweeps","in PNGS, number of sweeps over the grid",
                            "nmcd.c",ctx.sweeps,&(ctx.sweeps),NULL));
    PetscCall(PetscOptionsString("-view","custom view of solution,residual,cr in ascii_matlab",
                            "nmcd",viewname,viewname,sizeof(viewname),&view));
    PetscOptionsEnd();

    // options consistency checking
    if (csweeps < 1) {
        SETERRQ(PETSC_COMM_SELF,1,"do at least 1 sweep over points at coarsest level in PNGS");
    }
    if (cycles < 1) {
        SETERRQ(PETSC_COMM_SELF,2,"do at least 1 V-cycle");
    }
    if (totlevs < 1) {
        SETERRQ(PETSC_COMM_SELF,3,"use at least 1 grid level");
    }
    if (ctx.maxits < 1) {
        SETERRQ(PETSC_COMM_SELF,4,"do at least 1 Newton iteration at each point in PNGS");
    }
    if (ctx.sweeps < 1) {
        SETERRQ(PETSC_COMM_SELF,5,"do at least 1 sweep over points in PNGS");
    }
    if (ctx.quadpts < 1 || ctx.quadpts > 3) {
        SETERRQ(PETSC_COMM_SELF,6,"quadrature points n=1,2,3 only");
    }

    if (bratu) {
        ctx.gamma_lower = NULL;
        ctx.g_bdry = &uexact_liouville;
    }

    // allocate Level stack
    PetscCall(PetscMalloc1(totlevs,&levs));

    // create DMDA for coarsest level: 3x3 grid on square Omega
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(levs[0].ldc.dal)));
    // next line allows  -da_grid_x mx -da_grid_y my  or  -da_refine L  to set
    // coarsest level
    PetscCall(DMSetFromOptions(levs[0].ldc.dal));
    PetscCall(DMSetUp(levs[0].ldc.dal));  // must be called BEFORE SetUniformCoordinates
    // note LDCRefine() duplicates bounding box for finer-level DMDA
    if (bratu)
        PetscCall(DMDASetUniformCoordinates(levs[0].ldc.dal,0.0,1.0,0.0,1.0,0.0,0.0));
    else
        PetscCall(DMDASetUniformCoordinates(levs[0].ldc.dal,-2.0,2.0,-2.0,2.0,0.0,0.0));

    // create Level stack by creating coarsest LDC using levs[0].dmda,
    // then refining jtop times; note:
    //   * each Level has an LDC
    //   * each level has allocated g,ell,y,z Vecs, except level 0 has no y
    for (j=0; j<totlevs; j++) {
        levs[j]._level = j;
        if (j == 0) {
            PetscCall(LDCCreateCoarsest(ldcinfo,&(levs[0].ldc)));
        } else {
            PetscCall(LDCRefine(&(levs[j-1].ldc),&(levs[j].ldc)));
        }
        PetscCall(DMSetApplicationContext(levs[j].ldc.dal,&ctx));
        PetscCall(DMGetGlobalVector(levs[j].ldc.dal,&(levs[j].g)));
        PetscCall(DMGetGlobalVector(levs[j].ldc.dal,&(levs[j].ell)));
        PetscCall(DMGetGlobalVector(levs[j].ldc.dal,&(levs[j].z)));
        if (j == 0)
            levs[j].y = NULL;
        else
            PetscCall(DMGetGlobalVector(levs[j].ldc.dal,&(levs[j].y)));
    }
    jtop = totlevs - 1;

    // check admissibility of the finest-level boundary condition
    PetscCall(AssertBoundaryValuesAdmissible(levs[jtop].ldc.dal,&ctx));

    // generate finest-level obstacle gamlow as Vec
    if (bratu)
        gamlow = NULL;
    else {
        PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&gamlow));
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,gamma_lower,gamlow,&ctx));
    }

    // create finest-level initial iterate w^J:  w^J = uexact + bumpsize * bump
    PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&w));
    PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&tmpfine));
    if (bratu) {
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,uexact_liouville,w,&ctx));
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,bump1_fcn,tmpfine,&ctx));
    } else {
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,uexact_obstacle,w,&ctx));
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,bump4_fcn,tmpfine,&ctx));
    }
    PetscCall(VecAXPY(w,bumpsize,tmpfine));  // w <- bumpsize * tmpfine + w
    PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&tmpfine));
    // PetscCall(VecSet(w,0.0));  // not really an alternative because we
                                  // require w^J admissible to start

    // check admissibility of w^J
    PetscCall(VecLessThanOrEqual(levs[jtop].ldc.dal,gamlow,w,&admis));
    if (!admis) {
        SETERRQ(PETSC_COMM_SELF,3,"initial iterate is not admissible\n");
    }

    // report ranges for initial w and corresponding residual f^J(w)
    if (monitorranges)
        PetscCall(MonitorRanges(levs[jtop].ldc.dal,w,0,&ctx));

    // one NMCD V-cycle with one smoother iteration at each level
    if (totlevs == 1)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "single-level solving using %d smoother iterations ...\n",cycles));
    else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "solving using %d V-cycles with %d levels ...\n",cycles,totlevs));
    PetscCall(VecSet(levs[jtop].ell,0.0)); // assumes f(x,y)=0 on RHS
    for (viter = 0; viter < cycles; viter++) {
        // report norm of CR residual
        if (monitor)
            PetscCall(MonitorCRNorm(levs[jtop].ldc.dal,NULL,gamlow,w,viter,&ctx));
        // set-up finest-level chiupp,chilow from initial iterate w
        PetscCall(LDCSetFinestUpDCs(w,NULL,gamlow,&(levs[jtop].ldc)));
        // initialize iterate at top level
        PetscCall(VecCopy(w,levs[jtop].g));
        // downward direction
        for (j = jtop; j >= 1; j--) {
            // compute LDCs:  chiupp,chilow for levs[j-1]  by mono restrict
            //                phiupp,philow for levs[j]    by subtraction
            PetscCall(LDCSetLevel(&(levs[j].ldc)));
            // get allocated temporaries
            PetscCall(DMGetGlobalVector(levs[j].ldc.dal,&gplusy));
            PetscCall(DMGetGlobalVector(levs[j].ldc.dal,&tmpfine));
            PetscCall(DMGetGlobalVector(levs[j-1].ldc.dal,&tmpcoarse));
            // smooth in D^j to compute y^j
            PetscCall(VecSet(levs[j].y,0.0));
            PetscCall(ProjectedNGS(&(levs[j].ldc),PETSC_FALSE,levs[j].ell,
                                   levs[j].g,levs[j].y,&ctx));
            if (monitorvcycles)
                PetscCall(UpdateIndentPrintRange(levs[j].y,"y",jtop,j));
            // compute g^j-1 using injection:
            //   g^j-1 = R^dot(g^j + y^j)
            PetscCall(VecWAXPY(gplusy,1.0,levs[j].g,levs[j].y));
            PetscCall(Q1Inject(levs[j].ldc.dal,levs[j-1].ldc.dal,gplusy,&(levs[j-1].g)));
            // construct ell^j-1:
            //   ell^j-1 = f^j-1(g^j-1) + R (ell^j - f^j(g^j + y^j))
            PetscCall(ApplyOperatorF(levs[j-1].ldc.dal,levs[j-1].g,levs[j-1].ell,&ctx));
            PetscCall(ApplyOperatorF(levs[j].ldc.dal,gplusy,tmpfine,&ctx));
            PetscCall(VecAYPX(tmpfine,-1.0,levs[j].ell));
            PetscCall(Q1Restrict(levs[j].ldc.dal,levs[j-1].ldc.dal,tmpfine,&tmpcoarse));
            PetscCall(VecAXPY(levs[j-1].ell,1.0,tmpcoarse));
            // restore temporaries
            PetscCall(DMRestoreGlobalVector(levs[j].ldc.dal,&gplusy));
            PetscCall(DMRestoreGlobalVector(levs[j].ldc.dal,&tmpfine));
            PetscCall(DMRestoreGlobalVector(levs[j-1].ldc.dal,&tmpcoarse));
        }
        // coarse solve in U^0 to compute z^0
        if (levs[0].ldc._printinfo) {
            PetscCall(LDCCheckDCRanges(levs[0].ldc));
        }
        PetscCall(VecSet(levs[0].z,0.0));
        for (k = 0; k < csweeps; k++) {
            PetscCall(ProjectedNGS(&(levs[0].ldc),PETSC_TRUE,levs[0].ell,
                                   levs[0].g,levs[0].z,&ctx));
        }
        if (monitorvcycles)
            PetscCall(UpdateIndentPrintRange(levs[0].z,"z",jtop,0));
        // upward direction
        for (j = 1; j <= jtop; j++) {
            // compute z^j using prolongation:
            //   z^j = P z^j-1 + y^j
            PetscCall(Q1Interpolate(levs[j-1].ldc.dal,levs[j].ldc.dal,levs[j-1].z,&(levs[j].z)));
            PetscCall(VecAXPY(levs[j].z,1.0,levs[j].y));
            // smooth in U^j to compute z^j
            PetscCall(ProjectedNGS(&(levs[j].ldc),PETSC_TRUE,levs[j].ell,
                                   levs[j].g,levs[j].z,&ctx));
            if (monitorvcycles)
                PetscCall(UpdateIndentPrintRange(levs[j].z,"z",jtop,j));
        }
        // update fine-level iterate:
        //   w <- w + z^J
        PetscCall(VecAXPY(w,1.0,levs[jtop].z));

        // report range on current w and f^J(w)
        if (monitorranges)
            PetscCall(MonitorRanges(levs[jtop].ldc.dal,w,viter+1,&ctx));
    } // for viter ...
    // report final norm of CR residual
    if (monitor)
        PetscCall(MonitorCRNorm(levs[jtop].ldc.dal,NULL,gamlow,w,viter,&ctx));

    if (counts) {
        // note calls to ApplyOperatorF() and ProjectedNGS() are
        // collective but flops are per-process, so we need a reduction
        PetscCall(PetscGetFlops(&lflops));
        PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,
                                PETSC_COMM_WORLD));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "flops = %.3e,  residual calls = %d,  PNGS calls = %d\n",
                              flops,ctx.residualcount,ctx.ngscount));
    }

    if (view) {
        PetscViewer  file;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "viewing iterate, exact solution, residual, CR residual into ascii_matlab file %s\n",viewname));
        PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&file));
        PetscCall(PetscViewerSetType(file,PETSCVIEWERASCII));
        PetscCall(PetscViewerFileSetMode(file,FILE_MODE_WRITE));
        PetscCall(PetscViewerFileSetName(file,viewname));
        PetscCall(PetscViewerPushFormat(file,PETSC_VIEWER_ASCII_MATLAB));
        PetscCall(PetscObjectSetName((PetscObject)w,"w"));
        PetscCall(VecView(w,file));
        PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&uexact));
        if (bratu)
            PetscCall(VecFromFormula(levs[jtop].ldc.dal,uexact_liouville,uexact,&ctx));
        else
            PetscCall(VecFromFormula(levs[jtop].ldc.dal,uexact_obstacle,uexact,&ctx));
        PetscCall(PetscObjectSetName((PetscObject)uexact,"uexact"));
        PetscCall(VecView(uexact,file));
        PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&F));
        PetscCall(ApplyOperatorF(levs[jtop].ldc.dal,w,F,&ctx));
        PetscCall(PetscObjectSetName((PetscObject)F,"F"));
        PetscCall(VecView(F,file));
        PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&Fhat));
        PetscCall(CRFromResidual(levs[jtop].ldc.dal,NULL,gamlow,w,F,Fhat));
        PetscCall(PetscObjectSetName((PetscObject)F,"Fhat"));
        PetscCall(VecView(Fhat,file));
        PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&uexact));
        PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&F));
        PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&Fhat));
        PetscCall(PetscViewerPopFormat(file));
        PetscCall(PetscViewerDestroy(&file));
    }

    // report on numerical error
    PetscCall(DMGetGlobalVector(levs[jtop].ldc.dal,&uexact));
    if (bratu)
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,uexact_liouville,uexact,&ctx));
    else
        PetscCall(VecFromFormula(levs[jtop].ldc.dal,uexact_obstacle,uexact,&ctx));
    PetscCall(VecAXPY(w,-1.0,uexact));    // u <- u + (-1.0) uexact
    PetscCall(VecNorm(w,NORM_INFINITY,&errinf));
    PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&uexact));
    PetscCall(DMDAGetLocalInfo(levs[jtop].ldc.dal,&finfo));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                          finfo.mx,finfo.my,errinf));

    // restore or destroy it all
    if (gamlow)
        PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&gamlow));
    PetscCall(DMRestoreGlobalVector(levs[jtop].ldc.dal,&w));
    for (j=0; j<totlevs; j++) {
        PetscCall(DMRestoreGlobalVector(levs[j].ldc.dal,&(levs[j].g)));
        PetscCall(DMRestoreGlobalVector(levs[j].ldc.dal,&(levs[j].ell)));
        PetscCall(DMRestoreGlobalVector(levs[j].ldc.dal,&(levs[j].z)));
        if (j > 0)
            PetscCall(DMRestoreGlobalVector(levs[j].ldc.dal,&(levs[j].y)));
        PetscCall(LDCDestroy(&(levs[j].ldc))); // destroys levs[j].dmda
    }
    PetscCall(PetscFree(levs));
    PetscCall(PetscFinalize());
    return 0;
}

// check that g_bdry(x,y) is above gamma_lower(x,y); error if not
PetscErrorCode AssertBoundaryValuesAdmissible(DM da, ObsCtx* user) {
    DMDALocalInfo  info;
    PetscInt       i, j;
    PetscReal      hx, hy, x, y;
    if (user->gamma_lower == NULL)  // so gamma_lower=-infty
        return 0;
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1);
    hy = 4.0 / (PetscReal)(info.my - 1);
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = -2.0 + j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = -2.0 + i * hx;
            if (user->g_bdry(x,y,user) < user->gamma_lower(x,y,user)) {
                PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                    "ERROR: g_bdry(x,y) < gamma_lower(x,y) at x=%.6e,y=%.6e\n",
                    x,y));
                SETERRQ(PETSC_COMM_SELF,1,
                    "assertion fails (boundary values are NOT above obstacle)");
            }
        }
    }
    return 0;
}

PetscErrorCode MonitorCRNorm(DM da, Vec gamupp, Vec gamlow, Vec w,
                             PetscInt iter, ObsCtx *ctx) {
    Vec        F, Fhat;
    PetscReal  Fnorm;
    PetscCall(DMGetGlobalVector(da,&F));
    PetscCall(DMGetGlobalVector(da,&Fhat));
    PetscCall(ApplyOperatorF(da,w,F,ctx));
    PetscCall(CRFromResidual(da,NULL,gamlow,w,F,Fhat));
    PetscCall(VecNorm(Fhat,NORM_2,&Fnorm));
    PetscCall(DMRestoreGlobalVector(da,&F));
    PetscCall(DMRestoreGlobalVector(da,&Fhat));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  %d CR norm %14.12e\n",iter,Fnorm));
    return 0;
}

PetscErrorCode MonitorRanges(DM da, Vec w, PetscInt iter, ObsCtx *ctx) {
    Vec F;
    if (iter == 0)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  initial ranges:\n"));
    else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  iterate %2d ranges:\n",iter));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    "));
    PetscCall(VecPrintRange(w,"w","",PETSC_TRUE));
    PetscCall(DMGetGlobalVector(da,&F));
    PetscCall(ApplyOperatorF(da,w,F,ctx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"    "));
    PetscCall(VecPrintRange(F,"f^J(w)","",PETSC_TRUE));
    PetscCall(DMRestoreGlobalVector(da,&F));
    return 0;
}

PetscBool _NodeOnBdry(DMDALocalInfo info, PetscInt i, PetscInt j) {
    return (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1);
}

// FLOPS: 2 + (16 + 5 + 7) = 30
PetscReal _IntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                        const PetscReal uu[4], const PetscReal ff[4],
                        PetscInt r, PetscInt s, ObsCtx *user) {
    const Q1GradRef  du    = Q1DEval(uu,r,s),
                     dchiL = Q1dchi[L][r][s];
    return Q1GradInnerProd(du,dchiL)
           - Q1Eval(ff,r,s) * Q1chi[L][r][s];
}

// compute F_ij, the discretized nonlinear operator:
//     F(u)[v] = int_Omega grad u . grad v - f v
// giving the vector of nodal values (residuals w/o ell)
//     F_ij = F(u)[psi_ij]
// where i,j is a node and psi_ij is the hat function
// at boundary nodes we have
//     F_ij = u_ij - g(x_i,y_j)
PetscErrorCode ApplyOperatorF(DM da, Vec u, Vec F, ObsCtx *user) {
    DMDALocalInfo    info;
    Vec              uloc;
    const Q1Quad1D   q = Q1gausslegendre[user->quadpts-1];
    const PetscInt   li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
    PetscInt         i, j, l, PP, QQ, r, s;
    const PetscReal  **au;
    PetscReal        hx, hy, detj, **FF, x, y, uu[4], ff[4];

    // grid info and arrays
    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 4.0 / (PetscReal)(info.mx - 1),
    hy = 4.0 / (PetscReal)(info.my - 1),
    detj = 0.25 * hx * hy;
    PetscCall(DMGetLocalVector(da,&uloc));
    PetscCall(DMGlobalToLocal(da,u,INSERT_VALUES,uloc));
    PetscCall(DMDAVecGetArrayRead(da, uloc, &au));
    PetscCall(DMDAVecGetArray(da, F, &FF));

    // set up Q1 FEM tools for this grid
    PetscCall(Q1Setup(user->quadpts,da,-2.0,2.0,-2.0,2.0));

    // clear residuals (because we sum over elements)
    // or assign F for Dirichlet nodes
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = -2.0 + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = -2.0 + i * hx;
            FF[j][i] = (_NodeOnBdry(info,i,j)) ? au[j][i] - user->g_bdry(x,y,user)
                                               : 0.0;
        }
    }

    // Sum over elements, with upper-right node indices i,j, to compute F for
    // interior nodes.  Note we own elements down or left of owned nodes, but
    // in parallel the integral also needs to include non-owned (halo) elements
    // up or right of owned nodes.  See diagram in Chapter 9 of Bueler (2021)
    // for indexing and element ownership.
    for (j = info.ys; j <= info.ys + info.ym; j++) {
        if ((j == 0) || (j > info.my-1))  // does element actually exist?
            continue;
        y = -2.0 + j * hy;
        for (i = info.xs; i <= info.xs + info.xm; i++) {
            if ((i == 0) || (i > info.mx-1))  // does element actually exist?
                continue;
            x = -2.0 + i * hx;
            // this element, down or left of node i,j, is adjacent to an owned
            // and interior node; so get values of rhs f at corners of element
            ff[0] = user->f_rhs(x,y,user);
            ff[1] = user->f_rhs(x-hx,y,user);
            ff[2] = user->f_rhs(x-hx,y-hy,user);
            ff[3] = user->f_rhs(x,y-hy,user);
            // get values of iterate u at corners of element, using Dirichlet
            // value if known (to generate symmetric Jacobian)
            uu[0] = _NodeOnBdry(info,i,  j)
                    ? user->g_bdry(x,y,user)       : au[j][i];
            uu[1] = _NodeOnBdry(info,i-1,j)
                    ? user->g_bdry(x-hx,y,user)    : au[j][i-1];
            uu[2] = _NodeOnBdry(info,i-1,j-1)
                    ? user->g_bdry(x-hx,y-hy,user) : au[j-1][i-1];
            uu[3] = _NodeOnBdry(info,i,  j-1)
                    ? user->g_bdry(x,y-hy,user)    : au[j-1][i];
            // loop over corners of element i,j; l is local (elementwise)
            // index of the corner and PP,QQ are global indices of same corner
            for (l = 0; l < 4; l++) {
                PP = i + li[l];
                QQ = j + lj[l];
                // only update residual if we own node and it is not boundary
                if (PP >= info.xs && PP < info.xs + info.xm
                    && QQ >= info.ys && QQ < info.ys + info.ym
                    && !_NodeOnBdry(info,PP,QQ)) {
                    // loop over quadrature points to contribute to residual
                    // for this l corner of this i,j element
                    for (r = 0; r < q.n; r++) {
                        for (s = 0; s < q.n; s++) {
                            FF[QQ][PP] += detj * q.w[r] * q.w[s]
                                          * _IntegrandRef(hx,hy,l,uu,ff,r,s,user);
                        }
                    }
                }
            }
        }
    }

    PetscCall(DMDAVecRestoreArrayRead(da, uloc, &au));
    PetscCall(DMDAVecRestoreArray(da, F, &FF));
    PetscCall(DMRestoreLocalVector(da,&uloc));

    // FLOPS: only count flops per quadrature point in residual computations:
    //   4 + 30 = 34
    // note q.n^2 quadrature points per element
    PetscCall(PetscLogFlops(34.0 * q.n * q.n * info.xm * info.ym));
    (user->residualcount)++;
    return 0;
}

// FIXME want complementarity residual too?

// for owned, interior nodes i,j, we define the pointwise residual corresponding
// to the hat function psi_ij:
//     rho(c) = F(u + c psi_ij)[psi_ij] - b_ij
//            = int_Omega (grad u + c grad psi_ij) . grad psi_ij - f psi_ij
//              - b_ij
// note b_ij is outside the integral; also note
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij
// note the integral is over four elements, each with four quadrature
// points "+" (in the default -lb_quadpts 2 case), which we traverse in the
// order 0,1,2,3 given:
//     j+1  *-------*-------*
//          | +   + | +   + |
//          |   1   |   0   |
//          | +   + | +   + |
//     j    *-------*-------*
//          | +   + | +   + |
//          |   2   |   3   |
//          | +   + | +   + |
//     j-1  *-------*-------*
//         i-1      i      i+1
// and each of the four elements has nodes with local (reference element)
// indices:
//        1 *-------* 0
//          |       |
//          |       |
//          |       |
//        2 *-------* 3

// evaluate integrand of rho(c) at a point xi,eta in the reference element,
// for the hat function at corner L (i.e. chi_L = psi_ij from caller)
// FLOPS: 2 + (16 + 5 + 4 + 7 + 5) = 39
PetscErrorCode _rhoIntegrandRef(PetscReal hx, PetscReal hy, PetscInt L,
                 PetscReal c, const PetscReal uu[4], const PetscReal ff[4],
                 PetscInt r, PetscInt s,
                 PetscReal *rho, PetscReal *drhodc, ObsCtx *user) {
    const Q1GradRef du    = Q1DEval(uu,r,s),
                    dchiL = Q1dchi[L][r][s];
    if (rho)
        *rho = Q1GradInnerProd(Q1GradAXPY(c,dchiL,du),dchiL)
               - Q1Eval(ff,r,s) * Q1chi[L][r][s];
    if (drhodc)
        *drhodc = Q1GradInnerProd(dchiL,dchiL);
    return 0;
}

// for owned, interior node i,j, evaluate rho(c) and rho'(c)
PetscErrorCode _rhoFcn(DMDALocalInfo *info, PetscInt i, PetscInt j,
                       PetscReal c, const PetscReal **ag, PetscReal **az,
                       PetscReal *rho, PetscReal *drhodc, ObsCtx *user) {
    // i+oi[k],j+oj[k] gives index of upper-right node of element k
    // ll[k] gives local (ref. element) index of the i,j node on the k element
    const PetscInt  oi[4] = {1, 0, 0, 1},
                    oj[4] = {1, 1, 0, 0},
                    ll[4] = {2, 3, 0, 1};
    const PetscReal hx = 4.0 / (PetscReal)(info->mx - 1),
                    hy = 4.0 / (PetscReal)(info->my - 1),
                    detj = 0.25 * hx * hy;
    const Q1Quad1D  q = Q1gausslegendre[user->quadpts-1];
    PetscInt  k, ii, jj, r, s;
    PetscReal x, y, ff[4], uu[4], prho, pdrhodc, tmp;

    *rho = 0.0;
    if (drhodc)
        *drhodc = 0.0;
    // loop around 4 elements adjacent to global index node i,j
    for (k=0; k < 4; k++) {
        // global index of this element
        ii = i + oi[k];
        jj = j + oj[k];
        // field values for f and u = g + z on this element
        x = -2.0 + ii * hx;
        y = -2.0 + jj * hy;
        ff[0] = user->f_rhs(x,y,user);
        ff[1] = user->f_rhs(x-hx,y,user);
        ff[2] = user->f_rhs(x-hx,y-hy,user);
        ff[3] = user->f_rhs(x,y-hy,user);
        uu[0] = _NodeOnBdry(*info,ii,  jj)   ?
                user->g_bdry(x,y,user)       : ag[jj][ii]     + az[jj][ii];
        uu[1] = _NodeOnBdry(*info,ii-1,jj)   ?
                user->g_bdry(x-hx,y,user)    : ag[jj][ii-1]   + az[jj][ii-1];
        uu[2] = _NodeOnBdry(*info,ii-1,jj-1) ?
                user->g_bdry(x-hx,y-hy,user) : ag[jj-1][ii-1] + az[jj-1][ii-1];
        uu[3] = _NodeOnBdry(*info,ii,  jj-1) ?
                user->g_bdry(x,y-hy,user)    : ag[jj-1][ii]   + az[jj-1][ii];
        // loop over quadrature points in this element, summing to get rho
        for (r = 0; r < q.n; r++) {
            for (s = 0; s < q.n; s++) {
                // ll[k] is local (elementwise) index of the corner (= i,j)
                _rhoIntegrandRef(hx,hy,ll[k],c,uu,ff,r,s,&prho,&pdrhodc,user);
                tmp = detj * q.w[r] * q.w[s];
                *rho += tmp * prho;
                if (drhodc)
                    *drhodc += tmp * pdrhodc;
            }
        }
    }
    // FLOPS per quadrature point in the four elements: 6 + 39 = 45
    PetscCall(PetscLogFlops(45.0 * q.n * q.n * 4.0));
    return 0;
}

// do projected nonlinear Gauss-Seidel (processor-block) sweeps on equation
//     F(g + z)[psi_ij] = ell_ij   for all nodes i,j
// in defect form, i.e. for z, with g fixed, and subject to bi-obstacle constraints
//     lower <= z <= upper
// note psi_ij is the hat function and
//     F(u)[v] = int_Omega grad u . grad v - f v,
// and ell is a nodal field (linear functional)
// for each interior node i,j we define
//     rho(c) = F(g + z + c psi_ij)[psi_ij] - ell_ij
// and do Newton iterations
//     c <-- c - rho(c) / rho'(c)
// followed by the bi-obstacle projection,
//     c <-- min{c, upper_ij - z_ij}
//     c <-- max{c, lower_ij - z_ij}
// note that since F() is linear, in fact:
//     rho'(c) = int_Omega grad psi_ij . grad psi_ij
// sets boundary values: z=0
PetscErrorCode ProjectedNGS(LDC *ldc, PetscBool updcs, Vec ell, Vec g, Vec z,
                            ObsCtx *user) {
    DMDALocalInfo   info;
    Vec             gloc, zloc;
    PetscBool       haveupper = PETSC_FALSE, havelower = PETSC_FALSE;
    PetscInt        i, j, k, totalits=0, l;
    const PetscReal **aell, **ag, **aupper, **alower;
    PetscReal       c, rho, drhodc, **az;

    PetscCall(Q1Setup(user->quadpts,ldc->dal,-2.0,2.0,-2.0,2.0)); // set up quadrature
    PetscCall(DMDAGetLocalInfo(ldc->dal,&info));

    // need local vector for stencil width of g and z in parallel
    PetscCall(DMGetLocalVector(ldc->dal,&gloc));
    PetscCall(DMGetLocalVector(ldc->dal,&zloc));
    PetscCall(DMGlobalToLocal(ldc->dal,g,INSERT_VALUES,gloc)); // ghosts in g

    // set-up for bi-obstacles
    if (updcs) {
        if (ldc->chiupp) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->chiupp,&aupper));
            haveupper = PETSC_TRUE;
        }
        if (ldc->chilow) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->chilow,&alower));
            havelower = PETSC_TRUE;
        }
    } else {
        if (ldc->phiupp) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->phiupp,&aupper));
            haveupper = PETSC_TRUE;
        }
        if (ldc->philow) {
            PetscCall(DMDAVecGetArrayRead(ldc->dal,ldc->philow,&alower));
            havelower = PETSC_TRUE;
        }
    }

    // NGS sweeps over interior nodes
    if (ell)
        PetscCall(DMDAVecGetArrayRead(ldc->dal,ell,&aell));
    PetscCall(DMDAVecGetArrayRead(ldc->dal,gloc,&ag));
    for (l=0; l<user->sweeps; l++) {
        // update ghosts in z
        PetscCall(DMGlobalToLocal(ldc->dal,z,INSERT_VALUES,zloc));
        PetscCall(DMDAVecGetArray(ldc->dal,zloc,&az));
        for (j = info.ys; j < info.ys + info.ym; j++) {
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (_NodeOnBdry(info,i,j)) {
                    az[j][i] = 0.0;
                    continue;
                }
                // i,j is owned interior node; do projected Newton iterations
                c = 0.0;
                for (k = 0; k < user->maxits; k++) {
                    // evaluate rho(c) and rho'(c) for current c
                    PetscCall(_rhoFcn(&info,i,j,c,ag,az,&rho,&drhodc,user));
                    if (ell)
                        rho -= aell[j][i];
                    if (drhodc == 0.0)
                        SETERRQ(PETSC_COMM_SELF,1,"drhodc == 0 at i,j=%d,%d",i,j);
                    c = c - rho / drhodc;  // Newton step
                    // bi-obstacle projection
                    if (haveupper)
                        c = PetscMin(c, aupper[j][i] - az[j][i]);
                    if (havelower)
                        c = PetscMax(c, alower[j][i] - az[j][i]);
                    totalits++;
                }
                az[j][i] += c;
            }
        }
        PetscCall(DMDAVecRestoreArray(ldc->dal,zloc,&az));
        PetscCall(DMLocalToGlobal(ldc->dal,zloc,INSERT_VALUES,z)); // z has changed
    }
    PetscCall(DMDAVecRestoreArrayRead(ldc->dal,gloc,&ag));
    if (ell)
        PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ell,&aell));

    PetscCall(DMRestoreLocalVector(ldc->dal,&gloc));
    PetscCall(DMRestoreLocalVector(ldc->dal,&zloc));
    if (updcs) {
        if (ldc->chiupp)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->chiupp,&aupper));
        if (ldc->chilow)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->chilow,&alower));
    } else {
        if (ldc->phiupp)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->phiupp,&aupper));
        if (ldc->philow)
            PetscCall(DMDAVecRestoreArrayRead(ldc->dal,ldc->philow,&alower));
    }

    // add flops for Newton iteration arithmetic; note rhoFcn() already counts flops
    PetscCall(PetscLogFlops(8 * totalits));
    (user->ngscount)++;
    return 0;
}
