"""Solve poisson equation  - triangle u = f  by FAS multigrid.

Poisson equation problem (weak form) is
    F(u)[v] = ell[v]
where v is a test function and
    F(u)[v] = dot(grad(u), grad(v)) * dx
    ell[v] = f * v * dx
and boundary conditions are homogeneous Dirichlet.

Solver is FAS V-cycles.  Going downward the V-cycle uses solve() for a
down-smoother, forms the FAS correction equation, and decrements the level.
The coarsest problem is solved by LU.  Going upward the V-cycle prolongs and
adds the computed correction and then uses solve() as an up-smoother.

The smoother is a KSP-only solve, with two CG+ICC iterations.

The FAS correction equation is
    Fc(uc)[vc] = ellc[vc]
where Fc has same formula as F and
    ellc[vc] = R'(resf)[vc] + Fc(R(uf))[vc]
and
    resf[v] = Ff(uf)[v] - ellf[v]
and R' is canonical restriction and R is injection.  After solution the fine
level iterate is updated
    u <- u + P(uc - R(uf))
where P is prolongation.
"""

from firedrake import *

levels = 4
vcycles = 1
smootherparams = {"snes_type": "ksponly",
                  "snes_max_linear_solve_fail": 2, # don't error when KSP reports DIVERGED_ITS
                  "ksp_converged_reason": None,
                  "ksp_type": "cg",
                  #"ksp_rtol": 0.01,
                  "ksp_max_it": 2,
                  "pc_type": "icc"}
                  #"ksp_type": "preonly",
                  #"pc_type": "lu"}
downparams = smootherparams.copy()
#downparams["pc_sor_forward"] = None
upparams = smootherparams.copy()
#upparams["pc_sor_backward"] = None
coarseparams = {"snes_type": "ksponly",
                #"ksp_converged_reason": None,
                "ksp_type": "preonly",
                "pc_type": "lu"}

# set up mesh hierarchy
cmesh = UnitSquareMesh(2, 2)  # as small as practical for nontriviality
hierarchy = MeshHierarchy(cmesh, levels-1)

# set up function spaces and some functions on each level
VV = [FunctionSpace(hierarchy[i], "CG", 1) for i in range(levels)]
u = [Function(VV[i]) for i in range(levels)] # values initialized to zero
v = [TestFunction(VV[i]) for i in range(levels)]
F = [dot(grad(u[i]), grad(v[i])) * dx for i in range(levels)]
bcs = [DirichletBC(VV[i], zero(), (1, 2, 3, 4)) for i in range(levels)]

# need to compute the problem form for the restriction of u
Ru = [Function(VV[i]) for i in range(levels)]
FRu = [dot(grad(Ru[i]), grad(v[i])) * dx for i in range(levels)]

# blank lists for functions to be created on descent by FAS formulas
ell = [None for i in range(levels)]
res = ell.copy()

# rhs and exact solution on fine level
fine = levels-1
x, y = SpatialCoordinate(hierarchy[fine])
f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

# evaluate error using function-space norms, L2 or Linf
useL2 = True
def error(u):
    udiff = Function(VV[fine]).interpolate(u - exact)
    if useL2:
        return sqrt(assemble(dot(udiff, udiff) * dx))
    else:
        with udiff.dat.vec_ro as vudiff:
            return abs(vudiff).max()[1]

# fine-mesh problem
ell[fine] = f * v[fine] * dx
res[fine] = F[fine] - ell[fine]

print('initial             |residual|=%.6f  |error|=%.6f' \
      % (norm(assemble(res[fine])), error(u[fine])))

for j in range(vcycles):
    # down-smoothing and creation of FAS correction equation
    for i in range(levels-1,0,-1):   # levels-1, ..., 1
        # smoother application
        solve(res[i] == 0, u[i], bcs=bcs[i], options_prefix = 'down%d' % i,
              solver_parameters=downparams)
        # construct FAS correction problem with right-hand side
        #     ell[i-1] = R'(res(u[i]))) + F[i-1](R(u[i]))
        # where R' is canonical restriction and R is injection
        Rres = Function(VV[i-1])
        restrict(assemble(res[i]), Rres) # assemble() needed because
                                         # res[i] is UFL form only
        inject(u[i], Ru[i-1])
        ell[i-1] = Rres * v[i-1] * dx + dot(grad(Ru[i-1]), grad(v[i-1])) * dx
        res[i-1] = F[i-1] - ell[i-1]
        inject(u[i], u[i-1])  # for initial value on next level; since u[i-1]=Ru[i-1],
                              # equivalent to setting initial error iterate to zero

    # coarse-mesh problem and direct solution
    solve(res[0] == 0, u[0], bcs=bcs[0], options_prefix = 'coarse',
          solver_parameters=coarseparams)
    print('%s|coarse cor|=%.6f' % ((levels-1)*'  ', norm(u[0])))

    # prolongation and up-smoothing
    for i in range(1,levels):   # 1, ..., levels-1
        Pcor = Function(VV[i])
        prolong(assemble(u[i-1] - Ru[i-1]), Pcor)  # assemble() needed because
                                                   # subtraction is UFL
        u[i] += Pcor
        solve(res[i] == 0, u[i], bcs=bcs[i], options_prefix = 'up%d' % i,
              solver_parameters=upparams)

    # report on result of cycle
    print('FAS V-cycle %d (%d levels) |residual|=%.6f  |error|=%.6f' \
          % (j+1, levels, norm(assemble(res[fine])), error(u[fine])))

u[fine].rename("u")
uexact = Function(VV[fine]).interpolate(exact)
uexact.rename("uexact")
udiff = Function(VV[fine]).interpolate(u[fine] - exact)
udiff.rename("udiff")
File("u-poisson.pvd").write(u[fine],uexact,udiff)
