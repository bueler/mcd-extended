"""Solve Bratu equation in 1D by FAS multigrid:
   -u'' - lambda e^u = f(x)
with Dirichlet conditions u(0)=u(1)=0 and
   f(x) = 9 pi^2 sin(3 pi x) − lambda exp(sin(3 pi x)).
The exact solution of this problem is u(x) = sin(3 pi x).

The goal is to solve the same problem as in
    Bueler (2021) "The full approximation storage multigrid scheme:
    A 1D finite element example", arXiv:2101.05408.
Specifically to start, compare to:
    ./fas1.py -K 1 -mms -niters 1 -cyclemax 1 -show

Weak form is
    F(u)[v] = ell[v]
where v is a test function and
    F(u)[v] = ( dot(grad(u), grad(v)) - lambda * exp(u) * v ) * dx
    ell[v] = f * v * dx

Solver is FAS V-cycles.  Going downward the V-cycle uses solve() for a
down-smoother, forms the FAS correction equation, and decrements the level.
The coarsest problem is solved by LU.  Going upward the V-cycle prolongs and
adds the computed correction and then uses solve() as an up-smoother.  The
smoother is a fixed number of Newton iterations with a fixed number of GS sweeps.
(FIXME: backward GS on up?)

The FAS correction equation is
    Fc(uc)[vc] = ellc[vc]
where Fc has same formula as F and
    ellc[vc] = R'(resf)[vc] + Fc(R(uf))[vc]
and
    resf[v] = ellf[v] - Ff(uf)[v]
and R' is canonical restriction and R is injection.  After solution the fine
level iterate is updated
    u <- u + P(uc - R(uf))
where P is prolongation.
"""

from firedrake import *

levels = 2
vcycles = 1
lam = 1.0
smootherparams = {"snes_rtol": 1.0,  # always succeed after one newton step
                  #"snes_view": None,
                  "snes_max_linear_solve_fail": 200, # don't error when KSP reports DIVERGED_ITS
                  "snes_converged_reason": None,
                  "ksp_converged_reason": None,
                  #"ksp_monitor": None,
                  "ksp_type": "richardson",
                  "ksp_max_it": 1,
                  "pc_type": "sor",
                  "pc_sor_forward": None}
coarseparams = {"snes_converged_reason": None,
                "ksp_converged_reason": None,
                "ksp_type": "preonly",
                "pc_type": "lu"}

# set up mesh hierarchy
cmesh = UnitIntervalMesh(2)
hierarchy = MeshHierarchy(cmesh, levels-1)

# set up function spaces and problem on each level
VV = [FunctionSpace(hierarchy[i], "CG", 1) for i in range(levels)]
u = [Function(VV[i]) for i in range(levels)] # values initialized to zero
v = [TestFunction(VV[i]) for i in range(levels)]
F = [( dot(grad(u[i]), grad(v[i])) - lam * exp(u[i]) * v[i] ) * dx \
     for i in range(levels)]
bcs = [DirichletBC(VV[i], zero(), (1, 2)) for i in range(levels)]

# need to compute the problem form for the restriction of u
Ru = [Function(VV[i]) for i in range(levels)]
FRu = [( dot(grad(Ru[i]), grad(v[i])) - lam * exp(Ru[i]) * v[i] ) * dx \
       for i in range(levels)]

# blank lists for functions to be created on descent by FAS formulas
ell = [None for i in range(levels)]
res = ell.copy()

# rhs and exact solution on fine level
fine = levels-1
xx = SpatialCoordinate(hierarchy[fine])  # with one output arg, returns list
x = xx[0]
f = 9.0 * pi*pi * sin(3.0*pi*x) - lam * exp(sin(3.0*pi*x))
exact = sin(3.0*pi*x)

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

# do FAS V-cycles
for j in range(vcycles):
    # down-smoothing and creation of FAS correction equation
    for i in range(levels-1,0,-1):   # levels-1, ..., 1
        # smoother application
        solve(res[i] == 0, u[i], bcs=bcs[i], options_prefix = 'down%d' % i,
              solver_parameters=smootherparams)
        # construct FAS correction problem with right-hand side
        #     ell[i-1] = R'(res(u[i]))) + F[i-1](R(u[i]))
        # where R' is canonical restriction and R is injection
        Rres = Function(VV[i-1])
        restrict(assemble(res[i]), Rres) # assemble() needed because
                                         # res[i] is UFL form only
        inject(u[i], Ru[i-1])

        # check that injection is acting as expected
        if False:
            print(u[i].dat.data)
            print(Ru[i-1].dat.data)
        # check that canonical restriction is acting as expected
        if False:
            xf = SpatialCoordinate(hierarchy[i])
            print(Function(VV[i]).interpolate(xf[0]).dat.data)
            print(assemble(res[i]).dat.data)
            xc = SpatialCoordinate(hierarchy[i-1])
            print(Function(VV[i-1]).interpolate(xc[0]).dat.data)
            print(Rres.dat.data)

        ell[i-1] = Rres * v[i-1] * dx + FRu[i-1]
        res[i-1] = F[i-1] - ell[i-1]
        inject(u[i], u[i-1])  # for initial value on next level
                              # equivalent to setting initial error iterate to zero
                              # but restrict() seems better?

    # coarse-mesh problem and direct solution
    solve(res[0] == 0, u[0], bcs=bcs[0], options_prefix = 'coarse',
          solver_parameters=coarseparams, )
    print('%s|coarse cor|=%.6f' % ((levels-1)*'  ', norm(u[0])))

    # prolongation and up-smoothing
    for i in range(1,levels):   # 1, ..., levels-1
        Pcor = Function(VV[i])
        prolong(assemble(u[i-1] - Ru[i-1]), Pcor)  # assemble() needed because
                                                   # subtraction is UFL
        u[i] += Pcor
        solve(res[i] == 0, u[i], bcs=bcs[i], options_prefix = 'up%d' % i,
              solver_parameters=smootherparams)

    # report on result of cycle
    print('vcycle %d (%d levels) |residual|=%.6f  |error|=%.6f' \
          % (j+1, levels, norm(assemble(res[fine])), error(u[fine])))

u[fine].rename("u")
uexact = Function(VV[fine]).interpolate(exact)
uexact.rename("uexact")
File("u-bratu1.pvd").write(u[fine],uexact)

#print(u[fine].dat.data)
#uexact = Function(VV[fine]).interpolate(exact)
#print(uexact.dat.data)
