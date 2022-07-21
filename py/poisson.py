"""solve poisson equation by 2-mesh FAS multigrid.
problem is
    F(u)[w] = ell[w]
where
    F(u)[w] = dot(grad(u), grad(w)) * dx
    ell[w] = f * w * dx
FAS equation is
    Fc(uc)[wc] = ellc[wc]
where Fc has same formula as F and
    ellc[wc] = R'(ellf[.] - Ff(uf)[.])[wc] + Fc(R(uf))[wc]
and R' is canonical restriction and R is injection
"""

from firedrake import *

# set up mesh hierarchy and fine-mesh data
cmesh = UnitSquareMesh(2, 2)  # as small as practical for nontriviality
hierarchy = MeshHierarchy(cmesh, 1)  # just 2 levels, coarse and fine
mesh = hierarchy[-1]
x, y = SpatialCoordinate(mesh)
f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

# fine-mesh problem can be described outside of V-cycles
V1 = FunctionSpace(hierarchy[-1], "CG", 1)
u1 = Function(V1)
#print(norm(u1))  # it is zero vector
w1 = TestFunction(V1)
F1 = dot(grad(u1), grad(w1)) * dx
ell1 = f * w1 * dx
residual1 = ell1 - F1
bcs1 = DirichletBC(V1, zero(), (1, 2, 3, 4))
params1 = {"snes_type": "ksponly",
           #"ksp_converged_reason": None,
           "ksp_max_it": 1,
           "ksp_type": "richardson",
           "pc_type": "sor"}

def error(u):
    expect = Function(V1).interpolate(exact)
    return norm(assemble(u - expect))

print('initial       |residual|=%.6f  |error|=%.6f' \
      % (norm(assemble(residual1)), error(u1)))

vcycles = 2
for j in range(vcycles):
    # fine-mesh smoother application
    solve(residual1 == 0, u1, bcs=bcs1,
          solver_parameters=params1)
    print('fine smoother |residual|=%.6f  |error|=%.6f' \
          % (norm(assemble(residual1)), error(u1)))

    # coarse-mesh FAS problem and direct solution
    V0 = FunctionSpace(hierarchy[0], "CG", 1)
    u0 = Function(V0)
    w0 = TestFunction(V0)
    F0 = dot(grad(u0), grad(w0)) * dx
    # build FAS rhs:  ell = R'(residual1(u1)) + F0(R(u1))
    # where R' is canonical restriction and R is injection
    r0 = Function(V0)
    restrict(assemble(residual1), r0)
    Ru = Function(V0)
    inject(u1, Ru)
    ell0 = r0 * w0 * dx + dot(grad(Ru), grad(w0)) * dx
    residual0 = ell0 - F0
    bcs0 = DirichletBC(V0, zero(), (1, 2, 3, 4))
    params0 = {"snes_type": "ksponly",
               #"ksp_converged_reason": None,
               "ksp_type": "preonly",
               "pc_type": "lu"}
    solve(residual0 == 0, u0, bcs=bcs0,
          solver_parameters=params0)
    print('  |coarse correction|=%.6f' % (norm(u0)))

    # prolong correction, add it, and do a smoother
    Pc1 = Function(V1)
    prolong(assemble(u0 - Ru), Pc1)
    u1 += Pc1
    solve(residual1 == 0, u1, bcs=bcs1,
          solver_parameters=params1)
    print('%d cycles      |residual|=%.6f  |error|=%.6f' \
          % (j+1, norm(assemble(residual1)), error(u1)))
