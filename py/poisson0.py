"""solve poisson equation by Firedrake's usual multigrid"""

from firedrake import *

cmesh = UnitSquareMesh(2, 2)  # as small as practical for nontriviality
hierarchy = MeshHierarchy(cmesh, 1)  # just 2 levels, coarse and fine
mesh = hierarchy[-1]

Vf = FunctionSpace(mesh, "CG", 1)
u = Function(Vf)
v = TestFunction(Vf)

x, y = SpatialCoordinate(mesh)
f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)

F = dot(grad(u), grad(v)) * dx - f*v*dx
bcs = DirichletBC(Vf, zero(), (1, 2, 3, 4))

exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

Vc = FunctionSpace(cmesh, "CG", 1)

def run_solve(parameters):
    """ function that takes in set of parameters and returns the solution """
    solve(F == 0, u, bcs=bcs,
          solver_parameters=parameters)
    return u

def error(u):
    expect = Function(Vf).interpolate(exact)
    return norm(assemble(u - expect))

params = {"snes_type": "ksponly",
          #"snes_view": None,
          "ksp_converged_reason": None,
          "ksp_monitor": None,
          "ksp_type": "cg",
          "mg_levels_ksp_type": "richardson",
          "mg_levels_pc_type": "sor",
          "pc_type": "mg"}
u = run_solve(params)
print('MG V-cycle + CG error', error(u))
#u.rename("u")
#File("solution.pvd").write(u)
