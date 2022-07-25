"""solve poisson equation by Firedrake's usual multigrid"""

from firedrake import *

levels = 2
vcycles = 1
cmesh = UnitSquareMesh(2, 2)  # as small as practical for nontriviality
hierarchy = MeshHierarchy(cmesh, levels-1)
mesh = hierarchy[-1]

Vf = FunctionSpace(mesh, "CG", 1)
u = Function(Vf)
v = TestFunction(Vf)

x, y = SpatialCoordinate(mesh)
f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

F = dot(grad(u), grad(v)) * dx - f*v*dx
bcs = DirichletBC(Vf, zero(), (1, 2, 3, 4))

Vc = FunctionSpace(cmesh, "CG", 1)

def run_solve(parameters):
    """ function that takes in set of parameters and returns the solution """
    solve(F == 0, u, bcs=bcs,
          solver_parameters=parameters)
    return u

def error(u):
    expect = Function(Vf).interpolate(exact)
    return norm(assemble(u - expect))

print('initial                   |residual|=%.6f  |error|=%.6f' \
      % (norm(assemble(-F)), error(u)))
params = {"snes_type": "ksponly",
          "snes_max_linear_solve_fail": 2, # don't error when KSP reports DIVERGED_ITS
          #"snes_view": None,
          #"ksp_converged_reason": None,
          #"ksp_monitor_true_residual": None,
          "ksp_type": "richardson", # classical V-cycles
          "ksp_max_it": vcycles,
          "pc_type": "mg",
          "mg_levels_ksp_type": "richardson",
          "mg_levels_ksp_max_it": 1,
          "mg_levels_ksp_converged_reason": None,
          "mg_levels_pc_type": "sor",
          "mg_coarse_ksp_type": "preonly",
          "mg_coarse_pc_type": "lu"}
u = run_solve(params)
print('MG V-cycles (%d cycles, %d levels)  |residual|=%.6f  |error|=%.6f' \
      % (vcycles, levels, norm(assemble(-F)), error(u)))
u.rename("u")
File("u-poisson0.pvd").write(u)
