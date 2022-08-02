"""Attempt to solve Bratu equation in 2D by single-level NGS using PatchSNES:
   -triangle u - lambda e^u = 0
The exact solution of this problem is given by Liouville (1853); see
Exercise 7.12 in Bueler (2021), "PETSc for Partial Differential Equations"
and c/ch7/solns/bratu2D.c in https://github.com/bueler/p4pdes.
Dirichlet conditions on unit square are taken from exact solution.
"""

from firedrake import *

newtoncg = {
    "snes_type": "newtonls",
    #"snes_view": None,
    "snes_converged_reason": None,
    "snes_monitor": None,
    "ksp_type": "cg",
    "ksp_converged_reason": None,
    "pc_type": "jacobi",
}

npcnewtoncg = {
    "mat_type": "matfree",
    "snes_type": "nrichardson",
    #"snes_view": None,
    "snes_converged_reason": None,
    "snes_monitor": None,
    "snes_npc_side": "left",
    "npc_snes_type": "newtonls",
    "npc_snes_linesearch_type": "basic",
    "npc_ksp_type": "cg",
    "npc_ksp_converged_reason": None,
    "npc_pc_type": "jacobi",
    #"npc_pc_type": "python",
    #"npc_pc_python_type": "firedrake.AssembledPC",
    #"assembled_ksp_type": "preonly",
    #"assembled_pc_type": "lu",
}

ngssweep = {
    "mat_type": "matfree",
    "snes_type": "nrichardson",
    #"snes_view": None,
    "snes_converged_reason": None,
    "snes_monitor": None,
    "snes_npc_side": "left",
    "npc_snes_type": "python",
    "npc_snes_python_type": "firedrake.PatchSNES",
    "npc_snes_max_it": 1,
    "npc_snes_convergence_test": "skip",
    "npc_snes_linesearch_type": "basic",
    "npc_snes_linesearch_damping": 1.0,
    "npc_patch_snes_patch_construct_type": "star",
    "npc_patch_snes_patch_partition_of_unity": True,
    "npc_patch_snes_patch_sub_mat_type": "seqaij",
    "npc_patch_snes_patch_local_type": "additive",
    "npc_patch_snes_patch_symmetrise_sweep": False,
    "npc_patch_sub_snes_type": "newtonls",
    "npc_patch_sub_snes_linesearch_type": "basic",
    "npc_patch_sub_ksp_type": "preonly",
    "npc_patch_sub_pc_type": "lu",
    #"npc_patch_snes_patch_sub_snes_atol": 1.0e-11,
    #"npc_patch_snes_patch_sub_snes_rtol": 1.0e-11,
}

mesh = UnitSquareMesh(4,4)

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
v = TestFunction(V)

lam = 1.0
F = ( dot(grad(u), grad(v)) - lam * exp(u) * v ) * dx

# exact solution; see pages 196, 197 of Bueler 2021 book
x, y = SpatialCoordinate(mesh)
r2 = (x+1.0)*(x+1.0) + (y+1.0)*(y+1.0)
qq = r2 * r2 + 1.0
omega = r2 / (qq * qq)
exact = ln(32.0 * omega)

# bcs from exact solution
bcs = DirichletBC(V, Function(V).interpolate(exact), (1, 2, 3, 4))

# evaluate error using function-space L2 norm
def error(u):
    udiff = Function(V).interpolate(u - exact)
    return sqrt(assemble(dot(udiff, udiff) * dx))

print('initial  |error|=%.3e' % (error(u)))
#solve(F == 0, u, bcs=bcs, options_prefix = 's', solver_parameters=newtoncg)
#solve(F == 0, u, bcs=bcs, options_prefix = 's', solver_parameters=npcnewtoncg)
solve(F == 0, u, bcs=bcs, options_prefix = 's', solver_parameters=ngssweep)
print('final    |error|=%.3e' % (error(u)))

u.rename("u")
uexact = Function(V).interpolate(exact)
uexact.rename("uexact")
File("bratu2patch.pvd").write(u,uexact)
