"""ATTEMPT to solve the following obstacle problem.  The plan is to solve by
single-level projected NGS using PatchSNES.  (I know this is inefficient, but
it is to be the smoother in an FAS scheme.)  I am currently blocked by
not knowing how to implement *projected* NGS.

ISSUES:
  * I seem to only know how to get nonlinear *Jacobi* sweeps; setting 
      "npc_patch_snes_patch_local_type": "multiplicative"
    to get nonlinear GS sweeps generates a seg fault.
  * I don't know how to intervene in the patch-wise Newton step to change it
    to project back up to the obstacle if it over shoots.  So for now the
    solver ignores the obstacle psi, and therefore just solves a Dirichlet
    problem (PDE).

Solve laplacian obstacle problem in 2D square Omega = (-2,2) x (-2,2):
  u >= psi           on Omega
  - triangle u = 0   where u > psi
  u = g              on boundary of Omega
Formulas for psi (a hemisphere) and g are given in the code.

The exact solution of this problem is given in Chapter 12 of Bueler (2021),
"PETSc for Partial Differential Equations".  See c/ch12/obstacle.c
at https://github.com/bueler/p4pdes.  Dirichlet conditions are taken from
the exact solution.
"""

import numpy as np
from firedrake import *

ngssweep = {
    "mat_type": "matfree",
    "snes_type": "nrichardson",
    "snes_converged_reason": None,
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
    "npc_patch_snes_patch_local_type": "additive",   # this is nonlinear Jacobi?
    #"npc_patch_snes_patch_local_type": "multiplicative",  # this would be NGS? seg faults!
    "npc_patch_snes_patch_symmetrise_sweep": False,
    "npc_patch_sub_snes_type": "newtonls",
    "npc_patch_sub_snes_linesearch_type": "basic",
    "npc_patch_sub_ksp_type": "preonly",
    "npc_patch_sub_pc_type": "lu",
}

mesh = SquareMesh(8,8,4.0)
mesh.coordinates.dat.data[:] -= 2.0       # square is now (-2,2) x (-2,2)
# mesh coordinates as numpy arrays
xa = np.array(mesh.coordinates.dat.data_ro[:,0])
ya = np.array(mesh.coordinates.dat.data_ro[:,1])
V = FunctionSpace(mesh, "CG", 1)

# z = psi(x,y) is a hemispherical obstacle, made C^1 with "skirt" at r=r0
# uses numpy arrays to set values conditionally
r = np.sqrt(xa * xa + ya * ya)
r0 = 0.9
psi0 = np.sqrt(1.0 - r0 * r0)
dpsi0 = - r0 / psi0
psi = Function(V)
psi.dat.data[:] = psi0 + dpsi0 * (r - r0);
psi.dat.data[r <= r0] = np.sqrt(1.0 - r[r <= r0] * r[r <= r0])

# exact solution; see Chapter 12 text; solves Laplace eqn where uexact > psi
afree = 0.697965148223374
A     = 0.680259411891719
B     = 0.471519893402112
uexact = Function(V)
uexact.dat.data[:] = psi.dat.data_ro
uexact.dat.data[r > afree] = - A * np.log(r[r > afree]) + B

# Laplace equation weak form
u = Function(V)
v = TestFunction(V)
F = dot(grad(u), grad(v)) * dx

# Dirichlet boundary conditions from exact solution
bcs = DirichletBC(V, Function(V).interpolate(uexact), (1, 2, 3, 4))

# GOAL:  change NGS to projected NGS so this solves obstacle problem!
solve(F == 0, u, bcs=bcs, options_prefix = 's', solver_parameters=ngssweep)

# outputs allow comparing u to both psi and uexact
psi.rename("psi")
uexact.rename("uexact")
u.rename("u")
File("obstacle-result.pvd").write(psi,uexact,u)
