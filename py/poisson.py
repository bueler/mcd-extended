# see https://www.firedrakeproject.org/demos/geometric_multigrid.py for original source of this example

from firedrake import *
from firedrake.mg.utils import coarse_node_to_fine_node_map

cmesh = UnitSquareMesh(2, 2)  # as small as practical for nontriviality
hierarchy = MeshHierarchy(cmesh, 1)  # just 2 levels, coarse and fine
mesh = hierarchy[-1]

Vf = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(Vf)
v = TestFunction(Vf)

a = dot(grad(u), grad(v))*dx
bcs = DirichletBC(Vf, zero(), (1, 2, 3, 4))

# For a forcing function, we will use a product of sines such that we
# know the exact solution and can compute an error.
x, y = SpatialCoordinate(mesh)
f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
L = f*v*dx
exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

Vc = FunctionSpace(cmesh, "CG", 1)

def test_prolong():
    """ prolong a function f from coarse mesh hierarchy[0] to fine mesh hierarchy[-1] and write out .pvd for f on each mesh """
    xc, yc = SpatialCoordinate(cmesh)
    exactc = sin(pi*xc)*tan(pi*xc*0.25)*sin(pi*yc)
    f_coarse = Function(Vc).interpolate(exactc)
    f_prolonged = Function(Vf)
    prolong(f_coarse,f_prolonged)
    f_coarse.rename("f_coarse")
    f_prolonged.rename("f_prolonged")
    File("fc.pvd").write(f_coarse)
    File("fc-prolonged.pvd").write(f_prolonged)

def test_restrict():
    """ restrict a function f from fine mesh hierarchy[-1] to coarse mesh hierarchy[0] and write out .pvd for f on each mesh; officially these functions are in the dual spaces """
    f_fine = Function(Vf).interpolate(exact)
    f_restricted = Function(Vc)
    restrict(f_fine,f_restricted)
    f_fine.rename("f_fine")
    f_restricted.rename("f_restricted")
    File("ff.pvd").write(f_fine)
    File("ff-restricted.pvd").write(f_restricted)

def test_monotone_restrict():
    """ monotone restrict a function f from fine mesh hierarchy[-1] to coarse mesh hierarchy[0] and write out .pvd for f on each mesh; thanks to Lawrence Mitchell for this code; note the negative function f(x,y) = -xy generates nonzero monotone restriction onto coarsest mesh UnitSquareMesh(2, 2) """
    f_fine = Function(Vf).interpolate(- x * y)
    f_restricted = Function(Vc)
    c2fmap = coarse_node_to_fine_node_map(Vc, Vf)
    f_restricted.dat.data_with_halos[:] = \
        f_fine.dat.data_ro_with_halos[c2fmap.values_with_halo].max(axis=1)
    #print(f_restricted.dat.data_ro)
    f_fine.rename("f_fine")
    f_restricted.rename("f_restricted")
    File("fff.pvd").write(f_fine)
    File("fff-mrestricted.pvd").write(f_restricted)

def run_solve(parameters):
    """ function that takes in set of parameters and returns the solution """
    u = Function(Vf)
    solve(a == L, u, bcs=bcs, solver_parameters=parameters)
    return u

def error(u):
    expect = Function(Vf).interpolate(exact)
    return norm(assemble(u - expect))

#u = run_solve({"ksp_type": "preonly", "pc_type": "lu"})
#print('LU solve error', error(u))
u = run_solve({"ksp_type": "cg", "pc_type": "mg"})
print('MG V-cycle + CG error', error(u))
#u.rename("u")
#File("solution.pvd").write(u)

test_prolong()
test_restrict()
test_monotone_restrict()
