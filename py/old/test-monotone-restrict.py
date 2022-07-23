from firedrake import *
from firedrake.mg.utils import coarse_node_to_fine_node_map

cmesh = UnitSquareMesh(2, 2)  # as small as practical for nontriviality
hierarchy = MeshHierarchy(cmesh, 1)  # just 2 levels, coarse and fine
mesh = hierarchy[-1]

Vf = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
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

def test_inject():
    """ inject a function f from fine mesh hierarchy[-1] to coarse mesh hierarchy[0] and write out .pvd for f on each mesh """
    f_fine = Function(Vf).interpolate(exact)
    f_injected = Function(Vc)
    inject(f_fine,f_injected)
    f_fine.rename("f_fine")
    f_injected.rename("f_injected")
    File("fff.pvd").write(f_fine)
    File("fff-injected.pvd").write(f_injected)

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
    File("ffff.pvd").write(f_fine)
    File("ffff-mrestricted.pvd").write(f_restricted)

test_prolong()
test_restrict()
test_inject()
test_monotone_restrict()
