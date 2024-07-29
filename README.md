# mcd-extended

This project's original goal was to extend the multilevel constraint decomposition method of [Tai (2003)](https://doi.org/10.1007/s002110200404), thus the repository name.  The new method is in fact called _full approximation scheme constraint decomposition_ (FASCD), and it is a joint extension of Tai's work and the FAS scheme of [Brandt (1977)](https://www.ams.org/journals/mcom/1977-31-138/S0025-5718-1977-0431719-X/S0025-5718-1977-0431719-X.pdf).

FASCD is implemented in Python using the [Firedrake](https://www.firedrakeproject.org/index.html) finite element library at [Patrick Farrell's `fascd` repository](https://bitbucket.org/pefarrell/fascd/src/master/).

The paper itself is complete and published:

  * E. Bueler & P. E. Farrell (2024), _A full approximation scheme multilevel method for nonlinear variational inequalities_, SIAM J. Sci. Comput. 46 (4) [10.1137/23M1594200](https://doi.org/10.1137/23M1594200)
