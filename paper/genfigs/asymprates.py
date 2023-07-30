#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

# from V-cycle results, mcd-extended/paper/results/drape_fascd.txt
# line J:  [k, |r_ss(w^0)|, |r_ss(w^k)|]
resnormball = [[ 1, 2.977687712552e+00, 5.592593521511e-16],  # J=0, single level
               [ 3, 4.597248896467e+00, 9.299029163684e-15],  # J=1, two-level
               [ 6, 4.019378820472e+00, 3.548401366476e-14],  # ...
               [ 7, 3.518652347563e+00, 4.898369539597e-13],
               [ 9, 3.989265817522e+00, 5.346032356444e-13],
               [11, 5.149110069199e+00, 1.683050131619e-12],
               [11, 7.072193804686e+00, 6.219113662073e-12],
               [12, 9.929467647243e+00, 6.630087376663e-12],
               [13, 1.401524016881e+01, 5.806872030903e-12],
               [13, 1.981030784949e+01, 1.876069712594e-11],
               [14, 2.801197654214e+01, 1.004260356580e-11]]
resnormspir = [[ 1, 1.250364911422e+01, 1.441420557646e-15],
               [ 4, 3.581030295829e+01, 1.356576646565e-14],
               [ 5, 5.195274377477e+01, 2.683842615342e-11],
               [ 7, 7.324445644144e+01, 8.422103502963e-12],
               [ 9, 1.069563682455e+02, 1.703081752902e-11],
               [10, 1.517759488345e+02, 6.241169671924e-11],
               [11, 2.119960943342e+02, 6.428954633272e-11],
               [12, 3.016400256848e+02, 1.293809842620e-10],
               [13, 4.275614401535e+02, 1.177739303183e-10],
               [14, 6.031903763444e+02, 1.310547032348e-10],
               [18, 8.529374614123e+02, 5.472658274103e-10]]

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize = 20.0

plt.figure(figsize=(8, 6))
J = np.arange(1,11) # lev = J + 1, but skip single level J=0
mcoarse = 4
#m0 = (mcoarse + 1)**2 + mcoarse**2    # because diagonals are crossed
mJ = (mcoarse * 2**J + 1)**2 + (mcoarse * 2**J)**2
rnball = np.array(resnormball)
kball = rnball[1:,0]
rhoball = (rnball[1:,2] / rnball[1:,1])**(1.0/kball)
plt.semilogx(mJ, rhoball, 'ko:', ms=10.0, markerfacecolor='w', label='ball')
rnspir = np.array(resnormspir)
kspir = rnspir[1:,0]
rhospir = (rnspir[1:,2] / rnspir[1:,1])**(1.0/kspir)
plt.semilogx(mJ, rhospir, 'k*:', ms=12.0, label='spiral')
plt.xlabel(r'degrees of freedom $m_J$', fontsize=fsize)
plt.ylabel(r'asymptotic rate $\rho_J$', fontsize=fsize)
#plt.gca().set_xticklabels(mJ)
plt.gca().tick_params(axis='both', which='major', labelsize=16.0)
plt.legend(loc='upper left', fontsize=fsize)
plt.axis('tight')
plt.xlim(100.0,1.0e8)
writeout('asymprates.pdf')
