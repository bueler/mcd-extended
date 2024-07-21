#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

# data from Table 4 in paper
n = 15 * 2**np.arange(0,9) + 1
mJ2D = n**2
mJ3D = n[:5]**3
F2D = [1,2,2,2,2,2,3,3,3]
F3D = [1,2,2,2,3]

plt.clf()
plt.semilogx(mJ2D,F2D,'+',ms=12.0,mew=2.0,label='FMG cycles (2D)')
plt.semilogx(mJ3D,F3D,'+',ms=14.0,mew=2.0,label='FMG cycles (3D)')
plt.xlabel(r'$m_J$  (number of unknowns)')
plt.ylabel('FASCD iterations')
plt.legend(loc='upper left', fontsize=12.0)
plt.xlim([10.0,1.0e8])
plt.ylim([0.0,4.0])
plt.gca().set_yticks(range(1,5))
writeout('advdiff.png')
