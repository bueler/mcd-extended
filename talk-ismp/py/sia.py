#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

# data from Table 5 in paper
n = np.array([641, 1281, 2561, 5121, 10241, 20481])**2
FMG = [2, 2, 2, 2, 1, 1]

plt.clf()
plt.semilogx(n,FMG,'+',ms=14.0,mew=2.0,label='FMG cycles')
plt.xlabel(r'$m_J$  (number of unknowns)')
plt.ylabel('FASCD iterations')
plt.legend(loc='upper left', fontsize=12.0)
plt.xlim([1.0e5,1.0e9])
plt.ylim([0.0,3.0])
plt.gca().set_yticks(range(1,5))
writeout('sia.png')
