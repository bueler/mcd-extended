#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

# data from Table 2 in paper
mJ = [41, 145, 545, 2.1e3, 8.3e3, 3.3e4, 1.3e5, 5.3e5, 2.1e6, 8.4e6, 3.4e7]
ballV = [1,2,3,7,9,11,11,12,13,13,14]
ballF = [1,2,3,3,4,5,4,4,3,2,2]
spirV = [1,4,5,7,9,10,11,12,13,14,18]
spirF = [1,2,3,4,4,5,5,5,6,6,6]

names = ['ballitersV', 'ballitersVlog', 'ballitersVF', 'bothitersVF']
for j in range(4):
    plt.clf()
    plt.semilogx(mJ,ballV,'o',ms=12.0,color='C0',label='V-cycles (ball problem)')
    if j == 1:
        p = np.polyfit(np.log(mJ),ballV,1)
        fitstr = r'$O(log(m_J))$'
        plt.semilogx(mJ,p[0]*np.log(mJ)+p[1],'--',color='C0',label=fitstr)
    if j >= 2:
        plt.semilogx(mJ,ballF,'+',ms=16.0,mew=2.0,color='C0',label='FMG cycles (ball problem)')
    if j == 3:
        plt.semilogx(mJ,spirV,'o',ms=8.0,color='C1',label='V-cycles (spiral problem)')
        plt.semilogx(mJ,spirF,'+',ms=12.0,mew=2.0,color='C1',label='FMG cycles (spiral problem)')
    plt.xlabel(r'$m_J$  (number of unknowns)')
    plt.ylabel('FASCD iterations')
    plt.legend(loc='upper left', fontsize=12.0)
    plt.xlim([10.0,1.0e8])
    plt.ylim([0.0,19.0])
    plt.gca().set_yticks(range(2,20,2))
    writeout(names[j] + '.png')
