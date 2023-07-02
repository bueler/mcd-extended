#!/usr/bin/env python

# see run.sh (and output rawFD.txt), to generate input data data.txt

import numpy as np
import matplotlib.pyplot as plt
import sys

INFILE = 'data.txt'
#PROBMARKER = ['*','o']
#PARMARKERSIZE = [10.0,14.0,18.0,8.0,12.0,16.0]

SHOW = False
def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))

assert len(v[:,0]) == 5

N = v[:,0]**2
flops = v[:,1]
exps = v[:,2]
time = v[:,3]

# fit to show  (flops) = O(N^1):
#qflops = np.polyfit(np.log(N),np.log(flops),1)

# exps/N versus N
plt.figure(figsize=(7,6))
plt.semilogx(N,exps/N,'ko:',ms=14.0)
plt.xlabel('N = degrees of freedom',fontsize=18.0)
plt.ylabel('exps / N',fontsize=18.0)
plt.grid(True)
plt.axis([1e5, 1e8, 0, 50])
writeout('bratuexps.png')

# time/N versus N
plt.figure(figsize=(7,6))
plt.semilogx(N,1.0e6 * time/N,'ko:',ms=14.0)
plt.xlabel('N = degrees of freedom',fontsize=18.0)
plt.ylabel('$\mu s$ / N',fontsize=18.0)
plt.grid(True)
plt.axis([1e5, 1e8, 0, 2.5])
writeout('bratutime.png')
