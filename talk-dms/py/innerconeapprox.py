#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=15.0

def makeaxes(xmin,xmax,ymin,ymax):
    x = np.linspace(xmin,xmax,2)
    y = np.linspace(ymin,ymax,2)
    zero = np.zeros(np.shape(x))
    plt.plot(x,zero,'k',lw=1.0)
    plt.plot(zero,y,'k',lw=1.0)

plt.figure(figsize=(8,6))
xm, ym = 0.6, 0.4
makeaxes(-0.4, xm, -0.25, ym)
cornerx = [-0.3, -0.29, -0.2, -0.1]
cornery = [-0.2, -0.14, -0.09, 0.0]
sh = 0.02
for n, xc, yc in zip(range(4), cornerx, cornery):
    plt.fill([xc,xm,xm,xc,xc],[yc,yc,ym,ym,yc],'k',alpha=0.13)
    plt.plot(cornerx[n],cornery[n],'k.',ms=10.0)
    plt.text(cornerx[n]+(n-3)*sh,cornery[n]-2*sh,r'$\chi^%d$' % (3-n),
             fontsize=fsize)
plt.plot(0.0,0.0,'k.',ms=10.0)
plt.text(0.0+sh,0.0+sh,r'$0$',fontsize=fsize)
plt.text(-0.4,0.35,r'$\mathcal{V}^3$',
         fontsize=fsize)
plt.text(0.4,0.03,r'$\mathcal{U}^0=\mathcal{D}^0$',
         fontsize=fsize)
plt.text(0.33,cornery[2]+0.02,r'$\mathcal{U}^1=\mathcal{D}^0+\mathcal{D}^1$',
         fontsize=fsize)
plt.text(0.27,cornery[1]+0.01,r'$\mathcal{U}^2=\mathcal{D}^0+\mathcal{D}^1+\mathcal{D}^2$',
         fontsize=fsize)
plt.text(0.19,cornery[0]+0.01,
         r'$\mathcal{U}^3=\mathcal{D}^0+\mathcal{D}^1+\mathcal{D}^2+\mathcal{D}^3$',
         fontsize=fsize)
plt.axis('tight')
plt.axis('off')
plt.axis('equal')
writeout('innerconeapprox.png')
