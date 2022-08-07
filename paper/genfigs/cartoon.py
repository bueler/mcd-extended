#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize = 20.0

def makeaxes(xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, 2)
    y = np.linspace(ymin, ymax, 2)
    zero = np.zeros(np.shape(x))
    plt.plot(x, zero, 'k', lw = 1.0)
    plt.plot(zero, y, 'k', lw = 1.0)

plt.figure(figsize=(8, 6))
makeaxes(-0.25, 1.2, -0.1, 0.8)
plt.text(-0.05, -0.07, r'$0$', fontsize=fsize)

# filled gray box
psi1 = 0.2
psi2 = 0.35
plt.fill([psi1, 1.2, 1.2, psi1, psi1],
         [psi2, psi2, 0.8, 0.8, psi2],
         'k', alpha = 0.2)
plt.text(0.9, 0.6, r'$\mathcal{K}$', fontsize=fsize)

# put gray dots suggesting unboundedness
plt.plot(0.01 + np.array([1.25, 1.30, 1.35]), 0.6 * np.array([1,1,1]), 'k.', ms = 12.0, markeredgecolor='w', alpha = 0.3)
plt.plot(0.7 * np.array([1,1,1]), 0.01 + np.array([0.85, 0.90, 0.95]), 'k.', ms = 12.0, markeredgecolor='w', alpha = 0.3)
plt.plot([1.25, 1.30, 1.35], [0.85, 0.90, 0.95], 'k.', ms = 12.0, markeredgecolor='w', alpha = 0.3)

plt.text(1.28, -0.02, r'$\mathcal{V}_1$', fontsize=fsize)
plt.plot(psi1, 0.0, 'k.', ms = 12.0)
plt.plot([psi1, 1.2], [0.0, 0.0], 'k', lw = 2.5)
plt.text(psi1 - 0.07, - 0.09, r'$\psi(x_1)$', fontsize=fsize)
plt.text(0.7, - 0.09, r'$\mathcal{K}_1$', fontsize=fsize)
plt.gca().annotate('', xy=(0.7, psi2 - 0.3), xytext=(0.7, psi2 - 0.05),
                   arrowprops=dict(arrowstyle='->, head_length=0.8, head_width=0.5'))
plt.text(0.72, 0.15, r'$R_1$', fontsize=fsize)

plt.text(-0.02, 0.89, r'$\mathcal{V}_2$', fontsize=fsize)
plt.plot(0.0, psi2, 'k.', ms = 12.0)
plt.plot([0.0, 0.0], [psi2, 0.8], 'k', lw = 2.5)
plt.text(- 0.22, psi2 - 0.02, r'$\psi(x_2)$', fontsize=fsize)
plt.text(- 0.12, 0.6, r'$\mathcal{K}_2$', fontsize=fsize)
plt.gca().annotate('', xy=(psi1 - 0.15, 0.6), xytext=(psi1 - 0.05, 0.6),
                   arrowprops=dict(arrowstyle='->, head_length=0.8, head_width=0.5'))
plt.text(0.07, 0.65, r'$R_2$', fontsize=fsize)

plt.axis('tight')
plt.axis('off')
plt.axis('equal')
writeout('cartoon.pdf')
