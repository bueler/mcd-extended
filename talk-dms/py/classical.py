import numpy as np
import matplotlib.pyplot as plt

_SHOW = False
def writeout(outname):
    if _SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname, bbox_inches='tight')

# data from Table 2 in paper
mJ = [41, 145, 545, 2.1e3, 8.3e3, 3.3e4, 1.3e5, 5.3e5, 2.1e6, 8.4e6, 3.4e7]
ballV = [1,2,3,7,9,11,11,12,13,13,14]
ballF = [1,2,3,3,4,5,4,4,3,2,2]
spirV = [1,4,5,7,9,10,11,12,13,14,18]
spirF = [1,2,3,4,4,5,5,5,6,6,6]

# ball iterations: V only, with fit
plt.semilogx(mJ,ballV,'o',label='V-cycles (ball problem)')
p = np.polyfit(np.log(mJ),ballV,1)
print(p)
fitstr = r'$O(log(m_J))$'
line, = plt.semilogx(mJ,p[0]*np.log(mJ)+p[1],'b--',label=fitstr)
plt.xlabel(r'$m_J$  (number of unknowns)')
plt.legend(loc='upper left', fontsize=12.0)
plt.xlim([10.0,1.0e8])
plt.ylim([0.0,19.0])
plt.gca().set_yticks(range(2,20,2))
writeout('balliterslog.png')
line.remove()

# ball iterations: both V and F
#plt.semilogx(mJ,ballV,'o',label='V-cycles (ball problem)')
plt.semilogx(mJ,ballF,'+',ms=14.0,label='F-cycles (ball problem)')
plt.xlabel(r'$m_J$  (number of unknowns)')
plt.legend(loc='upper left', fontsize=12.0)
plt.xlim([10.0,1.0e8])
plt.ylim([0.0,19.0])
plt.gca().set_yticks(range(2,20,2))
writeout('balliters.png')

# add spiral iterations
plt.semilogx(mJ,spirV,'o',label='V-cycles (spiral problem)')
plt.semilogx(mJ,spirF,'+',ms=14.0,label='F-cycles (spiral problem)')
plt.xlabel(r'$m_J$  (number of unknowns)')
plt.legend(loc='upper left', fontsize=12.0)
plt.xlim([10.0,1.0e8])
plt.ylim([0.0,19.0])
plt.gca().set_yticks(range(2,20,2))
writeout('bothiters.png')
