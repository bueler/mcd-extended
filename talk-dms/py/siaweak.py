import numpy as np
import matplotlib.pyplot as plt

_SHOW = False
def writeout(outname):
    if _SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname, bbox_inches='tight')

# data from Table 5 in paper
P = ['1','4','16','64','256','1024']
#iters = [3,3,3,3,2,2]
time = [98.0,98.0,124.0,136.0,95.0,177.0]
plt.clf()
#plt.bar(P,iters)
#plt.ylabel('FMG cycles')
#plt.ylim([0.0,4.0])
#plt.gca().set_yticks(range(1,5))
#writeout('siaweakiters.png')
plt.bar(P,time)
plt.xlabel('processors')
plt.ylabel('run time (s)')
plt.ylim([0.0,200.0])
plt.gca().set_yticks([50.0,100.0,150.0,200.0])
writeout('siaweaktime.png')
