import numpy as np
import pyximport
pyximport.install(build_in_temp=False, inplace=True)

import doublevsint

data = np.load('/home/mspacek/data/ptc18/tr1/tr1_chanclust51_ch1,24,25_3PCs.npy')[:1000]
data *= 10
ddata = np.float64(data)
idata = np.int64(data)

doublevsint.doublevsint(ddata, idata)

'''
# result:
dd2=182234330.288176, double took 2.249582 sec
id2=170997252, int took 2.344817 sec

So, int math isn't any faster than float math.

'''
