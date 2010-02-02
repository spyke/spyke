"""Tests threadpool code from http://code.activestate.com/recipes/576519,
saved as spyke.threadpool"""

from spyke import threadpool
from multiprocessing import cpu_count
import numpy as np
import time

data = np.random.random((54, 30*50000))
print(data)
data -= 0.5 # centered on 0
data *= 2**15
data = np.int16(data)
print(data)

tmt = time.time()
ncpus = cpu_count()
p = threadpool.Pool(ncpus)
print(p.map(np.mean, data))
print('%d thread pool took %.3f' % (ncpus, time.time()-tmt))

tnp = time.time()
print(np.mean(data, axis=1))
print('single thread numpy took %.3f' % (time.time()-tnp))
