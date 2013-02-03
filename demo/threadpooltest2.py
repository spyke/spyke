"""Tests threadpool code from http://code.activestate.com/recipes/576519,
saved as spyke.threadpool. Threaded code runs about ncpus times faster than
unthreaded code, yet unlike multiprocessing, allows for shared memory. Only
when you're inside a numpy loop is the GIL released, and multithreading possible.
Or, same goes if you explicitlyl release the GIL inside Cython code"""

import numpy as np
import pyximport
pyximport.install(build_in_temp=False, inplace=True)

from spyke import threadpool
from multiprocessing import cpu_count
import time

import spyke.util as util

a = np.random.random(8*10000000)
tmt = time.time()
util.dostuffthreads(a)
print('dostuffthreads took %.3f' % (time.time()-tmt))
tst = time.time()
util.dostuff(a)
print('single thread took %.3f' % (time.time()-tst))

'''
tpool = time.time()
ncpus = cpu_count()
#ncpus = 4
pool = threadpool.Pool(ncpus)
print('pool creation took %.3f' % (time.time()-tpool))

a = np.random.random(8*10000000)
units = np.split(a, ncpus)

tmt = time.time()
pool.map(util.dostuff, units) # multithread over units
print('%d thread took %.3f' % (ncpus, time.time()-tmt))
tst = time.time()
util.dostuff(a)
print('single thread took %.3f' % (time.time()-tst))
pool.terminate()
'''



'''

def sort_inplace(data):
    """Sort data in-place along last axis"""
    data.sort()

sortdata1 = np.random.random((8, 1e7))
sortdata2 = sortdata1.copy()

tmt = time.time()
pool.map(sort_inplace, sortdata1) # multithread over rows in data
print(sortdata1)
print('%d thread numpy sort took %.3f' % (ncpus, time.time()-tmt))

tst = time.time()
sort_inplace(sortdata2)
print(sortdata2)
print('single thread numpy sort took %.3f' % (time.time()-tst))

del sortdata1
del sortdata2

tpool = time.time()
pool.terminate()
print('pool termination took %.3f' % (time.time()-tpool))







tpool = time.time()
ncpus = cpu_count()
pool = threadpool.Pool(ncpus)
print('pool creation took %.3f' % (time.time()-tpool))

#data = np.random.random((16, 1e7))
data = np.empty((64, 30*500000), dtype=np.int16)
#data -= 0.5 # centered on 0
#data *= 2**15
#data = np.int16(data)

tmt = time.time()
results = pool.map(np.mean, data) # multithread over rows in data
results = np.asarray(results)
print(results)
print('%d thread numpy mean took %.3f' % (ncpus, time.time()-tmt))

tst = time.time()
results = np.mean(data, axis=1)
print(results)
print('single thread numpy mean took %.3f' % (time.time()-tst))

del data

tpool = time.time()
pool.terminate()
print('pool termination took %.3f' % (time.time()-tpool))
'''
