import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import util # .pyx file
import time

"""Run some tests"""
data = np.int16(np.random.randint(-2**15, 2**15-1, 52*(500e3+1)))
data.shape = 52, 500e3+1
print('data: %r' % data)

tmed = time.clock()
mydata = np.abs(data) # does a copy
print('copy took %.3f sec' % (time.clock()-tmed))
result = np.median(mydata, axis=-1) # doesn't modify data, does internal copy
print('np.median took %.3f sec' % (time.clock()-tmed))
print('np.median: %r' % np.int16(result))
print('mydata: %r' % mydata)

tcy = time.clock()
mydata = np.abs(data) # does a copy
print('copy took %.3f sec' % (time.clock()-tcy))
result = util.median_inplace_2Dshort(mydata)
print('Cython median took %.3f sec' % (time.clock()-tcy))
print('Cython median: %r' % result)
print('mydata: %r' % mydata)

tcy = time.clock()
mydata = np.abs(data) # does a copy
print('copy took %.3f sec' % (time.clock()-tcy))
result = util.mean_2Dshort(mydata)
print('Cython mean took %.3f sec' % (time.clock()-tcy))
print('Cython mean: %r' % result)
print('mydata: %r' % mydata)
