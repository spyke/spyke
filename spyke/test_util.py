import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import util # .pyx file
import time

'''
"""Median tests"""
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
'''

"""Test argsharpness2D"""
signal = np.array([ -216,  -246,  -482,  -831, -1043, -1104, -1104, -1156, -1214, -1183, -1129, -1161, -1288, -1049,  -261,   872,  1871,  2502,  2701,  2317, 1153,  -885, -3352, -5442, -6487, -6525, -5820, -4547, -2827, -1071,   322,  1236,  1815,  2226,  2521,  2609,  2583,  2650,  2902,  2972, 2675,  2186,  1899,  1872,  1813,  1460,   837,   386,   351,   606], dtype=np.int16)
signal.shape = 1, len(signal)

ext, extti, npoints, sharp, nsegments = util.argsharpness2D(signal)
