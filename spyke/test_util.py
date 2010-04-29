import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import util # .pyx file
import time
'''
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
'''

from pylab import figure, gca, scatter, show
import scipy.io
import wx


RED = '#FF0000'
ORANGE = '#FF7F00'
YELLOW = '#FFFF00'
GREEN = '#00FF00'
CYAN = '#00FFFF'
LIGHTBLUE = '#007FFF'
BLUE = '#0000FF'
VIOLET = '#7F00FF'
MAGENTA = '#FF00FF'
WHITE = '#FFFFFF'
BROWN = '#AF5050'
GREY = '#555555' # reserve as junk cluster colour

COLOURS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA, WHITE, BROWN]


data = scipy.io.loadmat('/data/ptc18/14_full.mat')
data = data['data']
#data = np.float32(data)
nd = data.shape[1]
data = data[:1000] # keep only the 1st 10000 data points for now
clusteris = util.gradient_ascent(data, sigma=3.0, alpha=3.0)


ncolours = len(COLOURS)
colouris = clusteris % ncolours
colours = np.asarray(COLOURS)[colouris]
#colours[clusteris == -1] = GREY # unclassified points
f = figure()
a = gca()
f.canvas.SetBackgroundColour(wx.BLACK)
f.set_facecolor('black')
f.set_edgecolor('black')
a.set_axis_bgcolor('black')
scatter(data[:, 0], data[:, 1], s=1, c=colours, edgecolors='none')
show()
