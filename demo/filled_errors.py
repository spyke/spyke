"""Demo plotting filled error ranges around a line"""

from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import pylab as pl

x = np.arange(0, 1, 0.1)
y = np.random.random(len(x)) / 2 + 0.25
stdy = 0.02 + np.random.random(len(x)) * 0.02

'''
vert = mlab.poly_between(x, y+stdy, y-stdy)
vert = np.asarray(vert).T
verts = [vert]
'''
# can also use axes.fill() instead of a poly collection, or directly use axes.fill_between()
#pcol = mpl.collections.PolyCollection(verts, facecolors='r', edgecolors='none', alpha=0.2)
a = pl.gca()
pcol = a.fill_between(x, y+stdy, y-stdy, facecolors='r', edgecolors='none', alpha=0.2)
a.add_collection(pcol)
a.plot(x, y, 'r-')
a.set_xlim((0, 1))
