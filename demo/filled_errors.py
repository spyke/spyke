"""Demo plotting filled error ranges around lines"""

from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import pylab as pl


yoffs = np.arange(10)
nplots = len(yoffs)
x = np.arange(0, 1, 0.1)
npoints = len(x)
y = np.zeros((nplots, len(x)))
stdy = np.zeros((nplots, len(x)))
verts = np.zeros((nplots, 2*npoints, 2)) # each timepoint has a +ve and a -ve value
for ploti, yoff in enumerate(yoffs):
    y[ploti] = np.random.random(len(x)) / 2 + 0.25 + yoff
    stdy[ploti] = 0.2 + np.random.random(len(x)) * 0.2
    vert = mlab.poly_between(x, y[ploti]-stdy[ploti], y[ploti]+stdy[ploti])
    vert = np.asarray(vert).T
    verts[ploti] = vert

# can also use axes.fill() instead of a poly collection, or directly use axes.fill_between()
pcol = mpl.collections.PolyCollection(verts, facecolors='r', edgecolors='none', alpha=0.2)
a = pl.gca()
#pcol = a.fill_between(x, y+stdy, y-stdy, facecolors='r', edgecolors='none', alpha=0.2)
a.add_collection(pcol)
for ploti in range(nplots):
    a.plot(x, y[ploti], 'r-')
a.set_xlim((0, 1))
