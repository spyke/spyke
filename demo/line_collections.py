"""Shows how a LineCollection could be used in place of a bunch of Line2Ds to 
represent a plot slot. Inits faster, but otherwise I'm not sure if there's a
speed difference. Might get rid of flicker when refreshing selected spikes
in latest mpl

Problem: not sure how to set visibility of individual lines in line collection,
might not be possible. Seems you can only set the visibility of the lc as a whole.
Instead, it's probably best to just pass it as many segments
as you need when you need them, according to the segments data.
You're not generating variable numbers of python objects each time I don't think,
like you are when makeing Line2Ds, so hopefully having variable numbers of channels
per plot slot won't cause a slowdown with LCs like it did with Line2Ds (which is
what made me init all Line2D plots to have the same number of chans, 54, but most of
which were left disabled). 
"""

from __future__ import division

import sys
import time
from PyQt4 import QtGui

import numpy as np

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

nplots = 1
nchans = 54
npoints = 50
x = np.tile(np.arange(0, 1, 1/npoints), nplots*nchans).reshape(nplots, nchans, npoints)
y = np.random.random(x.shape)
# add y offsets:
for i in range(nchans):
    y[:, i, :] += i

segments = np.zeros((nplots, nchans, npoints, 2))
segments[:, :, :, 0] = x
segments[:, :, :, 1] = y

linestyle = '-'
linewidth = 1#0.2
zorder = 4
visible = np.bool8(np.random.random_integers(0, 1, nchans))

class MyFigureCanvasQTAgg(FigureCanvasQTAgg):
    def __init__(self):
        figure = Figure()
        FigureCanvasQTAgg.__init__(self, figure)
        self.ax = figure.add_axes([0, 0, 1, 1])
        t0 = time.time()
        #self.init_lines()
        self.init_lc()
        self.ax.set_ylim([0, nchans])
        print('initing artists took %.3f sec' % (time.time()-t0))

    def init_lines(self):
        for i in range(nplots):
            for j in range(nchans):
                line = Line2D(segments[i, j, :, 0], segments[i, j, :, 1],
                              visible=visible, linestyle=linestyle, linewidth=linewidth,
                              zorder=zorder)
                self.ax.add_line(line)

    def init_lc(self):
        self.lcs = []
        for i in range(nplots):
            lc = LineCollection(segments[i], visible=True,
                                linestyles=linestyle, linewidths=linewidth,
                                zorder=zorder)
            self.ax.add_collection(lc)
            self.lcs.append(lc)
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyFigureCanvasQTAgg()
    window.show()
    sys.exit(app.exec_())
