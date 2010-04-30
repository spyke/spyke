import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

from climbing import gradient_ascent # .pyx file

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

COLOURS = np.asarray([RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA, WHITE, BROWN])


data = scipy.io.loadmat('/data/ptc18/14_full.mat')
data = data['data']
#data = np.float32(data)
nd = data.shape[1]
data = data[:10000] # keep only the 1st 10000 data points for now
clusteris, clusters = gradient_ascent(data, sigma=7.0, alpha=3.0)
nclusters = len(clusters)

ncolours = len(COLOURS)
samplecolours = COLOURS[clusteris % ncolours]
clustercolours = COLOURS[np.arange(nclusters) % ncolours]
#colours[clusteris == -1] = GREY # unclassified points
f = figure()
a = gca()
f.canvas.SetBackgroundColour(wx.BLACK)
f.set_facecolor('black')
f.set_edgecolor('black')
a.set_axis_bgcolor('black')
scatter(data[:, 0], data[:, 1], s=1, c=samplecolours, edgecolors='none')
scatter(clusters[:, 0], clusters[:, 1], c=clustercolours)
show()
