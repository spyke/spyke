"""Trying out Gaussian mixture modelling using PyMix"""

import numpy as np
import scipy.io
import pymix.mixture as pm
from pylab import figure, gca, scatter
from matplotlib.patches import Ellipse
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
GREY = '#555555'
WHITE = '#FFFFFF'
BROWN = '#AF5050'
DARKGREY = '#222222' # reserve as junk cluster colour

COLOURS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA, GREY, WHITE, BROWN]

data = scipy.io.loadmat('/data/ptc18/14_full.mat')
data = data['data']
nd = data.shape[1]
data = data[:10000] # keep only the 1st 10000 data points for now
mins = data.min(axis=0)
maxs = data.max(axis=0)
ranges = maxs - mins

ndistribs = 50
distribs = []
for i in range(ndistribs):
    mu = np.random.random(nd)
    mu = mins + mu * ranges
    xd = pm.NormalDistribution(mu[0], 30)
    yd = pm.NormalDistribution(mu[1], 30)
    distrib = pm.ProductDistribution([xd, yd])
    distribs.append(distrib)

pmdata = pm.DataSet()
pmdata.fromArray(data)

m = pm.MixtureModel(ndistribs, np.ones(ndistribs)/ndistribs, distribs)
#m.modelInitialization(pmdata) # this hangs? only for multivariate distribs, works fine for productdistribs
posterior, loglikelihood = m.EM(pmdata, 50, 0.1)

clusteris = m.classify(pmdata, entropy_cutoff=None, silent=True)

ncolours = len(COLOURS)
colouris = clusteris % ncolours
colours = np.asarray(COLOURS)[colouris]

f = figure()
a = gca()
f.canvas.SetBackgroundColour(wx.BLACK)
f.set_facecolor('black')
f.set_edgecolor('black')
a.set_axis_bgcolor('black')
scatter(data[:, 0], data[:, 1], s=1, c=colours, edgecolors='none')
for i, d, in enumerate(m.components):
    xd, yd = d.distList
    e = Ellipse(xy=(xd.mu, yd.mu), width=xd.sigma*2, height=yd.sigma*2, fill=False)
    a.add_artist(e)
    #e.set_clip_box(ax.bbox)
    #e.set_alpha(rand())
    #e.set_facecolor(rand(3))
    c = COLOURS[i % ncolours]
    e.set_edgecolor(c)

# also do uncoloured scatter plot for comparison
f = figure()
a = gca()
f.canvas.SetBackgroundColour(wx.BLACK)
f.set_facecolor('black')
f.set_edgecolor('black')
a.set_axis_bgcolor('black')
scatter(data[:, 0], data[:, 1], s=1, c='white', edgecolors='none')
for i, d, in enumerate(m.components):
    xd, yd = d.distList
    e = Ellipse(xy=(xd.mu, yd.mu), width=xd.sigma*2, height=yd.sigma*2, fill=False)
    a.add_artist(e)
    #e.set_clip_box(ax.bbox)
    #e.set_alpha(rand())
    #e.set_facecolor(rand(3))
    c = COLOURS[i % ncolours]
    e.set_edgecolor(c)
