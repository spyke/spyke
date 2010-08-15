"""Trying out Gaussian mixture modelling using PyMix"""

import numpy as np
from numpy import inf
import scipy.io
import pymix.mixture as pm
from pylab import figure, gca, scatter
from matplotlib.patches import Ellipse, Rectangle
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
nd = data.shape[1]
data = data[:10000] # keep only the 1st 10000 data points for now
mins = data.min(axis=0)
maxs = data.max(axis=0)
ranges = maxs - mins


pos = np.asarray([
[-27, 1305],
[33, 1299],
[-36, 1256],
[-34, 1237],
[27, 1250],
[-35, 1213],
[-36, 1176],
[-38, 1151],
[-38, 1044],
[12, 1036],
[-19, 995],
[-37, 962],
[-23, 925],
[55, 950],
[23, 931],
[-20, 873],
[20, 855],
[23, 800],
[-7, 781],
[-28, 741],
[30, 548],
[-16, 482],
[9, 289],
[-31, 272],
[-35, 167],
[-19, 155],
[30, 151],
[62, 149],
[-35, 110],
[-24, 100],
[38, 102],
[-30, 73],
[-24, 49]])

ndistribs = 50
distribs = []
for mu in pos:
    #mu = np.random.random(nd)
    #mu = mins + mu * ranges
    xd = pm.NormalDistribution(mu[0], 30)
    yd = pm.NormalDistribution(mu[1], 30)
    distrib = pm.ProductDistribution([xd, yd])
    distribs.append(distrib)
# add some extra random ones
for i in range(len(pos), ndistribs):
    mu = np.random.random(nd)
    mu = mins + mu * ranges
    xd = pm.NormalDistribution(mu[0], 30)
    yd = pm.NormalDistribution(mu[1], 30)
    distrib = pm.ProductDistribution([xd, yd])
    distribs.append(distrib)


'''
# add background noise distrib, see 2006 Bar-Hillel
#datamu = data.mean(axis=0)
#datasigma = data.std(axis=0)
#k = 2
#xd = pm.NormalDistribution(datamu[0], k*datasigma[0])
#yd = pm.NormalDistribution(datamu[1], k*datasigma[1])
xmin, ymin = data.min(axis=0)
xmax, ymax = data.max(axis=0)
#xmean, ymean = np.mean([xmin, xmax]), np.mean([ymin, ymax])
width, height = xmax-xmin, ymax-ymin
xd = pm.UniformDistribution(xmin, xmax)
yd = pm.UniformDistribution(ymin, ymax)
distrib = pm.ProductDistribution([xd, yd])
distribs.append(distrib)
compFix = [0] * ndistribs
compFix[-1] = 1 # flag to make last distrib have fixed params
'''

pmdata = pm.DataSet()
pmdata.fromArray(data)

m = pm.MixtureModel(ndistribs, np.ones(ndistribs)/ndistribs, distribs,
                    compFix=None)
#m.modelInitialization(pmdata) # this hangs? only for multivariate distribs, works fine for productdistribs
posterior, loglikelihood = m.EM(pmdata, 50, 0.1)
#posterior, loglikelihood = m.randMaxEM(pmdata, 20, 100, 0.5, silent=False)

cids = m.classify(pmdata, entropy_cutoff=0.5, silent=True)

ncolours = len(COLOURS)
colouris = cids % ncolours
colours = np.asarray(COLOURS)[colouris]
colours[cids == -1] = GREY # unclassified points

f = figure()
a = gca()
f.canvas.SetBackgroundColour(wx.BLACK)
f.set_facecolor('black')
f.set_edgecolor('black')
a.set_axis_bgcolor('black')
scatter(data[:, 0], data[:, 1], s=1, c=colours, edgecolors='none')
for i, d, in enumerate(m.components):
    xd, yd = d.distList
    if xd.__class__ == pm.NormalDistribution:
        e = Ellipse(xy=(xd.mu, yd.mu), width=xd.sigma*2, height=yd.sigma*2, fill=False)
    else:
        e = Rectangle(xy=(xmin, ymin), width=width, height=height, fill=False)
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
    if xd.__class__ == pm.NormalDistribution:
        e = Ellipse(xy=(xd.mu, yd.mu), width=xd.sigma*2, height=yd.sigma*2, fill=False)
    else:
        e = Rectangle(xy=(xmin, ymin), width=width, height=height, fill=False)
    a.add_artist(e)
    #e.set_clip_box(ax.bbox)
    #e.set_alpha(rand())
    #e.set_facecolor(rand(3))
    c = COLOURS[i % ncolours]
    e.set_edgecolor(c)
