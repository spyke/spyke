"""Trying out Gaussian mixture modelling using PyMix"""

import numpy as np
import scipy.io
import pymix.mixture as pm
from pylab import figure, scatter

data = scipy.io.loadmat('/data/ptc18/14_full.mat')
data = data['data']
nd = data.shape[1]
data = data[:10000] # keep only the 1st 10000 data points for now
mins = data.min(axis=0)
maxs = data.max(axis=0)
ranges = maxs - mins

ndistribs = 100
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
#m.modelInitialization(pmdata) # this hangs?
m.EM(pmdata, 50, 0.1)

clusteris = m.classify(pmdata)
figure()
scatter(data[:, 0], data[:, 1], s=1, c=clusteris, edgecolors='none')
