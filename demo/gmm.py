"""Trying out Gaussian mixture modelling using PyMix"""

import numpy as np
import scipy.io
import pymix.mixture as pm

data = scipy.io.loadmat('/data/ptc18/14_full.mat')
data = data['data']
nd = data.shape[1]
data = data[:10000] # keep only the 1st 10000 data points for now
mins = data.min(axis=0)
maxs = data.max(axis=0)
ranges = maxs - mins

#sigma = np.array([1, 0], [0, 1])

ndistribs = 35
distribs = []
sigma = np.diag(np.ones(nd) * 5000)
for i in range(ndistribs):
    mu = np.random.random(nd)
    mu = mins + mu * ranges
    #sigma = np.random.random(nd)
    #sigma = -1 + 2*sigma # random diagonal values from -1 to 1
    #sigma = np.diag(sigma) # cov matrix with random diagonal values, and zeros elsewhere
    distribs.append(pm.MultiNormalDistribution(nd, mu, sigma))

pmdata = pm.DataSet()
pmdata.fromArray(data)

m = pm.MixtureModel(ndistribs, np.ones(ndistribs)/ndistribs, distribs)
#m.modelInitialization(pmdata) # this hangs
m.EM(pmdata, 10, 500)
