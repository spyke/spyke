import numpy as np
import pyximport
pyximport.install()
from climbing import climb # .pyx file

from pylab import figure, gca, scatter, show
import scipy.io
import time


def makefigure():
    f = figure()
    f.subplots_adjust(0, 0, 1, 1)
    f.set_facecolor('black')
    f.set_edgecolor('black')
    a = gca()
    a.set_axis_bgcolor('black')
    return f


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

#data = np.load('/home/mspacek/data/ptc18/tr1/14-tr1-mseq32_40ms_7deg/2010-05-20_17.18.12_full_scaled_x0_y0_Vpp_t.npy')
#data = data[:50000, :4].copy() # limit npoints and ndims, copy to make it contig
data = np.load('/home/mspacek/data/ptc18/tr1/tr1_chanclust51_ch1,24,25_3PCs.npy')
nd = data.shape[1]
sigma = 0.175 * np.sqrt(nd)
alpha = 1.0
rmergex = 0.25
rneighx = 4
minmovex = 0.00001 # increasing doesn't seem to make much of a speed difference
maxstill = 200 # decreasing doesn't seem to make much of a speed difference
maxnnomerges = 1000
minpoints = 5

'''
import pstats, cProfile
s = """
climb(data, sigma, alpha,
      rneighx=rneighx, rmergex=rmergex,
      minmove=minmove, maxstill=maxstill,
      maxnnomerges=maxnnomerges, minpoints=minpoints)
"""
cProfile.runctx(s, globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
'''
t0 = time.time()
cids, pos = climb(data, sigma, alpha,
                  rneighx=rneighx, rmergex=rmergex,
                  minmovex=minmovex, maxstill=maxstill,
                  maxnnomerges=maxnnomerges, minpoints=minpoints)
print('climb took %.3f sec' % (time.time()-t0))
print cids
print pos
'''
cids, positions = results

nclusters = len(positions)

ncolours = len(COLOURS)
samplecolours = COLOURS[cids % ncolours]
clustercolours = COLOURS[np.arange(nclusters) % ncolours]
#colours[cids == -1] = GREY # unclassified points

# plot x vs y
f = makefigure()
scatter(data[:, 0], data[:, 1], s=1, c=samplecolours, edgecolors='none')
scatter(positions[:, 0], positions[:, 1], c=clustercolours)

if data.shape[1] > 2:
    # plot Vpp vs y
    f = makefigure()
    scatter(data[:, 2], data[:, 1], s=1, c=samplecolours, edgecolors='none')
    scatter(positions[:, 2], positions[:, 1], c=clustercolours)

if data.shape[1] > 3:
    # plot t vs y
    f = makefigure()
    scatter(data[:, 3], data[:, 1], s=1, c=samplecolours, edgecolors='none')
    scatter(positions[:, 3], positions[:, 1], c=clustercolours)


show()
'''
