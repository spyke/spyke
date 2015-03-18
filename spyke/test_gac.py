import numpy as np
import pyximport
pyximport.install(build_in_temp=False, inplace=True)
from gac import gac # .pyx file

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
VIOLET = '#9F3FFF'
MAGENTA = '#FF00FF'
WHITE = '#FFFFFF'
BROWN = '#AF5050'
GREY = '#555555' # reserve as junk cluster colour

COLOURS = np.asarray([RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA, WHITE, BROWN])
'''
data = np.load('/home/mspacek/data/ptc18/tr1/14-tr1-mseq32_40ms_7deg/2010-05-20_17.18.12_full_scaled_x0_y0_Vpp_t.npy')
data = data[:50000, :4].copy() # limit npoints and ndims, copy to make it contig
'''

# real data, just keep the first two dimensions, need to copy to make it contiguous:
data = np.load('/home/mspacek/data/ptc18/tr1/tr1_chanclust51_ch1,24,25_3PCs.npy')[:, :2].copy()
sigma = 0.2

'''
#g = np.float32(np.random.normal(scale=1, size=(100000, 2)))
#np.save('/home/mspacek/Desktop/g.npy', g)
data = np.load('/home/mspacek/Desktop/g.npy')
sigma = 0.2
'''
'''
#x = np.float32(np.random.normal(scale=0.1, size=(100000)))
#y = np.float32(np.random.normal(scale=0.1, size=(100000)))
#c1 = np.column_stack([3*x, y]) # center origin, elongated along x axis
#c2 = np.column_stack([2*x, y+0.5]) # (0, 0.5) origin, elongated along x axis
#g2 = np.vstack([c1, c2])
#plot(g2[:, 0], g2[:, 1], 'k.', ms=1)
#np.save('/home/mspacek/Desktop/g2.npy', g2)
data = np.load('/home/mspacek/Desktop/g2.npy')
sigma = 0.1
'''
'''
#x = np.float32(np.random.normal(scale=0.1, size=(100000)))
#y = np.float32(np.random.normal(scale=0.1, size=(100000)))
#c1 = np.column_stack([x, 2*y]) # center origin, elongated along y axis
#c2 = np.column_stack([2*x+0.5, y+0.5]) # (0.5, 0.5) origin, elongated along x axis
#c3 = np.column_stack([np.sqrt(2)*x+0.5, np.sqrt(2)*x-0.25+y/3]) # (0.5, -0.25) origin, elongated along y=x
#g3 = np.vstack([c1, c2, c3])
#plot(g3[:, 0], g3[:, 1], 'k.', ms=1)
#np.save('/home/mspacek/Desktop/g3.npy', g3)
data = np.load('/home/mspacek/Desktop/g3.npy')
sigma = 0.1
'''

nd = data.shape[1]
#sigma = 0.19#0.175 * np.sqrt(nd)
rmergex = 0.25
rneighx = 4
alpha = 1.0
maxgrad = 1000
minmovex = 0.00001 # increasing doesn't seem to make much of a speed difference
maxnnomerges = 1000
minpoints = 5
'''
import pstats, cProfile
s = """
cids, pos = gac(data, sigma, alpha,
                rneighx=rneighx, rmergex=rmergex,
                minmovex=minmovex,
                maxnnomerges=maxnnomerges, minpoints=minpoints)
"""
cProfile.runctx(s, globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
'''
t0 = time.time()
cids, poshist = gac(data, sigma, rmergex=rmergex, rneighx=rneighx,
                    alpha=alpha, maxgrad=maxgrad, minmovex=minmovex,
                    maxnnomerges=maxnnomerges, minpoints=minpoints)
print('gac took %.3f sec' % (time.time()-t0))
print cids

np.save('/home/mspacek/Desktop/poshist_manhattan_real_sigma_0.2.npy', poshist)
#np.save('/home/mspacek/Desktop/poshist_euclid_real_sigma_0.2.npy', poshist)

#np.save('/home/mspacek/Desktop/poshist_manhattan_real_sigma_0.25.npy', poshist)
#np.save('/home/mspacek/Desktop/poshist_euclid_real_sigma_0.25.npy', poshist)

#np.save('/home/mspacek/Desktop/poshist_manhattan_real_sigma_0.3.npy', poshist)
#np.save('/home/mspacek/Desktop/poshist_euclid_real_sigma_0.3.npy', poshist)

#np.save('/home/mspacek/Desktop/poshist_manhattan_g.npy', poshist)
#np.save('/home/mspacek/Desktop/poshist_euclid_g.npy', poshist)

#np.save('/home/mspacek/Desktop/poshist_manhattan_g2.npy', poshist)
#np.save('/home/mspacek/Desktop/poshist_euclid_g2.npy', poshist)

#np.save('/home/mspacek/Desktop/poshist_manhattan_g3.npy', poshist)
#np.save('/home/mspacek/Desktop/poshist_euclid_g3.npy', poshist)

#import pdb; pdb.set_trace()

'''
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
'''
# correct result:

M=22516.M=15173.M=10333.M=7091.M=4931.M=3460.M=2498.M=1852.M=1396.M=1086.M=850.M=668.M=554.M=458.M=388.M=333.M=280.M=249.M=215.M=183.M=161.M=141.M=130.M=119.M=112.M=103.M=97.M=91.M=78.M=75.M=70.M=66.M=60.M=57.M=52.M=48.M=47.M=45...M=44.M=42.M=39.M=37..M=35.M=34.M=33.M=32.M=30.M=29..M=27..M=26.M=25..M=24..M=23.M=22...M=20..M=19........M=18..M=17....M=16........................M=15...........M=14..............M=13.........M=12.........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
8 points (0.0%) and 8 clusters deleted for having less than 5 points each
nniters: 1138
nclusters: 4
sigma: 0.303, rneigh: 1.212, rmerge: 0.076, alpha: 1.000
nmoving: 0, minmove: 0.000003
still array:
[200, 200, 200, 200, ]
gac took 59.251 sec
[0, 0, 0, ..., 1, 2, 2]
[[ 0.88302857,  0.84062564,  0.48508993],
 [ 0.56020129, -1.09298003,  0.05824601],
 [ 0.18563971,  0.04221672, -1.32369912],
 [-1.45865393,  0.36367384,  0.25685614]]
'''
