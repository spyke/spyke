"""
TODO: explicitly test detection of spikes that cross block boundaries

DONE: fix multiple detection of spike on ch3 at ~ t=23600 in file 87

"""

import time
#import timeit
import spyke
import wx
refresh('spyke')
from spyke.detect import BipolarAmplitudeFixedThresh
f = spyke.surf.File(r'C:\data\ptc15\87 - track 7c spontaneous craziness.srf')
#f = spyke.surf.File('/home/mspacek/Desktop/Work/spyke/data/large_data.srf')
f.parse()
wx.Yield()
s = spyke.core.Stream(ctsrecords=f.highpassrecords)
det = BipolarAmplitudeFixedThresh(stream=s)
det.trange = (0, 1000000)
t0 = time.clock()
spikes = det.search()
print 'whole search took %f sec' % (time.clock()-t0)
#timeit.Timer('det.search()', 'from __main__ import det').timeit(1)


>>> spikes[0][:,:20]
array([[ 1480,  2040,  5600,  6880,  9400, 14960, 15120, 21920, 23600, 23640, 23680, 24640, 24760, 25080, 25600, 26720, 26920, 27240, 27560, 28040],
       [   51,    20,    51,    47,    50,    26,    20,    10,    12,    33,     3,    10,    21,    32,    21,    28,     7,    46,     7,    30]],



#import cProfile
#cProfile.run('spikes = det.search()')
