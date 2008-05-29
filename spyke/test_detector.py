"""
TODO: explicitly test detection of spikes that cross block boundaries

TODO: fix multiple detection of spike on ch3 at ~ t=23600 in file 87

"""

import time
import spyke
import wx
refresh('spyke')
from spyke.detect import BipolarAmplitudeFixedThresh
f = spyke.surf.File(r'C:\data\ptc15\87 - track 7c spontaneous craziness.srf')
#f = spyke.surf.File('/home/mspacek/Desktop/Work/spyke/data/large_data.srf')
f.parse()
#wx.Yield()
s = spyke.core.Stream(ctsrecords=f.highpassrecords)
det = BipolarAmplitudeFixedThresh(stream=s)
det.trange = (0, 1000000)
t0 = time.clock()
spikes = det.search()
print 'whole search took %f sec' % (time.clock()-t0)

#import cProfile
#cProfile.run('spikes = det.search()')
    ti:19488
    ti:19273
    ti:21851
    ti:18114
