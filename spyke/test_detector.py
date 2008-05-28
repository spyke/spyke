"""
TODO: explicitly test detection of spikes that cross block boundaries

TODO: fix multiple detection of spike on ch3 at ~ t=23600 in file 87

"""

import time
import spyke
refresh('spyke')
from spyke.detect import BipolarAmplitudeFixedThresh
f = spyke.surf.File(r'C:\data\ptc15\87 - track 7c spontaneous craziness.srf')
f.parse()
s = spyke.core.Stream(ctsrecords=f.highpassrecords)
det = BipolarAmplitudeFixedThresh(stream=s)
det.trange = (0, 10000000)
t0 = time.clock()
spikes = det.search()
print 'whole search took %f sec' % (time.clock()-t0)

#import cProfile
#cProfile.run('spikes = det.search()')
