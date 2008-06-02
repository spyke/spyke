"""
TODO: explicitly test detection of spikes that cross block boundaries

"""

import time
#import timeit
import spyke
import wx
refresh('spyke')
from spyke.detect import BipolarAmplitudeFixedThresh
f = spyke.surf.File('/data/ptc15/87 - track 7c spontaneous craziness.srf')
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

# correct result for first 20 spikes in ptc15/87:
array([[ 1480,  2040,  5600,  6880,  9400, 14960, 15120, 21920, 23600,
        23640, 23680, 24640, 24760, 25080, 25600, 26720, 26920, 27240,
        27560, 28040],
       [   51,    20,    51,    47,    50,    26,    20,    10,    12,
           33,     3,    10,    21,    32,    21,    28,     7,    46,
            7,    30]], dtype=int64)



# this one has a not so good trigger on chan 20, then ch33 doesn't seem to be locked out long enough
wavetrange: (38013040, 39015040), cutrange: (38014040, 39014040)
found 10 spikes in total
inside .search() took 0.256 sec
array([[38014360, 38015880, 38016200, 38017360, 38017600, 38017840,
        38017920, 38018520, 38019200, 38021360],
       [      30,       20,       33,       20,        2,       20,
               2,       51,       31,       30]], dtype=int64)

#a = np.zeros((54,25000), dtype=np.int16)
a = np.zeros((54,25000), dtype=np.float32)
timeit.Timer('a += 1', 'from __main__ import a').timeit(100)/100







import cProfile
cProfile.run('spikes = det.search()')
