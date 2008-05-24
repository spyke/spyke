import time

refresh('spyke')
from spyke.detect import BipolarAmplitudeFixedThresh

f = spyke.surf.File(r'C:\data\ptc15\87 - track 7c spontaneous craziness.srf')
f.parse()
s = spyke.core.Stream(ctsrecords=f.highpassrecords)
det = BipolarAmplitudeFixedThresh(stream=s)

det.trange = (0, 10000000)

t0 = time.clock()
spikes = det.search(method='all')
t1 = time.clock()
print t1-t0, 'sec'

# TODO: explicitly test detection of spikes that cross block boundaries
