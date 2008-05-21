refresh('spyke')
from spyke.detect import BipolarAmplitudeFixedThresh

f = spyke.surf.File(r'C:\data\ptc15\87 - track 7c spontaneous craziness.srf')
f.parse()
s = spyke.core.Stream(ctsrecords=f.highpassrecords)
det = BipolarAmplitudeFixedThresh(stream=s)

det.trange = (0, 1500000)
spikes = det.search()
