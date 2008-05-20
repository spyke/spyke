refresh('spyke')
f = spyke.surf.File(r'C:\data\ptc15\87 - track 7c spontaneous craziness.srf')
f.parse()
s = spyke.core.Stream(ctsrecords=f.highpassrecords)
from spyke.detect import BipolarAmplitudeFixedThresh
det = BipolarAmplitudeFixedThresh(stream=s)

spikes = det.search()
