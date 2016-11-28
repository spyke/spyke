"""Demo how to load and interpret KiloSort results that have been exported to .npy files"""

from __future__ import division
from __future__ import print_function


def intround(x):
    return np.int64(np.round(x))

spiketis = np.load('spike_times.npy').ravel() # integers relative to start of .dat file
nids = np.load('spike_clusters.npy').ravel()
#spiketemplates = np.load('spike_templates.npy') # seems to be identical to nids

# not sure what this is for, might be components, but there's another bigger array for PCs:
#templatefeatures = np.load('template_features.npy') # nspikes x 16, float32

# find maxchan for each template:
templates = np.load('templates.npy') # ntemplates, nt, nchans, Fortran contig
ntemplates, nt, nchans = templates.shape
# reshape templates to ntemplates, nchans, nt by swapping axes (can't just assign new shape!)
templates = np.swapaxes(templates, 1, 2)
templates = np.ascontiguousarray(templates) # make C-contig
nid2maxchani = abs(templates).max(axis=2).argmax(axis=1) # one per template
chans = np.arange(1, 32+1) # A1x32 are 1-based, at least in Blackrock
nid2maxchan = chans[nid2maxchani]
spikemaxchans = nid2maxchan[nids]
#chanmap = np.load('channel_map.npy') # 0 to 31, these come out 0-based from ks
#chanpos = np.load('channel_positions.npy') # (x, y) coords
#nid2maxchan = chanmap[nid2maxchani]
#spikemaxchans = nid2maxchan[nids].ravel() # one per spike
'''
# plot maxchan waveforms of all templates:
figure()
for i, template in enumerate(templates):
    maxchani = nid2maxchani[i]
    plot(template[maxchani, :], '-')
show()
'''
t0 = 3400 # us
sampfreq = 30000 # Hz
spikets = t0 + spiketis/sampfreq*1e6 # us, value to punch into spyke's time box
t_ch_nid = np.column_stack((intround(spikets), spikemaxchans, nids)) # spike time, chan, nid
print(t_ch_nid[:20])
