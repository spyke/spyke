"""Spike sorting classes and frame"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

import os
import sys
import time
import datetime
from copy import copy
import operator

import wx

import numpy as np
#from scipy.cluster.hierarchy import fclusterdata
#import pylab

from spyke.core import WaveForm, Gaussian, MAXLONGLONG, R, toiter
from spyke import wxglade_gui
from spyke.detect import TW, SPIKEDTYPE

MAXCHANTOLERANCE = 100 # um

SPLITTERSASH = 360
SORTSPLITTERSASH = 117
SPIKESORTPANELWIDTHPERCOLUMN = 120
SORTFRAMEHEIGHT = 950

"""
TODO: before extracting features from events, first align all chans wrt maxchan.
Keep tabs on how far and in what direction each chan had to be realigned. Maybe
take sum(abs(phase1V*realignments)) over all chans in the event, (weighted by
amount of signal at phase1 on that chan) and call that another feature.
Events with lots of realignment are more likely BPAPs, or are certainly a different
mode of spike than those with very little realignment. To find the min of each chan
reliably even in noise, find all the local minima within some trange of the maxchan
phase1t (say +/- max allowable dphase/2, ie ~ +/- 175 us) and outside of any preceding
lockouts. Then, take the temporal median of all the local minima found within those constraints,
and align the channel to that. That gives you some confidence about reslilience to noise.
"""

class Sort(object):
    """A spike sorting session, in which you can do multiple Detection runs,
    build Neurons up from spikes in those Detection runs, and then use Neurons
    to sort Spikes.
    Formerly known as a Session, and before that, a Collection.
    A .sort file is a single pickled Sort object"""
    SAVEWAVES = False # save each spike's .wave to .sort file?
    def __init__(self, detector=None, stream=None):
        self.__version__ = 0.1
        self.detector = detector # this Sort's current Detector object
        self.detections = {} # history of detection runs
        self.stream = stream
        self.probe = stream.probe # only one probe design per sort allowed
        self.converter = stream.converter

        # recarray of all spikes detected in this Sort across all Detection runs,
        # whether sorted or not. Each record has a unique .id field
        # Sorted spike IDs also go in their respective Neuron's .spikeis dict
        self.spikes = np.recarray(0, SPIKEDTYPE)
        # most neurons will have an associated cluster, but not necessarily all -
        # some neurons may be purely hand sorted, one spike at a time
        self.neurons = {}
        self.clusters = {} # dict of multidim ellipsoid params
        #self.trash = {} # discarded spikes, disabled, not very useful, adds complexity

        self.uris_sorted_by = 't'
        self.uris_reversed = False

        # how much to scale each dim for better viewing in cluster plots
        self.SCALE = {'x0': 3}

        self._detid = 0 # used to count off unqiue Detection run IDs
        self._sid = 0 # used to count off unique spike IDs
        self._nid = 0 # used to count off unique neuron IDs

    def get_stream(self):
        return self._stream

    def set_stream(self, stream=None):
        """Restore sampfreq and shcorrect to stream when binding/modifying
        stream to self"""
        try:
            stream.sampfreq = self.sampfreq
            stream.shcorrect = self.shcorrect
        except AttributeError:
            pass # either stream is None or self.sampfreq/shcorrect aren't bound
        self._stream = stream

    stream = property(get_stream, set_stream)

    def __getstate__(self):
        """Get object state for pickling"""
        t0 = time.clock()
        # copy it cuz we'll be making changes, this is fast because it's just a shallow copy
        d = self.__dict__.copy()
        # don't pickle the stream, cuz it relies on an open .srf file
        # spikes and wavedata arrays are (potentially) saved separately
        # all the others can be regenerated from the spikes array
        for attr in ['_stream', 'st', 'ris_by_time', 'uris', 'Xcols', 'spikes', 'wavedata']:
            try: del d[attr]
            except KeyError: pass
        return d

    def __setstate__(self, d):
        """Restore self on unpickle per usual, but also restore
        .st, .ris_by_time, and .uris"""
        self.__dict__ = d
        #self.spikes = np.recarray(10, dtype=SPIKEDTYPE)
        #self.update_spike_lists()

    def append_spikes(self, spikes):
        """Append spikes recarray to self.spikes recarray, update associated
        spike lists, and lock down sampfreq and shcorrect attribs"""
        nspikes = len(self.spikes)
        nnewspikes = len(spikes)
        self.spikes.resize(nspikes+nnewspikes, refcheck=False) # resize in-place
        self.spikes[nspikes:] = spikes # append
        self.update_spike_lists()
        try:
            if self.sampfreq != self.stream.sampfreq:
                raise RuntimeError("Sort.sampfreq = %s doesn't match Stream.sampfreq = %s while appending spikes"
                                   % (self.sampfreq, self.stream.sampfreq))
            if self.shcorrect != self.stream.shcorrect:
                raise RuntimeError("Sort.shcorrect = %s doesn't match Stream.shcorrect = %s while appending spikes"
                                   % (self.shcorrect, self.stream.shcorrect))
        except AttributeError: # self's attribs haven't been set yet
            self.sampfreq = self.stream.sampfreq
            self.shcorrect = self.stream.shcorrect
            self.tres = self.stream.tres # for convenience

    def update_spike_lists(self):
        """Update self.st sorted array of all spike times, ris_by_time array,
        and self.uris list containing row indices of unsorted spikes"""
        #spikes = self.spikes.values() # pull list out of dict
        nspikes = len(self.spikes)
        #self.spikes.ri = np.arange(nspikes, dtype=np.uint32) # spike recarray row indices
        st = self.spikes.t # all spike times
        sids = self.spikes.id # all spike ids
        # self.st and self.ris_by_time are required for quick raster plotting
        # can't assume spikes come out of recarray sorted in time
        # (detections may not be in temporal order)
        self.ris_by_time = st.argsort() # recarray indices of all spikes, sorted by time
        self.st = st[self.ris_by_time] # array of current spike times
        self.update_uris() # uris is an array of recarray indices of unsorted spikes

    def update_uris(self):
        """Update uris, which is an array of recarray indices of unsorted spikes,
        used by spike virtual listctrl"""
        nids = self.spikes.nid
        self.uris, = np.where(nids == -1) # -1 indicates spike has no nid assigned to it
        # order it by .uris_sorted_by and .uris_reversed
        if self.uris_sorted_by != 't': self.sort_uris()
        if self.uris_reversed: self.reverse_uris()

    def sort_uris(self, sort_by):
        """Sort recarray row indices of unsorted spikes according to
        sort_by"""
        vals = self.spikes[self.uris][sort_by] # vals from just the unsorted rows and the desired column
        urisis = vals.argsort() # indices into uris, sorted by sort_by
        self.uris = self.uris[urisis] # uris are now sorted by sorty_by
        self.uris_sorted_by = sort_by # update

    def reverse_uris(self):
        """Reverse uris"""
        # is there a way to reverse an array in-place, like a list?
        # maybe swap the start and end points, and set stride to -1?
        self.uris = self.uris[::-1]

    def get_spikes_sortedby(self, attr='id'):
        """Return array of all spikes, sorted by attribute 'attr'"""
        vals = self.spikes[attr]
        ris = vals.argsort()
        spikes = self.spikes[ris]
        return spikes

    def get_wave(self, ri):
        """Return WaveForm corresponding to spikes recarray row ri"""
        spikes = self.spikes
        wave = spikes.wave[ri]
        if wave != None: return wave

        # try self.wavedata ndarray
        det = spikes.detection[ri].detector
        chans = det.chans[spikes.chanis[ri]] # dereference
        try:
            wavedata = self.wavedata[ri]
            t0 = spikes.t0[ri]
            tend = spikes.tend[ri]
            ts = np.arange(t0, tend, self.tres) # build them up
            # only include data relevant to this spike
            wavedata = wavedata[0:len(chans), 0:len(ts)]
            return WaveForm(data=wavedata, ts=ts, chans=chans)
        except IndexError: pass

        # try getting it from the stream
        if self.stream == None:
            raise RuntimeError("No stream open, can't get wave for %s %d" %
                               (spikes[ri], spikes[ri].id))
        if det.srffname != self.stream.srffname:
            msg = ("Spike %d was extracted from .srf file %s.\n"
                   "The currently opened .srf file is %s.\n"
                   "Can't get spike %d's wave" %
                   (spikes[ri].id, det.srffname, self.stream.srffname, spikes[ri].id))
            wx.MessageBox(msg, caption="Error", style=wx.OK|wx.ICON_EXCLAMATION)
            raise RuntimeError(msg)
        wave = self.stream[spikes.t0[ri] : spikes.tend[ri]]
        return wave[chans]

    def get_param_matrix(self, dims=None):
        """Organize parameters in dims from all spikes into a
        data matrix for clustering"""
        t0 = time.clock()
        nspikes = len(self.spikes)
        nparams = len(dims)
        try:
            # self.Xcols stores all currently created columns of any potential param matrix X
            assert len(self.Xcols.values()[0]) == nspikes
        except (AttributeError, AssertionError):
            # not created yet, or change in number of spikes
            self.Xcols = {}

        try:
            for dim in dims:
                #print('asserting dim %r is in Xcols' % dim)
                assert dim in self.Xcols
        except AssertionError: # column is missing
            #spikes = self.get_spikes_sortedby('id')
            for dim in dims:
                if dim not in self.Xcols: # add missing column
                    #self.Xcols[dim] = np.asarray([ s[dim] for s in spikes ], dtype=np.float32)
                    if dim in self.SCALE:
                        # scale this dim appropriately
                        self.Xcols[dim] = self.SCALE[dim] * np.float32(self.spikes[dim])
                    else:
                        self.Xcols[dim] = np.float32(self.spikes[dim])

        X = np.column_stack([ self.Xcols[dim] for dim in dims ])
        print("Getting param matrix took %.3f sec" % (time.clock()-t0))
        return X

    def apply_cluster(self, cluster):
        """Apply cluster to spike data - calculate which spikes fall within the
        cluster's multidimensional ellipsoid. Return spikes recarray row indices"""

        # consider all the dimensions in this cluster that have non-zero scale
        dims = [ dim for dim, val in cluster.scale.items() if val != 0 ]
        X = self.get_param_matrix(dims=dims)

        # To find which points fall within the ellipsoid, need to do the inverse of all
        # the operations that translate and rotate the ellipsoid, in the correct order.
        # Need to do those operations on the points, not on the ellipsoid parameters.
        # That way, we can figure out which points to pick out, and then we
        # pick them out of the original set of unmodified points

        # undo the translation, in place
        dim2coli = {}
        for i, dim in enumerate(dims):
            X[:, i] -= cluster.pos[dim]
            dim2coli[dim] = i # build up dim to X column index mapping while we're at it

        # build up dict of groups of rotations which not only have the same set of 3 dims,
        # but are also ordered according to the right hand rule
        rotgroups = {} # key is ordered tuple of dims, value is list of corresponding ori values
        nonrotdims = copy(dims) # dims that aren't in any rotated projection, init to all dims and remove one by one
        for dim in dims:
            val = cluster.ori[dim]
            if val == {}: # this dim has an empty orientation value
                continue
            for reldims, ori in val.items():
                rotdim = dim, reldims[0], reldims[1] # tuple of rotated dim and the (ordered) 2 dims it was done wrt
                for d in rotdim:
                    try: nonrotdims.remove(d) # remove dim from nonrotdims
                    except ValueError: pass
                if rotdim in rotgroups: # is rotdim already in rotgroups?
                    raise RuntimeError("can't have more than one rotation value for a given dim and its relative dims")
                elif (rotdim[2], rotdim[0], rotdim[1]) in rotgroups: # same set of dims exist, but are rotated around rotdim[2]
                    rotgroups[(rotdim[2], rotdim[0], rotdim[1])][1] = ori
                elif (rotdim[1], rotdim[2], rotdim[0]) in rotgroups: # same set of dims exist, but are rotated around rotdim[1]
                    rotgroups[(rotdim[1], rotdim[2], rotdim[0])][2] = ori
                else: # no ring permutation of these dims is in rotgroups, add it
                    rotgroups[rotdim] = [ori, 0, 0] # leave the other two slots available for ring permutations

        # TODO: check at start of method to make sure if a cluster.ori[dim] != {},
        # that its entries all have non-zero ori values. This might not be the case
        # if an ori value is set to something, then set back to 0 in the GUI. Fixing
        # this will stop unnecessary plugging in of values into the rotation matrix,
        # though this isn't really a big performance hit it seems

        # First take the non-oriented dims, however many there are, and plug them
        # into the ellipsoid eq'n. That'll init your trutharray. Then go
        # through each rotgroup in succession and AND its results to the trutharray,
        # and you're done!
        sumterms = np.zeros(len(X)) # ellipsoid eq'n sum of terms over non-rotated dimensions
        for dim in nonrotdims:
            x = X[:, dim2coli[dim]] # pull correct column out of X for this non-rotated dim
            A = cluster.scale[dim]
            sumterms += x**2/A**2
        trutharray = (sumterms <= 1) # which points in nonrotdims space fall inside the ellipsoid?

        # for each rotation group, undo the rotation by taking product of inverse of
        # rotation matrix (which == its transpose) with the detranslated points
        for rotdims, oris in rotgroups.items():
            Xrot = np.column_stack([ X[:, dim2coli[dim]] for dim in rotdims ]) # pull correct columns out of X for this rotgroup
            Xrot = (R(oris[0], oris[1], oris[2]).T * Xrot.T).T
            Xrot = np.asarray(Xrot) # convert from np.matrix back to np.array to prevent from taking matrix power
            # which points are inside the ellipsoid?
            x = Xrot[:, 0]; A = cluster.scale[rotdims[0]]
            y = Xrot[:, 1]; B = cluster.scale[rotdims[1]]
            z = Xrot[:, 2]; C = cluster.scale[rotdims[2]]
            trutharray *= (x**2/A**2 + y**2/B**2 + z**2/C**2 <= 1) # AND with interior points from any previous rotgroups

        # spikes row indices of points that fall within ellipsoids of all rotgroups
        ris, = np.where(trutharray)
        #assert len(i) > 0, "no points fall within the ellipsoid"
        #Xin = X[i] # pick out those points
        #spikes = np.asarray(self.get_spikes_sortedby('id'))[i]
        spikeis = self.spikes.id[ris]
        return spikeis
    '''
    def get_component_matrix(self, dims=None, weighting=None):
        """Convert spike param matrix into pca/ica data for clustering"""

        import mdp # can't delay this any longer
        X = self.get_param_matrix(dims=dims)
        if weighting == None:
            return X
        if weighting.lower() == 'ica':
            node = mdp.nodes.FastICANode()
        elif weighting.lower() == 'pca':
            node = mdp.nodes.PCANode()
        else:
            raise ValueError, 'unknown weighting %r' % weighting
        node.train(X)
        features = node.execute(X) # returns all available components
        #self.node = node
        #self.weighting = weighting
        #self.features = features
        return features
    '''
    def create_neuron(self):
        """Create and return a new Neuron with a unique ID"""
        neuron = Neuron(self, self._nid)
        self._nid += 1 # inc for next unique neuron
        self.neurons[neuron.id] = neuron # add neuron to self
        return neuron

    def get_ids(self, cids, spikes):
        """Convert a list of cluster ids into 2 dicts: n2sids maps neuron IDs to
        spike IDs; s2nids maps spike IDs to neuron IDs"""
        cids = np.asarray(cids)
        cids = cids - cids.min() # make sure cluster IDs are 0-based
        uniquecids = set(cids)
        nclusters = len(uniquecids)
        # neuron ID to spike IDs (plural) mapping
        n2sids = dict(zip(uniquecids, [ [] for i in range(nclusters) ]))
        s2nids = {} # spike ID to neuron ID mapping
        for spike, nid in zip(spikes, cids):
            s2nids[spike.id] = nid
            n2sids[nid].append(spike.id)
        return n2sids, s2nids

    def write_spc_input(self):
        """Generate input data file to SPC"""
        X = self.get_component_matrix()
        # write to space-delimited .dat file. Each row is a spike, each column a param
        spykedir = os.path.dirname(__file__)
        dt = str(datetime.datetime.now())
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        self.spcdatfname = os.path.join(spykedir, 'spc', dt+'.dat')
        self.spclabfname = os.path.join(spykedir, 'spc', dt+'.dg_01.lab') # not sure why spc adds the dg_01 part
        f = open(self.spcdatfname, 'w')
        for params in X: # write text data to file, one row at a time
            params.tofile(f, sep='  ', format='%.6f')
            f.write('\n')
        f.close()

    def parse_spc_lab_file(self, fname=None):
        """Parse output .lab file from SPC. Each row in the file is the assignment of each spin
        (datapoint) to a cluster, one row per temperature datapoint. First column is temperature
        run number (0-based). 2nd column is the temperature. All remaining columns correspond
        to the datapoints in the order presented in the input .dat file. Returns (Ts, cids)"""
        #spikes = self.get_spikes_sortedby('id')
        if fname == None:
            dlg = wx.FileDialog(None, message="Open SPC .lab file",
                                defaultDir=r"C:\Documents and Settings\Administrator\Desktop\Charlie\From", defaultFile='',
                                wildcard="All files (*.*)|*.*|.lab files (*.lab)|*.lab|",
                                style=wx.OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                fname = dlg.GetPath()
            dlg.Destroy()
        data = np.loadtxt(fname, dtype=np.float32)
        Ts = data[:, 1] # 2nd column
        cids = np.int32(data[:, 2:]) # 3rd column on
        print('Parsed %r' % fname)
        return Ts, cids

    def parse_charlies_output(self, fname=r'C:\Documents and Settings\Administrator\Desktop\Charlie\From\2009-07-20\clustered_events_coiflet_T0.125.txt'):
        nids = np.loadtxt(fname, dtype=int) # one neuron id per spike
        return nids

    def write_spc_app_input(self):
        """Generate input data file to spc_app"""
        spikes = self.get_spikes_sortedby('id')
        X = self.get_component_matrix()
        # write to tab-delimited data file. Each row is a param, each column a spike (this is the transpose of X)
        # first row has labels "AFFX", "NAME", and then spike ids
        # first col has labels "AFFX", and then param names
        f = open(r'C:\home\mspacek\Desktop\Work\SPC\Weizmann\spc_app\spc_app_input.txt', 'w')
        f.write('AFFX\tNAME\t')
        for spike in spikes:
            f.write('s%d\t' % spike.id)
        f.write('\n')
        for parami, param in enumerate(['Vpp', 'dphase', 'x0', 'y0', 'sx', 'sy', 'theta']):
            f.write(param+'\t'+param+'\t')
            for val in X[:, parami]:
                f.write('%f\t' % val)
            f.write('\n')
        f.close()

    def hcluster(self, t=1.0):
        """Hierarchically cluster self.spikes

        TODO: consider doing multiple cluster runs. First, cluster by spatial location (x0, y0).
        Then split those clusters up by Vpp. Then those by spatial distrib (sy/sx, theta),
        then by temporal distrib (dphase, s1, s2). This will ensure that the lousier params will
        only be considered after the best ones already have, and therefore that you start off
        with pretty good clusters that are then only slightly refined using the lousy params
        """
        spikes = self.get_spikes_sortedby('id')
        X = self.get_component_matrix()
        print X
        cids = fclusterdata(X, t=t, method='single', metric='euclidean') # try 'weighted' or 'average' with 'mahalanobis'
        n2sids, s2nids = self.get_ids(cids, spikes)
        return n2sids

    def export2Charlie(self, fname='spike_data', onlymaxchan=False, nchans=3, npoints=32):
        """Export spike data to a text file, one spike per row.
        Columns are x0, y0, followed by most prominent npoints datapoints
        (1/4, 3/4 wrt spike time) of each nearest nchans. This is to
        give to Charlie to do WPD and SPC on"""
        if onlymaxchan:
            nchans = 1
        assert np.log2(npoints) % 1 == 0, 'npoints is not a power of 2'
        # get ti - time index each spike is assumed to be centered on
        self.spikes[0].update_wave(self.stream) # make sure it has a wave
        ti = int(round(self.spikes[0].wave.data.shape[-1] / 4)) # 13 for 50 kHz, 6 for 25 kHz
        dims = len(self.spikes), 2+nchans*npoints
        output = np.empty(dims, dtype=np.float32)
        dm = self.detector.dm
        chanis = np.arange(len(dm.data))
        coords = np.asarray(dm.coords)
        xcoords = coords[:, 0]
        ycoords = coords[:, 1]
        spikeis = self.spikes.keys() # self.spikes is a dict!
        spikeis.sort()
        for spikei in spikeis:
            spike = self.spikes[spikei]
            chani = spike.chani # max chani
            x0, y0 = spike.x0, spike.y0
            if onlymaxchan:
                nearestchanis = np.asarray([chani])
            else:
                # find closest chans to x0, y0
                d2s = (xcoords - x0)**2 + (ycoords - y0)**2 # squared distances
                sortis = d2s.argsort()
                nearestchanis = chanis[sortis][0:nchans] # pick the first nchan nearest chans
                if chani not in nearestchanis:
                    print("WARNING: max chani %d is not among the %d chanis nearest "
                          "(x0, y0) = (%.1f, %.1f) for spike %d at t=%d"
                          % (chani, nchans, x0, y0, spikei, spike.t))
            if spike.wave.data == None:
                spike.update_wave(self.stream)
            row = [x0, y0]
            for chani in nearestchanis:
                chan = dm.chans[chani] # dereference
                try:
                    data = spike.wave[chan].data[0] # pull out singleton dimension
                except IndexError: # empty array
                    data = np.zeros(data.shape[-1], data.dtype)
                row.extend(data[ti-npoints/4:ti+npoints*3/4])
            output[spikei] = row
        dt = str(datetime.datetime.now())
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        fname += '.' + dt + '.txt'
        np.savetxt(fname, output, fmt='%.1f', delimiter=' ')

    '''
    def match(self, templates=None, weighting='signal', sort=True):
        """Match templates to all .spikes with nearby maxchans,
        save error values to respective templates.

        Note: slowest step by far is loading in the wave data from disk.
        (First match is slow, subsequent ones are ~ 15X faster.)
        Unless something's done about that in advance, don't bother optimizing here much.
        Right now, once waves are loaded, performance is roughly 20000 matches/sec

        TODO: Nick's alternative to gaussian distance weighting: have two templates: a mean template, and an stdev
        template, and weight the error between each matched spike and the mean on each chan at each timepoint by
        the corresponding stdev value (divide the error by the stdev, so that timepoints with low stdev are more
        sensitive to error)

        TODO: looks like I still need to make things more nonlinear - errors at high signal values aren't penalized enough,
        while errors at small signal values are penalized too much. Try cubing both signals, then taking sum(err**2)

        DONE: maybe even better, instead of doing an elaborate cubing of signal, followed by a rather elaborate
        gaussian spatiotemporal weighting of errors, just take difference of signals, and weight the error according
        to the abs(template_signal) at each point in time and across chans. That way, error in parts of the signal far from
        zero are considered more important than deviance of perhaps similar absolute value for signal close to zero

        """
        templates = templates or self.templates.values() # None defaults to matching all templates
        sys.stdout.write('matching')
        t0 = time.clock()
        nspikes = len(self.spikes)
        dm = self.detector.dm
        for template in templates:
            template.err = [] # overwrite any existing .err attrib
            tw = template.tw
            templatewave = template.wave[template.chans] # pull out template's enabled chans
            #stdev = template.get_stdev()[template.chans] # pull out template's enabled chans
            #stdev[stdev == 0] = 1 # replace any 0s with 1s - TODO: what's the best way to avoid these singularities?
            weights = template.get_weights(weighting=weighting, sstdev=self.detector.slock/2,
                                           tstdev=self.detector.tlock/2) # Gaussian weighting in space and/or time
            for spike in self.spikes.values():
                # check if spike.maxchan is outside some minimum distance from template.maxchan
                if dm[template.maxchan, spike.maxchan] > MAXCHANTOLERANCE: # um
                    continue # don't even bother
                if spike.wave.data == None or template.tw != TW: # make sure their data line up
                    spike.update_wave(tw) # this slows things down a lot, but is necessary
                # slice template's enabled chans out of spike, calculate sum of squared weighted error
                # first impression is that dividing by stdev makes separation worse, not better
                #err = (templatewave.data - spike.wave[template.chans].data) / stdev * weights # low stdev means more sensitive to error
                spikewave = spike.wave[template.chans] # pull out template's enabled chans from spike
                if weighting == 'signal':
                    weights = np.abs(np.asarray([templatewave.data, spikewave.data])).max(axis=0) # take elementwise max of abs of template and spike data
                err = (templatewave.data - spikewave.data) * weights # weighted error
                err = (err**2).sum(axis=None) # sum of squared weighted error
                template.err.append((spike.id, int(round(err))))
            template.err = np.asarray(template.err, dtype=np.int64)
            if sort and len(template.err) != 0:
                i = template.err[:, 1].argsort() # row indices that sort by error
                template.err = template.err[i]
            sys.stdout.write('.')
        print '\nmatch took %.3f sec' % (time.clock()-t0)
    '''

class Neuron(object):
    """A collection of spikes that have been deemed somehow, whether manually
    or automatically, to have come from the same cell. A Neuron's waveform
    is the mean of its member spikes"""
    def __init__(self, sort, id=None):
        self.sort = sort
        self.id = id # neuron id
        self.wave = WaveForm() # init to empty waveform
        self.spikeis = set() # indices of spikes that make up this neuron
        self.t = 0 # relative reference timestamp, here for symmetry with fellow spike rec (obj.t comes up sometimes)
        self.plt = None # Plot currently holding self
        #self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        self.cluster = None
        #self.srffname # not here, let's allow neurons to have spikes from different files?
    '''
    def get_spikes(self):
        return self.sort.si2spikes(self.spikeis)

    spikes = property(get_spikes)
    '''
    def update_wave(self, stream):
        """Update mean waveform, should call this every time .spikes are modified.
        Setting .spikes as a property to do so automatically doesn't work, because
        properties only catch name binding of spikes, not modification of an object
        that's already been bound"""
        spikes = self.sort.spikes
        if len(self.spikeis) == 0: # no member spikes, perhaps I should be deleted?
            raise RuntimeError("neuron %d has no spikes and its waveform can't be updated" % self.id)
            #self.wave = WaveForm() # empty waveform
            #return self.wave
        spikeis = np.asarray(list(self.spikeis))
        ris = spikes.id.searchsorted(spikeis)

        # build up union of chans and relative timepoints of all member spikes
        chans, ts = set(), set()
        for ri in ris:
            det = spikes[ri].detection.detector
            chans.update(det.chans[spikes[ri].chanis])
            spikets = np.arange(spikes[ri].t0, spikes[ri].tend, self.sort.tres) # build them up
            ts.update(spikets - spikes[ri].t) # timepoints wrt spike time, not absolute
        chans = np.asarray(list(chans))
        ts = np.asarray(list(ts))
        chans.sort() # Neuron's chans are a sorted union of chans of all its member spikes
        ts.sort() # ditto for timepoints

        # take mean of chans of data from spikes with potentially different
        # chans and time windows wrt their spike
        shape = len(chans), len(ts)
        data = np.zeros(shape, dtype=np.float32) # collect data that corresponds to chans and ts
        nspikes = np.zeros(shape, dtype=np.uint32) # nspikes that have contributed to each point in data
        for ri in ris:
            wave = self.sort.get_wave(ri)
            #spikes[ri].wave = wave # bind to spike
            wavedata = wave.data
            det = spikes[ri].detection.detector
            spikechans = det.chans[spikes[ri].chanis]
            spikets = np.arange(spikes[ri].t0, spikes[ri].tend, self.sort.tres)
            # get chan indices into chans corresponding to spikechans, chans is a superset of spikechans
            chanis = chans.searchsorted(spikechans)
            # get timepoint indices into ts corresponding to wave.ts timepoints relative to their spike time
            tis = ts.searchsorted(spikets - spikes[ri].t)
            # there must be an easier way of doing the following:
            rowis = np.tile(False, len(chans))
            rowis[chanis] = True
            colis = np.tile(False, len(ts))
            colis[tis] = True
            i = np.outer(rowis, colis) # 2D boolean array for indexing into data
            # this method doesn't work, destination indices are assigned to in the wrong order:
            '''
            rowis = np.tile(chanis, len(tis))
            colis = np.tile(tis, len(chanis))
            i = rowis, colis
            '''
            # accumulate appropriate data points (add int16 to float32, keep as AD units)
            data[i] += wavedata.ravel()
            nspikes[i] += 1 # increment spike counts at appropriate data points
        # some entries in nspikes can be 0 - this raises an 'invalid' error instead
        # of a div by 0 error because those same entries in data are also 0, so we
        # get 0/0. This can be dealt with by temporarily ignoring invalid errors
        # from numpy, using np.seterr. Or instead, we can replace all the zeros in
        # nspikes with 1s, and get 0/1 which wouldn't raise any errors
        nspikes = np.maximum(nspikes, np.ones(shape, dtype=np.float32)) # element-wise max
        #np.seterr(invalid='ignore')
        data /= nspikes # normalize each data point appropriately
        #np.seterr(invalid='raise') # restore error level
        self.wave.data = data
        self.wave.chans = chans
        self.wave.ts = ts
        #print('neuron[%d].wave.chans = %r' % (self.id, chans))
        #print('neuron[%d].wave.ts = %r' % (self.id, ts))
        return self.wave
    '''
    def get_stdev(self):
        """Return 2D array of stddev of each timepoint of each chan of member spikes.
        Assumes self.update_wave has already been called"""
        data = []
        # TODO: speed this up by pre-allocating memory and then filling in the array
        for spike in self.spikes:
            data.append(spike.wave.data) # collect spike's data
        stdev = np.asarray(data).std(axis=0)
        return stdev
    '''

    def get_chans(self):
        return self.wave.chans # self.chans just refers to self.wave.chans

    chans = property(get_chans)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        d['plt'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        d.pop('itemID', None) # remove tree item ID, if any, since that'll have changed anyway on unpickle
        return d

    '''
    def get_maxchan(self):
        """Find maxchan at t=0 in mean waveform, constrained to enabled chans

        Notes:
            - don't recenter self.chans on maxchan, leave actual chan selection to user
            - maxchan should however be constrained to currently enabled chans
        """
        if self.wave.data == None or self.chans == None:
            return None
        data = self.wave.data
        ts = self.wave.ts
        t0i, = np.where(ts == 0) # find column index that corresponds to t=0
        assert len(t0i) == 1 # make sure there's only one reference timepoint
        maxchani = abs(data[self.chans, t0i]).argmax() # find index into self.chans with greatest abs(signal) at t=0
        #maxchani = abs(data).max(axis=1).argmax() # ignore sign, find max across columns, find row with greatest max
        maxchan = self.chans[maxchani] # dereference
        return maxchan
    '''
    '''
    def get_tw(self):
        return self._tw

    def set_tw(self, tw):
        """Reset self's time range relative to t=0 spike time,
        update slice of member spikes, and update mean waveform"""
        self._tw = tw
        for spike in self.spikes.values():
            spike.update_wave(tw=tw)
        self.update_wave()

    tw = property(get_tw, set_tw)

    def get_weights(self, weighting=None, sstdev=None, tstdev=None):
        """Returns unity, spatial, temporal, or spatiotemporal Gaussian weights
        for self's enabled chans in self.wave.data, given spatial and temporal
        stdevs"""
        nchans = len(self.wave.chans)
        nt = len(self.wave.data[0]) # assume all chans have the same number of timepoints
        if weighting == None:
            weights = 1
        elif weighting == 'spatial':
            weights = self.get_gaussian_spatial_weights(sstdev) # vector
        elif weighting == 'temporal':
            weights = self.get_gaussian_temporal_weights(tstdev) # vector
        elif weighting == 'spatiotemporal':
            sweights = self.get_gaussian_spatial_weights(sstdev)
            tweights = self.get_gaussian_temporal_weights(tstdev)
            weights = np.outer(sweights, tweights) # matrix, outer product of the two
        elif weighting == 'signal':
            weights = None # this is handled by caller
        #print '\nweights:\n%r' % weights
        return weights

    def get_gaussian_spatial_weights(self, stdev):
        """Return a vector that weights self.chans according to a 2D gaussian
        centered on self.maxchan with standard deviation stdev in um"""
        g = Gaussian(mean=0, stdev=stdev)
        d = self.sort.detector.dm[self.maxchan, self.chans] # distances between maxchan and all enabled chans
        weights = g[d]
        weights.shape = (-1, 1) # vertical vector with nchans rows, 1 column
        return weights

    def get_gaussian_temporal_weights(self, stdev):
        """Return a vector that weights timepoints in self's mean waveform
        by a gaussian centered on t=0, with standard deviation stdev in us"""
        g = Gaussian(mean=0, stdev=stdev)
        ts = self.wave.ts # template mean timepoints relative to t=0 spike time
        weights = g[ts] # horizontal vector with 1 row, nt timepoints
        return weights

    def __del__(self):
        """Is this run on 'del template'?"""
        for spike in self.spikes:
            spike.template = None # remove self from all spike.template fields

    def pop(self, spikeid):
        return self.spikes.pop(spikeid)
    '''

'''
class Match(object):
    """Holds all the settings of a match run. A match run is when you compare each
    template to all of the detected but unsorted spikes in Sort.spikes, plot an
    error histogram for each template, and set the error threshold for each to
    decide which spikes match the template. Fast, simple, no noise spikes to worry
    about, but is susceptible to spike misalignment. Compare with a Rip"""

    def match(self):
        pass


class Rip(object):
    """Holds all the Rip settings. A rip is when you take each template and
    slide it across the entire file. A spike is detected and
    sorted at timepoints where the error between template and file falls below
    some threshold. Slow, and requires distinguishing a whole lotta noise spikes"""

    def rip(self):
        pass


class MatchRip(Match, Rip):
    """A hybrid of the two. Rip each template across all of the unsorted spikes
    instead of across the entire file. Compared to a Match, a MatchRip can better
    handle unsorted unspikes that are misaligned, with the downside that you now
    have a lot of noise spikes to distinguish as well, but not as many as in a normal Rip"""

    def matchrip(self):
        pass
'''

class SortFrame(wxglade_gui.SortFrame):
    """Sort frame"""
    def __init__(self, *args, **kwargs):
        wxglade_gui.SortFrame.__init__(self, *args, **kwargs)
        self.spykeframe = self.Parent
        ncols = self.sort.probe.ncols
        size = (SPLITTERSASH + SPIKESORTPANELWIDTHPERCOLUMN * ncols,
                SORTFRAMEHEIGHT)
        self.SetSize(size)
        self.splitter.SetSashPosition(SPLITTERSASH) # do this here because wxGlade keeps messing this up
        self.sort_splitter.SetSashPosition(SORTSPLITTERSASH)

        self.listTimer = wx.Timer(owner=self.list)

        self.list.lastSelectedIDs = []
        self.tree.lastSelectedItems = []
        self.tree.selectedItemIDs = []
        self.tree.selectedItems = []

        columnlabels = ['sID', 'x0', 'y0', 'time'] # spike list column labels
        for coli, label in enumerate(columnlabels):
            self.list.InsertColumn(coli, label)
        #for coli in range(len(columnlabels)): # this needs to be in a separate loop it seems
        #    self.list.SetColumnWidth(coli, wx.LIST_AUTOSIZE_USEHEADER) # resize columns to fit
        # hard code column widths for precise control, autosize seems buggy
        for coli, width in {0:40, 1:40, 2:60, 3:80}.items(): # (sid, x0, y0, time)
            self.list.SetColumnWidth(coli, width)

        self.list.Bind(wx.EVT_TIMER, self.OnListTimer)
        self.list.Bind(wx.EVT_RIGHT_DOWN, self.OnListRightDown)
        self.list.Bind(wx.EVT_KEY_DOWN, self.OnListKeyDown)
        #self.tree.Bind(wx.EVT_LEFT_DOWN, self.OnTreeLeftDown) # doesn't fire when clicking on non focused item, bug #4448
        self.tree.Bind(wx.EVT_LEFT_UP, self.OnTreeLeftUp) # need this to catch clicking on non focused item, bug #4448
        self.tree.Bind(wx.EVT_RIGHT_DOWN, self.OnTreeRightDown)
        self.tree.Bind(wx.EVT_KEY_DOWN, self.OnTreeKeyDown)
        self.tree.Bind(wx.EVT_KEY_UP, self.OnTreeKeyUp)
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnTreeSelectChanged)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def get_sort(self):
        return self.spykeframe.sort

    def set_sort(self):
        raise RuntimeError("SortFrame's .sort not settable")

    sort = property(get_sort, set_sort) # make this a property for proper behaviour after unpickling

    def OnSize(self, evt):
        """Re-save reflines_background after resizing the frame"""
        # resize doesn't actually happen until after this handler exits,
        # so we have to CallAfter
        wx.CallAfter(self.DrawRefs)
        evt.Skip()

    def OnSplitterSashChanged(self, evt):
        """Re-save reflines_background after resizing the SortPanel(s)
        with the frame's primary splitter"""
        print('in OnSplitterSashChanged')
        wx.CallAfter(self.DrawRefs)

    def OnClose(self, evt):
        # remove 'Frame' from class name
        frametype = self.__class__.__name__.lower().replace('frame', '')
        self.spykeframe.HideFrame(frametype)

    def OnListSelect(self, evt):
        """Restart list selection timer
        listTimer explanation: on any selection event, start or restart the timer for say
        1 msec. Then, when timer runs down, run self.list.GetSelections() and compare to
        previous list of selections, and execute your plots accordingly. This makes all
        the sel event handling fast"""
        self.listTimer.Stop()
        # only fire one timer event after specified interval
        self.listTimer.Start(milliseconds=1, oneShot=True)

    def OnListDeselect(self, evt):
        self.OnListSelect(evt)

    def OnListTimer(self, evt):
        """Run when started timer runs out and triggers a TimerEvent"""
        sort = self.sort
        selectedRows = self.list.GetSelections()
        ris = sort.uris[selectedRows]
        sids = sort.spikes.id[ris]
        remove_sids = [ sid for sid in self.list.lastSelectedIDs if sid not in sids ]
        add_sids = [ sid for sid in sids if sid not in self.list.lastSelectedIDs ]
        self.RemoveItemsFromPlot([ 's'+str(sid) for sid in remove_sids ])
        self.AddItems2Plot([ 's'+str(sid) for sid in add_sids ])
        self.list.lastSelectedIDs = sids # save for next time

    def OnListRightDown(self, evt):
        """Toggle selection of the clicked list item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection ListEvent happening before the MouseEvent, or something"""
        print('in OnListRightDown')
        pt = evt.GetPosition()
        row, flags = self.list.HitTest(pt)
        sid = self.sort.spikes.id[self.sort.uris[row]]
        print('spikeID is %r' % id)
        # this would be nice, but doesn't work (?) cuz apparently somehow the
        # selection ListEvent happens before MouseEvent that caused it:
        #selected = not self.list.IsSelected(row)
        #self.list.Select(row, on=int(not selected))
        # here is a yucky workaround:
        try:
            self.spikesortpanel.used_plots['s'+str(id)] # is it plotted?
            selected = True # if so, item must be selected
            print('spike %d in used_plots' % id)
        except KeyError:
            selected = False # item is not selected
            print('spike %d not in used_plots' % id)
        self.list.Select(row, on=not selected) # toggle selection, this fires sel spike, which updates the plot

    def OnListColClick(self, evt):
        """Sort .uris according to column clicked.

        TODO: keep track of currently selected spikes and currently focused spike,
        clear the selection, then reselect those same spikes after sorting is done,
        and re-focus the same spike. Scroll into view of the focused spike (maybe
        that happens automatically). Right now, the selection remains in the list
        as-is, regardless of the entries that change beneath it"""
        col = evt.GetColumn()
        field = self.list.COL2FIELD[col]
        s = self.sort
        # for speed, check if already sorted by field
        if s.uris_sorted_by == field: # already sorted, reverse the order
            s.reverse_uris()
            s.uris_reversed = not s.uris_reversed # update reversed flag
        else: # not yet sorted by field
            s.sort_uris(field)
            s.uris_sorted_by = field # update
            s.uris_reversed = False # update
        self.list.RefreshItems()

    def OnListKeyDown(self, evt):
        """Spike list key down evt"""
        key = evt.GetKeyCode()
        if key == wx.WXK_TAB:
            self.tree.SetFocus() # change focus to tree
        elif key in [wx.WXK_SPACE, wx.WXK_RETURN]:
            self.list.ToggleFocusedItem()
            return # evt.Skip() seems to prevent toggling, or maybe it untoggles
        elif key in [ord('A'), wx.WXK_LEFT]:
            self.MoveCurrentSpikes2Neuron(which='selected')
        elif key in [ord('C'), ord('N')]: # wx.WXK_SPACE doesn't seem to work
            self.MoveCurrentSpikes2Neuron(which='new')
        elif key in [wx.WXK_DELETE, ord('D')]:
            #self.MoveCurrentSpikes2Trash()
            print('individual spike deletion disabled, not very useful feature')
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        evt.Skip()

    def OnTreeSelectChanged(self, evt=None):
        """Due to bugs #2307 and #626, a SEL_CHANGED event isn't fired when
        (de)selecting the currently focused item in a tree with the wx.TR_MULTIPLE
        flag set, as it is here. So, this handler has to be called manually on mouse
        and keyboard events. Don't think there's any harm in calling this handler twice
        for all other cases where a SEL_CHANGED event is fired"""
        print('in OnTreeSelectChanged')
        if evt: # just use this as a hack to update currently focused
            item = evt.GetItem()
            if item:
                self.tree._focusedItem = item
                print('currently focused item: %s' % self.tree.GetItemText(item))
            return # don't allow the selection event to actually happen?????????????
        selectedItemIDs = self.tree.GetSelections()
        selectedItems = [ self.tree.GetItemText(itemID) for itemID in selectedItemIDs ]
        # update list of selected tree items for OnTreeRightDown's benefit
        self.tree.selectedItemIDs = selectedItemIDs
        self.tree.selectedItems = selectedItems
        print(selectedItems)
        removeItems = [ item for item in self.tree.lastSelectedItems if item not in selectedItems ]
        addItems = [ item for item in selectedItems if item not in self.tree.lastSelectedItems ]
        self.RemoveItemsFromPlot(removeItems)
        self.AddItems2Plot(addItems)
        self.tree.lastSelectedItems = selectedItems # save for next time

    def OnTreeLeftDown(self, evt):
        print('in OnTreeLeftDown')
        pt = evt.GetPosition()
        itemID, flags = self.tree.HitTest(pt)
        if itemID.IsOk(): # if we've clicked on an item
            # leave selection event uncaught, call selection handler
            # after OS has finished doing the actual (de)selecting
            wx.CallAfter(self.OnTreeSelectChanged)

    def OnTreeLeftUp(self, evt):
        """Need this to catch clicking on non focused item, bug #4448"""
        print('in OnTreeLeftUp')
        self.OnTreeLeftDown(evt)

    def OnTreeRightDown(self, evt):
        """Toggle selection of the clicked item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection TreeEvent happening before the MouseEvent"""
        print('in OnTreeRightDown')
        pt = evt.GetPosition()
        itemID, flags = self.tree.HitTest(pt)
        if not itemID.IsOk(): # if we haven't clicked on an item
            return
        # first, restore all prior selections in the tree (except our item) that were cleared by the right click selection event
        for itID in self.tree.selectedItemIDs: # rely on tree.selectedItemIDs being judiciously kept up to date
            self.tree.SelectItem(itID)
        if itemID not in self.tree.selectedItemIDs: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(itemID, select=False)
        self.OnTreeSelectChanged() # now plot accordingly

    def OnTreeSpaceUp(self, evt):
        """Toggle selection of the clicked item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection TreeEvent happening before the SPACE KeyUp event.
        This causes an annoying flicker of the currently selected items, because they
        all unfortunately get deselected on the uncatchable SPACE keydown event,
        and aren't reselected until the SPACE keyup event"""
        print('in OnTreeSpaceUp')
        itemID = self.tree.GetFocusedItem()
        # first, restore all prior selections in the tree (except our item) that were cleared by the space selection event
        for itID in self.tree.selectedItemIDs: # rely on tree.selectedItemIDs being judiciously kept up to date
            self.tree.SelectItem(itID)
        if itemID not in self.tree.selectedItemIDs: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(itemID, select=False)
        self.OnTreeSelectChanged() # now plot accordingly

    def OnTreeKeyDown(self, evt):
        key = evt.GetKeyCode()
        #print 'key down: %r' % key
        if key == wx.WXK_TAB:
            self.list.SetFocus() # change focus to list
        elif key == wx.WXK_RETURN: # space only triggered on key up, see bug #4448
            self.tree.ToggleFocusedItem()
            #wx.CallAfter(self.OnTreeSelectChanged)
            self.OnTreeSelectChanged()
            # evt.Skip() seems to prevent toggling, or maybe it untoggles
        elif key in [wx.WXK_DELETE, ord('D'),]:
            self.MoveCurrentItems2List()
        elif key == ord('A'): # allow us to add from spike list even if tree is in focus
            self.MoveCurrentSpikes2Neuron(which='selected')
        elif key in [ord('C'), ord('N')]: # ditto for creating a new neuron
            self.MoveCurrentSpikes2Neuron(which='new')
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        elif key in [wx.WXK_UP, wx.WXK_DOWN]: # keyboard selection hack around multiselect bug
            wx.CallAfter(self.OnTreeSelectChanged)
            #self.OnTreeSelectChanged()
        self.tree.selectedItemIDs = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnTreeKeyUp(self, evt):
        key = evt.GetKeyCode()
        #print 'key up: %r' % key
        if key == wx.WXK_SPACE: # space only triggered on key up, see bug #4448
            if evt.ControlDown():
                #wx.CallAfter(self.OnTreeSelectChanged)
                self.OnTreeSelectChanged()
            else:
                #self.tree.ToggleFocusedItem()
                #wx.CallAfter(self.OnTreeSelectChanged)
                #self.OnTreeSelectChanged()
                self.OnTreeSpaceUp(evt)
                #evt.Skip() seems to prevent toggling, or maybe it untoggles
        self.tree.selectedItemIDs = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnSortTree(self, evt):
        root = self.tree.GetRootItem()
        if root: # tree isn't empty
            self.tree.SortChildren(root)
            self.RelabelNeurons(root)

    def RelabelNeurons(self, root):
        """Consecutively relabel neurons according to their vertical order in the TreeCtrl.
        Relabeling happens both in the TreeCtrl and in the .sort.neurons dict"""
        itemIDs = self.tree.GetTreeChildren(root) # get all children of root in order from top to bottom
        items = [ self.tree.GetItemText(itemID) for itemID in itemIDs ]
        neurons = {} # build up a new neurons dict
        for neuroni, item in enumerate(items):
            assert item[0] == 'n'
            neuron = self.sort.neurons[int(item[1:])]
            neuron.id = neuroni # update its id
            neurons[neuron.id] = neuron # add it to its key in neuron dict
            self.tree.SetItemText(neuron.itemID, 'n'+str(neuron.id)) # update its entry in the tree
        self.sort.neurons = neurons # overwrite the dict
        self.sort._nid = neuroni + 1 # reset unique Neuron ID counter to make next added neuron consecutive
        print('TODO: relabel all nids in sort.spikes recarray as well!')

    def DrawRefs(self):
        """Redraws refs and resaves background of sort panel(s)"""
        self.spikesortpanel.draw_refs()
        #self.chartsortpanel.draw_refs()

    def AddItems2Plot(self, items):
        self.spikesortpanel.addItems(items)
        #self.chartsortpanel.addItems(items)

    def RemoveItemsFromPlot(self, items):
        self.spikesortpanel.removeItems(items)
        #self.chartsortpanel.removeItems(items)

    def UpdateItemsInPlot(self, items):
        self.spikesortpanel.updateItems(items)
        #self.chartsortpanel.updateItems(items)

    # TODO: should self.OnTreeSelectChanged() (update plot) be called more often at the end of many of the following methods?:

    def AddNeuron2Tree(self, neuron):
        """Add a neuron to the tree control"""
        root = self.tree.GetRootItem()
        if not root.IsOk(): # if tree doesn't have a valid root item
            root = self.tree.AddRoot('Neurons')
        neuron.itemID = self.tree.AppendItem(root, 'n'+str(neuron.id)) # add neuron to tree
        #self.tree.SetItemPyData(neuron.itemID, neuron) # associate neuron tree item with neuron
    '''
    def RemoveNeuronFromTree(self, neuron):
        """Remove neuron and all its spikes from the tree"""
        self.MoveSpikes2List(neuron, neuron.spikeis)
        try:
            self.tree.Delete(neuron.itemID)
            del neuron.itemID # make it clear that neuron is no longer in tree
        except AttributeError:
            pass # neuron.itemID already deleted due to recursive call
    '''
    def RemoveNeuron(self, neuron):
        """Remove neuron and all its spikes from the tree and the Sort"""
        #self.RemoveNeuronFromTree(neuron)
        self.MoveSpikes2List(neuron, neuron.spikeis)
        try:
            del self.sort.neurons[neuron.id] # maybe already be removed due to recursive call
            del self.sort.clusters[neuron.id] # may or may not exist
        except KeyError:
            pass
        self.tree.RefreshItems()

    def MoveSpikes2Neuron(self, spikeis, neuron=None):
        """Assign spikes from sort.spikes to a neuron, and update mean wave.
        Also, move spikes from the spike list control to a neuron in the tree.
        If neuron is None, create a new one"""
        spikeis = toiter(spikeis)
        spikes = self.sort.spikes
        ris = spikes.id.searchsorted(spikeis)
        createdNeuron = False
        if neuron == None:
            neuron = self.sort.create_neuron()
            #self.AddNeuron2Tree(neuron)
            #createdNeuron = True
        neuron.spikeis.update(spikeis) # update the set
        spikes.neuron[ris] = neuron
        self.sort.update_uris()
        self.list.SetItemCount(len(self.sort.uris))
        self.list.RefreshItems() # refresh the list
        # TODO: selection doesn't seem to be working, always jumps to top of list
        #self.list.Select(row) # automatically select the new item at that position
        #self.AddSpikes2Tree(neuron.itemID, spikeis) # disable for huge cluster creation
        self.tree.RefreshItems()
        neuron.wave.data = None # signify it needs an update when it's actually needed
        #neuron.update_wave(self.sort.stream) # update mean neuron waveform
        '''
        if createdNeuron:
            #self.tree.Expand(root) # make sure root is expanded
            self.tree.Expand(neuron.itemID) # expand neuron
            self.tree.UnselectAll() # unselect all items in tree
            self.tree.SelectItem(neuron.itemID) # select the newly created neuron
            self.OnTreeSelectChanged() # now plot accordingly
        '''
        return neuron
    '''
    def AddSpikes2Tree(self, parent, spikeis):
        """Add spikes to the tree, where parent is a tree itemID"""
        spikeis = toiter(spikeis)
        if type(spikeis) == set:
            spikeis = np.asarray(list(spikeis))
        spikes = self.sort.spikes
        ris = spikes.id.searchsorted(spikeis)
        for ri, si in zip(ris, spikeis):
            # add spike to tree, save its itemID
            itemID = self.tree.AppendItem(parent, 's'+str(si))
            self.sort.spikes[ri].itemID = itemID
    '''
    def MoveSpikes2List(self, neuron, spikeis):
        """Move spikes from a neuron in the tree back to the list control.
        Make sure to call neuron.update_wave() at some appropriate time after
        calling this method"""
        spikeis = toiter(spikeis)
        if type(spikeis) == set:
            spikeis = np.asarray(list(spikeis))
        if len(spikeis) == 0:
            return # nothing to do
        spikes = self.sort.spikes
        neuron.spikeis.difference_update(spikeis) # remove spikeis from their neuron
        ris = spikes.id.searchsorted(spikeis)
        spikes.neuron[ris] = None # unbind neuron of spikeis in recarray
        #itemIDs = spikes.itemID[ris]
        #for itemID in itemIDs:
        #    self.tree.Delete(itemID) # update tree
        #spikes.itemID[ris] = None # no longer applicable
        self.sort.update_uris()
        self.list.SetItemCount(len(self.sort.uris))
        self.list.RefreshItems() # refresh the list
        self.tree.RefreshItems()

    def MoveCurrentSpikes2Neuron(self, which='selected'):
        if which == 'selected':
            neuron = self.GetFirstSelectedNeuron()
        elif which == 'new':
            neuron = None # indicates we want a new neuron
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        selected_uris = self.sort.uris[selected_rows]
        spikeis = self.sort.spikes.id[selected_uris]
        neuron = self.MoveSpikes2Neuron(spikeis, neuron) # if neuron was None, it isn't any more
        if neuron != None and neuron.plt != None: # if it exists and it's plotted
            self.UpdateItemsInPlot(['n'+str(neuron.id)]) # update its plot

    def MoveCurrentItems2List(self):
        for itemID in self.tree.GetSelections():
            if itemID: # check if spike's tree parent (neuron) has already been deleted
                item = self.tree.GetItemText(itemID)
                id = int(item[1:])
                sort = self.sort
                if item[0] == 's': # it's a spike
                    ri, = np.where(sort.spikes.id == id) # returns an array
                    ri = int(ri)
                    neuron = sort.spikes.neuron[ri]
                    self.MoveSpikes2List(neuron, id)
                    if len(neuron.spikeis) == 0:
                        self.RemoveNeuron(neuron) # remove empty Neuron
                    else:
                        neuron.update_wave(sort.stream) # update mean neuron waveform
                else: # it's a neuron
                    neuron = sort.neurons[id]
                    self.RemoveNeuron(neuron) # remove Neuron and all its Spikes
        self.OnTreeSelectChanged() # update plot

    def GetFirstSelectedNeuron(self):
        for itemID in self.tree.GetSelections():
            item = self.tree.GetItemText(itemID)
            id = int(item[1:])
            sort = self.sort
            if item[0] == 's': # it's a spike, get its neuron
                ri, = np.where(sort.spikes.id == id) # returns an array
                ri = int(ri)
                return sort.spikes.neuron[ri]
            else: # it's a neuron
                return sort.neurons[id]
        return None
