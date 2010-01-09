"""Spike sorting classes and frame"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

import os
import sys
import time
import datetime
from copy import copy
import operator
import random

import wx

import numpy as np
#from scipy.cluster.hierarchy import fclusterdata
#import pylab

from spyke.core import TW, WaveForm, Gaussian, MAXLONGLONG, R, toiter
from spyke import wxglade_gui

MAXCHANTOLERANCE = 100 # um

SPLITTERSASH = 360
SORTSPLITTERSASH = 117
NSSPLITTERSASH = 30
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
    and sort spikes into Neurons.
    Formerly known as a Session, and before that, a Collection.
    A .sort file is a single pickled Sort object"""
    DEFWAVEDATANSPIKES = 100000 # length (nspikes) to init contiguous wavedata array
    TW = TW # save a reference
    def __init__(self, detector=None, stream=None):
        self.__version__ = 0.2
        self.detector = detector # this Sort's current Detector object
        self.detections = {} # history of detection runs
        self.stream = stream
        self.probe = stream.probe # only one probe design per sort allowed
        self.converter = stream.converter

        # most neurons will have an associated cluster, but not necessarily all -
        # some neurons may be purely hand sorted, one spike at a time
        self.neurons = {}
        self.clusters = {} # dict of multidim ellipsoid params

        self.uris_sorted_by = 't'
        self.uris_reversed = False

        # how much to scale each dim for better viewing in cluster plots
        self.SCALE = {'x0': 5, 'Vpp': 0.5} #, 'IC1': 0.05, 'IC2': 0.05}

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
        tres = stream.tres
        twts = np.arange(self.TW[0], self.TW[1], tres) # temporal window timepoints wrt thresh xing or phase1t
        twts += twts[0] % tres # get rid of mod, so twts go through zero
        self.twts = twts
        # time window indices wrt thresh xing or 1st phase:
        self.twi = int(round(twts[0] / tres)), int(round(twts[-1] / tres))
        #info('twi = %s' % (self.twi,))

    stream = property(get_stream, set_stream)

    def __getstate__(self):
        """Get object state for pickling"""
        # copy it cuz we'll be making changes, this is fast because it's just a shallow copy
        d = self.__dict__.copy()
        # don't pickle the stream, cuz it relies on an open .srf file
        # spikes and wavedata arrays are (potentially) saved separately
        # all the others can be regenerated from the spikes array
        for attr in ['_stream', 'st', 'ris_by_time', 'uris', 'spikes', 'wavedatas', 'wavedatascumsum']:
            try: del d[attr]
            except KeyError: pass
        return d

    def get_nspikes(self):
        try: return len(self.spikes)
        except AttributeError: return 0

    nspikes = property(get_nspikes)

    def append_spikes(self, spikes):
        """Append spikes struct array to self.spikes struct array, update
        associated spike lists, and lock down sampfreq and shcorrect attribs"""
        if self.nspikes == 0: # (re)init
            self.spikes = spikes
        else: # append
            oldnspikes = self.nspikes # save
            shape = list(self.spikes.shape)
            shape[0] += len(spikes)
            self.spikes.resize(shape, refcheck=False) # resize in-place
            self.spikes[oldnspikes:] = spikes # append
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
        st = self.spikes['t'] # all spike times
        sids = self.spikes['id'] # all spike ids
        # self.st and self.ris_by_time are required for quick raster plotting
        # can't assume spikes come out of struct array sorted in time
        # (detections may not be in temporal order)
        self.ris_by_time = st.argsort() # struct array row indices of all spikes, sorted by time
        self.st = st[self.ris_by_time] # array of current spike times
        self.update_uris() # uris is an array of struct array indices of unsorted spikes

    def update_uris(self):
        """Update uris, which is an array of struct array indices of unsorted spikes,
        used by spike virtual listctrl"""
        nids = self.spikes['nid']
        self.uris, = np.where(nids == -1) # -1 indicates spike has no nid assigned to it
        # order it by .uris_sorted_by and .uris_reversed
        if self.uris_sorted_by != 't': self.sort_uris()
        if self.uris_reversed: self.reverse_uris()

    def sort_uris(self, sort_by):
        """Sort struct array row indices of unsorted spikes according to
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

    def init_wavedata(self, nchans=None, nt=None):
        self.wavedatas = []
        self.wavedatanchans = nchans
        self.wavedatant = nt
        self.append_wavedata()
        self.update_wavedatacumsum()

    def append_wavedata(self):
        nspikes = self.DEFWAVEDATANSPIKES
        nchans = self.wavedatanchans
        nt = self.wavedatant
        self.wavedatas.append(np.zeros((nspikes, nchans, nt), dtype=np.int16))

    def update_wavedatacumsum(self):
        """Call this every time self.wavedatas changes length, or any of its contained
        wavedata arrays changes length"""
        self.wavedatascumsum = np.asarray([ len(wd) for wd in self.wavedatas ]).cumsum() # update

    def get_wavedata(self, ris):
        """Get wave data, potentially spread across multiple wavedata 3D arrays
        in .wavedatas list, corresponding to ris. Returns wavedatas in order of
        sorted ris, not necessarily in original ris order"""
        ris = toiter(ris)
        if len(ris) == 1: # optimize for this special case
            ri = ris[0]
            wavedatai = self.wavedatascumsum.searchsorted(ri, side='right')
            wd = self.wavedatas[wavedatai]
            if wavedatai > 0:
                ri -= self.wavedatascumsum[wavedatai-1] # decr by nspikes in all previous wavedata arrays
            return wd[ri]
        # len(ris) > 1
        # first figure out which arrays in wavedatas the row indices ris correspond to
        ris.sort() # make sure they're sorted, no guarantee the results come out in the original order
        wavedatais = self.wavedatascumsum.searchsorted(ris, side='right') # these are in sorted order
        uniquewavedatais = np.unique(wavedatais) # also sorted
        startis = wavedatais.searchsorted(uniquewavedatais, side='left')
        endis = wavedatais.searchsorted(uniquewavedatais, side='right')
        slicedwavedatas = []
        for wavedatai, starti, endi in zip(uniquewavedatais, startis, endis):
            localris = ris[starti:endi]
            wd = self.wavedatas[wavedatai]
            if wavedatai > 0:
                 localris -= self.wavedatascumsum[wavedatai-1] # decr by nspikes in all previous wavedata arrays
            slicedwavedatas.append(wd[localris])
        return np.concatenate(slicedwavedatas)

    def set_wavedata(self, ri, wavedata, phase1ti):
        """Set 2D array wavedata to row index ri, in appropriate 3D wavedata
        array in self.wavedatas, and align it in time with all the rest according
        to phase1ti"""
        # first figure out which array in wavedatas the row index ri corresponds to
        wavedatai = self.wavedatascumsum.searchsorted(ri, side='right')
        if wavedatai > len(self.wavedatas)-1: # out of range of all wavedata arrays
            '''
            try:
                # resize last one
                wd = self.wavedatas[-1]
                shape = list(wd.shape) # allows assignment
                shape[0] += self.DEFWAVEDATANSPIKES
                print('resizing wavedata to %r' % shape)
                wd.resize(shape, refcheck=False)
            except MemoryError: # not enough contig memory to resize that one
            '''
            # append new wavedata array
            self.append_wavedata()
            self.update_wavedatacumsum()
            wavedatai = self.wavedatascumsum.searchsorted(ri, side='right') # update
        # now do the actual assignment
        wd = self.wavedatas[wavedatai]
        ri -= self.wavedatascumsum[wavedatai-1] # decr by nspikes in all previous wavedata arrays
        # TODO: ri comes out -ve when wavedatai == 0, but happens to be exactly right. Does it come out -ve for all wavedatai?
        nchans = len(wavedata)
        startti = -self.twi[0] - phase1ti # always +ve, usually 0 unless spike had some lockout near its start
        wd[ri, 0:nchans, startti:] = wavedata

    def get_wave(self, ri):
        """Return WaveForm corresponding to spikes struct array row ri"""
        spikes = self.spikes
        #ri = int(ri) # make sure it isn't stuck in a numpy scalar

        # try self.wavedata ndarray
        chan = spikes['chan'][ri]
        nchans = spikes['nchans'][ri]
        chans = spikes['chans'][ri, :nchans]
        t0 = spikes['t0'][ri]
        tend = spikes['tend'][ri]
        phase1ti = spikes['phase1ti'][ri]
        startti = -self.twi[0] - phase1ti # always +ve, usually 0 unless spike had some lockout near its start
        try:
            wavedata = self.get_wavedata(ri)
            ts = np.arange(t0, tend, self.tres) # build them up
            # only include data relevant to this spike
            wavedata = wavedata[0:nchans, startti:]
            return WaveForm(data=wavedata, ts=ts, chans=chans)
        except AttributeError: pass

        # try getting it from the stream
        if self.stream == None:
            raise RuntimeError("No stream open, can't get wave for %s %d" %
                               (spikes[ri], spikes[ri].id))
        detid = spikes['detid'][ri]
        det = self.detections[detid].detector
        if det.srffname != self.stream.srffname:
            msg = ("Spike %d was extracted from .srf file %s.\n"
                   "The currently opened .srf file is %s.\n"
                   "Can't get spike %d's wave" %
                   (spikes[ri].id, det.srffname, self.stream.srffname, spikes[ri].id))
            wx.MessageBox(msg, caption="Error", style=wx.OK|wx.ICON_EXCLAMATION)
            raise RuntimeError(msg)
        wave = self.stream[t0:tend]
        return wave[chans]

    def get_srffnameroot(self):
        """Return root name (without extension) of .srf file, checking to make
        sure that all detections come from the same .srf file, and match the
        currently opened one"""
        allsrffnames = set([ detection.detector.srffname for detection in self.detections.values() ])
        if len(allsrffnames) == 0:
            srffname = self.stream.srffname
        else:
            srffname = allsrffnames.pop()
        srffnameroot = srffname.partition('.srf')[0]
        if len(allsrffnames) != 0:
            raise ValueError("Can't figure out srffnameroot, because detections come from "
                             "different .srf files")
        if srffname != self.stream.srffname:
            raise ValueError("Can't figure out srffnameroot, because currently open .srf "
                             "file doesn't match the one in the detections")
        return srffnameroot

    def export(self, path=''):
        """Export stimulus textheader, din and/or spike data to binary files in path in
        the classic way for use in neuropy"""
        # first export the din to path, using the source .srf fname of first
        # detection as its name
        print('Exporting data to %r' % path)
        if hasattr(self, 'stream'):
            srffnameroot = self.get_srffnameroot()
            if hasattr(self.stream.srff, 'displayrecords'):
                self.exporttextheader(srffnameroot, path)
            if hasattr(self.stream.srff, 'digitalsvalrecords'):
                self.exportdin(srffnameroot, path)
        if len(self.neurons) != 0:
            self.exportspikes(path)

    def exporttextheader(self, srffnameroot, path=''):
        """Export stimulus text header to path"""
        displayrecords = self.stream.srff.displayrecords
        if len(displayrecords) != 1:
            print("Can't figure out which display record to export stimulus text header from")
            return
        textheader = displayrecords[0].Header.python_tbl
        textheaderfname = srffnameroot + '.textheader'
        print(textheaderfname)
        f = open(os.path.join(path, textheaderfname), 'w')
        f.write(textheader) # save it
        f.close()

    def exportdin(self, srffnameroot, path=''):
        """Export stimulus din to binary file in path"""
        dinfname = srffnameroot + '.din'
        print(dinfname)
        dinfiledtype=[('TimeStamp', '<i8'), ('SVal', '<i8')] # pairs of int64s
        # upcast SVal field from uint16 to int64, creates a copy, but it's not too expensive
        digitalsvalrecords = self.stream.srff.digitalsvalrecords.astype(dinfiledtype)
        digitalsvalrecords.tofile(os.path.join(path, dinfname)) # save it

    def exportspikes(self, path=''):
        """Export spike data to binary files in path, one file per neuron"""
        spikes = self.spikes
        dt = str(datetime.datetime.now()) # get an export timestamp
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        spikefoldername = dt + '.best.sort'
        path = os.path.join(path, spikefoldername)
        os.mkdir(path)
        for nid, neuron in self.neurons.items():
            spikeis = neuron.spikeis # should be sorted
            ris = spikes['id'].searchsorted(spikeis)
            spikets = spikes['t'][ris]
            # pad filename with leading zero to always make template (t) ID at least 2 digits long
            neuronfname = '%s_t%02d.spk' % (dt, nid)
            print(neuronfname)
            spikets.tofile(os.path.join(path, neuronfname)) # save it

    def get_param_matrix(self, dims=None, viz_scaled=False):
        """Organize parameters in dims from all spikes into a
        data matrix, each column corresponds to a dim"""
        # np.column_stack returns a copy, not modifying the original array
        X = np.column_stack([ np.float32(self.spikes[dim]) for dim in dims ])
        if viz_scaled:
            # scale select columns for better visualization
            for dim, col in zip(dims, X.T): # iterate over columns
                if dim in self.SCALE:
                    col *= self.SCALE[dim]
        return X

    def apply_cluster(self, cluster):
        """Apply cluster to spike data - calculate which spikes fall within the
        cluster's multidimensional ellipsoid. Return spike indices in an array view"""

        # consider all the dimensions in this cluster that have non-zero scale
        dims = [ dim for dim, val in cluster.scale.items() if val != 0 ]
        # get same X that was used for visualization
        X = self.get_param_matrix(dims=dims, viz_scaled=True)

        # To find which points fall within the ellipsoid, need to do the inverse of all
        # the operations that translate and rotate the ellipsoid, in the correct order.
        # Need to do those operations on the points, not on the ellipsoid parameters.
        # That way, we can figure out which points to pick out, and then we
        # pick them out of the original set of unmodified points

        # undo the translation, in place
        SCALE = self.SCALE
        dim2coli = {}
        for i, dim in enumerate(dims):
            X[:, i] -= cluster.pos[dim] * SCALE.get(dim, 1)
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
            A = cluster.scale[dim] * SCALE.get(dim, 1)
            sumterms += x**2/A**2
        trutharray = (sumterms <= 1) # which points in nonrotdims space fall inside the ellipsoid?

        # for each rotation group, undo the rotation by taking product of inverse of
        # rotation matrix (which == its transpose) with the detranslated points
        for rotdims, oris in rotgroups.items():
            Xrot = np.column_stack([ X[:, dim2coli[dim]] for dim in rotdims ]) # pull correct columns out of X for this rotgroup
            Xrot = (R(oris[0], oris[1], oris[2]).T * Xrot.T).T
            Xrot = np.asarray(Xrot) # convert from np.matrix back to np.array to prevent from taking matrix power
            # which points are inside the ellipsoid?
            x = Xrot[:, 0]; A = cluster.scale[rotdims[0]] * SCALE.get(rotdims[0], 1)
            y = Xrot[:, 1]; B = cluster.scale[rotdims[1]] * SCALE.get(rotdims[1], 1)
            z = Xrot[:, 2]; C = cluster.scale[rotdims[2]] * SCALE.get(rotdims[2], 1)
            trutharray *= (x**2/A**2 + y**2/B**2 + z**2/C**2 <= 1) # AND with interior points from any previous rotgroups

        # spikes row indices of points that fall within ellipsoids of all rotgroups
        ris, = np.where(trutharray)
        #assert len(i) > 0, "no points fall within the ellipsoid"
        #Xin = X[i] # pick out those points
        #spikes = np.asarray(self.get_spikes_sortedby('id'))[i]
        spikeis = self.spikes['id'][ris]
        return spikeis

    def align_neuron(self, nid, to):
        """Align all neuron nid's spikes by their max or min"""
        neuron = self.neurons[nid]
        spikes = self.spikes
        nris = spikes['id'].searchsorted(neuron.spikeis) # row indices of spikes that belong to this neuron
        V1s = spikes['V1'][nris]
        V2s = spikes['V2'][nris]
        if to == 'max':
            nriis = V1s < 0 # indices into nris of spikes aligned to the min phase
        elif to == 'min':
            nriis = V1s > 0 # indices into nris of spikes aligned to the max phase
        else: raise ValueError()
        ris = nris[nriis] # row indices of spikes that need realigning
        dphasetis = spikes['phase2ti'][ris] - spikes['phase1ti'][ris]
        dphases = spikes['dphase'][ris]
        # shift values
        spikes['phase1ti'][ris] -= dphasetis
        spikes['phase2ti'][ris] -= dphasetis
        spikes['t0'][ris] += dphases
        spikes['t'][ris] += dphases
        spikes['tend'][ris] += dphases
        # now swap names
        spikes['phase1ti'][ris], spikes['phase2ti'][ris] = spikes['phase2ti'][ris], spikes['phase1ti'][ris]
        spikes['V1'][ris], spikes['V2'][ris] = spikes['V2'][ris], spikes['V1'][ris]
        spikes['dphase'][ris] = -spikes['dphase'][ris]
        # update wavedata for each shifted spike
        for ri, spike in zip(ris, spikes[ris]):
            wave = self.stream[spike['t0']:spike['tend']]
            chans = spike['chans'][:spike['nchans']]
            wave = wave[chans]
            self.set_wavedata(ri, wave.data, spike['phase1ti'])
        neuron.update_wave() # update mean waveform
        # TODO: trigger a redraw for all of this neuron's plotted spikes
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
            s2nids[spike['id']] = nid
            n2sids[nid].append(spike['id'])
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
            f.write('s%d\t' % spike['id'])
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
    '''
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
        dims = self.nspikes, 2+nchans*npoints
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
        t0 = time.time()
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
        print '\nmatch took %.3f sec' % (time.time()-t0)
    '''

class Neuron(object):
    """A collection of spikes that have been deemed somehow, whether manually
    or automatically, to have come from the same cell. A Neuron's waveform
    is the mean of its member spikes"""
    def __init__(self, sort, id=None):
        self.sort = sort
        self.id = id # neuron id
        self.wave = WaveForm() # init to empty waveform
        self.spikeis = np.array([], dtype=int) # indices of spikes that make up this neuron
        self.t = 0 # relative reference timestamp, here for symmetry with fellow spike rec (obj.t comes up sometimes)
        self.plt = None # Plot currently holding self
        self.cluster = None
        #self.srffname # not here, let's allow neurons to have spikes from different files?
    '''
    def get_spikes(self):
        return self.sort.si2spikes(self.spikeis)

    spikes = property(get_spikes)
    '''
    def update_wave(self):
        """Update mean waveform, should call this every time .spikes are modified.
        Setting .spikes as a property to do so automatically doesn't work, because
        properties only catch name binding of spikes, not modification of an object
        that's already been bound"""
        sort = self.sort
        spikes = sort.spikes
        if len(self.spikeis) == 0: # no member spikes, perhaps I should be deleted?
            raise RuntimeError("neuron %d has no spikes and its waveform can't be updated" % self.id)
            #self.wave = WaveForm() # empty waveform
            #return self.wave
        spikeis = self.spikeis
        ris = spikes['id'].searchsorted(spikeis)

        t0 = time.time()
        chanss = spikes['chans'][ris]
        nchanss = spikes['nchans'][ris]
        chanslist = [ chans[:nchans] for chans, nchans in zip(chanss, nchanss) ]
        chanpopulation = np.concatenate(chanslist)
        neuronchans = np.unique(chanpopulation)
        print('first loop took %.3f sec' % (time.time()-t0))

        t0 = time.time()
        try:
            wavedatas = sort.get_wavedata(ris)
        except MemoryError:
            # grab a random subset of spikes to use to calculate the mean
            k = 200
            print('Taking random sample of %d spikes instead of all of them' % k)
            ris = random.sample(ris, k=k) # ris is now a list, not array, but that doesn't matter
            wavedatas = sort.get_wavedata(ris)
        if wavedatas.ndim == 2: # should be 3, get only 2 if len(ris) == 1
            wavedatas.shape = 1, wavedatas.shape[0], wavedatas.shape[1] # give it a singleton 3rd dim
        maxnt = wavedatas.shape[-1]
        shape = len(neuronchans), maxnt
        data = np.zeros(shape, dtype=np.float32)
        nspikes = np.zeros(maxnt, dtype=np.int32)
        twi0 = -sort.twi[0] # num points from tref backwards to first timepoint in window
        phase1tis = spikes['phase1ti'][ris]
        starttis = twi0 - phase1tis # always +ve, usually 0 unless spike had some lockout near its start
        for chans, wavedata, startti in zip(chanslist, wavedatas, starttis):
            chanis = neuronchans.searchsorted(chans) # each spike's chans is a subset of neuronchans
            data[chanis] += wavedata[:len(chans)] # accumulate
            nspikes[startti:] += 1 # inc spike count for timepoints for this spike
        print('2nd loop took %.3f sec' % (time.time()-t0))
        t0 = time.time()
        #nspikes = np.maximum(nspikes, np.ones(shape, dtype=np.float32)) # element-wise max, avoids div by 0
        #np.seterr(invalid='ignore')
        data /= nspikes # normalize each data point appropriately
        #np.seterr(invalid='raise') # restore error level
        bins = list(neuronchans) + [sys.maxint] # concatenate rightmost bin edge
        hist, bins = np.histogram(chanpopulation, bins=bins)
        newneuronchans = neuronchans[hist >= len(ris)/2]
        chanis = neuronchans.searchsorted(newneuronchans)
        self.wave.data = data[chanis]
        self.wave.chans = newneuronchans
        self.wave.ts = sort.twts
        #print('neuron[%d].wave.chans = %r' % (self.id, chans))
        #print('neuron[%d].wave.ts = %r' % (self.id, ts))
        print('mean calc took %.3f sec' % (time.time()-t0))
        return self.wave

    def remove_chans(self, remchans, reextract=True):
        """Remove remchans from all member spikes, re-extract their params.
        This is to get rid of spurious clustering using spatial mean
        when the maxchan alternates between two chans"""
        #import pdb; pdb.set_trace()
        sort = self.sort
        spikes = sort.spikes
        spikeis = self.spikeis
        ris = spikes['id'].searchsorted(spikeis)
        twi0 = -sort.twi[0] # num points from tref backwards to first timepoint in window
        for ri in ris:
            s = spikes[ri]
            wavedata = sort.get_wavedata(ri)
            nchans = s['nchans']
            startti = twi0 - s['phase1ti'] # always +ve, usually 0 unless spike had some lockout near its start
            wavedata = wavedata[0:nchans, startti:]
            chans = s['chans'][:nchans]
            for remchan in remchans:
                chani, = np.where(chans == remchan)
                if len(chani) != 0: # delete it
                    wavedata = np.delete(wavedata, chani, axis=0)
                    sort.set_wavedata(ri, wavedata, s['phase1ti'])
                    chans = np.delete(chans, chani)
                    nchans = len(chans)
                    s['nchans'] = nchans
                    s['chans'][:nchans] = chans
                    if reextract:
                        detid = s['detid']
                        det = sort.detections[detid].detector
                        chanis = det.chans.searchsorted(chans) # det.chans are always sorted
                        x = det.siteloc[chanis, 0] # 1D array (row)
                        y = det.siteloc[chanis, 1]
                        maxchani = int(np.where(chans == s['chan'])[0])
                        s['x0'], s['y0'] = sort.extractor.extractXY(wavedata, x, y,
                                                                    s['phase1ti'], s['phase2ti'],
                                                                    maxchani)
        # TODO: replot only the spikes whose params have changed, and keep their scalar params intact
        # so you can see that they moved
        self.wave = WaveForm() # reset to empty waveform so mean is recalculated
        # trigger resaving of .spike and .wave files next time .sort is saved,
        # since their associated array contents have changed
        try: del sort.spikefname
        except AttributeError: pass
        try: del sort.wavefname
        except AttributeError: pass

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

    def get_nspikes(self):
        return len(self.spikeis)

    nspikes = property(get_nspikes)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        d['plt'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
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
        self.ns_splitter.SetSashPosition(NSSPLITTERSASH)

        #self.slist.Bind(wx.EVT_RIGHT_DOWN, self.OnSListRightDown)
        #self.slist.Bind(wx.EVT_KEY_DOWN, self.OnSListKeyDown)

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

    def OnNListSelect(self, evt):
        selectedRows = self.nlist.getSelection()
        nids = set(np.asarray(list(self.sort.neurons))[selectedRows])
        remove_nids = self.nlist.lastSelectedIDs.difference(nids)
        add_nids = nids.difference(self.nlist.lastSelectedIDs)
        self.RemoveItemsFromPlot([ 'n'+str(nid) for nid in remove_nids ])
        self.AddItems2Plot([ 'n'+str(nid) for nid in add_nids ])
        self.nlist.lastSelectedIDs = nids # save for next time
        if self.nslist.neuron != None and self.nslist.neuron.id in remove_nids:
            self.nslist.neuron = None
        elif len(nids) == 1:
            nid = list(nids)[0]
            self.nslist.neuron = self.sort.neurons[nid]

    def OnNSListSelect(self, evt):
        sort = self.sort
        selectedRows = self.nslist.getSelection()
        sids = set(self.nslist.neuron.spikeis[selectedRows])
        remove_sids = self.nslist.lastSelectedIDs.difference(sids)
        add_sids = sids.difference(self.nslist.lastSelectedIDs)
        self.RemoveItemsFromPlot([ 's'+str(sid) for sid in remove_sids ])
        self.AddItems2Plot([ 's'+str(sid) for sid in add_sids ])
        self.nslist.lastSelectedIDs = sids # save for next time

    def OnSListSelect(self, evt):
        sort = self.sort
        selectedRows = self.slist.getSelection()
        ris = sort.uris[selectedRows]
        sids = set(sort.spikes['id'][ris])
        remove_sids = self.slist.lastSelectedIDs.difference(sids)
        add_sids = sids.difference(self.slist.lastSelectedIDs)
        self.RemoveItemsFromPlot([ 's'+str(sid) for sid in remove_sids ])
        self.AddItems2Plot([ 's'+str(sid) for sid in add_sids ], ris=ris)
        self.slist.lastSelectedIDs = sids # save for next time
    '''
    def OnSListRightDown(self, evt):
        """Toggle selection of the clicked list item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection ListEvent happening before the MouseEvent, or something"""
        print('in OnSListRightDown')
        pt = evt.GetPosition()
        row, flags = self.slist.HitTest(pt)
        sid = self.sort.spikes['id'][self.sort.uris[row]]
        print('spikeID is %r' % sid)
        # this would be nice, but doesn't work (?) cuz apparently somehow the
        # selection ListEvent happens before MouseEvent that caused it:
        #selected = not self.slist.IsSelected(row)
        #self.slist.Select(row, on=int(not selected))
        # here is a yucky workaround:
        try:
            self.spikesortpanel.used_plots['s'+str(sid)] # is it plotted?
            selected = True # if so, item must be selected
            print('spike %d in used_plots' % sid)
        except KeyError:
            selected = False # item is not selected
            print('spike %d not in used_plots' % sid)
        self.slist.Select(row, on=not selected) # toggle selection, this fires sel spike, which updates the plot
    '''
    def OnSListColClick(self, evt):
        """Sort .uris according to column clicked.

        TODO: keep track of currently selected spikes and currently focused spike,
        clear the selection, then reselect those same spikes after sorting is done,
        and re-focus the same spike. Scroll into view of the focused spike (maybe
        that happens automatically). Right now, the selection remains in the list
        as-is, regardless of the entries that change beneath it"""
        col = evt.GetColumn()
        field = self.slist.COL2FIELD[col]
        s = self.sort
        # for speed, check if already sorted by field
        if s.uris_sorted_by == field: # already sorted, reverse the order
            s.reverse_uris()
            s.uris_reversed = not s.uris_reversed # update reversed flag
        else: # not yet sorted by field
            s.sort_uris(field)
            s.uris_sorted_by = field # update
            s.uris_reversed = False # update
        self.slist.RefreshItems()

    def OnAlignMax(self, evt):
        self.Align('max')

    def OnAlignMin(self, evt):
        self.Align('min')

    def Align(self, to):
        selectedRows = self.nlist.getSelection()
        if len(selectedRows) != 1:
            raise RuntimeError("Exactly 1 neuron must be selected for spike alignment")
        row = selectedRows[0]
        nid = list(self.sort.neurons)[row]
        self.sort.align_neuron(nid, to)

    def DrawRefs(self):
        """Redraws refs and resaves background of sort panel(s)"""
        self.spikesortpanel.draw_refs()

    def AddItems2Plot(self, items, ris=None):
        try: self.spikesortpanel.addItems(items, ris=ris)
        except RuntimeError: # probably a neuron with no spikes
            pass

    def RemoveItemsFromPlot(self, items):
        try: self.spikesortpanel.removeItems(items)
        except KeyError:
            # probably a neuron with no spikes that was never added to plot.
            # catching this might risk hiding deeper flaws, but seems to work for now
            pass

    def UpdateItemsInPlot(self, items):
        self.spikesortpanel.updateItems(items)

    def RemoveNeuron(self, neuron):
        """Remove neuron and all its spikes from the GUI and the Sort"""
        #self.RemoveNeuronFromTree(neuron)
        self.MoveSpikes2List(neuron, neuron.spikeis)
        try:
            del self.sort.neurons[neuron.id] # maybe already be removed due to recursive call
            del self.sort.clusters[neuron.id] # may or may not exist
        except KeyError:
            pass
        self.nlist.SetItemCount(len(self.sort.neurons))
        self.nlist.RefreshItems()
        if neuron == self.nslist.neuron:
            self.nslist.neuron = None

    def MoveSpikes2Neuron(self, spikeis, neuron=None):
        """Assign spikes from sort.spikes to a neuron, and update mean wave.
        If neuron is None, create a new one"""
        spikeis = toiter(spikeis)
        spikes = self.sort.spikes
        ris = spikes['id'].searchsorted(spikeis)
        createdNeuron = False
        if neuron == None:
            neuron = self.sort.create_neuron()
        neuron.spikeis = np.union1d(neuron.spikeis, spikeis) # update
        spikes['nid'][ris] = neuron.id
        self.sort.update_uris()
        self.slist.SetItemCount(len(self.sort.uris))
        self.slist.RefreshItems() # refresh the list
        if neuron == self.nslist.neuron:
            self.nslist.neuron = neuron # this triggers a refresh
        # TODO: selection doesn't seem to be working, always jumps to top of list
        #self.slist.Select(row) # automatically select the new item at that position
        neuron.wave.data = None # signify it needs an update when it's actually needed
        #neuron.update_wave() # update mean neuron waveform
        return neuron

    def MoveSpikes2List(self, neuron, spikeis):
        """Move spikes from a neuron back to the unsorted spike list control.
        Make sure to call neuron.update_wave() at some appropriate time after
        calling this method"""
        spikeis = toiter(spikeis)
        if len(spikeis) == 0:
            return # nothing to do
        spikes = self.sort.spikes
        neuron.spikeis = np.setdiff1d(neuron.spikeis, spikeis) # return what's in first arr and not in the 2nd
        ris = spikes['id'].searchsorted(spikeis)
        spikes['nid'][ris] = -1 # unbind neuron id of spikeis in struct array
        self.sort.update_uris()
        self.slist.SetItemCount(len(self.sort.uris))
        self.slist.RefreshItems() # refresh the spike list
        # this only makes sense if the neuron is currently selected in the nlist:
        if neuron == self.nslist.neuron:
            self.nslist.neuron = neuron # this triggers a refresh

    def MoveCurrentSpikes2Neuron(self, which='selected'):
        if which == 'selected':
            neuron = self.GetFirstSelectedNeuron()
        elif which == 'new':
            neuron = None # indicates we want a new neuron
        selected_rows = self.slist.getSelection()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        selected_uris = self.sort.uris[selected_rows]
        spikeis = self.sort.spikes['id'][selected_uris]
        neuron = self.MoveSpikes2Neuron(spikeis, neuron) # if neuron was None, it isn't any more
        if neuron != None and neuron.plt != None: # if it exists and it's plotted
            self.UpdateItemsInPlot(['n'+str(neuron.id)]) # update its plot
