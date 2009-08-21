"""Spike sorting classes and frame"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

import os
import sys
import time
import datetime
import copy
import operator

import wx

import numpy as np
#from scipy.cluster.hierarchy import fclusterdata
#import pylab

from spyke.core import WaveForm, Gaussian, MAXLONGLONG
from spyke import wxglade_gui
from spyke.plot import CMAP, CMAPWITHJUNK
from spyke.detect import Spike, TW

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
        self.detector = detector # this sort session's current Detector object
        self.detections = {} # history of detection runs
        self.stream = stream
        self.probe = stream.probe # only one probe design per sort allowed
        self.converter = stream.converter

        # all unsorted spikes detected in this sort session across all Detection runs, indexed by unique ID
        # sorted spikes go in their respective Neuron's .spikes dict
        self.spikes = {}
        self.neurons = {} # first hierarchy of neurons
        self.trash = {} # discarded spikes

        self._spikes_sorted_by = 't'
        self._spikes_reversed = False

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
        d = self.__dict__.copy() # copy it cuz we'll be making changes, this doesn't seem to be a slow step
        del d['_stream'] # don't pickle the stream, cuz it relies on an open .srf file
        for attr in ['_st', '_spikes_by_time', '_spikes']:
            # all are temporary, and can be regenerated from .spikes
            try: del d[attr]
            except KeyError: pass
        return d

    def __setstate__(self, d):
        """Restore self on unpickle per usual, but also restore
        ._st, ._spikes_by_time, and ._spikes"""
        self.__dict__ = d
        self.update_spike_lists()

    def append_spikes(self, spikes):
        """Append spikes to self.spikes dict, update associated spike lists,
        and lockdown sampfreq and shcorrect attribs. Don't add a new spike
        from a new detection if the identical spike is already in self.spikes"""
        newspikes = set(spikes.values()).difference(self.spikes.values())
        duplicates = set(spikes.values()).difference(newspikes)
        if duplicates:
            print('not adding duplicate spikes %r' % [ spike.id for spike in duplicates ])
        uniquespikes = {}
        for newspike in newspikes:
            uniquespikes[newspike.id] = newspike
        self.spikes.update(uniquespikes)
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
        """Update self._st sorted array of spike times, _spikes_by_time array,
        and self._spikes list"""
        spikes = self.spikes.values() # pull list out of dict
        st = np.asarray([ spike.t for spike in spikes ]) # spike times
        # can't assume spikes come out of dict sorted in time
        sti = st.argsort() # spike time indices
        # self._st and self._spikes_by_time are required for quick raster plotting
        self._st = st[sti] # array of current spike times
        self._spikes_by_time = np.asarray(spikes)[sti] # array of current spikes sorted by time
        self._spikes = spikes # indep list of current spikes, by however they were last sorted
        self._spikes.sort(key=operator.attrgetter(self._spikes_sorted_by),
                          reverse=self._spikes_reversed)

    def get_spikes_sortedby(self, attr='id'):
        """Return list of spikes, sorted by attribute 'attr'"""
        spikes = self.spikes.values()
        spikes.sort(key=operator.attrgetter(attr)) # sort in-place by spike attribute
        return spikes

    def extractXY(self, method):
        """Extract XY parameters from spikes using extraction method"""
        if len(self.spikes) == 0:
            raise RuntimeError("No spikes to extract XY parameters from")
        print("Extracting parameters from spikes")
        t0 = time.clock()
        if method.lower() == 'spatial mean':
            for spike in self.spikes.values():
                det = spike.detection.detector
                spike.x0, spike.y0 = det.get_spike_spatial_mean(spike)
        elif method.lower() == 'gaussian fit':
            for spike in self.spikes.values():
                det = spike.detection.detector
                spike.x0, spike.y0 = det.get_gaussian_fit(spike)
        else:
            raise ValueError("Unknown XY parameter extraction method %r" % method)
        print("Extracting XY parameters from all %d spikes using %s took %.3f sec" %
              (len(self.spikes), method.lower(), time.clock()-t0))

    def get_param_matrix(self):
        """Organize parameters from all spikes into a data matrix for clustering"""
        try:
            return self.X # see if param matrix has already been calculated and saved
        except AttributeError:
            nspikes = len(self.spikes)
            nparams = 4
            #nparams = 9
            X = np.empty((nspikes, nparams), dtype=np.float32)
            spikes = self.get_spikes_sortedby('id')
            for i, s in enumerate(spikes):
                X[i] = [s.x0, s.y0, s.Vpp, s.dphase]
                '''
                V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta = s.p # underlying model parameters
                # for clustering, substitute V1/V2 with Vpp, and mu1/mu2 with dphase
                X[i] = [s.Vpp, s.dphase, s1, s2, x0, y0, sx, sy, theta]
                #X[i] = [x0, y0]
                '''
            '''
            # normalize each column in X (ie each param) from [0, 1]
            X -= X.min(axis=0) # have them all start from 0
            X /= X.max(axis=0) # normalize
            # now weight some parameters more than others. This affects the euclidean distance
            # between clusters, which affects their agglomeration/density.
            # maybe the ideal parameter weights can come from openopt...
            X[:, 0] *= 2 # Vpp
            X[:, 2] *= 5 # x0
            X[:, 3] *= 10 # y0
            '''
            self.X = X # save
            return X

    def get_cluster_data(self, weighting='pca'):
        """Convert spike param matrix into pca/ica data for clustering"""

        import mdp # can't delay this any longer

        try:
            if self.weighting == weighting:
                return self.features # return previously saved features
            else:
                raise AttributeError("self.weighting != weighting")
        except AttributeError: # self.weighting and/or self.features don't exist, or self.weighting != weighting
            X = self.get_param_matrix()
            if weighting.lower() == 'ica':
                node = mdp.nodes.FastICANode()
            elif weighting.lower() == 'pca':
                node = mdp.nodes.PCANode()
            else:
                raise ValueError, 'unknown weighting %r' % weighting
            node.train(X)
            features = node.execute(X) # returns all available components
            self.node = node
            self.weighting = weighting
            self.features = features
            return features

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
        X = self.get_cluster_data()
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
    '''
    def parse_spc_lab_file_slow(self, fname=None):
        """Parse output .lab file from SPC. Each row is the cluster assignment of each spin
        (datapoint) to a cluster, one row per temperature datapoint. First column is temperature
        run number (0-based). 2nd column is the temperature. All remaining columns correspond
        to the datapoints in the order presented in the input .dat file. Returns cidsT dict"""
        #spikes = self.get_spikes_sortedby('id')
        if fname == None:
            fname = self.spclabfname
        f = open(fname, 'r')
        #n2sidsT = {} # at different temperatures
        #s2nidsT = {} # at different temperatures
        cidsT = {} # at different temperatures
        for row in f:
            row = np.fromstring(row, sep=' ') # array of floats
            T = row[1] # row[0] is run #
            cids = np.int32(row[2:])
            cidsT[T] = cids
            #n2sidsT[T], s2nidsT[T] = self.get_ids(cids, spikes)
        f.close()
        #return n2sidsT, s2nidsT
        #return n2sidsT
        return cidsT
    '''
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
        X = self.get_cluster_data()
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
        X = self.get_cluster_data()
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
        self.spikes[0].update_wave(stream=self.stream) # make sure it has a wave
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
                spike.update_wave(stream=self.stream)
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

class Detection(object):
    """A spike detection run, which happens every time Search is pressed.
    When you're merely searching for the previous/next spike with
    F2/F3, that's not considered a detection run"""
    def __init__(self, sort, detector, id=None, datetime=None, spikes=None):
        self.sort = sort # parent sort session
        self.detector = detector # Detector object used in this Detection run
        self.id = id
        self.datetime = datetime
        self._spikes = spikes # list of spikes collected from Detector.search

    def __eq__(self, other):
        """Compare detection runs by their ._spikes lists"""
        return np.all(self._spikes == other._spikes)

    def set_spikeids(self):
        """Give each spike an ID, inc sort's _sid spike ID counter after each one.
        Stick a references to all spikes into a .spikes dict, using spike IDs as the keys"""
        self.spikes = {}
        for s in self._spikes:
            s.id = self.sort._sid
            self.sort._sid += 1 # inc for next unique SpikeModel
            s.detection = self
            try:
                s.wave
            except AttributeError:
                s.wave = WaveForm() # init to empty waveform
            #s.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
            #s.plt = None # Plot currently holding self
            #s.neuron = None # neuron currently associated with
            self.spikes[s.id] = s
    '''
    deprecated: use spike.chans instead:
    def get_slock_chans(self, maxchan):
        """Get or generate list of chans within spatial lockout of maxchan, use
        spatial lockout of self.detector
        Note this can't be used as the getter in a property, I think cuz you can't pass
        args to a getter"""
        try:
            return self._slock_chans[maxchan]
        except KeyError:
            det = self.detector
            # NOTE: dm is now always a full matrix, where its row indices always correspond
            # to channel indices, so no need for messing around with indices into indices...
            #chans = np.asarray(det.chans) # chans that correspond to rows/columns in det.dm
            #maxchani, = np.where(chans == maxchan) # get index into det.dm that corresponds to maxchan
            chans, = np.where(det.dm[maxchan].flat <= det.slock) # flat removes the singleton dimension
            chans = list(chans)
            self._slock_chans[maxchan] = chans # save for quick retrieval next time
            return chans
    '''

class Neuron(object):
    """A collection of spikes that have been deemed somehow, whether manually
    or automatically, to have come from the same cell. A Neuron's waveform
    is the mean of its member spikes"""
    def __init__(self, sort, id=None, parent=None):
        self.sort = sort # parent sort session
        self.id = id # neuron id
        self.wave = WaveForm() # init to empty waveform
        self.spikes = {} # member spikes that make up this neuron
        self.t = 0 # relative reference timestamp, here for symmetry with fellow obj Spike (obj.t comes up sometimes)
        self.plt = None # Plot currently holding self
        self.itemID = None # tree item ID, set when self is displayed as an entry in the TreeCtrl
        self.ellipsoid_params = None
        #self.surffname # not here, let's allow neurons to have spikes from different files?


    def update_wave(self):
        """Update mean waveform, should call this every time .spikes are modified.
        Setting .spikes as a property to do so automatically doesn't work, because
        properties only catch name binding of spikes, not modification of an object
        that's already been bound"""
        if self.spikes == {}: # no member spikes, perhaps I should be deleted?
            raise RuntimeError, "neuron %d has no spikes and its waveform can't be updated" % self.id
            #self.wave = WaveForm() # empty waveform
            #return self.wave
        chans, ts = set(), set() # build up union of chans and relative timepoints of all member spikes
        for spike in self.spikes.values():
            chans = chans.union(spike.chans)
            spikets = np.arange(spike.t0, spike.tend, self.sort.tres) # build them up
            ts = ts.union(spikets - spike.t) # timepoints wrt spike time, not absolute
        chans = np.asarray(list(chans))
        ts = np.asarray(list(ts))
        chans.sort() # Neuron's chans are a sorted union of chans of all its member spikes
        ts.sort() # ditto for timepoints

        # take mean of chans of data from spikes with potentially different chans and time windows wrt their spike
        shape = len(chans), len(ts)
        data = np.zeros(shape, dtype=np.float32) # collect data that corresponds to chans and ts
        nspikes = np.zeros(shape, dtype=np.uint32) # keep track of how many spikes have contributed to each point in data
        for spike in self.spikes.values():
            if spike.wave.data == None: # empty WaveForm
                spike.update_wave()
            wave = spike.wave[chans] # has intersection of spike.wave.chans and chans
            # get chan indices into chans corresponding to wave.chans, chans is a superset of wave.chans
            chanis = chans.searchsorted(wave.chans)
            #chanis = [ np.where(chans==chan)[0][0] for chan in wave.chans ]
            # get timepoint indices into ts corresponding to wave.ts timepoints relative to their spike time
            tis = ts.searchsorted(wave.ts - spike.t)
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
            data[i] += wave.data.ravel() # accumulate appropriate data points
            nspikes[i] += 1 # increment spike counts at appropriate data points
        data /= nspikes # normalize each data point appropriately
        self.wave.data = data
        self.wave.chans = chans
        self.wave.ts = ts
        print 'neuron[%d].wave.chans = %r' % (self.id, chans)
        print 'neuron[%d].wave.ts = %r' % (self.id, ts)
        return self.wave

    def get_stdev(self):
        """Return 2D array of stddev of each timepoint of each chan of member spikes.
        Assumes self.update_wave has already been called"""
        data = []
        # TODO: speed this up by pre-allocating memory and then filling in the array
        for spike in self.spikes.values():
            data.append(spike.wave.data) # collect spike's data
        stdev = np.asarray(data).std(axis=0)
        return stdev

    def get_chans(self):
        return self.wave.chans # self.chans just refers to self.wave.chans

    chans = property(get_chans)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        d['plt'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        d['itemID'] = None # clear tree item ID, since that'll have changed anyway on unpickle
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

        self.lastSelectedListSpikes = []
        self.lastSelectedTreeObjects = []

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
        raise RuntimeError, "SortFrame's .sort not settable"

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
        print 'in OnSplitterSashChanged'
        wx.CallAfter(self.DrawRefs)

    def OnClose(self, evt):
        frametype = self.__class__.__name__.lower().replace('frame', '') # remove 'Frame' from class name
        self.spykeframe.HideFrame(frametype)

    def OnListSelect(self, evt):
        """Restart list selection timer
        listTimer explanation: on any selection event, start or restart the timer for say 1 msec.
        Then, when timer runs down, run self.list.GetSelections() and compare to previous list of
        selections, and execute your plots accordingly. This makes all the sel event handling fast"""
        self.listTimer.Stop()
        self.listTimer.Start(milliseconds=1, oneShot=True) # only fire one timer event after specified interval

    def OnListDeselect(self, evt):
        self.OnListSelect(evt)

    def OnListTimer(self, evt):
        """Run when started timer runs out and triggers a TimerEvent"""
        selectedRows = self.list.GetSelections()
        selectedListSpikes = [ self.sort._spikes[row] for row in selectedRows ]
        removeSpikes = [ spike for spike in self.lastSelectedListSpikes if spike not in selectedListSpikes ]
        addSpikes = [ spike for spike in selectedListSpikes if spike not in self.lastSelectedListSpikes ]
        self.RemoveObjectsFromPlot(removeSpikes)
        self.AddObjects2Plot(addSpikes)
        self.lastSelectedListSpikes = selectedListSpikes # save for next time

    def OnListRightDown(self, evt):
        """Toggle selection of the clicked list item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection ListEvent happening before the MouseEvent, or something"""
        print 'in OnListRightDown'
        pt = evt.GetPosition()
        row, flags = self.list.HitTest(pt)
        spike = self.sort._spikes[row]
        print 'spikeID is %r' % spike.id
        # this would be nice, but doesn't work (?) cuz apparently somehow the
        # selection ListEvent happens before MouseEvent that caused it:
        #selected = not self.list.IsSelected(row)
        #self.list.Select(row, on=int(not selected))
        # here is a yucky workaround:
        try:
            self.spikesortpanel.used_plots['s'+str(spike.id)] # is it plotted?
            selected = True # if so, item must be selected
            print 'spike %d in used_plots' % spike.id
        except KeyError:
            selected = False # item is not selected
            print 'spike %d not in used_plots' % spike.id
        self.list.Select(row, on=not selected) # toggle selection, this fires sel spike, which updates the plot

    def OnListColClick(self, evt):
        """Sort ._spikes according to column clicked.

        TODO: keep track of currently selected spikes and currently focused spike,
        clear the selection, then reselect those same spikes after sorting is done,
        and re-focus the same spike. Scroll into view of the focused spike (maybe
        that happens automatically). Right now, the selection remains in the list
        as-is, regardless of the entries that change beneath it"""
        col = evt.GetColumn()
        attr = self.list.COL2ATTR[col]
        s = self.sort
        # for speed, check if already sorted by attr
        if s._spikes_sorted_by == attr: # already sorted, reverse the order
            s._spikes.reverse() # in-place
            s._spikes_reversed = not s._spikes_reversed # update reversed flag
        else: # not sorted by attr
            s._spikes.sort(key=operator.attrgetter(attr)) # in-place
            s._spikes_sorted_by = attr # update
            s._spikes_reversed = False # update
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
            self.MoveCurrentSpikes2Trash()
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        evt.Skip()

    def OnTreeSelectChanged(self, evt=None):
        """Due to bugs #2307 and #626, a SEL_CHANGED event isn't fired when
        (de)selecting the currently focused item in a tree with the wx.TR_MULTIPLE
        flag set, as it is here. So, this handler has to be called manually on mouse
        and keyboard events. Don't think there's any harm in calling this handler twice
        for all other cases where a SEL_CHANGED event is fired"""
        print 'in OnTreeSelectChanged'
        if evt: # just use this as a hack to update currently focused
            item = evt.GetItem()
            if item:
                self.tree._focusedItem = item
                print 'currently focused item: %s' % self.tree.GetItemText(item)
            return # don't allow the selection event to actually happen?????????????
        self.tree._selectedItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        print [ self.tree.GetItemText(item) for item in self.tree._selectedItems ]
        selectedTreeObjects = [] # objects could be a mix of Spikes and Neurons
        for itemID in self.tree._selectedItems:
            item = self.tree.GetItemPyData(itemID)
            selectedTreeObjects.append(item)
        removeObjects = [ obj for obj in self.lastSelectedTreeObjects if obj not in selectedTreeObjects ]
        addObjects = [ obj for obj in selectedTreeObjects if obj not in self.lastSelectedTreeObjects ]
        self.RemoveObjectsFromPlot(removeObjects)
        self.AddObjects2Plot(addObjects)
        self.lastSelectedTreeObjects = selectedTreeObjects # save for next time

    def OnTreeLeftDown(self, evt):
        print 'in OnTreeLeftDown'
        pt = evt.GetPosition()
        itemID, flags = self.tree.HitTest(pt)
        if itemID.IsOk(): # if we've clicked on an item
            # leave selection event uncaught, call selection handler
            # after OS has finished doing the actual (de)selecting
            wx.CallAfter(self.OnTreeSelectChanged)

    def OnTreeLeftUp(self, evt):
        """Need this to catch clicking on non focused item, bug #4448"""
        print 'in OnTreeLeftUp'
        self.OnTreeLeftDown(evt)

    def OnTreeRightDown(self, evt):
        """Toggle selection of the clicked item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection TreeEvent happening before the MouseEvent"""
        print 'in OnTreeRightDown'
        pt = evt.GetPosition()
        itemID, flags = self.tree.HitTest(pt)
        if not itemID.IsOk(): # if we haven't clicked on an item
            return
        obj = self.tree.GetItemPyData(itemID) # either a Spike or a Neuron
        # first, restore all prior selections in the tree (except our item) that were cleared by the right click selection event
        for itemID in self.tree._selectedItems: # rely on tree._selectedItems being judiciously kept up to date
            self.tree.SelectItem(itemID)
        if obj.itemID not in self.tree._selectedItems: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(obj.itemID, select=False)
        self.OnTreeSelectChanged() # now plot accordingly

    def OnTreeSpaceUp(self, evt):
        """Toggle selection of the clicked item, without changing selection
        status of any other items. This is a nasty hack required to get around
        the selection TreeEvent happening before the SPACE KeyUp event.
        This causes an annoying flicker of the currently selected items, because they
        all unfortunately get deselected on the uncatchable SPACE keydown event,
        and aren't reselected until the SPACE keyup event"""
        print 'in OnTreeSpaceUp'
        itemID = self.tree.GetFocusedItem()
        obj = self.tree.GetItemPyData(itemID) # either a Spike or a Neuron
        # first, restore all prior selections in the tree (except our item) that were cleared by the space selection event
        for itemID in self.tree._selectedItems: # rely on tree._selectedItems being judiciously kept up to date
            self.tree.SelectItem(itemID)
        if obj.itemID not in self.tree._selectedItems: # if it wasn't selected before, it is now, so no need to do anything
            pass
        else: # it was selected before, it still will be now, so need to deselect it
            self.tree.SelectItem(obj.itemID, select=False)
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
            self.MoveCurrentObjects2List()
        elif key == ord('A'): # allow us to add from spike list even if tree is in focus
            self.MoveCurrentSpikes2Neuron(which='selected')
        elif key in [ord('C'), ord('N')]: # ditto for creating a new neuron
            self.MoveCurrentSpikes2Neuron(which='new')
        elif evt.ControlDown() and key == ord('S'):
            self.spykeframe.OnSave(evt) # give it any old event, doesn't matter
        elif key in [wx.WXK_UP, wx.WXK_DOWN]: # keyboard selection hack around multiselect bug
            wx.CallAfter(self.OnTreeSelectChanged)
            #self.OnTreeSelectChanged()
        self.tree._selectedItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
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
        self.tree._selectedItems = self.tree.GetSelections() # update list of selected tree items for OnTreeRightDown's benefit
        evt.Skip()

    def OnSortTree(self, evt):
        root = self.tree.GetRootItem()
        if root: # tree isn't empty
            self.tree.SortChildren(root)
            self.RelabelNeurons(root)
    '''
    def OnMatchNeuron(self, evt):
        """Match spikes in spike list against first selected neuron, populate err column"""
        errcol = 4 # err is in 4th column (0-based)
        neuron = self.GetFirstSelectedNeuron()
        if not neuron: # no neurons selected
            return
        self.sort.match(neurons=[neuron])
        sid2err = dict(neuron.err) # maps spike ID to its error for this neuron
        for rowi in range(self.list.GetItemCount()):
            sid = int(self.list.GetItemText(rowi))
            try:
                err = str(sid2err[sid])
            except KeyError: # no err for this sid because the spike and neuron don't overlap enough
                err = ''
            erritem = self.list.GetItem(rowi, errcol)
            erritem.SetText(err)
            self.list.SetItem(erritem)
    '''
    def RelabelNeurons(self, root):
        """Consecutively relabel neurons according to their vertical order in the TreeCtrl.
        Relabeling happens both in the TreeCtrl and in the .sort.neurons dict"""
        neurons = self.tree.GetTreeChildrenPyData(root) # get all children of root in order from top to bottom
        self.sort.neurons = {} # clear the dict, gc won't kick in cuz we still have a ref
        for neuroni, neuron in enumerate(neurons):
            neuron.id = neuroni # update its id
            self.sort.neurons[neuron.id] = neuron # add it to its key in neuron dict
            self.tree.SetItemText(neuron.itemID, 'n'+str(neuron.id)) # update its entry in the tree
        self.sort._nid = neuroni + 1 # reset unique Neuron ID counter to make next added neuron consecutive

    def DrawRefs(self):
        """Redraws refs and resaves background of sort panel(s)"""
        self.spikesortpanel.draw_refs()
        #self.chartsortpanel.draw_refs()

    def AddObjects2Plot(self, objects):
        #print 'objects to add: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.addObjects(objects)
        #self.chartsortpanel.addObjects(objects)

    def RemoveObjectsFromPlot(self, objects):
        #print 'objects to remove: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.removeObjects(objects)
        #self.chartsortpanel.removeObjects(objects)

    def UpdateObjectsInPlot(self, objects):
        #print 'objects to update: %r' % [ obj.id for obj in objects ]
        self.spikesortpanel.updateObjects(objects)
        #self.chartsortpanel.updateObjects(objects)

    # TODO: should self.OnTreeSelectChanged() (update plot) be called more often at the end of many of the following methods?:

    def CreateNeuron(self):
        """Create, select, and return a new neuron"""
        neuron = Neuron(self.sort, self.sort._nid, parent=None)
        self.sort._nid += 1 # inc for next unique neuron
        self.sort.neurons[neuron.id] = neuron # add neuron to sort session
        self.AddNeuron2Tree(neuron)
        return neuron

    def AddNeuron2Tree(self, neuron):
        """Add a neuron to the tree control"""
        root = self.tree.GetRootItem()
        if not root.IsOk(): # if tree doesn't have a valid root item
            root = self.tree.AddRoot('Neurons')
        neuron.itemID = self.tree.AppendItem(root, 'n'+str(neuron.id)) # add neuron to tree
        self.tree.SetItemPyData(neuron.itemID, neuron) # associate neuron tree item with neuron

    def RemoveNeuron(self, neuron):
        """Remove neuron and all its spikes from the tree and the sort session"""
        for spike in neuron.spikes.values():
            self.MoveSpike2List(spike)
        self.tree.Delete(neuron.itemID)
        try:
            del self.sort.neurons[neuron.id]
        except KeyError: # it's already been removed due to a recursive call
            pass

    def MoveSpike2Neuron(self, spike, row, neuron=None):
        """Move a spike spike from unsorted sort.spikes to a neuron.
        Also, move it from a list control row to a neuron in the tree.
        If neuron is None, create a new one
        """
        # make sure this spike isn't already a member of this neuron,
        # or of any other neuron
        for n in self.sort.neurons.values():
            if spike in n.spikes.values():
                print "Can't move: spike %d is identical to a member spike in neuron %d" % (spike.id, n.id)
                return
        createdNeuron = False
        if neuron == None:
            neuron = self.CreateNeuron()
            createdNeuron = True
        del self.sort.spikes[spike.id] # remove spike from unsorted sort.spikes
        self.sort.update_spike_lists()
        self.list.RefreshItems() # refresh the list
        # TODO: selection doesn't seem to be working, always jumps to top of list
        self.list.Select(row) # automatically select the new item at that position
        neuron.spikes[spike.id] = spike # add spike to neuron
        neuron.update_wave() # update mean neuron waveform
        spike.neuron = neuron # bind neuron to spike
        self.AddSpike2Tree(neuron.itemID, spike)
        if createdNeuron:
            #self.tree.Expand(root) # make sure root is expanded
            self.tree.Expand(neuron.itemID) # expand neuron
            self.tree.UnselectAll() # unselect all items in tree
            self.tree.SelectItem(neuron.itemID) # select the newly created neuron
            self.OnTreeSelectChanged() # now plot accordingly
        return neuron

    def MoveSpike2Trash(self, spike, row):
        """Move spike from spike list to trash"""
        del self.sort.spikes[spike.id] # remove spike from unsorted sort.spikes
        self.sort.update_spike_lists()
        # TODO: selection doesn't seem to be working, always jumps to top of list
        self.list.RefreshItems() # refresh the list
        self.list.Select(row) # automatically select the new item at that position
        self.sort.trash[spike.id] = spike # add it to trash
        print 'moved spike %d to trash' % spike.id

    def AddSpike2Tree(self, parent, spike):
        """Add a spike to the tree, where parent is a tree itemID"""
        spike.itemID = self.tree.AppendItem(parent, 's'+str(spike.id)) # add spike to tree, save its itemID
        self.tree.SetItemPyData(spike.itemID, spike) # associate spike tree item with spike

    def MoveSpike2List(self, spike):
        """Move a spike spike from a neuron in the tree back to the list control"""
        # make sure this spike isn't already in sort.spikes
        if spike in self.sort.spikes.values():
            # would be useful to print out the guilty spike id in the spike list, but that would require a more expensive search
            print "Can't move: spike %d (x0=%d, y0=%d, t=%d) in neuron %d is identical to an unsorted spike in the spike list" \
                  % (spike.id, int(round(spike.x0)), int(round(spike.y0)), spike.t, spike.neuron.id)
            return
        neuron = spike.neuron
        del neuron.spikes[spike.id] # del spike from its neuron's spike dict
        spike.neuron = None # unbind spike's neuron from itself
        self.sort.spikes[spike.id] = spike # restore spike to unsorted sort.spikes
        self.sort.update_spike_lists()
        # GUI operations:
        self.tree.Delete(spike.itemID)
        spike.itemID = None # no longer applicable
        if len(neuron.spikes) == 0:
            self.RemoveNeuron(neuron) # remove empty Neuron
        else:
            neuron.update_wave() # update mean neuron waveform
        self.list.RefreshItems() # refresh the list

    def MoveCurrentSpikes2Neuron(self, which='selected'):
        if which == 'selected':
            neuron = self.GetFirstSelectedNeuron()
        elif which == 'new':
            neuron = None # indicates we want a new neuron
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        for row in selected_rows:
            spike = self.sort._spikes[row]
            if spike.wave.data != None: # only move it to neuron if it's got wave data
                neuron = self.MoveSpike2Neuron(spike, row, neuron) # if neuron was None, it isn't any more
            else:
                print "can't add spike %d to neuron because its data isn't accessible" % spike.id
        if neuron != None and neuron.plt != None: # if it exists and it's plotted
            self.UpdateObjectsInPlot([neuron]) # update its plot

    def MoveCurrentObjects2List(self):
        for itemID in self.tree.GetSelections():
            if itemID: # check if spike's tree parent (neuron) has already been deleted
                obj = self.tree.GetItemPyData(itemID)
                if type(obj) == Spike:
                    self.MoveSpike2List(obj)
                elif type(obj) == Neuron:
                    self.RemoveNeuron(obj) # remove Neuron and all its Spikes
        self.OnTreeSelectChanged() # update plot

    def MoveCurrentSpikes2Trash(self):
        """Move currently selected spikes in spike list to trash"""
        selected_rows = self.list.GetSelections()
        # remove from the bottom to top, so each removal doesn't affect the row index of the remaining selections
        selected_rows.reverse()
        for row in selected_rows:
            spike = self.sort._spikes[row]
            self.MoveSpike2Trash(spike, row)

    def GetFirstSelectedNeuron(self):
        for itemID in self.tree.GetSelections():
            obj = self.tree.GetItemPyData(itemID)
            if type(obj) == Neuron:
                return obj
            # no neuron selected, check to see if an spike is selected in the tree, grab its neuron
            elif type(obj) == Spike:
                return obj.neuron
        return None


