"""Spike sorting classes and window"""

from __future__ import division
from __init__ import __version__

__authors__ = ['Martin Spacek', 'Reza Lotun']

import os
import sys
import time
import datetime
from copy import copy
import operator
import random
import shutil

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

import numpy as np
#from scipy.cluster.hierarchy import fclusterdata
#import pylab

from core import TW, WaveForm, Gaussian, MAXLONGLONG, R
from core import toiter, savez, intround, lstrip, rstrip, lrstrip, timedelta2usec
from core import SpykeToolWindow, NList, NSList, USList, ClusterChange, rmserror
from plot import SpikeSortPanel

MAXCHANTOLERANCE = 100 # um

MAINSPLITTERPOS = 300
SPIKESORTPANELWIDTHPERCOLUMN = 120
SORTWINDOWHEIGHT = 1080

MEANWAVESAMPLESIZE = 1000

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
    """A spike sorting session, in which you can detect spikes and sort them into Neurons.
    A .sort file is a single pickled Sort object"""
    DEFWAVEDATANSPIKES = 100000 # length (nspikes) to init contiguous wavedata array
    TW = TW # save a reference
    def __init__(self, detector=None, stream=None):
        self.__version__ = __version__
        self.detector = detector # this Sort's current Detector object
        self.stream = stream
        self.probe = stream.probe # only one probe design per sort allowed
        self.converter = stream.converter

        self.neurons = {}
        self.clusters = {} # neurons with multidm params scaled for plotting
        self.norder = [] # stores order of neuron ids display in nlist

        self.usids_sorted_by = 't'
        self.usids_reversed = False

    def get_nextnid(self):
        """nextnid is used to retrieve the next unique neuron ID"""
        nids = self.neurons.keys()
        if len(nids) == 0:
            return 0
        else:
            return max(nids) + 1

    nextnid = property(get_nextnid)

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
        twts = np.arange(self.TW[0], self.TW[1], tres) # temporal window timepoints wrt thresh xing or spike time
        twts += twts[0] % tres # get rid of mod, so twts go through zero
        self.twts = twts
        # time window indices wrt thresh xing or 1st phase:
        self.twi = intround(twts[0] / tres), intround(twts[-1] / tres)
        #info('twi = %s' % (self.twi,))

    stream = property(get_stream, set_stream)

    def __getstate__(self):
        """Get object state for pickling"""
        # copy it cuz we'll be making changes, this is fast because it's just a shallow copy
        d = self.__dict__.copy()
        # Don't pickle the stream, cuz it relies on an open .srf file.
        # Spikes and wavedata arrays are (potentially) saved separately.
        # usids can be regenerated from the spikes array.
        for attr in ['spikes', 'wavedata', 'usids']:
            # keep _stream during normal pickling for multiprocessing, but remove it
            # manually when pickling to .sort
            try: del d[attr]
            except KeyError: pass
        return d

    def get_nspikes(self):
        try: return len(self.spikes)
        except AttributeError: return 0

    nspikes = property(get_nspikes)

    def update_usids(self):
        """Update usids, which is an array of struct array indices of unsorted spikes"""
        nids = self.spikes['nid']
        self.usids, = np.where(nids == -1) # -1 indicates spike has no nid assigned to it
        # FIXME: disable sorting for now
        # order it by .usids_sorted_by and .usids_reversed
        #if self.usids_sorted_by != 't': self.sort_usids('t')
        #if self.usids_reversed: self.reverse_usids()

    def sort_usids(self, sort_by):
        """Sort struct array row indices of unsorted spikes according to
        sort_by"""
        vals = self.spikes[self.usids][sort_by] # vals from just the unsorted rows and the desired column
        self.usids = self.usids[vals.argsort()] # usids are now sorted by sorty_by
        self.usids_sorted_by = sort_by # update

    def reverse_usids(self):
        """Reverse usids"""
        # is there a way to reverse an array in-place, like a list?
        # maybe swap the start and end points, and set stride to -1?
        self.usids = self.usids[::-1]

    def get_spikes_sortedby(self, attr='id'):
        """Return array of all spikes, sorted by attribute 'attr'"""
        vals = self.spikes[attr]
        spikes = self.spikes[vals.argsort()]
        return spikes

    def get_wave(self, sid):
        """Return WaveForm corresponding to spike sid"""
        spikes = self.spikes
        nchans = spikes['nchans'][sid]
        chans = spikes['chans'][sid, :nchans]
        t0 = spikes['t0'][sid]
        t1 = spikes['t1'][sid]
        try:
            wavedata = self.wavedata[sid, 0:nchans]
            ts = np.arange(t0, t1, self.tres) # build them up
            return WaveForm(data=wavedata, ts=ts, chans=chans)
        except AttributeError: pass

        # try getting it from the stream
        if self.stream == None:
            raise RuntimeError("No stream open, can't get wave for %s %d" %
                               (spikes[sid], sid))
        det = self.detector
        if det.srffname != self.stream.srffname:
            msg = ("Spike %d was extracted from .srf file %s.\n"
                   "The currently opened .srf file is %s.\n"
                   "Can't get spike %d's wave" %
                   (sid, det.srffname, self.stream.srffname, sid))
            raise RuntimeError(msg)
        wave = self.stream[t0:t1]
        return wave[chans]

    def exportptcsfiles(self, basepath):
        """Export spike data to binary .ptcs files under basepath, one file per recording"""
        spikes = self.spikes
        dt = str(datetime.datetime.now()) # get an export datetime stamp
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        srffnames = self.stream.srffnames
        try: # self.stream is a TrackStream?
            streamtranges = self.stream.streamtranges # includes offsets
        except AttributeError: # self.stream is a normal Stream
            streamtranges = [[self.stream.t0, self.stream.t1]]
        print('exporting clustered spikes to:')
        # do a separate export for each recording
        for srffname, streamtrange in zip(srffnames, streamtranges):
            self.exportptcsfile(dt, srffname, streamtrange, basepath)

    def exportptcsfile(self, dt, srffname, streamtrange, basepath):
        """Export spike data to binary .ptcs file in basepath, constrain spikes
        to given streamtrange"""
        userdescr = ''
        nsamplebytes = 4
        # build up list of PTCSNeuronRecords that have spikes in this streamrange,
        # and tally their spikes
        recs = []
        nspikes = 0
        nids = sorted(self.neurons)
        for nid in nids:
            neuron = self.neurons[nid]
            spikets = self.spikes['t'][neuron.sids] # should be sorted
            # constrain to spikes within streamtrange
            lo, hi = spikets.searchsorted(streamtrange)
            spikets = spikets[lo:hi]
            if len(spikets) == 0:
                continue # don't save empty neurons
            rec = PTCSNeuronRecord(neuron, spikets, nsamplebytes=nsamplebytes)
            recs.append(rec)
            nspikes += len(spikets)
        nneurons = len(recs)

        # write the file
        srffnameroot = lrstrip(srffname, '../', '.srf')
        path = os.path.join(basepath, srffnameroot)
        try: os.mkdir(path)
        except OSError: pass # path already exists?
        fname = dt + '.ptcs'
        fullfname = os.path.join(path, fname)
        with open(fullfname, 'wb') as f:
            header = PTCSHeader(self, nneurons, nspikes, userdescr, nsamplebytes,
                                fullfname, dt, srffname)
            header.write(f)
            for rec in recs:
                rec.write(f)
        print(fullfname)

    def exportspkfiles(self, basepath):
        """Export spike data to binary .spk files under basepath, one file per neuron"""
        spikes = self.spikes
        dt = str(datetime.datetime.now()) # get an export datetime stamp
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        spikefoldername = dt + '.best.sort'
        srffnames = self.stream.srffnames
        try: # self.stream is a TrackStream?
            streamtranges = self.stream.streamtranges # includes offsets
        except AttributeError: # self.stream is a normal Stream
            streamtranges = [[self.stream.t0, self.stream.t1]]
        print('exporting clustered spikes to:')
        # do a separate export for each recording
        for srffname, streamtrange in zip(srffnames, streamtranges):
            srffnameroot = lrstrip(srffname, '../', '.srf')
            path = os.path.join(basepath, srffnameroot)
            try: os.mkdir(path)
            except OSError: pass # path already exists?
            # if any existing folders in srffname path end with the name '.best.sort',
            # then remove the '.best' from their name
            for name in os.listdir(path):
                fullname = os.path.join(path, name)
                if os.path.isdir(fullname) and fullname.endswith('.best.sort'):
                    #os.rename(fullname, rstrip(fullname, '.best.sort') + '.sort')
                    shutil.rmtree(fullname) # aw hell, just delete them to minimize junk
            path = os.path.join(path, spikefoldername)
            os.mkdir(path)
            for nid, neuron in self.neurons.items():
                spikets = spikes['t'][neuron.sids] # should be sorted
                # limit to spikes within streamtrange
                lo, hi = spikets.searchsorted(streamtrange)
                spikets = spikets[lo:hi]
                if len(spikets) == 0:
                    continue # don't generate 0 byte files
                # pad filename with leading zeros to always make template ID 3 digits long
                neuronfname = '%s_t%03d.spk' % (dt, nid)
                spikets.tofile(os.path.join(path, neuronfname)) # save it
            print(path)

    def exporttschid(self, basepath):
        """Export int64 (timestamp, channel, neuron id) 3 tuples to binary file"""
        raise NotImplementedError('needs to be redone to work with multiple streams')
        spikes = self.spikes[self.spikes['nid'] != -1] # probably shouldn't export unsorted spikes
        dt = str(datetime.datetime.now()) # get an export timestamp
        dt = dt.split('.')[0] # ditch the us
        dt = dt.replace(' ', '_')
        dt = dt.replace(':', '.')
        srffnameroot = srffnameroot.replace(' ', '_')
        tschidfname = dt + '_' + srffnameroot + '.tschid'
        tschid = np.empty((len(spikes), 3), dtype=np.int64)
        tschid[:, 0] = spikes['t']
        tschid[:, 1] = spikes['chan']
        tschid[:, 2] = spikes['nid']
        tschid.tofile(os.path.join(path, tschidfname)) # save it
        print(tschidfname)

    def exportdin(self, basepath):
        """Export stimulus din(s) to binary .din file(s) in basepath"""
        try: # self.stream is a TrackStream?
            streams = self.stream.streams
        except AttributeError: # self.stream is a normal Stream
            streams = [self.stream]
        dinfiledtype=[('TimeStamp', '<i8'), ('SVal', '<i8')] # pairs of int64s
        print('exporting DIN(s) to:')
        for stream in streams:
            digitalsvalrecords = stream.srff.digitalsvalrecords
            if len(digitalsvalrecords) == 0: # no din to export for this stream
                continue
            srffnameroot = lrstrip(stream.srff.fname, '../', '.srf')
            path = os.path.join(basepath, srffnameroot)
            try: os.mkdir(path)
            except OSError: pass # path already exists?
            dinfname = srffnameroot + '.din'
            fullfname = os.path.join(path, dinfname)
            # upcast SVal field from uint16 to int64, creates a copy, but it's not too expensive
            digitalsvalrecords = digitalsvalrecords.astype(dinfiledtype)
            # convert to normal n x 2 int64 array
            digitalsvalrecords = digitalsvalrecords.view(np.int64).reshape(-1, 2)
            # calculate offset for din values, get time delta between stream i and stream 0
            td = timedelta2usec(stream.datetime - streams[0].datetime)
            digitalsvalrecords[:, 0] += td # add offset
            digitalsvalrecords.tofile(fullfname) # save it
            print(fullfname)

    def exporttextheader(self, basepath):
        """Export stimulus text header(s) to .textheader file(s) in basepath"""
        try: # self.stream is a TrackStream?
            streams = self.stream.streams
        except AttributeError: # self.stream is a normal Stream
            streams = [self.stream]
        print('exporting text header(s) to:')
        for stream in streams:
            displayrecords = stream.srff.displayrecords
            if len(displayrecords) == 0: # no textheader to export for this stream
                continue
            if len(displayrecords) > 1:
                print("*** WARNING: multiple display records for file %r\n"
                      "Exporting textheader from only the first display record"
                      % stream.srff.fname)
            srffnameroot = lrstrip(stream.srff.fname, '../', '.srf')
            path = os.path.join(basepath, srffnameroot)
            try: os.mkdir(path)
            except OSError: pass # path already exists?
            textheader = displayrecords[0].Header.python_tbl
            textheaderfname = srffnameroot + '.textheader'
            fullfname = os.path.join(path, textheaderfname)
            f = open(fullfname, 'w')
            f.write(textheader) # save it
            f.close()
            print(fullfname)

    def exportall(self, basepath):
        """Export spike data, stimulus textheader, and din to path in
        the classic way for use in neuropy"""
        self.exportspikes(basepath)
        self.exportdin(basepath)
        self.exporttextheader(basepath)

    def exportlfp(self, basepath):
        """Export LFP data to binary .lfp file"""
        raise NotImplementedError('needs to be redone to work with multiple streams')
        srffnameroot = srffnameroot.replace(' ', '_')
        lfpfname = srffnameroot + '.lfp'
        lps = lpstream
        wave = lps[lps.t0:lps.t1]
        uVperAD = lps.converter.AD2uV(1)
        savez(os.path.join(path, lfpfname), compress=True,
              data=wave.data, chans=wave.chans,
              t0=lps.t0, t1=lps.t1, tres=lps.tres, # for easy ts reconstruction
              uVperAD=uVperAD) # save it
        print(lfpfname)

    def get_param_matrix(self, dims=None, scale=True):
        """Organize parameters in dims from all spikes into a data matrix,
        each column corresponds to a dim"""
        # np.column_stack returns a copy, not modifying the original array
        data = []
        for dim in dims:
            data.append( np.float32(self.spikes[dim]) )
        data = np.column_stack(data)
        if scale:
            x0std = self.spikes['x0'].std()
            assert x0std != 0
            for dim, d in zip(dims, data.T): # d iterates over columns
                d -= d.mean()
                if dim in ['x0', 'y0']:
                    d /= x0std
                #elif dim == 't': # the longer the recording in hours, the greater the scaling in time
                #    trange = d.max() - d.min()
                #    tscale = trange / (60*60*1e6)
                #    d *= tscale / d.std()
                else: # normalize all other dims by their std
                    d /= d.std()
        return data

    def create_neuron(self, id=None, inserti=None):
        """Create and return a new Neuron with a unique ID"""
        if id == None:
            id = self.nextnid
        if id in self.neurons:
            raise RuntimeError('Neuron %d already exists' % id)
        id = int(id) # get rid of numpy ints
        neuron = Neuron(self, id)
        # add neuron to self
        self.neurons[neuron.id] = neuron
        if inserti == None:
            self.norder.append(neuron.id)
        else:
            self.norder.insert(inserti, neuron.id)
        return neuron

    def remove_neuron(self, id):
        try:
            del self.neurons[id] # may already be removed due to recursive call
            del self.clusters[id]
            self.norder.remove(id)
        except KeyError, ValueError:
            pass
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
        ti = intround(self.spikes[0].wave.data.shape[-1] / 4) # 13 for 50 kHz, 6 for 25 kHz
        dims = self.nspikes, 2+nchans*npoints
        output = np.empty(dims, dtype=np.float32)
        dm = self.detector.dm
        chanis = np.arange(len(dm.data))
        coords = np.asarray(dm.coords)
        xcoords = coords[:, 0]
        ycoords = coords[:, 1]
        sids = self.spikes.keys() # self.spikes is a dict!
        sids.sort()
        for sid in sids:
            spike = self.spikes[sid]
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
                          % (chani, nchans, x0, y0, sid, spike.t))
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
            output[sid] = row
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
                template.err.append((spike.id, intround(err)))
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
        self.sids = np.array([], dtype=int) # indices of spikes that make up this neuron
        self.t = 0 # relative reference timestamp, here for symmetry with fellow spike rec (obj.t comes up sometimes)
        self.plt = None # Plot currently holding self
        self.cluster = None
        #self.srffname # not here, let's allow neurons to have spikes from different files?

    def get_chans(self):
        if self.wave.data == None:
            self.update_wave()
        return self.wave.chans # self.chans just refers to self.wave.chans

    chans = property(get_chans)

    def get_chan(self):
        if self.wave.data == None:
            self.update_wave()
        return self.wave.chans[self.wave.data.ptp(axis=1).argmax()] # chan with max Vpp

    chan = property(get_chan)

    def get_nspikes(self):
        return len(self.sids)

    nspikes = property(get_nspikes)

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy()
        d['plt'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        return d

    def update_wave(self):
        """Update mean waveform"""
        sort = self.sort
        spikes = sort.spikes
        if len(self.sids) == 0: # no member spikes, perhaps I should be deleted?
            raise RuntimeError("neuron %d has no spikes and its waveform can't be updated" % self.id)
            #self.wave = WaveForm() # empty waveform
            #return self.wave
        sids = self.sids
        if len(sids) > MEANWAVESAMPLESIZE:
            print('neuron %d: update_wave() taking random sample of %d spikes instead '
                  'of all %d of them' % (self.id, MEANWAVESAMPLESIZE, len(sids)))
            sids = np.asarray(random.sample(sids, MEANWAVESAMPLESIZE))

        chanss = spikes['chans'][sids]
        nchanss = spikes['nchans'][sids]
        chanslist = [ chans[:nchans] for chans, nchans in zip(chanss, nchanss) ] # list of arrays
        chanpopulation = np.concatenate(chanslist)
        neuronchans = np.unique(chanpopulation)

        wavedata = sort.wavedata[sids]
        if wavedata.ndim == 2: # should be 3, get only 2 if len(sids) == 1
            wavedata.shape = 1, wavedata.shape[0], wavedata.shape[1] # give it a singleton 3rd dim
        maxnt = wavedata.shape[-1]
        maxnchans = len(neuronchans)
        data = np.zeros((maxnchans, maxnt))
        # all spike have same nt, but not necessarily nchans, keep track of
        # how many spikes contributed to each of neuron's chans
        nspikes = np.zeros((maxnchans, 1), dtype=int)
        for chans, wd in zip(chanslist, wavedata):
            chanis = neuronchans.searchsorted(chans) # each spike's chans is a subset of neuronchans
            data[chanis] += wd[:len(chans)] # accumulate
            nspikes[chanis] += 1 # inc spike count for this spike's chans
        #t0 = time.time()
        data /= nspikes # normalize all data points appropriately
        # keep only those chans that at least 1/2 the spikes contributed to
        bins = list(neuronchans) + [sys.maxint] # concatenate rightmost bin edge
        hist, bins = np.histogram(chanpopulation, bins=bins)
        newneuronchans = neuronchans[hist >= len(sids)/2]
        chanis = neuronchans.searchsorted(newneuronchans)
        # update this Neuron's Waveform object
        self.wave.data = data[chanis]
        self.wave.chans = newneuronchans
        self.wave.ts = sort.twts
        return self.wave

    def __sub__(self, other):
        """Return difference array between self and other neurons' waveforms
        on common channels"""
        selfwavedata, otherwavedata = self.getCommonWaveData(other)
        return selfwavedata - otherwavedata

    def getCommonWaveData(self, other):
        chans = np.intersect1d(self.chans, other.chans, assume_unique=True)
        if len(chans) == 0:
            raise ValueError('no common chans')
        if self.chan not in chans or other.chan not in chans:
            raise ValueError("maxchans aren't part of common chans")
        selfchanis = self.chans.searchsorted(chans)
        otherchanis = other.chans.searchsorted(chans)
        return self.wave.data[selfchanis], other.wave.data[otherchanis]

    def align(self, to):
        if to == 'best':
            self.alignbest()
        else:
            self.alignminmax(to)
        self.update_wave() # update mean waveform
        # trigger resaving of .wave file on next save
        try: del self.sort.wavefname
        except AttributeError: pass

    def alignbest(self, method=rmserror):
        """Align all of this neuron's spikes by best fit"""
        s = self.sort
        # TODO: make maxshift a f'n of interpolation factor
        nt = s.wavedata.shape[2] # num timepoints in each waveform
        maxshift = 2 # shift +/- this many timepoints
        maxshiftus = maxshift * s.stream.tres
        shifts = range(-maxshift, maxshift+1) # from -maxshift to maxshift, inclusive
        print('neuron %d stdev before alignbest: %f uV' % (self.id, self.stdevwaveerrors()))
        srffopen = s.stream.is_open()
        if not srffopen:
            print("WARNING: .srf file(s) not available, padding waveforms with up to +/- %d "
                  "points of fake data" % maxshift)
        for sid in self.sids:
            # TODO: use at most only maxchan and immediate neighbour chans
            spike = s.spikes[sid]
            nspikechans = spike['nchans']
            spikechans = spike['chans'][:nspikechans]
            spikedata = s.wavedata[sid, :nspikechans]
            # get chans common to neuron and spike
            chans = np.intersect1d(self.chans, spikechans, assume_unique=True)
            spikechanis = spikechans.searchsorted(chans)
            neuronchanis = self.chans.searchsorted(chans)
            # widespikedata holds spikedata plus extra data on either side
            # to allow for full width slicing for all time shifts:
            if srffopen:
                wave = s.stream[-maxshiftus+spike['t0'] : spike['t1']+maxshiftus]
                wavechanis = wave.chans.searchsorted(spikechans)
                widespikedata = wave.data[wavechanis]
            else:
                # Only add fake values at start and end if .srf file isn't available.
                # Problem is fake values will make stdev keep improving indefinitely, until you've
                # done so many shifts, there potentially isn't any real data left for some spikes
                widespikedata = np.zeros((nspikechans, maxshift+nt+maxshift))
                widespikedata[:, maxshift:-maxshift] = spikedata
                widespikedata[:, :maxshift] = spikedata[:, 0, None] # pad start with first point per chan
                widespikedata[:, -maxshift:] = spikedata[:, -1, None] # pad end with last point per chan
            widespikesubdata = widespikedata[spikechanis]
            neuronsubdata = self.wave.data[neuronchanis]
            errors = np.zeros(len(shifts)) # init
            for shifti, shift in enumerate(shifts):
                t0i = maxshift + shift
                shiftedspikesubdata = widespikesubdata[:, t0i:t0i+nt]
                errors[shifti] = method(shiftedspikesubdata, neuronsubdata)
            bestshift = shifts[errors.argmin()]
            if bestshift != 0: # no need to update sort.wavedata[sid] if there's no shift
                t0i = maxshift + bestshift
                s.wavedata[sid][:nspikechans] = widespikedata[:, t0i:t0i+nt]
        print('neuron %d stdev after alignbest: %f uV' % (self.id, self.stdevwaveerrors()))

    def waveerrors(self, method=rmserror):
        """Return array of differences between self and all member spikes"""
        s = self.sort
        errors = np.zeros(self.nspikes)
        for spikei, sid in enumerate(self.sids):
            spike = s.spikes[sid]
            nspikechans = spike['nchans']
            spikechans = spike['chans'][:nspikechans]
            spikedata = s.wavedata[sid, :nspikechans]
            # get chans common to neuron and spike
            chans = np.intersect1d(self.chans, spikechans, assume_unique=True)
            spikechanis = spikechans.searchsorted(chans)
            neuronchanis = self.chans.searchsorted(chans)
            spikesubdata = spikedata[spikechanis]
            neuronsubdata = self.wave.data[neuronchanis]
            errors[spikei] = method(spikesubdata, neuronsubdata)
        return errors

    def stdevwaveerrors(self, method=rmserror):
        """Return stdev of difference between self and all member spikes, in uV"""
        stdev = self.waveerrors(method).std()
        return self.sort.converter.AD2uV(stdev)

    def alignminmax(self, to):
        """Align all of this neuron's spikes by their min or max"""
        s = self.sort
        spikes = s.spikes
        nsids = self.sids # ids of spikes that belong to this neuron

        V0s = spikes['V0'][nsids]
        V1s = spikes['V1'][nsids]
        Vss = np.column_stack((V0s, V1s))
        alignis = spikes['aligni'][nsids]
        b = np.column_stack((alignis==0, alignis==1)) # 2D boolean array
        if to == 'min':
            i = Vss[b] > 0 # indices into nsids of spikes aligned to the max phase
        elif to == 'max':
            i = Vss[b] < 0 # indices into nsids of spikes aligned to the min phase
        else:
            raise ValueError('unknown to %r' % to)
        sids = nsids[i] # ids of spikes that need realigning
        n = len(sids) # num spikes that need realigning
        print("Realigning %d spikes" % n)
        if n == 0: # nothing to do
            return

        multichanphasetis = spikes['phasetis'][sids] # n x nchans x 2 arr
        chanis = spikes['chani'][sids] # len(n) arr
        # phasetis of max chan of each spike, convert from uint8 to int32 for safe math
        phasetis = np.int32(multichanphasetis[np.arange(n), chanis]) # n x 2 arr
        # NOTE: phasetis aren't always in temporal order!
        dphasetis = phasetis[:, 1] - phasetis[:, 0] # could be +ve or -ve
        dphases = spikes['dphase'][sids] # stored as +ve

        # for each spike, decide whether to add or subtract dphase to/from its temporal values
        ordered  = dphasetis > 0 # in temporal order
        reversed = dphasetis < 0 # in reversed temporal order
        alignis = spikes['aligni'][sids]
        alignis0 = alignis == 0
        alignis1 = alignis == 1
        dphasei = np.zeros(n, dtype=int)
        # add dphase to temporal values to align to later phase
        dphasei[ordered & alignis0 | reversed & alignis1] = 1
        # subtact dphase from temporal values to align to earlier phase
        dphasei[ordered & alignis1 | reversed & alignis0] = -1

        #dalignis = -np.int32(alignis)*2 + 1 # upcast aligni from 1 byte to an int before doing arithmetic on it
        dts = dphasei * dphases
        dtis = -dphasei * abs(dphasetis)
        # shift values
        spikes['t'][sids] += dts
        spikes['t0'][sids] += dts
        spikes['t1'][sids] += dts
        spikes['phasetis'][sids] += dtis[:, None, None] # update wrt new t0i
        spikes['aligni'][sids[alignis0]] = 1
        spikes['aligni'][sids[alignis1]] = 0

        # update wavedata for each shifted spike
        for sid, spike in zip(sids, spikes[sids]):
            wave = s.stream[spike['t0']:spike['t1']]
            nchans = spike['nchans']
            chans = spike['chans'][:nchans]
            wave = wave[chans]
            s.wavedata[sid, 0:nchans] = wave.data
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
    '''

class PTCSHeader(object):
    """
    Polytrode clustered spikes file header:
    
    formatversion: int64 (start at version 1)
    ndescrbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    descr: ndescrbytes of ASCII text
        (padded with spaces if needed for 8 byte alignment)
    nneurons: uint64 (number of neurons)
    nspikes: uint64 (total number of spikes)
    nsamplebytes: uint64 (number of bytes per template waveform sample)
    samplerate: float64 (Hz)
    """
    FORMATVERSION = 1 # overall .ptcs file format version, not header format version
    def __init__(self, sort, nneurons, nspikes, userdescr, nsamplebytes,
                 fullfname, dt, srffname):
        self.sort = sort
        self.nneurons = nneurons
        self.nspikes = nspikes
        self.userdescr = userdescr
        self.nsamplebytes = nsamplebytes
        homelessfullfname = lstrip(fullfname, os.path.expanduser('~'))
        # For description dictionary, could create a dict and convert it
        # to a string, but that wouldn't guarantee key order. Instead,
        # build string rep of description dict with guaranteed key order:
        d = ("{'file_type': '.ptcs (polytrode clustered spikes) file', "
             "'original_fname': %r, 'extraction_datetime': %r, "
             "'recording_fname': %r, 'electrode_name': %r"
             % (homelessfullfname, dt, srffname, sort.stream.probe.name))
        if userdescr:
            d += ", 'user_descr': %r" % userdescr
        d += "}"
        d = d.encode('ascii') # ensure it's pure ASCII
        rem = len(d) % 8
        npad = 8 - rem if rem else 0 # num spaces to pad with for 8 byte alignment
        d += ' ' * npad
        assert len(d) % 8 == 0
        self.descr = d

    def write(self, f):
        s = self.sort
        np.int64(self.FORMATVERSION).tofile(f) # formatversion
        np.uint64(len(self.descr)).tofile(f) # ndescrbytes
        f.write(self.descr) # descr
        np.uint64(self.nneurons).tofile(f) # nneurons
        np.uint64(self.nspikes).tofile(f) # nspikes
        np.uint64(self.nsamplebytes).tofile(f) # nsamplebytes
        np.float64(s.sampfreq).tofile(f) # samplerate


class PTCSNeuronRecord(object):
    """
    Polytrode clustered spikes file neuron record:
    
    nid: int64 (signed neuron id, could be -ve, could be non-contiguous with previous)
    ptid: int64 (polytrode/tetrode/electrode ID, for multi electrode recordings,
                 defaults to -1)
    ndescrbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment, defaults to 0)
    descr: ndescrbytes of ASCII text
        (padded with spaces if needed for 8 byte alignment)
    clusterscore: float64
    xpos: float64 (um)
    ypos: float64 (um)
    zpos: float64 (um) (defaults to NaN)
    nchans: uint64 (num chans in template waveforms)
    chans: nchans * uint64 (IDs of channels in template waveforms)
    maxchan: uint64 (ID of max channel in template waveforms)
    nt: uint64 (num timepoints per template waveform channel)
    nwavedatabytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    wavedata: nchans * nt * nsamplebytes
        (float template waveform data, in uV, padded with zeros if
         needed for 8 byte alignment)
    nspikes: uint64 (number of spikes in this neuron)
    spike timestamps: nspikes * uint64 (us, should be sorted)
    """
    def __init__(self, neuron, spikets=None, descr='', nsamplebytes=None):
        self.neuron = neuron
        self.spikets = spikets # constrained to stream range, may be < neuron.sids
        self.descr = descr
        assert len(self.descr) % 8 == 0
        self.wavedtype = {2: np.float16, 4: np.float32, 8: np.float64}[nsamplebytes]
        nbytes = self.wavedtype(neuron.wave.data).nbytes
        rem = nbytes % 8
        self.nwavedatapadbytes = 8 - rem if rem else 0
        self.nwavedatabytes = nbytes + self.nwavedatapadbytes
        assert self.nwavedatabytes % 8 == 0
        
    def write(self, f):
        n = self.neuron
        AD2uV = n.sort.converter.AD2uV
        np.int64(n.id).tofile(f) # nid
        np.int64(-1).tofile(f) # ptid
        np.uint64(len(self.descr)).tofile(f) # ndescrbytes
        f.write(self.descr) # descr
        np.float64(np.nan).tofile(f) # clusterscore
        np.float64(n.cluster.pos['x0']).tofile(f) # xpos (um)
        np.float64(n.cluster.pos['y0']).tofile(f) # ypos (um)
        np.float64(np.nan).tofile(f) # zpos (um)
        np.uint64(len(n.wave.chans)).tofile(f) # nchans
        np.uint64(n.wave.chans).tofile(f) # chans
        np.uint64(n.chan).tofile(f) # maxchan
        np.uint64(len(n.wave.ts)).tofile(f) # nt
        np.uint64(self.nwavedatabytes).tofile(f) # nwavedatabytes
        self.wavedtype(AD2uV(n.wave.data)).tofile(f) # wavedata (uV, nchans * nt * nsamplebytes)
        np.zeros(self.nwavedatapadbytes, dtype=np.uint8).tofile(f) # 0 padding
        np.uint64(len(self.spikets)).tofile(f) # nspikes
        np.uint64(self.spikets).tofile(f) # spike timestamps (us)


class SortWindow(SpykeToolWindow):
    """Sort window"""
    def __init__(self, parent, pos=None):
        SpykeToolWindow.__init__(self, parent, flags=QtCore.Qt.Tool)
        self.spykewindow = parent
        ncols = self.sort.probe.ncols
        size = (MAINSPLITTERPOS + SPIKESORTPANELWIDTHPERCOLUMN * ncols, SORTWINDOWHEIGHT)
        self.setWindowTitle("Sort Window")
        self.move(*pos)
        self.resize(*size)

        toolbar = self.setupToolbar()

        self._source = None # source cluster for comparison
        self.nlist = NList(self)
        self.nlist.setToolTip('Neuron list')
        self.nslist = NSList(self)
        self.nslist.setToolTip('Sorted spike list')
        self.uslist = USList(self) # should really be multicolumn tableview
        self.uslist.setToolTip('Unsorted spike list')
        self.panel = SpikeSortPanel(self)

        self.hsplitter = QtGui.QSplitter(Qt.Horizontal)
        self.hsplitter.addWidget(self.nlist)
        self.hsplitter.addWidget(self.nslist)

        self.vsplitter = QtGui.QSplitter(Qt.Vertical)
        self.vsplitter.addWidget(self.hsplitter)
        self.vsplitter.addWidget(self.uslist)

        self.mainsplitter = QtGui.QSplitter(Qt.Horizontal)
        self.mainsplitter.addWidget(self.vsplitter)
        self.mainsplitter.addWidget(self.panel)
        #self.mainsplitter.moveSplitter(MAINSPLITTERPOS, 1) # only works after self is shown

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.mainsplitter)

        mainwidget = QtGui.QWidget(self)
        mainwidget.setLayout(layout)
        self.setCentralWidget(mainwidget)

        #QtCore.QMetaObject.connectSlotsByName(self)

    def setupToolbar(self):
        toolbar = QtGui.QToolBar("toolbar", self)
        toolbar.setFloatable(True)

        actionDeleteClusters = QtGui.QAction("Del", self)
        actionDeleteClusters.setToolTip('Delete cluster')
        self.connect(actionDeleteClusters, QtCore.SIGNAL("triggered()"),
                     self.on_actionDeleteClusters_triggered)
        toolbar.addAction(actionDeleteClusters)

        actionMergeClusters = QtGui.QAction("M", self)
        actionMergeClusters.setToolTip('Merge clusters')
        self.connect(actionMergeClusters, QtCore.SIGNAL("triggered()"),
                     self.on_actionMergeClusters_triggered)
        toolbar.addAction(actionMergeClusters)

        actionChanSplitClusters = QtGui.QAction("/", self)
        actionChanSplitClusters.setToolTip('Split clusters by channels')
        self.connect(actionChanSplitClusters, QtCore.SIGNAL("triggered()"),
                     self.on_actionChanSplitClusters_triggered)
        toolbar.addAction(actionChanSplitClusters)

        toolbar.addSeparator()

        actionRenumberClusters = QtGui.QAction("#", self)
        actionRenumberClusters.setToolTip('Renumber clusters')
        self.connect(actionRenumberClusters, QtCore.SIGNAL("triggered()"),
                     self.on_actionRenumberClusters_triggered)
        toolbar.addAction(actionRenumberClusters)

        toolbar.addSeparator()

        actionFocusCurrentCluster = QtGui.QAction("C", self)
        actionFocusCurrentCluster.setToolTip('Focus current cluster')
        self.connect(actionFocusCurrentCluster, QtCore.SIGNAL("triggered()"),
                     self.on_actionFocusCurrentCluster_triggered)
        toolbar.addAction(actionFocusCurrentCluster)

        actionFocusCurrentSpike = QtGui.QAction("X", self)
        actionFocusCurrentSpike.setToolTip('Focus current spike')
        self.connect(actionFocusCurrentSpike, QtCore.SIGNAL("triggered()"),
                     self.on_actionFocusCurrentSpike_triggered)
        toolbar.addAction(actionFocusCurrentSpike)

        actionSelectRandomSpikes = QtGui.QAction("R", self)
        actionSelectRandomSpikes.setToolTip('Select random sample of spikes of current cluster')
        self.connect(actionSelectRandomSpikes, QtCore.SIGNAL("triggered()"),
                     self.on_actionSelectRandomSpikes_activated)
        toolbar.addAction(actionSelectRandomSpikes)

        nsamplesComboBox = QtGui.QComboBox(self)
        nsamplesComboBox.setToolTip('Number of spikes per cluster to randomly select')
        nsamplesComboBox.setFocusPolicy(Qt.NoFocus)
        nsamplesComboBox.addItems(['1', '5', '10', '20', '50'])
        nsamplesComboBox.setCurrentIndex(3)
        toolbar.addWidget(nsamplesComboBox)
        self.connect(nsamplesComboBox, QtCore.SIGNAL("activated(int)"),
                     self.on_actionSelectRandomSpikes_activated)
        self.nsamplesComboBox = nsamplesComboBox

        toolbar.addSeparator()

        actionAlignMin = QtGui.QAction("Align min", self)
        actionAlignMin.setToolTip("Align neurons' spikes to min")
        self.connect(actionAlignMin, QtCore.SIGNAL("triggered()"),
                     self.on_actionAlignMin_triggered)
        toolbar.addAction(actionAlignMin)

        actionAlignMax = QtGui.QAction("Align max", self)
        actionAlignMax.setToolTip("Align neurons' spikes to max")
        self.connect(actionAlignMax, QtCore.SIGNAL("triggered()"),
                     self.on_actionAlignMax_triggered)
        toolbar.addAction(actionAlignMax)

        actionAlignBest = QtGui.QAction("Align best", self)
        actionAlignBest.setToolTip("Align neurons' spikes by best fit")
        self.connect(actionAlignBest, QtCore.SIGNAL("triggered()"),
                     self.on_actionAlignBest_triggered)
        toolbar.addAction(actionAlignBest)

        toolbar.addSeparator()

        actionFindPrevMostSimilar = QtGui.QAction("<", self)
        actionFindPrevMostSimilar.setToolTip("Find previous most similar cluster")
        self.connect(actionFindPrevMostSimilar, QtCore.SIGNAL("triggered()"),
                     self.on_actionFindPrevMostSimilar_triggered)
        toolbar.addAction(actionFindPrevMostSimilar)

        actionFindNextMostSimilar = QtGui.QAction(">", self)
        actionFindNextMostSimilar.setToolTip("Find next most similar cluster")
        self.connect(actionFindNextMostSimilar, QtCore.SIGNAL("triggered()"),
                     self.on_actionFindNextMostSimilar_triggered)
        toolbar.addAction(actionFindNextMostSimilar)

        return toolbar

    def get_sort(self):
        return self.spykewindow.sort

    sort = property(get_sort) # make this a property for proper behaviour after unpickling

    def closeEvent(self, event):
        self.spykewindow.HideWindow('Sort')

    def keyPressEvent(self, event):
        """Alpha character keypresses are by default caught by the child lists for quickly
        scrolling down to and selecting list items. However, the appropriate alpha keypresses have
        been set in the child lists to be ignored, so they propagate up to here"""
        key = event.key()
        if key == Qt.Key_Escape: # deselect all spikes and all clusters
            self.clear()
        elif key in [Qt.Key_Delete, Qt.Key_D]: # D ignored in SpykeListViews
            self.on_actionDeleteClusters_triggered()
        elif key == Qt.Key_M: # ignored in SpykeListViews
            self.on_actionMergeClusters_triggered()
        elif key == Qt.Key_Slash: # ignored in SpykeListViews
            self.on_actionChanSplitClusters_triggered()
        elif key == Qt.Key_NumberSign: # ignored in SpykeListViews
            self.on_actionRenumberClusters_triggered()
        elif key == Qt.Key_C: # ignored in SpykeListViews
            self.on_actionFocusCurrentCluster_triggered()
        elif key == Qt.Key_X: # ignored in SpykeListViews
            self.on_actionFocusCurrentSpike_triggered()
        elif key == Qt.Key_R: # ignored in SpykeListViews
            self.on_actionSelectRandomSpikes_activated()
        elif key == Qt.Key_B: # ignored in SpykeListViews
            self.on_actionAlignBest_triggered()
        elif key == Qt.Key_Comma: # ignored in SpykeListViews
            self.on_actionFindPrevMostSimilar_triggered()
        elif key == Qt.Key_Period: # ignored in SpykeListViews
            self.on_actionFindNextMostSimilar_triggered()
        else:
            SpykeToolWindow.keyPressEvent(self, event) # pass it on

    def clear(self):
        """Clear selections in this order: unsorted spikes, sorted spikes,
        secondary selected neuron, neurons"""
        spw = self.spykewindow
        clusters = spw.GetClusters()
        if len(self.uslist.selectedIndexes()) > 0:
            self.uslist.clearSelection()
        elif len(self.nslist.selectedIndexes()) > 0:
            self.nslist.clearSelection()
        elif len(clusters) == 2 and self._source in clusters:
            clusters.remove(self._source)
            spw.SelectClusters(clusters, on=False)
        else:
            self.nlist.clearSelection()

    def on_actionDeleteClusters_triggered(self):
        """Del button click"""
        spw = self.spykewindow
        clusters = spw.GetClusters()
        s = self.sort
        spikes = s.spikes
        sids = []
        for cluster in clusters:
            sids.append(cluster.neuron.sids)
        sids = np.concatenate(sids)

        # save some undo/redo stuff
        message = 'delete clusters %r' % [ c.id for c in clusters ]
        cc = ClusterChange(sids, spikes, message)
        cc.save_old(clusters, s.norder)

        # deselect and delete clusters
        spw.DelClusters(clusters)
        if len(s.clusters) > 0:
            # select cluster that replaces the first of the deleted clusters in norder
            selrows = [ cc.oldnorder.index(oldunid) for oldunid in cc.oldunids ]
            if len(selrows) > 0:
                selrow = selrows[0]
                nlist = spw.windows['Sort'].nlist
                nlist.selectRows(selrow) # TODO: this sets selection, but not focus
            #else: # first of deleted clusters was last in norder, don't select anything

        # save more undo/redo stuff
        newclusters = []
        cc.save_new(newclusters, s.norder)
        spw.AddClusterChangeToStack(cc)
        print(cc.message)

    def on_actionMergeClusters_triggered(self):
        """Merge button (M) click. For simple merging of clusters, easier to
        use than running climb() on selected clusters using a really big sigma to force
        them to all merge"""
        spw = self.spykewindow
        clusters = spw.GetClusters()
        s = self.sort
        spikes = s.spikes
        sids = [] # spikes to merge
        for cluster in clusters:
            sids.append(cluster.neuron.sids)
        # merge any selected usids as well
        sids.append(spw.GetUnsortedSpikes())
        sids = np.concatenate(sids)
        if len(sids) == 0:
            return

        # save some undo/redo stuff
        message = 'merge clusters %r' % [ c.id for c in clusters ]
        cc = ClusterChange(sids, spikes, message)
        cc.save_old(clusters, s.norder)

        # get ordered index of first selected cluster, if any
        inserti = None
        if len(clusters) > 0:
            inserti = s.norder.index(clusters[0].id)

        # delete selected clusters and deselect selected usids
        spw.DelClusters(clusters, update=False)
        self.uslist.clearSelection()

        # create new cluster
        t0 = time.time()
        newnid = None # merge into new highest nid
        if len(clusters) > 0:
            newnid = min([ nid for nid in cc.oldunids ]) # merge into lowest selected nid
        if newnid == -1: # never merge into a junk cluster
            newnid = None # incorporate junk into new real cluster
        newcluster = spw.CreateCluster(update=False, id=newnid, inserti=inserti)
        neuron = newcluster.neuron
        self.MoveSpikes2Neuron(sids, neuron, update=False)
        plotdims = spw.GetClusterPlotDimNames()
        newcluster.update_pos()

        # save more undo/redo stuff
        cc.save_new([newcluster], s.norder)
        spw.AddClusterChangeToStack(cc)

        # now do some final updates
        spw.UpdateClustersGUI()
        spw.ColourPoints(newcluster)
        #print('applying clusters to plot took %.3f sec' % (time.time()-t0))
        # select newly created cluster
        spw.SelectClusters(newcluster)
        cc.message += ' into cluster %d' % newcluster.id
        print(cc.message)

    def on_actionChanSplitClusters_triggered(self):
        """Split by channels button (/) click"""
        self.spykewindow.cluster('chansplit')

    def on_actionRenumberClusters_triggered(self):
        """Renumber clusters consecutively from 0, ordered by y position, on "#" button
        click. Sorting by y position makes user inspection of clusters more orderly,
        makes the presence of duplicate clusters more obvious, and allows for maximal
        spatial separation between clusters of the same colour, reducing colour
        conflicts"""
        spw = self.spykewindow
        s = self.sort
        spikes = s.spikes

        if s.norder == range(len(s.norder)):
            print('nothing to renumber: clusters IDs already ordered and contiguous')
            return

        # deselect current selections
        selclusters = spw.GetClusters()
        oldselcids = [ cluster.id for cluster in selclusters ]
        spw.SelectClusters(selclusters, on=False)

        # delete junk cluster, if it exists
        if -1 in s.clusters:
            s.remove_neuron(-1)
            print('deleted junk cluster -1')

        # get lists of unique old cids and new cids
        olducids = sorted(s.clusters) # make sure they're in order
        # this is a bit confusing: find indices that would sort olducids by y pos, but then
        # what you really want is to find the y pos *rank* of each olducid, so you need to
        # take argsort again:
        newucids = np.asarray([ s.clusters[cid].pos['y0'] for cid in olducids ]).argsort().argsort()
        cw = spw.windows['Cluster']
        oldclusters = s.clusters.copy()
        oldneurons = s.neurons.copy()
        dims = spw.GetClusterPlotDimNames()
        for oldcid, newcid in zip(olducids, newucids):
            newcid = int(newcid) # keep as Python int, not numpy int
            if oldcid == newcid:
                continue # no need to waste time removing and recreating this cluster
            # change all occurences of oldcid to newcid
            cluster = oldclusters[oldcid]
            cluster.id = newcid # this indirectly updates neuron.id
            # update cluster and neuron dicts
            s.clusters[newcid] = cluster
            s.neurons[newcid] = cluster.neuron
            sids = cluster.neuron.sids
            spikes['nid'][sids] = newcid
        # remove any orphaned cluster ids
        for oldcid in olducids:
            if oldcid not in newucids:
                del s.clusters[oldcid]
                del s.neurons[oldcid]
        # reset norder
        s.norder = sorted(s.neurons)

        # now do some final updates
        spw.UpdateClustersGUI()
        spw.ColourPoints(s.clusters.values())
        # reselect the previously selected (but now renumbered) clusters - helps user keep track
        newselcids = newucids[np.searchsorted(olducids, oldselcids)]
        spw.SelectClusters([s.clusters[cid] for cid in newselcids])
        # all cluster changes in stack are no longer applicable, reset cchanges
        del spw.cchanges[:]
        spw.cci = -1
        print('renumbering complete')

    def on_actionFocusCurrentCluster_triggered(self):
        """Move focus to location focus of currently selected (single) cluster"""
        spw = self.spykewindow
        try:
            cluster = spw.GetCluster()
        except RuntimeError, msg:
            print(msg)
            return
        cw = spw.windows['Cluster']
        dims = spw.GetClusterPlotDimNames()
        cw.glWidget.focus = np.float32([ cluster.normpos[dim] for dim in dims ])
        cw.glWidget.panTo() # pan to new focus
        cw.glWidget.updateGL()

    def on_actionFocusCurrentSpike_triggered(self):
        """Move focus to location of currently selected (single) spike"""
        spw = self.spykewindow
        try:
            sid = spw.GetSpike()
        except RuntimeError, msg:
            print(msg)
            return
        cw = spw.windows['Cluster']
        cw.glWidget.focus = cw.glWidget.points[sid]
        cw.glWidget.panTo() # pan to new focus
        cw.glWidget.updateGL()

    def on_actionSelectRandomSpikes_activated(self):
        """Select random sample of spikes in current cluster(s), or random sample
        of unsorted spikes if no cluster(S) selected"""
        nsamples = int(self.nsamplesComboBox.currentText())
        if len(self.nslist.neurons) > 0:
            slist = self.nslist
        else:
            slist = self.uslist
        slist.clearSelection() # emits selectionChanged signal, .reset() doesn't
        slist.selectRandom(nsamples)

    def on_actionAlignMin_triggered(self):
        self.Align('min')

    def on_actionAlignMax_triggered(self):
        self.Align('max')

    def on_actionAlignBest_triggered(self):
        self.Align('best')

    def on_actionFindPrevMostSimilar_triggered(self):
        self.findMostSimilarCluster('previous')

    def on_actionFindNextMostSimilar_triggered(self):
        self.findMostSimilarCluster('next')

    def findMostSimilarCluster(self, which='next'):
        try:
            source = self.getClusterComparisonSource()
        except RuntimeError, errmsg:
            print(errmsg)
            return
        destinations = self.sort.clusters.values()
        destinations.remove(source)
        errors = []
        dests = []
        # compare source neuron waveform to all destination neuron waveforms
        for dest in destinations:
            try:
                error = rmserror(source.neuron, dest.neuron)
            except ValueError: # not comparable
                continue
            errors.append(error)
            dests.append(dest)
        if len(errors) == 0:
            print("no sufficiently overlapping clusters to compare to")
            return
        errors = np.asarray(errors)
        dests = np.asarray(dests)
        desterrsortis = errors.argsort()

        if which == 'next':
            self._cmpid += 1
        elif which == 'previous':
            self._cmpid -= 1
        else: raise ValueError('unknown which: %r' % which)
        self._cmpid = max(self._cmpid, 0)
        self._cmpid = min(self._cmpid, len(dests)-1)

        dest = dests[desterrsortis][self._cmpid]
        self.spykewindow.SelectClusters(dest)
        desterr = errors[desterrsortis][self._cmpid]
        print('n%d to n%d rmserror: %.2f uV' %
             (source.id, dest.id, self.sort.converter.AD2uV(desterr)))

    def getClusterComparisonSource(self):
        selclusters = self.spykewindow.GetClusters()
        errmsg = 'unclear which cluster to use as source for comparison'
        if len(selclusters) == 1:
            source = selclusters[0]
            self._source = source
            self._cmpid = -1 # reset
        elif len(selclusters) == 2:
            source = self._source
            if source not in selclusters:
                raise RuntimeError(errmsg)
            # deselect old destination cluster:
            selclusters.remove(source)
            self.spykewindow.SelectClusters(selclusters, on=False)
        else:
            self._source = None # reset for tidiness
            raise RuntimeError(errmsg)
        return source

    def Align(self, to):
        selclusters = self.spykewindow.GetClusters()
        nids = [ cluster.id for cluster in selclusters ]
        for nid in nids:
            self.sort.neurons[nid].align(to)
        # auto-refresh all plots
        self.panel.updateAllItems()

    def RemoveNeuron(self, neuron, update=True):
        """Remove neuron and all its spikes from the GUI and the Sort"""
        self.MoveSpikes2List(neuron, neuron.sids, update=update)
        self.sort.remove_neuron(neuron.id)
        if update:
            self.nlist.updateAll()

    def MoveSpikes2Neuron(self, sids, neuron=None, update=True):
        """Assign spikes from sort.spikes to a neuron, and update mean wave.
        If neuron is None, create a new one"""
        sids = toiter(sids)
        spikes = self.sort.spikes
        createdNeuron = False
        if neuron == None:
            neuron = self.sort.create_neuron()
        neuron.sids = np.union1d(neuron.sids, sids) # update
        spikes['nid'][sids] = neuron.id
        if update:
            self.sort.update_usids()
            self.uslist.updateAll()
        if neuron in self.nslist.neurons:
            self.nslist.neurons = self.nslist.neurons # triggers nslist refresh
        # TODO: selection doesn't seem to be working, always jumps to top of list
        #self.uslist.Select(row) # automatically select the new item at that position
        neuron.wave.data = None # triggers an update when it's actually needed
        #neuron.update_wave() # update mean neuron waveform
        return neuron

    def MoveSpikes2List(self, neuron, sids, update=True):
        """Move spikes from a neuron back to the unsorted spike list control.
        Make sure to call neuron.update_wave() at some appropriate time after
        calling this method"""
        sids = toiter(sids)
        if len(sids) == 0:
            return # nothing to do
        spikes = self.sort.spikes
        neuron.sids = np.setdiff1d(neuron.sids, sids) # return what's in first arr and not in the 2nd
        spikes['nid'][sids] = -1 # unbind neuron id of sids in struct array
        if update:
            self.sort.update_usids()
            self.uslist.updateAll()
        # this only makes sense if the neuron is currently selected in the nlist:
        if neuron in self.nslist.neurons:
            self.nslist.neurons = self.nslist.neurons # this triggers a refresh
        neuron.wave.data = None # triggers an update when it's actually needed

