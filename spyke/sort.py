"""Spike sorting classes and window"""

from __future__ import division
from __init__ import __version__

__authors__ = ['Martin Spacek', 'Reza Lotun']

import numpy as np
import pyximport
pyximport.install()
import util # .pyx file

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
from numpy import sqrt
import scipy
#from scipy.cluster.hierarchy import fclusterdata

import pylab as pl

import core
from core import TW, WaveForm, Gaussian, MAXLONGLONG, R
from core import toiter, savez, intround, lstrip, rstrip, lrstrip, pad, td2usec, td2days
from core import SpykeToolWindow, NList, NSList, USList, ClusterChange
from core import lrrep2Darrstripis, rollwin2D
from surf import EPOCH
from plot import SpikeSortPanel, WHITE

MAXCHANTOLERANCE = 100 # um

MAINSPLITTERPOS = 300
VSPLITTERPOS = 1 # make horizontal sort slider use as little vertical space as possible
HSPLITTERPOS = intround(MAINSPLITTERPOS * 3 / 4) # maximize nlist width
SPIKESORTPANELWIDTHPERCOLUMN = 120
SORTWINDOWHEIGHT = 1080

MEANWAVESAMPLESIZE = 1000


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

    def get_nextnid(self):
        """nextnid is used to retrieve the next unique single unit ID"""
        nids = self.neurons.keys()
        if len(nids) == 0:
            return 1 # single unit nids start at 1
        else:
            return max(max(nids) + 1, 1) # at least 1

    nextnid = property(get_nextnid)

    def get_nextmuid(self):
        """nextmuid is used to retrieve the next unique multiunit ID"""
        nids = self.neurons.keys()
        if len(nids) == 0:
            return -1 # multiunit ids start at -1
        else:
            return min(min(nids) - 1, -1) # at most -1

    nextmuid = property(get_nextmuid)

    def get_good(self):
        """Return array of nids marked by user as 'good'"""
        good = []
        for neuron in self.neurons.values():
            try:
                if neuron.good:
                    good.append(neuron.id)
            except AttributeError: # neuron is from older sort, no .good attrib
                neuron.good = False
        return np.asarray(good)

    good = property(get_good)

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
        # Spikes and wavedata arrays are (potentially) saved separately.
        # usids and PCs/ICs can be regenerated from the spikes array.
        for attr in ['spikes', 'wavedata', 'usids', 'comp', 'compkind', 'compsids', 'compchans']:
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
        self.usids, = np.where(nids == 0) # 0 means unclustered

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

    def get_mean_wave(self, sids, nid=None):
        """Return the mean and std waveform of spike waveforms in sids"""
        spikes = self.spikes
        nsids = len(sids)
        if nsids > MEANWAVESAMPLESIZE:
            s = ("get_mean_wave() taking random sample of %d spikes instead of all %d of them"
                 % (MEANWAVESAMPLESIZE, nsids))
            if nid != None:
                s = "neuron %d: " % nid + s
            print(s)
            sids = np.asarray(random.sample(sids, MEANWAVESAMPLESIZE))
            nsids = len(sids) # update
    
        chanss = spikes['chans'][sids]
        nchanss = spikes['nchans'][sids]
        chanslist = [ chans[:nchans] for chans, nchans in zip(chanss, nchanss) ] # list of arrays
        chanpopulation = np.concatenate(chanslist)
        groupchans = np.unique(chanpopulation) # comes out sorted
    
        wavedata = self.wavedata[sids]
        if wavedata.ndim == 2: # should be 3, get only 2 if nsids == 1
            wavedata.shape = 1, wavedata.shape[0], wavedata.shape[1] # give it a singleton 3rd dim
        nt = wavedata.shape[-1]
        maxnchans = len(groupchans)
        data = np.zeros((maxnchans, nt))
        # all spike have same nt, but not necessarily nchans, keep track of
        # how many spikes contributed to each of the group's chans
        nspikes = np.zeros((maxnchans, 1), dtype=int)
        for chans, wd in zip(chanslist, wavedata):
            chanis = groupchans.searchsorted(chans) # each spike's chans is a subset of groupchans
            data[chanis] += wd[:len(chans)] # accumulate
            nspikes[chanis] += 1 # inc spike count for this spike's chans
        #t0 = time.time()
        data /= nspikes # normalize all data points appropriately, this is now the mean
        var = np.zeros((maxnchans, nt))
        for chans, wd in zip(chanslist, wavedata):
            chanis = groupchans.searchsorted(chans) # each spike's chans is a subset of groupchans
            var[chanis] += (wd[:len(chans)] - data[chanis]) ** 2 # accumulate 2nd moment
        var /= nspikes # normalize all data points appropriately, this is now the variance
        std = np.sqrt(var)
        # keep only those chans that at least 1/2 the spikes contributed to
        bins = list(groupchans) + [sys.maxint] # concatenate rightmost bin edge
        hist, bins = np.histogram(chanpopulation, bins=bins)
        chans = groupchans[hist >= nsids/2]
        chanis = groupchans.searchsorted(chans)
        data = data[chanis]
        std = std[chanis]
        return WaveForm(data=data, std=std, chans=chans)

    def exportptcsfiles(self, sortpath=None, basepath=None):
        """Export spike data to binary .ptcs files under basepath, one file per recording"""
        spikes = self.spikes
        exportdt = str(datetime.datetime.now()) # get an export datetime stamp
        exportdt = exportdt.split('.')[0] # ditch the us
        try: # self.stream is a TrackStream?
            streams = self.stream.streams
        except AttributeError: # self.stream is a normal Stream
            streams = [self.stream]
        print('exporting clustered spikes to:')
        # do a separate export for each recording
        for stream in streams:
            # get time delta between stream i and stream 0, could be 0:
            td = stream.datetime - streams[0].datetime
            self.exportptcsfile(stream, td, exportdt, sortpath, basepath)

    def exportptcsfile(self, stream, td, exportdt, sortpath, basepath):
        """Export spike data to binary .ptcs file in basepath. Constrain to spikes in
        stream, and undo any time delta in spike times"""
        userdescr = ''
        nsamplebytes = 4 # float32
        # build up list of PTCSNeuronRecords that have spikes in this stream,
        # and tally their spikes
        nrecs = []
        nspikes = 0
        # only export neurons marked as "good", could be single or multi unit:
        for nid in sorted(self.good):
            neuron = self.neurons[nid]
            spikets = self.spikes['t'][neuron.sids] # should be a sorted copy
            assert spikets.flags['OWNDATA'] # should now be safe to modify in place
            spikets -= td2usec(td) # export spike times relative to t=0 of this recording
            # only include spikes that occurred during this recording
            lo, hi = spikets.searchsorted([stream.t0, stream.t1])
            spikets = spikets[lo:hi]
            if len(spikets) == 0:
                continue # don't save empty neurons
            nrec = PTCSNeuronRecord(neuron, spikets, nsamplebytes, descr='')
            nrecs.append(nrec)
            nspikes += len(spikets)
        nneurons = len(nrecs)

        # write the file
        path = os.path.join(basepath, stream.srcfnameroot)
        try: os.mkdir(path)
        except OSError: pass # path already exists?
        fname = exportdt.replace(' ', '_')
        fname = fname.replace(':', '.')
        fname = fname + '.ptcs'
        fullfname = os.path.join(path, fname)
        with open(fullfname, 'wb') as f:
            header = PTCSHeader(self, sortpath, stream, nneurons, nspikes, userdescr,
                                nsamplebytes, fullfname, exportdt)
            header.write(f)
            for nrec in nrecs:
                nrec.write(f)
        print(fullfname)

    def exportgdffiles(self, basepath=None):
        """Export spike and stim data to text .gdf files under basepath, one file per
        recording"""
        spikes = self.spikes
        exportdt = str(datetime.datetime.now()) # get an export datetime stamp
        exportdt = exportdt.split('.')[0] # ditch the us
        try: # self.stream is a TrackStream?
            streams = self.stream.streams
        except AttributeError: # self.stream is a normal Stream
            streams = [self.stream]
        # only export spikes from neurons marked as "good", could be single or multi unit:
        good = np.zeros(len(spikes), dtype=bool) # init False array
        for nid in self.good:
            good[self.neurons[nid].sids] = True
        sids, = np.where(good == True) # sids to export across all streams
        print('exporting clustered spikes to:')
        # do a separate export for each recording
        for stream in streams:
            # get time delta between stream i and stream 0, could be 0:
            td = stream.datetime - streams[0].datetime
            self.exportgdffile(sids, stream, td, exportdt, basepath)

    def exportgdffile(self, sids, stream, td, exportdt, basepath):
        """Export spikes in sids to text .gdf file in basepath. Constrain to spikes in
        stream, and undo any time delta in spike times. Also export stim data. 1st column is
        event id, 2nd column is event time in ms res"""
        nids = self.spikes['nid'][sids] # not modifying in place, no need for a copy
        assert nids.max() < 1000 # don't confuse any nids with stim event ids
        spikets = self.spikes['t'][sids] # should be a sorted copy
        assert spikets.flags['OWNDATA'] # should now be safe to modify in place
        spikets -= td2usec(td) # export spike times relative to t=0 of this recording
        # only include spikes that occurred during this recording
        lo, hi = spikets.searchsorted([stream.t0, stream.t1])
        spikets = spikets[lo:hi]
        nids = nids[lo:hi]
        nspikes = len(spikets)
        svalrecs = stream.srff.digitalsvalrecords
        stimts = svalrecs['TimeStamp'] # stimulus raster times
        svals = np.int64(svalrecs['SVal']) # stim vals at those times, convert from uint16
        changeis = np.where(np.diff(svals) != 0)[0] + 1 # indices at which svals change
        # keep first sval, plus just the ones that change:
        changeis = np.concatenate([[0], changeis])
        stimts = stimts[changeis]
        svals = svals[changeis]
        nsvals = len(svals)
        nevents = nsvals + nspikes
        idts = np.empty((nevents, 2), dtype=np.int64) # init array to export as text
        # stick stim events at start for now:
        idts[:nsvals, 0] = svals + 1000 # .gdf stim event ids start from 1000
        idts[:nsvals, 1] = intround(stimts / 1e3) # convert to int ms resolution
        # followed by the spikes:
        idts[nsvals:, 0] = nids
        idts[nsvals:, 1] = intround(spikets / 1e3) # convert to int ms resolution
        # now sort them in time:
        sortis = idts[:, 1].argsort()
        idts = idts[sortis]

        # write the file
        path = os.path.join(basepath, stream.srcfnameroot)
        try: os.mkdir(path)
        except OSError: pass # path already exists?
        fname = exportdt.replace(' ', '_')
        fname = fname.replace(':', '.')
        fname = fname + '.gdf'
        fullfname = os.path.join(path, fname)
        np.savetxt(fullfname, idts, '%d') # default delimiter is ' '
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
            streamtranges = np.int64([[self.stream.t0, self.stream.t1]])
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
        spikes = self.spikes[self.spikes['nid'] > 0] # don't export unsorted/multiunit spikes
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
            # NOTE: offset correction is a bad idea. Leave disabled. Spike times and DIN should
            # be exported without offsets in their timestamps. .ptcs files have a datetime field,
            # which can be used later to calculate offsets between recordings
            '''
            # calculate offset for din values, get time delta between stream i and stream 0
            td = td2usec(stream.datetime - streams[0].datetime)
            digitalsvalrecords[:, 0] += td # add offset
            '''
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

    def get_param_matrix(self, kind=None, sids=None, dims=None, selchans=None, scale=True):
        """Organize dims parameters from sids into a data matrix, each column
        corresponding to a dim. To do PCA/ICA clustering on all spikes, one maxchan at
        a time, caller needs to call this multiple times, one for each set of
        maxchan unique spikes,"""
        spikes = self.spikes
        if sids == None:
            sids = spikes['id'] # default to all spikes
        comps = np.any([ dim.startswith('c') and dim[-1].isdigit() for dim in dims ])
        if comps:
            X = self.get_component_matrix(kind, sids, chans=selchans)
        data = []
        for dim in dims:
            if dim in spikes.dtype.fields:
                data.append( np.float32(spikes[dim][sids]) )
            elif dim.startswith('c') and dim[-1].isdigit():
                compid = int(lstrip(dim, 'c'))
                data.append( np.float32(X[:, compid]) )
            else:
                raise RuntimeError('unknown dim %r' % dim)
        # np.column_stack returns a copy, not modifying the original array
        data = np.column_stack(data)
        if scale:
            # ensure 0 mean, and unit variance/stdev
            x0std = spikes['x0'].std()
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

    def get_component_matrix(self, kind, sids, chans=None):
        """Find set of chans common to all sids, and do PCA/ICA on those waveforms. Or,
        if chans are specified, limit PCA/ICA to them"""
        import mdp # delay as late as possible
        spikes = self.spikes
        chanss = spikes['chans'][sids]
        nchanss = spikes['nchans'][sids]
        chanslist = [ cs[:ncs] for cs, ncs in zip(chanss, nchanss) ] # list of arrays
        allchans = core.intersect1d(chanslist) # find intersection
        if not chans: # empty list, or None
            chans = allchans
        diffchans = np.setdiff1d(chans, allchans) # values in chans but not in allchans
        chans = np.intersect1d(chans, allchans) # values in both
        if len(diffchans) > 0:
            print('WARNING: ignored chans %r not common to all spikes' % list(diffchans))
        nchans = len(chans)
        nspikes = len(sids)
        if nchans == 0:
            raise RuntimeError("Spikes have no common chans for %s" % kind)
        if (hasattr(self, 'comp') and
            hasattr(self, 'compkind') and kind == self.compkind and
            hasattr(self, 'compsids') and np.all(sids == self.compsids) and
            hasattr(self, 'compchans') and np.all(chans == self.compchans)):
            print('using saved %ss from chans %r of %d spikes' % (kind[:-1], list(chans), nspikes))
            return self.comp # no need to recalculate

        # collect data from chans from all spikes:
        if kind not in ['PCA', 'ICA', 'PCA+ICA']:
            raise ValueError('unknown kind %r' % kind)
        nt = self.wavedata.shape[2]
        print('doing %s on chans %r of %d spikes' % (kind, list(chans), nspikes))
        # MDP complains of roundoff errors with float32 for large covariance matrices
        data = np.zeros((nspikes, nchans, nt), dtype=np.float64)
        for sii, sid in enumerate(sids):
            spikechans = chanslist[sii]
            spikechanis = np.searchsorted(spikechans, chans)
            data[sii] = self.wavedata[sid][spikechanis]
        t0 = time.time()
        if kind == 'PCA':
            data.shape = nspikes, nchans*nt # flatten timepoints of all chans into columns
            #comp = mdp.pca(data, output_dim=5, svd=False)
            comp = mdp.pca(data, output_dim=5) # keep just 1st 5 components
        else: # kind in ['ICA', 'PCA+ICA']:
            # ensure nspikes >= ndims**2 for good ICA convergence
            maxncomp = intround(sqrt(nspikes))
            if kind == 'ICA':
                # for speed, keep only the largest 14% of points, per chan. Largest points are
                # probably the most important ones
                mean = data.mean(axis=0) # mean across all spikes
                datai = abs(mean).argsort(axis=1)[:, ::-1] # highest to lowest amplitude points, per chan
                ntkeep = nt // 7
                datai = datai[:, :ntkeep]
                datai += np.row_stack(np.arange(nchans)) * nt
                datai = datai.ravel() # 1D of len nchans*ntkeep
                data.shape = nspikes, nchans*nt # flatten timepoints of all chans into columns
                data = data[:, datai] # nspikes x (nchans*ntkeep)
                print datai
                if data.shape[1] > maxncomp:
                    mean = data.mean(axis=0) # mean across all spikes, gives nchans*ntkeep vector
                    pointis = abs(mean).argsort()[::-1] # highest to lowest amplitude points
                    pointis = pointis[:maxncomp]
                    data = data[:, pointis]
                    print('restrict to maxncomp=%d: %r' % (maxncomp, datai[:, pointis]))
            else: # kind == 'PCA+ICA'
                # do PCA first, to reduce dimensionality and speed up ICA:
                data.shape = nspikes, nchans*nt # flatten timepoints of all chans into columns
                ncomp = min(maxncomp, 7*nchans) # keep up to 7 components per chan on average
                print('ncomp = %d' % ncomp)
                data = mdp.pca(data, output_dim=ncomp)
            print('data.shape = %r' % (data.shape,))
            if data.shape[0] <= data.shape[1]:
                raise RuntimeError('need more observations than dimensions for ICA')
            trycount = 0
            while True:
                try:
                    node = mdp.nodes.FastICANode()
                    comp = node(data)
                    pm = node.get_projmatrix()
                    comp = comp[:, np.any(pm, axis=0)] # keep only the non zero columns
                    if comp.shape[1] < 3: # need at least 3 columns
                        raise RuntimeError()
                    break
                except:
                    print('ICA failed, retrying...')
                    trycount += 1
                    if trycount > 10:
                        break # give up
            # sort ICs by decreasing kurtosis, as in Scholz et al, 2004 (or rather,
            # opposite to their approach, which picked ICs with most negative kurtosis)
            ## TODO: maybe an alternative to this is to ues a HitParade node, which apparently
            ## returns the "largest local maxima" of the previous node
            ## Another possibility might be to sort according to the energy in each column
            ## of node.filter (see sorting of components at end of JADENode). See McKeown 2003.
            
            ## TODO: damn, what's the different between a node's filters and a node's
            ## projection matrix????????????? They're the same shape.. Are they perhaps the
            ## inverse or pseudo inverse of each other?
            
            ## TODO: sounds like nonlineariy g and fine_g = 'gaus' or maybe 'tanh' might be
            ## better choice than default 'pow3', though they might be slower. See Hyvarinen 1999
            
            ## TODO: perhaps when using PCA before ICA, since the PCA comes out ordered by
            ## captured variance, maybe the ICs will come out that way too, and I don't
            ## need to measure kurtosis anymore? Nope, not the case it seems.
            
            k = scipy.stats.kurtosis(comp, axis=0) # find kurtosis of each IC (column)
            ki = k.argsort()[::-1] # decreasing order of kurtosis
            #std = comp.std(axis=0)
            #stdi = std.argsort()[::-1] # decreasing order of std
            comp = comp[:, ki] # sort 'em
            #comp = comp[:, :5] # keep just 1st 5 components
            #print(pm)
            #print('k:', k)
            #print('by k: ', ki)
            #print('std:', std)
            #print('by std: ', stdi)
            '''
            import pylab as pl
            pl.figure()
            pl.imshow(pm)
            pl.colorbar()
            pl.title('original projmatrix')
            pl.figure()
            pl.imshow(pm[:, ki])
            pl.colorbar()
            pl.title('decreasing kurtosis projmatrix')
            pl.figure()
            pl.imshow(pm[:, stdi])
            pl.colorbar()
            pl.title('decreasing std projmatrix')
            '''
        self.comp = comp
        print('%s took %.3f sec' % (kind, time.time()-t0))
        self.compkind = kind
        self.compsids = sids
        self.compchans = copy(chans) # make sure this isn't just a pointer to panel.selected_chans
        unids = list(np.unique(spikes['nid'][sids])) # set of all nids that sids span
        for nid in unids:
            # don't update pos of junk cluster, if any, since it might not have any chans
            # common to all its spikes, and therefore can't have PCA/ICA done on it
            if nid != 0:
                self.clusters[nid].update_comppos()
        return self.comp

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

    def alignbest(self, sids, chans, method=core.rmserror):
        """Align all sids on chans by best fit according to error method.
        chans are assumed to be a subset of channels of sids. Return sids
        that were actually moved and therefore need to be marked as dirty"""
        spikes = self.spikes
        streamopen = self.stream.is_open()
        nspikes = len(sids)
        nchans = len(chans)
        wd = self.wavedata
        nt = wd.shape[2] # num timepoints in each waveform
        # TODO: make maxshift a f'n of interpolation factor
        maxshift = 2 # shift +/- this many timepoints
        #maxshiftus = maxshift * self.stream.tres
        shifts = range(-maxshift, maxshift+1) # from -maxshift to maxshift, inclusive
        print("padding waveforms with up to +/- %d points of fake data" % maxshift)

        # collect spike waveform data on chans, and calc mean
        '''
        if len(sids) > MEANWAVESAMPLESIZE:
            print('taking random sample of %d spikes instead of all %d of them' %
                  (MEANWAVESAMPLESIZE, nspikes)
            sidis = np.random.randint(0, nspikes, MEANWAVESAMPLESIZE)
        else:
            sidis = np.arange(nspikes)
        '''
        subsd = np.zeros((nspikes, nchans, nt), dtype=wd.dtype)
        spikechanis = np.zeros((nspikes, nchans), dtype=np.int64)
        for sidi, sid in enumerate(sids):
            spike = spikes[sid]
            nspikechans = spike['nchans']
            spikechans = spike['chans'][:nspikechans]
            spikechanis[sidi] = spikechans.searchsorted(chans)
            subsd[sidi] = wd[sid, spikechanis[sidi]]
        meandata = subsd.mean(axis=0) # float64

        # choose best shifted waveform for each spike
        # widesd holds spike data plus extra data on either side
        # to allow for full width slicing for all time shifts:
        maxnchans = spikes['nchans'].max() # of all spikes in sort
        widesd = np.zeros((maxnchans, maxshift+nt+maxshift), dtype=wd.dtype)        
        shiftedsubsd = subsd.copy()
        tempshifts = np.zeros((len(shifts), maxnchans, nt), dtype=wd.dtype)
        tempsubshifts = np.zeros((len(shifts), nchans, nt), dtype=wd.dtype)
        errors = np.zeros(len(shifts))
        dirtysids = []
        for sidi, sid in enumerate(sids):
            # for speed, always add fake values at start and end. Only load
            # real data when explicitly ask for it via reloadSpikes()
            '''
            if streamopen:
                wave = self.stream[-maxshiftus+spike['t0'] : spike['t1']+maxshiftus]
                chanis = wave.chans.searchsorted(chans)
                widesd = wave.data[chanis]
            else: # add fake values at start and end if .srf file isn't available.
            '''
            sd = wd[sid] # spike data
            chanis = spikechanis[sidi]
            widesd[:, maxshift:-maxshift] = sd # 2D
            widesd[:, :maxshift] = sd[:, 0, None] # pad start with first point per chan
            widesd[:, -maxshift:] = sd[:, -1, None] # pad end with last point per chan
            errors.fill(0.0) # reset
            for shifti, shift in enumerate(shifts):
                t0i = maxshift + shift
                tempshifts[shifti] = widesd[:, t0i:t0i+nt]
                tempsubshifts[shifti] = tempshifts[shifti, chanis]
                errors[shifti] = method(tempsubshifts[shifti], meandata)
            bestshifti = errors.argmin()
            bestshift = shifts[bestshifti]
            if bestshift != 0: # no need to update sort.wavedata[sid] if there's no shift
                # update time values:
                dt = bestshift * self.tres # time to shift by, signed, in us
                spikes['t'][sid] += dt # should remain halfway between t0 and t1
                spikes['t0'][sid] += dt
                spikes['t1'][sid] += dt
                # might result in some out of bounds phasetis because the original phases
                # have shifted off the ends. Opposite sign, referencing within wavedata:
                spikes['phasetis'][sid] -= bestshift
                # update sort.wavedata
                wd[sid] = tempshifts[bestshifti]
                shiftedsubsd[sidi] = tempsubshifts[bestshifti]
                dirtysids.append(sid) # mark sid as dirty
        AD2uV = self.converter.AD2uV
        stdevbefore = AD2uV(subsd.std(axis=0).mean())
        stdevafter = AD2uV(shiftedsubsd.std(axis=0).mean())
        print('stdev went from %.3f to %.3f uV' % (stdevbefore, stdevafter))
        return dirtysids

    def alignminmax(self, sids, to):
        """Align sids by their min or max. Return those that were actually moved
        and therefore need to be marked as dirty"""
        if not self.stream.is_open():
            raise RuntimeError("no open stream to reload spikes from")
        spikes = self.spikes
        V0s = spikes['V0'][sids]
        V1s = spikes['V1'][sids]
        Vss = np.column_stack((V0s, V1s))
        alignis = spikes['aligni'][sids]
        b = np.column_stack((alignis==0, alignis==1)) # 2D boolean array
        if to == 'min':
            i = Vss[b] > 0 # indices into sids of spikes aligned to the max phase
        elif to == 'max':
            i = Vss[b] < 0 # indices into sids of spikes aligned to the min phase
        else:
            raise ValueError('unknown to %r' % to)
        sids = sids[i] # sids that need realigning
        nspikes = len(sids)
        print("realigning %d spikes" % nspikes)
        if nspikes == 0: # nothing to do
            return [] # no sids to mark as dirty

        multichanphasetis = spikes['phasetis'][sids] # nspikes x nchans x 2 arr
        chanis = spikes['chani'][sids] # nspikes arr of max chanis
        # phasetis of max chan of each spike, convert from uint8 to int32 for safe math
        phasetis = np.int32(multichanphasetis[np.arange(nspikes), chanis]) # nspikes x 2 arr
        # NOTE: phasetis aren't always in temporal order!
        dphasetis = phasetis[:, 1] - phasetis[:, 0] # could be +ve or -ve
        dphases = spikes['dphase'][sids] # stored as +ve

        # for each spike, decide whether to add or subtract dphase to/from its temporal values
        ordered  = dphasetis > 0 # in temporal order
        reversed = dphasetis < 0 # in reversed temporal order
        alignis = spikes['aligni'][sids]
        alignis0 = alignis == 0
        alignis1 = alignis == 1
        dphasei = np.zeros(nspikes, dtype=int)
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
        self.reloadSpikes(sids)
        return sids # mark all sids as dirty

    def reloadSpikes(self, sids, fixtvals=False, usemeanchans=False):
        """Update wavedata of designated spikes from stream. Optionally fix incorrect
        time values from .sort 0.3 files. Optionally choose new set of channels for all
        sids based on the chans closest to the mean of the sids. It's the caller's
        responsibility to mark sids as dirty and trigger resaving of .wave file"""
        stream = self.stream
        if not stream.is_open():
            raise RuntimeError("no open stream to reload spikes from")
        spikes = self.spikes
        ver_lte_03 = float(self.__version__) <= 0.3
        print('reloading %d spikes' % len(sids))
        if usemeanchans:
            if ver_lte_03:
                raise RuntimeError("Best not to choose new chans from mean until after "
                                   "converting to .sort >= 0.4")
            print('choosing new channel set for all selected spikes')
            # get mean waveform of all sids, then find the mean's chan with max Vpp, then
            # choose det.maxnchansperspike channels around that maxchan
            meanwave = self.get_mean_wave(sids)
            # mean chan with max Vpp:
            maxchan = meanwave.chans[meanwave.data.ptp(axis=1).argmax()]
            det = self.detector
            maxchani = det.chans.searchsorted(maxchan)
            distances = det.dm.data[maxchani]
            # keep the maxnchansperspike closest chans to maxchan, including maxchan:
            chanis = distances.argsort()[:det.maxnchansperspike]
            meanchans = det.chans[chanis]
            meanchans.sort() # keep them sorted
            print('meanchans: %r' % meanchans)
            furthestchan = det.chans[chanis[-1]]
            print('furthestchan: %d' % furthestchan)
            furthestchani = meanchans.searchsorted(furthestchan)
            nmeanchans = len(meanchans)
            # just to be sure:
            assert nmeanchans == det.maxnchansperspike
            assert maxchan in meanchans

        if fixtvals and ver_lte_03:
            """In sort.__version__ <= 0.3, t, t0, t1, and phasetis were not updated
            during alignbest() calls. To fix this, load new data with old potentially
            incorrect t0 and t1 values, and compare this new data to existing old data
            in wavedata array. Find where the non-repeating parts of the old data fits
            into the new, and calculate the correction needed to fix the time values,
            and also reload new data according to these corrected time values."""
            if usemeanchans == True:
                # this could be complicated, avoid it
                raise RuntimeError("Best not to fix time values and simultaneously choose "
                                   "new chans from mean. Do one, then the other")
            print('fixing potentially wrong time values during spike reloading')
            nfixed = 0
            for sid in sids:
                #print('reloading sid: %d' % sid)
                spike = spikes[sid]
                nchans = spike['nchans']
                chans = spike['chans'][:nchans]
                od = self.wavedata[sid, 0:nchans] # old data
                # indices that strip const values from left and right ends:
                lefti, righti = lrrep2Darrstripis(od)
                od = od[:, lefti:righti] # stripped old data
                # load new data, use old incorrect t0 and t1, but they should be wide
                # enough to encompass the old data:
                newwave = stream[spike['t0']:spike['t1']]
                newwave = newwave[chans]
                nd = newwave.data # new data
                width = od.shape[1] # rolling window width
                assert width <= nd.shape[1]
                odinndis = np.where((rollwin2D(nd, width) == od).all(axis=1).all(axis=1))[0]
                assert len(odinndis) == 1 # ensure exactly one hit of old data in new
                odinndi = odinndis[0] # pull it out
                dnt = odinndi - lefti # num timepoints to correct by, signed
                #print('dnt: %d' % dnt)
                if dnt != 0:
                    dt = dnt * self.tres # time to correct by, signed, in us
                    spikes['t'][sid] += dt # should remain halfway between t0 and t1
                    spikes['t0'][sid] += dt
                    spikes['t1'][sid] += dt
                    # might result in some out of bounds phasetis because the original phases
                    # have shifted off the ends. Opposite sign, referencing within wavedata:
                    spikes['phasetis'][sid] -= dnt
                    spike = spikes[sid] # update local var
                    # reload spike data again now that t0 and t1 have changed
                    newwave = stream[spike['t0']:spike['t1']]
                    newwave = newwave[chans]
                    nfixed += 1
                self.wavedata[sid, 0:nchans] = newwave.data # update wavedata
            print('fixed time values of %d spikes' % nfixed)
        else: # assume time values for all spikes are accurate
            if usemeanchans:
                # update spikes array entries for all sids:
                spikes['nchans'][sids] = nmeanchans
                spikes['chans'][sids] = meanchans # using max num chans, assign full array
            for sid in sids:
                spike = spikes[sid]
                wave = stream[spike['t0']:spike['t1']]
                if usemeanchans:
                    # check that each spike's maxchan is in meanchans:
                    chan = spike['chan']
                    if chan not in meanchans:
                        # replace furthest chan with spike's maxchan:
                        print("spike %d: replacing furthestchan %d with spike's maxchan %d"
                              % (sid, furthestchan, chan))
                        spikes['chans'][sid][furthestchani] = chan
                nchans = spike['nchans']
                chans = spike['chans'][:nchans]
                wave = wave[chans]
                self.wavedata[sid, 0:nchans] = wave.data
        print('reloaded %d spikes' % len(sids))
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
        self.good = False # user can mark this neuron as "good" if so desired
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
        # don't save any calculated PCs/ICs:
        #d.pop('comp', None)
        #d.pop('compkind', None)
        #d.pop('compsids', None)
        #d.pop('compchans', None)
        # don't save plot self is assigned to, since that'll change anyway on unpickle
        d['plt'] = None
        return d

    def get_wave(self):
        """Check for valid mean and std waveform before returning it"""
        # many neuron waveforms saved in old .sort files won't have a wave.std field
        try: self.wave.std
        except AttributeError: return self.update_wave()
        if self.wave == None or self.wave.data == None or self.wave.std == None:
            return self.update_wave()
        else:
            return self.wave # return existing waveform

    def update_wave(self):
        """Update mean and std of self's waveform"""
        sort = self.sort
        spikes = sort.spikes
        if len(self.sids) == 0: # no member spikes, perhaps I should be deleted?
            raise RuntimeError("neuron %d has no spikes and its waveform can't be updated"
                               % self.id)
        meanwave = sort.get_mean_wave(self.sids, nid=self.id)

        # update self's Waveform object
        self.wave.data = meanwave.data
        self.wave.std = meanwave.std
        self.wave.chans = meanwave.chans
        self.wave.ts = sort.twts
        return self.wave

    def __sub__(self, other):
        """Return difference array between self and other neurons' waveforms
        on common channels"""
        selfwavedata, otherwavedata = self.getCommonWaveData(other.chan, other.chans,
                                                             other.wave.data)
        return selfwavedata - otherwavedata

    def getCommonWaveData(self, otherchan, otherchans, otherwavedata):
        """Return waveform data common to self's chans and otherchans, while
        requiring that both include the other's maxchan"""
        chans = np.intersect1d(self.chans, otherchans, assume_unique=True)
        if len(chans) == 0:
            raise ValueError('no common chans')
        if self.chan not in chans or otherchan not in chans:
            raise ValueError("maxchans aren't part of common chans")
        selfchanis = self.chans.searchsorted(chans)
        otherchanis = otherchans.searchsorted(chans)
        return self.wave.data[selfchanis], otherwavedata[otherchanis]
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

    formatversion: int64 (currently version 2, identical to version 1)
    ndescrbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    descr: ndescrbytes of ASCII text
        (padded with null bytes if needed for 8 byte alignment)

    nneurons: uint64 (number of neurons)
    nspikes: uint64 (total number of spikes)
    nsamplebytes: uint64 (number of bytes per template waveform sample)
    samplerate: uint64 (Hz)

    npttypebytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    pttype: npttypebytes of ASCII text
        (padded with null bytes if needed for 8 byte alignment)
    nptchans: uint64 (total num chans in polytrode)
    chanpos: nptchans * 2 * float64
        (array of (x, y) positions, in um, relative to top of polytrode,
         indexed by 0-based channel IDs)
    nsrcfnamebytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    srcfname: nsrcfnamebytes of ASCII text
        (source file name, probably .srf, padded with null bytes if needed for 8 byte alignment)
    datetime: float64
        (absolute datetime corresponding to t=0 us timestamp, stored as days since
         epoch: December 30, 1899 at 00:00)
    ndatetimestrbytes: uint64 
    datetimestr: ndatetimestrbytes of ASCII text
        (human readable string representation of datetime, preferrably ISO 8601,
         padded with null bytes if needed for 8 byte alignment)
    """
    FORMATVERSION = 2 # overall .ptcs file format version, not header format version
    def __init__(self, sort, sortpath, stream, nneurons, nspikes, userdescr,
                 nsamplebytes, fullfname, exportdt):
        self.sort = sort
        self.stream = stream
        self.nneurons = nneurons
        self.nspikes = nspikes
        self.userdescr = userdescr
        self.nsamplebytes = nsamplebytes
        homelessfullfname = lstrip(fullfname, os.path.expanduser('~'))
        sortfname = sort.sortfname
        sortfullfname = os.path.join(sortpath, sortfname)
        sortfmoddt = str(datetime.datetime.fromtimestamp(os.path.getmtime(sortfullfname)))
        sortfmoddt = sortfmoddt.split('.')[0] # ditch the us
        sortfsize = os.path.getsize(sortfullfname) # in bytes
        # For description dictionary, could create a dict and convert it
        # to a string, but that wouldn't guarantee key order. Instead,
        # build string rep of description dict with guaranteed key order:
        d = ("{'file_type': '.ptcs (polytrode clustered spikes) file', "
             "'original_fname': %r, 'export_time': %r, "
             "'sort': {'fname': %r, 'path': %r, 'fmtime': %r, 'fsize': %r}"
             % (homelessfullfname, exportdt,
                sortfname, sortpath, sortfmoddt, sortfsize))
        if userdescr:
            d += ", 'user_descr': %r" % userdescr
        d += "}"
        try: eval(d)
        except: raise ValueError("descr isn't a valid dictionary:\n%r" % d)
        self.descr = pad(d, align=8)
        self.srcfname = pad(lstrip(stream.fname, '../'), align=8)
        self.pttype = pad(stream.probe.name, align=8)
        self.dt = stream.datetime
        self.dtstr = pad(self.dt.isoformat(), align=8)

    def write(self, f):
        s = self.sort
        np.int64(self.FORMATVERSION).tofile(f) # formatversion
        np.uint64(len(self.descr)).tofile(f) # ndescrbytes
        f.write(self.descr) # descr
        
        np.uint64(self.nneurons).tofile(f) # nneurons
        np.uint64(self.nspikes).tofile(f) # nspikes
        np.uint64(self.nsamplebytes).tofile(f) # nsamplebytes
        np.uint64(s.sampfreq).tofile(f) # samplerate

        np.uint64(len(self.pttype)).tofile(f) # npttypebytes
        f.write(self.pttype) # pttype
        np.uint64(s.stream.probe.nchans).tofile(f) # nptchans
        np.float64(s.stream.probe.siteloc_arr()).tofile(f) # chanpos
        np.uint64(len(self.srcfname)).tofile(f) # nsrcfnamebytes
        f.write(self.srcfname) # srcfname
        np.float64(td2days(self.dt - EPOCH)).tofile(f) # datetime (in days)
        np.uint64(len(self.dtstr)).tofile(f) # ndatetimestrbytes
        f.write(self.dtstr)


class PTCSNeuronRecord(object):
    """
    Polytrode clustered spikes file neuron record:
    
    nid: int64 (signed neuron id, could be -ve, could be non-contiguous with previous)
    ndescrbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment, defaults to 0)
    descr: ndescrbytes of ASCII text
        (padded with null bytes if needed for 8 byte alignment)
    clusterscore: float64
    xpos: float64 (um)
    ypos: float64 (um)
    zpos: float64 (um) (defaults to NaN)
    nchans: uint64 (num chans in template waveforms)
    chanids: nchans * uint64 (0 based IDs of channels in template waveforms)
    maxchanid: uint64 (0 based ID of max channel in template waveforms)
    nt: uint64 (num timepoints per template waveform channel)
    nwavedatabytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    wavedata: nwavedatabytes of nsamplebytes sized floats
        (template waveform data, laid out as nchans * nt, in uV,
         padded with null bytes if needed for 8 byte alignment)
    nwavestdbytes: uint64 (nbytes, keep as multiple of 8 for nice alignment)
    wavestd: nwavestdbytes of nsamplebytes sized floats
        (template waveform standard deviation, laid out as nchans * nt, in uV,
         padded with null bytes if needed for 8 byte alignment)
    nspikes: uint64 (number of spikes in this neuron)
    spike timestamps: nspikes * uint64 (us, should be sorted)
    """
    def __init__(self, neuron, spikets=None, nsamplebytes=None, descr=''):
        n = neuron
        AD2uV = n.sort.converter.AD2uV
        self.neuron = neuron
        self.spikets = spikets # constrained to stream range, may be < neuron.sids
        self.wavedtype = {2: np.float16, 4: np.float32, 8: np.float64}[nsamplebytes]
        if n.wave.data == None or n.wave.std == None: # some may have never been displayed
            n.update_wave()
        # wavedata and wavestd are nchans * nt * nsamplebytes long:
        self.wavedata = pad(self.wavedtype(AD2uV(n.wave.data)), align=8)
        self.wavestd = pad(self.wavedtype(AD2uV(n.wave.std)), align=8)
        self.descr = pad(descr, align=8)
        
    def write(self, f):
        n = self.neuron
        np.int64(n.id).tofile(f) # nid
        np.uint64(len(self.descr)).tofile(f) # ndescrbytes
        f.write(self.descr) # descr
        np.float64(np.nan).tofile(f) # clusterscore
        np.float64(n.cluster.pos['x0']).tofile(f) # xpos (um)
        np.float64(n.cluster.pos['y0']).tofile(f) # ypos (um)
        np.float64(np.nan).tofile(f) # zpos (um)
        np.uint64(len(n.wave.chans)).tofile(f) # nchans
        np.uint64(n.wave.chans).tofile(f) # chanids
        np.uint64(n.chan).tofile(f) # maxchanid
        np.uint64(len(n.wave.ts)).tofile(f) # nt
        np.uint64(self.wavedata.nbytes).tofile(f) # nwavedatabytes
        self.wavedata.tofile(f) # wavedata 
        np.uint64(self.wavestd.nbytes).tofile(f) # nwavestdbytes
        self.wavestd.tofile(f) # wavestd 
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
        self.slider = QtGui.QSlider(Qt.Horizontal, self)
        self.slider.setInvertedControls(True)
        self.slider.setToolTip('Position of sliding spike selection time window')
        self.connect(self.slider, QtCore.SIGNAL("valueChanged(int)"),
                     self.on_slider_valueChanged)
        self.connect(self.slider, QtCore.SIGNAL("sliderPressed()"),
                     self.on_slider_sliderPressed)

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
        self.vsplitter.addWidget(self.slider)
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

        actionDelete = QtGui.QAction("Del", self)
        actionDelete.setToolTip('Delete clusters\nSHIFT: Delete spikes')
        self.connect(actionDelete, QtCore.SIGNAL("triggered()"),
                     self.on_actionDelete_triggered)
        toolbar.addAction(actionDelete)

        actionMergeClusters = QtGui.QAction("M", self)
        actionMergeClusters.setToolTip('Merge clusters')
        self.connect(actionMergeClusters, QtCore.SIGNAL("triggered()"),
                     self.on_actionMergeClusters_triggered)
        toolbar.addAction(actionMergeClusters)

        actionToggleClustersGood = QtGui.QAction("G", self)
        actionToggleClustersGood.setToolTip('Toggle clusters as "good"')
        self.connect(actionToggleClustersGood, QtCore.SIGNAL("triggered()"),
                     self.on_actionToggleClustersGood_triggered)
        toolbar.addAction(actionToggleClustersGood)

        actionLabelMultiunit = QtGui.QAction("-", self)
        actionLabelMultiunit.setToolTip('Label clusters as multiunit')
        self.connect(actionLabelMultiunit, QtCore.SIGNAL("triggered()"),
                     self.on_actionLabelMultiunit_triggered)
        toolbar.addAction(actionLabelMultiunit)

        actionChanSplitClusters = QtGui.QAction("/", self)
        actionChanSplitClusters.setToolTip('Split clusters by channels')
        self.connect(actionChanSplitClusters, QtCore.SIGNAL("triggered()"),
                     self.on_actionChanSplitClusters_triggered)
        toolbar.addAction(actionChanSplitClusters)

        actionRandomSplit = QtGui.QAction("\\", self)
        actionRandomSplit.setToolTip('Randomly split clusters exceeding %d' % core.MAXNCLIMBPOINTS)
        self.connect(actionRandomSplit, QtCore.SIGNAL("triggered()"),
                     self.on_actionRandomSplit_triggered)
        toolbar.addAction(actionRandomSplit)

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

        actionFocusCurrentSpike = QtGui.QAction("V", self)
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
        nsamplesComboBox.addItems(['1', '5', '10', '20', '50', '100'])
        nsamplesComboBox.setCurrentIndex(3)
        toolbar.addWidget(nsamplesComboBox)
        self.connect(nsamplesComboBox, QtCore.SIGNAL("activated(int)"),
                     self.on_actionSelectRandomSpikes_activated)
        self.nsamplesComboBox = nsamplesComboBox

        toolbar.addSeparator()

        alignlabel = QtGui.QLabel('Align:')
        toolbar.addWidget(alignlabel)

        actionAlignMin = QtGui.QAction("min", self)
        actionAlignMin.setToolTip("Align neurons' spikes to min")
        self.connect(actionAlignMin, QtCore.SIGNAL("triggered()"),
                     self.on_actionAlignMin_triggered)
        toolbar.addAction(actionAlignMin)

        actionAlignMax = QtGui.QAction("max", self)
        actionAlignMax.setToolTip("Align neurons' spikes to max")
        self.connect(actionAlignMax, QtCore.SIGNAL("triggered()"),
                     self.on_actionAlignMax_triggered)
        toolbar.addAction(actionAlignMax)

        actionAlignBest = QtGui.QAction("best", self)
        actionAlignBest.setToolTip("Align neurons' spikes by best fit")
        self.connect(actionAlignBest, QtCore.SIGNAL("triggered()"),
                     self.on_actionAlignBest_triggered)
        toolbar.addAction(actionAlignBest)

        toolbar.addSeparator()

        actionReloadSpikes = QtGui.QAction("Reload", self)
        actionReloadSpikes.setToolTip("Reload selected spikes\nSHIFT: Use mean waveform "
                                      "to choose chans to reload")
        self.connect(actionReloadSpikes, QtCore.SIGNAL("triggered()"),
                     self.on_actionReloadSpikes_triggered)
        toolbar.addAction(actionReloadSpikes)

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

        actionPlotClusterHist = QtGui.QAction("H", self)
        actionPlotClusterHist.setToolTip("Plot cluster histogram, calculate overlap index "
                                         "if 2 clusters selected")
        self.connect(actionPlotClusterHist, QtCore.SIGNAL("triggered()"),
                     self.on_actionPlotClusterHist_triggered)
        toolbar.addAction(actionPlotClusterHist)

        actionSave = QtGui.QAction("S", self)
        actionSave.setToolTip("Save sort panel to file")
        self.connect(actionSave, QtCore.SIGNAL("triggered()"),
                     self.on_actionSave_triggered)
        toolbar.addAction(actionSave)

        return toolbar

    def get_sort(self):
        return self.spykewindow.sort

    sort = property(get_sort) # make this a property for proper behaviour after unpickling

    def closeEvent(self, event):
        self.spykewindow.HideWindow('Sort')

    def mousePressEvent(self, event):
        """These are mostly passed on up from spyke list views and sort panel. Left
        clicks are (or should be) filtered out"""
        buttons = event.buttons()
        if buttons == QtCore.Qt.MiddleButton:
            self.on_actionSelectRandomSpikes_activated()
        elif buttons == QtCore.Qt.RightButton:
            self.clear()

    def keyPressEvent(self, event):
        """Alpha character keypresses are by default caught by the child lists for quickly
        scrolling down to and selecting list items. However, the appropriate alpha
        keypresses have been set in the child lists to be ignored, so they propagate
        up to here"""
        key = event.key()
        modifiers = event.modifiers()
        shift = Qt.ShiftModifier == modifiers # only modifier is shift
        ctrl = Qt.ControlModifier == modifiers # only modifier is ctrl
        if key == Qt.Key_Escape: # deselect all spikes and all clusters
            self.clear()
        elif key == Qt.Key_Delete:
            if shift:
                self.on_actionDeleteSpikes_triggered() # del selected spikes
            else:
                self.on_actionDeleteClusters_triggered() # del selected clusters
        elif key == Qt.Key_M: # ignored in SpykeListViews
            self.on_actionMergeClusters_triggered()
        elif key == Qt.Key_G: # ignored in SpykeListViews
            self.on_actionToggleClustersGood_triggered()
        elif key == Qt.Key_Minus: # ignored in SpykeListViews
            self.on_actionLabelMultiunit_triggered()
        elif key == Qt.Key_Slash: # ignored in SpykeListViews
            self.on_actionChanSplitClusters_triggered()
        elif key == Qt.Key_Backslash: # ignored in SpykeListViews
            self.on_actionRandomSplit_triggered()
        elif key == Qt.Key_NumberSign: # ignored in SpykeListViews
            self.on_actionRenumberClusters_triggered()
        elif key == Qt.Key_C: # ignored in SpykeListViews
            self.on_actionFocusCurrentCluster_triggered()
        elif key == Qt.Key_V: # ignored in SpykeListViews
            self.on_actionFocusCurrentSpike_triggered()
        elif key == Qt.Key_R: # ignored in SpykeListViews
            self.on_actionSelectRandomSpikes_activated()
        elif key == Qt.Key_Space: # ignored in SpykeListViews
            if ctrl:
                SpykeToolWindow.keyPressEvent(self, event) # pass it on
            else:
                #self.on_actionSelectRandomSpikes_activated()
                self.spykewindow.on_clusterButton_clicked()
        elif key == Qt.Key_B: # ignored in SpykeListViews
            self.on_actionAlignBest_triggered()
        elif key == Qt.Key_Comma: # ignored in SpykeListViews
            self.on_actionFindPrevMostSimilar_triggered()
        elif key == Qt.Key_Period: # ignored in SpykeListViews
            self.on_actionFindNextMostSimilar_triggered()
        elif key == Qt.Key_H: # ignored in SpykeListViews
            self.on_actionPlotClusterHist_triggered()
        elif key == Qt.Key_S: # ignored in SpykeListViews
            self.on_actionSave_triggered()
        elif key in [Qt.Key_Enter, Qt.Key_Return]:
            # this is handled at a lower level by on_actionItem_activated
            # in the various listview controls
            pass
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

    def on_actionDelete_triggered(self):
        """Del or SHIFT+Del click"""
        if QtGui.QApplication.instance().keyboardModifiers() == Qt.ShiftModifier:
            self.on_actionDeleteSpikes_triggered()
        else:
            self.on_actionDeleteClusters_triggered()

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

    def on_actionDeleteSpikes_triggered(self):
        """SHIFT+Del button click"""
        self.spykewindow.DeleteSpikes()

    def on_actionMergeClusters_triggered(self):
        """Merge button (M) click. Merge selected clusters. Easier to use than
        running climb() on selected clusters using a really big sigma to force
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
        newnid = None # merge by default into new highest nid
        if len(clusters) > 0:
            oldunids = np.asarray(cc.oldunids)
            suids = oldunids[oldunids > 0]
            if len(suids) > 0:
                newnid = suids.min() # merge into lowest selected single unit nid
        newcluster = spw.CreateCluster(update=False, id=newnid, inserti=inserti)
        neuron = newcluster.neuron
        self.MoveSpikes2Neuron(sids, neuron, update=False)
        plotdims = spw.GetClusterPlotDims()
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

    def on_actionToggleClustersGood_triggered(self):
        """'Good' button (G) click. For simple merging of clusters, easier to
        use than running climb() on selected clusters using a really big sigma to force
        them to all merge"""
        spw = self.spykewindow
        clusters = spw.GetClusters()
        for cluster in clusters:
            cluster.neuron.good = not cluster.neuron.good
        self.nlist.updateAll() # nlist item colouring will change as a result

    def on_actionLabelMultiunit_triggered(self):
        """- button click. Label all selected clusters as multiunit by deleting them
        and creating new ones with -ve IDs"""
        spw = self.spykewindow
        clusters = spw.GetClusters()
        s = self.sort
        spikes = s.spikes
        # only relabel single unit clusters:
        clusters = [ cluster for cluster in clusters if cluster.id > 0 ]
        if len(clusters) == 0:
            return
        sids = []
        for cluster in clusters:
            sids.append(cluster.neuron.sids)
        sids = np.concatenate(sids)

        # save some undo/redo stuff
        message = 'label as multiunit clusters %r' % [ c.id for c in clusters ]
        cc = ClusterChange(sids, spikes, message)
        cc.save_old(clusters, s.norder)

        # delete old clusters
        inserti = s.norder.index(clusters[0].id)
        # collect cluster sids before cluster deletion
        sidss = [ cluster.neuron.sids for cluster in clusters ]
        spw.DelClusters(clusters, update=False)

        # create new multiunit clusters
        newclusters = []
        for sids in sidss:
            muid = s.get_nextmuid()
            newcluster = spw.CreateCluster(update=False, id=muid, inserti=inserti)
            neuron = newcluster.neuron
            self.MoveSpikes2Neuron(sids, neuron, update=False)
            newcluster.update_pos()
            newclusters.append(newcluster)
            inserti += 1

        # select newly labelled multiunit clusters
        spw.SelectClusters(newclusters)

        # save more undo/redo stuff
        cc.save_new(newclusters, s.norder)
        spw.AddClusterChangeToStack(cc)
        print(cc.message)

    def on_actionChanSplitClusters_triggered(self):
        """Split by channels button (/) click"""
        self.spykewindow.chansplit()

    def on_actionRandomSplit_triggered(self):
        """Randomly split each selected cluster if it has a population
        of spikes > MAXNCLIMBPOINTS"""
        self.spykewindow.randomsplit()

    def on_actionRenumberClusters_triggered(self):
        """Renumber single unit clusters consecutively from 1, ordered by y position,
        on "#" button click. Do the same for multiunit (-ve number) clusters, starting
        from -1. Sorting by y position makes user inspection of clusters more orderly,
        makes the presence of duplicate clusters more obvious, and allows for maximal
        spatial separation between clusters of the same colour, reducing colour
        conflicts"""
        spw = self.spykewindow
        s = self.sort
        spikes = s.spikes

        # get spatially and numerically ordered lists of new ids
        oldids = np.asarray(s.norder)
        oldsuids = oldids[oldids > 0]
        oldmuids = oldids[oldids < 0]
        # this is a bit confusing: find indices that would sort old ids by y pos, but then
        # what you really want is to find the y pos *rank* of each old id, so you need to
        # take argsort again:
        newsuids = np.asarray([ s.clusters[cid].pos['y0'] for cid in oldsuids ]).argsort().argsort() + 1
        newmuids = np.asarray([ s.clusters[cid].pos['y0'] for cid in oldmuids ]).argsort().argsort() + 1
        newmuids = -newmuids
        # multiunit, followed by single unit, no 0 junk cluster. Can't seem to do it the other
        # way around as of Qt 4.7.2 - it seems QListViews don't like having a -ve value in
        # the last entry. Doing so causes all 2 digit values in the list to become blank,
        # suggests a spacing calculation bug. Reproduce by making last entry multiunit, undoing,
        # then redoing. Actually, maybe the bug is it doesn't like having a number in the last
        # entry with fewer digits than the preceding entry. Only seems to be a problem when
        # setting self.setUniformItemSizes(True).
        newids = np.concatenate([newmuids, newsuids])

        # test
        if np.all(oldids == newids):
            print('nothing to renumber: cluster IDs already ordered in y0 and contiguous')
            return
        oldids = np.concatenate([oldmuids, oldsuids]) # update for replacing oldids with newids

        # deselect current selections
        selclusters = spw.GetClusters()
        oldselids = [ cluster.id for cluster in selclusters ]
        spw.SelectClusters(selclusters, on=False)

        # delete junk cluster, if it exists
        if 0 in s.clusters:
            s.remove_neuron(0)
            print('deleted junk cluster 0')
        if 0 in oldselids:
            oldselids.remove(0)

        # replace old ids with new ids
        cw = spw.windows['Cluster']
        oldclusters = s.clusters.copy() # no need to deepcopy, just copy refs, not clusters
        dims = spw.GetClusterPlotDims()
        for oldid, newid in zip(oldids, newids):
            newid = int(newid) # keep as Python int, not numpy int
            if oldid == newid:
                continue # no need to waste time removing and recreating this cluster
            # change all occurences of oldid to newid
            cluster = oldclusters[oldid]
            cluster.id = newid # this indirectly updates neuron.id
            # update cluster and neuron dicts
            s.clusters[newid] = cluster
            s.neurons[newid] = cluster.neuron
            sids = cluster.neuron.sids
            spikes['nid'][sids] = newid

        # remove any orphaned cluster ids
        for oldid in oldids:
            if oldid not in newids:
                del s.clusters[oldid]
                del s.neurons[oldid]

        # reset norder
        s.norder = []
        s.norder.extend(sorted([ int(newid) for newid in newmuids ])[::-1])
        s.norder.extend(sorted([ int(newid) for newid in newsuids ]))

        # now do some final updates
        spw.UpdateClustersGUI()
        spw.ColourPoints(s.clusters.values())
        # reselect the previously selected (but now renumbered) clusters,
        # helps user keep track
        oldiis = [ list(oldids).index(oldselid) for oldselid in oldselids ]
        newselids = newids[oldiis]
        spw.SelectClusters([s.clusters[cid] for cid in newselids])
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
        gw = spw.windows['Cluster'].glWidget
        dims = spw.GetClusterPlotDims()
        gw.focus = np.float32([ cluster.normpos[dim] for dim in dims ])
        gw.panTo() # pan to new focus
        gw.updateGL()

    def on_actionFocusCurrentSpike_triggered(self):
        """Move focus to location of currently selected (single) spike"""
        spw = self.spykewindow
        try:
            sid = spw.GetSpike()
        except RuntimeError, msg:
            print(msg)
            return
        gw = spw.windows['Cluster'].glWidget
        pointis = gw.sids.searchsorted(sid)
        gw.focus = gw.points[pointis]
        gw.panTo() # pan to new focus
        gw.updateGL()

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

    def on_actionReloadSpikes_triggered(self):
        spw = self.spykewindow
        sids = spw.GetAllSpikes()
        sort = self.sort
        usemeanchans = False
        if QtGui.QApplication.instance().keyboardModifiers() == Qt.ShiftModifier:
            usemeanchans = True
        self.sort.reloadSpikes(sids, fixtvals=True, usemeanchans=usemeanchans)
        # add sids to the set of dirtysids to be resaved to .wave file:
        spw.dirtysids.update(sids)
        # update neuron templates:
        unids = np.unique(sort.spikes['nid'][sids])
        neurons = [ sort.neurons[nid] for nid in unids ]
        for neuron in neurons:
            neuron.update_wave() # update affected mean waveforms
        # auto-refresh all plots
        self.panel.updateAllItems()

    def on_actionFindPrevMostSimilar_triggered(self):
        self.findMostSimilarCluster('previous')

    def on_actionFindNextMostSimilar_triggered(self):
        self.findMostSimilarCluster('next')

    def on_actionPlotClusterHist_triggered(self):
        """Plot histogram of selected clusters along a single dimension. If one cluster
        selected, plot its distribution along its first (x) dimension. If two clusters
        are selected, project them onto axis connecting their centers, and calculate
        separation index between them. Find max of the two stdevs of projections of points
        from both clusters. Take ratio of distance between the cluster centers and
        3*maxstdevs to get separation index. An index < 1 suggests the two clusters are
        not significantly separated from each other.
                
        Another way would be to simply take the fraction of area that the two distribs
        overlap. For the two distribs, at each bin, take min value of the two. Add up all
        those min values, and divide by the mass of the smaller distrib.

        Or, could instead take sqrt of Jensen Shannon divergence, which is a metric.
        """
        spw = self.spykewindow
        clusters = spw.GetClusters()
        if len(clusters) == 0:
            print("no clusters selected")
            return
        if len(clusters) == 2:
            calc_measures = True
        else:
            calc_measures = False
            projdimi = 0
        # get param matrix X for points in all clusters, given current dim and
        # channel selection:
        dims = spw.GetClusterPlotDims()
        sids = np.concatenate([ cluster.neuron.sids for cluster in clusters ])
        sids.sort()
        try:
            X, sids = spw.get_param_matrix(sids=sids, dims=dims, scale=True)
        except RuntimeError, errmsg:
            print(errmsg)
            return
        points = [] # list of projection of each cluster's points onto dimi
        for cluster in clusters:
            sidis = sids.searchsorted(cluster.neuron.sids)
            points.append(np.ascontiguousarray(X[sidis]))
        if calc_measures:
            t0 = time.time()
            sepmetric = util.NDsepmetric(*points)
            print('sepmetric calc took %.3f sec' % (time.time()-t0))
            # centers of both clusters, use median:
            c0 = np.median(points[0], axis=0) # ndims vector
            c1 = np.median(points[1], axis=0)
            # line connecting the centers of the two clusters, wrt c0
            line = c1-c0
            line /= np.linalg.norm(line) # make it unit length
            #print('c0=%r, c1=%r, line=%r' % (c0, c1, line))
        else:
            line = np.zeros(len(dims))
            line[projdimi] = 1.0 # pick out just the one component
            c0 = 0.0 # set origin at 0
        # calculate projection of each cluster's points onto line
        projs = []
        for cpoints in points:
            projs.append(np.dot(cpoints-c0, line))
        if calc_measures:
            d = np.linalg.norm(np.median(projs[1]) - np.median(projs[0]))
            # measure whether centers are at least 3 of the bigger stdevs away from
            # each other:
            sepindex = d / (3 * max(projs[0].std(), projs[1].std()))
            #print('std0=%f, std1=%f, d=%f' % (projs[0].std(), projs[1].std(), d))
        proj = np.concatenate(projs)
        nbins = intround(np.sqrt(len(proj))) # seems like a good heuristic
        #print('nbins = %d' % nbins)
        edges = np.histogram(proj, bins=nbins)[1]
        hists = []
        for ci, cluster in enumerate(clusters):
            hists.append(np.histogram(projs[ci], bins=edges)[0])
        hist = np.concatenate([hists]) # one cluster hist per row
        masses = np.asarray([ h.sum() for h in hist ])
        sortedmassi = masses.argsort()
        # Take the fraction of area that the two distribs overlap.
        # At each bin, take min value of the two distribs. Add up all those min values,
        # and divide by the mass of the smaller distrib.
        if calc_measures:
            overlaparearatio = hist.min(axis=0).sum() / masses[sortedmassi[0]]
            djs = core.DJS(hists[0], hists[1])
        # plotting:
        ledges = edges[:-1] # keep just the left edges, discard the last right edge
        assert len(ledges) == nbins
        binwidth = ledges[1] - ledges[0]
        # plot:
        mplw = spw.OpenWindow('MPL')
        a = mplw.ax
        a.clear()
        windowtitle = "clusters %r" % ([ cluster.id for cluster in clusters ])
        print(windowtitle)
        mplw.setWindowTitle(windowtitle)
        if calc_measures:
            #title = ("sep index=%.3f, overlap area ratio=%.3f, DJS=%.3f, sqrt(DJS)=%.3f"
            #         % (sepindex, overlaparearatio, djs, np.sqrt(djs)))
            title = ("sep metric=%.3f, sep index=%.3f, overlap area ratio=%.3f, DJS=%.3f"
                     % (sepmetric, sepindex, overlaparearatio, djs))
            print(title)
            a.set_title(title)
        cs = core.rgb2hex([ cluster.color for cluster in clusters ])
        for i, c in enumerate(cs):
            if c == WHITE:
                cs[i] = 'black'
        # plot the smaller cluster last, to maximize visibility:
        for i in sortedmassi[::-1]:
            a.bar(ledges, hist[i], width=binwidth, color=cs[i], edgecolor=cs[i])
        mplw.figurecanvas.draw()

    def on_actionSave_triggered(self):
        """Save sort panel to file"""
        f = self.panel.figure

        # copied from matplotlib.backend_qt4.NavigationToolbar2QT.save_figure():
        filetypes = f.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        default_filetype = f.canvas.get_default_filetype()

        start = f.canvas.get_default_filename()
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        #from matplotlib.backends.qt4_compat import _getSaveFileName
        fname = QtGui.QFileDialog.getSaveFileName(self.panel, "Save sort panel to",
                                                  start, filters, selectedFilter)
        if fname:
            fname = str(fname) # convert from QString
            try:
                f.canvas.print_figure(fname, facecolor=None, edgecolor=None)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self.panel, "Error saving file", str(e),
                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)
            print('sort panel saved to %r' % fname)
        
    def on_slider_valueChanged(self, slideri):
        self.nslist.clearSelection() # emits selectionChanged signal, .reset() doesn't
        if self.nslist.model().sliding == False:
            self.nslist.model().sids.sort() # change from nid order to sid order
            self.nslist.updateAll() # update to reflect new ordering
            self.nslist.model().sliding = True
        nsamples = int(self.nsamplesComboBox.currentText())
        rows = np.arange(slideri, slideri+nsamples)
        self.nslist.selectRows(rows)

    def on_slider_sliderPressed(self):
        """Make slider click (without movement) highlight the first nsamples
        or fewer spikes when slider is at 0 position"""
        slideri = self.slider.value()
        if slideri == 0:
            nsamples = int(self.nsamplesComboBox.currentText())
            nsamples = min(nsamples, self.nslist.model().nspikes)
            rows = np.arange(nsamples)
            self.nslist.selectRows(rows)

    def update_slider(self):
        """Update slider limits and step sizes"""
        nsamples = int(self.nsamplesComboBox.currentText())
        nsids = len(self.nslist.sids)
        ulim = max(nsids-nsamples, 1) # upper limit
        self.slider.setRange(0, ulim)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(nsamples)

    def findMostSimilarCluster(self, which='next'):
        """If no chans selected, compare source to next or previous most similar cluster
        based on chans the two have in common, while requiring the two have each others'
        max chans in common. If chans have been selected, use them as a starting set of
        chans to compare on"""
        try:
            source = self.getClusterComparisonSource()
        except RuntimeError, errmsg:
            print(errmsg)
            return
        destinations = self.sort.clusters.values()
        destinations.remove(source)
        selchans = np.sort(self.panel.chans_selected)
        if len(selchans) > 0:
            srcchans = np.intersect1d(source.neuron.wave.chans, selchans)
            if len(srcchans) == 0:
                print("source cluster doesn't overlap with selected chans")
                return
        else:
            srcchans = source.neuron.wave.chans
        errors = []
        dests = []
        # try and compare source neuron waveform to all destination neuron waveforms
        for dest in destinations:
            if dest.neuron.wave.data == None: # hasn't been calculated yet
                dest.neuron.update_wave()
            dstchans = dest.neuron.wave.chans
            if len(selchans) > 0:
                if not set(selchans).issubset(dstchans):
                    continue
                dstchans = selchans
            cmpchans = np.intersect1d(srcchans, dstchans)
            if len(cmpchans) == 0: # not comparable
                continue
            # ensure maxchan of both source and dest neuron are both in cmpchans
            if source.neuron.chan not in cmpchans or dest.neuron.chan not in cmpchans:
                continue
            srcwavedata = source.neuron.wave[cmpchans].data
            dstwavedata = dest.neuron.wave[cmpchans].data
            error = core.rms(srcwavedata - dstwavedata)
            errors.append(error)
            dests.append(dest)
        if len(errors) == 0:
            print("no sufficiently overlapping clusters on selected chans to compare to")
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
            self._cmpid = -1 # init/reset
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
        """Align all implicitly selected spikes to min or max, or best fit
        on selected chans"""        
        s = self.sort
        spikes = s.spikes
        spw = self.spykewindow
        sids = np.concatenate((spw.GetClusterSpikes(), spw.GetUnsortedSpikes()))
        if to == 'best':
            selchans = spw.get_selchans(sids)
            # find which chans are common to all sids
            chanss = spikes['chans'][sids]
            nchanss = spikes['nchans'][sids]
            chanslist = [ chans[:nchans] for chans, nchans in zip(chanss, nchanss) ] # array list
            common_chans = core.intersect1d(chanslist) # find intersection
            # check selected chans
            for selchan in selchans:
                if selchan not in common_chans:
                    raise RuntimeError("chan %d not common to all spikes, pick from %r"
                                       % (selchan, list(common_chans)))
            print('doing best fit alignment on %d spikes on chans %r' % (len(sids), selchans))
            dirtysids = s.alignbest(sids, selchans)
        else: # to in ['min', 'max']
            print('doing %s alignment on %d spikes' % (to, len(sids)))
            dirtysids = s.alignminmax(sids, to)
        print('aligned %d spikes' % len(dirtysids))
        unids = np.unique(spikes['nid'][dirtysids])
        neurons = [ s.neurons[nid] for nid in unids ]
        for neuron in neurons:
            neuron.update_wave() # update affected mean waveforms
        # add dirtysids to the set to be resaved to .wave file:
        spw.dirtysids.update(dirtysids)
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
        neuron.sids = np.setdiff1d(neuron.sids, sids) # return what's in 1st arr and not in 2nd
        spikes['nid'][sids] = 0 # unbind neuron id of sids in spikes struct array
        if update:
            self.sort.update_usids()
            self.uslist.updateAll()
        # this only makes sense if the neuron is currently selected in the nlist:
        if neuron in self.nslist.neurons:
            self.nslist.neurons = self.nslist.neurons # this triggers a refresh
        neuron.wave.data = None # triggers an update when it's actually needed

