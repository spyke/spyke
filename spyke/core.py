"""Core classes and functions used throughout spyke"""

from __future__ import division
from __future__ import with_statement

__authors__ = ['Martin Spacek', 'Reza Lotun']

import cPickle
import gzip
import hashlib
import time
from datetime import timedelta
import os
import sys
import random
import string
from copy import copy
import datetime

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

import numpy as np
from numpy import pi

# set some numpy options - these should hold for all modules in spyke
np.set_printoptions(precision=3)
np.set_printoptions(threshold=1000)
np.set_printoptions(edgeitems=5)
np.set_printoptions(linewidth=150)
np.set_printoptions(suppress=True)
# make overflow, underflow, div by zero, and invalid all raise errors
# this really should be the default in numpy...
np.seterr(all='raise')

import probes
from probes import uMap54_1a, uMap54_1b, uMap54_1c, uMap54_2a, uMap54_2b

UNIXEPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0) # UNIX epoch: Jan 1, 1970

MU = '\xb5' # greek mu symbol
MICRO = 'u'

DEFHIGHPASSSAMPFREQ = 50000 # default (possibly interpolated) high pass sample frequency, in Hz
DEFHIGHPASSSHCORRECT = True
KERNELSIZE = 12 # apparently == number of kernel zero crossings, but that seems to depend on the phase of the kernel, some have one less. Anyway, total number of points in the kernel is this plus 1 (for the middle point) - see Blanche2006
assert KERNELSIZE % 2 == 0 # I think kernel size needs to be even
NCHANSPERBOARD = 32 # TODO: stop hard coding this

TW = -500, 500 # spike time window range, us, centered on thresh xing or main phase of spike

MAXLONGLONG = 2**63-1
MAXNBYTESTOFILE = 2**31 # max array size safe to call .tofile() on in Numpy 1.5.0 on Windows

MAXNCLIMBPOINTS = 25000

CHANFIELDLEN = 256 # channel string field length at start of .resample file

INVPI = 1 / pi


class EmptyClass(object):
    pass


class Converter(object):
    """Simple object to store intgain and extgain values and
    provide methods to convert between AD and uV values, even
    when a .srf file (and associated Stream where intgain
    and extgain are stored) isn't available"""
    def __init__(self, intgain, extgain):
        self.intgain = intgain
        self.extgain = extgain

    def AD2uV(self, AD):
        """Convert rescaled AD values to float32 uV
        Biggest +ve voltage is 10 million uV, biggest +ve rescaled signed int16 AD val
        is half of 16 bits, then divide by internal and external gains

        TODO: unsure: does the DT3010 acquire from -10 to 10 V at intgain == 1 and encode
        that from 0 to 4095?
        """
        return np.float32(AD) * 10000000 / (2**15 * self.intgain * self.extgain)

    def uV2AD(self, uV):
        """Convert uV to signed rescaled int16 AD values"""
        return np.int16(np.round(uV * (2**15 * self.intgain * self.extgain) / 10000000))


class WaveForm(object):
    """Just a container for data, std of data, timestamps, and channels.
    Sliceable in time, and indexable in channel space. Only really used for
    convenient plotting. Everything else uses the sort.wavedata array, and
    related sort.spikes fields"""
    def __init__(self, data=None, std=None, ts=None, chans=None):
        self.data = data # in AD, potentially multichannel, depending on shape
        self.std = std # std of data
        self.ts = ts # timestamps array in us, one for each sample (column) in data
        self.chans = chans # channel ids corresponding to rows in .data

    def __getitem__(self, key):
        """Make waveform data sliceable in time, and directly indexable by channel id(s).
        Return a new WaveForm"""
        
        # check for std field, won't exist for old saved Waveforms in .sort files:
        try: self.std
        except AttributeError: self.std = None
        
        if type(key) == slice: # slice self in time
            if self.ts == None:
                return WaveForm() # empty WaveForm
            else:
                lo, hi = self.ts.searchsorted([key.start, key.stop])
                data = self.data[:, lo:hi]
                if self.std == None:
                    std = None
                else:
                    std = self.std[:, lo:hi]
                ts = self.ts[lo:hi]
                #if np.asarray(data == self.data).all() and np.asarray(ts == self.ts).all():
                #    return self # no need for a new WaveForm - but new WaveForms aren't expensive, only new data are
                return WaveForm(data=data, std=std, ts=ts, chans=self.chans) # return a new WaveForm
        else: # index into self by channel id(s)
            keys = toiter(key)
            #try: assert (self.chans == np.sort(self.chans)).all() # testing code
            #except AssertionError: import pdb; pdb.set_trace() # testing code
            try:
                assert set(keys).issubset(self.chans), "requested channels outside of channels in waveform"
                #assert len(set(keys)) == len(keys), "same channel specified more than once" # this is fine
            except AssertionError:
                raise IndexError('invalid index %r' % key)
            #i = self.chans.searchsorted(keys) # indices into rows of data
            # best not to assume that chans are sorted, often the case in LFP data:
            i = [ int(np.where(chan == self.chans)[0]) for chan in keys ] # indices into rows of data
            data = self.data[i] # grab the appropriate rows of data
            if self.std == None:
                std = None
            else:
                std = self.std[i]
            return WaveForm(data=data, std=std, ts=self.ts, chans=keys) # return a new WaveForm

    def __len__(self):
        """Number of data points in time"""
        nt = len(self.ts)
        assert nt == self.data.shape[1] # obsessive
        return nt

    def _check_add_sub(self, other):
        """Check a few things before adding or subtracting waveforms"""
        if self.data.shape != other.data.shape:
            raise ValueError("Waveform shapes %r and %r don't match" %
                             (self.data.shape, other.data.shape))
        if self.chans != other.chans:
            raise ValueError("Waveform channel ids %r and %r don't match" %
                             (self.chans, other.chans))

    def __add__(self, other):
        """Return new waveform which is self+other. Keep self's timestamps"""
        self._check_add_sub(other)
        return WaveForm(data=self.data+other.data,
                        ts=self.ts, chans=self.chans)

    def __sub__(self, other):
        """Return new waveform which is self-other. Keep self's timestamps"""
        self._check_add_sub(other)
        return WaveForm(data=self.data-other.data,
                        ts=self.ts, chans=self.chans)
    '''
    def get_padded_data(self, chans):
        """Return self.data corresponding to self.chans,
        padded with zeros for chans that don't exist in self"""
        common = set(self.chans).intersection(chans) # overlapping chans
        dtype = self.data.dtype # self.data corresponds to self.chans
        padded_data = np.zeros((len(chans), len(self.ts)), dtype=dtype) # padded_data corresponds to chans
        chanis = [] # indices into self.chans corresponding to overlapping chans
        commonis = [] # indices into chans corresponding to overlapping chans
        for chan in common:
            chani, = np.where(chan == np.asarray(self.chans))
            commoni, = np.where(chan == np.asarray(chans))
            chanis.append(chani)
            commonis.append(commoni)
        chanis = np.concatenate(chanis)
        commonis = np.concatenate(commonis)
        padded_data[commonis] = self.data[chanis] # for overlapping chans, overwrite the zeros with data
        return padded_data
    '''

class TrackStream(object):
    """A collection of streams, all from the same track. This is used to simultaneously
    cluster all spikes from many (or all) recordings from the same track. Designed to have
    as similar an interface as possible to a normal Stream. srffs needs to be a list of
    open and parsed surf.File objects, in temporal order"""
    def __init__(self, srffs, trackfname, kind='highpass', sampfreq=None, shcorrect=None):
        # to prevent pickling problems, don't bind srffs
        self.fname = trackfname
        self.kind = kind
        streams = []
        self.streams = streams # bind right away so setting sampfreq and shcorrect will work
        # collect appropriate streams from srffs
        if kind == 'highpass':
            for srff in srffs:
                streams.append(srff.hpstream)
        elif kind == 'lowpass':
            for srff in srffs:
                streams.append(srff.lpstream)
        else: raise ValueError('Unknown stream kind %r' % kind)

        datetimes = [stream.datetime for stream in streams]
        if not (np.diff(datetimes) >= timedelta(0)).all():
            raise RuntimeError(".srf files aren't in temporal order")

        """Generate tranges, an array of all the contiguous data ranges in all the
        streams in self. These are relative to the start of acquisition (t=0) in the first
        stream. Also generate streamtranges, an array of each stream's t0 and t1"""
        tranges = []
        streamtranges = []
        for stream in streams:
            td = stream.datetime - datetimes[0] # time delta between stream i and stream 0
            for trange in stream.tranges:
                t0 = td2usec(td + timedelta(microseconds=int(trange[0])))
                t1 = td2usec(td + timedelta(microseconds=int(trange[1])))
                tranges.append([t0, t1])
            streamt0 = td2usec(td + timedelta(microseconds=int(stream.t0)))
            streamt1 = td2usec(td + timedelta(microseconds=int(stream.t1)))
            streamtranges.append([streamt0, streamt1])
        self.tranges = np.int64(tranges)
        self.streamtranges = np.int64(streamtranges)
        self.t0 = self.streamtranges[0, 0]
        self.t1 = self.streamtranges[-1, 1]

        self.layout = streams[0].layout # assume they're identical
        intgains = np.asarray([ stream.converter.intgain for stream in streams ])
        if max(intgains) != min(intgains):
            import pdb; pdb.set_trace() # investigate which are the deviant .srf files
            raise NotImplementedError("not all .srf files have the same intgain")
            # TODO: find recording with biggest intgain, call that value maxintgain. For each
            # recording, scale its AD values by its intgain/maxintgain when returning a slice
            # from its stream. Note that this ratio should always be a factor of 2, so all you
            # have to do is bitshift, I think. Then, have a single converter for the
            # trackstream whose intgain value is set to maxintgain
        self.converter = streams[0].converter # they're identical
        self.srffnames = [srff.fname for srff in srffs]
        self.rawsampfreq = streams[0].rawsampfreq # assume they're identical
        self.rawtres = streams[0].rawtres # assume they're identical
        contiguous = np.asarray([stream.contiguous for stream in streams])
        if not contiguous.all() and kind == 'highpass': # don't bother reporting again for lowpass
            fnames = [ s.fname for s, c in zip(streams, contiguous) if not c ]
            print("some .srf files are non contiguous:")
            for fname in fnames:
                print(fname)
        probe = streams[0].probe
        if not np.all([type(probe) == type(stream.probe) for stream in streams]):
            raise RuntimeError("some .srf files have different probe types")
        self.probe = probe # they're identical

        # set sampfreq and shcorrect for all streams
        if kind == 'highpass':
            self.sampfreq = sampfreq or DEFHIGHPASSSAMPFREQ # desired sampling frequency
            self.shcorrect = shcorrect or DEFHIGHPASSSHCORRECT
        else: # kind == 'lowpass'
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default

    def is_open(self):
        return np.all([stream.is_open() for stream in self.streams])

    def open(self):
        for stream in self.streams:
            stream.open()

    def close(self):
        for stream in self.streams:
            stream.close()

    def get_chans(self):
        return self.streams[0].chans # assume they're identical

    def set_chans(self, chans):
        for stream in self.streams:
            stream.chans = chans

    chans = property(get_chans, set_chans)

    def get_nchans(self):
        return len(self.chans)

    nchans = property(get_nchans)

    def get_sampfreq(self):
        return self.streams[0].sampfreq # they're identical

    def set_sampfreq(self, sampfreq):
        for stream in self.streams:
            stream.sampfreq = sampfreq

    sampfreq = property(get_sampfreq, set_sampfreq)

    def get_tres(self):
        return self.streams[0].tres # they're identical

    tres = property(get_tres)

    def get_shcorrect(self):
        return self.streams[0].shcorrect # they're identical

    def set_shcorrect(self, shcorrect):
        for stream in self.streams:
            stream.shcorrect = shcorrect

    shcorrect = property(get_shcorrect, set_shcorrect)
    '''
    # having this would make sense, but it isn't currently needed:
    def get_datetime(self):
        return self.streams[0].datetime # datetime of first stream

    datetime = property(get_datetime)
    '''
    def pickle(self):
        """Just a way to pickle all the .srf files associated with self"""
        for stream in self.streams:
            stream.pickle()

    def __getitem__(self, key):
        """Figure out which stream(s) the slice spans (usually just one, sometimes 0 or
        2), send the request to the stream(s), generate the appropriate timestamps, and
        return the waveform"""
        if key.step not in [None, 1]:
            raise ValueError('unsupported slice step size: %s' % key.step)
        tres = self.tres
        start, stop = max(key.start, self.t0), min(key.stop, self.t1) # stay in bounds
        streamis = []
        # TODO: this could probably be more efficient by not iterating over all streams:
        for streami, trange in enumerate(self.streamtranges):
            if (trange[0] <= start < trange[1]) or (trange[0] <= stop < trange[1]):
                streamis.append(streami)
        ts = np.arange(start, stop, tres)
        data = np.zeros((self.nchans, len(ts)), dtype=np.int16) # any gaps will have zeros
        for streami in streamis:
            stream = self.streams[streami]
            abst0 = self.streamtranges[streami, 0] # absolute start time of stream
            # find start and end offsets relative to abst0
            relt0 = max(start - abst0, 0) # stay within stream's lower limit
            relt1 = min(stop - abst0, stream.t1 - stream.t0) # stay within stream's upper limit
            # source slice times:
            st0 = relt0 + stream.t0
            st1 = relt1 + stream.t0
            sdata = stream[st0:st1].data # source data
            # destination time indices:
            dt0i = (abst0 + relt0 - start) // tres # absolute index
            dt1i = dt0i + sdata.shape[1]
            data[:, dt0i:dt1i] = sdata
        return WaveForm(data=data, ts=ts, chans=self.chans)


class Stream(object):
    """Data stream object - provides convenient stream interface to .srf files.
    Maps from timestamps to record index of stream data to retrieve the
    approriate range of waveform data from disk"""
    def __init__(self, srff, kind='highpass', sampfreq=None, shcorrect=None):
        """Takes a sorted temporal (not necessarily evenly-spaced, due to pauses in recording)
        sequence of ContinuousRecords: either HighPassRecords or LowPassMultiChanRecords.
        sampfreq arg is useful for interpolation. Assumes that all HighPassRecords belong
        to the same probe. srff must be open and parsed"""
        self.srff = srff
        self.kind = kind
        if kind == 'highpass':
            self.records = srff.highpassrecords
        elif kind == 'lowpass':
            self.records = srff.lowpassmultichanrecords
        else: raise ValueError('Unknown stream kind %r' % kind)

        # assume same layout for all records of type "kind"
        self.layout = self.srff.layoutrecords[self.records['Probe'][0]]
        intgain = self.layout.intgain
        extgain = int(self.layout.extgain[0]) # assume same extgain for all chans in layout
        self.converter = Converter(intgain, extgain)
        self.nADchans = self.layout.nchans # always constant
        self.rawsampfreq = self.layout.sampfreqperchan
        self.rawtres = intround(1 / self.rawsampfreq * 1e6) # us
        if kind == 'highpass':
            ADchans = self.layout.ADchanlist
            if list(self.layout.ADchanlist) != range(self.nADchans):
                raise ValueError("ADchans aren't contiguous from 0, highpass recordings are "
                                 "nonstandard, and assumptions made for resampling are wrong")
            # probe chans, as opposed to AD chans. Don't know yet of any probe
            # type whose chans aren't contiguous from 0 (see probes.py)
            self.chans = np.arange(self.nADchans)
            self.sampfreq = sampfreq or DEFHIGHPASSSAMPFREQ # desired sampling frequency
            self.shcorrect = shcorrect or DEFHIGHPASSSHCORRECT
        else: # kind == 'lowpass'
            # probe chan values are already parsed from LFP probe description
            self.chans = self.layout.chans
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default
        probename = self.layout.electrode_name
        probename = probename.replace(MU, 'u') # replace any 'micro' symbols with 'u'
        probetype = eval('probes.' + probename) # yucky. TODO: switch to a dict with keywords?
        self.probe = probetype() # instantiate it

        rts = self.records['TimeStamp'] # array of record timestamps
        NumSamples = np.unique(self.records['NumSamples'])
        if len(NumSamples) > 1:
            raise RuntimeError("Not all continuous records are of the same length. "
                               "NumSamples = %r" % NumSamples)
        rtlen = NumSamples / self.nADchans * self.rawtres
        # Check whether rts values are all equally spaced, indicating there were no
        # pauses in recording
        diffrts = np.diff(rts)
        self.contiguous = (np.diff(diffrts) == 0).all() # could also call diff(rts, n=2)
        if self.contiguous:
            try: assert np.unique(diffrts) == rtlen
            except AssertionError: import pdb; pdb.set_trace()
            self.tranges = np.int64([[rts[0], rts[-1]+rtlen]]) # keep it 2D
        else:
            if kind == 'highpass': # don't bother reporting again for lowpass
                print('NOTE: time gaps exist in %s, possibly due to pauses' % self.fname)
            # build up self.tranges
            splitis = np.where(diffrts != rtlen)[0] + 1
            splits = np.split(rts, splitis) # list of arrays of contiguous rts
            tranges = []
            for split in splits: # for each array of contiguous rts
                tranges.append([split[0], split[-1]+rtlen])
            self.tranges = np.int64(tranges)
        self.t0 = self.tranges[0, 0]
        self.t1 = self.tranges[-1, 1]

    def is_open(self):
        return self.srff.is_open()

    def open(self):
        self.srff.open()

    def close(self):
        self.srff.close()

    def get_fname(self):
        return self.srff.fname

    fname = property(get_fname)

    def get_srffnames(self):
        return [self.srff.fname]

    srffnames = property(get_srffnames)

    def get_srcfnameroot(self):
        return lrstrip(self.fname, '../', '.srf')

    srcfnameroot = property(get_srcfnameroot)

    def get_nchans(self):
        return len(self.chans)

    nchans = property(get_nchans)

    def get_sampfreq(self):
        return self._sampfreq

    def set_sampfreq(self, sampfreq):
        """On .sampfreq change, delete .kernels (if set), and update .tres"""
        self._sampfreq = sampfreq
        try:
            del self.kernels
        except AttributeError:
            pass
        self.tres = intround(1 / self.sampfreq * 1e6) # us, for convenience

    sampfreq = property(get_sampfreq, set_sampfreq)

    def get_shcorrect(self):
        return self._shcorrect

    def set_shcorrect(self, shcorrect):
        """On .shcorrect change, deletes .kernels (if set)"""
        self._shcorrect = shcorrect
        try:
            del self.kernels
        except AttributeError:
            pass

    shcorrect = property(get_shcorrect, set_shcorrect)

    def get_datetime(self):
        return self.srff.datetime

    datetime = property(get_datetime)

    def pickle(self):
        self.srff.pickle()

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us. Returns the corresponding WaveForm
        object, which has as its attribs the 2D multichannel waveform array as well
        as the timepoints, potentially spanning multiple ContinuousRecords"""
        if key.step not in [None, 1]:
            raise ValueError('unsupported slice step size: %s' % key.step)

        nADchans = self.nADchans
        rawtres = self.rawtres
        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at
            # key.start and key.stop
            xs = KERNELSIZE * rawtres
        else:
            xs = 0
        # get a slightly greater range of raw data (with xs) than might be needed:
        t0xsi = (key.start - xs) // rawtres # round down to nearest mult of rawtres
        t1xsi = ((key.stop + xs) // rawtres) + 1 # round up to nearest mult of rawtres
        # stay within stream limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, self.t0 // rawtres)
        t1xsi = min(t1xsi, self.t1 // rawtres)
        # convert back to us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        tsxs = np.arange(t0xs, t1xs, rawtres)
        ntxs = len(tsxs)
        # init data as int32 so we have bitwidth to rescale and zero, then convert to int16
        dataxs = np.zeros((nADchans, ntxs), dtype=np.int32) # any gaps will have zeros
        # first and last record indices corresponding to the slice
        loreci, hireci = self.records['TimeStamp'].searchsorted([t0xs, t1xs], side='right')
        # always get back at least 1 record
        records = self.records[max(loreci-1, 0):max(hireci, 1)]

        # load up data+excess, from all relevant records
        # TODO: fix code duplication
        #tload = time.time()
        if self.kind == 'highpass': # straightforward
            for record in records: # iterating over highpass records
                d = self.srff.loadContinuousRecord(record) # get record's data
                nt = d.shape[1]
                t0i = record['TimeStamp'] // rawtres
                t1i = t0i + nt
                # source indices
                st0i = max(t0xsi - t0i, 0)
                st1i = min(t1xsi - t0i, nt)
                # destination indices
                dt0i = max(t0i - t0xsi, 0)
                dt1i = min(t1i - t0xsi, ntxs)
                dataxs[:, dt0i:dt1i] = d[:, st0i:st1i]
        else: # kind == 'lowpass', need to load chans from subsequent records
            nt = records[0]['NumSamples'] / nADchans # assume all lpmc records are same length
            d = np.zeros((nADchans, nt), dtype=np.int32)
            for record in records: # iterating over lowpassmultichan records
                for chani in range(nADchans):
                    lprec = self.srff.lowpassrecords[record['lpreci']+chani]
                    d[chani] = self.srff.loadContinuousRecord(lprec)
                t0i = record['TimeStamp'] // rawtres
                t1i = t0i + nt
                # source indices
                st0i = max(t0xsi - t0i, 0)
                st1i = min(t1xsi - t0i, nt)
                # destination indices
                dt0i = max(t0i - t0xsi, 0)
                dt1i = min(t1i - t0xsi, ntxs)
                dataxs[:, dt0i:dt1i] = d[:, st0i:st1i]
        #print('record.load() took %.3f sec' % (time.time()-tload))

        # bitshift left to scale 12 bit values to use full 16 bit dynamic range, same as
        # * 2**(16-12) == 16. This provides more fidelity for interpolation, reduces uV per
        # AD to about 0.02
        dataxs <<= 4 # data is still int32 at this point

        # do any resampling if necessary, returning only self.chans data
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs)
            #print('resample took %.3f sec' % (time.time()-tresample))
        else: # don't resample, just cut out self.chans data, if necessary
            if self.kind == 'highpass':
                if range(nADchans) != list(self.chans):
                    # some chans are disabled. This is kind of a hack, but works because
                    # because ADchans map to probe chans 1 to 1, and both start from 0
                    dataxs = dataxs[self.chans]
            else: # self.kind == 'lowpass'
                # self.chans is a sorted subset of layout.chans, layout.chans are not sorted:
                chanis = [ int(np.where(chan == self.layout.chans)[0]) for chan in self.chans ]
                dataxs = dataxs[chanis]
        # now trim down to just the requested time range
        lo, hi = tsxs.searchsorted([key.start, key.stop])
        data = dataxs[:, lo:hi]
        ts = tsxs[lo:hi]

        data = np.int16(data) # should be safe to convert back down to int16 now
        return WaveForm(data=data, ts=ts, chans=self.chans)

    def resample(self, rawdata, rawts):
        """Return potentially sample-and-hold corrected and Nyquist interpolated
        data and timepoints. See Blanche & Swindale, 2006"""
        #print('sampfreq, rawsampfreq, shcorrect = (%r, %r, %r)' %
        #      (self.sampfreq, self.rawsampfreq, self.shcorrect))
        rawtres = self.rawtres # us
        tres = self.tres # us
        resamplex = intround(self.sampfreq / self.rawsampfreq) # resample factor: n output resampled points per input raw point
        assert resamplex >= 1, 'no decimation allowed'
        N = KERNELSIZE

        # pretty basic assumption which might change if chans are disabled:
        #assert self.nchans == len(self.chans) == len(ADchans)
        # check if kernels have been generated already
        try:
            self.kernels
        except AttributeError:
            ADchans = self.layout.ADchanlist
            self.kernels = self.get_kernels(ADchans, resamplex, N)

        # convolve the data with each kernel
        nrawts = len(rawts)
        # all the interpolated points have to fit in between the existing raw
        # points, so there's nrawts - 1 of each of the interpolated points:
        #nt = nrawts + (resamplex-1) * (nrawts - 1)
        # the above can be simplified to:
        nt = nrawts*resamplex - (resamplex - 1)
        tstart = rawts[0]
        ts = np.arange(tstart, tstart+tres*nt, tres) # generate interpolated timepoints
        #print 'len(ts) is %r' % len(ts)
        assert len(ts) == nt
        # resampled data, leave as int32 for convolution, then convert to int16:
        data = np.empty((self.nchans, nt), dtype=np.int32)
        #print 'data.shape = %r' % (data.shape,)
        #tconvolve = time.time()
        tconvolvesum = 0
        # assume chans map onto ADchans 1 to 1, ie chan 0 taps off of ADchan 0
        # this way, only the chans that are actually needed are resampled and returned
        for chani, chan in enumerate(self.chans):
            for point, kernel in enumerate(self.kernels[chan]):
                """np.convolve(a, v, mode)
                for mode='same', only the K middle values are returned starting at n = (M-1)/2
                where K = len(a)-1 and M = len(v) - 1 and K >= M
                for mode='valid', you get the middle len(a) - len(v) + 1 number of values"""
                #tconvolveonce = time.time()
                row = np.convolve(rawdata[chan], kernel, mode='same')
                #tconvolvesum += (time.time()-tconvolveonce)
                #print 'len(rawdata[ADchani]) = %r' % len(rawdata[ADchani])
                #print 'len(kernel) = %r' % len(kernel)
                #print 'len(row): %r' % len(row)
                # interleave by assigning from point to end in steps of resamplex
                # index to start filling data from for this kernel's points:
                ti0 = (resamplex - point) % resamplex
                # index of first data point to use from convolution result 'row':
                rowti0 = int(point > 0)
                # discard the first data point from interpolant's convolutions, but not for
                # raw data's convolutions, since interpolated values have to be bounded on both
                # sides by raw values?
                data[chani, ti0::resamplex] = row[rowti0:]
        #print('convolve loop took %.3f sec' % (time.time()-tconvolve))
        #print('convolve calls took %.3f sec total' % (tconvolvesum))
        #tundoscaling = time.time()
        data >>= 16 # undo kernel scaling, shift 16 bits right in place, same as //= 2**16
        #print('undo kernel scaling took %.3f sec total' % (time.time()-tundoscaling))
        return data, ts

    def get_kernels(self, ADchans, resamplex, N):
        """Generate a different set of kernels for each ADchan to correct each ADchan's
        s+h delay.

        TODO: when resamplex > 1 and shcorrect == False, you only need resamplex - 1 kernels.
        You don't need a kernel for the original raw data points. Those won't be shifted,
        so you can just interleave appropriately.

        TODO: take DIN channel into account, might need to shift all highpass ADchans
        by 1us, see line 2412 in SurfBawdMain.pas. I think the layout.sh_delay_offset field
        may tell you if and by how much you should take this into account

        WARNING! TODO: not sure if say ADchan 4 will always have a delay of 4us, or only if
        it's preceded by AD chans 0, 1, 2 and 3 in the channel gain list - I suspect the latter
        is the case, but right now I'm coding the former. Note that there's a
        srff.layout.sh_delay_offset field that describes the sh delay for first chan of probe.
        Should probably take this into account, although it doesn't affect relative delays
        between chans, I think. I think it's usually 1us.
        """
        i = ADchans % NCHANSPERBOARD # ordinal position of each chan in the hold queue
        if self.shcorrect:
            dis = 1 * i # per channel delays, us
            # TODO: stop hard coding 1us delay per ordinal position
        else:
            dis = 0 * i
        ds = dis / self.rawtres # normalized per channel delays
        wh = hamming # window function
        h = np.sinc # sin(pi*t) / pi*t
        kernels = [] # list of list of kernels, indexed by [ADchani][resample point]
        for ADchan in ADchans:
            d = ds[ADchan] # delay for this chan
            kernelrow = []
            for point in xrange(resamplex): # iterate over resampled points per raw point
                t0 = point/resamplex # some fraction of 1
                tstart = -N/2 - t0 - d
                tend = tstart + (N+1)
                # kernel sample timepoints, all of length N+1, float32s to match voltage
                # data type
                t = np.arange(tstart, tend, 1, dtype=np.float32)
                kernel = wh(t, N) * h(t) # windowed sinc, sums to 1.0, max val is 1.0
                # rescale to get values up to 2**16, convert to int32
                kernel = np.int32(np.round(kernel * 2**16))
                kernelrow.append(kernel)
            kernels.append(kernelrow)
        return kernels


class TSFStream(Stream):
    """Stream based on wavedata from a .tsf file instead of a .srf file"""
    def __init__(self, fname, wavedata, siteloc, rawsampfreq, masterclockfreq,
                 extgain, intgain, sampfreq=None, shcorrect=None):
        self._fname = fname
        self.wavedata = wavedata
        nchans, nt = wavedata.shape
        self.chans = np.arange(nchans) # this sets self.nchans
        self.nt = nt
        self.nADchans = self.nchans
        self.ADchans = np.arange(self.nADchans)
        self.layout = EmptyClass()
        self.layout.ADchanlist = self.ADchans # for the sake of self.resample()
        probematch = False
        for probetype in [uMap54_1a, uMap54_1b, uMap54_1c, uMap54_2a, uMap54_2b]:
            probe = probetype()
            if (probe.siteloc_arr() == siteloc).all():
                self.probe = probe
                probematch = True
                break
        if not probematch:
            raise ValueError("siteloc in %s doesn't match known probe type" % fname)
        self.rawsampfreq = rawsampfreq
        self.rawtres = intround(1 / self.rawsampfreq * 1e6) # us
        self.masterclockfreq = masterclockfreq
        self.extgain = extgain
        self.intgain = intgain
        self.converter = Converter(intgain, extgain)
        self.sampfreq = sampfreq or DEFHIGHPASSSAMPFREQ # desired sampling frequency
        self.shcorrect = shcorrect or DEFHIGHPASSSHCORRECT
        self.t0 = 0 # us
        self.t1 = nt * self.rawtres
        self.tranges = np.int64([[self.t0, self.t1]])

    def open(self):
        pass

    def is_open(self):
        return True

    def close(self):
        pass

    def get_fname(self):
        return self._fname

    fname = property(get_fname)

    def get_srcfnameroot(self):
        return lrstrip(self.fname, '../', '.tsf')

    srcfnameroot = property(get_srcfnameroot)

    def get_datetime(self):
        """.tsf files don't currently have a datetime stamp, return Unix epoch instead"""
        return UNIXEPOCH

    datetime = property(get_datetime)
    
    def __getstate__(self):
        """Get object state for pickling"""
        # copy it cuz we'll be making changes, this is fast because it's just a shallow copy
        d = self.__dict__.copy()
        try: del d['wavedata'] # takes up way too much space
        except KeyError: pass
        return d

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us. Returns the corresponding WaveForm
        object, which has as its attribs the 2D multichannel waveform array as well
        as the timepoints, potentially spanning multiple ContinuousRecords. Lots of
        unavoidable code duplication from Stream.__getitem__"""
        if key.step not in [None, 1]:
            raise ValueError('unsupported slice step size: %s' % key.step)

        nADchans = self.nADchans
        rawtres = self.rawtres
        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at
            # key.start and key.stop
            xs = KERNELSIZE * rawtres
        else:
            xs = 0
        # get a slightly greater range of raw data (with xs) than might be needed:
        t0xsi = (key.start - xs) // rawtres # round down to nearest mult of rawtres
        t1xsi = ((key.stop + xs) // rawtres) + 1 # round up to nearest mult of rawtres
        # stay within stream limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, self.t0 // rawtres)
        t1xsi = min(t1xsi, self.t1 // rawtres)
        # convert back to us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        tsxs = np.arange(t0xs, t1xs, rawtres)
        ntxs = len(tsxs)

        # init data as int32 so we have bitwidth to rescale and zero, then convert to int16
        dataxs = np.int32(self.wavedata[:, t0xsi:t1xsi])

        # bitshift left to scale 12 bit values to use full 16 bit dynamic range, same as
        # * 2**(16-12) == 16. This provides more fidelity for interpolation, reduces uV per
        # AD to about 0.02
        dataxs <<= 4 # data is still int32 at this point

        # do any resampling if necessary, returning only self.chans data
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs)
            #print('resample took %.3f sec' % (time.time()-tresample))
        else: # don't resample, just cut out self.chans data, if necessary
            if range(nADchans) != list(self.chans):
                # some chans are disabled. This is kind of a hack, but works because
                # because ADchans map to probe chans 1 to 1, and both start from 0
                dataxs = dataxs[self.chans]
        # now trim down to just the requested time range
        lo, hi = tsxs.searchsorted([key.start, key.stop])
        data = dataxs[:, lo:hi]
        ts = tsxs[lo:hi]

        data = np.int16(data) # should be safe to convert back down to int16 now
        return WaveForm(data=data, ts=ts, chans=self.chans)

            
class SpykeToolWindow(QtGui.QMainWindow):
    """Base class for all of spyke's tool windows"""
    def __init__(self, parent, flags=Qt.Tool):
        QtGui.QMainWindow.__init__(self, parent, flags)
        self.maximized = False

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F11:
            self.toggleMaximized()
        else:
            QtGui.QMainWindow.keyPressEvent(self, event) # pass it on

    def mouseDoubleClickEvent(self, event):
        """Doesn't catch window titlebar doubleclicks for some reason (window manager
        catches them?). Have to doubleclick on a part of the window with no widgets in it"""
        self.toggleMaximized()

    def closeEvent(self, event):
        # remove 'Window' from class name
        windowtype = type(self).__name__.replace('Window', '')
        self.parent().HideWindow(windowtype)

    def toggleMaximized(self):
        if not self.maximized:
            self.normalPos, self.normalSize = self.pos(), self.size()
            dw = QtGui.QDesktopWidget()
            rect = dw.availableGeometry(self)
            self.setGeometry(rect)
            self.maximized = True
        else: # restore
            self.resize(self.normalSize)
            self.move(self.normalPos)
            self.maximized = False


class SpykeListView(QtGui.QListView):
    def __init__(self, parent):
        QtGui.QListView.__init__(self, parent)
        self.sortwin = parent
        #self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QtGui.QListView.ExtendedSelection)
        self.setLayoutMode(QtGui.QListView.Batched) # prevents lockup during huge layout ops
        self.setResizeMode(QtGui.QListView.Adjust) # recalculates layout on resize
        self.setUniformItemSizes(True) # speeds up listview
        self.setFlow(QtGui.QListView.LeftToRight) # default is TopToBottom
        self.setWrapping(True)
        self.setBatchSize(300)
        #self.setViewMode(QtGui.QListView.IconMode)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        ctrldown = bool(Qt.ControlModifier & modifiers)
        ctrlup = not ctrldown
        if (key in [Qt.Key_M, Qt.Key_G, Qt.Key_Minus, Qt.Key_Slash, Qt.Key_Backslash,
                    Qt.Key_NumberSign, Qt.Key_C, Qt.Key_V, Qt.Key_R, Qt.Key_B,
                    Qt.Key_Comma, Qt.Key_Period, Qt.Key_H]
            or ctrlup and key == Qt.Key_Space):
            event.ignore() # pass it on up to the parent
        else:
            QtGui.QListView.keyPressEvent(self, event) # handle it as usual

    def selectionChanged(self, selected, deselected, prefix=None):
        """Plot neurons or spikes on list item selection"""
        QtGui.QListView.selectionChanged(self, selected, deselected)
        panel = self.sortwin.panel
        addis = [ i.data().toInt()[0] for i in selected.indexes() ]
        remis = [ i.data().toInt()[0] for i in deselected.indexes() ]
        panel.removeItems([ prefix+str(i) for i in remis ])
        panel.addItems([ prefix+str(i) for i in addis ])
        #print("done selchanged, %r, addis=%r, remis=%r" % (prefix, addis, remis))

    def updateAll(self):
        self.model().updateAll()

    def get_nrows(self):
        return self.model().rowCount()

    nrows = property(get_nrows)

    def selectRows(self, rows, on=True, scrollTo=True):
        """Row selection in listview is complex. This makes it simpler"""
        ## TODO: There's a bug here, where if you select the last two neurons in nlist,
        ## (perhaps these last two need to be near a list edge), merge them, and then
        ## undo,then merge again (instead of just redoing), then undo again, they're
        ## both selected, but only the first is replotted because the selchanged event
        ## is only passed the first of the two as being newly selected. If however,
        ## before remerging, you clear the selection, or select something else, and then
        ## go back and select those same two neurons and merge, and undo, it works fine,
        ## and the selchanged event gets both items as newly selected. Seems like a Qt
        ## bug, or at least some very subtle timing problem of some kind. This might have
        ## something to do with reflow when changing list contents, but even resetting
        ## listview behaviour to default doesn't make this go away. Also, seems to happen
        ## for selection of one index at a time, and for doing it all in one go with a
        ## QItemSelection.
        
        rows = toiter(rows)
        m = self.model()
        sm = self.selectionModel()
        if on:
            flag = sm.Select
        else:
            flag = sm.Deselect
        #print('start select=%r loop for rows %r' % (on, rows))
        '''
        # unnecessarily emits nrows selectionChanged signals, causes slow
        # plotting in mpl commit 50fc548465b1525255bc2d9f66a6c7c95fd38a75 (pre
        # 1.0) and later:
        [ sm.select(m.index(row), flag) for row in rows ]
        '''
        # emits single selectionChanged signal, more efficient, but causes a bit of
        # flickering, or at least used to in Qt 4.7.0:
        sel = QtGui.QItemSelection()
        for row in rows:
            index = m.index(row)
            #print('row: %r, index: %r' % (row, index))
            sel.select(index, index) # topleft to bottomright
        #print('sel has indexes, rows, cols, data:')
        #for index in sel.indexes():
        #    print(index, index.row(), index.column(), index.data())
        sm.select(sel, flag)
        #print('end select loop')
        
        if scrollTo and on and len(rows) > 0: # scroll to last row that was just selected
            self.scrollTo(m.index(rows[-1]))

    def selectedRows(self):
        """Return list of selected rows"""
        return [ i.row() for i in self.selectedIndexes() ]

    def rowSelected(self, row):
        """Simple way to check if a row is selected"""
        return self.model().index(row) in self.selectedIndexes()

    def get_nrowsSelected(self):
        return len(self.selectedIndexes())

    nrowsSelected = property(get_nrowsSelected)

    def selectRandom(self, start, stop, nsamples):
        """Select random sample of rows"""
        start = max(0, start)
        if stop == -1:
            stop = self.nrows
        stop = min(self.nrows, stop)
        nrows = stop - start
        nsamples = min(nsamples, nrows)
        rows = random.sample(xrange(start, stop), nsamples)
        self.selectRows(rows, scrollTo=False)


class NList(SpykeListView):
    """Neuron list view"""
    def __init__(self, parent):
        SpykeListView.__init__(self, parent)
        self.setModel(NListModel(parent))
        self.setItemDelegate(NListDelegate(parent))
        self.connect(self, QtCore.SIGNAL("activated(QModelIndex)"),
                     self.on_actionItem_activated)

    def selectionChanged(self, selected, deselected):
        SpykeListView.selectionChanged(self, selected, deselected, prefix='n')
        selnids = [ i.data().toInt()[0] for i in self.selectedIndexes() ]
        #if 1 <= len(selnids) <= 3: # populate nslist if exactly 1, 2 or 3 neurons selected
        self.sortwin.nslist.neurons = [ self.sortwin.sort.neurons[nid] for nid in selnids ]
        #else:
        #    self.sortwin.nslist.neurons = []

    def on_actionItem_activated(self, index):
        sw = self.sortwin
        sw.parent().ui.plotButton.click()


class NSList(SpykeListView):
    """Spike list view"""
    def __init__(self, parent):
        SpykeListView.__init__(self, parent)
        self.setModel(NSListModel(parent))
        self.connect(self, QtCore.SIGNAL("activated(QModelIndex)"),
                     self.on_actionItem_activated)

    def selectionChanged(self, selected, deselected):
        SpykeListView.selectionChanged(self, selected, deselected, prefix='s')

    def on_actionItem_activated(self, index):
        sw = self.sortwin
        if sw.sort.stream.is_open():
            sid = self.sids[index.row()]
            spike = sw.sort.spikes[sid]
            sw.parent().seek(spike['t'])
        else:
            sw.parent().ui.plotButton.click()

    def get_neurons(self):
        return self.model().neurons

    def set_neurons(self, neurons):
        """Every time neurons are set, clear any existing selection and update data model"""
        self.clearSelection() # remove any plotted sids, at least for now
        self.model().neurons = neurons

    neurons = property(get_neurons, set_neurons)

    def get_sids(self):
        return self.model().sids

    sids = property(get_sids)

    def selectRandom(self, nsamples):
        """Select up to nsamples random rows per neuron"""
        if self.model().sliding == True:
            self.neurons = self.neurons # trigger NSListModel.set_neurons() call
            self.model().sliding = False
        for neuron in self.neurons:
            allrows = self.sids.searchsorted(neuron.sids)
            nsamples = min(nsamples, len(allrows))
            rows = random.sample(allrows, nsamples)
            self.selectRows(rows, scrollTo=False)


class USList(SpykeListView):
    """Unsorted spike list view"""
    def __init__(self, parent):
        SpykeListView.__init__(self, parent)
        self.setModel(USListModel(parent))
        self.connect(self, QtCore.SIGNAL("activated(QModelIndex)"),
                     self.on_actionItem_activated)

    def selectionChanged(self, selected, deselected):
        SpykeListView.selectionChanged(self, selected, deselected, prefix='s')

    def on_actionItem_activated(self, index):
        sw = self.sortwin
        if sw.sort.stream.is_open():
            sid = sw.sort.usids[index.row()]
            spike = sw.sort.spikes[sid]
            sw.parent().seek(spike['t'])
        else:
            sw.parent().ui.plotButton.click()

    def selectRandom(self, nsamples):
        """Select up to nsamples random rows"""
        SpykeListView.selectRandom(self, 0, -1, nsamples)


class SpykeAbstractListModel(QtCore.QAbstractListModel):
    def __init__(self, parent):
        QtCore.QAbstractListModel.__init__(self, parent)
        self.sortwin = parent

    def updateAll(self):
        """Emit dataChanged signal so that view updates itself immediately.
        Hard to believe this doesn't already exist in some form"""
        i0 = self.createIndex(0, 0) # row, col
        i1 = self.createIndex(self.rowCount()-1, 0) # seems this isn't necessary
        #self.dataChanged.emit(i0, i0) # seems to refresh all, though should only refresh 1st row
        self.dataChanged.emit(i0, i1) # refresh all


class NListModel(SpykeAbstractListModel):
    """Model for neuron list view"""
    def rowCount(self, parent=None):
        try:
            # update nlist tooltip before returning, only +ve nids count as neurons:
            nneurons = (np.asarray(self.sortwin.sort.norder) > 0).sum()
            self.sortwin.nlist.setToolTip("Neuron list\n%d neurons" % nneurons)
            return len(self.sortwin.sort.norder)
        except AttributeError: # sort doesn't exist
            return 0

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            neurons = self.sortwin.sort.neurons
            norder = self.sortwin.sort.norder
            try:
                nid = norder[index.row()]
            except IndexError:
                print('WARNING: tried to index non-existent row %d' % index.row())
            #print('.data(): row=%d, val=%d' % (index.row(), nid))
            if role == Qt.DisplayRole:
                return nid # no need to use QVariant() apparently
            elif role == Qt.ToolTipRole:
                neuron = neurons[nid]
                pos = neuron.cluster.pos
                return ('nid: %d\n' % nid +
                        '%d spikes\n' % neuron.nspikes +
                        't: %d us\n' % pos['t'] +
                        'x0: %.4g um\n' % pos['x0'] +
                        'y0: %.4g um\n' % pos['y0'] +
                        'Vpp: %.4g uV\n' % pos['Vpp'] +
                        'sx: %.4g um\n' % pos['sx'] +
                        'dphase: %.4g us' % pos['dphase'])
            # this stuff is handled in NListDelegate:
            '''
            elif role == Qt.ForegroundRole:
                if nid in self.sortwin.sort.get_good():
                    return QtGui.QBrush(QtGui.QColor(255, 255, 255))
            elif role == Qt.BackgroundRole:
                if nid in self.sortwin.sort.get_good():
                    return QtGui.QBrush(QtGui.QColor(0, 128, 0))
            '''
class SListModel(SpykeAbstractListModel):
    """Base model for spike list models"""
    def spiketooltip(self, spike):
        return ('sid: %d\n' % spike['id'] +
                'nid: %d\n' % spike['nid'] +
                't: %d us\n' % spike['t'] +
                'x0: %.4g um\n' % spike['x0'] +
                'y0: %.4g um\n' % spike['y0'] +
                'Vpp: %.4g uV\n' % spike['Vpp'] +
                'sx: %.4g um\n' % spike['sx'] +
                'dphase: %.4g us' % spike['dphase'])


class NSListModel(SListModel):
    """Model for neuron spikes list view"""
    def __init__(self, parent):
        SpykeAbstractListModel.__init__(self, parent)
        self._neurons = []
        self.nspikes = 0

    def get_neurons(self):
        return self._neurons

    def set_neurons(self, neurons):
        self._neurons = neurons
        if neurons:
            self.sids = np.concatenate([ neuron.sids for neuron in neurons ])
            self.sids.sort() # keep them sorted
            self.sortwin.slider.setEnabled(True)
        else:
            self.sids = np.empty(0, dtype=np.int32)
            self.sortwin.slider.setEnabled(False)
        self.nspikes = len(self.sids)
        # triggers new calls to rowCount() and data(), and critically, clears selection
        # before moving slider to pos 0, which triggers slider.valueChanged:
        self.reset()
        self.sortwin.slider.setValue(0) # reset position to 0
        self.sortwin.update_slider() # update limits and step sizes
        self.sliding = False

    neurons = property(get_neurons, set_neurons)

    def rowCount(self, parent=None):
        # update nslist tooltip before returning:
        self.sortwin.nslist.setToolTip("Sorted spike list\n%d spikes" % self.nspikes)
        return self.nspikes

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in [Qt.DisplayRole, Qt.ToolTipRole]:
            sid = int(self.sids[index.row()])
            if role == Qt.DisplayRole:
                return sid
            elif role == Qt.ToolTipRole:
                spike = self.sortwin.sort.spikes[sid]
                return self.spiketooltip(spike)


class USListModel(SListModel):
    """Model for unsorted spike list view"""
    def rowCount(self, parent=None):
        try:
            nspikes = len(self.sortwin.sort.usids)
            # update uslist tooltip before returning:
            self.sortwin.uslist.setToolTip("Unsorted spike list\n%d spikes" % nspikes)
            return nspikes
        except AttributeError: # sort doesn't exist
            return 0

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role in [Qt.DisplayRole, Qt.ToolTipRole]:
            sid = int(self.sortwin.sort.usids[index.row()])
            if role == Qt.DisplayRole:
                return sid
            elif role == Qt.ToolTipRole:
                spike = self.sortwin.sort.spikes[sid]
                return self.spiketooltip(spike)


class NListDelegate(QtGui.QStyledItemDelegate):
    """Delegate for neuron list view, modifies appearance of items"""
    def __init__(self, parent):
        QtGui.QStyledItemDelegate.__init__(self, parent)
        self.sortwin = parent
        palette = QtGui.QApplication.palette()
        self.selectedgoodbrush = QtGui.QBrush(QtGui.QColor(0, 0, 255)) # blue
        self.unselectedgoodbrush = QtGui.QBrush(QtGui.QColor(0, 128, 0)) # mid green
        self.selectedbrush = palette.highlight()
        self.unselectedbrush = palette.base()
        self.selectedgoodpen = QtGui.QPen(Qt.white)
        self.unselectedgoodpen = QtGui.QPen(Qt.white)
        self.selectedpen = QtGui.QPen(palette.highlightedText().color())
        self.unselectedpen = QtGui.QPen(palette.text().color())
        self.focusedpen = QtGui.QPen(Qt.gray, 0, Qt.DashLine)
        self.focusedpen.setDashPattern([1, 1])
        self.focusedpen.setCapStyle(Qt.FlatCap)

    def paint(self, painter, option, index):
        """Change background colour for nids designated as "good"""
        model = index.model()
        nid = model.data(index) # should come out as an int
        good = nid in self.sortwin.sort.get_good()
        # don't care whether self is active or inactive, only care about
        # selection, "good", and focused states
        selected = option.state & QtGui.QStyle.State_Selected
        focused = option.state & QtGui.QStyle.State_HasFocus
        painter.save()
        # paint background:
        painter.setPen(QtGui.QPen(Qt.NoPen))
        if selected:
            if good:
                painter.setBrush(self.selectedgoodbrush)
            else: # use default selection brush
                painter.setBrush(self.selectedbrush)
        else: # unselected
            if good:
                painter.setBrush(self.unselectedgoodbrush)
            else: # use default background brush
                painter.setBrush(self.unselectedbrush)
        painter.drawRect(option.rect)
        # paint focus rect:
        if focused:
            rect = copy(option.rect)
            painter.setBrush(Qt.NoBrush) # no need to draw bg again
            painter.setPen(self.focusedpen)
            rect.adjust(0, 0, -1, -1) # make space for outline
            painter.drawRect(rect)
        # paint foreground:
        value = index.data(Qt.DisplayRole)
        if selected:
            if good:
                painter.setPen(self.selectedgoodpen)
            else: # use default selection pen
                painter.setPen(self.selectedpen)
        else: # unselected
            if good:
                painter.setPen(self.unselectedgoodpen)
            else: # use default background pen
                painter.setPen(self.unselectedpen)
        text = value.toString()
        painter.drawText(option.rect, Qt.AlignCenter, text)
        painter.restore()


class ClusterTabSpinBox(QtGui.QSpinBox):
    """Intercept CTRL+Z key event for cluster undo instead of spinbox edit undo"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.topLevelWidget().on_actionUndo_triggered()
        else:
            QtGui.QSpinBox.keyPressEvent(self, event) # handle it as usual


class ClusterTabDoubleSpinBox(QtGui.QDoubleSpinBox):
    """Intercept CTRL+Z key event for cluster undo instead of spinbox edit undo"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.topLevelWidget().on_actionUndo_triggered()
        else:
            QtGui.QDoubleSpinBox.keyPressEvent(self, event) # handle it as usual


class ClusteringGroupBox(QtGui.QGroupBox):
    """Make ENTER key event activate the cluster button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().ui.clusterButton.click()
        else:
            QtGui.QGroupBox.keyPressEvent(self, event) # handle it as usual


class PlottingGroupBox(QtGui.QGroupBox):
    """Make ENTER key event activate the plot button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().ui.plotButton.click()
        else:
            QtGui.QGroupBox.keyPressEvent(self, event) # handle it as usual


class XCorrsGroupBox(QtGui.QGroupBox):
    """Make ENTER key event activate the correlograms plot button"""
    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.topLevelWidget().ui.plotXcorrsButton.click()
        else:
            QtGui.QGroupBox.keyPressEvent(self, event) # handle it as usual


class Stack(list):
    """A list that doesn't allow -ve indices"""
    def __getitem__(self, key):
        if key < 0:
            raise IndexError('stack index %d out of range' % key)
        return list.__getitem__(self, key)


class ClusterChange(object):
    """Stores info for undoing/redoing a change to any set of clusters"""
    def __init__(self, sids, spikes, message):
        self.sids = sids
        self.spikes = spikes
        self.message = message

    def __repr__(self):
        return self.message

    def save_old(self, oldclusters, oldnorder):
        self.oldnids = self.spikes['nid'][self.sids] # this seems to create a copy
        self.oldunids = [ c.id for c in oldclusters ]
        self.oldposs = [ c.pos.copy() for c in oldclusters ]
        self.oldnormposs = [ c.normpos.copy() for c in oldclusters ]
        self.oldnorder = copy(oldnorder)

    def save_new(self, newclusters, newnorder):
        self.newnids = self.spikes['nid'][self.sids] # this seems to create a copy
        self.newunids = [ c.id for c in newclusters ]
        self.newposs = [ c.pos.copy() for c in newclusters ]
        self.newnormposs = [ c.normpos.copy() for c in newclusters ]
        self.newnorder = copy(newnorder)

'''
def save(fname, arr):
    """Taken from np.lib.npyio.save and np.lib.format.write_array to write
    a big array to a .npy file in reasonably sized chunks at a time so
    as not to trigger msvc >= 2**31 byte fwrite() call problems which happen
    even in win64. See http://projects.scipy.org/numpy/ticket/1660 and
    http://bugs.python.org/issue9015.
    Rendered unnecessary by Christoph Gohlke's numpy patch!

    # test code:
    fourgbplus = 2**32 + 2**16
    testbytes = np.arange(8, dtype=np.int8).reshape(1, -1) # make it 2D
    arr = testbytes.repeat(fourgbplus // testbytes.nbytes, axis=0)
    save('test', arr)
    np.save('test_np', arr) # compare the two files in hex editor
    # files should have MD5 (128 bit) hash: 99BFB5B8E2FA2DB93092C5454AAF9388
    """
    if not arr.flags.c_contiguous:
        if not arr.flags.f_contiguous:
            raise ValueError("array is not contiguous")
        raise NotImplementedError('saving f-contig arrays not tested')
        arr = arr.T # transpose to convert from f-contig to c-contig

    if not fname.endswith('.npy'):
        fname = fname + '.npy'
    f = open(fname, 'wb')

    version = 1, 0 # .npy format version
    format = np.lib.format
    f.write(format.magic(*version))
    format.write_array_header_1_0(f, format.header_data_from_array_1_0(arr))

    nchunks = int(np.ceil(arr.nbytes / MAXNBYTESTOFILE))
    arrravel = arr.ravel()
    for chunki in range(nchunks):
        lo, hi = MAXNBYTESTOFILE*chunki, MAXNBYTESTOFILE*(chunki+1)
        arrravel[lo:hi].tofile(f) # these are contiguous views, not copies
    f.close()
'''
def savez(file, *args, **kwargs):
    """Save several arrays into a single, possibly compressed, binary file.
    Taken from numpy.io.lib.savez. Add a compress=False|True keyword, and
    allow for any file extension. For full docs, see numpy.savez()"""

    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile
    import tempfile
    import numpy.lib.format as format

    compress = kwargs.pop('compress', False) # defaults to False
    assert type(compress) == bool
    namedict = kwargs
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError, "Cannot use un-named variables and keyword %s" % key
        namedict[key] = val

    compression = zipfile.ZIP_STORED # no compression
    if compress:
        compression = zipfile.ZIP_DEFLATED # compression
    zip = zipfile.ZipFile(file, mode="w", compression=compression)
    # place to write temporary .npy files before storing them in the zip
    direc = tempfile.gettempdir()
    todel = []
    for key, val in namedict.iteritems():
        fname = key + '.npy'
        filename = os.path.join(direc, fname)
        todel.append(filename)
        fid = open(filename,'wb')
        format.write_array(fid, np.asanyarray(val))
        fid.close()
        zip.write(filename, arcname=fname)
    zip.close()
    for name in todel:
        os.remove(name)

def get_sha1(fname, blocksize=2**20):
    """Gets the sha1 hash of fname (with full path)"""
    m = hashlib.sha1()
    with file(fname, 'rb') as f:
        # continually update hash until EOF
        while True:
            block = f.read(blocksize)
            if not block:
                break
            m.update(block)
    return m.hexdigest()

def intround(n):
    """Round to the nearest integer, return an integer. Works on arrays,
    saves on parentheses, nothing more"""
    if iterable(n): # it's a sequence, return as an int64 array
        return np.int64(np.round(n))
    else: # it's a scalar, return as normal Python int
        return int(round(n))

def iterable(x):
    """Check if the input is iterable, stolen from numpy.iterable()"""
    try:
        iter(x)
        return True
    except TypeError:
        return False

def toiter(x):
    """Convert to iterable. If input is iterable, returns it. Otherwise returns it in a list.
    Useful when you want to iterate over something (like in a for loop),
    and you don't want to have to do type checking or handle exceptions
    when it isn't a sequence"""
    if iterable(x):
        return x
    else:
        return [x]

def tocontig(x):
    """Return C contiguous copy of array x if it isn't C contiguous already"""
    if not x.flags.c_contiguous:
        x = x.copy()
    return x
'''
# use np.vstack instead:
def cvec(x):
    """Return x as a column vector. x must be a scalar or a vector"""
    x = np.asarray(x)
    assert x.squeeze().ndim in [0, 1]
    try:
        nrows = len(x)
    except TypeError: # x is scalar?
        nrows = 1
    x.shape = (nrows, 1)
    return x
'''
def is_empty(x):
    """Check if sequence is empty. There really should be a np.is_empty function"""
    print("WARNING: not thoroughly tested!!!")
    x = np.asarray(x)
    if np.prod(x.shape) == 0:
        return True
    else:
        return False

def cut(ts, trange):
    """Returns timestamps, where tstart <= timestamps <= tend
    Copied and modified from neuropy rev 149"""
    lo, hi = argcut(ts, trange)
    return ts[lo:hi] # slice it

def argcut(ts, trange):
    """Returns timestamp slice indices, where tstart <= timestamps <= tend
    Copied and modified from neuropy rev 149"""
    tstart, tend = trange[0], trange[1]
    '''
    # this is what we're trying to do:
    return ts[ (ts >= tstart) & (ts <= tend) ]
    ts.searchsorted([tstart, tend]) method does it faster, because it assumes ts are ordered.
    It returns an index where the values would fit in ts. The index is such that
    ts[index-1] < value <= ts[index]. In this formula ts[ts.size]=inf and ts[-1]= -inf
    '''
    lo, hi = ts.searchsorted([tstart, tend]) # returns indices where tstart and tend would fit in ts
    # can probably avoid all this end inclusion code by using the 'side' kwarg, not sure if I want end inclusion anyway
    '''
    if tend == ts[min(hi, len(ts)-1)]: # if tend matches a timestamp (protect from going out of index bounds when checking)
        hi += 1 # inc to include a timestamp if it happens to exactly equal tend. This gives us end inclusion
        hi = min(hi, len(ts)) # limit hi to max slice index (==max value index + 1)
    '''
    return lo, hi

def eucd(coords):
    """Generates Euclidean distance matrix from a
    sequence of n dimensional coordinates. Nice and fast.
    Written by Willi Richert
    Taken from:
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/498246
    on 2006/11/11
    """
    coords = np.asarray(coords)
    n, m = coords.shape
    delta = np.zeros((n, n), dtype=np.float64)
    for d in xrange(m):
        data = coords[:, d]
        delta += (data - data[:, np.newaxis]) ** 2
    return np.sqrt(delta)

def revcmp(x, y):
    """Does the reverse of cmp():
    Return negative if y<x, zero if y==x, positive if y>x"""
    return cmp(y, x)


class Gaussian(object):
    """Gaussian function, works with ndarray inputs"""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        """Called when self is called as a f'n.
        Don't bother normalizing by 1/(sigma*np.sqrt(2*pi)),
        don't care about normalizing the integral,
        just want to make sure that f(0) == 1"""
        return np.exp( -(x-self.mu)**2 / (2*self.sigma**2) )

    def __getitem__(self, x):
        """Called when self is indexed into"""
        return self(x)


def g(x0, sx, x):
    """1-D Gaussian"""
    return np.exp( -(x-x0)**2 / (2*sx**2) )

def g2(x0, y0, sx, sy, x, y):
    """2-D Gaussian"""
    arg = -(x-x0)**2 / (2*sx**2) - (y-y0)**2 / (2*sy**2)
    return np.exp(arg)

def g3(x0, y0, z0, sx, sy, sz, x, y, z):
    """3-D Gaussian"""
    return np.exp( -(x-x0)**2 / (2*sx**2) - (y-y0)**2 / (2*sy**2) - (z-z0)**2 / (2*sz**2) )

def cauchy(x0, gx, x):
    """1-D Cauchy. See http://en.wikipedia.org/wiki/Cauchy_distribution"""
    #return INVPI * gx/((x-x0)**2+gx**2)
    gx2 = gx * gx
    return gx2 / ((x-x0)**2 + gx2)

def cauchy2(x0, y0, gx, gy, x, y):
    """2-D Cauchy"""
    #return INVPI * gx/((x-x0)**2+gx**2) * gy/((y-y0)**2+gy**2)
    return (gx*gy)**2 / ((x-x0)**2 + gx**2) / ((y-y0)**2 + gy**2)

def Vf(Im, x0, y0, z0, sx, sy, sz, x, y, z):
    """1/r voltage decay function in 2D space
    What to do with the singularity so that the leastsq gets a smooth differentiable f'n?"""
    #if np.any(x == x0) and np.any(y == y0) and np.any(z == z0):
    #    raise ValueError, 'V undefined at singularity'
    return Im / (4*pi) / np.sqrt( sx**2 * (x-x0)**2 + sy**2 * (y-y0)**2 + sz**2 * (z-z0)**2)

def dgdmu(mu, sigma, x):
    """Partial of g wrt mu"""
    return (x - mu) / sigma**2 * g(mu, sigma, x)

def dgdsigma(mu, sigma, x):
    """Partial of g wrt sigma"""
    return (x**2 - 2*x*mu + mu**2) / sigma**3 * g(mu, sigma, x)

def dg2dx0(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt x0"""
    return g(y0, sy, y) * dgdmu(x0, sx, x)

def dg2dy0(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt y0"""
    return g(x0, sx, x) * dgdmu(y0, sy, y)

def dg2dsx(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt sx"""
    return g(y0, sy, y) * dgdsigma(x0, sx, x)

def dg2dsy(x0, y0, sx, sy, x, y):
    """Partial of g2 wrt sy"""
    return g(x0, sx, x) * dgdsigma(y0, sy, y)

def RM(theta):
    """Return 2D (2x2) rotation matrix, with theta counterclockwise rotation in radians"""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class Poo(object):
    """Poo function, works with ndarray inputs"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        """Called when self is called as a f'n"""
        return (1+self.a*x) / (self.b+self.c*x**2)

    def __getitem__(self, x):
        """Called when self is indexed into"""
        return self(x)


def hamming(t, N):
    """Return y values of Hamming window at sample points t"""
    #if N == None:
    #    N = (len(t) - 1) / 2
    return 0.54 - 0.46 * np.cos(pi * (2*t + N)/N)

def hex2rgb(hexcolours):
    """Convert colours RGB hex string list into an RGB int array"""
    if type(hexcolours) not in (list, tuple):
        hexcolours = [hexcolours] # enclose in list
    rgb = []
    for s in hexcolours:
        s = s[len(s)-6:len(s)] # get last 6 characters
        r, g, b = s[0:2], s[2:4], s[4:6]
        r, g, b = int(r, base=16), int(g, base=16), int(b, base=16)
        rgb.append((r, g, b))
    return np.uint8(rgb)

def hex2rgba(hexcolours, alpha=255):
    """Convert colours RGB hex string list into an RGBA int array"""
    assert type(alpha) == int and 0 <= alpha <= 255
    rgb = hex2rgb(hexcolours)
    alphas = np.repeat(alpha, len(rgb))
    alphas.shape = -1, 1 # make it 2D column vector
    return np.concatenate([rgb, alphas], axis=1)

def hex2floatrgba(hexcolours, alpha=255):
    """Convert colours RGB hex string list into an RGBA float array"""
    assert type(alpha) == int and 0 <= alpha <= 255
    rgba = hex2rgba(hexcolours, alpha)
    return np.float64(rgba) / 255.

def rgb2hex(rgbcolours):
    """Convert RGB int array into a hex string list"""
    if type(rgbcolours) not in (list, tuple):
        rgbcolours = [rgbcolours] # enclose in list
    hx = []
    for rgb in rgbcolours:
        r, g, b = rgb
        h = hex(r*2**16 + g*2**8 + b)
        h = lrstrip(h, '0x', 'L')
        pad = (6 - len(h)) * '0'
        h = '#' + pad + h
        hx.append(h)
    return hx

c = np.cos
s = np.sin

def Rx(t):
    """Rotation matrix around x axis, theta in radians"""
    return np.matrix([[1, 0,     0   ],
                      [0, c(t), -s(t)],
                      [0, s(t),  c(t)]])

def Ry(t):
    """Rotation matrix around y axis, theta in radians"""
    return np.matrix([[ c(t), 0, s(t)],
                      [ 0,    1, 0   ],
                      [-s(t), 0, c(t)]])

def Rz(t):
    """Rotation matrix around z axis, theta in radians"""
    return np.matrix([[c(t), -s(t), 0],
                      [s(t),  c(t), 0],
                      [0,     0,    1]])

def R(tx, ty, tz):
    """Return full 3D rotation matrix, given thetas in degress.
    Mayavi (tvtk actually) rotates axes in Z, X, Y order, for
    some unknown reason. So, we have to do the same. See:
    tvtk_classes.zip/actor.py:32
    tvtk_classes.zip/prop3d.py:67
    """
    # convert to radians, then take matrix product
    return Rz(tz*pi/180)*Rx(tx*pi/180)*Ry(ty*pi/180)
'''
def normdeg(angle):
    return angle % 360

def win2posixpath(path):
    path = path.replace('\\', '/')
    path = os.path.splitdrive(path)[-1] # remove drive name from start
    return path

def oneD2D(a):
    """Convert 1D array to 2D array. Can do this just as easily using a[None, :]"""
    a = a.squeeze()
    assert a.ndim == 1, "array has more than one non-singleton dimension"
    a.shape = 1, len(a) # make it 2D
    return a

def twoD1D(a):
    """Convert trivially 2D array to 1D array. Seems unnecessary. Just call squeeze()"""
    a = a.squeeze()
    assert a.ndim == 1, "array has more than one non-singleton dimension"
    return a
'''
def is_unique(a):
    """Check whether a has purely unique values in it"""
    u = np.unique(a)
    if len(a) != len(u):
        return False
    else:
        return True

def intersect1d(arrays, assume_unique=False):
    """Find the intersection of any number of 1D arrays.
    Return the sorted, unique values that are in all of the input arrays.
    Adapted from numpy.lib.arraysetops.intersect1d"""
    N = len(arrays)
    if N == 0:
        return np.asarray(arrays)
    arrays = list(arrays) # allow assignment
    if not assume_unique:
        for i, arr in enumerate(arrays):
            arrays[i] = np.unique(arr)
    aux = np.concatenate(arrays) # one long 1D array
    aux.sort() # sorted
    if N == 1:
        return aux
    shift = N-1
    return aux[aux[shift:] == aux[:-shift]]

def rowtake(a, i):
    """For each row in a, return values according to column indices in the
    corresponding row in i. Returned shape == i.shape"""
    assert a.ndim == 2
    assert i.ndim <= 2
    '''
    if i.ndim == 1:
        j = np.arange(a.shape[0])
    else: # i.ndim == 2
        j = np.repeat(np.arange(a.shape[0]), i.shape[1])
        j.shape = i.shape
    j *= a.shape[1]
    j += i
    return a.flat[j]
    '''
    # this is about 3X faster:
    if i.ndim == 1:
        return a[np.arange(a.shape[0]), i]
    else: # i.ndim == 2
        return a[np.arange(a.shape[0])[:, None], i]

def td2usec(td):
    """Convert datetime.timedelta to microseconds"""
    sec = td.total_seconds() # float
    usec = intround(sec * 1000000) # round to nearest us
    return usec

def td2days(td):
    """Convert datetime.timedelta to days"""
    sec = td.total_seconds() # float
    days = sec / 3600 / 24
    return days

def ordered(ts):
    """Check if ts is ordered"""
    # is difference between subsequent entries >= 0?
    return (np.diff(ts) >= 0).all()
    # or, you could compare the array to an explicitly sorted version of itself,
    # and see if they're identical

def concatenate_destroy(arrays):
    """Concatenate list of arrays along 0th axis, destroying them in the process.
    Doesn't duplicate everything in arrays, as does numpy.concatenate. Only
    temporarily duplicates one array at a time, saving memory"""
    if type(arrays) not in (list, tuple):
        raise TypeError('arrays must be list or tuple')
    arrays = list(arrays)
    nrows = 0
    a0 = arrays[0]
    subshape = a0.shape[1::] # dims excluding concatenation dim
    dtype = a0.dtype
    for i, a in enumerate(arrays):
        nrows += len(a)
        if a.shape[1::] != subshape:
            raise TypeError("array %d has subshape %r instead of %r" % (a.shape[1::], subshape))
        if a.dtype != dtype:
            raise TypeError("array %d has type %r instead of %r" % (a.dtype, dtype))
    shape = [nrows] + list(subshape)

    # use np.empty to size up to memory + virtual memory before throwing MemoryError
    a = np.empty(shape, dtype=dtype)
    rowi = 0
    narrays = len(arrays)
    for i in range(narrays):
        array = arrays.pop(0)
        nrows = len(array)
        a[rowi:rowi+nrows] = array # concatenate along 0th axis
        rowi += nrows
    return a

def lst2shrtstr(lst, sigfigs=4, brackets=False):
    """Return string representation of list, replacing any floats with potentially
    shorter representations with fewer sig figs. Any string items in list will be
    simplified by having their quotes removed"""
    gnumfrmt = string.join(['%.', str(sigfigs), 'g'], sep='')
    strlst = []
    for val in lst:
        try:
            strlst.append(gnumfrmt % val)
        except TypeError:
            strlst.append(val) # val isn't a general number
    s = string.join(strlst, sep=', ')
    if brackets:
        s = string.join(['[', s, ']'], sep='')
    return s

def rms(a, axis=None):
    """Return root-mean-squared value of array a along axis"""
    return np.sqrt(np.mean(a**2, axis))

def rmserror(a, b, axis=None):
    """Return root-mean-squared error between arrays a and b"""
    return rms(a - b, axis=axis)

def lstrip(s, strip):
    """What I think str.lstrip should really do"""
    if s.startswith(strip):
        return s[len(strip):] # strip it
    else:
        return s

def rstrip(s, strip):
    """What I think str.rstrip should really do"""
    if s.endswith(strip):
        return s[:-len(strip)] # strip it
    else:
        return s

def strip(s, strip):
    """What I think str.strip should really do"""
    return rstrip(lstrip(s, strip), strip)

def lrstrip(s, lstr, rstr):
    """Strip lstr from start of s and rstr from end of s"""
    return rstrip(lstrip(s, lstr), rstr)

def pad(x, align=8):
    """Pad x with null bytes so it's a multiple of align bytes long"""
    if type(x) == str: # or maybe unicode?
        return padstr(x, align=align)
    elif type(x) == np.ndarray:
        return padarr(x, align=align)
    else:
        raise TypeError('Unhandled type %r in pad()')

def padstr(x, align=8):
    """Pad string x with null bytes so it's a multiple of align bytes long"""
    nbytes = len(x)
    rem = nbytes % align
    npadbytes = align - rem if rem else 0 # nbytes to pad with for 8 byte alignment
    if npadbytes == 0:
        return x
    x = x.encode('ascii') # ensure it's pure ASCII, where each char is 1 byte
    x += '\0' * npadbytes # returns a copy, doesn't modify in place
    assert len(x) % align == 0
    return x

def padarr(x, align=8):
    """Flatten array x and pad with null bytes so it's a multiple of align bytes long"""
    nitems = len(x.ravel())
    nbytes = x.nbytes
    dtypenbytes = x.dtype.itemsize
    rem = nbytes % align
    npadbytes = align - rem if rem else 0 # nbytes to pad with for 8 byte alignment
    if npadbytes == 0:
        return x
    if npadbytes % dtypenbytes != 0:
        raise RuntimeError("Can't pad %d byte array to %d byte alignment" % (dtypenbytes, align))
    npaditems = npadbytes / dtypenbytes
    x = x.ravel().copy() # don't modify in place
    x.resize(nitems + npaditems, refcheck=False) # pads with npaditems zeros, each of length dtypenbytes
    assert x.nbytes % align == 0
    return x

def rollwin(a, width):
    """Return a.nd + 1 dimensional array, where the last dimension contains
    consecutively shifted windows of a of the given width, each shifted by 1
    along the last dimension of a. This allows for calculating rolling stats,
    as well as searching for the existence and position of subarrays in a
    larger array, all without having to resort to Python loops or making
    copies of a.

    Taken from:
        http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
        http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
        http://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy

    Ex 1:
    >>> x = np.arange(10).reshape((2,5))
    >>> rollwin(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])    
    >>> np.mean(rollwin(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    Ex 2:
    >>> a = np.arange(10)
    >>> np.random.shuffle(a)
    >>> a
    array([7, 3, 6, 8, 4, 0, 9, 2, 1, 5])
    >>> rollwin(a, 3) == [8, 4, 0]
    array([[False, False, False],
           [False, False, False],
           [False, False, False],
           [ True,  True,  True],
           [False, False, False],
           [False, False, False],
           [False, False, False],
           [False, False, False]], dtype=bool)
    >>> np.all(rollwin(a, 3) == [8, 4, 0], axis=1)
    array([False, False, False,  True, False, False, False, False], dtype=bool)
    >>> np.where(np.all(rollwin(a, 3) == [8, 4, 0], axis=1))[0][0]
    3
    """
    shape = a.shape[:-1] + (a.shape[-1] - width + 1, width)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rollwin2D(a, width):
    """A modified version of rollwin. Allows for easy columnar search of 2D
    subarray b within larger 2D array a, assuming both have the same number of
    rows.
    
    Ex:
    >>> a
    array([[44, 89, 34, 67, 11, 92, 22, 72, 10, 81],
           [52, 40, 29, 35, 67, 10, 24, 23, 65, 51],
           [70, 58, 14, 34, 11, 66, 47, 68, 11, 56],
           [70, 55, 47, 30, 39, 79, 71, 70, 67, 33]])    
    >>> b
    array([[67, 11, 92],
           [35, 67, 10],
           [34, 11, 66],
           [30, 39, 79]])
    >>> np.where((rollwin2D(a, 3) == b).all(axis=1).all(axis=1))[0]
    array([3])
    """
    assert a.ndim == 2
    shape = (a.shape[1] - width + 1, a.shape[0], width)
    strides = (a.strides[-1],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def argcolsubarr2D(a, b):
    """Return column index of smaller subarray b within bigger array a. Both
    must be 2D and have the same number of rows. Raises IndexError if b is not
    a subarray of a"""
    assert a.ndim == b.ndim == 2
    assert a.shape[0] == b.shape[0] # same nrows
    width = b.shape[1] # ncols in b
    return np.where((rollwin2D(a, width) == b).all(axis=1).all(axis=1))[0]

def lrrep2Darrstripis(a):
    """Return left and right slice indices that strip repeated values from all rows
    from left and right ends of 2D array a, such that a[:, lefti:righti] gives you
    the stripped version.

    Ex:
    >>> a
    array([[44, 44, 44, 44, 89, 34, 67, 11, 92, 22, 72, 10, 81, 81, 81],
           [52, 52, 52, 52, 40, 29, 35, 67, 10, 24, 23, 65, 51, 51, 51],
           [70, 70, 70, 70, 58, 14, 34, 11, 66, 47, 68, 11, 56, 56, 56],
           [70, 70, 70, 70, 55, 47, 30, 39, 79, 71, 70, 67, 33, 33, 33]])
    >>> lrrep2Darrstripis(a)
    (3, -2)
    """
    assert a.ndim == 2
    left = a[:, :1] # 2D column vector
    right = a[:, -1:] # 2D column vector
    leftcolis = argcolsubarr2D(a, left)
    if len(leftcolis) == 1: # only 1 hit, at the far left edge
        lefti = 0
    else: # multiple hits, get slice index of rightmost consecutive hit
        lefti = max(np.where(np.diff(leftcolis) == 1)[0]) + 1
    rightcolis = argcolsubarr2D(a, right)
    if len(rightcolis) == 1: # only 1 hit, at the far right edge
        righti = a.shape[1]
    else: # multiple hits, get slice index of leftmost consecutive hit
        righti = -(max(np.where(np.diff(rightcolis)[::-1] == 1)[0]) + 1)
    return lefti, righti

def normpdf(p, lapcorrect=1e-10):
    """Ensure p is normalized (sums to 1). Return p unchanged if it's already normalized.
    Otherwise, return it normalized. I guess this treats p as a pmf, not strictly a pdf.
    Optional apply Laplacian correction to avoid 0s"""
    p = np.asarray(p)
    if lapcorrect and (p == 0).any():
        p += lapcorrect
    psum = p.sum()
    if not np.allclose(psum, 1.0) and psum > 0: # make sure the probs sum to 1
        #print("p sums to %f instead of 1, normalizing" % psum)
        p = np.float64(p) # copy and ensure it's float before modifying in-place
        p /= psum
    return p

def DKL(p, q):
    """Kullback-Leibler divergence from true probability distribution p to arbitrary
    distribution q"""
    assert len(p) == len(q)
    p, q = normpdf(np.asarray(p)), normpdf(np.asarray(q))
    return sum(p * np.log2(p/q))
    
def DJS(p, q):
    """Jensen-Shannon divergence, a symmetric measure of divergence between
    distributions p and q"""
    assert len(p) == len(q)
    p, q = normpdf(np.asarray(p)), normpdf(np.asarray(q))
    m = (p + q) / 2
    return (DKL(p, m) + DKL(q, m)) / 2

def updatenpyfilerows(fname, rows, arr):
    """Given a numpy formatted binary file (usually with .npy extension,
    but not necessarily), update 0-based rows (first dimension) of the
    array stored in the file from arr. Works for arrays of any rank >= 1"""
    assert len(arr) >= 1 # has at least 1 row
    f = open(fname, 'r+b') # open in read+write binary mode
    # get .npy format version:
    major, minor = np.lib.format.read_magic(f)
    assert (major == 1 and minor == 0)
    # read header to move file pointer to start of array in file
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
    assert shape == arr.shape
    assert fortran_order == np.isfortran(arr)
    assert dtype == arr.dtype
    arroffset = f.tell()
    rowsize = arr[0].size * dtype.itemsize # nbytes per row
    # sort rows so that we move efficiently from start to end of file
    rows = sorted(rows) # rows might be a set, list, tuple, or array, convert to list
    # update rows in file
    for row in rows:
        f.seek(arroffset + row*rowsize) # seek from start of file, row is 0-based
        f.write(arr[row])
    f.close()
