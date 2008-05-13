"""Core classes and functions used throughout spyke"""

from __future__ import division
from __future__ import with_statement

__authors__ = ['Martin Spacek', 'Reza Lotun']

import cPickle
import gzip
import hashlib

import numpy as np

from spyke import probes


class WaveForm(object):
    """Waveform object, has data, timestamps, chan2i, and sample frequency attribs"""
    def __init__(self, data=None, ts=None, chan2i=None, sampfreq=None):
        self.data = data # always in uV? potentially multichannel, depending on shape
        self.ts = ts # timestamps array, one for each sample (column) in data
        self.chan2i = chan2i # converts from chan id to .data row index
        self.sampfreq = sampfreq # Hz

    def __getitem__(self, key):
        """Make waveform data directly indexable.
        Maybe this is where data should be interpolated?"""
        return self.data[self.chan2i[key]]

    def __len__(self):
        """Number of data points in time"""
        return self.data.shape[-1]


class Stream(object):
    """Streaming object - provides convenient stream interface to .srf files.
    Maps from timestamps to record index of stream data to retrieve the
    approriate range of waveform data from disk. Converts from AD units to uV"""
    DEFAULTINTERPSAMPFREQ = 50000 # default interpolated sample rate, in Hz

    def __init__(self, ctsrecords=None, sampfreq=None):
        """Takes a sorted temporal (not necessarily evenly-spaced, due to pauses in recording)
        sequence of ContinuousRecords: either HighPassRecords or LowPassMultiChanRecords.
        sampfreq arg is useful for interpolation"""
        self.ctsrecords = ctsrecords
        self.layout = ctsrecords[0].layout # layout record for this stream
        # if no sampfreq passed in, use sampfreq of the raw data
        self.sampfreq = sampfreq or self.layout.sampfreqperchan
        self.nchans = len(self.layout.chanlist)
        # converts from chan id to data array row index, identical to
        # .layout.chanlist unless there are channel gaps in .layout
        self.chan2i = dict(zip(self.layout.chanlist, range(self.nchans)))
        # array of ctsrecord timestamps
        self.rts = np.asarray([ctsrecord.TimeStamp for ctsrecord in self.ctsrecords])
        probename = self.layout.electrode_name
        probename = probename.replace('\xb5', 'u') # replace any 'micro' symbols with 'u'
        probetype = eval('probes.' + probename) # yucky. TODO: switch to a dict with keywords?
        self.probe = probetype()

        self.t0 = self.rts[0] # us, time that recording began
        self.tres = intround(1 / self.sampfreq * 1e6) # us, for convenience
        lastctsrecordtw = self.ctsrecords[-1].NumSamples / self.probe.nchans * self.tres
        self.tend = self.rts[-1] + lastctsrecordtw  # time of last recorded data point


    def __len__(self):
        """Total number of timepoints? Length in time? Interp'd or raw?"""
        raise NotImplementedError()

    def __getitem__(self, key, endinclusive=False):
        """Called when Stream object is indexed into using [] or with a slice object, indicating
        start and end timepoints in us. Returns the corresponding WaveForm object, which has as
        its attribs the 2D multichannel waveform array as well as the timepoints, potentially
        spanning multiple ContinuousRecords"""

        # for now, accept only slice objects as keys
        assert key.__class__ == slice

        # Find the first and last records corresponding to the slice. If the start of the slice
        # matches a record's timestamp, start with that record. If the end of the slice matches a record's
        # timestamp, end with that record (even though you'll only potentially use the one timepoint from
        # that record, depending on the value of 'endinclusive')"""
        lorec, hirec = self.rts.searchsorted([key.start, key.stop], side='right')

        # We always want to get back at least 1 record (ie records[0:1]). When slicing, we need to do
        # lower bounds checking (don't go less than 0), but not upper bounds checking
        cutrecords = self.ctsrecords[max(lorec-1, 0):max(hirec, 1)]
        for record in cutrecords:
            try:
                record.waveform
            except AttributeError:
                # to save time, only load the waveform if not already loaded
                record.load()

        # join all waveforms, returns a copy
        data = np.concatenate([record.waveform for record in cutrecords], axis=1)
        # all ctsrecords should be using the same layout, use tres from the first one
        tres = cutrecords[0].layout.tres

        # build up waveform timepoints, taking into account any time gaps in
        # between records due to pauses in recording
        ts = []
        for record in cutrecords:
            tstart = record.TimeStamp
            # number of timepoints (columns) in this record's waveform
            nt = record.waveform.shape[-1]
            ts.extend(range(tstart, tstart + nt*tres, tres))
            #del record.waveform # save memory by unloading waveform data from records that aren't needed anymore
        ts = np.asarray(ts)
        lo, hi = ts.searchsorted([key.start, key.stop])
        data = data[:, lo:hi+endinclusive]
        ts = ts[lo:hi+endinclusive]

        # interp and s+h correct here
        data, ts = self.interp(data, ts, self.sampfreq)

        # transform AD values to uV
        extgain = self.ctsrecords[0].layout.extgain
        intgain = self.ctsrecords[0].layout.intgain
        data = self.ADVal_to_uV(data, intgain, extgain)

        # return a WaveForm object
        return WaveForm(data=data, ts=ts, chan2i=self.chan2i, sampfreq=self.sampfreq)


    def ADVal_to_uV(self, adval, intgain, extgain):
        """Convert AD values to micro-volts"""
        # Delphi code:
        # Round((ADValue - 2048)*(10 / (2048
        #                  * ProbeArray[m_ProbeIndex].IntGain
        #                  * ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan]))
        #                  * V2uV);
        return (adval - 2048) * (10 / (2048 * intgain * extgain[0]) * 1e6)

    def interp(self, data, ts, sampfreq=None, kind='nyquist'):
        """Returns interpolated and sample-and-hold corrected data and
        timepoints, at the given sample frequency"""
        if kind == 'nyquist':
            # do Nyquist interpolation and S+H correction here, find a scipy function that'll do Nyquist interpolation?
            # TODO: Implement this!
            return data, ts
        else:
            raise ValueError, 'Unknown kind of interpolation %r' % kind

    def plot(self, chanis=None, trange=None):
        """Creates a simple matplotlib plot of the specified chanis over trange"""
        import pylab as pl # wouldn't otherwise need this, so import here
        from pylab import get_current_fig_manager as gcfm

        try:
            # see if neuropy is available
            from neuropy.Core import lastcmd, neuropyScalarFormatter, neuropyAutoLocator
        except ImportError:
            pass

        if chanis == None:
            # all high pass records should have the same chanlist
            if self.ctsrecords[0].__class__ == HighPassRecord:
                chanis = self.records[0].layout.chanlist
            # same goes for lowpassmultichanrecords, each has its own set of chanis,
            # derived previously from multiple single layout.chanlists
            elif self.ctsrecords[0].__class__ == LowPassMultiChanRecord:
                chanis = self.ctsrecords[0].chanis
            else:
                raise ValueError, 'unknown record type %s in self.records' % self.ctsrecords[0].__class__
        nchans = len(chanis)
        if trange == None:
            trange = (self.rts[0], self.rts[0]+100000)
        # make a waveform object
        wf = self[trange[0]:trange[1]]
        figheight = 1.25+0.2*nchans
        self.f = pl.figure(figsize=(16, figheight))
        self.a = self.f.add_subplot(111)

        try:
            gcfm().frame.SetTitle(lastcmd())
        except NameError:
            pass
        except AttributeError:
            pass

        try:
            # better behaved tick label formatter
            self.formatter = neuropyScalarFormatter()
            # use a thousands separator
            self.formatter.thousandsSep = ','
            # better behaved tick locator
            self.a.xaxis.set_major_locator(neuropyAutoLocator())
            self.a.xaxis.set_major_formatter(self.formatter)
        except NameError:
            pass
        for chanii, chani in enumerate(chanis):
            # upcast to int32 to prevent int16 overflow
            self.a.plot(wf.ts/1e3, (np.int32(wf.data[chanii])-2048+500*chani)/500., '-', label=str(chani))
        #self.a.legend()
        self.a.set_xlabel('time (ms)')
        self.a.set_ylabel('channel id')

        # assumes chanis are sorted
        self.a.set_ylim(chanis[0]-1, chanis[-1]+1)
        bottominches = 0.75
        heightinches = 0.15+0.2*nchans
        bottom = bottominches / figheight
        height = heightinches / figheight
        self.a.set_position([0.035, bottom, 0.94, height])
        pl.show()


class Spike(WaveForm):
    """A spike event"""
    def __init__(self, waveform=None, channel=None, event_time=None):
        self.data = waveform.data # potentially multichannel
        self.ts = waveform.ts
        self.sampfreq = waveform.sampfreq
        self.channel = channel # trigger channel
        self.event_time = event_time
        self.name = str(self)

    def __str__(self):
        return 'Channel ' + str(self.channel) + ' time: ' + \
                str(self.event_time)

    def __hash__(self):
        return hash(str(self.channel) + str(self.event_time) + \
                                                        str(self.data))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Template(object):
    """A spike template is simply a collection of spikes"""
    def __init__(self,):
        self.active_channels = []
        self._spikes = set() # sets have remove() method, lists don't
        self.name = str(self)
        self._init_props = False

    def remove(self, spike):
        self._spikes.remove(spike)

    def __repr__(self):
        return repr(self._spikes)

    def _set_props(self, spike):
        dim = spike.data.shape
        num_chans = dim[0]
        self.active_channels = [True] * num_chans

    def add(self, spike):
        """Add a spike"""
        # trigger intialization of certain properties
        if not self._init_props:
            self._set_props(spike)
            self._init_props = True
        self._spikes.add(spike)

    def __iter__(self):
        return iter(self._spikes)

    def __len__(self):
        return len(self._spikes)

    def mean(self):
        """Return the mean of all the contained spikes"""
        if len(self) == 0:
            return None
        # TODO: use numpy's mean() method
        sample = iter(self).next()
        dim = sample.data.shape
        _mean = Spike(sample)
        _mean.data = np.asarray([0.0] * dim[0] * dim[1]).reshape(dim)
        for num, spike in enumerate(self):
            _mean.data += spike.data
        _mean.data = _mean.data / (num + 1)
        return _mean

    def __eq__(self, oth):
        otype = isinstance(oth, Template)
        return hash(self) == hash(oth) if otype else False

    def __hash__(self):
        """XXX hmmm how probable would collisions be using this...?
        Could base this on the member _spikes instead of just the mean"""
        return hash(str(self.mean()) + str(self))

    def __str__(self):
        return 'Template (' + str(len(self)) + ')'


class HybridList(set):
    def append(self, item):
        self.add(item)


class Collection(object):
    """A container for Templates. Collections are associated with Surf Files.
    By default a Collection represents a single sorting session. Initially
    detected spikes will be added to a default set of spikes in a collection.
    These spikes will be differentiated through a combination of algorithmic
    and/or manual sorting"""
    def __init__(self, file=None):
        self.templates = HybridList()
        self.unsorted_spikes = HybridList() # these represent unsorted spikes
        self.recycle_bin = HybridList()
        self.surf_hash = '' # SHA1 hash of surf file

    def verify_surf(self, surf_file):
        """Verify that this collection corresponds to the surf file"""
        return spyke.get_sha1(surf_file) == self.data_hash

    def __len__(self):
        return len(self.templates)

    def __str__(self):
        """Pretty print the contents of the Collection"""
        s = []
        for t in self:
            s.extend([str(t), '\n'])
            for sp in t:
                s.extend(['\t', str(sp), '\n'])
        return ''.join(s)

    def __iter__(self):
        for template in self.templates:
            yield template


class SpykeError(Exception):
    """Base spyke error"""
    pass


class CollectionError(SpykeError):
    """Problem with collection file"""
    pass


def get_sha1(fname, blocksize=2**20):
    """Gets the sha1 hash of fname (with full path)"""
    m = hashlib.sha1()
    # automagically clean up after ourselves
    with file(fname, 'rb') as f:

        # continually update hash until EOF
        while True:
            block = f.read(blocksize)
            if not block:
                break
            m.update(block)

    return m.hexdigest()

def load_collection(fname):
    """Loads a collection file. Returns None if fname is not a collection"""
    with file(fname, 'rb') as f:
        try:
            g = gzip.GzipFile(fileobj=f, mode='rb')
            col = cPickle.load(g)
        except Exception, e:
            raise CollectionError(str(e))
        g.close()
    return col

def write_collection(collection, fname):
    """Writes a collection to fname"""
    with file(fname, 'wb') as f:
        try:
            g = gzip.GzipFile(fileobj=f, mode='wb')
            cPickle.dump(collection, g, -1)
        except Exception, e:
            raise CollectionError(str(e))
        g.close()

def intround(n):
    """Round to the nearest integer, return an integer.
    Saves on parentheses"""
    return int(round(n))
