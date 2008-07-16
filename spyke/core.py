"""Core classes and functions used throughout spyke"""

from __future__ import division
from __future__ import with_statement

__authors__ = ['Martin Spacek', 'Reza Lotun']

import cPickle
import gzip
import hashlib
import time
import os

import wx

import numpy as np
import scipy.signal

from spyke import probes

MU = '\xb5' # greek mu symbol

DEFHIGHPASSSAMPFREQ = 50000 # default (possibly interpolated) high pass sample frequency, in Hz
DEFHIGHPASSSHCORRECT = False


class WaveForm(object):
    """Waveform object holds raw and interpolated data, timestamps, channels, and sampling frequency.
    Sliceable in time, and indexable in channel space. Resamples data as needed"""
    def __init__(self, data=None, ts=None, chans=None):
        self.data = data # in uV, potentially multichannel, depending on shape
        self.ts = ts # timestamps array in us, one for each sample (column) in data
        self.chans = chans # channel ids corresponding to rows in .data. If None, channel ids == data row indices

        # change of plans. interpolation stuff is going back in Stream, keep WaveForm as just a dumb container for data and ts and chans. nothing more

    def get_data(self):
        """Return s+h corrected and potentially interpolated data"""
        #print 'getting data from WaveForm object: %r' % self
        try:
            return self._data
        except AttributeError:
            self._data, self._ts = self.resample()
            return self._data

    def set_rawdata(self, rawdata):
        """Set raw data. This should really only happen on init"""
        self._rawdata = rawdata

    data = property(get_data, set_rawdata)

    def get_ts(self):
        """Return potentially interpolated timestamps corresponding to data"""
        try:
            return self._ts
        except AttributeError:
            self.data # accessing this property should generate resampled ._ts
            return self._ts

    def set_rawts(self, rawts):
        """Set raw data timepoints. This should really only happen on init"""
        self._rawts = rawts

    ts = property(get_ts, set_rawts)

    def get_sampfreq(self):
        return self.stream.sampfreq

    def set_sampfreq(self, sampfreq):
        """Delete current potentially interpolated data and ts
        on resetting desired sampfreq in .stream"""
        try:
            del self._data
            del self._ts
        except AttributeError:
            pass
        self.stream.sampfreq = sampfreq

    sampfreq = property(get_sampfreq, set_sampfreq)

    def resample(self):
        """Return potentially sample-and-hold corrected and Nyquist interpolated
        data and timepoints. See Blanche & Swindale, 2006"""
        print 'sampfreq, rawsampfreq, shcorrect = (%r, %r, %r)' % (self.sampfreq, self.rawsampfreq, self.shcorrect)
        if self.sampfreq == self.rawsampfreq and self.shcorrect == False:
            return self._rawdata, self._rawts # don't need to do anything, fast
        rawtres = intround(1 / self.rawsampfreq * 1e6) # us
        tres = intround(1 / self.sampfreq * 1e6) # us
        # number of raw samples per interpolated sampe (0.5 when interpolating from 25 to 50 kHz):
        nrspis = self.rawsampfreq / self.sampfreq # inverse of interpolation factor
        # normalize timepoints so that sampling period delta t = 1, ie divide all timepoints by tres
        #tns = self._rawts / rawtres # raw timepoints normalized - should I bother introunding here?
        N = 12 # number of kernel points

        # generate separate kernels per chan to correct each channel's s+h delay
        # TODO: does chan 0 have 0 delay, or a delay of 1us???
        chans = self.chans or range(len(self._rawdata)) # None indicates channel ids == data row indices
        nchansperboard = 32 # TODO: stop hard coding number of chans per board (32)
        i = np.asarray(chans) % nchansperboard # ordinal position of each chan in the hold queue
        dis = 1 * i # per channel delays, us. TODO: stop hard coding 1us delay per ordinal position
        ds = dis / rawtres # normalized per channel delays
        kernels = []
        wh = hamming
        h = np.sinc # sin(pi*t) / pi*t
        for d in ds:
            t = np.arange(start=-N/2-d, stop=N/2-d+nrspis, step=nrspis) # kernel sample points
            kernel = wh(t, N) * h(t)
            kernels.append(kernel)
        kernels = np.asarray(kernels)

        # TODO: need _rawdata to be 6 data points longer at either end, then do full convolution, then discard them after convolution
        #data = scipy.signal.convolve(self._rawdata, kernels, mode='full') # takes nD arrays, allowing us to convolve each chan (rows in a) with its appropriate kernel in (rows of) v, all in one fell swoop. No for loop required. In comparison, np.convolve only works on 1D arrays for some reason
        data = []
        for chani, chan in enumerate(chans):
            row = np.convolve(self._rawdata[chani], kernels[chani], mode='full')
            data.append(row)
        data = np.asarray(data)
        #ts = np.arange(self._rawts[0], self._rawts[-1] + rawtres, tres) # generate interpolated timepoints
        ts = np.arange(self._rawts[0], self._rawts[-1] + tres, tres) # generate interpolated timepoints
        #assert data.shape[1] == len(ts)
        print 'data.shape = %r' % (data.shape,)
        return data, ts

    def __getitem__(self, key):
        """Make waveform data sliceable in time, and directly indexable by channel id.
        Return a WaveForm if slicing.
        TODO: Right now, this Works off of raw data for consistency and simplicity,
        although this means you can't slice at interpolated timepoints, which might be bad..."""
        if key.__class__ == slice: # slice self, return a WaveForm
            if self.ts == None:
                rawdata = None
                rawts = None
            else:
                lo, hi = self._rawts.searchsorted([key.start, key.stop])
                rawdata = self._rawdata[:, lo:hi]
                rawts = self._rawts[lo:hi]
            if np.asarray(rawdata == self._rawdata).all() and np.asarray(rawts == self._rawts).all():
                return self # no need for a new WaveForm
            else:
                return WaveForm(data=rawdata, ts=rawts, chans=self.chans,
                                rawsampfreq=self.rawsampfreq, sampfreq=self.sampfreq,
                                shcorrect=self.shcorrect)
        else: # index into self by channel id, return that channel's data
            if self.chans == None: # contiguous chans, simple and fast
                return self.data[key] # TODO: should probably use .take here for speed
            else: # non contiguous chans
                try:
                    self.chan2i # converts from chan id to data array row index
                except AttributeError:
                    nchans = len(self.chans)
                    self.chan2i = dict(zip(self.chans, range(nchans)))
                return self.data[self.chan2i[key]] # TODO: should probably use .take here for speed

    def __len__(self):
        """Number of data points in time"""
        nt = len(self.ts)
        assert nt == self.data.shape[1] # obsessive
        return nt

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        try:
            del d['_data'] # to save space and time, don't pickle the potentially interpolated data
            del d['_ts'] # or timestamps
        except KeyError:
            pass
        #d['stream'] = None # we'll need to do something in Stream that keeps its attibs like .sampfreq, .rawsampfreq., etc, upon pickling, while still not trying to access an unavailable .srf file - maybe set its ctsrecords to []?
        return d


class Stream(object):
    """Data stream object - provides convenient stream interface to .srf files.
    Maps from timestamps to record index of stream data to retrieve the
    approriate range of waveform data from disk. Converts from AD units to uV"""

    def __init__(self, ctsrecords=None, sampfreq=None, shcorrect=None, endinclusive=False):
        """Takes a sorted temporal (not necessarily evenly-spaced, due to pauses in recording)
        sequence of ContinuousRecords: either HighPassRecords or LowPassMultiChanRecords.
        sampfreq arg is useful for interpolation"""
        self.ctsrecords = ctsrecords
        self.layout = self.ctsrecords[0].layout # layout record for this stream
        self.srffname = os.path.basename(self.layout.f.name) # filename excluding path
        self.rawsampfreq = self.layout.sampfreqperchan
        try:
            self.ctsrecords[0].lowpassrecords # it's a low pass stream
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default
        except AttributeError: # it's a high pass stream
            self.sampfreq = sampfreq or DEFHIGHPASSSAMPFREQ # desired sampling frequency
            self.shcorrect = shcorrect or DEFHIGHPASSSHCORRECT
        self.endinclusive = endinclusive

        self.nchans = len(self.layout.chanlist)
        self.chans = self.layout.chanlist
        if self.chans == range(self.nchans): # if it's contiguous
            self.chans = None # use this as a signal to indicate so to the WaveForm
        # array of ctsrecord timestamps
        self.rts = np.asarray([ctsrecord.TimeStamp for ctsrecord in self.ctsrecords])
        probename = self.layout.electrode_name
        probename = probename.replace(MU, 'u') # replace any 'micro' symbols with 'u'
        probetype = eval('probes.' + probename) # yucky. TODO: switch to a dict with keywords?
        self.probe = probetype() # instantiate it

        self.t0 = self.rts[0] # us, time that recording began
        lastctsrecordtw = intround(self.ctsrecords[-1].NumSamples / self.probe.nchans * self.tres)
        self.tend = self.rts[-1] + lastctsrecordtw  # time of last recorded data point

    def get_sampfreq(self):
        return self._sampfreq

    def set_sampfreq(self, sampfreq):
        """Updates .tres on .sampfreq change"""
        self._sampfreq = sampfreq
        self.tres = intround(1 / self.sampfreq * 1e6) # us, for convenience

    sampfreq = property(get_sampfreq, set_sampfreq)

    def __len__(self):
        """Total number of timepoints? Length in time? Interp'd or raw?"""
        raise NotImplementedError

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object, indicating
        start and end timepoints in us. Returns the corresponding WaveForm object, which has as
        its attribs the 2D multichannel waveform array as well as the timepoints, potentially
        spanning multiple ContinuousRecords"""

        # for now, accept only slice objects as keys
        assert key.__class__ == slice
        # key.step == -1 indicates we want the returned Waveform reversed in time
        # key.step == None behaves the same as key.step == 1
        assert key.step in [None, 1, -1]
        if key.step == -1:
            start, stop = key.stop, key.start # reverse start and stop, now start should be < stop
        else:
            start, stop = key.start, key.stop

        # Find the first and last records corresponding to the slice. If the start of the slice
        # matches a record's timestamp, start with that record. If the end of the slice matches a record's
        # timestamp, end with that record (even though you'll only potentially use the one timepoint from
        # that record, depending on the value of 'endinclusive')"""
        lorec, hirec = self.rts.searchsorted([start, stop], side='right') # TODO: this might need to be 'left' for step=-1

        # We always want to get back at least 1 record (ie records[0:1]). When slicing, we need to do
        # lower bounds checking (don't go less than 0), but not upper bounds checking
        cutrecords = self.ctsrecords[max(lorec-1, 0):max(hirec, 1)]
        for record in cutrecords:
            try:
                record.data
            except AttributeError:
                # to save time, only load the waveform if it's not already loaded
                record.load()

        # join all waveforms, returns a copy. Also, convert to float32 here,
        # instead of in .AD2uV(), since we're doing a copy here anyway.
        # Use float32 cuz it uses half the memory, and is also a little faster as a result.
        # Don't need float64 precision anyway.
        data = np.concatenate([np.float32(record.data) for record in cutrecords], axis=1)
        # all ctsrecords should be using the same layout, use tres from the first one
        tres = cutrecords[0].layout.tres

        # build up waveform timepoints, taking into account any time gaps in
        # between records due to pauses in recording
        ts = []
        for record in cutrecords:
            tstart = record.TimeStamp
            # number of timepoints (columns) in this record's waveform
            nt = record.data.shape[1]
            ts.extend(range(tstart, tstart + nt*tres, tres))
            #del record.data # save memory by unloading waveform data from records that aren't needed anymore
        ts = np.asarray(ts, dtype=np.int64) # force timestamps to be int64
        lo, hi = ts.searchsorted([start, stop])
        data = data[:, lo:hi+self.endinclusive] # TODO: is this the slowest step? use .take instead?
        #data = data.take(np.arange(lo, hi+self.endinclusive), axis=1) # doesn't seem to help performance
        ts = ts[lo:hi+self.endinclusive]
        #ts = ts.take(np.arange(lo, hi+self.endinclusive)) # doesn't seem to help performance

        # reverse data if need be
        if key.step == -1:
            data = data[:, ::key.step]
            ts = ts[::key.step]

        # transform AD values to uV, assume all chans in ctsrecords have same gain
        extgain = self.ctsrecords[0].layout.extgain
        intgain = self.ctsrecords[0].layout.intgain
        data = self.AD2uV(data, intgain, extgain)

        # return a WaveForm object
        return WaveForm(data=data, ts=ts, chans=self.chans, stream=self)

    def AD2uV(self, data, intgain, extgain):
        """Convert AD values in data to uV
        TODO: stop hard-coding 2048, should be (maxval of AD board + 1) / 2
        TODO: stop hard-coding 10V, should be max range at intgain of 1
        """
        return (data - 2048) * (10 / (2048 * intgain * extgain[0]) * 1000000)

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


class SpykeListCtrl(wx.ListCtrl):
    """ListCtrl with a couple of extra methods defined"""
    def GetSelections(self):
        """Return row indices of selected list items.
        wx.ListCtrl lacks something like this as a method"""
        selected_rows = []
        first = self.GetFirstSelected()
        if first == -1: # no more selected rows
            return selected_rows
        selected_rows.append(first)
        last = first
        while True:
            next = self.GetNextSelected(last)
            if next == -1: # no more selected rows
                return selected_rows
            selected_rows.append(next)
            last = next

    def InsertRow(self, row, data):
        """Insert data in list at row position.
        data is a list of strings or numbers, one per column.
        wx.ListCtrl lacks something like this as a method"""
        row = self.InsertStringItem(row, str(data[0])) # inserts data's first column
        for coli, val in enumerate(data[1:]): # insert the rest of data's columns
            self.SetStringItem(row, coli+1, str(val))

    def DeleteItemByData(self, data):
        """Delete first item whose first column matches data"""
        row = self.FindItem(0, str(data)) # start search from row 0
        assert row != -1, "couldn't find data %r in SpykeListCtrl" % str(data)
        success = self.DeleteItem(row) # remove from event listctrl
        assert success, "couldn't delete data %r from SpykeListCtrl" % str(data)


class SpykeTreeCtrl(wx.TreeCtrl):
    """TreeCtrl with overridden OnCompareItems().
    Also has a couple of helper functions"""
    def OnCompareItems(self, item1, item2):
        """Compare templates in tree according to the vertical
        position of the maxchan in space, for sorting purposes.
        Called by self.SortChildren

        TODO: sort member events by event times
        """
        try:
            self.SiteLoc
        except AttributeError:
            sortframe = self.GetTopLevelParent()
            self.SiteLoc = sortframe.session.probe.SiteLoc # do this here and not in __init__ due to binding order
        obj1 = self.GetItemPyData(item1)
        obj2 = self.GetItemPyData(item2)
        try: # make sure they're both templates by checking for .events attrib
            obj1.events
            obj2.events
            ycoord1 = self.SiteLoc[obj1.maxchan][1] # what about if the template's maxchan is still None?
            ycoord2 = self.SiteLoc[obj2.maxchan][1]
            return cmp(ycoord1, ycoord2)
        except AttributeError:
            raise RuntimeError, "can't yet deal with sorting events in tree"

    def GetTreeChildrenPyData(self, itemID):
        """Returns PyData of all children of item in
        order from top to bottom in a list"""
        children = []
        childID, cookie = self.GetFirstChild(itemID)
        while childID:
            child = self.GetItemPyData(childID)
            children.append(child)
            childID, cookie = self.GetNextChild(itemID, cookie)
        return children

    def GetTreeChildren(self, itemID):
        """Returns list of itemIDs of all children of item in
        order from top to bottom of the tree"""
        childrenIDs = []
        childID, cookie = self.GetFirstChild(itemID)
        while childID:
            childrenIDs.append(child)
            childID = self.GetNextChild(itemId, cookie)
        return childrenIDs

'''
class HybridList(set):
    """A set with an append() method like a list"""
    def append(self, item):
        self.add(item)
'''

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

def intround(n):
    """Round to the nearest integer, return an integer.
    Saves on parentheses"""
    return int(round(n))

def iterable(x):
    """Check if the input is iterable, stolen from numpy.iterable()"""
    try:
        iter(x)
        return True
    except:
        return False

def toiter(x):
    """Convert to iterable. If input is iterable, returns it. Otherwise returns it in a list.
    Useful when you want to iterate over a Record (like in a for loop),
    and you don't want to have to do type checking or handle exceptions
    when the Record isn't a sequence"""
    if iterable(x):
        return x
    else:
        return [x]

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


class Gaussian(object):
    """Gaussian function, works with ndarray inputs"""
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def f(self, x):
        mu = self.mean
        sigma = self.stdev
        # don't bother normalizing by 1/(sigma*np.sqrt(2*np.pi)), don't care about normalizing the integral,
        # just want to make sure that f(0) == 1
        return np.exp(- ((x-mu)**2 / (2*sigma**2)) )

    def __getitem__(self, x):
        return self.f(x)

def hamming(t, N=None):
    """Return y values of Hamming window at sample points t,
    centered on the middle sample"""
    if N == None:
        N = (len(t) - 1) / 2
    return 0.54 - 0.46 * np.cos(np.pi * (2*t + N)/N)
