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

from spyke import probes

MU = '\xb5' # greek mu symbol


class WaveForm(object):
    """Waveform object, has data, timestamps, channels, and sample frequency attribs
    Index directly into it by channel(s) using Waveform[chanis]"""
    def __init__(self, data=None, ts=None, chans=None, sampfreq=None):
        self.data = data # in uV, potentially multichannel, depending on shape
        self.ts = ts # timestamps array in us, one for each sample (column) in data
        self.chans = chans # channel ids corresponding to rows in .data. If None, channel ids == data row indices
        self.sampfreq = sampfreq # Hz

    def __getitem__(self, key):
        """Make waveform data sliceable in time, and directly indexable by channel id.
        Maybe this is where data should be interpolated?"""
        if key.__class__ == slice: # slice self, return a new WaveForm
            if self.ts == None:
                data = None
                ts = None
            else:
                lo, hi = self.ts.searchsorted([key.start, key.stop])
                data = self.data[:, lo:hi]
                ts = self.ts[lo:hi]
            return WaveForm(data=data, ts=ts,
                            chans=self.chans, sampfreq=self.sampfreq)
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
        return self.data.shape[-1]


class Stream(object):
    """Data stream object - provides convenient stream interface to .srf files.
    Maps from timestamps to record index of stream data to retrieve the
    approriate range of waveform data from disk. Converts from AD units to uV"""
    DEFAULTINTERPSAMPFREQ = 50000 # default interpolated sample rate, in Hz

    def __init__(self, ctsrecords=None, sampfreq=None, units='uV', endinclusive=False):
        """Takes a sorted temporal (not necessarily evenly-spaced, due to pauses in recording)
        sequence of ContinuousRecords: either HighPassRecords or LowPassMultiChanRecords.
        sampfreq arg is useful for interpolation"""
        self.ctsrecords = ctsrecords
        self.layout = self.ctsrecords[0].layout # layout record for this stream
        self.srffname = os.path.basename(self.layout.f.name) # filename excluding path
        # if no sampfreq passed in, use sampfreq of the raw data
        self.sampfreq = sampfreq or self.layout.sampfreqperchan
        self.units = units
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
        self.tres = intround(1 / self.sampfreq * 1e6) # us, for convenience
        lastctsrecordtw = intround(self.ctsrecords[-1].NumSamples / self.probe.nchans * self.tres)
        self.tend = self.rts[-1] + lastctsrecordtw  # time of last recorded data point

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
                record.waveform
            except AttributeError:
                # to save time, only load the waveform if not already loaded
                record.load()

        # join all waveforms, returns a copy. Also, convert to float32 here,
        # instead of in .AD2uV(), since we're doing a copy here anyway.
        # Use float32 cuz it uses half the memory, and is also a little faster as a result.
        # Don't need float64 precision anyway.
        # TODO: do interpolation here too, which will also need more memory.
        data = np.concatenate([np.float32(record.waveform) for record in cutrecords], axis=1)
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
        ts = np.asarray(ts, dtype=np.int64) # force timestamps to be int64
        lo, hi = ts.searchsorted([start, stop])
        data = data[:, lo:hi+self.endinclusive] # TODO: is this the slowest step? use .take instead?
        #data = data.take(np.arange(lo, hi+self.endinclusive), axis=1) # doesn't seem to help performance
        ts = ts[lo:hi+self.endinclusive]
        #ts = ts.take(np.arange(lo, hi+self.endinclusive)) # doesn't seem to help performance

        # interp and s+h correct here, reverse if need be
        data, ts = self.interp(data, ts, self.sampfreq)
        if key.step == -1:
            data = data[:, ::key.step]
            ts = ts[::key.step]

        # transform AD values to uV, assume all chans in ctsrecords have same gain
        extgain = self.ctsrecords[0].layout.extgain
        intgain = self.ctsrecords[0].layout.intgain
        data = self.AD2uV(data, intgain, extgain)

        # return a WaveForm object - TODO: does this return a copy or just a ref to data? I think just a ref
        return WaveForm(data=data, ts=ts, chans=self.chans, sampfreq=self.sampfreq)

    def AD2uV(self, data, intgain, extgain):
        """Convert AD values in data to uV
        Delphi code:
        Round((ADValue - 2048)*(10 / (2048
                         * ProbeArray[m_ProbeIndex].IntGain
                         * ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan]))
                         * V2uV);

        TODO: stop hard-coding 2048, should be (maxval of AD board + 1) / 2
        TODO: stop hard-coding 10V, should be max range at intgain of 1
        """
        return (data - 2048) * (10 / (2048 * intgain * extgain[0]) * 1000000)

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
        for coli, val in enumerate(data[1:]):
            self.SetStringItem(row, coli+1, str(val))


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
