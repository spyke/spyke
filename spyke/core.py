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
DEFHIGHPASSSHCORRECT = True
KERNELSIZE = 12 # apparently == number of kernel zero crossings, but that seems to depend on the phase of the kernel, some have one less. Anyway, total number of points in the kernel is this plus 1 (for the middle point) - see Blanche2006
assert KERNELSIZE % 2 == 0 # I think kernel size needs to be even
NCHANSPERBOARD = 32 # TODO: stop hard coding this

MAXLONGLONG = 2**63-1

class WaveForm(object):
    """Just a container for data, timestamps, and channels.
    Sliceable in time, and indexable in channel space"""
    def __init__(self, data=None, ts=None, chans=None):
        self.data = data # in uV, potentially multichannel, depending on shape
        self.ts = ts # timestamps array in us, one for each sample (column) in data
        self.chans = chans # channel ids corresponding to rows in .data. If None, channel ids == data row indices

    def __getitem__(self, key):
        """Make waveform data sliceable in time, and directly indexable by channel id.
        Return a WaveForm if slicing"""
        if key.__class__ == slice: # slice self in time, return a WaveForm
            if self.ts == None:
                data = None
                ts = None
            else:
                lo, hi = self.ts.searchsorted([key.start, key.stop])
                data = self.data[:, lo:hi]
                ts = self.ts[lo:hi]
            if np.asarray(data == self.data).all() and np.asarray(ts == self.ts).all():
                return self # no need for a new WaveForm
            else:
                return WaveForm(data=data, ts=ts, chans=self.chans)
        else: # index into self by channel id, return that channel's data, assume non contiguous chans
            try:
                self.chan2i # converts from chan id to data array row index
            except AttributeError:
                nchans = len(self.chans)
                self.chan2i = dict(zip(self.chans, range(nchans)))
            chans = [ self.chan2i[chan] for chan in toiter(key) ] # allow key to be list of chans
            # TODO: should probably use .take here for speed
            # eliminate any length 1 dimensions, ie if we're only returning one channel of data,
            # ensure the array is only rank 1, not rank 2
            return self.data[chans].squeeze()

    def __len__(self):
        """Number of data points in time"""
        nt = len(self.ts)
        assert nt == self.data.shape[1] # obsessive
        return nt


class Stream(object):
    """Data stream object - provides convenient stream interface to .srf files.
    Maps from timestamps to record index of stream data to retrieve the
    approriate range of waveform data from disk. Converts from AD units to uV

    TODO: might need to do something in Stream that keeps its attibs like .sampfreq,
    .rawsampfreq., etc, upon pickling, while still not trying to access an unavailable
    .srf file - maybe set its ctsrecords to []?
    """
    def __init__(self, ctsrecords=None, sampfreq=None, shcorrect=None, endinclusive=False):
        """Takes a sorted temporal (not necessarily evenly-spaced, due to pauses in recording)
        sequence of ContinuousRecords: either HighPassRecords or LowPassMultiChanRecords.
        sampfreq arg is useful for interpolation. Assumes that all HighPassRecords belong
        to the same probe"""
        self.ctsrecords = ctsrecords
        self.layout = self.ctsrecords[0].layout # layout record for this stream
        #self.fname = self.ctsrecords[0].srff.fname # full filename with path, needed for unpickling?
        fname = self.ctsrecords[0].srff.fname # full filename with path
        self.srffname = os.path.basename(fname) # filename excluding path
        self.rawsampfreq = self.layout.sampfreqperchan
        self.rawtres = intround(1 / self.rawsampfreq * 1e6) # us, for convenience
        self.nchans = len(self.layout.ADchanlist)
        try: # are ctsrecords LowPassMultiChanRecords?
            self.ctsrecords[0].lowpassrecords # yes, it's a low pass stream
            self.chans = self.layout.chans # probe chan values already parsed from LFP probe description
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default
        except AttributeError: # ctsrecords are HighPassRecords
            self.chans = range(self.nchans) # probe chans, as opposed to AD chans, don't know yet of any probe
                                            # type whose chans aren't contiguous from 0 (see probes.py)
            self.sampfreq = sampfreq or DEFHIGHPASSSAMPFREQ # desired sampling frequency
            self.shcorrect = shcorrect or DEFHIGHPASSSHCORRECT
        self.endinclusive = endinclusive
        self.rts = np.asarray([ctsrecord.TimeStamp for ctsrecord in self.ctsrecords]) # array of ctsrecord timestamps
        probename = self.layout.electrode_name
        probename = probename.replace(MU, 'u') # replace any 'micro' symbols with 'u'
        probetype = eval('probes.' + probename) # yucky. TODO: switch to a dict with keywords?
        self.probe = probetype() # instantiate it

        self.t0 = self.rts[0] # us, time that recording began
        lastctsrecordtw = intround(self.ctsrecords[-1].NumSamples / self.probe.nchans * self.rawtres)
        self.tend = self.rts[-1] + lastctsrecordtw  # time of last recorded data point

    def get_sampfreq(self):
        return self._sampfreq

    def set_sampfreq(self, sampfreq):
        """Deletes .kernels (if set), and updates .tres on .sampfreq change"""
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
        """Deletes .kernels (if set) on .shcorrect change"""
        self._shcorrect = shcorrect
        try:
            del self.kernels
        except AttributeError:
            pass

    shcorrect = property(get_shcorrect, set_shcorrect)

    def __len__(self):
        """Total number of timepoints? Length in time? Interp'd or raw?"""
        raise NotImplementedError

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object, indicating
        start and end timepoints in us. Returns the corresponding WaveForm object, which has as
        its attribs the 2D multichannel waveform array as well as the timepoints, potentially
        spanning multiple ContinuousRecords"""
        #tslice = time.clock()

        # for now, accept only slice objects as keys
        assert key.__class__ == slice
        # key.step == -1 indicates we want the returned Waveform reversed in time
        # key.step == None behaves the same as key.step == 1
        assert key.step in [None, 1, -1]
        if key.step == -1:
            start, stop = key.stop, key.start # reverse start and stop, now start should be < stop
        else:
            start, stop = key.start, key.stop

        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at our desired start and stop
            xs = KERNELSIZE * self.rawtres
        else:
            xs = 0

        # Find the first and last records corresponding to the slice. If the start of the slice
        # matches a record's timestamp, start with that record. If the end of the slice matches a record's
        # timestamp, end with that record (even though you'll only potentially use the one timepoint from
        # that record, depending on the value of 'endinclusive')"""
        lorec, hirec = self.rts.searchsorted([start-xs, stop+xs], side='right') # TODO: this might need to be 'left' for step=-1

        # We always want to get back at least 1 record (ie records[0:1]). When slicing, we need to do
        # lower bounds checking (don't go less than 0), but not upper bounds checking
        cutrecords = self.ctsrecords[max(lorec-1, 0):max(hirec, 1)]
        for record in cutrecords:
            try:
                record.data
            except AttributeError:
                record.load() # to save time, only load the waveform if it's not already loaded
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
            nt = record.data.shape[1] # number of timepoints (columns) in this record's waveform
            ts.extend(range(tstart, tstart + nt*tres, tres))
            del record.data # save memory by unloading waveform data from records that aren't needed anymore
        ts = np.asarray(ts, dtype=np.int64) # force timestamps to be int64
        lo, hi = ts.searchsorted([start-xs, stop+xs])
        data = data[:, lo:hi+self.endinclusive] # .take doesn't seem to be any faster
        ts = ts[lo:hi+self.endinclusive] # .take doesn't seem to be any faster

        # reverse data if need be
        if key.step == -1:
            data = data[:, ::key.step]
            ts = ts[::key.step]

        # transform AD values to uV, assume all chans in ctsrecords have same gain
        extgain = self.ctsrecords[0].layout.extgain
        intgain = self.ctsrecords[0].layout.intgain
        data = self.AD2uV(data, intgain, extgain)
        #print 'raw data shape before resample: %r' % (data.shape,)

        # do any resampling if necessary
        if resample:
            #tresample = time.clock()
            data, ts = self.resample(data, ts)
            #print 'resample took %.3f sec' % (time.clock()-tresample)

        # now get rid of any excess
        if xs:
            lo, hi = ts.searchsorted([start, stop])
            data = data[:, lo:hi+self.endinclusive]
            ts = ts[lo:hi+self.endinclusive]

        #print 'data and ts shape after rid of xs: %r, %r' % (data.shape, ts.shape)
        #print 'Stream slice took %.3f sec' % (time.clock()-tslice)

        # return a WaveForm object
        return WaveForm(data=data, ts=ts, chans=self.chans)
    '''
    def __setstate__(self, d):
        """Restore self on unpickle per usual, but also restore open .srf file
        for all records that self relies on, so they can once again read from the file"""
        self.__dict__ = d
        f = open(self.fname, 'rb')
        for ctsrecord in self.ctsrecords:
            ctsrecord.f = f # reset the open srf file for each ctsrecord
        self.layout.f = f # reset it for this stream's layout record as well
    '''
    def AD2uV(self, data, intgain, extgain):
        """Convert AD values in data to uV
        TODO: stop hard-coding 2048, should be (maxval of AD board + 1) / 2
        TODO: stop hard-coding 10V, should be max range at intgain of 1
        """
        return (data - 2048) * (10 / (2048 * intgain * extgain[0]) * 1000000)

    def resample(self, rawdata, rawts):
        """Return potentially sample-and-hold corrected and Nyquist interpolated
        data and timepoints. See Blanche & Swindale, 2006

        TODO: should interpolation be multithreaded?
        TODO: self.kernels should be deleted when selected chans change, self.nchans should be updated
        """
        #print 'sampfreq, rawsampfreq, shcorrect = (%r, %r, %r)' % (self.sampfreq, self.rawsampfreq, self.shcorrect)
        rawtres = self.rawtres # us
        tres = self.tres # us
        resamplex = intround(self.sampfreq / self.rawsampfreq) # resample factor: n output resampled points per input raw point
        assert resamplex >= 1, 'no decimation allowed'
        N = KERNELSIZE

        ADchans = self.layout.ADchanlist
        assert self.nchans == len(self.chans) == len(ADchans) # pretty basic assumption which might change if chans are disabled
        # check if kernels have been generated already
        try:
            self.kernels
        except AttributeError:
            self.kernels = self.get_kernels(ADchans, resamplex, N)

        # convolve the data with each kernel
        nrawts = len(rawts)
        # all the interpolated points have to fit in between the existing raw
        # points, so there's nrawts - 1 of each of the interpolated points:
        #nt = nrawts + (resamplex-1) * (nrawts - 1)
        # the above can be simplified to:
        nt = nrawts*resamplex - (resamplex - 1)
        tstart = rawts[0]
        ts = np.arange(start=tstart, stop=tstart+tres*nt, step=tres) # generate interpolated timepoints
        #print 'len(ts) is %r' % len(ts)
        assert len(ts) == nt
        data = np.empty((self.nchans, nt), dtype=np.float32) # resampled data, float32 uses half the space
        #print 'data.shape = %r' % (data.shape,)
        for ADchani, ADchan in enumerate(ADchans):
            for point, kernel in enumerate(self.kernels[ADchani]):
                """np.convolve(a, v, mode)
                for mode='same', only the K middle values are returned starting at n = (M-1)/2
                where K = len(a)-1 and M = len(v) - 1 and K >= M
                for mode='valid', you get the middle len(a) - len(v) + 1 number of values"""
                row = np.convolve(rawdata[ADchani], kernel, mode='same')
                #print 'len(rawdata[ADchani]) = %r' % len(rawdata[ADchani])
                #print 'len(kernel) = %r' % len(kernel)
                #print 'len(row): %r' % len(row)
                # interleave by assigning from point to end in steps of resamplex
                ti0 = (resamplex - point) % resamplex # index to start filling data from for this kernel's points
                rowti0 = int(point > 0) # index of first data point to use from convolution result 'row'
                data[ADchani, ti0::resamplex] = row[rowti0:] # discard the first data point from interpolant's convolutions, but not for raw data's convolutions
        return data, ts

    def get_kernels(self, ADchans, resamplex, N):
        """Generate a different set of kernels for each ADchan to correct each ADchan's s+h delay.

        TODO: when resamplex > 1 and shcorrect == False, you only need resamplex - 1 kernels.
        You don't need a kernel for the original raw data points. Those won't be shifted,
        so you can just interleave appropriately.

        TODO: take DIN channel into account, might need to shift all highpass ADchans
        by 1us, see line 2412 in SurfBawdMain.pas. I think the layout.sh_delay_offset field may tell you
        if and by how much you should take this into account

        WARNING! TODO: not sure if say ADchan 4 will always have a delay of 4us, or only if it's preceded by AD chans
        0, 1, 2 and 3 in the channel gain list - I suspect the latter is the case, but right now I'm coding the former
        """
        i = np.asarray(ADchans) % NCHANSPERBOARD # ordinal position of each chan in the hold queue
        if self.shcorrect:
            dis = 1 * i # per channel delays, us. TODO: stop hard coding 1us delay per ordinal position
        else:
            dis = 0 * i
        ds = dis / self.rawtres # normalized per channel delays
        wh = hamming # window function
        h = np.sinc # sin(pi*t) / pi*t
        kernels = [] # list of list of kernels, indexed by [ADchani][resample point]
        for ADchani in range(len(ADchans)):
            d = ds[ADchani] # delay for this chan
            kernelrow = []
            for point in range(resamplex): # iterate over resampled points per raw point
                t0 = point/resamplex # some fraction of 1
                tstart = -N/2 - t0 - d
                tstop = tstart + (N+1)
                t = np.arange(start=tstart, stop=tstop, step=1) # kernel sample timepoints, all of length N+1
                kernel = wh(t, N) * h(t) # windowed sinc
                kernelrow.append(kernel)
            kernels.append(kernelrow)
        return kernels


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

    def ToggleFocusedItem(self):
        """Toggles selection of focused list item"""
        itemID = self.GetFocusedItem()
        if itemID == -1: # no item focused
            return
        selectedIDs = self.GetSelections()
        if itemID in selectedIDs: # is already selected
            self.Select(itemID, on=0) # deselect it
        else: # isn't selected
            self.Select(itemID, on=1)


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
            self.SiteLoc = sortframe.sort.probe.SiteLoc # do this here and not in __init__ due to binding order
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

    def GetFocusedItem(self):
        """This relies on _focusedItem being set externally by handling
        tree events appropriately"""
        return self._focusedItem
    '''
    def GetSelectedItems(self):
        """This relies on _selectedItems being set externally by handling
        tree events appropriately. I think this differs from .GetSelections()
        in that the currently focused item isn't incorrectly assumed to
        also be a selected item"""
        return self._selectedItems
    '''
    def ToggleFocusedItem(self):
        """Toggles selection of focused tree item"""
        item = self.GetFocusedItem()
        if item in self.GetSelections(): # already selected
            select = False # deselect
        else: # not selected
            select = True # select it
        self.SelectItem(item, select)
        #print 'SpykeTreeCtrl.ToggleFocusedItem() not implemented yet, use Ctrl+Space instead'

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
        Don't bother normalizing by 1/(sigma*np.sqrt(2*np.pi)),
        don't care about normalizing the integral,
        just want to make sure that f(0) == 1"""
        return np.exp(- ((x-self.mu)**2 / (2*self.sigma**2)) )

    def __getitem__(self, x):
        """Called when self is indexed into"""
        return self(x)

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
    return 0.54 - 0.46 * np.cos(np.pi * (2*t + N)/N)
