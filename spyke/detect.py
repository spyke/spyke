"""Event detection algorithms

TODO: check if python's cygwincompiler.py module has been updated to deal with
      extra version fields in latest mingw
"""

from __future__ import division

__authors__ = ['Martin Spacek, Reza Lotun']

import itertools
import sys
import time
#import processing

import wx

import numpy as np

import spyke.surf
from spyke.core import WaveForm, toiter, argcut, intround, eucd
#from detect_weave import BipolarAmplitudeFixedThresh_Weave
try:
    import detect_cy
    from detect_cy import BipolarAmplitude_Cy, DynamicMultiphasic_Cy
    cy_module = detect_cy
except ImportError: # detect_cy isn't available
    import simple_detect_cy
    from simple_detect_cy import BipolarAmplitude_Cy
    cy_module = simple_detect_cy

class RandomWaveTranges(object):
    """Iterator that spits out time ranges of width bs with
    excess bx that begin randomly from within the given trange.
    Optionally spits out no more than maxntranges tranges"""
    def __init__(self, trange, bs, bx=0, maxntranges=None):
        self.trange = trange
        self.bs = bs
        self.bx = bx
        self.maxntranges = maxntranges
        self.ntranges = 0

    def next(self):
        # random int within trange
        if self.maxntranges != None and self.ntranges >= self.maxntranges:
            raise StopIteration
        t0 = np.random.randint(low=self.trange[0], high=self.trange[1])
        tend = t0 + self.bs
        self.ntranges += 1
        return (t0-self.bx, tend+self.bx)

    def __iter__(self):
        return self


class Detector(object):
    """Event detector base class"""
    #DEFALGORITHM = 'BipolarAmplitude'
    DEFALGORITHM = 'DynamicMultiphasic'
    DEFTHRESHMETHOD = 'Dynamic'
    DEFNOISEMETHOD = 'median'
    DEFNOISEMULT = 3.5
    DEFFIXEDTHRESH = 40 # uV
    DEFFIXEDNOISEWIN = 1000000 # 1s
    DEFDYNAMICNOISEWIN = 10000 # 10ms
    DEFMAXNEVENTS = 15
    DEFBLOCKSIZE = 1000000 # us, waveform data block size
    RANDOMBLOCKSIZE = 100000 # us, block size to use if we're randomly sampling
    DEFSLOCK = 175 # um
    DEFTLOCK = 300 # us
    DEFRANDOMSAMPLE = False

    MAXAVGFIRINGRATE = 1000 # Hz, assume no chan will trigger more than this rate of events on average within a block
    BLOCKEXCESS = 1000 # us, extra data as buffer at start and end of a block while searching for events. Only useful for ensuring event times within the actual block time range are accurate. Events detected in the excess are discarded

    def __init__(self, stream, chans=None,
                 threshmethod=None, noisemethod=None, noisemult=None, fixedthresh=None,
                 fixednoisewin=None, dynamicnoisewin=None,
                 trange=None, maxnevents=None, blocksize=None,
                 slock=None, tlock=None, randomsample=None):
        """Takes a data stream and sets various parameters"""
        self.srffname = stream.srffname # used to potentially reassociate self with stream on unpickling
        self.stream = stream
        self.chans = chans or range(self.stream.nchans) # None means search all channels
        self.nchans = len(self.chans)
        self.dm = self.get_full_chan_distance_matrix() # channel distance matrix, identical for all Detectors on the same probe
        self.threshmethod = threshmethod or self.DEFTHRESHMETHOD
        self.noisemethod = noisemethod or self.DEFNOISEMETHOD
        self.noisemult = noisemult or self.DEFNOISEMULT
        self.fixedthresh = fixedthresh or self.DEFFIXEDTHRESH
        self.fixednoisewin = fixednoisewin or self.DEFFIXEDNOISEWIN # us
        self.dynamicnoisewin = dynamicnoisewin or self.DEFDYNAMICNOISEWIN # us
        self.trange = trange or (stream.t0, stream.tend)
        self.maxnevents = maxnevents or self.DEFMAXNEVENTS # return at most this many events, applies across chans
        self.blocksize = blocksize or self.DEFBLOCKSIZE
        self.slock = slock or self.DEFSLOCK
        self.tlock = tlock or self.DEFTLOCK
        self.randomsample = randomsample or self.DEFRANDOMSAMPLE

    def search(self):
        """Search for events. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results

        TODO: remove any events that happen right at the first or last timepoint in the file,
        since we can't say when an interrupted rising or falling edge would've reached peak
        """
        t0 = time.clock()

        self.thresh = self.get_thresh() # this could probably go in __init__ without problems
        print '.get_thresh() took %.3f sec' % (time.clock()-t0)

        if self.randomsample:
            # random sampling ignores Detector's blocksize and enforces its own specially sized one
            bs = self.RANDOMBLOCKSIZE
        else:
            bs = self.blocksize
        bx = self.BLOCKEXCESS
        bs_sec = bs/1000000 # from us to sec
        maxneventsperchanperblock = bs_sec * self.MAXAVGFIRINGRATE # num elements per chan to preallocate before searching a block
        # reset this at the start of every search
        self.nevents = 0 # total num events found across all chans so far by this Detector
        # these hold temp eventtimes and maxchans for .searchblock, reused on every call
        self._eventtimes = np.empty(self.nchans*maxneventsperchanperblock, dtype=np.int64)
        self._maxchans = np.empty(self.nchans*maxneventsperchanperblock, dtype=int)
        self.tilock = self.us2nt(self.tlock)

        wavetranges, (bs, bx, direction) = self.get_blockranges(bs, bx)

        # should probably do a check here to see if it's worth using multiple processes
        # so, check if wavetrange is big enough, and/or if maxnevents is big enough
        # if random sampling, use only a single process?
        #pool = processing.Pool() # spawns as many worker processes as there are CPUs/cores on the machine
        self.events = [] # list of 2D event arrays returned by .searchblockprocess(), one array per block
        results = [] # stores ApplyResult objects returned by pool.applyAsync
        for wavetrange in wavetranges:
            args = (wavetrange, direction)
            # NOTE: might need to make a dummyDetector object with the right attribs to prevent mpl stuff and everything else from being copied over to each new spawned process???
            #result = pool.applyAsync(self.searchblockprocess, args=args, callback=self.handle_eventarr)
            #results.append(result) # not really necessary
            try:
                eventarr = self.searchblockprocess(*args)
                self.handle_eventarr(eventarr)
            except ValueError: # we've found all the events we need
                break # out of wavetranges loop
        #print 'done queueing tasks, result objects are: %r' % results
        #pool.close() # prevent any more tasks from being added to pool
        #pool.join() # wait until all tasks are done

        try:
            events = np.concatenate(self.events, axis=1)
        except ValueError: # self.events is an empty list
            events = np.asarray(self.events)
            events.shape = (2, 0)
        print '\nfound %d events in total' % events.shape[1]
        print 'inside .search() took %.3f sec' % (time.clock()-t0)
        return events

    def searchblockprocess(self, wavetrange, direction):
        """This is what a worker process executes"""
        print 'in searchblockprocess, self.nevents=%r, self.maxnevents=%r' % (self.nevents, self.maxnevents)
        if self.nevents >= self.maxnevents:
            raise ValueError # skip this iteration. TODO: this should really cancel all enqueued tasks
        tlo, thi = wavetrange # tlo could be > thi
        bx = self.BLOCKEXCESS
        cutrange = (tlo+bx, thi-bx) # range without the excess, ie time range of events to actually keep
        #print 'wavetrange: %r, cutrange: %r' % (wavetrange, cutrange)
        wave = self.stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed
        if self.randomsample:
            maxnevents = 1 # how many more we're looking for in the next block
        else:
            maxnevents = self.maxnevents - self.nevents
        eventarr = self.searchblock(wave, cutrange, maxnevents) # drop into C loop
        return eventarr

    def handle_eventarr(self, eventarr):
        """Blocking callback, called every time a worker process completes a task"""
        print 'handle_eventarr got: %r' % eventarr
        if eventarr == None:
            return
        nnewevents = eventarr.shape[1] # number of columns
        #wx.Yield() # allow GUI to update
        if self.randomsample and eventarr.tolist() in np.asarray(self.events).tolist():
            # check if eventarr is a duplicate of any that are already in .events, if so,
            # don't append this new eventarr, and don't inc self.nevents. Duplicates are possible
            # in random sampling cuz we might end up with blocks with overlapping tranges.
            # Converting to lists for the check is probably slow cuz, but at least it's legible and correct
            sys.stdout.write('found duplicate random sampled event')
        elif nnewevents != 0:
            self.events.append(eventarr)
            self.nevents += nnewevents # update
            sys.stdout.write('.')

    def get_blockranges(self, bs, bx):
        """Generate time ranges for slightly overlapping blocks of data,
        given blocksize and blockexcess"""
        wavetranges = []
        bs = abs(bs)
        bx = abs(bx)
        if self.trange[1] >= self.trange[0]: # search forward
            direction = 1
        else: # self.trange[1] < self.trange[0], # search backward
            bs = -bs
            bx = -bx
            direction = -1

        if self.randomsample:
            # wavetranges is an iterator that spits out random ranges starting from within
            # self.trange, and of width bs + 2bx
            if direction == -1:
                raise ValueError, "Check trange - I'd rather not do a backwards random search"
            wavetranges = RandomWaveTranges(self.trange, bs, bx)
        else:
            es = range(self.trange[0], self.trange[1], bs) # left (or right) edges of data blocks
            for e in es:
                wavetranges.append((e-bx, e+bs+bx)) # time range of waveform to give to .searchblock
            # last wavetrange surpasses self.trange[1] by some unknown amount, fix that here:
            wavetranges[-1] = (wavetranges[-1][0], self.trange[1]+bx) # replace with a new tuple
        return wavetranges, (bs, bx, direction)

    # leave the stream be, let it be pickled
    '''
    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['_stream'] # don't pickle the stream, cuz it relies on ctsrecords, which rely on open .srf file
        return d
    '''
    def get_stream(self):
        return self._stream

    def set_stream(self, stream=None):
        """Check that self's srf file matches stream's srf file before binding stream"""
        if stream == None or stream.srffname != self.srffname:
            self._stream = None
        else:
            self._stream = stream # it's from the same file, bind it

    stream = property(get_stream, set_stream)

    def get_full_chan_distance_matrix(self):
        """Get full channel distance matrix, in um"""
        chans_coords = self.stream.probe.SiteLoc.items() # list of tuples
        chans_coords.sort() # sort by chanid
        # TODO: what if this probe is missing some channel ids, ie chans aren't consecutive in layout?
        # That'll screw up indexing into the distance matrix, unless we insert dummy entries in the matrix
        # for those chans missing from the layout
        chans = [ chan_coord[0] for chan_coord in chans_coords ] # pull out the sorted chans and check them
        assert chans == range(len(chans)), 'is probe layout channel list not consecutive starting from 0?'
        coords = [ chan_coord[1] for chan_coord in chans_coords ] # pull out the coords, now in channel id order
        return eucd(coords)

    def get_thresh(self):
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            thresh = np.ones(self.nchans, dtype=np.float32) * self.fixedthresh
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed thresh, calculate from start of stream
            """randomly sample DEFFIXEDNOISEWIN's worth of data from the entire file in blocks of self.RANDOMBLOCKSIZE
            NOTE: this samples with replacement, so it's possible, though unlikely, that some parts of the data
            will contribute more than once to the noise calculation
            This sometimes causes an 'unhandled exception' for BipolarAmplitude algorithm, don't know why
            """
            nblocks = intround(self.DEFFIXEDNOISEWIN / self.RANDOMBLOCKSIZE)
            wavetranges = RandomWaveTranges(self.trange, bs=self.RANDOMBLOCKSIZE, bx=0, maxntranges=nblocks)
            data = []
            for wavetrange in wavetranges:
                data.append(self.stream[wavetrange[0]:wavetrange[1]].data)
            data = np.concatenate(data, axis=1)
            noise = self.get_noise(data)
            thresh = noise * self.noisemult
        elif self.threshmethod == 'Dynamic':
            thresh = np.zeros(self.nchans, dtype=np.float32) # this will be calculated on the fly in the Cython loop
        else:
            raise ValueError
        print 'thresh = %s' % thresh
        assert thresh.dtype == np.float32
        return thresh

    def get_noise(self, data):
        """Calculates noise over last dim in data (time), using .noisemethod"""
        if self.noisemethod == 'median':
            return np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            return np.stdev(data, axis=-1)
        else:
            raise ValueError

    def us2nt(self, us):
        """Convert time in us to nearest number of eq'v timepoints in stream"""
        nt = intround(us / self.stream.tres)
        # prevent rounding nt down to 0. This way, even the smallest
        # non-zero us will get you at least 1 timepoint
        if nt == 0 and us != 0:
            nt = 1
        return nt


class BipolarAmplitude(Detector, BipolarAmplitude_Cy):
    """Bipolar amplitude detector"""
    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.algorithm = 'BipolarAmplitude'


class DynamicMultiphasic(Detector, DynamicMultiphasic_Cy):
    """Dynamic multiphasic detector"""
    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.algorithm = 'DynamicMultiphasic'
