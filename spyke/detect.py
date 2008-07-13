"""Event detection algorithms

TODO: use median based noise estimation instead of std based
      - estimate noise level dynamically with sliding window
        and independently for each channel

TODO: for MultiPhasic, do spatial lockout only during first 1/2 phase of trigger spike

TODO: for speed (esp comparing signal to thresh), consider converting all uV data
      from 64bit float to 16 bit integer (actually, just keep it as 16 bit to begin
      with) - ack, not being in uV would complicate things all over the place

TODO: check if python's cygwincompiler.py module has been updated to deal with
      extra version fields in latest mingw
"""

from __future__ import division

__authors__ = ['Martin Spacek, Reza Lotun']

import itertools
import sys
import time

import wx

import numpy as np

import spyke.surf
from spyke.core import WaveForm, toiter, argcut, intround, eucd
#from detect_weave import BipolarAmplitudeFixedThresh_Weave
from detect_cy import BipolarAmplitudeFixedThresh_Cy


class RandomWaveTranges(object):
    """Iterator that spits out time ranges of width bs with
    excess bx that begin randomly from within the given trange"""
    def __init__(self, trange, bs, bx):
        self.trange = trange
        self.bs = bs
        self.bx = bx

    def next(self):
        # random int within trange
        t0 = np.random.randint(low=self.trange[0], high=self.trange[1])
        tend = t0 + self.bs
        return (t0-self.bx, tend+self.bx)

    def __iter__(self):
        return self


class Detector(object):
    """Event detector base class"""
    DEFFIXEDTHRESH = 50 # uV
    DEFNOISEMETHOD = 'median'
    DEFNOISEMULT = 4
    DEFNOISEWINDOW = 10000000 # 10 sec
    DEFMAXNEVENTS = 500
    DEFBLOCKSIZE = 1000000 # waveform data block size, us
    RANDOMBLOCKSIZE = 10000 # block size to use if we're randomly sampling
    DEFSLOCK = 175 # um
    DEFTLOCK = 440 # us
    DEFRANDOMSAMPLE = True

    MAXAVGFIRINGRATE = 1000 # Hz, assume no chan will trigger more than this rate of events on average within a block
    BLOCKEXCESS = 1000 # us, extra data as buffer at start and end of a block while searching for events. Only useful for ensuring event times within the actual block time range are accurate. Events detected in the excess are discarded

    def __init__(self, stream, chans=None,
                 fixedthresh=None, noisemethod=None, noisemult=None, noisewindow=None,
                 trange=None, maxnevents=None, blocksize=None,
                 slock=None, tlock=None, randomsample=None):
        """Takes a data stream and sets various parameters"""
        self.srffname = stream.srffname # used to potentially reassociate self with stream on unpickling
        self.stream = stream
        self.chans = chans or range(self.stream.nchans) # None means search all channels
        self.dm = self.get_full_chan_distance_matrix() # channel distance matrix, identical for all Detectors on the same probe
        # assign all thresh and noise attribs, then reassign as None for subclasses where one of them doesn't apply
        self.fixedthresh = fixedthresh or self.DEFFIXEDTHRESH
        self.noisemethod = noisemethod or self.DEFNOISEMETHOD
        self.noisemult = noisemult or self.DEFNOISEMULT
        self.noisewindow = noisewindow or self.DEFNOISEWINDOW
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
        self._eventtimes = np.empty(len(self.chans)*maxneventsperchanperblock, dtype=np.int64)
        self._maxchans = np.empty(len(self.chans)*maxneventsperchanperblock, dtype=int)
        self.tilock = self.us2nt(self.tlock)

        wavetranges, (bs, bx, direction) = self.get_blockranges(bs, bx)

        events = [] # list of 2D event arrays returned by .searchblock(), one array per block
        for wavetrange in wavetranges: # iterate over time ranges with excess in them, one per block
            if self.nevents >= self.maxnevents:
                break # out of for loop
            tlo, thi = wavetrange # tlo could be > thi
            cutrange = (tlo+bx, thi-bx) # range without the excess, ie time range of events to actually keep
            #print 'wavetrange: %r, cutrange: %r' % (wavetrange, cutrange)
            wave = self.stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed
            if self.randomsample:
                maxnevents = 1 # how many more we're looking for in the next block
            else:
                maxnevents = self.maxnevents - self.nevents
            eventarr = self.searchblock(wave, cutrange, maxnevents) # TODO: this should be threaded
            nnewevents = eventarr.shape[1] # number of columns
            #wx.Yield() # allow GUI to update
            if self.randomsample and eventarr.tolist() in np.asarray(events).tolist():
                # check if eventarr is a duplicate of any that are already in .events, if so,
                # don't append this new event, and don't inc self.nevents. Duplicates are possible
                # in random sampling cuz we might end up with blocks with overlapping tranges
                # converting to lists for the check is probably slow cuz, but at least it's legible and correct
                sys.stdout.write('found duplicate random sampled event')
            elif nnewevents != 0:
                events.append(eventarr)
                self.nevents += nnewevents # update
                sys.stdout.write('.')
        events = np.concatenate(events, axis=1)
        print '\nfound %d events in total' % events.shape[1]
        print 'inside .search() took %.3f sec' % (time.clock()-t0)
        return events

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

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['_stream'] # don't pickle the stream, cuz it relies on ctsrecords, which rely on open .srf file
        return d

    def get_stream(self):
        return self._stream

    def set_stream(self, stream=None):
        """Check that self's srf file matches stream's srf file before binding stream"""
        if stream == None or stream.srffname != self.srffname:
            self._stream = None
        else:
            self._stream = stream # it's from the same file, bind it

    stream = property(get_stream, set_stream)
    '''
    def get_chans(self):
        return self._chans

    def set_chans(self, chans):
        if chans == None:
            chans = range(self.stream.nchans) # search all channels
        self._chans = toiter(chans) # need not be contiguous
        self._chans.sort() # make sure they're in order
        self.dm = self.get_chan_distance_matrix() # Euclidean channel distance matrix, in self.chans order

    chans = property(get_chans, set_chans)

    def get_chan_distance_matrix(self):
        """Get subset of channel distance matrix, in um, based on self.chans"""
        sl = self.stream.probe.SiteLoc
        coords = []
        for chan in self.chans:
            coords.append(sl[chan])
        return eucd(coords)
    '''
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

    def get_thresh(self, chan):
        """Calculate either median or stdev based threshold for a given chan"""
        if self.noisemethod == 'median':
            self.get_median_thresh(chan)
        elif  self.noisemethod == 'stdev':
            self.get_stdev_thresh(chan)

    def get_median_thresh(self, chan):
        return self.get_median_noise(chan) * self.MEDIAN_MULT

    def get_stdev_thresh(self, chan):
        return self.get_stdev_noise(chan) * self.STDEV_MULT

    def get_median_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def get_stdev_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def us2nt(self, us):
        """Convert time in us to nearest number of eq'v timepoints in stream"""
        nt = intround(self.stream.layout.sampfreqperchan * us / 1000000)
        # prevent rounding nt down to 0. This way, even the smallest
        # non-zero us will get you at least 1 timepoint
        if nt == 0 and us != 0:
            nt = 1
        return nt


class FixedThresh(Detector):
    """Base class for fixed threshold event detection,
    Uses the same single static threshold throughout the entire file,
    with an independent threshold for every channel"""

    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.threshmethod = 'FixedThresh'
        self.noisemult = None # doesn't apply for FixedThresh Detector

    def get_median_noise(self, chan):
        pass

    def get_median_thresh(self, chan):
        pass
    '''
        """Used to determine threshold and set initial state"""
        # get stdev for each channel along a STDEV_WINDOW window
        wave = self.stream[self.t0:self.t0 + STDEV_WINDOW]
        self.std = {}
        for chan, d in enumerate(wave.data):
            self.std[chan] = wave.data[chan].std()

        # set the threshold to be STDEV_MULT * standard deviation
        # each chan has a separate thresh
        self.thresholds = {}
        for chan, stdev in self.std.iteritems():
            self.thresholds[chan] = stdev * self.STDEV_MULT
    '''

class DynamicThresh(Detector):
    """Base class for dynamic threshold event detection,
    Uses varying thresholds throughout the entire file,
    depending on the local noise level

    Calculate noise level using, say, a 50ms sliding window centered on the
    timepoint you're currently testing for an event. Or, use fixed pos
    windows, pre calc noise for each of them, and take noise level from whichever
    window you happen to be in while checking a timepoint for thresh xing.
    """
    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.threshmethod = 'DynamicThresh'
        self.fixedthresh = None # doesn't apply for DynamicThresh Detector

    def get_median_noise(self, chan):
        pass

    def get_stdev_noise(self, chan):
        pass


class BipolarAmplitudeFixedThresh(FixedThresh,
                                  BipolarAmplitudeFixedThresh_Cy):
    """Bipolar amplitude fixed threshold detector,
    with fixed temporal lockout on all channels, plus a spatial lockout"""

    def __init__(self, *args, **kwargs):
        FixedThresh.__init__(self, *args, **kwargs)
        self.algorithm = 'BipolarAmplitude'


class MultiPhasic(FixedThresh):
    """Multiphasic filter - events triggered only when consecutive
    thresholds of opposite polarity occur on a given channel within
    a specified time window delta_t

    That is, either:

        1) s_i(t) > f and s_i(t + t') < -f
        2) s_i(t) < -f and s_it(t + t') > f

    for 0 < t' <= delta_t
    """

    STDEV_MULT = 4
    EVENT_PRE = 250
    EVENT_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000
    delta_t = 300

    def find(self):
        """Maintain state and search forward for an event"""

        # keep on sliding our search window forward to find events
        while True:

            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = np.where(numpy.abs(self.window.data[chan]) > thresh)[0]

                if len(_ev) <= 0:
                    continue

                thresh_vals = [(self.window.data[chan][ind], ind) \
                                    for ind in _ev.tolist()]
                # for each threshold value, scan forwrd in time delta_t
                # to see if an opposite threshold crossing occurred
                for i, tup in enumerate(thresh_vals):
                    val, ind = tup
                    sgn = numpy.sign(val)
                    t = self.window.ts[ind]
                    for cand_val, t_ind in thresh_vals[i + 1:]:
                        # check ahead only with delt_t
                        if self.window.ts[t_ind] - t > self.delta_t:
                            break
                        cand_sgn = numpy.sign(cand_val)
                        # check if threshold crossings are opposite
                        # polarity
                        if cand_sgn != sgn:
                            chan_events.append((ind, chan))
                            break

                for evt in self.yield_events(chan_events):
                    yield evt

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]


class DynamicMultiPhasic(FixedThresh):
    """Dynamic Multiphasic filter - events triggered only when consecutive
    thresholds of opposite polarity occured on a given channel within
    a specified time window delta_t, where the second threshold level is
    determined relative to the amplitude of the waveform peak/valley
    following initial phase trigger

    That is, either:

        1) s_i(t) > f and s_i(t + t') < f_pk - f'
    or  2) s_i(t) < -f and s_it(t + t') > f_val + f'

    for -delta_t < t' <= delta_t
    and where f' is the minimum amplitdude inflection in delta_t
    """

    STDEV_MULT = 4
    EVENT_PRE = 250
    EVENT_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000
    delta_t = 300

    def setup(self):
        FixedThresh.setup(self)
        self.f_inflect = {}
        # set f' to be 3.5 * standard deviation (see paper)
        for chan, val in self.std.iteritems():
            self.f_inflect[chan] = 3.5 * val

    def find(self):
        """Maintain state and search forward for a event"""

        # keep on sliding our search window forward to find events
        while True:

            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = np.where(numpy.abs(self.window.data[chan]) > thresh)[0]

                if len(_ev) <= 0:
                    continue

                thresh_vals = [(self.window.data[chan][ind], ind) \
                                    for ind in _ev.tolist()]

                # for each threshold value, scan forwrd in time delta_t
                # to see if an opposite threshold crossing occurred
                for val, ind in thresh_vals:

                    # scan forward to find local max or local min
                    extremal_ind = ind
                    extremal_val = val
                    #while True:
                    #    next_ind = extremal_ind + 1
                    #    next_val = self.window.data[chan][next_ind]
                    #    if abs(next_val) < abs(extremal_val):
                    #        break
                    #    extremal_val, extremal_ind = next_val, next_ind

                    # calculate our dynamic threshold
                    # TODO: make this more compact
                    if extremal_val < 0:
                        # a valley
                        dyn_thresh = extremal_val + self.f_inflect[chan]
                        dyn_events = np.where(self.window.data[chan] \
                                                        > dyn_thresh)[0]
                    else:
                        # a peak
                        dyn_thresh = extremal_val - self.f_inflect[chan]
                        dyn_events = np.where(self.window.data[chan] \
                                                        < dyn_thresh)[0]

                    dyn_vals = [(self.window.data[chan][_ind], _ind) \
                                    for _ind in dyn_events.tolist()]
                    t = self.window.ts[extremal_ind]
                    # check for next inflection
                    for dyn_val, t_ind in dyn_vals:
                        # check ahead only within +/- delta_t
                        t_prime = self.window.ts[t_ind]
                        if (t_prime > t - self.delta_t) and \
                                (t_prime <= t + self.delta_t):
                            break

                        event_val = extremal_val
                        event_ind = extremal_ind
                        if abs(dyn_val) > abs(extremal_val):
                            event_val = dyn_val
                            event_ind = t_ind
                        chan_events.append((event_ind, chan))
                        break

                # yield all the events we've found
                for evt in self.yield_events(chan_events):
                    yield evt

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]
