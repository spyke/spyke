"""Spike detection algorithms

TODO: use median based noise estimation instead of std based
      - estimate noise level dynamically with sliding window
        and independently for each channel
TODO: spatiotemporal lockout:
      - do spatial lock out only during first 1/2 phase of trigger spike
      - phases are part of same spike if less than 250us between each other

TODO: for speed, consider converting all uV data from 64bit float to 16 bit integer

DONE: Might need to use scipy.weave or some other low-level code
     (cython?) to make these algorithms fast enough, while allowing you to
     step through one timepoint at a time, which numpy might not let you do
     easily...
DONE: add a method to search forward until you find the next spike on any or on a
      specific set of channels, as opposed to searching forward over a known fixed
      trange. This would make testing easier too
"""

from __future__ import division

__authors__ = ['Reza Lotun, Martin Spacek']

import itertools
import sys

import numpy as np
from numpy import where
from scipy import weave

import spyke.surf
from spyke.core import WaveForm, toiter, cut, intround
#from spyke import Spike, Template, Collection



class Detector(object):
    """Spike detector base class"""
    DEFAULTTHRESHMETHOD = 'median'
    DEFTLOCK = 250 # us
    DEFSLOCK = 175 # um

    MAXAVGFIRINGRATE = 1000 # Hz, assume no chan will trigger more than this rate of events on average within a block. TODO: should be a property
    BLOCKSIZE = 10000000 # waveform data block size, us. TODO: should be a property
    MAXNSPIKETIS = BLOCKSIZE/1000000 * MAXAVGFIRINGRATE # length of array to preallocate before searching a block's channel
    BLOCKEXCESS = 1000 # us, extra data as buffer at start and end of a block while searching for spikes. Only useful for ensuring spike times within the actual block time range are accurate. Spikes detected in the excess are discarded


    def __init__(self, stream, chans=None, trange=None, maxnspikes=None,
                 tlock=None, slock=None):
        """Takes a data stream and sets various parameters"""
        self.stream = stream
        if chans == None:
            chans = range(self.stream.nchans) # search all channels
        self.chans = toiter(chans)
        self.nchans = len(self.chans)
        if trange == None:
            trange = (stream.t0, stream.tend)
        self.trange = trange
        if maxnspikes == None:
            maxnspikes = sys.maxint
        self.maxnspikes = maxnspikes # return at most this many spikes, applies across chans
        if tlock == None:
            tlock = self.DEFTLOCK
        self.tlock = tlock
        if slock == None:
            slock = self.DEFSLOCK
        self.slock = slock

    '''
    def __iter__(self):
        """Returns an iterator object. Called in for loops and in 'in' statements, and by the iter() f'n?.
        This is here to allow you to treat any of the detection classes directly as iterators"""
        spikes = iter(self.find())
        # why not just return the iterator, and let the outside world iterate over it, like this:?
        return spikes
        #while True:
        #    try:
        #        yield spikes.next()
        #    except StopIteration:
        #        break
    '''
    def get_threshold(self, chan, kind=DEFAULTTHRESHMETHOD):
        """Calculate either median or stdev based threshold for a given chan"""
        if kind == 'median':
            self.get_median_threshold(chan)
        elif  kind == 'stdev':
            self.get_stdev_threshold(chan)

    def get_median_threshold(self, chan):
        return self.get_median_noise(chan) * self.MEDIAN_MULT

    def get_stdev_threshold(self, chan):
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
    """Base class for fixed threshold spike detection,
    Uses the same single static threshold throughout the entire file,
    with an independent threshold for every channel"""

    STDEV_WINDOW = 10000000 # 10 sec
    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    #SEARCH_SPAN = 1000

    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.thresh = 50 # uV, TODO: calculate this from noise level

    '''
    def setup(self):
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

        # spike window: -SPIKE_PRE and +SPIKE_POST around spike, search window will be 1ms
        self.search_span = self.SEARCH_SPAN
        self.curr = self.t0 + self.SEARCH_SPAN # XXX: add an initial jump: TODO: why?
        self.window = self.stream[self.t0:self.t0 + self.search_span]

        self.lockout = self.LOCKOUT

    def yield_events(self, chan_events):
        """TODO: what does this do? need description here"""
        # sort event indices
        chan_events.sort()
        for event_index, chan in chan_events:
            # if the event is firing before our current location
            # then we're in lockout mode and should just continue
            if self.window.ts[event_index] < self.curr:
                continue
            # reposition window for each event
            self.curr = self.window.ts[event_index] - self.SPIKE_PRE
            spike = self.stream[self.curr:self.curr + \
                                self.SPIKE_PRE + self.SPIKE_POST]
            self.curr = self.curr + self.SPIKE_PRE + \
                            self.SPIKE_POST + self.lockout
            #self.window = self.stream[self.curr:self.curr + self.search_span]
            yield Spike(spike, chan, self.window.ts[event_index])
    '''

class DynamicThresh(Detector):
    """Base class for dynamic threshold spike detection,
    Uses varying thresholds throughout the entire file,
    depending on the local noise level

    Calculate noise level using, say, a 50ms sliding window centered on the
    timepoint you're currently testing for a spike. Or, use fixed pos
    windows, pre calc noise for each of them, and take noise level from whichever
    window you happen to be in while checking a timepoint for thresh xing.
    """

    def get_median_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def get_stdev_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass


class BipolarAmplitudeFixedThresh(FixedThresh):
    """Bipolar amplitude fixed threshold detector,
    with fixed temporal lockout on all channels, plus a spatial lockout"""

    def search(self, method='C'):
        """Search for spikes. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results"""
        # reset this at the start of every search
        self.totalnspikes = 0 # total num spikes found across all chans so far by this Detector

        # holds a channel's spike times, passed by assignment to C code.
        # no need for more than one max every other timepoint, can get away with less to save memory.
        # recordings not likely to have more than 2**32 timestamps, even when interpolated to 50 kHz,
        # so uint32 allows us at least 23 hour long recordings, don't think int64 is needed here
        if method == 'C':
            self.spiketis = np.zeros((self.nchans, self.MAXNSPIKETIS), dtype=np.uint32) # use this for .Csearchblock()
            searchblockfn = self.Csearchblock
        else:
            self.spiketis = np.zeros(self.MAXNSPIKETIS, dtype=np.uint32) # use this for .searchblock()
            searchblockfn = self.searchblock
        self.tilock = self.us2nt(self.tlock) # TODO: this should be a property, or maybe tlock should

        # generate time ranges for slightly overlapping blocks of data
        wavetranges = []
        cutranges = []
        BS = self.BLOCKSIZE
        BX = self.BLOCKEXCESS
        les = range(self.trange[0], self.trange[1], BS)  # left edges of data blocks
        for le in les:
            wavetranges.append((le-BX, le+BS+BX)) # time range of waveform to give to .searchblock
            cutranges.append((le, le+BS)) # time range of spikes to keep from those returned by .searchblock
        # last wavetrange and cutrange surpass self.trange[1], fix that here:
        wavetranges[-1] = (wavetranges[-1][0], self.trange[1]+BX)
        cutranges[-1] = (cutranges[-1][0], self.trange[1])

        spikes = {} # dict of arrays of spike times, one entry per chan
        for (lo, hi), cutrange in zip(wavetranges, cutranges): # iterate over blocks
            #print 'iterate block'
            if self.totalnspikes < self.maxnspikes:
                wave = self.stream[lo:hi] # a block (Waveform) of multichan data
                tsdict = searchblockfn(wave)
                for chan, ts in tsdict.iteritems(): # iterate over channels
                    cts = cut(ts, cutrange) # remove excess
                    try:
                        # add cut spikes for this chan. TODO: maybe append to list instead of array?
                        spikes[chan] = np.append(spikes[chan], cts)
                    except KeyError: # on first loop, spikes is an empty dict
                        spikes[chan] = cts
        return spikes

    def Csearchblock(self, wave):
        """Search across all chans in a manageable block of waveform
        data and return a dict of arrays of spike times, one entry per chan.
        Apply both temporal and spatial lockouts"""
        absdata = np.abs(wave.data) # TODO: try doing this in the C loop, see if it's faster
        nchans = self.nchans
        nt = wave.data.shape[1]
        nspikes = np.zeros(self.nchans, dtype=np.uint32) # holds nspikes found for each chan in this block
        totalnspikes = self.totalnspikes # total num of spikes found so far across all chans by this Detector
        maxnspikes = self.maxnspikes
        thresh = self.thresh
        # holds flags that indicate if a chan has recently crossed threshold, and is still searching for that spike's peak
        xthresh = np.zeros(nchans, dtype=np.int32)
        last = np.zeros(nchans) # holds last signal value per chan, floats in uV
        lock = np.zeros(nchans, dtype=np.int32) # holds number of lockout timepoints left per chan
        tilock = self.tilock
        #slock = self.slock
        spiketis = self.spiketis # init'd in self.search()

        code = r"""
        #line 262 "detect.py" // for debugging
        int nnewspikes=0; // num new spikes found in this f'n
        int ti=0; // current time index
        int chan=0; // current chan index
        int spikei=0; // current spike index for current chan
        double v; // current signal voltage, uV
        for ( ti=0; ti<nt && totalnspikes<maxnspikes; ti++ ) {
            // chan loop should go in random order, not numerical, to prevent a chan from
            // dominating with its spatial lockout. Also, there might be missing channels
            for ( chan=0; chan<nchans; chan++ ) {
                if ( lock(chan) > 0 ) // if this chan is still locked out
                    lock(chan)--; // decr this chan's temporal lockout
                else { // search for a thresh xing or a peak
                    v = absdata(chan, ti); // TODO: v should be a pointer to prevent this copy operation?
                    if ( xthresh(chan) == 0 ) { // we're looking for a thresh xing
                        if ( v >= thresh && xthresh(chan) == 0 ) { // met or exceeded threshold and
                                                                   // xthresh flag hasn't been set yet
                                                                   // TODO: check if >= is slower than >
                            xthresh(chan) = 1; // set crossed threshold flag for this chan
                            last(chan) = v; // update last value for this chan, have to wait til
                                            // next ti to decide if this is a peak
                        }
                    }
                    else { // we've found a thresh xing, now we're look for a peak
                        if ( v > last(chan) ) // if signal is still increasing
                            last(chan) = v; // update last value for this chan, have to wait til
                                            // next ti to decide if this is a peak
                        else { // signal is decreasing, save previous ti as spike
                            spikei = nspikes(chan)++; // 0-based spike index. assign, then increment
                            totalnspikes++;
                            nnewspikes++;
                            spiketis(chan, spikei) = ti-1; // save previous time index as that of the nth spike for this chan
                            xthresh(chan) = 0; // we've found the peak, clear crossed thresh flag for this chan
                            last(chan) = 0.0; // reset for search for next spike
                            lock(chan) = tilock; // apply temporal lockout for this chan
                            //lock(chan and nearby chans) = tilock; // set temporal lockout for this chan and all chans
                                                                  // within slock distance of it
                        }
                    }
                }
            }
        }
        return_val = nnewspikes;
        """
        nnewspikes = weave.inline(code, ['absdata', 'nchans', 'nt', 'nspikes', 'totalnspikes', 'maxnspikes',
                                         'thresh', 'xthresh', 'last', 'lock', 'tilock', 'spiketis'],
                                  type_converters=weave.converters.blitz,
                                  compiler='gcc')
        self.totalnspikes += nnewspikes # update
        spikes = {}
        for chan in range(nchans): # C loop assumes consecutive chans, so let's do the same here
            tis = spiketis[chan, :nspikes[chan]] # keep only the entries that were filled for this chan
            spikes[chan] = wave.ts[tis] # spike times in us
        return spikes

    def searchblock(self, wave):
        """Search across all chans in a manageable block of waveform
        data and return a dict of arrays of spike times, one entry per chan"""
        spikes = {}
        for chan in self.chans:
            if self.totalnspikes < self.maxnspikes:
                abschan = np.abs(wave[chan])
                tis = self.searchchan(abschan)
                spikes[chan] = wave.ts[tis] # spike times in us
        # TODO: apply spatial lockout here. Would have to iterate over all
        # timepoints again, which would be slow
        return spikes

    def searchchan(self, abschan):
        """Search a single chan of absval data for thresh xings, apply temporal lockout.
        If this crashes, it might be possible that self.spiketis was init'd too small"""
        nt = len(abschan)
        totalnspikes = self.totalnspikes # total num of spikes found by this Detector so far
        maxnspikes = self.maxnspikes
        thresh = self.thresh
        tilock = self.tilock
        spiketis = self.spiketis # init'd in self.search()

        code = r"""
        #line 341 "detect.py" // for debugging
        double last=0.0; // last signal value, uV
        int nnewspikes=0; // num new spikes found in this f'n
        int ti=0; // current time index
        while ( ti<nt && totalnspikes < maxnspikes ) { // enforce maxnspikes across single chan
            if ( abschan(ti) >= thresh ) { // if we've exceeded threshold
                while ( abschan(ti) > last ) { // while signal is still increasing
                    last = abschan(ti); // update last
                    ti++; // go to next timepoint
                }
                // now signal is decreasing, save last timepoint as spike
                totalnspikes++;
                nnewspikes++;
                spiketis(nnewspikes-1) = ti-1; // 0-based index into spiketis
                last = 0.0; // reset for search for next spike
                ti += tilock+1; // skip forward one temporal lockout, and go to next timepoint
            }
            else
                ti++; // no thresh xing, go to next timepoint
        }
        return_val = nnewspikes;
        """
        nnewspikes = weave.inline(code, ['abschan', 'nt', 'totalnspikes', 'maxnspikes', 'thresh', 'tilock', 'spiketis'],
                                  type_converters=weave.converters.blitz,
                                  compiler='gcc')
        self.totalnspikes += nnewspikes # update
        return spiketis[:nnewspikes]


class MultiPhasic(FixedThresh):
    """Multiphasic filter - spikes triggered only when consecutive
    thresholds of opposite polarity occur on a given channel within
    a specified time window delta_t

    That is, either:

        1) s_i(t) > f and s_i(t + t') < -f
        2) s_i(t) < -f and s_it(t + t') > f

    for 0 < t' <= delta_t
    """

    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000
    delta_t = 300

    def find(self):
        """Maintain state and search forward for a spike"""

        # keep on sliding our search window forward to find spikes
        while True:

            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = where(numpy.abs(self.window.data[chan]) > thresh)[0]

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
    """Dynamic Multiphasic filter - spikes triggered only when consecutive
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
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000
    delta_t = 300

    def setup(self):
        FixedThreshold.setup(self)
        self.f_inflect = {}
        # set f' to be 3.5 * standard deviation (see paper)
        for chan, val in self.std.iteritems():
            self.f_inflect[chan] = 3.5 * val

    def find(self):
        """Maintain state and search forward for a spike"""

        # keep on sliding our search window forward to find spikes
        while True:

            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = where(numpy.abs(self.window.data[chan]) > thresh)[0]

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
                        dyn_events = where(self.window.data[chan] \
                                                        > dyn_thresh)[0]
                    else:
                        # a peak
                        dyn_thresh = extremal_val - self.f_inflect[chan]
                        dyn_events = where(self.window.data[chan] \
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
