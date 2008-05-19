"""Spike detection algorithms


TODO: use median based noise estimation instead of std based
      - estimate noise level dynamically with sliding window
        and independently for each channel
TODO: spatiotemporal lockout:
      - rlock = 175um
      - lock out only first 1/2 phase of spike
      - phases are part of same spike if less than 250us between each other

TODO: Might need to use scipy.weave or some other low-level code
    (cython?) to make these algorithms fast enough, while allowing you to
    step through one timepoint at a time, which numpy might not let you do
    easily...

TODO: for speed, consider converting all uV data from 64bit float to 16 bit integer

TODO: add a method to search forward until you find the next spike on any or on a
      specific set of channels, as opposed to searching forward over a known fixed
      trange

"""

from __future__ import division

__authors__ = ['Reza Lotun']

import itertools

import spyke.surf
from spyke.stream import WaveForm
from spyke import Spike, Template, Collection

import numpy
from numpy import where


class Detector(object):
    """Spike detector base class"""
    DEFAULTTHRESHMETHOD = 'median'

    def __init__(self, stream=None, t0=None):
        """Takes a data stream, and optionally the time from which to start detection"""
        self.stream = stream
        if t0 == None:
            t0 = stream.t0
        self.t0 = t0
        self.setup()

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

    def self.get_median_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def self.get_stdev_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass


class FixedThresh(Detector):
    """Base class for fixed threshold spike detection,
    Uses the same single static threshold throughout the entire file,
    with an independent threshold for every channel"""

    STDEV_WINDOW = 10000000 # 10 sec
    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000

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


class DynamicThresh(Detector):
    """Base class for dynamic threshold spike detection,
    Uses varying thresholds throughout the entire file,
    depending on the local noise level

    Calculate noise level using, say, a 50ms sliding window centered on the
    timepoint you're currently testing for a spike


    """

    def self.get_median_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def self.get_stdev_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass


class BipolarAmplitudeFixedThresh(FixedThresh):
    """Bipolar amplitude threshold fixed threshold detector, with fixed lockout on all channels"""

    def search(self, trange=None):
        """Manage combining search results from individual multichannel chunks of waveform data"""
        if trange == None:
            trange = (stream.t0, stream.tend)
        wavetranges = # slightly overlapping chunks of 1sec each or so
        for wavetrange in wavetranges:
            wave = self.stream[wavetrange] # chunk is just a waveform
            eventis = self.searchwave(wave)
            compare to previous chunk in overlap area, make sure not counting spikes twice in overlap

    def Csearchwave(self, abschan, thresh, tilockout):
        """Maybe limit this to a single chan, deal with spatial lockout in python,
        do peak searching and temporal lockout on a single chan basis in C. Or, maybe use
        C for both chan and ti loop, but leave spatial lockout to a later step (which itself
        could be done in another C weave f'n)"""
        code = r"""
        #line xx "detect.py" (this is only useful for debugging)
        int ti=0 // current time index
        int last=0;
        int nspikes=0;
        int array spikeis[0..len(abschan)/2] // can't have more than one max every other timepoint
        tilockout-- // we will always skip ahead from ti+1, so dec tilockout by 1
        for (int ti=0; ti<nt; ) {
            if abschan[ti] >= thresh:
                if abschan[ti] > last: // signal is still increasing
                    last = absdata[chani*ti]
                    ti++ // go to next timepoint
                else: // signal is decreasing, save last timepoint as spike
                    nspikes++
                    spikeis[nspikes-1] = ti-1
                    ti += tilockout
            else:
                ti++ // go to next timepoint

        }
        return_val = 666;
        """
        return  weave.inline(code,
                           [],
                           type_converters=converters.blitz,
                           compiler='msvc')

    def searchwave(self, wave, thresh):
        """Look for thresh xings, find peak indices, do lockouts, return event indices"""
        absdata = np.abs(wave.data)
        threshxis = absdata > thresh # bipolar thresh xing indices for all chans
        # find peak indices. this is quite wasteful since we already know where the peaks mostly aren't.
        # Maybe use a sparse matrix here to prevent having to search through all the zeros?
        peakis = np.diff(absdata*threshxis)
        # now do spatial and temporal lockouts

        # problem: can't load all data in at once into a single big array to search through




    def find(self):
        """Maintain state and search forward for a spike"""
        # keep on sliding our search window forward to find spikes
        while True:
            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = where(numpy.abs(self.window.data[chan]) > thresh)[0]
                if len(_ev) > 0:
                    # scan forward to find local max
                    #                    x  <----  want to find this
                    #  have inds ---{ x     x
                    #      ---------x----------------  threshold
                    thresh_vals = [(abs(self.window.data[chan][ind]), ind) \
                                        for ind in _ev.tolist()]
                    max_val, max_ind = max(thresh_vals)
                    chan_events.append((max_ind, chan))

                for evt in self.yield_events(chan_events):
                    yield evt

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]


class MultiPhasic(FixedThreshold):
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


class DynamicMultiPhasic(FixedThreshold):
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
