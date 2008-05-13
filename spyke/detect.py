"""Spike detection algorithms"""

from __future__ import division

__authors__ = ['Reza Lotun']

import itertools

import spyke.surf
from spyke.stream import WaveForm
from spyke import Spike, Template, Collection

import numpy
from numpy import where


class Detector(object):
    """Spike detection base class"""
    def __init__(self, stream, t0):
        self.stream = stream
        self.t0 = t0
        self.setup()

    def setup():
        pass

    def __iter__(self):
        spikes = iter(self.find())
        while True:
            try:
                yield spikes.next()
            except:         # TODO: catch something specific
                break

    def find(self):
        pass


class FixedThreshold(Detector):
    STDEV_WINDOW = 1e7
    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3

    def setup(self):
        """Used to determine threshold and set initial state"""
        # get stdev for each channel along a STDEV_WINDOW window
        wave = self.stream[self.t0:self.t0 + STDEV_WINDOW]
        self.std = {}
        for chan, d in enumerate(wave.data):
            self.std[chan] = wave.data[chan].std()

        # set the threshold to be STDEV_MULT * standard deviation
        self.thresholds = {}
        for chan, val in self.std.iteritems():
            self.thresholds[chan] = val * self.STDEV_MULT

        # spike window: -0.25ms and +0.75ms around spike, search window will be 1ms
        self.search_span = self.SEARCH_SPAN
        self.curr = self.t0 + self.SEARCH_SPAN # XXX: add an initial jump
        self.window = self.stream[self.t0:self.t0 + self.search_span]

        self.lockout = self.LOCKOUT

    def yield_events(self, chan_events):
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
            #self.window = self.stream[self.curr:self.curr + \
            #                                    self.search_span]
            yield Spike(spike, chan, self.window.ts[event_index])


class SimpleThreshold(FixedThreshold):
    """Bipolar amplitude threshold, with fixed lockout on all channels"""

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
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3
    delta_t = 3e2

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
        2) s_i(t) < -f and s_it(t + t') > f_val + f'

    for -delta_t < t' <= delta_t
    and where f' is the minimum amplitdude inflection in delta_t
    """

    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3
    delta_t = 3e2

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
