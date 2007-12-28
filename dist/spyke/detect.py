from __future__ import division

""" Spike detection algorithms """

__authors__ = ['Reza Lotun']

import itertools

import spyke.surf
from spyke.stream import WaveForm

import numpy
from numpy import where

class Spike(WaveForm):
    """ A spike event """
    def __init__(self, waveform=None, channel=None, event_time=None):
        self.data = waveform.data
        self.ts = waveform.ts
        self.sampfreq = waveform.sampfreq
        self.channel = channel
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


class Template(set):
    """ A spike template is simply a collection of spikes. """
    def __init__(self, *args):
        set.__init__(self, *args)
        self.name = str(self)
        self.active_channels = None

    def mean(self):
        """ Returns the mean of all the contained spikes. """
        if len(self) == 0:
            return None

        sample = iter(self).next()
        dim = sample.data.shape
        _mean = Spike(sample)
        _mean.data = numpy.asarray([0.0] * dim[0] * dim[1]).reshape(dim)

        for num, spike in enumerate(self):
            _mean.data += spike.data

        _mean.data = _mean.data / (num + 1)

        return _mean

    def __hash__(self):
        # XXX hmmm how probable would collisions be using this...?
        return hash(str(self.mean()) + str(self))

    def __str__(self):
        return 'Template (' + str(len(self)) + ')'


class Collection(object):
    """ A container for Templates. Collections are associated with
    Surf Files. By default a Collection represents a single sorting session.
    Initially detected spikes will be added to a default set of spikes in a
    collection - these spikes will be differentiated through a combination
    of algorithmic and/or manual sorting.
    """
    def __init__(self, file=None):
        # XXX: populate this with pertinent info
        self.templates = []
        self.unsorted_spikes = []        # these represent unsorted spikes
        self.recycle_bin = []

    def __len__(self):
        return len(self.templates)

    def __str__(self):
        """ Pretty print the contents of the Collection."""
        s = []
        for t in self:
            s.extend([str(t), '\n'])
            for sp in t:
                s.extend(['\t', str(sp), '\n'])
        return ''.join(s)

    def __iter__(self):
        for template in self.templates:
            yield template


class Detector(object):
    """ Spike detection superclass. """
    def __init__(self, stream, init_time):
        self.stream = stream
        self.init_time = init_time
        self.setup()

    def setup():
        pass

    def __iter__(self):
        spikes = iter(self.find())
        while True:
            try:
                yield spikes.next()
            except:         # XXX: catch something specific
                break

    def find(self):
        pass


class FixedThreshold(Detector):
    THRESH_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3

    def setup(self):
        """ Used to determine threshold and set initial state. """
        # get the stdeviation for each channel along a 10s window
        ten_seconds = 1e7
        chunk = self.stream[self.init_time:self.init_time + ten_seconds]
        self.std = {}
        for chan, d in enumerate(chunk.data):
            self.std[chan] = chunk.data[chan].std()

        # set the threshold to be THRESH_MULT * standard deviation
        self.thresholds = {}
        for chan, val in self.std.iteritems():
            self.thresholds[chan] = val * self.THRESH_MULT

        # spike window: -0.25ms and +0.75ms around spike
        # our search window will be 1ms
        self.search_span = self.SEARCH_SPAN
        self.curr = self.init_time + self.SEARCH_SPAN # XXX: add an initial jump
        self.window = self.stream[self.init_time:self.init_time + \
                                                        self.search_span]

        self.lockout = self.LOCKOUT


class MultiPhasic(FixedThreshold):
    """ Multiphasic filter - spikes triggered only when consecutive
    thresholds of opposite polarity occured on a given channel within
    a specified time window delta_t

    That is, either:

        1) s_i(t) > f and s_i(t + t') < -f
        2) s_i(t) < -f and s_it(t + t') > f

    for 0 < t' <= delta_t
    """

    THRESH_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3
    delta_t = 1e2

    def find(self):
        # maintain state and search forward for a spike

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

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]


class DynamicMultiPhasic(FixedThreshold):
    """ Dynamic Multiphasic filter - spikes triggered only when consecutive
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

    THRESH_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3
    delta_t = 1e2
    f_inflect = None

    def setup(self):
        FixedThreshold.setup(self)
        self.f_inflect = {}
        # set f' to be 3.5 * standard deviation (see paper)
        for chan, val in self.std.iteritems():
            self.f_inflect[chan] = 3.5 * self.THRESH_MULT

    def find(self):
        # maintain state and search forward for a spike

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
                    extremal_val = self.window.data[chan][extremal_ind]
                    while True:
                        next_ind = extremal_ind + 1
                        next_val = self.window.data[chan][next_ind]
                        if abs(next_val) < abs(extremal_val):
                            break
                        extremal_val, extremal_ind = next_val, next_ind

                    print 'extreme: ', extremal_val, extremal_ind

                    # calculate our dynamic threshold
                    # TODO: make this more compact
                    if extremal_val < val:
                        # a valley
                        dyn_thresh = extremal_val + self.f_inflect[chan]
                        dyn_events = where(self.window.data[chan] \
                                                        > dyn_thresh)[0]
                        dyn_vals = [(self.window.data[chan][_ind], _ind) \
                                        for _ind in dyn_events.tolist()]
                        print dyn_vals

                    else:
                        # a peak
                        dyn_thresh = extremal_val - self.f_inflect[chan]
                        dyn_events = where(self.window.data[chan] \
                                                        < dyn_thresh)[0]
                        dyn_vals = [(self.window.data[chan][_ind], _ind) \
                                        for _ind in dyn_events.tolist()]
                        print dyn_vals


                    t = self.window.ts[ind]
                    # check for next inflection
                    for dyn_val, t_ind in dyn_vals:
                        # check ahead only within +/- delt_t
                        t_prime = self.window.ts[t_ind]
                        if abs(t_prime - t) > self.delta_t:
                            break

                        chan_events.append((extremal_ind, chan))
                        print 'events: ', chan_events
                        break

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

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]


class SimpleThreshold(FixedThreshold):
    """ Bipolar amplitude threshold, with fixed lockout on all channels."""

    def find(self):
        # maintain state and search forward for a spike

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

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]

