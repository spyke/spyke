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
        return hash(str(self.channel) + str(self.event_time))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Template(set):
    """ A spike template is simply a collection of spikes. """
    def __init__(self, *args):
        set.__init__(self, *args)
        self.name = str(self)

    def mean():
        if len(self) > 0:
            pass

        return None

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


class SimpleThreshold(Detector):
    """ Bipolar amplitude threshold, with fixed lockout on all channels."""

    THRESH_MULT = 10
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1e3
    LOCKOUT = 1e3

    def setup(self):
        """ Used to determine threshold and set initial state. """

        # get the stdeviation for each channel along a 10s window
        #ten_seconds = 1e7
        ten_seconds = 1e3
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

    def find(self):
        # maintain state and search forward for a spike

        # keep on sliding our search window forward to find spikes
        while True:

            # check if we have an channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = where(numpy.abs(self.window.data[chan]) > thresh)[0]
                if len(_ev) > 0:
                    # scan forward to find local max
                    #                    x  <----  want to find this
                    #  have inds ---{ x     x
                    #      ---------x----------------  threshold
                    #ev_inds = [(ind, chan) for ind in _ev.tolist()]
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

