""" Spike detection algorithms """

__authors__ = ['Reza Lotun']

import spyke.surf
from spyke.stream import WaveForm

import numpy
from numpy import where

class Spike(WaveForm):
    """ A spike event """
    pass

class Template(set):
    """ A collection of Spikes represent a template. """
    pass

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

    SPIKE_PRE = 250
    SPIKE_POST = 750

    def setup(self):
        """ Used to determine threshold and set initial state. """

        # get the stdeviation for each channel along a 10s window
        ten_seconds = 1e7
        chunk = self.stream[self.init_time:self.init_time + ten_seconds]
        self.std = {}
        for chan, d in enumerate(chunk.data):
            self.std[chan] = chunk.data[chan].std()

        # set the threshold to be twice standard deviation
        self.thresholds = {}
        for chan, val in self.std.iteritems():
            self.thresholds[chan] = val * 4

        # spike window: -0.25ms and +0.75ms around spike
        # our search window will be 1ms
        self.search_span = 1e3
        self.curr = self.init_time + 1e3 # XXX: add an initial jump
        self.window = self.stream[self.init_time:self.init_time + \
                                                        self.search_span]

        self.lockout = 1e3

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
                    chan_events.extend(_ev.tolist())

            # remove duplicates and sort
            chan_events = list(set(chan_events))
            chan_events.sort()

            for event_index in chan_events:

                # if the event is firing before our current location
                # then we're in lockout mode and should just continue
                if self.window.ts[event_index] < self.curr:
                    continue

                # reposition window for each event
                self.curr = self.window.ts[event_index] - self.SPIKE_PRE
                spike = self.stream[self.curr:self.curr + 1e3]
                self.curr = self.curr + 1e3 + self.lockout
                self.window = self.stream[self.curr:self.curr + \
                                                    self.search_span]
                yield spike

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]

