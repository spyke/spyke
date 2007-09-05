""" Spike detection algorithms """

__authors__ = ['Reza Lotun']

import spyke.surf
from spyke.stream import WaveForm

from numpy import where

def bipolar_amplitude():
    pass

def multiphasic_filter():
    pass

def dynamic_multiphasic_filter():
    pass

def nonlinear_energy_operator():
    pass

def hyper_ellipsoidal():
    pass

class Spike(WaveForm):
    """ A spike event """
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
        while True:
            try:
                yield self.find()
            except:         # XXX: catch something specific
                break

    def find(self):
        pass

class SimpleThreshold(Detector):
    """ Bipolar amplitude threshold, with fixed lockout on all channels."""

    def setup(self):
        """ Used to determine threshold and set initial state. """

        # get the stdeviation for each channel along a 10s window
        ten_seconds = 1e7
        chunk = self.stream[self.init_time:self.init_time + ten_seconds]
        self.std = {}
        for chan, d in enumerate(chunk):
            self.std[chan] = window[chan].std()

        # set the threshold to be twice standard deviation
        self.thresholds = {}
        for chan, val in self.std.iteritems():
            self.thresholds[chan] = val * 2

        # spike window: -0.25ms and +0.75ms around spike
        # our search window will be 1ms
        self.search_span = 1e3
        self.window = self.stream[self.init:self.init + self.search_span]

    def find(self):
        # maintain state and search forward for a spike

        # keep on sliding our large search window forward to find spikes
        # 
        while True:
            res = numpy.where(



            
        
        

