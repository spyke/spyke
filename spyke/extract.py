"""Spike parameter extraction"""

from __future__ import division

__authors__ = ['Martin Spacek']

import time

import numpy as np

from spyke.detect import update_wave


class Extractor(object):
    """Spike extractor base class"""
    #DEFXYMETHOD = 'spatial mean'
    def __init__(self, sort, XYmethod=None):
        """Takes a parent Sort session and sets various parameters"""
        self.sort = sort
        self.XYmethod = XYmethod # or DEFXYMETHOD

    def extract(self):
        """Extract spike parameters, store them as spike attribs. Every time
        you do a new extraction, (re)create a new .params recarray with the right
        set of params in it"""
        self.extractXY() # just x and y params for now

    def extractXY(self):
        """Extract XY parameters from spikes using XYmethod"""
        spikes = self.sort.spikes # recarray
        method = self.XYmethod
        if len(spikes) == 0:
            raise RuntimeError("No spikes to extract XY parameters from")
        print("Extracting parameters from spikes")
        t0 = time.clock()
        if method.lower() == 'spatial mean':
            f = self.get_spike_spatial_mean
        elif method.lower() == 'gaussian fit':
            f = self.get_gaussian_fit
        else:
            raise ValueError("Unknown XY parameter extraction method %r" % method)
        for spike in spikes:
            spike.x0, spike.y0 = f(spike) # save as spike attribs
        print("Extracting XY parameters from all %d spikes using %r took %.3f sec" %
              (len(spikes), method.lower(), time.clock()-t0))

    def get_spike_spatial_mean(self, spike):
        """Return weighted spatial mean of chans in spike according to their
        Vpp, to use as rough spatial origin of spike
        NOTE: sometimes neighbouring chans have inverted polarity, see ptc15.87.50880, 68840
        This is handled by giving them 0 weight. Maybe it's better to take abs instead?"""
        chanis = spike.chanis
        siteloc = spike.detection.detector.siteloc
        wavedata = spike.wavedata
        if wavedata == None:
            update_wave(spike, self.sort.stream)
            wavedata = spike.wave.data
        x = siteloc[chanis, 0] # 1D array (row)
        y = siteloc[chanis, 1]
        # phase2 - phase1 on all chans, should be +ve, at least on maxchan
        weights = (wavedata[:, spike.phase2ti] -
                   wavedata[:, spike.phase1ti])
        # replace any -ve weights with 0, convert to float before normalization
        weights = np.float32(np.where(weights >= 0, weights, 0))
        weights /= weights.sum() # normalized
        #weights = wave.data[spike.chanis, spike.ti] # Vp weights, unnormalized, some of these may be -ve
        # not sure if this is a valid thing to do, maybe just take abs instead, like when spike inverts across space
        #weights = np.where(weights >= 0, weights, 0) # replace -ve weights with 0
        #weights = abs(weights)
        x0 = (weights * x).sum()
        y0 = (weights * y).sum()
        return x0, y0

    def get_gaussian_fit(self, spike):
        raise NotImplementedError
