"""Spike parameter extraction"""

from __future__ import division

__authors__ = ['Martin Spacek']

import time

import numpy as np

from spyke.detect import get_wave


class Extractor(object):
    """Spike extractor base class"""
    #DEFXYMETHOD = 'spatial mean'
    def __init__(self, sort, XYmethod):
        """Takes a parent Sort session and sets various parameters"""
        self.sort = sort
        self.XYmethod = XYmethod # or DEFXYMETHOD
        self.choose_XY_fun()

    def choose_XY_fun(self):
        if self.XYmethod.lower() == 'spatial mean':
            self.extractXY = self.get_spatial_mean
        elif self.XYmethod.lower() == 'gaussian fit':
            self.extractXY = self.get_gaussian_fit
        else:
            raise ValueError("Unknown XY parameter extraction method %r" % method)

    def __getstate__(self):
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['extractXY'] # can't pickle an instance method, not sure why it even bothers trying
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.choose_XY_fun() # restore instance method

    def extract(self):
        """Extract spike parameters, store them as spike attribs. Every time
        you do a new extraction, (re)create a new .params recarray with the right
        set of params in it"""
        spikes = self.sort.spikes # recarray
        if len(spikes) == 0:
            raise RuntimeError("No spikes to extract XY parameters from")
        print("Extracting parameters from spikes")
        t0 = time.clock()
        for s in spikes:
            wave = get_wave(s, self.sort.stream)
            wavedata = wave.data
            det = s.detection.detector
            x = det.siteloc[s.chanis, 0] # 1D array (row)
            y = det.siteloc[s.chanis, 1]
            # just x and y params for now
            s.x0, s.y0 = self.extractXY(wavedata, x, y, s.phase1ti, s.phase2ti) # save as spike attribs
        print("Extracting parameters from all %d spikes using %r took %.3f sec" %
              (len(spikes), self.XYmethod.lower(), time.clock()-t0))

    def get_spatial_mean(self, wavedata, x, y, phase1ti, phase2ti):
        """Return weighted spatial mean of chans in spike according to their
        Vpp at the same timepoints as on the max chan, to use as rough
        spatial origin of spike. x and y are spatial coords of chans in wavedata.
        phase1ti and phase2ti are timepoint indices in wavedata at which the max chan
        hits its 1st and 2nd spike phases.
        TODO: maybe you get better clustering if you allow phase1ti and phase2ti to
        vary at least slightly for each chan, since they're never simultaneous across
        chans, and sometimes they're very delayed or advanced in time. Maybe just try finding
        max and min vals for each chan in some trange phase1ti-dt to phase2ti+dt for some dt
        NOTE: sometimes neighbouring chans have inverted polarity, see ptc15.87.50880, 68840"""

        # phase2 - phase1 on all chans, should be mostly +ve
        # int16 data isn't centered around V=0, but that doesn't matter since we want Vpp
        weights = wavedata[:, phase2ti] - wavedata[:, phase1ti]
        # convert to float before normalization, take abs of all weights
        weights = np.abs(np.float32(weights))
        weights /= weights.sum() # normalized
        # alternative approach: replace -ve weights with 0
        #weights = np.float32(np.where(weights >= 0, weights, 0))
        #try: weights /= weights.sum() # normalized
        #except FloatingPointError: pass # weird all -ve weights spike, replaced with 0s
        x0 = (weights * x).sum()
        y0 = (weights * y).sum()
        return x0, y0

    def get_gaussian_fit(self, spike, wavedata=None, siteloc=None):
        raise NotImplementedError
