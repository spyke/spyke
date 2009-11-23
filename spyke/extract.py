"""Spike parameter extraction"""

from __future__ import division

__authors__ = ['Martin Spacek']

import time

import numpy as np
from scipy.optimize import leastsq

from spyke.detect import get_wave
from spyke.core import g2

class LeastSquares(object):
    """Least squares Levenberg-Marquardt spatial gaussian fit of decay across chans"""
    def calc(self, x, y, V):
        t0 = time.clock()
        result = leastsq(self.cost, self.p0, args=(x, y, V), full_output=True)
                         #Dfun=None, full_output=True, col_deriv=False,
                         #maxfev=50, xtol=0.0001,
                         #diag=None)
        print('iters took %.3f sec' % (time.clock()-t0))
        self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        print('p0 = %r' % self.p0)
        print('p = %r' % self.p)
        print('%d iterations' % self.infodict['nfev'])
        print('mesg=%r, ier=%r' % (self.mesg, self.ier))

    def model(self, p, x, y):
        """2D Gaussian"""
        try: return p[0] * g2(p[1], p[2], p[3], p[3], x, y) # circularly symmetric Gaussian
        except Exception as err:
            print(err)
            import pdb; pdb.set_trace()

    def cost(self, p, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        return self.model(p, x, y) - V




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
            self.ls = LeastSquares()
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
        """Extract spike parameters, store them as spike attribs.
        TODO?: Every time you do a new extraction, (re)create a new
        .params struct array with the right set of params in it - not
        sure what I meant by this"""
        sort = self.sort
        spikes = sort.spikes # struct array
        nspikes = len(spikes)
        if nspikes == 0:
            raise RuntimeError("No spikes to extract XY parameters from")
        try: sort.wavedatas
        except AttributeError:
            raise RuntimeError("Sort has no saved wavedata in memory to extract parameters from")
        twi0 = -sort.twi[0] # num points from tref backwards to first timepoint in window
        print("Extracting parameters from spikes")
        t0 = time.time()
        for ri in np.arange(nspikes):
            wavedata = self.sort.get_wavedata(ri)
            detid = spikes['detid'][ri]
            det = sort.detections[detid].detector
            nchans = spikes['nchans'][ri]
            #nt = (spikes.tend[ri] - spikes.t0[ri]) // sort.tres
            nt = spikes['nt'][ri]
            #try: assert len(np.arange(spikes.t0[ri], spikes.tend[ri], sort.tres)) == nt
            #except AssertionError: import pdb; pdb.set_trace()
            phase1ti = spikes['phase1ti'][ri]
            phase2ti = spikes['phase2ti'][ri]
            startti = twi0 - phase1ti # always +ve, usually 0 unless spike had some lockout near its start
            wavedata = wavedata[0:nchans, startti:]
            chans = spikes['chans'][ri, :nchans]
            chanis = det.chans.searchsorted(chans) # det.chans are always sorted
            x = det.siteloc[chanis, 0] # 1D array (row)
            y = det.siteloc[chanis, 1]
            # just x and y params for now
            print('ri = %d' % ri)
            x0, y0 = self.extractXY(wavedata, x, y, phase1ti, phase2ti)
            spikes['x0'][ri] = x0
            spikes['y0'][ri] = y0
        print("Extracting parameters from all %d spikes using %r took %.3f sec" %
              (nspikes, self.XYmethod.lower(), time.time()-t0))

    def get_spatial_mean(self, wavedata, x, y, phase1ti, phase2ti):
        """Return weighted spatial mean of chans in spike according to their
        Vpp at the same timepoints as on the max chan, to use as rough
        spatial origin of spike. x and y are spatial coords of chans in wavedata.
        phase1ti and phase2ti are timepoint indices in wavedata at which the max chan
        hits its 1st and 2nd spike phases.
        NOTE: you get better clustering if you allow phase1ti and phase2ti to
        vary at least slightly for each chan, since they're never simultaneous across
        chans, and sometimes they're very delayed or advanced in time
        NOTE: sometimes neighbouring chans have inverted polarity, see ptc15.87.50880, 68840"""

        # find min on each chan around phase1ti, and max on each chan around phase2ti
        # dividing dti by 2 might seem safer, since not looking for other phase, just
        # looking for same phase maybe slightly shifted, but clusterability seems
        # a bit better when you leave dti be
        dti = self.sort.detector.dti
        V1 = wavedata[:, max(phase1ti-dti,0):phase1ti+dti].min(axis=1)
        V2 = wavedata[:, max(phase2ti-dti,0):phase2ti+dti].max(axis=1)
        weights = V2 - V1
        # old method: # phase2 - phase1 on all chans:
        # int16 data isn't centered around V=0, but that doesn't matter since we want Vpp
        #weights = wavedata[:, phase2ti] - wavedata[:, phase1ti]

        # convert to float before normalization, take abs of all weights
        # taking abs doesn't seem to affect clusterability
        weights = np.abs(np.float32(weights))
        weights /= weights.sum() # normalized
        # alternative approach: replace -ve weights with 0
        #weights = np.float32(np.where(weights >= 0, weights, 0))
        #try: weights /= weights.sum() # normalized
        #except FloatingPointError: pass # weird all -ve weights spike, replaced with 0s
        x0 = (weights * x).sum()
        y0 = (weights * y).sum()
        return x0, y0

    def get_gaussian_fit(self, wavedata, x, y, phase1ti, phase2ti):
        if len(wavedata) < 4: # can't fit Gaussian for spikes with low nchans
            print('\n\nspike has only %d chans\n\n' % len(wavedata))
            return self.get_spatial_mean(wavedata, x, y, phase1ti, phase2ti)
        dti = self.sort.detector.dti / 2
        V = wavedata[:, phase1ti]
        #V1 = wavedata[:, max(phase1ti-dti,0):phase1ti+dti].min(axis=1)
        #V2 = wavedata[:, max(phase2ti-dti,0):phase2ti+dti].max(axis=1)
        #Vpp = V2 - V1
        i = np.abs(V).argmax()
        ls = self.ls
        ls.p0 = [ V[i], x[i], y[i], 60 ]
        ls.calc(x, y, V)
        #except: import pdb; pdb.set_trace()
        # TODO: return modelled amplitude and sigma as well!
        return ls.p[1], ls.p[2]
