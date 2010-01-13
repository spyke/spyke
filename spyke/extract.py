"""Spike parameter extraction"""

from __future__ import division

__authors__ = ['Martin Spacek']

import time

import numpy as np
#np.seterr(under='ignore') # only enable this if getting underflow during gaussian_fit
from scipy.optimize import leastsq

from spyke.detect import get_wave
from spyke.core import g2


class LeastSquares(object):
    """Least squares Levenberg-Marquardt spatial gaussian fit of decay across chans"""
    def __init__(self):
        self.anisotropy = 2 # y vs x anisotropy in sigma of gaussian fit

    def calc(self, x, y, V):
        t0 = time.clock()
        try: result = leastsq(self.cost, self.p0, args=(x, y, V), full_output=True)
                         #Dfun=None, full_output=True, col_deriv=False,
                         #maxfev=50, xtol=0.0001,
                         #diag=None)
        except: import pdb; pdb.set_trace()
        print('iters took %.3f sec' % (time.clock()-t0))
        self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        print('p0 = %r' % self.p0)
        print('p = %r' % self.p)
        print('%d iterations' % self.infodict['nfev'])
        print('mesg=%r, ier=%r' % (self.mesg, self.ier))

    def model(self, p, x, y):
        """2D circularly symmetric Gaussian"""
        return p[0] * g2(p[1], p[2], p[3], p[3], x, y)

    def model2(self, p, x, y):
        """2D elliptical Gaussian"""
        try:
            return p[0] * g2(p[1], p[2], p[3], p[3]*self.anisotropy, x, y)
        except Exception as err:
            print(err)
            import pdb; pdb.set_trace()

    def cost(self, p, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        return self.model2(p, x, y) - V


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
        ''' # comment out ICA stuff
        ICs = np.matrix(np.load('ptc15.87.2000_waveform_ICs.npy'))
        invICs = ICs.I # not a square matrix, think it must do pseudoinverse
        '''
        for ri in np.arange(nspikes):
            wavedata = self.sort.get_wavedata(ri)
            detid = spikes['detid'][ri]
            det = sort.detections[detid].detector
            nchans = spikes['nchans'][ri]
            chans = spikes['chans'][ri, :nchans]
            maxchan = spikes['chan'][ri]
            maxchani = int(np.where(chans == maxchan)[0])
            chanis = det.chans.searchsorted(chans) # det.chans are always sorted
            wavedata = wavedata[0:nchans]
            ''' # comment out ICA stuff
            maxchanwavedata = wavedata[maxchani]
            weights = maxchanwavedata * invICs # weights of ICs for this spike's maxchan waveform
            spikes['IC1'][ri] = weights[0, 0]
            spikes['IC2'][ri] = weights[0, 1]
            '''
            #nt = (spikes.tend[ri] - spikes.t0[ri]) // sort.tres
            #nt = spikes['nt'][ri]
            #try: assert len(np.arange(spikes.t0[ri], spikes.tend[ri], sort.tres)) == nt
            #except AssertionError: import pdb; pdb.set_trace()
            phase1ti = spikes['phase1ti'][ri]
            phase2ti = spikes['phase2ti'][ri]
            startti = twi0 - phase1ti # always +ve, usually 0 unless spike had some lockout near its start
            tightwavedata = wavedata[:, startti:]
            x = det.siteloc[chanis, 0] # 1D array (row)
            y = det.siteloc[chanis, 1]
            # just x and y params for now
            #print('ri = %d' % ri)
            x0, y0 = self.extractXY(tightwavedata, x, y, phase1ti, phase2ti, maxchani)
            spikes['x0'][ri] = x0
            spikes['y0'][ri] = y0
        print("Extracting parameters from all %d spikes using %r took %.3f sec" %
              (nspikes, self.XYmethod.lower(), time.time()-t0))

    def get_spatial_mean(self, wavedata, x, y, phase1ti, phase2ti, maxchani):
        """Return weighted spatial mean of chans in spike according to their
        Vpp at the same timepoints as on the max chan, to use as rough
        spatial origin of spike. x and y are spatial coords of chans in wavedata.
        phase1ti and phase2ti are timepoint indices in wavedata at which the max chan
        hits its 1st and 2nd spike phases.
        NOTE: you get better clustering if you allow phase1ti and phase2ti to
        vary at least slightly for each chan, since they're never simultaneous across
        chans, and sometimes they're very delayed or advanced in time
        NOTE: sometimes neighbouring chans have inverted polarity, see ptc15.87.50880, 68840"""

        # find peaks on each chan around phase1ti and phase2ti
        # dividing dti by 2 might seem safer, since not looking for other phase, just
        # looking for same phase maybe slightly shifted, but clusterability seems
        # a bit better when you leave dti be
        #dti = self.sort.detector.dti // 2 # constant
        dti = max(abs(int(phase1ti) - int(phase2ti)) // 2, 1) # varies from spike to spike
        V1 = wavedata[maxchani, phase1ti]
        V2 = wavedata[maxchani, phase2ti]
        window1 = wavedata[:, max(phase1ti-dti,0):phase1ti+dti]
        window2 = wavedata[:, max(phase2ti-dti,0):phase2ti+dti]
        if V1 < V2: # phase1ti is a min on maxchan, phase2ti is a max
            #weights = np.float32(window1.min(axis=1))
            V1s = np.float32(window1.min(axis=1))
            V2s = np.float32(window2.max(axis=1))
            weights = V2s - V1s
        else: # phase1ti is a max on maxchan, phase2ti is a min
            #weights = np.float32(window1.max(axis=1))
            V1s = np.float32(window1.max(axis=1))
            V2s = np.float32(window2.min(axis=1))
            weights = V1s - V2s

        # convert to float before normalization, take abs of all weights
        # taking abs doesn't seem to affect clusterability
        weights = np.abs(weights)
        weights /= weights.sum() # normalized
        # alternative approach: replace -ve weights with 0
        #weights = np.float32(np.where(weights >= 0, weights, 0))
        #try: weights /= weights.sum() # normalized
        #except FloatingPointError: pass # weird all -ve weights spike, replaced with 0s
        x0 = (weights * x).sum()
        y0 = (weights * y).sum()
        return x0, y0

    def get_gaussian_fit(self, wavedata, x, y, phase1ti, phase2ti, maxchani):
        """Can't seem to prevent from getting a stupidly wide range of modelled
        x locations. Tried fitting V**2 instead of V (to give big chans more weight),
        tried removing chans that don't fit spatial Gaussian model very well, and
        tried fitting with a fixed sy to sx ratio (1.5 or 2). They all may have helped
        a bit, but the results are still way too scattered, and look far less clusterable
        than results from spatial mean. Plus, the LM algorithm keeps generating underflow
        errors for some reason. These can be turned off and ignored, but it's strange that
        it's choosing to explore the fit at such extreme values of sx (say 0.6 instead of 60)"""
        self.x, self.y, self.maxchani = x, y, maxchani # bind in case need to pass unmolested versions to get_spatial_mean()
        dti = self.sort.detector.dti / 2
        wavedata = np.float64(wavedata)
        V1 = wavedata[:, max(phase1ti-dti,0):phase1ti+dti].min(axis=1)
        V2 = wavedata[:, max(phase2ti-dti,0):phase2ti+dti].max(axis=1)
        V = V2 - V1
        #V /= 1000
        V = V**2 # fit Vpp squared, so that big chans get more consideration, and errors on small chans aren't as important
        ls = self.ls
        ls.p0 = np.asarray([ V[maxchani], x[maxchani], y[maxchani], 60 ])

        while True:
            if len(V) < 4: # can't fit Gaussian for spikes with low nchans
                print('\n\nonly %d fittable chans in spike \n\n' % len(V))
                return self.get_spatial_mean(wavedata, self.x, self.y, phase1ti, phase2ti, self.maxchani)
            ls.calc(x, y, V)
            if ls.ier == 2: # essentially perfect fit between data and model
                break
            err = np.sqrt(np.abs(ls.cost(ls.p, x, y, V)))
            errsortis = err.argsort() # indices into err and chans, that sort err from lowest to highest
            #errsortis = errsortis[-1:0:-1] # highest to lowest
            otheris = list(errsortis) # creates a copy, used for mean & std calc
            erri = errsortis[-1] # index of biggest error
            if erri == maxchani: # maxchan has the biggest error
                erri = errsortis[-2] # index of biggest error excluding that of maxchan
                otheris.remove(maxchani) # remove maxchan from mean & std calc
            otheris.remove(erri) # remove biggest error from mean & std calc
            others = err[otheris] # dereference
            #hist(others)
            #hist(err[[erri]])
            meanerr, meanstd = others.mean(), others.std()
            maxerr = err[erri]
            print('mean err: %.0f' % meanerr)
            print('stdev err: %.0f' % meanstd)
            print('deviant chani %d had err: %.0f' % (erri, maxerr))
            if maxerr > meanerr+3*meanstd: # it's a big outlier, probably messing up Gaussian fit a lot
                # remove the erri'th entry from x, y and V
                # allow calc to happen again
                print('removing deviant chani %d' % erri)
                x = np.delete(x, erri)
                y = np.delete(y, erri)
                V = np.delete(V, erri)
                if erri < maxchani:
                    maxchani -= 1 # decr maxchani appropriately
            else:
                break

        # TODO: return modelled amplitude and sigma as well!
        return ls.p[1], ls.p[2]
