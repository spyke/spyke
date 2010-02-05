"""Spike parameter extraction"""

from __future__ import division

__authors__ = ['Martin Spacek']

import time

import numpy as np
np.seterr(under='warn') # don't halt on underflow during gaussian_fit
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline

from spyke.core import g2


class LeastSquares(object):
    """Least squares Levenberg-Marquardt spatial gaussian fit of decay across chans"""
    def __init__(self):
        #self.anisotropy = 2 # y vs x anisotropy in sigma of gaussian fit
        self.A = None
        self.sx = 30
        self.sy = 30

    def calc(self, x, y, V):
        t0 = time.clock()
        try: result = leastsq(self.cost, self.p0, args=(x, y, V), full_output=True)
                         #Dfun=None, full_output=True, col_deriv=False,
                         #maxfev=50, xtol=0.0001,
                         #diag=None)
        except Exception as err:
            print(err)
            import pdb; pdb.set_trace()
        #print('iters took %.3f sec' % (time.clock()-t0))
        self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        #print('p0 = %r' % self.p0)
        #print('p = %r' % self.p)
        #print('%d iterations' % self.infodict['nfev'])
        #print('mesg=%r, ier=%r' % (self.mesg, self.ier))

    def model(self, p, x, y):
        """2D circularly symmetric Gaussian"""
        return self.A * g2(p[0], p[1], self.sx, self.sx, x, y)

    def model2(self, p, x, y):
        """2D elliptical Gaussian"""
        try:
            return self.A * g2(p[0], p[1], self.sx, self.sy, x, y)
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
        elif self.XYmethod.lower() == 'splines 1d fit':
            self.extractXY = self.get_splines_fit
        elif self.XYmethod.lower() == 'gaussian fit':
            self.extractXY = self.get_gaussian_fit
            self.ls = LeastSquares()
        else:
            raise ValueError("Unknown XY parameter extraction method %r" % self.XYmethod)

    def __getstate__(self):
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['extractXY'] # can't pickle an instance method, not sure why it even bothers trying
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.choose_XY_fun() # restore instance method

    def extract_all(self):
        """Extract spike parameters, store them as spike attribs"""
        sort = self.sort
        spikes = sort.spikes # struct array
        nspikes = len(spikes)
        if nspikes == 0:
            raise RuntimeError("No spikes to extract XY parameters from")
        try: sort.wavedata
        except AttributeError:
            raise RuntimeError("Sort has no saved wavedata in memory to extract parameters from")
        print("Extracting parameters from spikes")
        t0 = time.time()
        ''' # comment out ICA stuff
        ICs = np.matrix(np.load('ptc15.87.2000_waveform_ICs.npy'))
        invICs = ICs.I # not a square matrix, think it must do pseudoinverse
        '''
        import multiprocessing
        from spyke import threadpool
        ncpus = multiprocessing.cpu_count()
        pool = threadpool.Pool(ncpus)
        x0y0s = pool.map(self.extractspike, range(nspikes))
        pool.terminate() # pool.close() doesn't allow Python to exit when spyke is closed
        #pool.join() # unnecessary, hangs
        """
        for ri in xrange(nspikes):
            wavedata = sort.wavedata[ri]
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
            phase1ti = spikes['phase1ti'][ri]
            phase2ti = spikes['phase2ti'][ri]
            x = det.siteloc[chanis, 0] # 1D array (row)
            y = det.siteloc[chanis, 1]
            # just x and y params for now
            #print('ri = %d' % ri)
            x0, y0 = self.extract(wavedata, phase1ti, phase2ti, x, y, maxchani)
            spikes['x0'][ri] = x0
            spikes['y0'][ri] = y0
        """
        print("Extracting parameters from all %d spikes using %r took %.3f sec" %
              (nspikes, self.XYmethod.lower(), time.time()-t0))
        # trigger resaving of .spike file on next .sort save
        try: del sort.spikefname
        except AttributeError: pass

    # give each process a detector, then pass one spike record and one waveform to each
    # this assumes all spikes come from the same detector with the same siteloc and chans,
    # which is safe to assume anyway

    def extract(self, wavedata, phase1ti, phase2ti, x, y, maxchani):
        if len(wavedata) == 1: # only one chan, return its coords
            return int(x), int(y)
        weights = self.get_weights(wavedata, phase1ti, phase2ti, maxchani)
        return self.extractXY(weights, x, y, maxchani)

    def callextractspike(spike, wavedata):
        det = multiprocessing.current_process().detector
        return extractspike(spike, wavedata, det)

    def extractspike(spike, wavedata, det):
        nchans = spike['nchans']
        chans = spike['chans'][:nchans]
        maxchan = spike['chan']
        maxchani = int(np.where(chans == maxchan)[0])
        chanis = det.chans.searchsorted(chans) # det.chans are always sorted
        wavedata = wavedata[0:nchans]
        ''' # comment out ICA stuff
        maxchanwavedata = wavedata[maxchani]
        weights = maxchanwavedata * invICs # weights of ICs for this spike's maxchan waveform
        spikes['IC1'][ri] = weights[0, 0]
        spikes['IC2'][ri] = weights[0, 1]
        '''
        phase1ti = spike['phase1ti']
        phase2ti = spike['phase2ti']
        x = det.siteloc[chanis, 0] # 1D array (row)
        y = det.siteloc[chanis, 1]
        # just x and y params for now
        #print('ri = %d' % ri)
        x0, y0 = self.extract(wavedata, phase1ti, phase2ti, x, y, maxchani)
        return x0, y0
        #spikes['x0'][ri] = x0
        #spikes['y0'][ri] = y0

    def get_weights(self, wavedata, phase1ti, phase2ti, maxchani):
        """NOTE: you get better clustering if you allow phase1ti and phase2ti to
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
        return weights

    def get_spatial_mean(self, weights, x, y, maxchani):
        """Return weighted spatial mean of chans in spike according to their
        Vpp at the same timepoints as on the max chan, to use as rough
        spatial origin of spike. x and y are spatial coords of chans in wavedata.
        phase1ti and phase2ti are timepoint indices in wavedata at which the max chan
        hits its 1st and 2nd spike phases.
        NOTE: you get better clustering if you allow phase1ti and phase2ti to
        vary at least slightly for each chan, since they're never simultaneous across
        chans, and sometimes they're very delayed or advanced in time
        NOTE: sometimes neighbouring chans have inverted polarity, see ptc15.87.50880, 68840"""

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

    def get_splines_fit(self, w, x, y, maxchani):
        # w stands for weights
        xi = x.argsort()
        w, x, y = w[xi], x[xi], y[xi] # sort points by x values
        ux = np.unique(x)
        yw = np.empty(len(ux)) # these end up being the max interpolated weight values in each column
        y0s = np.empty(len(ux))
        xis = x.searchsorted(ux) # start indices of coords with identical x values
        # iterate over columns:
        import pdb; pdb.set_trace()
        for coli, starti in enumerate(xis):
            try:
                endi = xis[coli+1]
            except IndexError:
                endi = len(x)
            yc, wc = y[starti:endi], w[starti:endi] # y and w values for this column
            if len(yc) < 3: # not enough chans in this column to interpolate vertically, just find the max?
                assert len(yc) > 0
                yi = yc.argmax()
                yw[coli] = wc[yi]
                y0s[coli] = yc[yi]
            else:
                #k = min(max(3, len(yc)-2), 5)
                k = min(3, len(yc)-1)
                yi = yc.argsort() # UnivariateSpline requires monotonically ascending coordinates
                try:
                    us = UnivariateSpline(yc[yi], wc[yi], k=k)
                except UserWarning:
                    import pdb; pdb.set_trace()
                except:
                    import pdb; pdb.set_trace()
                yc2 = np.arange(yc.min(), yc.max(), 1) # span whole y range in steps of 1um
                wc2 = us(yc2)
                # if w[maxchani] > 0: # this is always the case - see get_weights
                wi = wc2.argmax()
                #else: # look for a min:
                #    wi = wc2.argmin()
                yw[coli] = wc2[wi]
                y0s[coli] = yc2[wi]
        # do normal full spatial mean for x values
        xw = np.abs(w)
        xw /= xw.sum() # normalized
        x0 = (xw * x).sum()
        # do column-wise spatial mean for y values
        yw = np.abs(yw)
        yw /= yw.sum() # normalized
        y0 = (yw * y0s).sum()
        return x0, y0

    def get_gaussian_fit(self, w, x, y, maxchani):
        """Can't seem to prevent from getting a stupidly wide range of modelled
        x locations. Tried fitting V**2 instead of V (to give big chans more weight),
        tried removing chans that don't fit spatial Gaussian model very well, and
        tried fitting with a fixed sy to sx ratio (1.5 or 2). They all may have helped
        a bit, but the results are still way too scattered, and look far less clusterable
        than results from spatial mean. Plus, the LM algorithm keeps generating underflow
        errors for some reason. These can be turned off and ignored, but it's strange that
        it's choosing to explore the fit at such extreme values of sx (say 0.6 instead of 60)"""
        #self.x, self.y, self.maxchani = x, y, maxchani # bind in case need to pass unmolested versions to get_spatial_mean()
        #w **= 2 # fit Vpp squared, so that big chans get more consideration, and errors on small chans aren't as important
        ls = self.ls
        x0, y0 = self.get_spatial_mean(w, x, y, maxchani)
        # or, init with just the coordinates of the max weight, doesn't save time
        #x0, y0 = x[maxchani], y[maxchani]
        ls.A = w[maxchani]
        ls.p0 = np.asarray([x0, y0])
        #ls.p0 = np.asarray([x[maxchani], y[maxchani]])
        ls.calc(x, y, w)
        return ls.p[0], ls.p[1]
        '''
        while True:
            if len(V) < 4: # can't fit Gaussian for spikes with low nchans
                print('\n\nonly %d fittable chans in spike \n\n' % len(V))
                return self.get_spatial_mean(weights, self.x, self.y, self.maxchani)
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
        '''
