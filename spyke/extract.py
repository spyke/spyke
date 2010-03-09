"""Spike parameter extraction"""

from __future__ import division

__authors__ = ['Martin Spacek']

import time
import multiprocessing as mp

import numpy as np
np.seterr(under='warn') # don't halt on underflow during gaussian_fit
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline
import pywt
import scipy.stats

from spyke.core import g2


def callextractspike(args):
    spike, wavedata = args
    ext = mp.current_process().extractor
    det = mp.current_process().detector
    return ext.extractspike(spike, wavedata, det)

def initializer(extractor, detector):
    #stream.srff.open() # reopen the .srf file which was closed for pickling, engage file lock
    #detector.sort.stream = stream
    #detector.sort.stream.srff = srff # restore .srff that was deleted from stream on pickling
    mp.current_process().extractor = extractor
    mp.current_process().detector = detector


class LeastSquares(object):
    """Least squares Levenberg-Marquardt spatial gaussian fit of decay across chans"""
    def __init__(self):
        self.A = None
        # TODO: mess with fixed sx and sy to find most clusterable vals, test on
        # 3 column data too
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
        self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        if self.debug:
            print('iters took %.3f sec' % (time.clock()-t0))
            print('p0 = %r' % self.p0)
            print('p = %r' % self.p)
            print('%d iterations' % self.infodict['nfev'])
            print('mesg=%r, ier=%r' % (self.mesg, self.ier))
    '''
    def model(self, p, x, y):
        """2D circularly symmetric Gaussian"""
        return self.A * g2(p[0], p[1], self.sx, self.sx, x, y)
    '''
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
        self.debug = False
        self.sort = sort
        self.XYmethod = XYmethod # or DEFXYMETHOD
        self.choose_XY_fun()

    def choose_XY_fun(self):
        if self.XYmethod.lower() == 'gaussian fit':
            self.extractXY = self.get_gaussian_fit
            self.ls = LeastSquares()
            self.ls.debug = self.debug
        elif self.XYmethod.lower() == 'spatial mean':
            self.extractXY = self.get_spatial_mean
        elif self.XYmethod.lower() == 'splines 1d fit':
            self.extractXY = self.get_splines_fit
        else:
            raise ValueError("Unknown XY parameter extraction method %r" % self.XYmethod)

    def __getstate__(self):
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['extractXY'] # can't pickle an instance method, not sure why it even bothers trying
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.choose_XY_fun() # restore instance method
    '''
    def extract_ICA(self):
        """This is just roughed in for now, had it in the extract_all_XY
        spike loop before"""
        ICs = np.matrix(np.load('ptc15.87.2000_waveform_ICs.npy'))
        invICs = ICs.I # not a square matrix, think it must do pseudoinverse

        for ri in xrange(nspikes):
            maxchanwavedata = wavedata[maxchani]
            # TODO: maybe normalize amplitude of spike to match that of the ICs (maybe keep everything normalized to 1). That way, You're really just looking at spike shape, and not allowing amplitude to add to the variability. Amplitude can remain a clusterable parameter via Vp or Vpp.
            weights = maxchanwavedata * invICs # weights of ICs for this spike's maxchan waveform
            spikes['IC0'][ri] = weights[0, 0]
            spikes['IC1'][ri] = weights[0, 1]
    '''
    def extract_all_wcs(self):
        """Extract wavelet coefficients from all spikes, store them as spike attribs"""
        self.ksis = [41, 11, 39, 40, 20] # best wave coeffs according kstest of wavedec of full ptc18.14 sort, using Haar wavelets
        ncoeffs = len(self.ksis)
        sort = self.sort
        spikes = sort.spikes # struct array
        wavedata = sort.wavedata
        nspikes = len(spikes)
        #ncoeffs = 53 # TODO: this only applies for V of length 50, stop hardcoding
        wcs = np.zeros((nspikes, ncoeffs))
        t0 = time.time()
        for spikei, (spike, wd) in enumerate(zip(spikes, wavedata)):
            nchans = spike['nchans']
            chans = spike['chans'][:nchans]
            maxchan = spike['chan']
            maxchani = int(np.where(chans == maxchan)[0])
            #chanis = det.chans.searchsorted(chans) # det.chans are always sorted
            wd = wd[:nchans]
            V = wd[maxchani]
            #wcs[spikei] = np.concatenate(pywt.wavedec(V, 'haar')) # flat array of wavelet coeffs
            wcs[spikei] = np.concatenate(pywt.wavedec(V, 'haar'))[self.ksis]
        #ks = np.zeros(ncoeffs)
        #p = np.zeros(ncoeffs)
        #for i in range(ncoeffs):
        #    ks[i], p[i] = scipy.stats.kstest(wcs[:, i], 'norm')
        #ksis = ks.argsort()[::-1] # ks indices sorted from biggest to smallest ks values
        # assign as params in spikes struct array
        for coeffi in range(ncoeffs):
            spikes['wc%d' % coeffi] = wcs[:, coeffi]
        print("Extracting wavelet coefficients from all %d spikes took %.3f sec" %
             (nspikes, time.time()-t0))
        # trigger resaving of .spike file on next .sort save
        try: del sort.spikefname
        except AttributeError: pass

    def extract_all_XY(self):
        """Extract XY parameters from all spikes, store them as spike attribs"""
        sort = self.sort
        spikes = sort.spikes # struct array
        # hack to get around numpy bug, see http://projects.scipy.org/numpy/ticket/1415:
        spikeslist = map(np.asarray, spikes)
        nspikes = len(spikes)
        if nspikes == 0:
            raise RuntimeError("No spikes to extract XY parameters from")
        try:
            wavedata = sort.wavedata
        except AttributeError:
            raise RuntimeError("Sort has no saved wavedata in memory to extract parameters from")
        print("Extracting XY parameters from spikes")
        t0 = time.time()
        if not self.debug: # use multiprocessing
            assert len(sort.detections) == 1
            det = sort.detector
            ncpus = mp.cpu_count() # 1 per core
            pool = mp.Pool(ncpus, initializer, (self, det)) # sends pickled copies to each process
            args = zip(spikeslist, wavedata)
            results = pool.map(callextractspike, args) # using chunksize=1 is a bit slower
            print('done with pool.map()')
            pool.close()
            # results is a list of (x0, y0) tuples, and needs to be unzipped
            spikes['x0'], spikes['y0'] = zip(*results)
        else:
            # give each process a detector, then pass one spike record and one waveform to
            # each this assumes all spikes come from the same detector with the same
            # siteloc and chans, which is safe to assume anyway
            initializer(self, sort.detector)
            for spike, wd in zip(spikes, wavedata):
                x0, y0 = callextractspike((spike, wd))
                spike['x0'] = x0
                spike['y0'] = y0
        print("Extracting XY parameters from all %d spikes using %r took %.3f sec" %
             (nspikes, self.XYmethod.lower(), time.time()-t0))
        # trigger resaving of .spike file on next .sort save
        try: del sort.spikefname
        except AttributeError: pass

    def extractspike(self, spike, wavedata, det):
        if self.debug or spike['id'] % 1000 == 0:
            print('%s: spike id: %d' % (mp.current_process().name, spike['id']))
        nchans = spike['nchans']
        chans = spike['chans'][:nchans]
        maxchan = spike['chan']
        maxchani = int(np.where(chans == maxchan)[0])
        chanis = det.chans.searchsorted(chans) # det.chans are always sorted
        wavedata = wavedata[:nchans]
        ''' # comment out ICA stuff
        maxchanwavedata = wavedata[maxchani]
        weights = maxchanwavedata * invICs # weights of ICs for this spike's maxchan waveform
        spikes['IC1'][ri] = weights[0, 0]
        spikes['IC2'][ri] = weights[0, 1]
        '''
        phasetis = spike['phasetis']
        aligni = spike['aligni']
        x = det.siteloc[chanis, 0] # 1D array (row)
        y = det.siteloc[chanis, 1]
        # just x and y params for now
        x0, y0 = self.extract(wavedata, maxchani, phasetis, aligni, x, y)
        return x0, y0

    def extract(self, wavedata, maxchani, phasetis, aligni, x, y):
        if len(wavedata) == 1: # only one chan, return its coords
            try:
                return int(x), int(y)
            except TypeError:
                print('%s: error trying int(x) and int(y)' % mp.current_process().name)
        # Vpp weights seem more clusterable than Vp weights
        weights = self.get_Vpp_weights(wavedata, maxchani, phasetis)
        #weights = self.get_Vp_weights(wavedata, maxchani, phasetis, aligni)
        return self.extractXY(weights, x, y, maxchani)

    def get_Vp_weights(self, wavedata, maxchani, phasetis, aligni):
        """Using just Vp instead of Vpp doesn't seem to improve clusterability"""
        dti = max((phasetis[1]-phasetis[0]) // 2, 1) # varies from spike to spike
        phaseti = phasetis[aligni]
        V = wavedata[maxchani, phaseti]
        window = wavedata[:, max(phaseti-dti,0):phaseti+dti]
        if V < 0:
            weights = np.float32(window.min(axis=1))
            weights = np.fmin(weights, 0) # clip any +ve values to 0
        else: # V >= 0
            weights = np.float32(window.max(axis=1))
            weights = np.fmax(weights, 0) # clip any -ve values to 0
        return weights

    def get_Vpp_weights(self, wavedata, maxchani, phasetis):
        """NOTE: you get better clustering if you allow phasetis to
        vary at least slightly for each chan, since they're never simultaneous across
        chans, and sometimes they're very delayed or advanced in time
        NOTE: sometimes neighbouring chans have inverted polarity, see ptc15.87.50880, 68840"""

        # find peaks on each chan around phasetis, assign weights by Vpp.
        # Dividing dti by 2 seems safer, since not looking for other phase, just
        # looking for same phase maybe slightly shifted
        #dti = self.sort.detector.dti // 2 # constant

        # TODO: try using dphase instead of dphase/2, check clusterability

        phasetis = np.int32(phasetis) # prevent over/underflow of uint8
        dti = max((phasetis[1]-phasetis[0]), 1) # varies from spike to spike
        Vs = wavedata[maxchani, phasetis]
        window0 = wavedata[:, max(phasetis[0]-dti,0):phasetis[0]+dti]
        window1 = wavedata[:, max(phasetis[1]-dti,0):phasetis[1]+dti]
        if Vs[0] < Vs[1]: # 1st phase is a min on maxchan, 2nd phase is a max
            #weights = np.float32(window0.min(axis=1))
            V0s = np.float32(window0.min(axis=1))
            V1s = np.float32(window1.max(axis=1))
            weights = V1s - V0s
        else: # 1st phase is a max on maxchan, 2nd phase is a min
            #weights = np.float32(window0.max(axis=1))
            V0s = np.float32(window0.max(axis=1))
            V1s = np.float32(window1.min(axis=1))
            weights = V0s - V1s
        return weights

    def get_spatial_mean(self, weights, x, y, maxchani):
        """Return weighted spatial mean of chans in spike according to their
        Vpp at the same timepoints as on the max chan, to use as rough
        spatial origin of spike. x and y are spatial coords of chans in wavedata"""

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
