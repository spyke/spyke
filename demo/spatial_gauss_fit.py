import numpy as np
from scipy.optimize import leastsq
import time

import spyke
#from spyke.core import g2

def g2(x0, y0, sx, sy, x, y):
    """2-D Gaussian"""
    try:
        return np.exp( -(x-x0)**2 / (2*sx**2) - (y-y0)**2 / (2*sy**2) )
    except:
        import pdb; pdb.set_trace()

class LeastSquares(object):
    """Least squares Levenberg-Marquardt spatial gaussian fit of decay across chans"""
    def calc(self, x, y, V):
        #self.x = x
        #self.y = y
        #self.V = V
        t0 = time.clock()
        result = leastsq(self.cost, self.p0, args=(x, y, V), full_output=True)
                         #Dfun=None, full_output=True, col_deriv=False,
                         #maxfev=50, xtol=0.0001,
                         #diag=None)
        print('iters took %.3f sec' % (time.clock()-t0))
        self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        print('p0 = %r' % self.p0)
        print('p = %r' % self.p)
        print '%d iterations' % self.infodict['nfev']
        print 'mesg=%r, ier=%r' % (self.mesg, self.ier)

    def model(self, p, x, y):
        """2D Gaussian"""
        return p[0] * g2(p[1], p[2], p[3], p[3], x, y) # circularly symmetric Gaussian

    def model2(self, p, x, y):
        """2D Gaussian"""
        return p[0] * g2(p[1], p[2], p[3], p[4], x, y) # elliptical Gaussian

    def model_both_peaks(self, p, x, y):
        """2D Gaussian"""
        m = g2(p[2], p[3], p[4], p[4], x, y)
        return np.concatenate((p[0]*m, p[1]*m))

    def cost(self, p, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        #import pdb; pdb.set_trace()
        return self.model(p, x, y) - V


sf = spyke.surf.File('/data/ptc15/87 - track 7c spontaneous craziness.srf')
sf.parse()
t1 = 26960
t2 = t1+10
chans = [9, 44, 8, 45, 7, 46, 6, 47, 5]
# take just the peak point on the maxchan, simultaneously across chans
V = sf.hpstream[t1:t2][chans].data.flatten()
# or, could also take Vpp simultaneously across chans
x = np.array([ sf.hpstream.probe.SiteLoc[chan][0] for chan in chans ])
y = np.array([ sf.hpstream.probe.SiteLoc[chan][1] for chan in chans ])
ls = LeastSquares()
ls.p0 = [-7729, -5, 787, 60]
ls.calc(x, y, V)




# try a 3 col probe to see if that's enough to give good convergence with a 2nd sigma:
# doesn't seem to
sf = spyke.surf.File('/data/ptc17/21-tr1-driftbar_longbar.srf')
sf.parse()
t1 = 6071960
t2 = t1+10
chans = [5,  6,  7,  8,  9, 44, 45, 46, 47]
# take just the peak point on the maxchan, simultaneously across chans
V = sf.hpstream[t1:t2][chans].data.flatten()
# or, could also take Vpp simultaneously across chans
x = np.array([ sf.hpstream.probe.SiteLoc[chan][0] for chan in chans ])
y = np.array([ sf.hpstream.probe.SiteLoc[chan][1] for chan in chans ])
ls = LeastSquares()
ls.model = ls.model2 # use 2 sigma model
ls.p0 = [-5555, 0, 682, 60, 60]
ls.calc(x, y, V)


# Try fitting both the peak and the valley at the same time, doesn't help, still get localization way
# off the side of the probe for some spikes
sf = spyke.surf.File('/data/ptc18/14-tr1-mseq32_40ms_7deg.srf')
sf.parse()
t1 = 19241760
#t2 = t1+10
t2 = t1 + 180
chans = [7, 8, 9, 10, 11, 41, 42, 43, 44, 45]
#V = sf.hpstream[t1:t2][chans].data.flatten()
#V = sf.hpstream[t2:t2+10][chans].data.flatten() - sf.hpstream[t1:t1+10][chans].data.flatten()
V1 = sf.hpstream[t1:t1+10][chans].data.flatten()
V2 = sf.hpstream[t2:t2+10][chans].data.flatten()
V = np.concatenate((V1, V2))
x = np.array([ sf.hpstream.probe.SiteLoc[chan][0] for chan in chans ])
y = np.array([ sf.hpstream.probe.SiteLoc[chan][1] for chan in chans ])
ls = LeastSquares()
ls.model = ls.model_both_peaks
ls.p0 = [-6916, 3818, 25, 550, 60]
ls.calc(x, y, V)

# maybe give it two sets of points, one for each peak, and give it an additional amplitude param for the 2nd peak


'''
- instead of fitting a gaussian, try fitting something peakier, like an exponential?
    - or, instead of fitting the voltages, fit the squared voltages? prolly wouldn't help, already using squared errors

- go back to spatial mean, but only use subset of chans for each spike to localize - say only the biggest 5 chans (or less if there aren't that many chans in the spike)
- or, weight the chans nonlinearly - maybe do squared weights? Have I tried this already?
    - maybe weight each chan according to its squared distance from the maxchan?

- build up a set of eigenvectors for each chan separately, get eigenvals for each chan, then use those in combination for each set of chans in a given spike to cluster - problem is you're still in a fairly high dimensional space, something like 3*nchansperspike
    - what about ICA instead

- or, just go back to using openopt, and use constraints
    - slower, but seemed much more reliable
        - won't be modelling nearly as many points this time, speed might not be an issue
    - constraining sigma might be good enough

- Nick: model gaussian spike sources in 2D space, generate voltages at site locations, then add some noise. Now you know exactly where your "spikes" are in space, and you can test various methods against them to figure out which method works best to localize them, or which works best in certain situations
'''
