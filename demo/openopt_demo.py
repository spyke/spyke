import numpy as np
import time
import openopt

import spyke
#from spyke.core import g2

def g2(x0, y0, sx, sy, x, y):
    """2-D Gaussian"""
    try:
        return np.exp( -(x-x0)**2 / (2*sx**2) - (y-y0)**2 / (2*sy**2) )
    except:
        import pdb; pdb.set_trace()



class NLLSPSpikeModel(object):
    """Nonlinear least squares problem solver from openopt, uses Shor's R-algorithm.
    This one can handle constraints"""
    FTOL = 1e-1 # function tolerance, openopt default is 1e-6
    XTOL = 1e-6 # variable tolerance
    GTOL = 1e-6 # gradient tolerance

    """Here come the constraints. For improved speed, might want to stop passing unnecessary args"""

    """constrain self.dmurange[0] <= dmu <= self.dmurange[1]
    TODO: maybe this contraint should be on the peak separation in the sum of Gaussians,
    instead of just on the mu params. Can probably remove the lower bound on the peak separation,
    especially if it's left at 0"""

    def model(self, p, x, y):
        """2D Gaussian"""
        return p[0] * g2(p[1], p[2], p[3], p[3], x, y) # circularly symmetric Gaussian

    def cost(self, p, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        #import pdb; pdb.set_trace()
        return self.model(p, x, y) - V # should maybe normalize error somehow, such that the stdev of the error is 1.0

    def calc(self, x, y, V):
        pr = openopt.NLLSP(self.cost, self.p0, args=(x, y, V))
                           #ftol=self.FTOL, xtol=self.XTOL, gtol=self.GTOL)
        # set lower and upper bounds on parameters:
        # limit mu1 and mu2 to within min(ts) and max(ts) - sometimes they fall outside,
        # esp if there was a poor lockout and you're triggering off a previous spike
        '''
        pr.lb[2], pr.ub[2] = min(ts), max(ts) # mu1
        pr.lb[3], pr.ub[3] = min(ts), max(ts) # mu2
        pr.lb[4], pr.ub[4] = 40, 250 # s1
        pr.lb[5], pr.ub[5] = 40, 250 # s2
        # limit x0 to within reasonable distance of vertical midline of probe
        pr.lb[6], pr.ub[6] = -50, 50 # x0
        pr.lb[8], pr.ub[8] = 20, 200 # sx
        pr.lb[9], pr.ub[9] = 20, 200 # sy
        pr.lb[10], pr.ub[10] = -np.pi/2, np.pi/2 # theta (radians)
        pr.c = [self.c0, self.c1, self.c2] # constraints
        '''
        pr.lb[1], pr.ub[1] = -50, 50 # um
        t0 = time.clock()
        pr.solve('scipy_leastsq')
        self.pr, self.p = pr, pr.xf
        print('iters took %.3f sec' % (time.clock()-t0))
        print('p0 = %r' % self.p0)
        print('p = %r' % self.p)
        print("%d NLLSP iterations" % pr.iter)



# Try fitting both the peak and the valley at the same time, doesn't help, still get localization way
# off the side of the probe for some spikes
sf = spyke.surf.File('/data/ptc18/14-tr1-mseq32_40ms_7deg.srf')
sf.parse()
t1 = 19241760
t2 = t1+10
#t2 = t1 + 180
chans = [7, 8, 9, 10, 11, 41, 42, 43, 44, 45]
V = sf.hpstream[t1:t2][chans].data.flatten()
#V = sf.hpstream[t2:t2+10][chans].data.flatten() - sf.hpstream[t1:t1+10][chans].data.flatten()
#V1 = sf.hpstream[t1:t1+10][chans].data.flatten()
#V2 = sf.hpstream[t2:t2+10][chans].data.flatten()
#V = np.concatenate((V1, V2))
x = np.array([ sf.hpstream.probe.SiteLoc[chan][0] for chan in chans ])
y = np.array([ sf.hpstream.probe.SiteLoc[chan][1] for chan in chans ])
ls = NLLSPSpikeModel()
#ls.model = ls.model_both_peaks
ls.p0 = [-6916, 25, 550, 60]
ls.calc(x, y, V)
