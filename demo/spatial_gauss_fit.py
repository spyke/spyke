import numpy as np
import scipy.optimize
from scipy.optimize import leastsq
from pylab import *
import time

import spyke
from spyke.core import g2


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
        print('ls.p0 = %r' % ls.p0)
        print('ls.p = %r' % ls.p)
        print '%d iterations' % self.infodict['nfev']
        print 'mesg=%r, ier=%r' % (self.mesg, self.ier)

    def model(self, p, x, y):
        """2D Gaussian"""
        return p[0] * g2(p[1], p[2], p[3], p[3], x, y) # circularly symmetric Gaussian

    def cost(self, p, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        #import pdb; pdb.set_trace()
        return self.model(p, x, y) - V



sf = spyke.surf.File('/data/ptc15/87 - track 7c spontaneous craziness.srf')
sf.parse()

t1 = 26960
t2 = 26970
chans = [9, 44, 8, 45, 7, 46, 6, 47, 5]

# take just the peak point on the maxchan, simultaneously across chans
V = sf.hpstream[t1:t2][chans].data.flatten()
# or, could also take Vpp simultaneously across chans

x = np.array([ sf.hpstream.probe.SiteLoc[chan][0] for chan in chans ])
y = np.array([ sf.hpstream.probe.SiteLoc[chan][1] for chan in chans ])
ls = LeastSquares()
ls.p0 = [-7729, -5, 787, 60]
ls.calc(x, y, V)

