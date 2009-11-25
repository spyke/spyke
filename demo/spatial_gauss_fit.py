import numpy as np
from scipy.optimize import leastsq
import time

import spyke
#from spyke.core import g2
from pylab import hist

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
        return p[0] * g2(p[1], p[2], p[3], p[3]*2, x, y) # elliptical Gaussian

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



sf = spyke.surf.File('/data/ptc18/14-tr1-mseq32_40ms_7deg.srf')
sf.parse()

t0 =   19241520
tend = 19242520
phase1ti = 12
phase2ti = 21
#chans = [41, 11, 42, 10, 43, 9, 44, 8, 45, 7]
chans = [41, 11, 42, 10, 43, 9, 8, 45, 7]
i = chans.index(43) #V.argmax() # index of maxchan

t0 = 7252960
tend = 7253960
phase1ti = 12
phase2ti = 22
#chans = [ 7,  8,  9, 10, 11, 41, 42, 43, 44, 45]
chans = [ 7,  9, 10, 11, 41, 42, 43, 44]
# might need to add the 2nd sigma back for a spike like this...
i = chans.index(9) #V.argmax() # index of maxchan

dti = 340 / 2
wavedata = np.float64(sf.hpstream[t0:tend][chans].data)
V1 = wavedata[:, max(phase1ti-dti,0):phase1ti+dti].min(axis=1)
V2 = wavedata[:, max(phase2ti-dti,0):phase2ti+dti].max(axis=1)
V = V2 - V1
V = V ** 2
maxchan = chans[i]
x = np.array([ sf.hpstream.probe.SiteLoc[chan][0] for chan in chans ])
y = np.array([ sf.hpstream.probe.SiteLoc[chan][1] for chan in chans ])
d2 = (x-x[i])**2 + (y-y[i])**2
d2is = d2.argsort() # indices that sort chans by increasing d2 from maxchan
Vis = V.argsort() # indices that sort chans by increasing V
Vis = Vis[-1:0:-1] # sort chans by decreasing V
#if d2is != Vis:
#    import pdb; pdb.set_trace()
#V = V[d2is]; x = x[d2is]; y = y[d2is]
ls = LeastSquares()
ls.model = ls.model2
ls.p0 = [V[i], x[i], y[i], 60]
ls.calc(x, y, V)

err = np.sqrt(np.abs(ls.cost(ls.p, x, y, V)))
errsortis = err.argsort() # indices into err and chans, lowest err to highest
errsortis = errsortis[-1:0:-1] # highest to lowest
for erri in errsortis:
    if chans[erri] == maxchan:
        continue
    otheris = list(errsortis) # creates a copy
    otheris.remove(erri) # maybe be careful about removing one with a lot of signal
    otheris.remove(i)
    others = err[otheris]
    hist(others)
    hist(err[[erri]])
    print('mean err: %.3f' % others.mean())
    print('stdev err: %.3f' % others.std())
    print('deviant chan %d had err: %.3f' % (chans[erri], err[erri]))
    break


'''
if abs(x) > 28*2, or maybe abs(sigma) > 60:
    while True:
        # get rid of severe outlying channels that don't match gaussian model
        find chan with most error (excluding maxchan)
        if its abs(error) is > 3*stdev from the mean abs(error) (of all the other chans?):
            remove it, redo the fitting
            # alternative test is to see if refitting with newly removed
            # chan significantly changes any of the parameter values, by say > 15%
            # Or, maybe check distance from next highest err, since mean doesn't
            # mean too much for such a small population
        else:
            break

        #if cost per chan (err per chan included) has dropped by at least x:
        #    continue
        #else:
        #    revert to previous set of chans
        #    break

    print out which chans were removed as a warning for debugging
'''
