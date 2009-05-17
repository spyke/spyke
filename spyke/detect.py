"""Spike detection and modelling"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
import sys
import time
import string

import wx
import pylab
import matplotlib as mpl

import numpy as np
#from scipy.optimize import leastsq, fmin_slsqp
import openopt
#import nmpfit

import spyke.surf
from spyke.core import WaveForm, toiter, argcut, intround, eucd, g, g2, RM
from text import SimpleTable


DMURANGE = (0, 500) # allowed time difference between peaks of modelled spike
TW = (-250, 750) # spike time window range, us, centered on thresh xing or 1st phase of spike

# save all Spike waveforms, even for those that have never been plotted or added to a neuron
#SAVEALLSPIKEWAVES = False
DONTSAVESPIKEWAVES = True


class FoundEnoughSpikesError(ValueError):
    pass


class RandomWaveTranges(object):
    """Iterator that spits out time ranges of width bs with
    excess bx that begin randomly from within the given trange.
    Optionally spits out no more than maxntranges tranges"""
    def __init__(self, trange, bs, bx=0, maxntranges=None):
        self.trange = trange
        self.bs = bs
        self.bx = bx
        self.maxntranges = maxntranges
        self.ntranges = 0

    def next(self):
        if self.maxntranges != None and self.ntranges >= self.maxntranges:
            raise StopIteration
        # random int within trange
        t0 = np.random.randint(low=self.trange[0], high=self.trange[1])
        tend = t0 + self.bs
        self.ntranges += 1
        return (t0-self.bx, tend+self.bx)

    def __iter__(self):
        return self


class DistanceMatrix(object):
    """Channel distance matrix, with rows in .data corresponding to
    .chans and .coords"""
    def __init__(self, SiteLoc):
        """SiteLoc is a dictionary of (x, y) tuples, with chans as the keys. See probes.py"""
        chans_coords = SiteLoc.items() # list of (chan, coords) tuples
        chans_coords.sort() # sort by chan
        self.chans = [ chan_coord[0] for chan_coord in chans_coords ] # pull out the sorted chans
        self.coords = [ chan_coord[1] for chan_coord in chans_coords ] # pull out the coords, now in chan order
        self.data = eucd(self.coords)
    '''
    # unused, therefore best left commented out:
    def __getitem__(self, key):
        """Make distance matrix data directly indexable by chan or chan pairs
        (instead of chani pairs). Return the distance between the chans in key.
        The order of the two chans in key doesn't matter, since .data is a symmetric matrix"""
        key = toiter(key)
        i0 = np.where(np.asarray(self.chans) == key[0]) # row index into .data of chan in key[0]
        if len(key) == 1:
            return self.data[i0].squeeze() # return a whole row of distances
        elif len(key) == 2:
            i1 = np.where(np.asarray(self.chans) == key[1]) # column index into .data of chan in key[1]
            return self.data[i0, i1] # return single distance value between the two specified chans
        else:
            raise ValueError, 'key must specify 1 or 2 chans'
    '''

class SpikeModel(object):
    """A model for fitting two voltage Gaussians to spike phases,
    plus a 2D spatial gaussian to model decay across channels"""
    def __init__(self):
        self.errs = []
        self.valid = False # modelled event is assumed not to be a spike until proven spike-worthy
        self.sxsyfactor = 3 # sx and sy need to be within this factor of each other

    def __eq__(self, other):
        """Compare SpikeModels by their parameter arrays"""
        if self.__class__ != other.__class__: # might be comparing a Spike with a Neuron
            return False
        #return np.all(self.p == other.p) # disable for now while not modelling
        return hash(self) == hash(other) # good enough for just simple detection

    def __hash__(self):
        """Unique hash value for self, based on modelled spike time and location.
        Required for effectively using SpikeModels in a Set"""
        return hash((self.t, self.maxchani)) # hash of their tuple, should guarantee uniqueness

    def __getitem__(self, key):
        """Return WaveForm for this spike given slice key"""
        assert key.__class__ == slice
        #stream = self.detection.detector.stream
        #if stream != None: # stream is available
        #    self.wave = stream[key] # let stream handle the slicing, save result
        #    return self.wave
        #elif self.wave != None: # stream unavailable, .wave from before last pickling is available
        if self.wave != None:
            return self.wave[key] # slice existing .wave
        else: # existing .wave unavailable
            return WaveForm() # return empty waveform

    def __getstate__(self):
        """Get object state for pickling"""
        #if SAVEALLSPIKEWAVES and self.wave.data == None:
        #    # make sure .wave is loaded before pickling to file
        #    self.update_wave()
        d = self.__dict__.copy() # this doesn't seem to be a slow step
        if DONTSAVESPIKEWAVES:
            d['wave'] = None # clear wave data to save space and time
            d['V'] = None
        d['errs'] = None
        d['plt'] = None # clear plot self is assigned to, since that'll have changed anyway on unpickle
        d['itemID'] = None # clear tree item ID, since that'll have changed anyway on unpickle
        return d

    def update_wave(self, stream=None, tw=None):
        """Load/update self's waveform, based either on data already present in
        self.V, or taken from the given stream. Optionally slice it according to
        tw around self's spike time"""
        if stream == None:
            assert self.V != None
            data = self.V
        else:
            wave = stream[self.ts[0] : self.ts[-1]+stream.tres] # end inclusive
            data = wave.data
            assert data.shape[1] == len(self.ts) # make sure I know what I'm doing
        self.wave = WaveForm(data=data, ts=self.ts, chans=self.chans)
        if tw != None:
            self.wave = self[self.t+tw[0] : self.t+tw[1]]
        return self.wave

    def plot(self):
        """Plot modelled and raw data for all chans, plus the single spatially
        positioned source time series, along with its 1 sigma ellipse"""
        # TODO: also plot the initial estimate of the model, according to p0, to see how the algoritm has changed wrt it
        ts, p = self.ts, self.p
        V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta = p
        uV2um = 45 / 100 # um/uV
        us2um = 75 / 1000 # um/us
        tw = ts[-1] - ts[0]
        f = pylab.figure()
        f.canvas.Parent.SetTitle('t=%d' % self.t)
        a = f.add_axes((0, 0, 1, 1), frameon=False, alpha=1.)
        self.f, self.a = f, a
        a.set_axis_off() # turn off the x and y axis
        f.set_facecolor('black')
        f.set_edgecolor('black')
        xmin, xmax = min(self.x), max(self.x)
        ymin, ymax = min(self.y), max(self.y)
        xrange = xmax - xmin
        yrange = ymax - ymin
        # this is set with an aspect ratio to mimic the effects of a.set_aspect('equal') without enforcing it
        f.canvas.Parent.SetSize((xrange*us2um*100, yrange*uV2um*8))
        thetadeg = theta*180/np.pi
        # plot stdev ellipse centered on middle timepoint, with bottom origin
        ellorig = x0, ymax-y0
        e = mpl.patches.Ellipse(ellorig, 2*sx, 2*sy, angle=thetadeg,
                                ec='#007700', fill=False, ls='dotted')
        a.add_patch(e)
        '''
        c = mpl.patches.Circle((0, yrange-15), radius=15, # for calibrating aspect ratio of display
                                ec='#ffffff', fill=False, ls='dotted')
        a.add_patch(c)
        '''
        # plot a radial arrow on the ellipse to make its vertical axis obvious. theta=0 should plot a vertical radial line
        arrow = mpl.patches.Arrow(ellorig[0], ellorig[1], -sy*np.sin(theta), sy*np.cos(theta),
                                  ec='#007700', fc='#007700', ls='solid')
        a.add_patch(arrow)
        for (V, x, y) in zip(self.V, self.x, self.y):
            t_ = (ts-ts[0]-tw/2)*us2um + x # in um, centered on the trace
            V_ = V*uV2um + (ymax-y) # in um, switch to bottom origin
            modelV_ = self.model(p, ts, x, y).ravel() * uV2um + (ymax-y) # in um, switch to bottom origin
            rawline = mpl.lines.Line2D(t_, V_, color='grey', ls='-', linewidth=1)
            modelline = mpl.lines.Line2D(t_, modelV_, color='red', ls='-', linewidth=1)
            a.add_line(rawline)
            a.add_line(modelline)
        t_ = (ts-ts[0]-tw/2)*us2um + x0 # in um
        modelsourceV_ = self.model(p, ts, x0, y0).ravel() * uV2um + (ymax-y0) # in um, switch to bottom origin
        modelsourceline = mpl.lines.Line2D(t_, modelsourceV_, color='lime', ls='-', linewidth=1)
        a.add_line(modelsourceline)
        a.autoscale_view(tight=True) # fit to enclosing figure
        a.set_aspect('equal') # this makes circles look like circles, and ellipses to tilt at the right apparent angle
        # plot vertical lines in all probe columns at self's modelled 1st and 2nd spike phase times
        colxs = list(set(self.x)) # x coords of probe columns
        ylims = a.get_ylim() # y coords of vertical line
        for colx in colxs: # plot one vertical line per spike phase per probe column
            t1_ = (self.phase1t-ts[0]-tw/2)*us2um + colx # in um
            t2_ = (self.phase2t-ts[0]-tw/2)*us2um + colx # in um
            vline1 = mpl.lines.Line2D([t1_, t1_], ylims, color='#004444', ls='dotted')
            vline2 = mpl.lines.Line2D([t2_, t2_], ylims, color='#440044', ls='dotted')
            a.add_line(vline1)
            a.add_line(vline2)

    def model(self, p, ts, x, y):
        """Sum of two Gaussians in time, modulated by a 2D spatial Gaussian.
        For each channel, return a vector of voltage values V of same length as ts.
        x and y are vectors of coordinates of each channel's spatial location.
        Output should be an (nchans, nt) matrix of modelled voltage values V"""
        V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta = p
        x, y = np.inner(RM(theta), np.asarray([x-x0, y-y0]).T) # make x, y distance to origin at x0, y0, and rotate by theta
        tmodel = V1*g(mu1, s1, ts) + V2*g(mu2, s2, ts)
        smodel = g2(0, 0, sx, sy, x, y)
        return np.outer(smodel, tmodel)

    def cost(self, p, ts, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        error = np.ravel(self.model(p, ts, x, y) - V)
        self.errs.append(np.abs(error).sum())
        #sys.stdout.write('%.1f, ' % np.abs(error).sum())
        return error

    def check_theta(self):
        """Ensure theta points along long axis of spatial model ellipse.
        Since theta always points along the sy axis, ensure sy is the long axis"""
        V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta = self.p
        if sx > sy:
            sx, sy = sy, sx # swap them so sy is the bigger of the two
            if theta > 0: # keep theta in [-pi/2, pi/2]
                theta = theta - np.pi/2
            else: # theta <= 0
                theta = theta + np.pi/2
            self.p = np.array([V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta])

    def get_paramstr(self, p=None):
        """Get formatted string of model parameter values"""
        p = p or self.p
        V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta = p
        s = ''
        s += 'V1, V2 = %d, %d uV\n' % (V1, V2)
        s += 'mu1, mu2 = %d, %d us\n' % (mu1, mu2)
        s += 's1, s2 = %d, %d us\n' % (s1, s2)
        s += 'x0, y0 = %d, %d um\n' % (x0, y0)
        s += 'sx, sy = %d, %d um\n' % (sx, sy)
        s += 'theta = %d deg' % (theta*180/np.pi)
        return s

    def print_paramstr(self, p=None):
        """Print formatted string of model parameter values"""
        print self.get_paramstr(p)


class NLLSPSpikeModel(SpikeModel):
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
    def c0(self, p, ts, x, y, V):
        """dmu lower bound constraint"""
        dmu = abs(p[3] - p[2])
        return self.dmurange[0] - dmu # <= 0, lower bound

    def c1(self, p, ts, x, y, V):
        """dmu upper bound constraint"""
        dmu = abs(p[3] - p[2])
        return dmu - self.dmurange[1] # <= 0, upper bound

    def c2(self, p, ts, x, y, V):
        """Constrain that sx and sy need to be within some factor of each other,
        ie constrain their ratio"""
        return max(p[8], p[9]) - self.sxsyfactor*min(p[8], p[9]) # <= 0

    # TODO: constrain V1 and V2 to have opposite sign, see ptc15.87.6920

    def calc(self, ts, x, y, V):
        self.ts = ts
        self.x = x
        self.y = y
        self.V = V
        pr = openopt.NLLSP(self.cost, self.p0, args=(ts, x, y, V),
                           ftol=self.FTOL, xtol=self.XTOL, gtol=self.GTOL)
        # set lower and upper bounds on parameters:
        # limit mu1 and mu2 to within min(ts) and max(ts) - sometimes they fall outside,
        # esp if there was a poor lockout and you're triggering off a previous spike
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
        pr.solve('nlp:ralg')
        self.pr, self.p = pr, pr.xf
        print "%d NLLSP iterations, cost f'n eval'd %d times" % (pr.iter, len(self.errs))
        self.check_theta()

    def __getstate__(self):
        """Get object state for pickling"""
        d = SpikeModel.__getstate__(self)
        # TODO: would be really nice to be able to keep the .pr attrib, for later inspection after unpickling of, say, bounds
        d['pr'] = None # don't pickle the openopt.NLLSP problem object, cuz it has lambdas which aren't picklable apparently
        return d


class Spike(NLLSPSpikeModel):
    """A Spike is just a subclass of a subclass of SpikeModel.
    Change inheritance to suit desired type of SpikeModel"""
    pass


class Detector(object):
    """Spike detector base class"""
    DEFTHRESHMETHOD = 'GlobalFixed' # GlobalFixed, ChanFixed, or Dynamic
    DEFNOISEMETHOD = 'median' # median or stdev
    DEFNOISEMULT = 3.5
    DEFFIXEDTHRESH = 50 # uV, used by GlobalFixed
    DEFPPTHRESHMULT = 1.5 # peak-to-peak threshold is this times thresh
    DEFFIXEDNOISEWIN = 10000000 # 10s, used by ChanFixed - this should really be a % of self.trange
    DEFDYNAMICNOISEWIN = 10000 # 10ms, used by Dynamic
    DEFMAXNSPIKES = 0
    DEFBLOCKSIZE = 1000000 # us, waveform data block size
    DEFSLOCK = 150 # spatial lockout radius, um
    DEFRANDOMSAMPLE = False

    BLOCKEXCESS = 1000 # us, extra data as buffer at start and end of a block while searching for spikes. Only useful for ensuring spike times within the actual block time range are accurate. Spikes detected in the excess are discarded

    def __init__(self, stream, chans=None,
                 threshmethod=None, noisemethod=None, noisemult=None, fixedthresh=None, ppthreshmult=None,
                 fixednoisewin=None, dynamicnoisewin=None,
                 trange=None, maxnspikes=None, blocksize=None,
                 slock=None, randomsample=None):
        """Takes a data stream and sets various parameters"""
        self.srffname = stream.srffname # used to potentially reassociate self with stream on unpickling
        self.stream = stream
        self.chans = chans or range(self.stream.nchans) # None means search all channels
        self.threshmethod = threshmethod or self.DEFTHRESHMETHOD
        self.noisemethod = noisemethod or self.DEFNOISEMETHOD
        self.noisemult = noisemult or self.DEFNOISEMULT
        self.fixedthresh = fixedthresh or self.DEFFIXEDTHRESH
        self.ppthreshmult = ppthreshmult or self.DEFPPTHRESHMULT
        self.fixednoisewin = fixednoisewin or self.DEFFIXEDNOISEWIN # us
        self.dynamicnoisewin = dynamicnoisewin or self.DEFDYNAMICNOISEWIN # us
        self.trange = trange or (stream.t0, stream.tend)
        self.maxnspikes = maxnspikes or self.DEFMAXNSPIKES # return at most this many spikes, applies across chans
        self.blocksize = blocksize or self.DEFBLOCKSIZE
        self.slock = slock or self.DEFSLOCK
        self.randomsample = randomsample or self.DEFRANDOMSAMPLE

        self.dmurange = DMURANGE # allowed time difference between peaks of modelled spike
        self.tw = TW # spike time window range, us, centered on 1st phase of spike

    def search(self):
        """Search for spikes. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results
        TODO: remove any spikes that happen right at the first or last timepoint in the file,
        since we can't say when an interrupted falling or rising edge would've reached peak
        """
        self.enabledSiteLoc = {}
        for chan in self.chans: # for all enabled chans
            self.enabledSiteLoc[chan] = self.stream.probe.SiteLoc[chan] # grab its (x, y) coordinate
        self.dm = DistanceMatrix(self.enabledSiteLoc) # distance matrix for the chans enabled for this search
        # TODO: instead of calling up the distance matrix and then checking for which chans are within slock over and over, since the slock is constant over an entire search, calculate each channel's neighbours once, and then call up that list of neighbours. Should be faster

        t0 = time.clock()
        self.thresh = self.get_thresh() # abs, in uV, one per chan in self.chans
        self.ppthresh = self.thresh * self.ppthreshmult # peak-to-peak threshold, abs, in uV
        #self.thresh = 50 # abs, in uV
        #self.ppthresh = self.thresh + 30 # peak-to-peak threshold, abs, in uV
        print 'thresh calcs took %.3f sec' % (time.clock()-t0)
        print 'thresh   = %s' % intround(self.thresh)
        print 'ppthresh = %s' % intround(self.ppthresh)

        bs = self.blocksize
        bx = self.BLOCKEXCESS
        wavetranges, (bs, bx, direction) = self.get_blockranges(bs, bx)

        nchans = len(self.chans) # number of enabled chans
        self.lockout = np.zeros(nchans, dtype=np.int64) # holds time indices until which each enabled chani is locked out
        self.nspikes = 0 # total num spikes found across all chans so far by this Detector, reset at start of every search
        spikes = [] # list of SpikeModels collected from .searchblock() call(s)

        t0 = time.clock()
        for wavetrange in wavetranges:
            try:
                spikes.extend(self.searchblock(wavetrange, direction))
            except FoundEnoughSpikesError:
                break
        spikets = [ s.t for s in spikes ]
        spikeis = np.argsort(spikets, kind='mergesort') # indices into spikes, ordered by spike time
        spikes = [ spikes[si] for si in spikeis ] # now guaranteed to be in temporal order
        print '\nfound %d spikes in total' % len(spikes)
        print 'inside .search() took %.3f sec' % (time.clock()-t0)
        return spikes

    def searchblock(self, wavetrange, direction):
        """Search a block of data, return a list of valid SpikeModels"""
        print 'searchblock():'
        print 'self.nspikes=%d, self.maxnspikes=%d, wavetrange=%s, direction=%d' % \
              (self.nspikes, self.maxnspikes, wavetrange, direction)
        if self.nspikes >= self.maxnspikes:
            raise FoundEnoughSpikesError # skip this iteration
        tlo, thi = wavetrange # tlo could be > thi
        bx = self.BLOCKEXCESS
        cutrange = (tlo+bx, thi-bx) # range without the excess, ie time range of spikes to actually keep
        #print 'wavetrange: %s, cutrange: %s' % (wavetrange, cutrange)
        wave = self.stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed
        wave = wave[self.chans] # get a WaveForm with just the enabled chans
        if self.randomsample:
            maxnspikes = 1 # how many more we're looking for in the next block
        else:
            maxnspikes = self.maxnspikes - self.nspikes

        tres = self.stream.tres
        twts = np.arange(self.tw[0], self.tw[1], tres) # temporal window timespoints wrt thresh xing or phase1t
        twts += twts[0] % tres # get rid of mod, so twts go through zero
        twi = intround(twts[0] / tres), intround(twts[-1] / tres) # time window indices wrt thresh xing or 1st phase
        print 'twi = %s' % (twi,)

        # want an nchan*2 array of [chani, x/ycoord]
        xycoords = [ self.enabledSiteLoc[chan] for chan in self.chans ] # (x, y) coords in chan order
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        self.siteloc = np.asarray([xcoords, ycoords]).T # index into with chani to get (x, y)

        spikes = self.threshwave(wave, twi)
        #spikes = self.modelspikes(spikes)

        # trim results from wavetrange down to just cutrange
        ts = np.asarray([ s.t for s in spikes ]) # get all spike times
        # searchsorted might be faster here instead of checking each and every element
        sis = (cutrange[0] < ts) * (ts < cutrange[1]) # boolean array of indices into spikes
        spikes = list(np.asarray(spikes)[sis])
        return spikes

    def threshwave(self, wave, twi):
        """Threshold wave data and return only events that roughly look like spikes
        TODO: would be nice to use some multichannel thresholding, instead of just single independent channel
            - e.g. obvious but small multichan spike at ptc15.87.23340
            - hyperellipsoidal?
            - take mean of sets of chans (say one set per chan, slock of chans around it), check when they exceed thresh, find max chan within that set at that time and report it as a threshold event
            - or slide some filter across the data that not only checks for thresh, but ppthresh as well
        """
        edges = np.diff(np.int8(abs(wave.data) >= np.vstack(self.thresh))) # indices where changing abs(signal) has crossed thresh
        edgeis = np.where(edges.T == 1) # indices of +ve edges, where increasing abs(signal) has crossed thresh
        edgeis = np.transpose(edgeis) # shape == (nti, 2), col0: ti, col1: chani. Rows are sorted increasing in time

        lockout = self.lockout
        spikes = []
        # check each edge for validity
        for ti, chani in edgeis:
            print 'trying thresh event at t=%d chan=%d' % (wave.ts[ti], self.chans[chani])
            if ti <= lockout[chani]: # is this thresh crossing timepoint locked out?
                print 'thresh event is locked out'
                continue # skip to next event

            # get data window wrt threshold crossing
            ti0 = max(ti+twi[0], lockout[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            tiend = min(ti+twi[1]+1, len(wave.ts)-1) # +1 makes it end inclusive, don't go further than last wave timepoint
            window = wave.data[chani, ti0:tiend] # window of data
            minti = window.argmin() # time of minimum in window, relative to ti0
            maxti = window.argmax() # time of maximum in window, relative to ti0
            phase1ti = min(minti, maxti) # wrt ti0
            phase2ti = max(minti, maxti)
            ti = ti0 + phase1ti # overwrite ti, make it phase1ti wrt 0th time index
            V1, V2 = window[phase1ti], window[phase2ti]
            print 'window params: t0=%d, phase1t=%d, tend=%d, mint=%d, maxt=%d, V1=%d, V2=%d' % \
                  (wave.ts[ti0], wave.ts[ti0+phase1ti], wave.ts[tiend], wave.ts[ti0+minti], wave.ts[ti0+maxti], V1, V2)

            # find all the enabled chanis within slock of chani, exclude chanis temporally locked-out at 1st phase:
            chanis, = np.where(self.dm.data[chani] <= self.slock) # at what col indices does the returned row fall within slock?
            chanis = np.asarray([ chi for chi in chanis if lockout[chi] < ti ])

            # find maxchan within chanis at 1st phase
            chanii = np.abs(wave.data[chanis, ti]).argmax() # index into chanis of new maxchan
            chani = chanis[chanii] # new max chani
            chan = self.chans[chani] # new max chan
            print 'new max chan=%d' % chan

            # get new data window using new maxchan and wrt 1st phase this time, instead of wrt the original thresh xing
            ti0 = max(ti+twi[0], lockout[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            tiend = min(ti+twi[1]+1, len(wave.ts)-1) # +1 makes it end inclusive, don't go further than last wave timepoint
            window = wave.data[chani, ti0:tiend]
            minti = window.argmin() # time of minimum in window, relative to ti0
            maxti = window.argmax() # time of maximum in window, relative to ti0
            minV, maxV = window[minti], window[maxti]
            phase1ti = min(minti, maxti) # wrt ti0
            phase2ti = max(minti, maxti)
            V1, V2 = window[phase1ti], window[phase2ti]

            # again, find all the enabled chanis within slock of new chani, exclude chanis locked-out at ti0:
            chanis, = np.where(self.dm.data[chani] <= self.slock) # at what col indices does the returned row fall within slock?
            chanis = np.asarray([ chi for chi in chanis if lockout[chi] < ti0 ])

            print 'window params: t0=%d, phase1t=%d, tend=%d, mint=%d, maxt=%d, V1=%d, V2=%d' % \
                  (wave.ts[ti0], wave.ts[ti0+phase1ti], wave.ts[tiend], wave.ts[ti0+minti], wave.ts[ti0+maxti], V1, V2)
            # check if this (still roughly defined) event crosses ppthresh, and some other requirements,
            # should help speed things up by rejecting obviously invalid events without having to run the model
            try:
                assert abs(V2 - V1) >= self.ppthresh[chani], \
                    "event doesn't cross ppthresh[chani=%d]=%.1f" % (chani, self.ppthresh[chani])
                assert ti0 < ti0+phase1ti < tiend, 'phase1t is at window endpoints, probably a mistrigger'
                assert np.sign(V1) == -np.sign(V2), 'phases must be of opposite sign'
                assert minV < 0, 'minV is %s V at t = %d' % (minV, wave.ts[ti0+minti])
                assert maxV > 0, 'maxV is %s V at t = %d' % (maxV, wave.ts[ti0+maxti])
            except AssertionError, message: # doesn't qualify as a spike
                print message
                continue # skip to next event

            # consider it a spike, save some attribs
            s = Spike()
            s.ti0, s.t0 = ti0, wave.ts[ti0]
            s.ti = ti0+phase1ti
            s.t = wave.ts[s.ti]
            s.ts = wave.ts[ti0:tiend]
            s.tiend, s.tend = tiend, wave.ts[tiend]
            s.V1, s.V2 = V1, V2
            chans = np.asarray(self.chans)[chanis] # dereference
            s.maxchani, s.chanis, s.chans = chani, chanis, chans
            s.x0, s.y0 = self.get_spike_spatial_mean(s, wave)
            s.valid = True
            spikes.append(s) # add to list of valid Spikes to return

            # update lockout
            lockout[chanis] = ti0 + phase2ti
            print 'lockout for chanis = %s' % wave.ts[lockout[chanis]]

        return spikes

    def get_spike_spatial_mean(self, spike, wave):
        """Return weighted spatial mean of chans in spike at designated spike
        time, to use as rough spatial origin of spike"""

        # take weighted spatial mean of chanis at phase1ti to estimate initial (x0, y0)
        x = self.siteloc[spike.chanis, 0] # 1D array (row)
        y = self.siteloc[spike.chanis, 1]
        chanweights = wave.data[spike.chanis, spike.ti] # unnormalized, some of these may be -ve
        # not sure if this is a valid thing to do, maybe just take abs instead, like when spike inverts across space
        #chanweights = np.where(chanweights >= 0, chanweights, 0) # replace -ve weights with 0
        chanweights = abs(chanweights)
        chanweights /= chanweights.sum() # normalized
        x0 = (chanweights * x).sum()
        y0 = (chanweights * y).sum()
        return x0, y0

    def modelspikes(self, events):
        """Model spike events that roughly look like spikes"""
        for ti, chani in events:
            print('trying to model thresh event at t=%d chan=%d'
                  % (wave.ts[ti], self.chans[chani]))
            # create a Spike model
            sm = Spike()
            chans = np.asarray(self.chans)[chanis] # dereference
            sm.chans, sm.maxchani, sm.nchans = chans, chani, nchans
            #sm.maxchanii, = np.where(sm.chanis == sm.maxchani) # index into chanis that returns maxchani
            sm.dmurange = self.dmurange
            print 'chans  = %s' % (chans,)
            print 'chanis = %s' % (chanis,)
            ts = wave.ts[ti0:tiend]
            x = siteloc[chanis, 0] # 1D array (row)
            y = siteloc[chanis, 1]
            V = wave.data[chanis, ti0:tiend]

            x0, y0 = self.get_spike_spatial_mean(sm, wave)

            # take weighted spatial mean of chanis at phase1ti to estimate initial (x0, y0)
            multichanwindow = wave.data[chanis, ti0:tiend]
            chanweights = multichanwindow[:, phase1ti] # unnormalized, some of these may be -ve
            chanweights = chanweights / chanweights.sum() # normalized
            chanweights = np.where(chanweights >= 0, chanweights, 0) # replace -ve weights with 0
            chanweights = chanweights / chanweights.sum() # renormalized
            x0 = (chanweights * x).sum()
            y0 = (chanweights * y).sum()


            print 'maxchan @ (%d, %d), (x0, y0)=(%.1f, %.1f)' % (siteloc[chani, 0], siteloc[chani, 1], x0, y0)
            """
            TODO: more intelligent estimate of sx and sy by taking signal differences between maxchan and two nearest chans. Get all chans with x vals different from max, and make a similar list for y vals. Out of each of those lists, get the nearest (in 2D) chan(s) to maxchan (pick one), find the signal value ratio between it and the maxchan at phase1ti, plug maxchan's (x or y) coord into g(), set it equal to the ratio, and solve for sigma (sx or sy).
            """
            # initial params
            p0 = [V1, V2, # Gaussian amplitudes (uV)
                  wave.ts[ti0+phase1ti], wave.ts[ti0+phase2ti], # temporal means mu1, mu2 (us)
                  60, 60, # temporal sigmas s1, s2 (us)
                  x0, y0, # spatial origin, (um)
                  60, 60, 0] # sx, sy (um), theta (radians)
            sm.p0 = np.asarray(p0)
            sm.calc(ts, x, y, V) # calculate spatiotemporal fit

            table = SimpleTable(np.asarray([sm.p0, sm.p]),
                                headers=('V1', 'V2', 'mu1', 'mu2', 's1', 's2', 'x0', 'y0', 'sx', 'sy', 'theta'),
                                stubs=('p0', 'p'),
                                fmt={'data_fmt': ['%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%d', '%.3f'],
                                     'data_aligns':      'r',
                                     'table_dec_above':  '-',
                                     'table_dec_below':  '-',
                                     'header_dec_below': ''})
            print table
            """
            The peak times of the modelled f'n may not correspond to the peak times of the two phases.
            Their amplitudes certainly need not correspond. So, here I'm reading values off of the modelled
            waveform instead of just the parameters of the constituent Gaussians that make it up
            """
            V1, V2, mu1, mu2, s1, s2, x0, y0, sx, sy, theta = sm.p
            modelV = sm.model(sm.p, sm.ts, x0, y0).ravel()
            modelminti = np.argmin(modelV)
            modelmaxti = np.argmax(modelV)
            phase1ti = min(modelminti, modelmaxti) # 1st phase might be the min or the max
            phase2ti = max(modelminti, modelmaxti) # 2nd phase might be the min or the max
            phase1t = ts[phase1ti]
            phase2t = ts[phase2ti]
            dphase = phase2t - phase1t
            V1 = modelV[phase1ti] # now refers to waveform's actual modelled peak, instead of Gaussian amplitude
            V2 = modelV[phase2ti]
            Vpp = abs(V2 - V1)
            absV1V2 = abs(np.array([V1, V2]))
            bigphase = max(absV1V2)
            smallphase = min(absV1V2)

            # save calculated params back to SpikeModel, save the SpikeModel in a dict
            sm.t = phase1t # phase1t is synonym for spike time
            sm.phase1t, sm.phase2t, sm.dphase = phase1t, phase2t, dphase
            sm.V1, sm.V2, sm.Vpp, sm.x0, sm.y0 = V1, V2, Vpp, x0, y0
            key = phase1t
            while key in self.sms: # if model timepoint doesn't make for a uniqe dict key (rarely the case)
                key += 0.1 # inc it slightly until it becomes unique
            self.sms[key] = sm # save the SpikeModel object for later inspection

            # check to see if modelled spike qualifies as an actual spike
            try:
                # ensure modelled spike time doesn't violate any existing lockout on any of its modelled chans
                assert (lockout[chanis] < ti0+phase1ti).all(), 'model spike time is locked out'
                assert wave.ts[ti0] < phase1t < wave.ts[tiend], "model spike time doesn't fall within time window"
                assert bigphase >= self.thresh[chani], \
                    "model (bigphase=%.1f) doesn't cross thresh[chani=%d]=%.1f " % (bigphase, chani, self.thresh[chani])
                assert Vpp >= self.ppthresh[chani], \
                    "model (Vpp=%.1f) doesn't cross ppthresh[chani=%d]=%.1f " % (Vpp, chani, self.ppthresh[chani])
                assert self.dmurange[0] <= dphase <= self.dmurange[1], \
                    'model phases separated by %f us (outside of dmurange=%s)' % (dphase, self.dmurange)
                assert np.sign(V1) == -np.sign(V2), 'model phases must be of opposite sign'
            except AssertionError, message: # doesn't qualify as a spike
                print '%s, spiket=%d' % (message, phase1t)
                continue
            # it's a spike, record it
            sm.valid = True
            sms.append(sm) # add to list of valid SpikeModels to return
            print 'found new spike: %d @ (%d, %d)' % (phase1t, intround(x0), intround(y0))
            """
            update spatiotemporal lockout

            TODO: maybe apply the same 2D gaussian spatial filter to the lockout in time, so chans further away
            are locked out for a shorter time. Use slock as a circularly symmetric spatial sigma
            TODO: center lockout on model (x0, y0) coords, instead of max chani - this could be dangerous - if model
            got it wrong, could get a whole lotta false +ve spike detections due to spatial lockout being way off

            lock out til one (TODO: maybe it should be two?) stdev after peak of 2nd phase,
            in case there's a noisy mini spike that might cause a trigger on the way down
            """
            lockout[chanis] = ti0 + phase2ti + intround(s2 / self.stream.tres)
            print 'lockout for chanis = %s' % wave.ts[lockout[chanis]]

    def check_spikes(self, spikes):
        """Checks for duplicate spikes between results from latest .searchblock() call,
        and previously saved spikes in this .search()"""
        if spikes == None:
            return
        nnewspikes = spikes.shape[1] # number of columns
        #wx.Yield() # allow GUI to update
        if self.randomsample and spikes.tolist() in np.asarray(self.spikes).tolist():
            # check if spikes is a duplicate of any that are already in .spikes, if so,
            # don't append this new spikes array, and don't inc self.nspikes. Duplicates are possible
            # in random sampling cuz we might end up with blocks with overlapping tranges.
            # Converting to lists for the check is probably slow cuz, but at least it's legible and correct
            sys.stdout.write('found duplicate spike')
        elif nnewspikes != 0:
            self.spikes.append(spikes)
            self.nspikes += nnewspikes # update
            sys.stdout.write('.')

    def get_blockranges(self, bs, bx):
        """Generate time ranges for slightly overlapping blocks of data,
        given blocksize and blockexcess"""
        wavetranges = []
        bs = abs(bs)
        bx = abs(bx)
        if self.trange[1] >= self.trange[0]: # search forward
            direction = 1
        else: # self.trange[1] < self.trange[0], # search backward
            bs = -bs
            bx = -bx
            direction = -1

        if self.randomsample:
            # wavetranges is an iterator that spits out random ranges starting from within
            # self.trange, and of width bs + 2bx
            if direction == -1:
                raise ValueError, "Check trange - I'd rather not do a backwards random search"
            wavetranges = RandomWaveTranges(self.trange, bs, bx)
        else:
            es = range(self.trange[0], self.trange[1], bs) # left (or right) edges of data blocks
            for e in es:
                wavetranges.append((e-bx, e+bs+bx)) # time range of waveform to give to .searchblock
            # last wavetrange surpasses self.trange[1] by some unknown amount, fix that here:
            wavetranges[-1] = (wavetranges[-1][0], self.trange[1]+bx) # replace with a new tuple
        return wavetranges, (bs, bx, direction)

    def get_sorted_sm(self, onlyvalid=False):
        """Return (only valid) SpikeModels in a sorted list of key:val tuples"""
        l = self.sms.items()
        l.sort() # according to key (spike time)
        if onlyvalid:
            l = [ (key, sm) for (key, sm) in l if sm.valid ]
        return l

    def plot_sm(self, reversed=True, onlyvalid=True):
        """Plot all spike models in self in (reversed) sorted order"""
        sortedsm = self.get_sorted_sm(onlyvalid)
        if reversed:
            sortedsm.reverse()
        for st, sm in sortedsm:
            sm.plot()

    # leave the stream be, let it be pickled
    '''
    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['_stream'] # don't pickle the stream, cuz it relies on ctsrecords, which rely on open .srf file
        return d
    '''
    def get_stream(self):
        return self._stream

    def set_stream(self, stream=None):
        """Check that self's srf file matches stream's srf file before binding stream"""
        if stream == None or stream.srffname != self.srffname:
            self._stream = None
        else:
            self._stream = stream # it's from the same file, bind it

    stream = property(get_stream, set_stream)

    def get_thresh(self):
        """Return array of thresholds, one per chan in self.chans,
        depending on threshmethod and noisemethod"""
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            thresh = np.tile(self.fixedthresh, len(self.chans))
            thresh = np.float32(thresh)
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed thresh
            """randomly sample self.fixednoisewin's worth of data from self.trange in
            blocks of self.blocksize. NOTE: this samples with replacement, so it's
            possible, though unlikely, that some parts of the data will contribute
            more than once to the noise calculation
            """
            print 'loading data to calculate noise'
            nblocks = intround(self.fixednoisewin / self.blocksize)
            wavetranges = RandomWaveTranges(self.trange, bs=self.blocksize, bx=0, maxntranges=nblocks)
            data = []
            for wavetrange in wavetranges:
                wave = self.stream[wavetrange[0]:wavetrange[1]]
                wave = wave[self.chans] # keep just the enabled chans
                data.append(wave.data)
            data = np.concatenate(data, axis=1)
            noise = self.get_noise(data)
            thresh = noise * self.noisemult
        elif self.threshmethod == 'Dynamic':
            # dynamic threshes are calculated on the fly during the search, so leave as zero for now
            thresh = np.zeros(len(self.chans), dtype=np.float32)
        else:
            raise ValueError
        assert len(thresh) == len(self.chans)
        assert thresh.dtype == np.float32
        return thresh

    def get_noise(self, data):
        """Calculates noise over last dim in data (time), using .noisemethod"""
        print 'calculating noise'
        if self.noisemethod == 'median':
            return np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            return np.stdev(data, axis=-1)
        else:
            raise ValueError
