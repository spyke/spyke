"""Event detection algorithms
"""

from __future__ import division

__authors__ = ['Martin Spacek, Reza Lotun']

import itertools
import sys
import time
import string
import processing
import threadpool

import wx

import numpy as np
#from scipy.optimize import leastsq, fmin_slsqp
import openopt
#import nmpfit

from pylab import *

import spyke.surf
from spyke.core import WaveForm, toiter, argcut, intround, cvec, eucd, g, g2


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

    def plot(self):
        """Plot modelled and raw data for all chans,
        plus the single spatially positioned source time series"""
        t, p = self.t, self.p
        f = figure()
        f.canvas.Parent.SetTitle('t=%d' % self.spiket)
        a = f.add_axes((0, 0, 1, 1), frameon=False, alpha=1.)
        self.f, self.a = f, a
        a.set_axis_off() # turn off the x and y axis
        f.set_facecolor('black')
        f.set_edgecolor('black')
        uV2um = 100 / 100 # um/uV
        us2um = 55 / 1000 # um/us
        xmin, xmax = min(self.x), max(self.x)
        ymin, ymax = min(self.y), max(self.y)
        xrange = xmax - xmin
        yrange = ymax - ymin
        f.canvas.Parent.SetSize((xrange*us2um*90, yrange*uV2um*2.5))
        for chanii, (V, x, y) in enumerate(zip(self.V, self.x, self.y)):
            t_ = (t-t[0])*us2um + x # in um
            V_ = V*uV2um + (ymax-y) # in um, switch to bottom origin
            modelV_ = self.model(p, t, x, y).ravel() * uV2um + (ymax-y) # switch to bottom origin
            rawline = mpl.lines.Line2D(t_, V_, color='grey', ls='-', linewidth=1)
            modelline = mpl.lines.Line2D(t_, modelV_, color='red', ls='-', linewidth=1)
            a.add_line(rawline)
            a.add_line(modelline)
        x0, y0 = p[6], p[7]
        t_ = (t-t[0])*us2um + x0
        modelsourceV_ = self.model(p, t, x0, y0).ravel() * uV2um + (ymax-y0) # switch to bottom origin
        modelsourceline = mpl.lines.Line2D(t_, modelsourceV_, color='lime', ls='-', linewidth=1)
        a.add_line(modelsourceline)
        a.autoscale_view(tight=True)
        # plot vertical lines in all probe columns at self's modelled 1st and 2nd spike phase times
        colxs = list(set(self.x)) # x coords of probe columns
        ylims = a.get_ylim() # y coords of vertical line
        for colx in colxs: # plot one vertical line per spike phase per probe column
            t1_ = (self.phase1t-t[0])*us2um + colx # in um
            t2_ = (self.phase2t-t[0])*us2um + colx # in um
            vline1 = mpl.lines.Line2D([t1_, t1_], ylims, color='#004444', ls=':')
            vline2 = mpl.lines.Line2D([t2_, t2_], ylims, color='#444400', ls=':')
            a.add_line(vline1)
            a.add_line(vline2)

    def cost(self, p, t, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        error = np.ravel(self.model(p, t, x, y) - V)
        self.errs.append(np.abs(error).sum())
        #sys.stdout.write('%.1f, ' % np.abs(error).sum())
        return error

    def model(self, p, t, x, y):
        """Sum of two Gaussians in time, modulated by a 2D spatial Gaussian.
        For each channel, return a vector of voltage values V of same length as t.
        x and y are vectors of coordinates of each channel's spatial location.
        Output should be an (nchans, nt) matrix of modelled voltage values V"""
        phase1V, mu1, s1, phase2V, mu2, s2, x0, y0, sx, sy, vinv = p
        # TODO: vx and vy should be in the rotated spatial coordinate space
        """
        dtx = np.abs(x - x0) * vxinv
        dty = np.abs(y - y0) * vyinv
        dt = dtx + dty
        """
        # TODO: if giving phase1 and phase2 different AP velocity delays, should probably constrain them to be of the same sign - scratch that: see ptc15.r87.26940 where the 1st phase increasingly leads the spike time as a f'n of distance, and the 2nd phase increasingly lags the spike time as a f'n of distance. You need to allow vinv1 and vin2 to be of opposite sign to successfully model this
        d = np.sqrt((x - x0)**2 + (y - y0)**2)
        dt = d * vinv
        # tile t vertically to make a 2D matrix of height nchans, so it can be broadcast across the mu+dt vectors in g()
        try:
            nchans = len(x)
        except TypeError: # x is scalar?
            nchans = 1
        t = np.tile(t, (nchans, 1))
        tprofile = phase1V*g(cvec(mu1+dt), s1, t) + phase2V*g(cvec(mu2+dt), s2, t) # 2D temporal profile matrix, one row per chan
        sprofile = cvec(g2(x0, y0, sx, sy, x, y)) # spatial profile column vector
        return sprofile * tprofile


class NLLSP(SpikeModel):
    """Nonlinear least squares problem solver from openopt, uses Shor's R-algorithm.
    This one can handle constraints"""
    def calc(self, t, x, y, V):
        self.t = t
        self.x = x
        self.y = y
        self.V = V
        pr = openopt.NLLSP(self.cost, self.p0, args=(t, x, y, V))
        pr.lb[6], pr.ub[6] = -50, 50 # x0
        pr.lb[8], pr.ub[8] = 20, 200 # sx
        pr.lb[9], pr.ub[9] = 20, 200 # sy
        pr.lb[10], pr.ub[10] = -2, 2 # vinv
        #pr.lb[11], pr.ub[11] = -2, 2 # vyinv
        """constrain self.dmurange[0] <= dmu <= self.dmurange[1]
        maybe this contraint should be on the peak separation in the sum of Gaussians,
        instead of just on the mu params
        can probably remove the lower bound on the peak separation, especially if it's left at 0.
        For improved speed, might want to stop passing unnecessary args"""
        c0 = lambda p, t, x, y, V: self.dmurange[0] - abs(p[4] - p[1]) # <= 0, lower bound
        c1 = lambda p, t, x, y, V: abs(p[4] - p[1]) - self.dmurange[1] # <= 0, upper bound
        # TODO: could constrain mu1 and mu2 to fall within min(t) and max(t) - sometimes they fall outside, esp if there was a poor lockout and you're triggering off a previous spike
        # TODO: could also say that sx and sy need to be within some fraction (50% ?) of each other, ie constrain their ratio
        pr.c = [c0, c1] # constraints
        pr.solve('nlp:ralg')
        self.pr, self.p = pr, pr.xf
        print '%d iterations' % pr.iter

'''
class LeastSquares(SpikeModel):
    """Least squares Levenberg-Marquardt fit of two voltage Gaussians
    to spike phases, plus a 2D spatial gaussian to model decay across channels"""
    def __init__(self):
        self.ftol = 1.49012e-8 # Relative error desired in the sum of squares
        self.xtol = 1.49012e-8 # Relative error desired in the approximate solution
        self.gtol = 0.0 # Orthogonality desired between the function vector and the columns of the Jacobian
        self.maxfev = 0 # The maximum number of calls to the function, 0 means unlimited
        # these are replicated here from the scipy.optimize.leastsq source code to get nice diagnostics while working around
        # a bug in the scipy code which happens when extra details like these are asked for with full_output = True
        self.errors = {0:"Improper input parameters.",
                       1:"Both actual and predicted relative reductions in the sum of squares\n  are at most %f" % self.ftol,
                       2:"The relative error between two consecutive iterates is at most %f" % self.xtol,
                       3:"Both actual and predicted relative reductions in the sum of squares\n  are at most %f and the relative error between two consecutive iterates is at \n  most %f" % (self.ftol, self.xtol),
                       4:"The cosine of the angle between func(x) and any column of the\n  Jacobian is at most %f in absolute value" % self.gtol,
                       5:"Number of calls to function has reached maxfev = %d." % self.maxfev,
                       6:"ftol=%f is too small, no further reduction in the sum of squares\n  is possible." % self.ftol,
                       7:"xtol=%f is too small, no further improvement in the approximate\n  solution is possible." % self.xtol,
                       8:"gtol=%f is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision." % self.gtol,
                       'unknown':"Unknown error."
                       }

    def calc(self, t, x, y, z, V):
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.V = V
        result = leastsq(self.cost, self.p0, args=(t, x, y, z, V),
                         Dfun=None, full_output=False, col_deriv=False,
                         ftol=self.ftol, xtol=self.xtol, gtol=self.gtol, maxfev=self.maxfev,
                         diag=None)
        #self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        self.p, self.ier = result
        self.mesg = self.errors[self.ier]
        #print '%d iterations' % self.infodict['nfev']
        print 'mesg=%r, ier=%r' % (self.mesg, self.ier)


class SLSQP(SpikeModel):
    """Sequential Least SQuares Programming with constraints.
    Oops, I'm dumb. SLSQP is a scalar nonlinear problem (NLP) solver.
    I need a nonlinear systems problem (NLSP),
    or a least squares nonlinear systems problem (LSNLSP) solver,
    aka a least squares problem (LSP) solver.
    """
    def tcalc(self, t, V):
        """Calculate least squares of temporal model"""
        self.t = t
        self.V = V
        i = 2*self.nchans
        """self.dmurange[0] <= dmu <= self.dmurange[1]"""

        ieqcon0 = lambda tp, t, V: abs(tp[i] - tp[i+2]) - self.dmurange[0] # constrain to be >= 0
        ieqcon1 = lambda tp, t, V: self.dmurange[1] - abs(tp[i] - tp[i+2]) # constrain to be >= 0
        # doesn't work, cuz self.tcost returns a vector, and slsqp expects a scalar:
        result = fmin_slsqp(self.tcost, self.tp0, args=(t, V),
                            bounds=[],
                            eqcons=[],
                            ieqcons=[ieqcon0, ieqcon1],
                            )
        self.tp, tmodelV, self.niters, self.ier, self.mesg = result
        print 'mesg=%r, ier=%r' % (self.mesg, self.ier)

    def scalc(self, x, y, V):
        """Calculate least squares of spatial model"""
        self.x = x
        self.y = y
        #self.V = V # don't overwrite, leave self.V as raw voltages, not tmodelled ones
        result = fmin_slsqp(self.scost, self.sp0, args=(x, y, V),
                            bounds=[],
                            eqcons=[],
                            ieqcons=[],
                            )
        self.sp, smodelV, self.niters, self.ier, self.mesg = result
        print 'mesg=%r, ier=%r' % (self.mesg, self.ier)


class NMPFit(SpikeModel):
    """Levenberg-Marquadt least-squares with nmpfit from NASA's STSCI Python pytools package.
    This one can handle constraints."""
    def tcalc(self, t, V):
        """Calculate least squares of temporal model"""
        self.t = t
        self.V = V
        i = 2*self.nchans
        """self.dmurange[0] <= dmu <= self.dmurange[1]"""
        p = nmpfit.mpfit(self.tcost, self.tp0, functkw={'t':t, 'V':V},
                         parinfo=None, fastnorm=1)
        print 'dont forget to try messing with fastnorm!'
        self.tp = p.params
        print 'output params seem to be unchanged wrt input params'

    def tcost(self, tp, fjac=None, t=None, V=None):
        """Distance of each point in temporal model to the target.
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        error = np.ravel(self.tmodel(tp, t) - V)
        sys.stdout.write('%.1f, ' % np.abs(error).sum())
        status = 0
        return [status, error]
'''

class Detector(object):
    """Event detector base class"""
    #DEFALGORITHM = 'BipolarAmplitude'
    DEFALGORITHM = 'DynamicMultiphasic'
    DEFTHRESHMETHOD = 'Dynamic'
    DEFNOISEMETHOD = 'median'
    DEFNOISEMULT = 3.5
    DEFFIXEDTHRESH = 40 # uV
    DEFFIXEDNOISEWIN = 1000000 # 1s
    DEFDYNAMICNOISEWIN = 10000 # 10ms
    DEFMAXNEVENTS = 0
    DEFBLOCKSIZE = 1000000 # us, waveform data block size
    DEFSLOCK = 150 # um
    DEFTLOCK = 300 # us
    DEFRANDOMSAMPLE = False

    MAXAVGFIRINGRATE = 1000 # Hz, assume no chan will trigger more than this rate of events on average within a block
    BLOCKEXCESS = 1000 # us, extra data as buffer at start and end of a block while searching for events. Only useful for ensuring event times within the actual block time range are accurate. Events detected in the excess are discarded

    def __init__(self, stream, chans=None,
                 threshmethod=None, noisemethod=None, noisemult=None, fixedthresh=None,
                 fixednoisewin=None, dynamicnoisewin=None,
                 trange=None, maxnevents=None, blocksize=None,
                 slock=None, tlock=None, randomsample=None):
        """Takes a data stream and sets various parameters"""
        self.srffname = stream.srffname # used to potentially reassociate self with stream on unpickling
        self.stream = stream
        self.chans = chans or range(self.stream.nchans) # None means search all channels
        #self.nchans = len(self.chans) # rather not bind this to self, cuz len(chans) can change between search() calls
        #self.fdm = DistanceMatrix(self.stream.probe.SiteLoc) # full channel distance matrix, ignores enabled self.chans, identical for all Detectors on the same probe
        self.threshmethod = threshmethod or self.DEFTHRESHMETHOD
        self.noisemethod = noisemethod or self.DEFNOISEMETHOD
        self.noisemult = noisemult or self.DEFNOISEMULT
        self.fixedthresh = fixedthresh or self.DEFFIXEDTHRESH
        self.fixednoisewin = fixednoisewin or self.DEFFIXEDNOISEWIN # us
        self.dynamicnoisewin = dynamicnoisewin or self.DEFDYNAMICNOISEWIN # us
        self.trange = trange or (stream.t0, stream.tend)
        self.maxnevents = maxnevents or self.DEFMAXNEVENTS # return at most this many events, applies across chans
        self.blocksize = blocksize or self.DEFBLOCKSIZE
        self.slock = slock or self.DEFSLOCK
        self.tlock = tlock or self.DEFTLOCK
        self.randomsample = randomsample or self.DEFRANDOMSAMPLE

    def search(self):
        """Search for events. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results
        TODO: remove any events that happen right at the first or last timepoint in the file,
        since we can't say when an interrupted rising or falling edge would've reached peak
        """
        t0 = time.clock()

        self.enabledSiteLoc = {}
        for chan in self.chans: # for all enabled chans
            self.enabledSiteLoc[chan] = self.stream.probe.SiteLoc[chan] # grab its (x, y) coordinate
        self.dm = DistanceMatrix(self.enabledSiteLoc) # distance matrix for the chans enabled for this search

        self.thresh = self.get_thresh() # this could probably go in __init__ without problems
        print '.get_thresh() took %.3f sec' % (time.clock()-t0)

        bs = self.blocksize
        bx = self.BLOCKEXCESS
        wavetranges, (bs, bx, direction) = self.get_blockranges(bs, bx)

        self.nevents = 0 # total num events found across all chans so far by this Detector, reset at start of every search
        self.sm = {} # dict of LeastSquares model objects, indexed by their modelled spike time
        self.events = [] # list of 2D event arrays returned by .searchblockthread(), one array per block

        ncpus = processing.cpuCount()
        nthreads = 1 # was ncpus + 1, getting some race conditions on multicore I think
        print 'ncpus: %d, nthreads: %d' % (ncpus, nthreads)
        pool = threadpool.ThreadPool(nthreads) # create a threading pool

        t0 = time.clock()
        for wavetrange in wavetranges:
            args = (wavetrange, direction)
            # TODO: handle exceptions
            request = threadpool.WorkRequest(self.searchblock, args=args, callback=self.handle_spikes)
            pool.putRequest(request)
            '''
            try:
                spikes = self.searchblock(*args)
                self.handle_spikes(spikes)
            except ValueError: # we've found all the events we need
                break # out of wavetranges loop
            '''
        print 'done queueing tasks'
        pool.wait()
        print 'tasks took %.3f sec' % (time.clock() - t0)
        #time.sleep(2) # pause so you can watch the worker threads in taskman before they exit

        try:
            events = np.concatenate(self.events, axis=1)
        except ValueError: # self.events is an empty list
            events = np.asarray(self.events)
            events.shape = (2, 0)
        print '\nfound %d events in total' % events.shape[1]
        print 'inside .search() took %.3f sec' % (time.clock()-t0)
        return events

    def searchblock(self, wavetrange, direction):
        """This is what a worker thread executes"""
        print 'searchblock(): self.nevents=%r, self.maxnevents=%r, wavetrange=%r, direction=%r' % (self.nevents, self.maxnevents, wavetrange, direction)
        if self.nevents >= self.maxnevents:
            raise ValueError # skip this iteration. TODO: this should really cancel all enqueued tasks
        tlo, thi = wavetrange # tlo could be > thi
        bx = self.BLOCKEXCESS
        cutrange = (tlo+bx, thi-bx) # range without the excess, ie time range of events to actually keep
        #print 'wavetrange: %r, cutrange: %r' % (wavetrange, cutrange)
        wave = self.stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed
        wave = wave[self.chans] # get a WaveForm with just the enabled chans
        nchans = len(self.chans) # number of enabled chans
        if self.randomsample:
            maxnevents = 1 # how many more we're looking for in the next block
        else:
            maxnevents = self.maxnevents - self.nevents

        # this should all be done in __init__ or at least in .search()?
        thresh = 50 # abs, in uV
        ppthresh = thresh + 30 # peak-to-peak threshold, abs, in uV
        dmurange = (0, 500) # allowed time difference between peaks of modelled spike
        twthresh = (-250, 750) # spike time window range, us, centered on threshold crossing
        tw = (-250, 750) # spike time window range, us, centered on 1st phase of spike
        trangeithresh = intround(twthresh / self.stream.tres) # spike time window range wrt thresh xing in number of timepoints
        trangei = intround(tw / self.stream.tres) # spike time window range wrt 1st phase in number of timepoints
        # want an nchan*2 array of [chani, x/ycoord]
        xycoords = [ self.enabledSiteLoc[chan] for chan in self.chans ] # (x, y) coords in chan order
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        siteloc = np.asarray([xcoords, ycoords]).T # [chani, (x, y)]
        '''
        TODO: would be nice to use some multichannel thresholding, instead of just single independent channel
            - e.g. obvious but small multichan spike at ptc15.87.23340
            - hyperellipsoidal?
            - take mean of sets of chans (say one set per chan, slock of chans around it), check when they exceed thresh, find max chan within that set at that time and report it as an event
            - or slide some filter across the data that not only checks for thresh, but ppthresh as well
        '''
        edges = np.diff(np.int8(abs(wave.data) >= thresh)) # indices where increasing or decreasing abs(signal) has crossed thresh
        events = np.where(np.transpose(edges == 1)) # indices of +ve edges, where increasing abs(signal) has crossed thresh
        events = np.transpose(events) # shape == (nti, 2), col0: ti, col1: chani, rows are sorted increasing in time

        lockout = np.zeros(nchans, dtype=np.int64) # holds time indices until which each enabled chani is locked out
        spikes = [] # list of spikes detected
        # threshold crossing event loop: events gives us indices into time and chans
        for ti, chani in events:
            print
            print 'trying thresh event at t=%d chan=%d' % (wave.ts[ti], self.chans[chani])
            if ti <= lockout[chani]: # is this thresh crossing timepoint locked out?
                print 'thresh event is locked out'
                continue # this event is locked out, skip to next event
            # get data window wrt threshold crossing
            ti0 = max(ti+trangeithresh[0], lockout[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            tiend = min(ti+trangeithresh[1], len(wave.ts)) # don't go further than last wave timepoint
            window = wave.data[chani, ti0:tiend]
            minti = window.argmin() # time of minimum in window, relative to ti0
            maxti = window.argmax() # time of maximum in window, relative to ti0
            phase1ti = min(minti, maxti) # wrt ti0
            phase1ti = ti0 + phase1ti # wrt 0th time index

            # find all the enabled chanis within slock of chani, exclude chanis temporally locked-out at phase1ti:
            chanis, = np.where(self.dm.data[chani] <= self.slock) # at what col indices does the returned row fall within slock?
            chanis = np.asarray([ chi for chi in chanis if lockout[chi] < phase1ti ])

            # find maxchan within chanis at phase1ti, redo all of the above calculations
            chanii = np.abs(wave.data[chanis, phase1ti]).argmax() # index into chanis of new maxchan
            chani = chanis[chanii] # new max chani
            chan = self.chans[chani] # new max chan
            print 'new max chan=%d' % chan
            # get new data window wrt phase1ti this time, instead of wrt the original thresh xing
            ti0 = max(phase1ti+trangei[0], lockout[chani]+1) # make sure any timepoints included prior to phase1ti aren't locked out
            tiend = min(phase1ti+trangei[1], len(wave.ts)) # don't go further than last wave timepoint
            window = wave.data[chani, ti0:tiend]
            minti = window.argmin() # time of minimum in window, relative to ti0
            maxti = window.argmax() # time of maximum in window, relative to ti0
            minV = window[minti]
            maxV = window[maxti]
            phase1ti = min(minti, maxti) # now it's back to wrt ti0 again
            phase2ti = max(minti, maxti)
            phase1V = window[phase1ti]
            phase2V = window[phase2ti]

            # again, find all the enabled chanis within slock of new chani, exclude chanis temporally locked-out at phase1ti:
            chanis, = np.where(self.dm.data[chani] <= self.slock) # at what col indices does the returned row fall within slock?
            chanis = np.asarray([ chi for chi in chanis if lockout[chi] < ti0 ])

            print 'window params: t0=%r, phase1t=%r, tend=%r, mint=%r, maxt=%r, phase1V=%r, phase2V=%r' % \
                (wave.ts[ti0], wave.ts[ti0+phase1ti], wave.ts[tiend], wave.ts[ti0+minti], wave.ts[ti0+maxti], phase1V, phase2V)
            # check if this (still roughly defined) event crosses ppthresh, and some other requirements,
            # should help speed things up by rejecting obviously invalid events without having to run the model
            try:
                assert abs(phase2V - phase1V) >= ppthresh, "event doesn't cross ppthresh"
                assert np.sign(phase1V) == -np.sign(phase2V), 'phases must be of opposite sign'
                assert minV < 0, 'minV is %s V at t = %d' % (minV, wave.ts[ti0+minti])
                assert maxV > 0, 'maxV is %s V at t = %d' % (maxV, wave.ts[ti0+maxti])
            except AssertionError, message: # doesn't qualify as a spike
                print message
                continue

            # create a SpikeModel
            sm = NLLSP()
            sm.chans, sm.maxchani, sm.chanis, sm.nchans = self.chans, chani, chanis, nchans
            sm.maxchanii, = np.where(sm.chanis == sm.maxchani) # index into chanis that returns maxchani
            sm.dmurange = dmurange
            print 'chans = %r' % (np.asarray(self.chans)[chanis],)
            print 'chanis = %r' % (chanis,)
            t = wave.ts[ti0:tiend]
            x = siteloc[chanis, 0]
            y = siteloc[chanis, 1]
            V = wave.data[chanis, ti0:tiend]
            # TODO: take weighted spatial mean of chanis at phase1ti to better estimate (x0, y0)
            p0 = [phase1V, wave.ts[ti0+phase1ti], 60, # 1st phase: phase1V (uV), mu1 (us), s1 (us)
                  phase2V, wave.ts[ti0+phase2ti], 60, # 2nd phase: phase2V (uV), mu1 (us), s2 (us)
                  siteloc[chani, 0], # x0 (um)
                  siteloc[chani, 1], # y0 (um)
                  60, 60, # sx, sy (um)
                  0] # vinv (us/um, ie s/m)
            '''
            if wave.ts[ti] == 26880:
                p0[6], p0[7], p0[10] = 0, 780, -0.5 # chan 7
            '''
            sm.p0 = p0
            sm.calc(t, x, y, V) # calculate spatiotemporal fit
            print '      V1,  mu1, s1,  V2,  mu2, s2,  x0,   y0, sx, sy, vinv'
            print 'p0 = %r' % list(intround(sm.p0))
            print 'p = %r' % list(intround(sm.p))
            """
            The peak times of the modelled f'n may not correspond to the peak times of the two phases.
            Their amplitudes certainly need not correspond. So, here I'm reading values off of the modelled
            waveform instead of just the parameters of the constituent Gaussians that make it up
            """
            phase1V, mu1, s1, phase2V, mu2, s2, x0, y0, sx, sy, vinv = sm.p
            modelV = sm.model(sm.p, sm.t, x0, y0).ravel()
            modelminti = np.argmin(modelV)
            modelmaxti = np.argmax(modelV)
            phase1ti = min(modelminti, modelmaxti) # 1st phase might be the min or the max
            phase2ti = max(modelminti, modelmaxti) # 2nd phase might be the min or the max
            phase1t = t[phase1ti]
            phase2t = t[phase2ti]
            phase1V = modelV[phase1ti]
            phase2V = modelV[phase2ti]
            absphaseV = abs(np.array([phase1V, phase2V]))
            bigphase = max(absphaseV)
            smallphase = min(absphaseV)

            self.sm[phase1t] = sm # save the SpikeModel object for later inspection
            sm.spiket = phase1t
            sm.phase1t = phase1t # synonym for spike time
            sm.phase2t = phase2t

            # check to see if modelled spike qualifies as an actual spike
            try:
                # ensure modelled spike time doesn't violate any existing lockout on any of its modelled chans
                assert (lockout[chanis] < ti0+phase1ti).all(), 'model spike time is locked out'
                assert wave.ts[ti0] < phase1t < wave.ts[tiend], "model spike time doesn't fall within time window"
                assert bigphase >= thresh, "model doesn't cross thresh (bigphase=%r)" % bigphase
                assert abs(phase2V - phase1V) >= ppthresh, "model doesn't cross ppthresh"
                dphase = phase2t - phase1t
                assert dmurange[0] <= dphase <= dmurange[1], 'model phases separated by %f us (outside of dmurange=%r)' % (dphase, dmurange)
                assert np.sign(phase1V) == -np.sign(phase2V), 'model phases must be of opposite sign'
            except AssertionError, message: # doesn't qualify as a spike
                print '%s, spiket=%d' % (message, phase1t)
                continue
            # it's a spike, record it
            spike = (phase1t, x0, y0) # (time, x0, y0) tuples
            spikes.append(spike)
            print 'found new spike: %r' % (list(intround(spike)),)
            # update spatiotemporal lockout
            # TODO: maybe apply the same 2D gaussian spatial filter to the lockout in time, so chans further away
            # are locked out for a shorter time. Use slock as a circularly symmetric spatial sigma
            # TODO: center lockout on model (x0, y0) coords, instead of max chani - this could be dangerous - if model got it wrong, could get a whole lotta false +ve spike detections due to spatial lockout being way off
            # TODO: lock each chan out separately wrt to its phase2ti, since all chans can now be potentially delayed wrt source - this will require calculating each channel's phase2ti separately...
            lockout[chanis] = ti0 + phase2ti + intround(s2 / self.stream.tres) # lock out til one stdev after peak of 2nd phase, in case there's a noisy mini spike that might cause a trigger on the way down
            print 'lockout for chanis = %r' % wave.ts[lockout[chanis]]

        spikes = np.asarray(spikes)
        # trim results from wavetrange down to just cutrange
        ts = spikes[:, 0] # spike times are in 0th column
        # searchsorted might be faster here instead of checking each and every element
        spikeis = (cutrange[0] < ts) * (ts < cutrange[1]) # boolean array
        spikes = spikes[spikeis]
        return spikes

    def handle_spikes(self, request, spikes):
        """Blocking callback, called every time a worker thread completes a task"""
        print 'handle_spikes got: %r' % spikes
        if spikes == None:
            return
        nnewevents = spikes.shape[1] # number of columns
        #wx.Yield() # allow GUI to update
        if self.randomsample and spikes.tolist() in np.asarray(self.events).tolist():
            # check if spikes is a duplicate of any that are already in .events, if so,
            # don't append this new spikes array, and don't inc self.nevents. Duplicates are possible
            # in random sampling cuz we might end up with blocks with overlapping tranges.
            # Converting to lists for the check is probably slow cuz, but at least it's legible and correct
            sys.stdout.write('found duplicate random sampled event')
        elif nnewevents != 0:
            self.events.append(spikes)
            self.nevents += nnewevents # update
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
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            thresh = np.ones(len(self.chans), dtype=np.float32) * self.fixedthresh
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed thresh, calculate from start of stream
            """randomly sample DEFFIXEDNOISEWIN's worth of data from the entire file in blocks of self.blocksize
            NOTE: this samples with replacement, so it's possible, though unlikely, that some parts of the data
            will contribute more than once to the noise calculation
            This sometimes causes an 'unhandled exception' for BipolarAmplitude algorithm, don't know why
            """
            print 'TODO: ChanFixed needs to respect enabled self.chans!'
            nblocks = intround(self.DEFFIXEDNOISEWIN / self.blocksize)
            wavetranges = RandomWaveTranges(self.trange, bs=self.blocksize, bx=0, maxntranges=nblocks)
            data = []
            for wavetrange in wavetranges:
                data.append(self.stream[wavetrange[0]:wavetrange[1]].data)
            data = np.concatenate(data, axis=1)
            noise = self.get_noise(data)
            thresh = noise * self.noisemult
        elif self.threshmethod == 'Dynamic':
            thresh = np.zeros(len(self.chans), dtype=np.float32) # this will be calculated on the fly in the Cython loop
        else:
            raise ValueError
        print 'thresh = %s' % thresh
        assert thresh.dtype == np.float32
        return thresh

    def get_noise(self, data):
        """Calculates noise over last dim in data (time), using .noisemethod"""
        if self.noisemethod == 'median':
            return np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            return np.stdev(data, axis=-1)
        else:
            raise ValueError


class BipolarAmplitude(Detector):
    """Bipolar amplitude detector"""
    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.algorithm = 'BipolarAmplitude'


class DynamicMultiphasic(Detector):
    """Dynamic multiphasic detector"""
    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.algorithm = 'DynamicMultiphasic'
