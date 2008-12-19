"""Event detection algorithms
"""

from __future__ import division

__authors__ = ['Martin Spacek, Reza Lotun']

import itertools
import sys
import time
import processing
import threadpool

import wx

import numpy as np
from scipy.optimize import leastsq

from pylab import *

import spyke.surf
from spyke.core import WaveForm, toiter, argcut, intround, eucd, g, g2


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
        # random int within trange
        if self.maxntranges != None and self.ntranges >= self.maxntranges:
            raise StopIteration
        t0 = np.random.randint(low=self.trange[0], high=self.trange[1])
        tend = t0 + self.bs
        self.ntranges += 1
        return (t0-self.bx, tend+self.bx)

    def __iter__(self):
        return self


class LeastSquares(object):
    """Least squares Levenberg-Marquardt fit of two voltage Gaussians
    to spike phases, plus a 2D spatial gaussian to model decay across channels"""
    def __init__(self):
        # initial parameter guess
        '''
        self.p0 = [-50, 150,  60, # 1st phase: amplitude (uV), mu (us), sigma (us)
                    50, 300, 120, # 2nd phase: amplitude (uV), mu (us), sigma (us)
                    None, # x (um)
                    None, # y (um)
                    60] # sigma_x == sigma_y (um)
        '''
        self.step = [0.1, 0.1, 1,
                     0.1, 0.1, 1,
                     0.2,
                     0.2,
                     0.1]

    def plot(self):
        t = self.t
        p = self.p
        for (V, x, y) in zip(self.V, self.x, self.y):
            figure()
            title('x, y = %r um' % ((x, y),))
            plot(t, V, 'k.-')
            plot(t,
                 g2(p[6], p[7], p[8], p[8], x, y) * p[0]*g(p[1], p[2], t),
                 'r-')
            plot(t,
                 g2(p[6], p[7], p[8], p[8], x, y) * p[3]*g(p[4], p[5], t),
                 'g-')
            plot(t,
                 g2(p[6], p[7], p[8], p[8], x, y) * (p[0]*g(p[1], p[2], t) + p[3]*g(p[4], p[5], t)),
                 'b-')
            gca().autoscale_view(tight=True, scalex=True, scaley=False) # tight fit to timepoints
            gca().set_ylim(-100, 100)
        figure()
        title('x, y = %r (model origin in space)' % ((p[6], p[7]),))
        plot(t, p[0]*g(p[1], p[2], t) + p[3]*g(p[4], p[5], t), 'm-')
        gca().autoscale_view(tight=True, scalex=True, scaley=False) # tight fit to timepoints
        gca().set_ylim(-100, 100)

    def calc(self, t, x, y, V):
        self.t = t
        self.x = x
        self.y = y
        self.V = V
        result = leastsq(self.cost, self.p0, args=(t, x, y, V),
                         Dfun=None, full_output=True, col_deriv=False,
                         maxfev=50, xtol=0.0001,
                         diag=None)
        #self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        self.p, self.infodict, self.mesg, self.ier = result
        print '%d iterations' % self.infodict['nfev']
        print 'mesg=%r, ier=%r' % (self.mesg, self.ier)

    def model(self, p, t, x, y):
        """Sum of two Gaussians in time, modulated by a 2D spatial Gaussian.
        For each channel, returns a vector of voltage values v of same length as t.
        x and y are vectors of coordinates of each channel's spatial location.
        Output should be an (nchans, nt) matrix of modelled voltage values V"""
        return np.outer(g2(p[6], p[7], p[8], p[8], x, y),
                        p[0]*g(p[1], p[2], t) + p[3]*g(p[4], p[5], t))

    def cost(self, p, t, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        return np.ravel(self.model(p, t, x, y) - V)
    '''
    def dcost(self, p, t, x, y, V):
        """Derivative of cost function wrt each parameter, returns Jacobian matrix"""
        # these all have the same length as t
        dfdp0 = np.ravel(np.outer(g2(p[6], p[7], p[8], p[8], x, y), g(p[1], p[2], t)))
        dfdp1 = p[0]*dgdmu(p[1], p[2], t)
        dfdp2 = p[0]*dgdsigma(p[1], p[2], t)
        dfdp3 = g(p[4], p[5], t)
        dfdp4 = p[3]*dgdmu(p[4], p[5], t)
        dfdp5 = p[3]*dgdsigma(p[4], p[5], t)
        dfdp6
        dfdp7
        dfdp8
        return np.asarray([dfdp0, dfdp1, dfdp2, dfdp3, dfdp4, dfdp5])
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
    DEFSLOCK = 100 # um
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
        self.nchans = len(self.chans)
        self.dm = self.get_full_chan_distance_matrix() # channel distance matrix, identical for all Detectors on the same probe
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

        self.thresh = self.get_thresh() # this could probably go in __init__ without problems
        print '.get_thresh() took %.3f sec' % (time.clock()-t0)

        bs = self.blocksize
        bx = self.BLOCKEXCESS
        # reset this at the start of every search
        self.nevents = 0 # total num events found across all chans so far by this Detector

        wavetranges, (bs, bx, direction) = self.get_blockranges(bs, bx)

        self.events = [] # list of 2D event arrays returned by .searchblockthread(), one array per block

        ncpus = processing.cpuCount()
        nthreads = ncpus # not too sure why, so you always have a worker thread waiting in the wings?
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
        # TODO: pull out just the enabled channels here: wave = wave[enabledchanis]
        dmi = self.dm[wave.chans][:, wave.chans] # channel distance matrix indexed into by chani instead of chan
        if self.randomsample:
            maxnevents = 1 # how many more we're looking for in the next block
        else:
            maxnevents = self.maxnevents - self.nevents

        # this should all be done in __init__ ?
        thresh = 50 # abs, in uV
        ppthresh = thresh + 30 # peak-to-peak threshold, abs, in uV
        dmurange = (0, 500) # time difference between means of spike phase Gaussians, us
        tw = (-250, 750) # spike time window range, us, centered on threshold crossing, maybe this should be a dynamic param, customized for each thresh crossing event, maybe based on mean (or median?) signal around the event
        twnt = intround(tw / self.stream.tres) # spike time window range in number of timepoints
        # self.stream.probe.SiteLoc is dict of chan:tuple
        # want a nchan*2 array of [chani, x/ycoord]
        xycoords = [ self.stream.probe.SiteLoc[chan] for chan in self.stream.chans ]
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        SiteLoc = np.asarray([xcoords, ycoords]).T # [chani, x/ycoord]

        # this should hopefully release the GIL
        # would be nice to use some multichannel thresholding, instead of just single independent channel
            # - e.g. obvious but small multichan spike at t=23340 on chan 41 in file ptc15/87
            # - hyperellipsoidal?
            # - take mean of sets of chans (say one set per chan, slock of chans around it), check when they exceed thresh, find max chan within that set at that time and report it as an event
            # - or slide some filter across the data that not only checks for thresh, but ppthresh as well
        edges = np.diff(np.int8(abs(wave.data) >= thresh)) # indices where increasing or decreasing abs(signal) has crossed thresh
        events = np.where(np.transpose(edges == 1)) # indices of +ve edges, where increasing abs(signal) has crossed thresh
        # TODO: filter events somehow in chan space using slock, so that you don't get more than one chan to test for a given event, even if it exceeds thresh on multiple chans - this will speed up the loop below by reducing unnecessary fitting runs
        events = np.transpose(events) # shape == (nti, 2), col0: ti, col1: chani, rows are sorted increasing in time

        lockouti = np.zeros(self.stream.nchans, dtype=np.int64) # holds time indices until which each channel is locked out

        spikes = [] # list of spikes detected
        self.ls = [] # list of LeastSquares models used
        ls = LeastSquares()
        for ti, chani in events: # for all threshold crossing events
            print
            print 'trying thresh event at t=%d chan=%d' % (wave.ts[ti], wave.chans[chani])
            if ti <= lockouti[chani]: # is this thresh crossing time locked out?
                print 'thresh event is locked out'
                continue # this event is locked out, skip to next event
            chan = wave.chans[chani]
            # find max short interval within time window of threshold crossing
            ti0 = max(ti+twnt[0], lockouti[chani]+1) # make sure any timepoints you're including prior to ti aren't locked out
            tiend = ti+twnt[1]
            window = wave.data[chani, ti0:tiend]
            #absmaxti = abs(window).argmax() # timepoint index of absolute maximum in window, relative to ti0
            #print 'original chan=%d has max %.1f' % (chan, window[absmaxti])
            # search for maxchan within slock at absmaxti
            chanis, = np.where(dmi[chani] <= self.slock)
            chanis = np.asarray([ chi for chi in chanis if lockouti[chani] < ti0 ]) # exclude any locked out channels from search
            #chani = chanis[ wave.data[chanis, absmaxti].argmax() ] # this is our new maxchan
            #chan = wave.chans[chani] # update
            #print 'new max chan=%d' % (chan)
            # find max and min within the time window, given (possibly new) maxchan
            #window = wave.data[chani, ti0:tiend]
            minti = window.argmin() # time of minimum in window, relative to ti0
            maxti = window.argmax() # time of maximum in window, relative to ti0
            minV = window[minti]
            maxV = window[maxti]
            phase1ti = min(minti, maxti)
            phase2ti = max(minti, maxti)
            phase1V = window[phase1ti]
            phase2V = window[phase2ti]
            print 'window params: t0=%r, tend=%r, mint=%r, maxt=%r, phase1V=%r, phase2V=%r' % \
                (wave.ts[ti0], wave.ts[tiend], wave.ts[ti0+minti], wave.ts[ti0+maxti], phase1V, phase2V)
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
            p0 = [phase1V, wave.ts[ti0+phase1ti], 60, # 1st phase: amplitude (uV), mu (us), sigma (us)
                  phase2V, wave.ts[ti0+phase2ti], 120, # 2nd phase: amplitude (uV), mu (us), sigma (us)
                  self.stream.probe.SiteLoc[chan][0], # x (um)
                  self.stream.probe.SiteLoc[chan][1], # y (um)
                  60] # sigma_x == sigma_y (um)
            ls.p0 = p0
            # find all the chans within slock of chani, exclude locked-out channels
            # TODO: exclude grounded channels, or maybe those should just be deselected?
            chanis, = np.where(dmi[chani] <= self.slock)
            chanis = [ chi for chi in chanis if lockouti[chani] < ti0 ]
            t = wave.ts[ti0:tiend]
            x = SiteLoc[chanis, 0]
            y = SiteLoc[chanis, 1]
            V = wave.data[chanis, ti0:tiend]
            ls.calc(t, x, y, V) # calculate least squares fit
            print 'leastsq got chanis = %r' % (chanis,)
            print 'p0 = %r' % (list(intround(ls.p0)),)
            print 'p = %r' % (list(intround(ls.p)),)
            self.ls.append(ls)
            # TODO: I should report some kind of measure of fit error here, and if error is big, plot the model on top of the data
            '''
            phase1i = np.argmin([ls.p[1], ls.p[4]]) # what was init'd as 1st phase may not have come out as such
            phase2i = np.argmax([ls.p[1], ls.p[4]])
            phase1t = [ls.p[1], ls.p[4]][phase1i]
            phase2t = [ls.p[1], ls.p[4]][phase2i]
            phase1V = [ls.p[0], ls.p[3]][phase1i]
            phase2V = [ls.p[0], ls.p[3]][phase2i]
            '''
            # the peak times of the modelled f'n may not correspond to the peak times of the two phases - their amplitudes certainly need not correspond. So, here I'm reading values off of the sum of Gaussians modelled f'n instead of the constituent Gaussians that make it up
            # get max and min modelled voltages at the modelled location
            modelV = ls.model(ls.p, t, ls.p[6], ls.p[7]).ravel()
            modelminti = np.argmin(modelV)
            modelmaxti = np.argmax(modelV)
            modelmint = t[modelminti]
            modelmaxt = t[modelmaxti]
            modelminV = modelV[modelminti]
            modelmaxV = modelV[modelmaxti]
            phase1i = np.argmin([modelminti, modelmaxti]) # 1st phase might be the min or the max
            phase2i = np.argmax([modelminti, modelmaxti]) # 2nd phase might be the min or the max
            phase1ti = [modelminti, modelmaxti][phase1i]
            phase2ti = [modelminti, modelmaxti][phase2i]
            phase1t = t[phase1ti]
            phase2t = t[phase2ti]
            phase1V = [modelminV, modelmaxV][phase1i]
            phase2V = [modelminV, modelmaxV][phase2i]
            bigphase = max(abs(modelminV), abs(modelmaxV))
            smallphase = min(abs(modelminV), abs(modelmaxV))
            # check params to see if event qualifies as spike
            try:
                assert bigphase >= thresh, "model doesn't cross thresh (bigphase=%r)" % bigphase
                assert abs(phase2V - phase1V) >= ppthresh, "model doesn't cross ppthresh"
                assert np.sign(phase1V) == -np.sign(phase2V), 'model phases must be of opposite sign'
                dphase = phase2t - phase1t
                assert dmurange[0] <= dphase <= dmurange[1], 'model phases separated by %f us (outside of dmurange=%r)' % (dphase, dmurange)
                # should probably add another here to ensure that (x, y) are reasonably close to within probe boundaries
            except AssertionError, message: # doesn't qualify as a spike
                print message
                continue
            # it's a spike, record it
            spike = (phase1t, ls.p[6], ls.p[7]) # (time, x, y) tuples
            spikes.append(spike)
            print 'found new spike: %r' % (list(intround(spike)),)
            # update spatiotemporal lockout
            # TODO: maybe apply the same 2D gaussian spatial filter to the lockout in time, so chans further away
            # are locked out for a shorter time, where slock is the circularly symmetric spatial sigma
            # TODO: center lockout on model x, y fit params, instead of chani that crossed thresh first
            lockouti[chanis] = intround(phase2t / self.stream.tres) # lock out til peak of 2nd phase

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

    def get_full_chan_distance_matrix(self):
        """Get full channel distance matrix, in um"""
        chans_coords = self.stream.probe.SiteLoc.items() # list of tuples
        chans_coords.sort() # sort by chanid
        # TODO: what if this probe is missing some channel ids, ie chans aren't consecutive in layout?
        # That'll screw up indexing into the distance matrix, unless we insert dummy entries in the matrix
        # for those chans missing from the layout
        chans = [ chan_coord[0] for chan_coord in chans_coords ] # pull out the sorted chans and check them
        assert chans == range(len(chans)), 'is probe layout channel list not consecutive starting from 0?'
        coords = [ chan_coord[1] for chan_coord in chans_coords ] # pull out the coords, now in channel id order
        return eucd(coords)

    def get_thresh(self):
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            thresh = np.ones(self.nchans, dtype=np.float32) * self.fixedthresh
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed thresh, calculate from start of stream
            """randomly sample DEFFIXEDNOISEWIN's worth of data from the entire file in blocks of self.blocksize
            NOTE: this samples with replacement, so it's possible, though unlikely, that some parts of the data
            will contribute more than once to the noise calculation
            This sometimes causes an 'unhandled exception' for BipolarAmplitude algorithm, don't know why
            """
            nblocks = intround(self.DEFFIXEDNOISEWIN / self.blocksize)
            wavetranges = RandomWaveTranges(self.trange, bs=self.blocksize, bx=0, maxntranges=nblocks)
            data = []
            for wavetrange in wavetranges:
                data.append(self.stream[wavetrange[0]:wavetrange[1]].data)
            data = np.concatenate(data, axis=1)
            noise = self.get_noise(data)
            thresh = noise * self.noisemult
        elif self.threshmethod == 'Dynamic':
            thresh = np.zeros(self.nchans, dtype=np.float32) # this will be calculated on the fly in the Cython loop
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
