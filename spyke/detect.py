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
    """Least squares Levenburg-Marquodt fit of two voltage Gaussians
    to spike phases, plus a 2D spatial gaussian to model decay across channels"""
    def __init__(self):
        # initial parameter guess
        self.p0 = [-50, 150,  60, # 1st phase: amplitude (uV), mu (us), sigma (us)
                    50, 300, 120, # 2nd phase: amplitude (uV), mu (us), sigma (us)
                    None, # x (um)
                    None, # y (um)
                    60] # sigma_x == sigma_y (um)

    def calc(self, t, x, y, V):
        result = leastsq(self.cost, self.p0, args=(t, x, y, V), Dfun=None, full_output=True, col_deriv=False)
        self.p, self.cov_p, self.infodict, self.mesg, self.ier = result
        return self.p

    def model(self, p, t, x, y):
        """Sum of two Gaussians in time, modulated by a 2D spatial Gaussian
        returns a vector of voltage values v of same length as t. x and y are
        vectors of x and y coordinates of each channel's spatial location. Output
        of this should be an (nchans, nt) matrix of modelled voltage values v"""
        return np.outer(g2(p[6], p[7], p[8], p[8], x, y),
                        p[0]*g(p[1], p[2], t) + p[3]*g(p[4], p[5], t))

    def cost(self, p, t, x, y, V):
        """Distance of each point to the 2D target function
        Returns a matrix of errors, channels in rows, timepoints in columns.
        Seems the resulting matrix has to be flattened into an array"""
        return np.ravel(self.model(p, t, x, y) - V)



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
    DEFSLOCK = 175 # um
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
        nthreads = ncpus + 1 # not too sure why, so you always have a worker thread waiting in the wings?
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
        time.sleep(2) # pause so you can watch the parent thread in taskman hang around after worker thread exit

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
        thresh1 = 50 # abs, in uV
        thresh2 = 30 # abs, in uV
        dmurange = (100, 500) # time difference between means of spike phase Gaussian, us
        tw = 1500 # spike time window, us
        twnt = intround(tw / self.stream.tres) # spike time window in number of timepoints
        # self.stream.probe.SiteLoc is dict of chan:tuple
        # want a nchan*2 array of [chani, x/ycoord]
        xycoords = [ self.stream.probe.SiteLoc[chan] for chan in self.stream.chans ]
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        siteloc = np.asarray([xcoords, ycoords]).T # [chani, x/ycoord]

        # this should hopefully release the GIL
        edges = np.diff(np.int8(abs(wave.data) > thresh1)) # indices where increasing or decreasing abs(signal) has crossed thresh1
        events = np.where(np.transpose(edges == 1)) # indices of +ve edges, where increasing abs(signal) has crossed thresh1
        # TODO: filter events somehow in chan space using slock, so that you don't get more than one chan to test for a given event, even if it exceeds thresh on multiple chans - this will speed up the loop below by reducing unnecessary fitting runs
        events = np.transpose(events) # shape == (nti, 2), col0: ti, col1: chani, rows are sorted increasing in time

        lockout = np.zeros(self.stream.nchans) # holds time indices until which each channel is locked out

        spikes = []
        ls = LeastSquares()
        for ti, chani in events: # for all threshold crossing events
            print
            print 'trying thresh event at t=%d chan=%d' % (wave.ts[ti], wave.chans[chani])
            if wave.ts[ti] <= lockout[chani]: # is this thresh crossing time locked out?
                print 'thresh event is locked out'
                continue # this event is locked out, skip to next event
            # find max and min for short interval following threshold crossing
            # might need to reduce ti a couple of points from thresh crossing for a nice gaussian fit
            chan = wave.chans[chani]
            ti0 = ti # for now at least
            tiend = ti0+twnt
            window = wave.data[chani, ti0:tiend]
            minti = window.argmin() # time of minimum in window, relative to ti0
            maxti = window.argmax() # time of maximum in window, relative to ti0
            minV = window[minti]
            maxV = window[maxti]
            phase1ti = min(minti, maxti)
            phase2ti = max(minti, maxti)
            phase1V = window[phase1ti]
            phase2V = window[phase2ti]
            # check if this (still roughly defined) event crosses thresh2, should help speed things up by rejecting obviously invalid events without having to run the model
            if abs(phase2V) < thresh2:
                print "event doesn't cross thresh2"
                continue
            assert minV < 0, 'minV is %s V at t = %d' % (minV, wave.ts[ti0+minti])
            assert maxV > 0, 'maxV is %s V at t = %d' % (maxV, wave.ts[ti0+maxti])
            p0 = [phase1V, wave.ts[ti0+phase1ti], 60, # 1st phase: amplitude (uV), mu (us), sigma (us)
                  phase2V, wave.ts[ti0+phase2ti], 120, # 2nd phase: amplitude (uV), mu (us), sigma (us)
                  self.stream.probe.SiteLoc[chan][0], # x (um)
                  self.stream.probe.SiteLoc[chan][1], # y (um)
                  60] # sigma_x == sigma_y (um)
            ls.p0 = p0
            print 'p0 = %r' % ([ intround(val) for val in ls.p0 ],)
            # find all the chans within slock of chani
            # TODO: this should probably exclude locked-out channels as well
            chanis, = np.where(dmi[chani] <= self.slock)
            t = wave.ts[ti0:tiend]
            x = siteloc[chanis, 0]
            y = siteloc[chanis, 1]
            V = wave.data[chanis, ti0:tiend]
            p = ls.calc(t, x, y, V) # calculate least squares fit
            print 'p = %r' % ([ intround(val) for val in ls.p ],)
            # TODO: I should report some kind of measure of fit error here
            phase1i = np.argmin([ls.p[1], ls.p[4]]) # what was init'd as 1st phase may not have come out as such
            phase2i = np.argmax([ls.p[1], ls.p[4]])
            phase1t = [ls.p[1], ls.p[4]][phase1i]
            phase2t = [ls.p[1], ls.p[4]][phase2i]
            phase1V = [ls.p[0], ls.p[3]][phase1i]
            phase2V = [ls.p[0], ls.p[3]][phase2i]
            # check params to see if event qualifies as spike
            try:
                assert abs(phase1V) >= thresh1, 'thresh1 not crossed by model'
                assert abs(phase2V) >= thresh2, 'thresh2 not crossed by model'
                assert np.sign(phase1V) == -np.sign(phase2V), 'phases must be of opposite sign'
                dphase = phase2t - phase1t
                assert dmurange[0] <= dphase <= dmurange[1], 'phases separated by %f us' % dphase
                # should probably add another here to ensure that (x, y) are reasonably close to within probe boundaries
            except AssertionError, message: # doesn't qualify as a spike
                print message
                continue
            # it's a spike, record it
            spikes.append((phase1t, ls.p[6], ls.p[7]))  # list of (time, x, y) tuples
            print 'found new spike: %r' % ([ intround(val) for val in spikes[-1] ],)
            # update spatiotemporal lockout
            # TODO: maybe apply the same 2D gaussian spatial filter to the lockout in time, so chans further away
            # are locked out for a shorter time, where slock is the circularly symmetric spatial sigma
            # TODO: center lockout on model x, y fit params, instead of chani that crossed thresh first
            lockout[chanis] = max(ls.p[1], ls.p[4]) # lock out til peak of last phase (1st or 2nd depending on what the model did)

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
