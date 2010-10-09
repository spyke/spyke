"""Spike detection and modelling"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

# this stuff needs to be near the top apparently
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import spyke.util # .pyx file

import sys
import time
import logging
import datetime
import multiprocessing as mp
from copy import copy

import wx

from scipy.weave import inline
#from scipy import ndimage
#from scipy.optimize import leastsq, fmin_slsqp
#import openopt
#import nmpfit

import spyke.surf
from spyke.core import eucd

#from spyke import threadpool
#from spyke.core import  WaveForm, toiter, argcut, intround, g, g2, RM
#from text import SimpleTable

#DMURANGE = 0, 500 # allowed time difference between peaks of modelled spike

logger = logging.Logger('detection')
shandler = logging.StreamHandler(strm=sys.stdout) # prints to screen
formatter = logging.Formatter('%(message)s')
shandler.setFormatter(formatter)
shandler.setLevel(logging.INFO) # log info level and higher to screen
logger.addHandler(shandler)
info = logger.info

DEBUG = False # print detection debug messages to log file? slows down detection

if DEBUG:
    # print detection info and debug msgs to file, and info msgs to screen
    dt = str(datetime.datetime.now()) # get a timestamp
    dt = dt.split('.')[0] # ditch the us
    dt = dt.replace(' ', '_')
    dt = dt.replace(':', '.')
    logfname = dt + '_detection.log'
    logf = open(logfname, 'w')
    fhandler = logging.StreamHandler(strm=logf) # prints to file
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.DEBUG) # log debug level and higher to file
    logger.addHandler(fhandler)
    debug = logger.debug


def callsearchblock(args):
    """Apply args to the Detector saved to the current process"""
    wavetrange, direction = args
    detector = mp.current_process().detector
    return detector.searchblock(wavetrange, direction)

def initializer(detector):
    """Save pickled copy of the Detector to the current process"""
    mp.current_process().detector = detector


class FoundEnoughSpikesError(ValueError):
    pass


class RandomWaveTranges(object):
    """Iterator that spits out time ranges of width bs with
    excess bx that begin randomly from within the given trange.
    Optionally spits out no more than maxntranges tranges"""
    def __init__(self, trange, bs, bx=0, maxntranges=None, replacement=True):
        self.trange = trange
        self.bs = bs
        self.bx = bx
        self.maxntranges = maxntranges
        self.ntranges = 0
        self.replacement = replacement
        # pool of possible start values of time ranges, all aligned to start of overall trange
        if not replacement:
            self.t0pool = np.arange(self.trange[0], self.trange[1], bs)

    def next(self):
        # on each iter, need to remove intpool values from t0-width to t0+width
        # use searchsorted to find out indices of these values in intpool, then use
        # those indices to remove the values: intpool = np.delete(intpool, indices)
        if self.maxntranges != None and self.ntranges >= self.maxntranges:
            raise StopIteration
        if not self.replacement and len(self.t0pool) == 0:
            raise StopIteration
        if self.replacement: # sample with replacement
            # start from random int within trange
            t0 = np.random.randint(low=self.trange[0], high=self.trange[1]-self.bs)
        else: # sample without replacement
            t0i = np.random.randint(low=0, high=len(self.t0pool)) # returns value < high
            t0 = self.t0pool[t0i]
            self.t0pool = np.delete(self.t0pool, t0i) # remove from pool
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
        self.chans = np.uint8([ chan_coord[0] for chan_coord in chans_coords ]) # pull out the sorted chans
        self.coords = [ chan_coord[1] for chan_coord in chans_coords ] # pull out the coords, now in chan order
        self.data = eucd(self.coords)


class Detector(object):
    """Spike detector base class"""
    DEFTHRESHMETHOD = 'Dynamic' # GlobalFixed, ChanFixed, or Dynamic
    DEFNOISEMETHOD = 'median' # median or stdev
    DEFNOISEMULT = 6
    DEFFIXEDTHRESHUV = 50 # uV, used by GlobalFixed, and as min thresh for ChanFixed
    DEFPPTHRESHMULT = 1.5 # peak-to-peak threshold is this times thresh
    DEFFIXEDNOISEWIN = 30000000 # 30s, used by ChanFixed - this should really be a % of self.trange
    DEFDYNAMICNOISEWIN = 10000 # 10ms, used by Dynamic
    DEFMAXNSPIKES = 0
    DEFBLOCKSIZE = 10000000 # 10s, waveform data block size
    DEFLOCKR = 150 # spatial lockout radius, um
    DEFINCLR = 150 # spatial include radius, um
    DEFDT = 400 # max time between phases of a single spike, us
    DEFRANDOMSAMPLE = False
    DEFEXTRACTPARAMSONDETECT = True

    # us, extra data as buffer at start and end of a block while detecting spikes.
    # Only useful for ensuring spike times within the actual block time range are
    # accurate. Spikes detected in the excess are discarded
    BLOCKEXCESS = 1000

    def __init__(self, sort, chans=None,
                 threshmethod=None, noisemethod=None, noisemult=None, fixedthreshuV=None,
                 ppthreshmult=None, fixednoisewin=None, dynamicnoisewin=None,
                 trange=None, maxnspikes=None, blocksize=None,
                 lockr=None, inclr=None, dt=None, randomsample=None,
                 extractparamsondetect=None):
        """Takes a parent Sort session and sets various parameters"""
        self.sort = sort
        self.srffname = sort.stream.srffname # for reference, store which .srf file this Detector is run on
        self.chans = np.asarray(chans) or np.arange(sort.stream.nchans) # None means search all channels
        self.threshmethod = threshmethod or self.DEFTHRESHMETHOD
        self.noisemethod = noisemethod or self.DEFNOISEMETHOD
        self.noisemult = noisemult or self.DEFNOISEMULT
        self.fixedthreshuV = fixedthreshuV or self.DEFFIXEDTHRESHUV
        self.ppthreshmult = ppthreshmult or self.DEFPPTHRESHMULT
        self.fixednoisewin = fixednoisewin or self.DEFFIXEDNOISEWIN # us
        self.dynamicnoisewin = dynamicnoisewin or self.DEFDYNAMICNOISEWIN # us
        self.trange = trange or (sort.stream.t0, sort.stream.tend)
        self.maxnspikes = maxnspikes or self.DEFMAXNSPIKES # return at most this many spikes
        self.blocksize = blocksize or self.DEFBLOCKSIZE
        self.lockr = lockr or self.DEFLOCKR
        self.inclr = inclr or self.DEFINCLR
        self.dt = dt or self.DEFDT
        self.randomsample = randomsample or self.DEFRANDOMSAMPLE
        self.extractparamsondetect = extractparamsondetect or self.DEFEXTRACTPARAMSONDETECT

        #self.dmurange = DMURANGE # allowed time difference between peaks of modelled spike
        self.datetime = None # date and time of last detect() call

    def get_chans(self):
        return self._chans

    def set_chans(self, chans):
        chans.sort() # ensure they're always sorted
        self._chans = np.int8(chans) # ensure they're always int8

    chans = property(get_chans, set_chans)

    def detect(self):
        """Search for spikes. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results

        TODO: remove any spikes that happen right at the first or last timepoint in the file,
        since we can't say when an interrupted falling or rising edge would've reached peak
        """
        self.calc_chans()
        sort = self.sort
        spikewidth = (sort.TW[1] - sort.TW[0]) / 1000000 # sec
        self.maxnt = int(sort.stream.sampfreq * spikewidth) # num timepoints to allocate per spike

        assert (self.dt < np.abs(sort.TW)).all() # necessary when calling sort.align_neuron()

        t0 = time.time()
        self.dti = int(self.dt // sort.stream.tres) # convert from numpy.int64 to normal int for inline C
        self.thresh = self.get_thresh() # abs, in AD units, one per chan in self.chans
        self.ppthresh = np.int16(np.round(self.thresh * self.ppthreshmult)) # peak-to-peak threshold, abs, in AD units
        AD2uV = sort.converter.AD2uV
        info('thresh calcs took %.3f sec' % (time.time()-t0))
        info('thresh   = %s' % AD2uV(self.thresh))
        info('ppthresh = %s' % AD2uV(self.ppthresh))

        bs = self.blocksize
        bx = self.BLOCKEXCESS
        wavetranges, (bs, bx, direction) = self.get_blockranges(bs, bx)

        self.nchans = len(self.chans) # number of enabled chans
        self.nspikes = 0 # total num spikes found across all chans so far by this Detector, reset at start of every search

        # want an nchan*2 array of [chani, x/ycoord]
        xycoords = [ self.enabledSiteLoc[chan] for chan in self.chans ] # (x, y) coords in chan order
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        self.siteloc = np.asarray([xcoords, ycoords]).T # index into with chani to get (x, y)

        t0 = time.time()

        if not DEBUG:
            # create a processing pool with as many processes as there are cores
            ncores = mp.cpu_count() # 1 per core
            pool = mp.Pool(ncores, initializer, (self,)) # send pickled copy of self to each process
            directions = [direction]*len(wavetranges)
            args = zip(wavetranges, directions)
            # TODO: FoundEnoughSpikesError is no longer being caught in multiprocessor code
            results = pool.map(callsearchblock, args, chunksize=1)
            pool.close()
            # results is a list of (spikes, wavedata) tuples, and needs to be unzipped
            spikeblocks, wavedatablocks = zip(*results)
            spikes = np.concatenate(spikeblocks)
            wavedata = np.concatenate(wavedatablocks) # along sid axis, other dims are identical
        else:
            # single process method, useful for debugging:
            spikes = np.zeros(0, self.SPIKEDTYPE) # init
            wavedata = np.zeros((0, 0, 0), np.int16) # init 3D array
            for wavetrange in wavetranges:
                try:
                    blockspikes, blockwavedata = self.searchblock(wavetrange, direction)
                except FoundEnoughSpikesError:
                    break
                nblockspikes = len(blockspikes)
                sshape = list(spikes.shape)
                sshape[0] += nblockspikes
                spikes.resize(sshape, refcheck=False)
                spikes[self.nspikes:self.nspikes+nblockspikes] = blockspikes

                wshape = list(wavedata.shape)
                if wshape == [0, 0, 0]:
                    wshape = blockwavedata.shape # init shape
                else:
                    wshape[0] += nblockspikes # just inc length
                wavedata.resize(wshape, refcheck=False)
                wavedata[self.nspikes:self.nspikes+nblockspikes] = blockwavedata

                self.nspikes += nblockspikes

        self.nspikes = len(spikes)
        assert len(wavedata) == self.nspikes
        # default -1 indicates no nid is set as of yet, reserve 0 for actual ids
        spikes['nid'] = -1
        spikes['cid'] = -1 # unused, always leave as -1
        info('\nfound %d spikes in total' % self.nspikes)
        info('inside .detect() took %.3f sec' % (time.time()-t0))
        # spikes might come out slightly out of temporal order, due to the way
        # the best peak is searched for forward and backwards in time on each edge
        t0 = time.time()
        i = spikes['t'].argsort()
        spikes = spikes[i] # ensure they're in temporal order
        wavedata = wavedata[i] # ditto for wavedata
        info("Sorting spikes and wavedata to ensure temporal order took %.3f sec" % (time.time()-t0))
        spikes['id'] = np.arange(self.nspikes) # assign ids now that they're in temporal order
        self.datetime = datetime.datetime.now()
        return spikes, wavedata

    def calc_chans(self):
        """Calculate lockout and inclusion chan neighbourhoods, max number of chans to use,
        and define the spike record dtype"""
        sort = self.sort
        self.enabledSiteLoc = {}
        for chan in self.chans: # for all enabled chans
            self.enabledSiteLoc[chan] = sort.stream.probe.SiteLoc[chan] # grab its (x, y) coordinate
        self.dm = DistanceMatrix(self.enabledSiteLoc) # distance matrix for the chans enabled for this search, sorted by chans
        # dict of neighbourhood of chanis for each chani
        self.locknbhdi = {} # for lockout around a spike
        self.inclnbhdi = {} # for inclusion of wavedata as part of a spike
        maxnchansperspike = 0
        for chani, distances in enumerate(self.dm.data): # iterate over rows of distances
            lockchanis, = np.uint8(np.where(distances <= self.lockr)) # at what col indices does the returned row fall within lockr?
            inclchanis, = np.uint8(np.where(distances <= self.inclr)) # at what col indices does the returned row fall within inclr?
            self.locknbhdi[chani] = lockchanis
            self.inclnbhdi[chani] = inclchanis
            maxnchansperspike = max(maxnchansperspike, len(inclchanis))
        self.maxnchansperspike = maxnchansperspike

        self.SPIKEDTYPE = [('id', np.int32), ('nid', np.int16), ('cid', np.int16),
                           ('chan', np.uint8), ('chans', np.uint8, self.maxnchansperspike),
                           ('nchans', np.uint8), ('chani', np.uint8),
                           # TODO: maybe it would be more efficient to store ti, t0i,
                           # and tendi wrt start of surf file instead of times in us?
                           ('t', np.int64), ('t0', np.int64), ('tend', np.int64),
                           ('V0', np.float32), ('V1', np.float32),
                           ('Vpp', np.float32),
                           ('phasetis', np.uint8, (self.maxnchansperspike, 2)),
                           ('aligni', np.uint8),
                           ('x0', np.float32), ('y0', np.float32),
                           ('sx', np.float32), ('sy', np.float32),
                           ('dphase', np.int16), # in us
                           #('w0', np.float32), ('w1', np.float32), ('w2', np.float32),
                           #('w3', np.float32), ('w4', np.float32),
                           ('s0', np.float32), ('s1', np.float32),
                           #('mVpp', np.float32),
                           #('mV0', np.float32), ('mV1', np.float32),
                           #('mdphase', np.float32),
                           ]

    def searchblock(self, wavetrange, direction):
        """Search a block of data, return a struct array of valid spikes,
        along with an array of their wavedata"""
        #info('searchblock():')
        stream = self.sort.stream
        #info('self.nspikes=%d, self.maxnspikes=%d, wavetrange=%s, direction=%d' %
        #     (self.nspikes, self.maxnspikes, wavetrange, direction))
        if self.nspikes >= self.maxnspikes:
            raise FoundEnoughSpikesError # skip this iteration
        tlo, thi = wavetrange # tlo could be > thi
        bx = self.BLOCKEXCESS
        cutrange = (tlo+bx, thi-bx) # range without the excess, ie time range of spikes to actually keep
        info('%s: wavetrange: %s, cutrange: %s' %
            (mp.current_process().name, wavetrange, cutrange))
        tslice = time.time()
        wave = stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed, ignores out of range data requests, returns up to stream limits
        print('%s: Stream slice took %.3f sec' %
             (mp.current_process().name, time.time()-tslice))
        tres = stream.tres

        if self.randomsample:
            maxnspikes = 1 # how many more we're looking for in the next block
        else:
            maxnspikes = self.maxnspikes - self.nspikes

        if self.threshmethod == 'Dynamic':
            # update thresh for each channel for this new block of data
            tnoise = time.time()
            noise = self.get_noise(wave.data) # float AD units
            info('%s: get_noise took %.3f sec' % (mp.current_process().name, time.time()-tnoise))
            self.thresh = noise * self.noisemult # float AD units
            self.thresh = np.int16(np.round(self.thresh)) # int16 AD units
            # clip so that fixedthresh <= self.thresh <= self.thresh.max()
            self.thresh = self.thresh.clip(self.fixedthresh, self.thresh.max())
            # peak-to-peak threshold, abs, in AD units
            self.ppthresh = np.int16(np.round(self.thresh * self.ppthreshmult))
            AD2uV = self.sort.converter.AD2uV
            info('%s: thresh:   %r' % (mp.current_process().name, AD2uV(self.thresh)))
            #info('%s: ppthresh: %r' % (mp.current_process().name, AD2uV(self.ppthresh)))

        tcheck_wave = time.time()
        spikes, wavedata = self.check_wave(wave, cutrange)
        info('%s: checking wave took %.3f sec' %
            (mp.current_process().name, time.time()-tcheck_wave))
        print('%s: found %d spikes' % (mp.current_process().name, len(spikes)))
        #import cProfile
        #cProfile.runctx('spikes, wavedata = self.check_wave(wave, cutrange)', globals(), locals())
        #spikes, wavedata = [], []
        return spikes, wavedata

    def check_wave(self, wave, cutrange):
        """Check which threshold exceeding peaks in wave data look like spikes
        and return only events that fall within cutrange. Search local spatiotemporal
        window around thresh exceeding peak for biggest peak-to-peak sharpness.
        Test that together they exceed Vpp thresh.

        TODO: keep an eye on broad spike at ptc15.87.1024880, about 340 us wide.
        Should be counted though
        """
        sort = self.sort
        AD2uV = sort.converter.AD2uV
        if self.extractparamsondetect:
            weights2gaussian = sort.extractor.weights2gaussian
            #weights2cauchy = sort.extractor.weights2cauchy
            #wavedata2spatial = sort.extractor.wavedata2spatial
            #wavedata2wcs = sort.extractor.wavedata2wcs
        lockouts = np.zeros(self.nchans, dtype=np.int64) # holds time indices for each enabled chan until which each enabled chani is locked out, updated on every found spike

        tsharp = time.time()
        sharp = spyke.util.sharpness2D(wave.data)
        info('%s: sharpness2D() took %.3f sec' %
            (mp.current_process().name, time.time()-tsharp))
        targthreshsharp = time.time()
        peakis = spyke.util.argthreshsharp(wave.data, self.thresh, sharp) # thresh exceeding peak indices
        info('%s: argthreshsharp() took %.3f sec' %
            (mp.current_process().name, time.time()-targthreshsharp))

        dti = self.dti
        twi = sort.twi
        sdti = dti // 2 # spatial dti - max dti allowed between maxchan and all other chans
        nspikes = 0
        npeaks = len(peakis)
        spikes = np.zeros(npeaks, self.SPIKEDTYPE) # nspikes will always be <= npeaks
        # TODO: test whether np.empty or np.zeros is faster overall in this case
        wavedata = np.empty((npeaks, self.maxnchansperspike, self.maxnt), dtype=np.int16)
        # check each peak for validity
        for ti, chani in peakis:
            if DEBUG: debug('*** trying thresh peak at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))
            # is this thresh exceeding peak locked out?
            if ti <= lockouts[chani]:
                if DEBUG: debug('peak is locked out')
                continue # skip to next peak

            # find all enabled chanis within locknbh of chani
            # lockouts are checked later
            chanis = self.locknbhdi[chani]
            nchans = len(chanis)

            # get sharpness window DT on either side of this peak
            t0i = max(ti-dti, 0) # check for lockouts a bit later
            tendi = ti+dti+1 # +1 makes it end inclusive, don't worry about slicing past end
            window = wave.data[chanis, t0i:tendi] # multichan data window, might not be contig

            # collect peak-to-peak sharpness for all chans
            # save max and adjacent sharpness timepoints for each chan, and keep track
            # of which of the two adjacent non locked out peaks is the sharpest
            localsharp = sharp[chanis, t0i:tendi]
            ppsharp = np.zeros(nchans, dtype=np.float32)
            maxsharpis = np.zeros(nchans, dtype=int)
            adjpeakis = np.zeros((nchans, 2), dtype=int)
            maxadjiis = np.zeros(nchans, dtype=int)
            for cii in range(nchans):
                localpeakis, = np.where(localsharp[cii] != 0.0)
                lastpeakii = len(localpeakis) - 1
                try: maxsharpii = abs(localsharp[cii, localpeakis]).argmax()
                except ValueError: continue # localpeakis is empty
                maxsharpi = localpeakis[maxsharpii]
                maxsharpis[cii] = maxsharpi
                # get one adjacent peak to left and right each, due to limits, either or
                # both may be identical to the max sharpness peak
                adjpeakis[cii] = localpeakis[[max(maxsharpii-1, 0), min(maxsharpii+1, lastpeakii)]]
                if localsharp[cii, maxsharpi] < 0:
                    maxadjii = localsharp[cii, adjpeakis[cii]].argmax() # look for +ve adj peak
                else:
                    maxadjii = localsharp[cii, adjpeakis[cii]].argmin() # look for -ve adj peak
                if maxadjii == 0 and (t0i+adjpeakis[cii, maxadjii] < lockouts[chanis[cii]]):
                    # adjacent peak comes before maxsharpi and is locked out
                    maxadjii = 1 # choose adjacent peak that falls after maxsharpi
                maxadjiis[cii] = maxadjii # save
                adjpi = adjpeakis[cii, maxadjii]
                ppsharp[cii] = localsharp[cii, maxsharpi] - localsharp[cii, adjpi]

            oldti = ti # save
            oldchani = chani # save

            # choose chan with biggest ppsharp as maxchan, check that this is identical to
            # the trigger chan, that its sharpest phase isn't locked out, that it falls within
            # cutrange, and that it meets both Vp and Vpp thresh criteria
            maxcii = abs(ppsharp).argmax()
            chani = chanis[maxcii] # update maxchan
            if chani != oldchani:
                if DEBUG: debug("triggered off peak on chan that isn't max ppsharpness for this event, pass on this peak and wait for the true sharpest peak to come later")
                continue
            maxsharpi = maxsharpis[maxcii]
            ti = t0i + maxsharpi # choose sharpest peak of maxchan, absolute
            # if sharpest peak is in the past, use it. If it's yet to come, wait for it
            if ti > oldti:
                if DEBUG: debug("triggered off early adjacent peak for this event, pass on this peak and wait for the true sharpest peak to come later")
                continue
            if ti <= lockouts[chani]: # sharpest peak is locked out
                if DEBUG: debug('sharpest peak at t=%d chan=%d is locked out' % (wave.ts[ti], self.chans[chani]))
                continue
            if not (cutrange[0] <= wave.ts[ti] <= cutrange[1]):
                if DEBUG:
                    # use %r since wave.ts[ti] is np.int64 and %d gives TypeError if > 2**31
                    debug("spike time %r falls outside cutrange for this searchblock "
                          "call, discarding" % wave.ts[ti])
                continue # skip to next peak
            # check that Vp thresh is exceeded by one of the two sharpest phases
            adjpi = adjpeakis[maxcii, maxadjiis[maxcii]]
            maxchanphasetis = np.array([maxsharpi, adjpi]) # relative to t0i, not necessarily in temporal order
            Vp = abs(window[maxcii, maxchanphasetis]).max() # grab biggest phase
            if Vp < self.thresh[chani]:
                if DEBUG: debug('peak at t=%d chan=%d and its adjacent peak are both < Vp' % (wave.ts[ti], self.chans[chani]))
                continue
            # check that Vpp thresh is exceeded by the two sharpest phases
            Vs = window[maxcii, maxchanphasetis]
            Vpp = abs(Vs).sum() # Vs are of opposite sign
            if Vpp < self.ppthresh[chani]:
                if DEBUG: debug('peaks at t=%r chan=%d are < Vpp' % (wave.ts[[ti, t0i+adjpi]], self.chans[chani]))
                continue
            if DEBUG: debug('found biggest thresh exceeding ppsharp at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))

            # get new spatiotemporal neighbourhood, with full window
            # align to -ve of the two sharpest peaks
            aligni = localsharp[maxcii, maxchanphasetis].argmin()
            #oldti = ti # save
            ti = t0i + maxchanphasetis[aligni] # new absolute time index to align to
            # cut new window
            oldt0i = t0i
            t0i = max(ti+twi[0], 0)
            tendi = ti+twi[1]+1 # end inclusive
            window = wave.data[chanis, t0i:tendi] # multichan data window, might not be contig
            maxcii, = np.where(chanis == chani)
            maxchanphasetis += oldt0i - t0i # relative to new t0i
            phasetis = np.zeros((nchans, 2), dtype=int) # holds phasetis for each lockchani
            phasetis[maxcii] = maxchanphasetis

            # pick corresponding peaks on other chans according to how
            # close they are to those on maxchan, Don't consider the sign of the peaks on each
            # chan, just their proximity in time. In other words, allow for spike inversion
            # across space
            localsharp = sharp[chanis, t0i:tendi]
            phaset0i, phaset1i = maxchanphasetis
            for cii in range(nchans):
                if cii == maxcii: # already set
                    continue
                localpeakis, = np.where(localsharp[cii] != 0.0)
                if len(localpeakis) == 0: # empty
                    phasetis[cii] = maxchanphasetis # use same tis as maxchan
                    continue
                lastpeakii = len(localpeakis) - 1
                # find peak on this chan that's temporally closest to primary phase on maxchan.
                # If two peaks are equally close, this picks the first one, although we should
                # probably pick the sharpest one instead:
                dt0is = abs(localpeakis-phaset0i)
                peak0ii = dt0is.argmin()
                # save primary phase for this cii
                dt0i = dt0is[peak0ii]
                if dt0i > sdti: # too distant in time
                    phasetis[cii, 0] = phaset0i # use same t0i as maxchan
                else: # give it its own t0i
                    phasetis[cii, 0] = localpeakis[peak0ii]
                # save 2ndary phase for this cii
                if phaset0i < phaset1i: # primary phase comes first (more common case)
                    peak1ii = peak0ii + 1 # 2ndary phase is 1 to the right
                else: # phaset1i < phaset0i, ie 2ndary phase comes first
                    peak1ii = peak0ii - 1 # 2ndary phase is 1 to the left
                dt1is = abs(localpeakis-phaset1i)
                try:
                    dt1i = dt1is[peak1ii]
                except IndexError: # no local peak relative to primary phase
                    phasetis[cii, 1] = phaset1i # use same t1i as maxchan
                    continue
                if dt1i > sdti: # too distant in time
                    phasetis[cii, 1] = phaset1i # use same t1i as maxchan
                else:
                    phasetis[cii, 1] = localpeakis[peak1ii]

            # find inclchanis, get corresponding indices into locknbhd of chanis
            inclchanis = self.inclnbhdi[chani]
            ninclchans = len(inclchanis)
            inclchans = self.chans[inclchanis]
            chan = self.chans[chani]
            inclchani = int(np.where(inclchans == chan)[0]) # != chani!
            inclciis = chanis.searchsorted(inclchanis)

            if DEBUG: debug("final window params: t0=%r, tend=%r, Vs=%r, phasets=\n%r"
                            % (wave.ts[t0i], wave.ts[tendi], list(AD2uV(Vs)), wave.ts[t0i+phasetis]))

            # build up spike record
            s = spikes[nspikes]
            s['t'] = wave.ts[ti]
            # leave each spike's chanis in sorted order, as they are in self.inclnbhdi,
            # important assumption used later on, like in sort.get_wave() and
            # Neuron.update_wave()
            ts = wave.ts[t0i:tendi]
            # use ts = np.arange(s['t0'], s['tend'], stream.tres) to reconstruct
            s['t0'], s['tend'] = wave.ts[t0i], wave.ts[tendi]
            inclphasetis = phasetis[inclciis]
            s['phasetis'][:ninclchans] = inclphasetis # wrt t0i
            s['aligni'] = aligni # 0 or 1
            s['dphase'] = int(abs(ts[phasetis[maxcii, 0]] - ts[phasetis[maxcii, 1]])) # in us
            s['V0'], s['V1'] = AD2uV(Vs) # in uV
            s['Vpp'] = AD2uV(Vpp) # in uV
            s['chan'], s['chans'][:ninclchans], s['nchans'] = chan, inclchans, ninclchans
            s['chani'] = inclchani
            inclwindow = window[inclciis]
            nt = inclwindow.shape[1] # isn't always full width if recording has gaps
            wavedata[nspikes, :ninclchans, :nt] = inclwindow
            if self.extractparamsondetect:
                # Get Vpp at each inclchan's phasetis, use as spatial weights:
                # see core.rowtake() or util.rowtake_cy() for indexing explanation:
                w = np.float32(inclwindow[np.arange(ninclchans)[:, None], inclphasetis])
                w = abs(w).sum(axis=1)
                x = self.siteloc[inclchanis, 0] # 1D array (row)
                y = self.siteloc[inclchanis, 1]
                s['x0'], s['y0'], s['sx'], s['sy'] = weights2gaussian(w, x, y, inclchani)
                #s['x0'], s['y0'], s['sx'], s['sy'] = weights2cauchy(w, x, y, inclchani)

            if DEBUG: debug('*** found new spike %d: %r @ (%d, %d)'
                            % (nspikes+self.nspikes, s['t'], self.siteloc[chani, 0], self.siteloc[chani, 1]))

            # Update lockouts for this spike.
            # Lock out to the latest of the 3 sharpest significant extrema. Some spikes
            # are more than biphasic, with a significant 3rd phase (see ptc18.14.24570980),
            # but never more than 3 significant phases. If the 3rd phase exceeds
            # threshold and falls within allowable dt of the previously considered last
            # phase, lock out to it. Otherwise, just lock out to 2nd phase. Doing this
            # reduces double triggers, yet maintains a minimalist lockout.
            '''
            if (phase2ti != lockout and
                phase2ti - lockout <= dti and
                abs(wave.data[chani, phase2ti]) >= self.thresh[chani]):
                lockout = phase2ti # absolute
            '''
            # locking out to phase2ti no longer seems necessary, now that each chan
            # has its own custom lockout
            '''
            p2ciis = (phase2tis - lockouts <= dti and
                      abs(wave.data[chanis, phase2tis]) >= self.thresh[chanis])) # bool array
            lockouts[p2ciis] = phase2tis[p2ciis]
            '''
            # give each chan a distinct lockout, based on how each chan's
            # sharpest phases line up with those of the maxchan. This fixes double
            # triggers that happen about 1% of the time (ptc18.14.7166200 & ptc18.14.9526000)
            lockouts[chanis] = t0i + phasetis.max(axis=1)
            if DEBUG: debug('lockouts=%r\nfor chans=%r' % (list(wave.ts[lockouts[chanis]]), list(self.chans[chanis])))

            nspikes += 1

        # shrink spikes and wavedata down to actual needed size
        spikes.resize(nspikes, refcheck=False)
        wds = wavedata.shape
        wavedata.resize((nspikes, wds[1], wds[2]), refcheck=False)
        return spikes, wavedata

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
                raise ValueError("Check trange - I'd rather not do a backwards random search")
            wavetranges = RandomWaveTranges(self.trange, bs, bx)
        else:
            es = range(self.trange[0], self.trange[1], bs) # left (or right) edges of data blocks
            for e in es:
                wavetranges.append((e-bx, e+bs+bx)) # time range of waveform to give to .searchblock
            # limit wavetranges to self.trange
            wavetranges[0] = self.trange[0], wavetranges[0][1]
            wavetranges[-1] = wavetranges[-1][0], self.trange[1]
        return wavetranges, (bs, bx, direction)

    def get_thresh(self):
        """Return array of thresholds in AD units, one per chan in self.chans,
        according to threshmethod and noisemethod"""
        self.fixedthresh = self.sort.converter.uV2AD(self.fixedthreshuV) # convert to AD units
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            thresh = np.tile(self.fixedthresh, len(self.chans))
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed thresh
            # randomly sample self.fixednoisewin's worth of data from self.trange in
            # blocks of self.blocksize, without replacement
            tload = time.time()
            print('loading data to calculate noise')
            if self.fixednoisewin >= abs(self.trange[1] - self.trange[0]): # sample width exceeds search trange
                wavetranges = [self.trange] # use a single block of data, as defined by trange
            else:
                nblocks = int(round(self.fixednoisewin / self.blocksize))
                wavetranges = RandomWaveTranges(self.trange, bs=self.blocksize, bx=0,
                                                maxntranges=nblocks, replacement=False)
            # preallocating memory doesn't seem to help here, all the time is in loading from stream:
            data = []
            for wavetrange in wavetranges:
                wave = self.sort.stream[wavetrange[0]:wavetrange[1]]
                wave = wave[self.chans] # keep just the enabled chans
                data.append(wave.data)
            data = np.concatenate(data, axis=1) # int16 AD units
            info('loading data to calc noise took %.3f sec' % (time.time()-tload))
            tnoise = time.time()
            noise = self.get_noise(data) # float AD units
            info('get_noise took %.3f sec' % (time.time()-tnoise))
            thresh = noise * self.noisemult # float AD units
            thresh = np.int16(np.round(thresh)) # int16 AD units
            thresh = thresh.clip(self.fixedthresh, thresh.max()) # clip so that all threshes are at least fixedthresh
        elif self.threshmethod == 'Dynamic':
            # dynamic threshes are calculated on the fly during the search, so leave as zero for now
            thresh = np.zeros(len(self.chans), dtype=np.int16)
        else:
            raise ValueError
        return thresh

    def get_noise(self, data):
        """Calculates noise over last dim in data (time), using .noisemethod"""
        #print('calculating noise')
        #ncores = mp.cpu_count()
        #pool = threadpool.Pool(ncores)
        if self.noisemethod == 'median':
            #noise = pool.map(self.get_median, data) # multithreads over rows in data
            #noise = np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004
            noise = spyke.util.median_inplace_2Dshort(np.abs(data)) / 0.6745 # see Quiroga2004
            #noise = np.mean(np.abs(data), axis=-1) / 0.6745 / 1.2
            #noise = util.mean_2Dshort(np.abs(data)) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            #noise = pool.map(self.get_stdev, data) # multithreads over rows in data
            noise = np.stdev(data, axis=-1)
        else:
            raise ValueError
        #pool.terminate() # pool.close() doesn't allow Python to exit when spyke is closed
        #pool.join() # unnecessary, hangs
        #return np.asarray(noise)
        return noise
    '''
    def get_median(self, data):
        """Return median value of multichan data, scaled according to Quiroga2004"""
        return np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004

    def get_stdev(self, data):
        """Return stdev of multichan data"""
        return np.stdev(data, axis=-1)
    '''
