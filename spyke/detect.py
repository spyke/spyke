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
from spyke.core import oneD2D, twoD1D #, WaveForm, toiter, argcut, intround, g, g2, RM
#from text import SimpleTable

#DMURANGE = 0, 500 # allowed time difference between peaks of modelled spike

logger = logging.Logger('detection')
shandler = logging.StreamHandler(strm=sys.stdout) # prints to screen
formatter = logging.Formatter('%(message)s')
shandler.setFormatter(formatter)
shandler.setLevel(logging.INFO) # log info level and higher to screen
logger.addHandler(shandler)
info = logger.info

DEBUG = True # print detection debug messages to log file? slows down detection

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
    wavetrange, direction = args
    detector = mp.current_process().detector
    return detector.searchblock(wavetrange, direction)

def initializer(detector, stream, srff):
    #stream.srff.open() # reopen the .srf file which was closed for pickling, engage file lock
    detector.sort.stream = stream
    detector.sort.stream.srff = srff # restore .srff that was deleted from stream on pickling
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
    DEFFIXEDTHRESH = 50 # uV, used by GlobalFixed, and as min thresh for ChanFixed
    DEFPPTHRESHMULT = 1.5 # peak-to-peak threshold is this times thresh
    DEFFIXEDNOISEWIN = 30000000 # 30s, used by ChanFixed - this should really be a % of self.trange
    DEFDYNAMICNOISEWIN = 10000 # 10ms, used by Dynamic
    DEFMAXNSPIKES = 0
    DEFBLOCKSIZE = 10000000 # 10s, waveform data block size
    DEFLOCKR = 150 # spatial lockout radius, um
    DEFINCLR = 100 # spatial include radius, um
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
        self.fixedthreshuV = fixedthreshuV or self.DEFFIXEDTHRESH
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

        stream = self.sort.stream
        stream.close() # make it picklable: close .srff, maybe .resample, and any file locks
        t0 = time.time()

        if not DEBUG:
            # create a processing pool with as many processes as there are CPUs/cores
            ncpus = min(mp.cpu_count(), 4) # 1 per core, max of 4, ie don't allow 8 "cores"
            pool = mp.Pool(ncpus, initializer, (self, stream, stream.srff)) # sends pickled copies to each process
            directions = [direction]*len(wavetranges)
            args = zip(wavetranges, directions)
            # TODO: FoundEnoughSpikesError is no longer being caught in multiprocessor code
            results = pool.map(callsearchblock, args, chunksize=1)
            pool.close()
            #pool.join() # unnecessary, I think
            blockspikes, blockwavedata = zip(*results) # results is a list of (spikes, wavedata) tuples, and needs to be unzipped
            spikes = np.concatenate(blockspikes)
            wavedata = np.concatenate(blockwavedata) # along sid axis, all other dims are identical
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

        stream.open()
        self.nspikes = len(spikes)
        assert len(wavedata) == self.nspikes
        # default -1 indicates no nid is set as of yet, reserve 0 for actual ids
        spikes['nid'] = -1
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

        self.SPIKEDTYPE = [('id', np.int32), ('nid', np.int16), ('chan', np.uint8),
                           ('chans', np.uint8, self.maxnchansperspike), ('nchans', np.uint8),
                           # TODO: maybe it would be more efficient to store ti, t0i,
                           # and tendi wrt start of surf file instead of times in us?
                           ('t', np.int64), ('t0', np.int64), ('tend', np.int64),
                           ('V0', np.float32), ('V1', np.float32),
                           ('Vpp', np.float32),
                           ('phaseti0', np.uint8), ('phaseti1', np.uint8),
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
        stream.open() # (re)open file that stream depends on (.resample or .srf), engage file lock
        info('%s: wavetrange: %s, cutrange: %s' %
            (mp.current_process().name, wavetrange, cutrange))
        tslice = time.time()
        wave = stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed, ignores out of range data requests, returns up to stream limits
        print('%s: Stream slice took %.3f sec' %
             (mp.current_process().name, time.time()-tslice))
        stream.close() # close file that stream depends on (.resample or .srf), release file lock
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
            self.thresh = self.thresh.clip(self.fixedthresh, self.thresh.max()) # clip so that all threshes are at least fixedthresh
            self.ppthresh = np.int16(np.round(self.thresh * self.ppthreshmult)) # peak-to-peak threshold, abs, in AD units
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
            wavedata2spatial = sort.extractor.wavedata2spatial
            #wavedata2wcs = sort.extractor.wavedata2wcs
        lockouts = np.zeros(self.nchans, dtype=np.int64) # holds time indices for each enabled chan until which each enabled chani is locked out, updated on every found spike

        tsharp = time.time()
        sharp = spyke.util.sharpness2D(wave.data)
        info('sharpness2D() took %.3f sec' % (time.time()-tsharp))
        targthreshsharp = time.time()
        peakis = spyke.util.argthreshsharp(wave.data, self.thresh, sharp) # thresh exceeding peak indices
        info('argthreshsharp() took %.3f sec' % (time.time()-targthreshsharp))

        dti = self.dti
        twi = sort.twi
        nspikes = 0
        npeaks = len(peakis)
        spikes = np.zeros(npeaks, self.SPIKEDTYPE) # nspikes will always be <= npeaks
        # TODO: test whether np.empty or np.zeros is faster overall in this case
        wavedata = np.empty((npeaks, self.maxnchansperspike, self.maxnt), dtype=np.int16)
        # check each peak for validity
        for ti, chani in peakis:
            if DEBUG: debug('*** trying thresh event at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))
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

            # find chan with biggest peak-to-peak sharpness, choose as maxchan. Also,
            # save max sharpness timepoints for each chan
            localsharp = sharp[chanis, t0i:tendi]
            ppsharp = np.zeros(nchans, dtype=np.float32)
            maxsharpis = np.zeros(nchans, dtype=int)
            adjpeakis = np.zeros(nchans, dtype=int)
            for cii in range(nchans):
                localpeakis, = np.where(localsharp[cii] != 0.0)
                lastpeakii = len(localpeakis) - 1
                try: maxsharpii = abs(localsharp[cii, localpeakis]).argmax()
                except ValueError: continue # localpeakis is empty
                maxsharpi = localpeakis[maxsharpii]
                maxsharpis[cii] = maxsharpi
                # get one adjacent peak to left and right each, due to limits, either or
                # both may be identical to the max sharpness peak
                adjpis = localpeakis[[max(maxsharpii-1, 0), min(maxsharpii+1, lastpeakii)]]
                if localsharp[cii, maxsharpi] < 0:
                    maxadjii = localsharp[cii, adjpis].argmax() # look for +ve adj peak
                else:
                    maxadjii = localsharp[cii, adjpis].argmin() # look for -ve adj peak
                adjpi = adjpis[maxadjii]
                adjpeakis[cii] = adjpi
                ppsharp[cii] = localsharp[cii, maxsharpi] - localsharp[cii, adjpi]

            #oldti = ti # save
            #oldchani = chani # save
            ciis = abs(ppsharp).argsort()[::-1] # biggest to smallest ppsharpness
            for cii in ciis: # test potential maxchans in decreasing ppsharpness order
                maxsharpi = maxsharpis[cii]
                ti = t0i + maxsharpi # align to sharpest peak of maxchan
                chani = chanis[cii] # update maxchan
                if ti <= lockouts[chani]: # peak is locked out
                    if DEBUG: debug('peak at t=%d chan=%d is locked out' % (wave.ts[ti], self.chans[chani]))
                    continue
                # check that Vp thresh is exceeded by one of the phases
                adjpi = adjpeakis[cii]
                Vp = abs(window[cii, [maxsharpi, adjpi]]).max() # grab biggest phase
                if Vp < self.thresh[chani]:
                    if DEBUG: debug('peak at t=%d chan=%d is < Vp' % (wave.ts[ti], self.chans[chani]))
                    continue
                # check that Vpp thresh is exceeded by the two phases
                Vpp = abs(window[cii, [maxsharpi, adjpi]]).sum()
                if Vpp < self.ppthresh[chani]:
                    if DEBUG: debug('peaks at t=%r chan=%d are < Vpp' % (wave.ts[[ti, t0i+adjpi]], self.chans[chani]))
                    continue
                if DEBUG: debug('found biggest thresh exceeding ppsharp at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))
                break
            else:
                if DEBUG: debug('all peaks are locked out')
                continue # skip to next event

            # get new spatiotemporal neighbourhood
            #oldchanis = chanis # save
            chanis = self.inclnbhdi[chani] # now take just inclnbhd instead of whole locknbhd
            nchans = len(chanis)
            t0i = max(ti+twi[0], 0)
            tendi = ti+twi[1]+1 # +1 makes it end inclusive
            window = wave.data[chanis, t0i:tendi] # multichan data window, might not be contig
            localsharp = sharp[chani, t0i:tendi] # single chan now
            maxsharpi = ti - t0i # relative

            # TODO: I think much of the following is no longer necessary, now that we're
            # searching for adjpeakis in ppsharp loop above:

            # grab adjacent phases to left and right of ti
            try:
                sharp0ti = np.where(localsharp[:maxsharpi] != 0.0)[0][-1] # relative to t0i
                sharp0 = localsharp[sharp0ti]
                phase0ti = t0i + sharp0ti # absolute
            except IndexError: # no phase to the left of ti
                sharp0 = 0.0
                phase0ti = 0
            #sharp1ti = maxsharpi
            #sharp1 = localsharp[sharp1ti]
            #phase1ti = ti
            try:
                sharp2ti = np.where(localsharp[maxsharpi+1:] != 0.0)[0][0] + maxsharpi + 1
                sharp2 = localsharp[sharp2ti]
                phase2ti = t0i + sharp2ti # absolute
            except IndexError: # no phase to the right of ti
                sharp2 = 0.0
                phase2ti = 0

            if phase0ti <= lockouts[chani]:
                sharp0 = 0.0 # don't consider this a viable phase

            # find sharpest adjacent phase, set potential lockout to later phase
            adjsharp = np.array([sharp0, sharp2])
            adjphasetis = np.array([phase0ti, phase2ti])
            adjsharpi = abs(adjsharp).argmax()
            adjphaseti = adjphasetis[adjsharpi] # either phase0ti or phase2ti
            if adjsharp[adjsharpi] == 0.0:
                if DEBUG: debug("couldn't find a matching adjacent phase to peak at "
                                "t=%r chan=%d" % (wave.ts[ti], self.chans[chani]))
                continue # skip to next event
            phasetis = np.sort([ti, adjphaseti]) # absolute, in temporal order
            dphase = phasetis[1] - phasetis[0]
            if dphase > dti:
                if DEBUG:
                    dt = dphase * sort.stream.tres
                    debug("sharpest adjacent phase to peak at t=%r is %d us away"
                          % (wave.ts[ti], dt))
                continue # skip to next event
            lockout = phasetis.max()
            Vs = wave.data[chani, phasetis]

            # ensure thresh is exceeded by the biggest of the two phases
            maxampli = abs(Vs).argmax()
            if abs(Vs[maxampli]) < self.thresh[chani]:
                if DEBUG:
                    debug("biggest peak at t=%r chan=%d gives only %.1f Vp"
                    % (wave.ts[phasetis[maxampli]], self.chans[chani], AD2uV(Vs[maxampli])))
                continue # skip to next event

            # ensure ppthresh is exceeded
            Vpp = abs(Vs[1] - Vs[0]) # don't maintain sign
            if Vpp < self.ppthresh[chani]:
                if DEBUG:
                    debug("matched adjacent peak to peak at t=%r chan=%d gives "
                          "only %.1f Vpp" % (wave.ts[ti], self.chans[chani], AD2uV(Vpp)))
                continue # skip to next event

            # align to sharpest -ve phase, set aligni
            aligni = localsharp[phasetis-t0i].argmin()
            #aligni = Vs.argmin() # could align by voltage instead
            oldti = ti # save
            ti = phasetis[aligni] # new absolute time index to align to (== oldti if sharpest phase is -ve)
            if ti != oldti: # need to update some variables
                t0i = ti+twi[0]
                tendi = ti+twi[1]+1 # end inclusive
                window = wave.data[chanis, t0i:tendi] # multichan data window, might not be contig
                #localsharp = sharp[chani, t0i:tendi] # not really necessary

            # now make phasetis relative to (potentially new) t0i
            phasetis -= t0i

            if not (cutrange[0] <= wave.ts[ti] <= cutrange[1]):
                if DEBUG:
                    # use %r since wave.ts[ti] is np.int64 and %d gives TypeError if > 2**31
                    debug("spike time %r falls outside cutrange for this searchblock "
                          "call, discarding" % wave.ts[ti])
                continue # skip to next event

            if DEBUG: debug("final window params: t0=%r, tend=%r, phasets=%r, Vs=%r"
                            % (wave.ts[t0i], wave.ts[tendi], wave.ts[t0i+phasetis], AD2uV(Vs)))

            # build up spike record

            s = spikes[nspikes]
            s['t'] = wave.ts[ti]
            # leave each spike's chanis in sorted order, as they are in self.inclnbhdi,
            # important assumption used later on, like in sort.get_wave() and
            # Neuron.update_wave()
            ts = wave.ts[t0i:tendi]
            # use ts = np.arange(s['t0'], s['tend'], stream.tres) to reconstruct
            s['t0'], s['tend'] = wave.ts[t0i], wave.ts[tendi]
            s['phaseti0'], s['phaseti1'] = phasetis # wrt t0i
            s['aligni'] = aligni # 0 or 1

            # TODO: add a sharpi field to designate which of the 2 phases is the sharpest
            # (main) phase - use this for extractor.get_Vp_weights()

            s['dphase'] = ts[phasetis[1]] - ts[phasetis[0]] # in us
            s['V0'], s['V1'] = AD2uV(Vs) # in uV
            s['Vpp'] = AD2uV(Vpp) # in uV
            chan = self.chans[chani]
            chans = self.chans[chanis]
            nchans = len(chans)
            s['chan'], s['chans'][:nchans], s['nchans'] = chan, chans, nchans
            wavedata[nspikes, 0:nchans] = window # aligned in time due to all having same nt
            if self.extractparamsondetect:
                # just x and y params for now
                x = self.siteloc[chanis, 0] # 1D array (row)
                y = self.siteloc[chanis, 1]
                maxchani = int(np.where(chans == chan)[0]) # != chani!
                # TODO: could call weights2spatialmean directly, since we already
                # have the multichan sharp array available, from which weights could be
                # more effectively extracted than with get_Vpp_weights()
                s['x0'], s['y0'], s['sx'], s['sy'] = wavedata2spatial(window, maxchani, phasetis, aligni, x, y)
                #s['w0'], s['w1'], s['w2'], s['w3'], s['w4'] = wavedata2wcs(window, maxchani)

            if DEBUG: debug('*** found new spike %d: %r @ (%d, %d)'
                            % (nspikes+self.nspikes, s['t'], self.siteloc[chani, 0], self.siteloc[chani, 1]))

            lockchanis = self.locknbhdi[chani]

            # Update lockouts for this spike.
            # Lock out to the latest of the 3 sharpest significant extrema. Some spikes
            # are more than biphasic, with a significant 3rd phase (see ptc18.14.24570980),
            # but never more than 3 significant phases. If the 3rd phase exceeds
            # threshold and falls within allowable dt of the previously considered last
            # phase, lock out to it. Otherwise, just lock out to 2nd phase. Doing this
            # reduces double triggers, yet maintains a minimalist lockout.

            if (phase2ti != lockout and
                phase2ti - lockout <= dti and
                abs(wave.data[chani, phase2ti]) >= self.thresh[chani]):
                lockout = phase2ti # absolute

            # TODO: give each chan a distinct lockout, based on how each chan's
            # sharpest phases line up with those of the maxchan. This will fix double
            # triggers that happen about 1% of the time (ptc18.14.7166200 & ptc18.14.9526000)
            # Lining up each included chan's sharpest phases with those of the maxchan
            # should also give better channel weights for spatial localization

            lockouts[lockchanis] = lockout # same for all chans in this spike
            if DEBUG:
                lockoutt = wave.ts[lockout]
                lockchans = self.chans[lockchanis]
                debug('lockout=%d for chans=%s' % (lockoutt, lockchans))

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
        #ncpus = mp.cpu_count()
        #pool = threadpool.Pool(ncpus)
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
