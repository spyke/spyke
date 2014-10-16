"""Spike detection and modelling"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

# this stuff needs to be near the top apparently
import numpy as np
import pyximport
pyximport.install(build_in_temp=False, inplace=True)
import util # .pyx file

import sys
import time
import logging
import datetime
import multiprocessing as mp
from multiprocessing import Process
ps = mp.current_process
from copy import deepcopy

'''
NOTE: as of Ubuntu 10.10, for some reason often get:

OSError: [Errno 4] Interrupted system call
> /usr/lib/python2.6/multiprocessing/forking.py(106)poll()
    105             if self.returncode is None:
--> 106                 pid, sts = os.waitpid(self.pid, flag)
    107                 if pid == self.pid:

or:

IOError: [Errno 4] Interrupted system call
/usr/lib/python2.6/multiprocessing/queues.py in get(self, block, timeout)
     89             self._rlock.acquire()
     90             try:
---> 91                 res = self._recv()
     92                 self._sem.release()
     93                 return res

which has to be caught and retried using _eintr_retry_call.
'''
import core
from core import eucd, issorted, concatenate_destroy, intround, g2, cauchy2

#DMURANGE = 0, 500 # allowed time difference between peaks of modelled spike

logger = logging.Logger('detection')
shandler = logging.StreamHandler(sys.stdout) # prints to screen
formatter = logging.Formatter('%(message)s')
shandler.setFormatter(formatter)
shandler.setLevel(logging.INFO) # log info level and higher to screen
logger.addHandler(shandler)
info = logger.info

DEBUG = False # print detection debug messages to log file? slows down detection
MPMETHOD = 'detectionprocess' #'singleprocess', 'detectionprocess', 'pool'

if DEBUG:
    # print detection info and debug msgs to file, and info msgs to screen
    dt = str(datetime.datetime.now()) # get a timestamp
    dt = dt.split('.')[0] # ditch the us
    dt = dt.replace(' ', '_')
    dt = dt.replace(':', '.')
    logfname = dt + '_detection.log'
    logf = open(logfname, 'w')
    fhandler = logging.StreamHandler(stream=logf) # prints to file
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.DEBUG) # log debug level and higher to file
    logger.addHandler(fhandler)
    debug = logger.debug

import errno
def _eintr_retry_call(func, *args):
    """Keeps retrying func in case an "OSError/IOError: [Errno 4] Interrupted system call"
    is raised for some mysterious reason. Modified from /usr/lib/python2.6/subprocess.py"""
    while True:
        try:
            return func(*args)
        except (OSError, IOError), e:
            if e.errno == errno.EINTR:
                continue
            raise

def callsearchblock(blockrange):
    """Run current process' Detector on blockrange"""
    detector = ps().detector
    return detector.searchblock(blockrange)

def initializer(detector):
    """Save pickled copy of the Detector to the current process"""
    # not exactly sure why, but deepcopy is crucial to prevent artefactual spikes!
    ps().detector = deepcopy(detector)
    ps().detector.sort.stream.open()
    

class RandomBlockRanges(object):
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
        t1 = t0 + self.bs
        self.ntranges += 1
        return (t0-self.bx, t1+self.bx)

    def __iter__(self):
        return self


class DistanceMatrix(object):
    """Channel distance matrix, with rows in .data corresponding to
    .chans and .coords"""
    def __init__(self, SiteLoc):
        """SiteLoc is a dictionary of (x, y) tuples, with chans as the keys. See probes.py"""
        chans_coords = SiteLoc.items() # list of (chan, coords) tuples
        chans_coords.sort() # sort by chan
        # pull out the sorted chans:
        self.chans = np.uint8([ chan_coord[0] for chan_coord in chans_coords ])
        # pull out the coords, now in chan order:
        self.coords = [ chan_coord[1] for chan_coord in chans_coords ]
        self.data = eucd(self.coords)


class DetectionProcess(mp.Process):
    """A temporary child process for doing some detection"""
    def run(self):
        for blocki, blockrange in zip(self.blockis, self.blockranges):
            blockspikes, blockwavedata = self.detector.searchblock(blockrange)
            self.q.put((blocki, blockspikes, blockwavedata))


class Detector(object):
    """Spike detector base class"""
    def __init__(self, sort=None):
        """Takes a parent Sort session and sets some parameters"""
        self.sort = sort
        # for reference, store .srf/.track filename(s) this Detector is run on
        self.fname = sort.stream.fname
        self.fixednoisewin = 30000000 # us, used by ChanFixed, should really be % of self.trange
        self.extractparamsondetect = True
        self.datetime = None # date and time of last detect() call
        # us, extra data as buffer at start and end of a block while detecting spikes.
        # Only useful for ensuring spike times within the actual block time range are
        # accurate. Spikes detected in the excess are discarded
        self.blockexcess = 1000

    def get_chans(self):
        return self._chans

    def set_chans(self, chans):
        chans.sort() # ensure they're always sorted
        self._chans = np.int8(chans) # ensure they're always int8

    chans = property(get_chans, set_chans)

    def get_srffnames(self):
        return self.sort.stream.srffnames

    srffnames = property(get_srffnames)

    def detect(self):
        """Search for spikes. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results"""
        self.calc_chans()
        sort = self.sort
        self.mpmethod = MPMETHOD
        spikewidth = (sort.tw[1] - sort.tw[0]) / 1000000 # sec
        # num timepoints to allocate per spike:
        self.maxnt = int(sort.stream.sampfreq * spikewidth)

        print('Detection trange: %r' % (self.trange,))

        t0 = time.time()
        # convert from numpy.int64 to normal int for inline C:
        self.dti = int(self.dt // sort.stream.tres)
        self.thresh = self.get_thresh() # abs, in AD units, one per chan in self.chans
        self.ppthresh = np.int16(np.round(self.thresh * self.ppthreshmult)) # abs, in AD units
        AD2uV = sort.converter.AD2uV
        info('thresh calcs took %.3f sec' % (time.time()-t0))
        info('thresh   = %s' % AD2uV(self.thresh))
        info('ppthresh = %s' % AD2uV(self.ppthresh))

        bs = self.blocksize
        bx = self.blockexcess
        blockranges = self.get_blockranges(bs, bx)
        nblocks = len(blockranges)

        self.nchans = len(self.chans) # number of enabled chans
        # total num spikes found across all chans so far by this Detector,
        # reset at start of every search:
        self.nspikes = 0

        # want an nchan*2 array of [chani, x/ycoord]
        xycoords = [ self.enabledSiteLoc[chan] for chan in self.chans ] # (x, y) in chan order
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        self.siteloc = np.asarray([xcoords, ycoords]).T # index into with chani to get (x, y)

        # prevent out of memory errors due to copying of large stream.wavedata array
        # when spawning multiple processes
        if type(self.sort.stream) == core.SimpleStream:
            self.mpmethod = 'singleprocess'

        ncores = mp.cpu_count()
        t0 = time.time()

        # mp.Pool is slightly faster than my own DetectionProcess
        if not DEBUG and self.mpmethod == 'pool': # use a pool of processes
            nprocesses = min(ncores, nblocks)
            # send pickled copy of self to each process
            pool = mp.Pool(nprocesses, initializer, (self,))
            results = pool.map(callsearchblock, blockranges, chunksize=1)
            pool.close()
            # results is a list of (spikes, wavedata) tuples, and needs to be unzipped
            spikes, wavedata = zip(*results)
        elif not DEBUG and self.mpmethod == 'detectionprocess':
            nprocesses = min(ncores, nblocks)
            dps = []
            q = mp.Queue()
            spikes = [None] * nblocks
            wavedata = [None] * nblocks
            for dpi in range(nprocesses):
                dp = DetectionProcess()
                # not exactly sure why, but deepcopy is crucial to prevent artefactual spikes!
                dp.detector = deepcopy(self)
                dp.detector.sort.stream.open()
                dp.blockis = range(dpi, nblocks, nprocesses)
                dp.blockranges = blockranges[dp.blockis]
                dp.q = q
                dp.start()
                dps.append(dp)
            for i in range(nblocks):
                #blocki, blockspikes, blockwavedata = dp.q.get() # defaults to block=True
                blocki, blockspikes, blockwavedata = _eintr_retry_call(dp.q.get)
                #print('got block %d results' % blocki)
                spikes[blocki] = blockspikes
                wavedata[blocki] = blockwavedata
            for dp in dps:
                dp.join()
                #_eintr_retry_call(dp.join) # eintr isn't raised anymore it seems
        else: # use a single process, useful for debugging or for .tsf files
            spikes = []
            wavedata = []
            for blockrange in blockranges:
                blockspikes, blockwavedata = self.searchblock(blockrange)
                spikes.append(blockspikes)
                wavedata.append(blockwavedata)

        spikes = concatenate_destroy(spikes)
        wavedata = concatenate_destroy(wavedata) # along sid axis, other dims are identical
        self.nspikes = len(spikes)
        assert len(wavedata) == self.nspikes
        # default -1 indicates no nid is set as of yet, reserve 0 for actual ids
        spikes['nid'] = 0
        info('\nfound %d spikes in total' % self.nspikes)
        info('inside .detect() took %.3f sec' % (time.time()-t0))
        if not issorted(spikes['t']):
            raise RuntimeError("spikes aren't sorted for some reason")
        spikes['id'] = np.arange(self.nspikes) # assign ids (should be in temporal order)
        self.datetime = datetime.datetime.now()
        return spikes, wavedata

    def calc_chans(self):
        """Calculate lockout and inclusion chan neighbourhoods, max number of chans to use,
        and define the spike record dtype"""
        sort = self.sort
        self.enabledSiteLoc = {}
        for chan in self.chans: # for all enabled chans
            self.enabledSiteLoc[chan] = sort.stream.probe.SiteLoc[chan] # get its (x, y)
        # distance matrix for the chans enabled for this search, sorted by chans:
        self.dm = DistanceMatrix(self.enabledSiteLoc)
        # dict of neighbourhood of chanis for each chani
        self.locknbhdi = {} # for lockout around a spike
        self.inclnbhdi = {} # for inclusion of wavedata as part of a spike
        maxnchansperspike = 0
        for chani, distances in enumerate(self.dm.data): # iterate over rows of distances
            # at what col indices does the returned row fall within lockr?:
            lockchanis, = np.uint8(np.where(distances <= self.lockr))
            # at what col indices does the returned row fall within inclr?:
            inclchanis, = np.uint8(np.where(distances <= self.inclr))
            self.locknbhdi[chani] = lockchanis
            self.inclnbhdi[chani] = inclchanis
            maxnchansperspike = max(maxnchansperspike, len(inclchanis))
        self.maxnchansperspike = maxnchansperspike

        self.SPIKEDTYPE = [('id', np.int32), ('nid', np.int16),
                           ('chan', np.uint8), ('nchans', np.uint8),
                           ('chans', np.uint8, (self.maxnchansperspike,)),
                           ('chani', np.uint8),
                           ('t', np.int64), ('t0', np.int64), ('t1', np.int64),
                           ('dt', np.int16), # time between peaks, in us
                           ('tis', np.uint8, (self.maxnchansperspike, 2)), # peak positions
                           ('aligni', np.uint8),
                           ('V0', np.float32), ('V1', np.float32), ('Vpp', np.float32),
                           ('x0', np.float32), ('y0', np.float32),
                           ('sx', np.float32), ('sy', np.float32),
                           ]

    def searchblock(self, blockrange):
        """Search a block of data, return a struct array of valid spikes,
        along with an array of their wavedata"""
        #info('searchblock():')
        stream = self.sort.stream
        cutrange = blockrange.copy() # trange of spikes to keep
        bx = self.blockexcess
        # if block doesn't falls at start or end of self.trange, remove excess:
        if cutrange[0] != self.trange[0]: cutrange[0] += bx
        if cutrange[1] != self.trange[1]: cutrange[1] -= bx
        info('%s: blockrange: %s, cutrange: %s' % (ps().name, blockrange, cutrange))
        tslice = time.time()
        # get WaveForm of multichan data, including excess, ignores out of range data requests:
        wave = stream(blockrange[0], blockrange[1])
        print('%s: Stream slice took %.3f sec' % (ps().name, time.time()-tslice))
        tres = stream.tres

        if self.threshmethod == 'Dynamic':
            # update threshold for each channel for this new block of data
            tnoise = time.time()
            noise = self.get_noise(wave.data) # float AD units
            info('%s: get_noise took %.3f sec' % (ps().name, time.time()-tnoise))
            self.thresh = noise * self.noisemult # float AD units
            self.thresh = np.int16(np.round(self.thresh)) # int16 AD units
            # clip so that fixedthresh <= self.thresh <= self.thresh.max()
            self.thresh = self.thresh.clip(self.fixedthresh, self.thresh.max())
            # peak-to-peak threshold, abs, in AD units
            self.ppthresh = np.int16(np.round(self.thresh * self.ppthreshmult))
            AD2uV = self.sort.converter.AD2uV
            #info('%s: thresh:   %r' % (ps().name, AD2uV(self.thresh)))
            #info('%s: ppthresh: %r' % (ps().name, AD2uV(self.ppthresh)))

        tcheck_wave = time.time()
        spikes, wavedata = self.check_wave(wave, cutrange)
        info('%s: checking wave took %.3f sec' % (ps().name, time.time()-tcheck_wave))

        # spikes might come out slightly out of temporal order, due to the way
        # the best peak is searched for forward and backwards in time on each edge
        #ttsort = time.time()
        i = spikes['t'].argsort()
        if len(i) > 0: # not empty
            spikes = spikes[i] # ensure they're in temporal order
            wavedata = wavedata[i] # ditto for wavedata
        #info("%s: temporal sorting took %.3f sec" % (ps().name, time.time()-ttsort))

        print('%s: found %d spikes' % (ps().name, len(spikes)))
        #import cProfile
        #cProfile.runctx('spikes, wavedata = self.check_wave(wave, cutrange)',
        #                globals(), locals())
        #spikes, wavedata = [], []
        return spikes, wavedata

    def check_wave(self, wave, cutrange):
        """Check which threshold-exceeding peaks in wave data look like spikes
        and return only events that fall within cutrange. Search local spatiotemporal
        window around threshold-exceeding peak for biggest peak-to-peak sharpness.
        Finally, test that the sharpest peak and its neighbour exceed Vp and Vpp thresholds"""
        sort = self.sort
        AD2uV = sort.converter.AD2uV
        if self.extractparamsondetect:
            weights2f = sort.extractor.weights2f
            f = g2 # 2D Gaussian
            #f = cauchy2 # 2D Cauchy
        # holds time indices for each enabled chan until which each enabled chani is
        # locked out, updated on every found spike
        lockouts = np.zeros(self.nchans, dtype=np.int64)

        tsharp = time.time()
        sharp = util.sharpness2D(wave.data)
        info('%s: sharpness2D() took %.3f sec' % (ps().name, time.time()-tsharp))
        targthreshsharp = time.time()
        # threshold-exceeding peak indices (2D, columns are [tis, cis])
        peakis = util.argthreshsharp(wave.data, self.thresh, sharp)
        info('%s: argthreshsharp() took %.3f sec' % (ps().name, time.time()-targthreshsharp))

        maxti = len(wave.ts) - 1
        dti = self.dti
        twi = sort.twi
        sdti = dti // 2 # spatial dti: max dti allowed between maxchan and all other chans
        nspikes = 0
        npeaks = len(peakis)
        spikes = np.zeros(npeaks, self.SPIKEDTYPE) # nspikes will always be <= npeaks
        # TODO: test whether np.empty or np.zeros is faster overall in this case
        wavedata = np.empty((npeaks, self.maxnchansperspike, self.maxnt), dtype=np.int16)
        # check each threshold-exceeding peak for validity:
        for ti, chani in peakis:
            if DEBUG: debug('*** trying thresh peak at t=%d chan=%d'
                            % (wave.ts[ti], self.chans[chani]))
            # is this threshold-exceeding peak locked out?
            lockoutchani = lockouts[chani]
            if ti <= lockoutchani:
                if DEBUG: debug('peak is locked out')
                continue # skip to next peak

            # find all enabled chanis within locknbh of chani
            # lockouts are checked later
            chanis = self.locknbhdi[chani]
            nchans = len(chanis)

            # get search window DT on either side of this peak, for checking sharpness
            t0i = max(ti-dti, 0) # check for lockouts a bit later
            t1i = ti+dti+1 # +1 makes it end inclusive, don't worry about slicing past end
            window = wave.data[chanis, t0i:t1i] # search window, might not be contig

            # Collect peak-to-peak sharpness for all chans. Save max and adjacent sharpness
            # timepoints for each chan, and keep track of which of the two adjacent non locked
            # out peaks is the sharpest
            localsharp = sharp[chanis, t0i:t1i] # sliced the same way as window
            ppsharp = np.zeros(nchans, dtype=np.float32)
            maxsharpis = np.zeros(nchans, dtype=int)
            adjpeakis = np.zeros((nchans, 2), dtype=int)
            maxadjiis = np.zeros(nchans, dtype=int)
            continuepeaki = False # signal to skip to next peaki
            for cii in range(nchans):
                localpeakis, = np.where(localsharp[cii] != 0.0)
                # keep only non-locked out localpeakis on this channel:
                localpeakis = localpeakis[(t0i+localpeakis) > lockouts[chanis[cii]]]
                if len(localpeakis) == 0:
                    continue # localpeakis is empty
                lastpeakii = len(localpeakis) - 1
                maxsharpii = abs(localsharp[cii, localpeakis]).argmax()
                maxsharpi = localpeakis[maxsharpii]
                maxsharpis[cii] = maxsharpi
                # Get one adjacent peak to left and right each. Due to limits, either or
                # both may be identical to the max sharpness peak
                adjpeakis[cii] = localpeakis[[max(maxsharpii-1, 0), min(maxsharpii+1,
                                              lastpeakii)]]
                if localsharp[cii, maxsharpi] < 0:
                    maxadjii = localsharp[cii, adjpeakis[cii]].argmax() # look for +ve adj peak
                else:
                    maxadjii = localsharp[cii, adjpeakis[cii]].argmin() # look for -ve adj peak
                maxadjiis[cii] = maxadjii # save
                adjpi = adjpeakis[cii, maxadjii]
                if maxsharpi != adjpi:
                    ppsharp[cii] = localsharp[cii, maxsharpi] - localsharp[cii, adjpi]
                else: # monophasic spike, set ppsharp == sharpness of single peak:
                    ppsharp[cii] = localsharp[cii, maxsharpi]
                    if chanis[cii] == chani: # trigger chan is monophasic
                        # ensure ppsharp of monophasic spike >= Vppthresh**2/dt, ie ensure that
                        # its Vpp exceeds Vppthresh and has zero crossings on either side,
                        # with no more than dt between. Avoids excessively wide
                        # monophasic peaks from being considered as spikes:
                        if DEBUG: debug("found monophasic spike")
                        if abs(ppsharp[cii]) < self.ppthresh[chani]**2 / dti:
                            continuepeaki = True
                            if DEBUG: debug("peak wasn't sharp enough for a monophasic spike")
                            break # out of cii loop

            if continuepeaki:
                continue # to next peaki
            oldti = ti # save
            oldchani = chani # save

            # choose chan with biggest ppsharp as maxchan, check that this is identical to
            # the trigger chan, that its sharpest peak isn't locked out, that it falls within
            # cutrange, and that it meets both Vp and Vpp threshold criteria
            maxcii = abs(ppsharp).argmax()
            chani = chanis[maxcii] # update maxchan
            if chani != oldchani:
                if DEBUG: debug("triggered off peak on chan that isn't max ppsharpness for "
                                "this event, pass on this peak and wait for the true "
                                "sharpest peak to come later")
                continue
            maxsharpi = maxsharpis[maxcii]
            ti = t0i + maxsharpi # choose sharpest peak of maxchan, absolute
            # if sharpest peak is in the past, use it. If it's yet to come, wait for it
            if ti > oldti:
                if DEBUG: debug("triggered off early adjacent peak for this event, "
                                "pass on this peak and wait for the true sharpest peak "
                                "to come later")
                continue
            if ti <= lockoutchani: # sharpest peak is locked out
                if DEBUG: debug('sharpest peak at t=%d chan=%d is locked out'
                                % (wave.ts[ti], self.chans[chani]))
                continue
            if not (cutrange[0] <= wave.ts[ti] <= cutrange[1]):
                if DEBUG:
                    # use %r since wave.ts[ti] is np.int64 and %d gives TypeError if > 2**31
                    debug("spike time %r falls outside cutrange for this searchblock "
                          "call, discarding" % wave.ts[ti])
                continue # skip to next peak
            # check that Vp threshold is exceeded by at least one of the two sharpest peaks
            adjpi = adjpeakis[maxcii, maxadjiis[maxcii]]
            # relative to t0i, not necessarily in temporal order:
            maxchantis = np.array([maxsharpi, adjpi])
            # voltages of the two sharpest peaks, convert int16 to int64 to prevent overflow
            Vs = np.int64(window[maxcii, maxchantis])
            Vp = abs(Vs).max() # grab biggest peak
            if Vp < self.thresh[chani]:
                if DEBUG: debug('peak at t=%d chan=%d and its adjacent peak are both < Vp'
                                % (wave.ts[ti], self.chans[chani]))
                continue
            # check that the two sharpest peaks together exceed Vpp threshold:
            Vpp = abs(Vs[0] - Vs[1]) # Vs are of opposite sign, unless monophasic
            if Vpp == 0: # monophasic spike
                Vpp = Vp # use Vp as Vpp
            
            if Vpp < self.ppthresh[chani]:
                if DEBUG: debug('peaks at t=%r chan=%d are < Vpp'
                                % (wave.ts[[ti, t0i+adjpi]], self.chans[chani]))
                continue

            if DEBUG: debug('found biggest thresh exceeding ppsharp at t=%d chan=%d'
                            % (wave.ts[ti], self.chans[chani]))

            # get new spatiotemporal neighbourhood, with full window
            # align to -ve of the two sharpest peaks
            aligni = localsharp[maxcii, maxchantis].argmin()
            #oldti = ti # save
            ti = t0i + maxchantis[aligni] # new absolute time index to align to
            # cut new window
            oldt0i = t0i
            t0i = max(ti+twi[0], 0)
            t1i = min(ti+twi[1]+1, maxti) # end inclusive
            window = wave.data[chanis, t0i:t1i] # multichan data window, might not be contig
            maxcii, = np.where(chanis == chani)
            maxchantis += oldt0i - t0i # relative to new t0i
            tis = np.zeros((nchans, 2), dtype=int) # holds time indices for each lockchani
            tis[maxcii] = maxchantis

            # pick corresponding peaks on other chans according to how close they are
            # to those on maxchan, Don't consider the sign of the peaks on each
            # chan, just their proximity in time. In other words, allow for spike
            # inversion across space
            localsharp = sharp[chanis, t0i:t1i]
            peak0ti, peak1ti = maxchantis
            for cii in range(nchans):
                if cii == maxcii: # already set
                    continue
                localpeakis, = np.where(localsharp[cii] != 0.0)
                # keep only non-locked out localpeakis on this channel:
                localpeakis = localpeakis[(t0i+localpeakis) > lockouts[chanis[cii]]]
                if len(localpeakis) == 0: # localpeakis is empty
                    tis[cii] = maxchantis # use same tis as maxchan
                    continue
                lastpeakii = len(localpeakis) - 1
                # find peak on this chan that's temporally closest to primary peak on maxchan.
                # If two peaks are equally close, this picks the first one, although we should
                # probably pick the sharpest one instead:
                dt0is = abs(localpeakis-peak0ti)
                peak0ii = dt0is.argmin()
                # save primary peak for this cii
                dt0i = dt0is[peak0ii]
                if dt0i > sdti: # too distant in time
                    tis[cii, 0] = peak0ti # use same t0i as maxchan
                else: # give it its own t0i
                    tis[cii, 0] = localpeakis[peak0ii]
                # save 2ndary peak for this cii
                if peak0ti < peak1ti: # primary peak comes first (more common case)
                    peak1ii = peak0ii + 1 # 2ndary peak is 1 to the right
                if len(localpeakis) == 1: # monophasic, set 2ndary peak same as primary
                    tis[cii, 1] = tis[cii, 0]
                    continue
                else: # peak1ti < peak0ti, ie 2ndary peak comes first
                    peak1ii = peak0ii - 1 # 2ndary peak is 1 to the left
                dt1is = abs(localpeakis-peak1ti)
                try:
                    dt1i = dt1is[peak1ii]
                except IndexError: # no local peak relative to primary peak
                    tis[cii, 1] = peak1ti # use same t1i as maxchan
                    continue
                if dt1i > sdti: # too distant in time
                    tis[cii, 1] = peak1ti # use same t1i as maxchan
                else:
                    tis[cii, 1] = localpeakis[peak1ii]

            # find inclchanis, get corresponding indices into locknbhd of chanis
            inclchanis = self.inclnbhdi[chani]
            ninclchans = len(inclchanis)
            inclchans = self.chans[inclchanis]
            chan = self.chans[chani]
            inclchani = int(np.where(inclchans == chan)[0]) # != chani!
            inclciis = chanis.searchsorted(inclchanis)

            if DEBUG: debug("final window params: t0=%r, t1=%r, Vs=%r, peakts=\n%r"
                            % (wave.ts[t0i], wave.ts[t1i], list(AD2uV(Vs)), wave.ts[t0i+tis]))

            # build up spike record
            s = spikes[nspikes]
            s['t'] = wave.ts[ti]
            # leave each spike's chanis in sorted order, as they are in self.inclnbhdi,
            # important assumption used later on, like in sort.get_wave() and
            # Neuron.update_wave()
            ts = wave.ts[t0i:t1i]
            # use ts = np.arange(s['t0'], s['t1'], stream.tres) to reconstruct
            s['t0'], s['t1'] = wave.ts[t0i], wave.ts[t1i]
            incltis = tis[inclciis]
            s['tis'][:ninclchans] = incltis # wrt t0i
            s['aligni'] = aligni # 0 or 1
            s['dt'] = int(abs(ts[tis[maxcii, 0]] - ts[tis[maxcii, 1]])) # in us
            s['V0'], s['V1'] = AD2uV(Vs) # in uV
            s['Vpp'] = AD2uV(Vpp) # in uV
            s['chan'], s['chans'][:ninclchans], s['nchans'] = chan, inclchans, ninclchans
            s['chani'] = inclchani
            inclwindow = window[inclciis]
            nt = inclwindow.shape[1] # isn't always full width if recording has gaps
            wavedata[nspikes, :ninclchans, :nt] = inclwindow
            if self.extractparamsondetect:
                # Get Vpp at each inclchan's tis, use as spatial weights:
                # see core.rowtake() or util.rowtake_cy() for indexing explanation:
                w = np.float32(inclwindow[np.arange(ninclchans)[:, None], incltis])
                w = abs(w).sum(axis=1)
                x = self.siteloc[inclchanis, 0] # 1D array (row)
                y = self.siteloc[inclchanis, 1]
                s['x0'], s['y0'], s['sx'], s['sy'] = weights2f(f, w, x, y, inclchani)

            if DEBUG: debug('*** found new spike %d: %r @ (%d, %d)'
                            % (nspikes+self.nspikes, s['t'], self.siteloc[chani, 0],
                               self.siteloc[chani, 1]))

            # give each chan a distinct lockout, based on how each chan's
            # sharpest peaks line up with those of the maxchan. Respect existing lockouts:
            # on each of the relvant chans, keep whichever lockout ends last
            lockouts[chanis] = np.max([lockouts[chanis], t0i+tis.max(axis=1)], axis=0)
            if DEBUG: debug('lockouts=%r\nfor chans=%r' %
                           (list(wave.ts[lockouts[chanis]]), list(self.chans[chanis])))
            nspikes += 1

        # shrink spikes and wavedata down to actual needed size
        spikes.resize(nspikes, refcheck=False)
        wds = wavedata.shape
        wavedata.resize((nspikes, wds[1], wds[2]), refcheck=False)
        return spikes, wavedata

    def get_blockranges(self, bs, bx):
        """Generate time ranges for slightly overlapping blocks of contiguous data that
        span self.trange, given blocksize and blockexcess"""
        stream = self.sort.stream
        bs = abs(bs)
        bx = abs(bx)
        if not self.trange[0] <= self.trange[1]: # not a forward search
            raise RuntimeError('backward detection not allowed')

        tranges = stream.tranges
        # pick out all tranges that overlap with self.trange
        trangesi = (self.trange[0] < tranges[:, 1]) & (tranges[:, 0] < self.trange[1])
        tranges = tranges[trangesi]

        blockranges = []
        for trange in tranges: # iterate over contiguous time ranges
            br = [] # list of blockranges for this trange
            # constrain in case self.trange falls within just one trange
            t0 = max(trange[0], self.trange[0])
            t1 = min(trange[1], self.trange[1])
            es = range(t0, t1, bs) # left edges of data blocks
            for e in es:
                br.append([e-bx, e+bs+bx]) # time range to give to .searchblock()
            br = np.asarray(br)
            # limit br to trange
            br[0, 0], br[-1, 1] = trange[0], trange[1]
            blockranges.append(br)

        blockranges = np.concatenate(blockranges)
        # limit blockranges to self.trange
        blockranges[0, 0], blockranges[-1, 1] = self.trange[0], self.trange[1]
        return np.asarray(blockranges)

    def get_thresh(self):
        """Return array of thresholds in AD units, one per chan in self.chans,
        according to threshmethod and noisemethod"""
        self.fixedthresh = self.sort.converter.uV2AD(self.fixedthreshuV) # convert to AD units
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed threshold
            thresh = np.tile(self.fixedthresh, len(self.chans))
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed threshold
            # randomly sample self.fixednoisewin's worth of data from self.trange in
            # blocks of self.blocksize, without replacement
            tload = time.time()
            print('loading data to calculate noise')
            if self.fixednoisewin >= abs(self.trange[1] - self.trange[0]):
                # sample width exceeds search trange
                blockranges = [self.trange] # use a single block of data, as defined by trange
            else:
                nblocks = intround(self.fixednoisewin / self.blocksize)
                blockranges = RandomBlockRanges(self.trange, bs=self.blocksize, bx=0,
                                                maxntranges=nblocks, replacement=False)
            # preallocating memory doesn't seem to help here, all the time is in loading
            # from stream:
            data = []
            for blockrange in blockranges:
                wave = self.sort.stream(blockrange[0], blockrange[1], self.chans)
                data.append(wave.data)
            data = np.concatenate(data, axis=1) # int16 AD units
            info('loading data to calc noise took %.3f sec' % (time.time()-tload))
            tnoise = time.time()
            noise = self.get_noise(data) # float AD units
            info('get_noise took %.3f sec' % (time.time()-tnoise))
            thresh = noise * self.noisemult # float AD units
            thresh = np.int16(np.round(thresh)) # int16 AD units
            # clip so that all thresholds are at least fixedthresh
            thresh = thresh.clip(self.fixedthresh, thresh.max())
        elif self.threshmethod == 'Dynamic':
            # dynamic thresholds are calculated on the fly during the search, so leave
            # as zero for now
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
            # np.abs does a copy, so modifying the result in-place is safe:
            noise = util.median_inplace_2Dshort(np.abs(data)) / 0.6745 # see Quiroga2004
            #noise = np.mean(np.abs(data), axis=-1) / 0.6745 / 1.2
            #noise = util.mean_2Dshort(np.abs(data)) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            #noise = pool.map(self.get_stdev, data) # multithreads over rows in data
            noise = np.std(data, axis=-1)
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
