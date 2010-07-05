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
#from spyke.core import WaveForm, toiter, argcut, intround, g, g2, RM
#from text import SimpleTable

#DMURANGE = 0, 500 # allowed time difference between peaks of modelled spike


def get_wave(obj, sort=None):
    """Return object's waveform, whether a spike record or a neuron,
    taken from sort.wavedata or sort.stream"""
    assert sort != None
    if type(obj) != np.void: # it's a Neuron
        n = obj
        if n.wave == None or n.wave.data == None:
            wave = n.update_wave() # call Neuron method
            return wave
        else:
            return n.wave # return existing neuron waveform
    # it's a spike record
    s = obj
    sid = int(s['id'])
    wave = sort.get_wave(sid)
    return wave


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
    wavetrange, direction = args
    detector = mp.current_process().detector
    return detector.searchblock(wavetrange, direction)

def initializer(detector, stream, srff):
    #stream.srff.open() # reopen the .srf file which was closed for pickling, engage file lock
    detector.sort.stream = stream
    detector.sort.stream.srff = srff # restore .srff that was deleted from stream on pickling
    mp.current_process().detector = detector

def minmax_filter(x, width=3):
    """An alternative to arglocalextrema, works on 2D arrays. Is about 10X slower
    than arglocalextrema for identical 1D arrays. Returns a boolean index array"""
    maxi = x == ndimage.maximum_filter(x, size=(1,width), mode='constant', cval=100000)
    mini = x == ndimage.minimum_filter(x, size=(1,width), mode='constant', cval=-100000)
    return maxi+mini

def argextrema1Damplitude(signal):
    """Return shape(signal) array of extremum amplitude and sign at each point
    in 1D int signal"""
    assert len(signal.shape) == 1
    # it's possible to have a local extremum at every timepoint on every chan
    ampl = np.zeros(signal.shape, dtype=np.float64) # +ve: max, -ve: min, abs: peak amplitude
    itemsize = signal.dtype.itemsize
    aitemsize = ampl.dtype.itemsize
    nt = len(signal)
    stride = int(signal.strides[0] // itemsize) # just timepoints to deal with
    astride = int(ampl.strides[0] // aitemsize)
    code = (r"""#line 110 "detect.py"
    int ti, n_ext, now, last, last2;

    n_ext = 0;
    // let's start from timepoint 2 (0-based)
    last = signal[stride]; // signal[(2-1)*stride] // signal 1 timepoint ago
    last2 = signal[0]; // signal[(2-2)*stride] // signal 2 timepoints ago
    for (ti=2; ti<nt; ti++) {
        now = signal[ti*stride]; // signal at current timepoint
        if ((last2 <= last && last > now) || (last2 >= last && last < now)) {
            // last is a max or min
            // alasti = (ti-1)*astride;
            ampl[(ti-1)*astride] = last; // save last amplitude at last index
            n_ext++;
        }
        // update for next loop
        last2 = last;
        last = now;
    }
    """)
    inline(code, ['signal', 'nt', 'stride', 'astride', 'ampl'])
    return ampl

def argextrema2Dsharpness(signal):
    """Return shape(signal) array of extremum sharpness and sign at
    each point in 2D int signal. Sharpness is defined as rise**2/run"""
    assert len(signal.shape) == 2
    # it's possible to have a local extremum at every timepoint on every chan
    sharp = np.zeros(signal.shape, dtype=np.float64) # +ve: max, -ve: min, abs: peak sharpness
    itemsize = signal.dtype.itemsize
    aitemsize = sharp.dtype.itemsize
    nchans = int(signal.shape[0])
    nt = int(signal.shape[1])
    stride0 = int(signal.strides[0] // itemsize)
    stride1 = int(signal.strides[1] // itemsize)
    astride0 = int(sharp.strides[0] // aitemsize)
    astride1 = int(sharp.strides[1] // aitemsize)
    extiw = np.empty(nt, dtype=np.int32) # for temp internal use
    code = (r"""#line 152 "detect.py"
    int ci, ti, ei, n_ext, cis0, i, ai, alasti, now, last, last2, thisti, lastti, nextti;
    double rise1, rise2, run1, run2;

    for (ci=0; ci<nchans; ci++) {
        n_ext = 0; // reset for each channel
        cis0 = ci*stride0;
        // let's start from timepoint 2 (0-based)
        last = signal[cis0 + stride1]; // signal[cis0 + (2-1)*stride1] // signal 1 timepoint ago
        last2 = signal[cis0]; // signal[cis0 + (2-2)*stride1] // signal 2 timepoints ago
        for (ti=2; ti<nt; ti++) {
            i = cis0 + ti*stride1;
            now = signal[i]; // signal at current timepoint
            if (last2 <= last && last > now) {
                // last is a max
                extiw[n_ext] = ti-1; // save previous time index
                alasti = ci*astride0 + (ti-1)*astride1;
                sharp[alasti] = 1; // +ve peak
                n_ext++;
            }
            else if (last2 >= last && last < now) {
                // last is a min
                extiw[n_ext] = ti-1; // save previous time index
                alasti = ci*astride0 + (ti-1)*astride1;
                sharp[alasti] = -1; // -ve peak
                n_ext++;
            }
            // update for next loop
            last2 = last;
            last = now;
        }
        // now calculate sharpness of each peak found on this channel
        for (ei=0; ei<n_ext; ei++) { // iterate over extremum indices, calc rise**2/run for each extremum
            thisti = extiw[ei];
            //lastti = 0;   // use first data point as previous reference for 1st extremum
            //nextti = nt-1; // use last data point as next reference for last extremum
            if (ei == 0) { lastti = 0; } // use first data point as previous reference for 1st extremum
            else { lastti = extiw[ei-1]; }
            if (ei == n_ext-1) { nextti = nt-1; } // use last data point as next reference for last extremum
            else { nextti = extiw[ei+1]; }
            // get total rise and run of both sides of this extremum
            i = cis0 + thisti*stride1;
            ai = ci*astride0 + thisti*astride1;
            rise1 = (double)signal[i] - (double)signal[cis0 + lastti*stride1];
            rise2 = (double)signal[i] - (double)signal[cis0 + nextti*stride1];
            run1 = (double)(thisti - lastti);
            run2 = (double)(nextti - thisti);
            //if (ei == 5) {
            //    printf("thisti %d;\n", thisti);
            //    printf("lastti %d;\n", lastti);
            //    printf("signal[%d] == %d\n", thisti, signal[i]);
            //    printf("signal[%d] == %d\n", lastti, signal[cis0 + lastti*stride1]);
            //    printf("%f, %f, %f, %f;\n", rise1, rise1*rise1, run1, rise1*rise1/run1); }
            sharp[ai] *= rise1*rise1/run1 + rise2*rise2/run2; // preserve existing sign in sharp
        }
    }
    """)
    inline(code, ['signal', 'nchans', 'nt', 'stride0', 'stride1', 'astride0', 'astride1',
           'extiw', 'sharp'])
    return sharp

def get_edges(wave, thresh):
    """Return n x 2 array (ti, chani) of indices of all threshold crossings
    in wave.data. Total wave.data should have no more than 2**31 elements in it"""
    '''
    # using pure numpy this way is slow:
    edges = np.diff(np.int8( np.abs(wave.data) >= np.vstack(self.thresh) )) # indices where changing abs(signal) has crossed thresh
    edgeis = np.where(edges.T == 1) # indices of +ve edges, where increasing abs(signal) has crossed thresh
    edgeis = np.transpose(edgeis) # columns are [ti, chani], rows temporally sorted
    for i, edgei in enumerate(edgeis):
        print("edge %d, (%d, %d)" % (i+1, edgei[0], edgei[1]))
    return edgeis
    '''
    data = wave.data
    assert data.size < 2**31 # we're sticking with signed int32 indices for speed
    itemsize = data.dtype.itemsize
    stride0 = int(data.strides[0] // itemsize)
    stride1 = int(data.strides[1] // itemsize)
    #assert (thresh >= 0).all() # assume it's passed as +ve
    # NOTE: taking abs(data) in advance doesn't seem faster than constantly calling abs() in the loop
    nchans = int(data.shape[0])
    nt = int(data.shape[1])
    #assert nchans == len(thresh)
    # TODO: this could be sped up by declaring a pointer to data and calculating byte
    # offsets directly in the C code, instead of relying on weave to do it for you
    code = (r"""#line 193 "detect.py"
    int nd = 2; // num dimensions of output edgeis array
    npy_intp dimsarr[2]; // can't use var nd as length, need a constant for msvc
    dimsarr[0] = 65536; // nrows, 2**16
    dimsarr[1] = 2;     // ncols
    PyArrayObject *edgeis = (PyArrayObject *) PyArray_SimpleNew(nd, dimsarr, NPY_INT);

    PyArray_Dims dims; // stores current dimension info of edgeis array
    dims.len = nd;
    dims.ptr = dimsarr;
    PyObject *OK;

    int nedges = 0;
    int i; // can get 12 hours of timestamps at 50 kHz with signed int32, but only 0.2 hours of sample
           // indices for 54 chans, but wave.data should be much shorter than that anyway
    for (int ti=1; ti<nt; ti++) { // start at 1'th timepoint so we can index back 1 timepoint into the past
        for (int ci=0; ci<nchans; ci++) {
            i = ci*stride0 + ti*stride1; // calculate only once for speed
            if (abs(data[i]) >= thresh[ci] && abs(data[i-stride1]) < thresh[ci]) {
                // abs(voltage) has crossed threshold
                if (nedges == PyArray_DIM(edgeis, 0)) { // allocate more rows to edgeis array
                    printf("allocating more memory!\n");
                    dims.ptr[0] *= 2; // double num rows in edgeis
                    OK = PyArray_Resize(edgeis, &dims, 0, NPY_ANYORDER); // 0 arg means don't check refcount or edgeis
                    if (OK == NULL) {
                        PyErr_Format(PyExc_TypeError, "can't resize edgeis");
                        return NULL;
                    }
                    // don't need 'OK' anymore I guess, see
                    // http://www.mail-archive.com/numpy-discussion@scipy.org/msg13013.html
                    Py_DECREF(OK);
                    printf("edgeis is now %d long\n", dims.ptr[0]);
                }
                // get pointer to i,jth entry in data, typecast appropriately,
                // then dereference the whole thing so you can assign
                // a value to it. Using PyArray_GETPTR2 macro is easier than
                // manually doing pointer math using strides, but might be slower?
                *((int *) PyArray_GETPTR2(edgeis, nedges, 0)) = ti; // assign to nedges'th row, col 0
                *((int *) PyArray_GETPTR2(edgeis, nedges, 1)) = ci; // assign to nedges'th row, col 1
                nedges++;
                // multi arg doesn't print right, even with %ld formatter, need a %lld formatter
                //printf("edge %d: (%d, %d)\n", nedges, ti, ci);
                // use this hack instead:
                //printf("edge %d: ", nedges);
                //printf("(%d, ", ti);
                //printf("%d)\n", ci);
            }
        }
    }

    // resize edgeis once more to reduce edgeis down to
    // just those values that were added to it
    dims.ptr[0] = nedges;
    OK = PyArray_Resize(edgeis, &dims, 0, NPY_ANYORDER);
    if (OK == NULL) {
        PyErr_Format(PyExc_TypeError, "can't resize edgeis");
        return NULL;
    }
    Py_DECREF(OK);
    //printf("shrunk edgeis to be %d long\n", dims.ptr[0]);
    //return_val = (PyObject *) edgeis;  // these two both
    return_val = PyArray_Return(edgeis); // seem to work
    """)
    edgeis = inline(code, ['data', 'nchans', 'nt', 'stride0', 'stride1', 'thresh'])
    print("%s: found %d edges" % (mp.current_process().name, len(edgeis)))
    return edgeis


class FoundEnoughSpikesError(ValueError):
    pass

class NoPeakError(ValueError):
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
    DEFDT = 370 # max time between phases of a single spike, us
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
        #self.lockouts = np.zeros(self.nchans, dtype=np.int64) # holds time indices until which each enabled chani is locked out, updated on every found spike
        #self.lockouts_us = np.zeros(nchans, dtype=np.int64) # holds times in us until which each enabled chani is locked out, updated only at end of each searchblock call
        self.nspikes = 0 # total num spikes found across all chans so far by this Detector, reset at start of every search
        #spikes = np.zeros(0, self.SPIKEDTYPE) # init

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
        # TODO: simplify the whole channel deselection and indexing approach, maybe
        # make all chanis always index into the full probe chan layout instead of the self.chans
        # that represent which chans are enabled for this detector. Also, maybe do away with
        # the whole slicing/indexing into a WaveForm object - just work directly on the .data
        # and .chans and .ts, convenience be damned
        tres = stream.tres
        #self.lockouts = np.int64((self.lockouts_us - wave.ts[0]) / tres)
        #self.lockouts[self.lockouts < 0] = 0 # don't allow -ve lockout indices
        #info('at start of searchblock:\n new wave.ts[0, end] = %s\n new lockouts = %s' %
        #     ((wave.ts[0], wave.ts[-1]), self.lockouts))

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

        tget_edges = time.time()
        edgeis = get_edges(wave, self.thresh)
        info('%s: get_edges() took %.3f sec' %
            (mp.current_process().name, time.time()-tget_edges))
        tcheck_edges = time.time()
        spikes, wavedata = self.check_edges(wave, edgeis, cutrange)
        info('%s: checking edges took %.3f sec' %
            (mp.current_process().name, time.time()-tcheck_edges))
        print('%s: found %d spikes' % (mp.current_process().name, len(spikes)))
        #import cProfile
        #cProfile.runctx('spikes = self.check_edges(wave, edgeis, cutrange)', globals(), locals())
        #spikes = []
        #self.lockouts_us = wave.ts[self.lockouts] # lockouts in us, use this to propagate lockouts to next searchblock call
        #info('at end of searchblock:\n lockouts = %s\n new lockouts_us = %s' %
        #     (self.lockouts, self.lockouts_us))
        return spikes, wavedata

    def check_edges(self, wave, edgeis, cutrange):
        """Check which edges (threshold crossings) in wave data look like spikes
        and return only events that fall within cutrange. Search in window
        forward from thresh for a peak, then in appropriate direction from
        that peak (based on sign of signal) for up to self.dt for another
        one of opposite sign. If you don't find a 2nd one that meets these
        criteria, it ain't a spike.

        TODO: would be nice to use some multichannel thresholding, instead
        of just single independent channel
            - e.g. obvious but small multichan spike at ptc15.87.23340
            - hyperellipsoidal?
            - take mean of sets of chans (say one set per chan, lockr of chans
            around it), check when they exceed thresh, find max chan within
            that set at that time and report it as a threshold event
            - or slide some filter across the data in space and time that not
            only checks for thresh, but ppthresh as well

        TODO: make lockout in space and time proportional to the size (and slope?) of signal
        on each chan at the 2nd phase on the maxchan
            - on the maxchan, lockout for some determined time after 2nd phase (say one dphase),
            on others lock out a proportionally less amount in time (say V2/V2maxchan*dphase)
            - should help with nearly overlapping spikes, such as at ptc15.87.89740
        - or more crudely?: for chans within lockr radius, lockout only those that
        exceeded thresh within the window

        TODO: search local window in space and time *simultaneously* for biggest signal,
        deal with biggest one first, maybe try to associate it with the nearest preceding thresh xing,
        then go back after applying lockout and deal with the
        smaller fry. ex. ptc15.87.125820 and 89760.
        Also, see error choosing the wrong maxchan due to sequential time-space-time-space
        search at ptc15.87.68420 (should detect grey maxchan 7, not slightly earlier magenta maxchan 46)
            - maybe partition the data into 2D tiles with some overlap

        TODO: keep an eye on broad spike at ptc15.87.1024880, about 340 us wide. Should be counted though
        """
        sort = self.sort
        AD2uV = sort.converter.AD2uV
        if self.extractparamsondetect:
            wavedata2spatial = sort.extractor.wavedata2spatial
            #wavedata2wcs = sort.extractor.wavedata2wcs
        #lockouts = self.lockouts
        lockouts = np.zeros(self.nchans, dtype=np.int64) # holds time indices for each enabled chan until which each enabled chani is locked out, updated on every found spike

        dti = self.dti
        twi = sort.twi
        nspikes = 0
        nedges = len(edgeis)
        spikes = np.zeros(nedges, self.SPIKEDTYPE) # nspikes will always be << nedgeis
        # TODO: test whether np.empty or np.zeros is faster overall in this case
        wavedata = np.empty((nedges, self.maxnchansperspike, self.maxnt), dtype=np.int16)
        # check each edge for validity
        for ti, chani in edgeis: # ti begins life as the threshold xing time index
            if DEBUG: debug('*** trying thresh event at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))
            # is this thresh crossing timepoint locked out?
            # make sure there's enough non-locked out signal before thresh crossing to give
            # at least twi[0]//2 datapoints before the peak - this avoids "spikes" that only
            # constitute a tiny blip right at the left edge of your data window, and then simply
            # decay slowly over the course of the window
            #if lockouts[chani] >= ti+twi[0]//2:
            if ti <= lockouts[chani]:
                if DEBUG: debug('thresh event is locked out')
                continue # skip to next event

            # find all enabled chanis within locknbhd of chani, exclude those locked-out at threshold xing
            #lockchanis = self.locknbhdi[chani]
            #chanis = lockchanis[lockouts[lockchanis] < ti]
            # no need to check lockouts, lockouts are checked in amplis loop below
            chanis = self.locknbhdi[chani]

            # get data window wrt threshold crossing
            # clip t0i to 0 since don't need full width wave just yet
            t0i = max(ti+twi[0], 0) # check for lockouts in amplis loop below,
            # not true: only need to search forward from thresh xing for the peak
            # actually: need to search forward and backward for sharpest peak, not biggest
            #t0i = ti # check for lockouts in amplis loop below
            tendi = ti+twi[1]+1 # +1 makes it end inclusive, don't worry about slicing past end
            window2D = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous

            # TODO: search 2*DT in either direction from thresh xing for sharpness values, save
            # the whole thing for companion search, then do subsearch of just DT in either direction
            # for sharpest extremum. Then simply pick the extrema to the left and right of the
            # found sharpest extremum, and you're done.

            # do spatiotemporal search for all local extrema in window,
            # decide which extremum is sharpest
            sharp = spyke.util.sharpness2D(window2D)
            # find max abs(sharpness) that isn't locked out
            sharpis = abs(sharp.ravel()).argsort() # to get chani and ti of each sort index, reshape to sharp.shape
            sharpis = sharpis[::-1] # reverse for highest to lowest abs(sharpness)
            ncols = window2D.shape[1]
            for sharpi in sharpis:
                rowi = sharpi // ncols
                coli = sharpi % ncols
                chani = chanis[rowi]
                ti = t0i + coli
                if ti > lockouts[chani]:# and abs(window2D[rowi, coli]) > self.thresh[chani]:
                    # extremum is not locked out
                    if DEBUG: debug('found peak at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))
                    break # found valid extremum with biggest relative sharpness
                else: # extremum is locked out (rare)
                    if DEBUG: debug('extremum at t=%d chan=%d is locked out' % (wave.ts[ti], self.chans[chani]))
            else:
                if DEBUG: debug('all extrema are locked out')
                continue # skip to next event

            # get 1D window +/- dti+1 around ti on chani, look for the other spike phase
            t0i = max(ti-dti-1, lockouts[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            #if t0i < 0:
            #    continue # too close to start of wave to get full width window, abort
            tendi = ti+dti+1 # +1 makes it end inclusive, don't worry about slicing past end
            window = wave.data[chani, t0i:tendi] # single chan window of data, not necessarily contiguous
            tiw = int(ti - t0i) # time index where ti falls wrt the window
            window.shape = 1, len(window) # make it 2D
            ampl = spyke.util.sharpness2D(window).squeeze() # this is 1D
            window = window.squeeze() # back to 1D
            # if tiw is a max, choose only extrema that are min, and vice versa
            if window[tiw-1] >= window[tiw] < window[tiw+1]: # main phase is a min
                ampl[ampl < 0] = 0 # null all the min entries, leave only max entries
            else: # main phase is a max
                ampl[ampl > 0] = 0 # null all the max entries, leave only min entries
            if (ampl == 0).all(): # no extrema left
                if DEBUG: debug("couldn't find a matching peak to extremum at "
                                "t=%d chan=%d" % (wave.ts[ti], self.chans[chani]))
                continue # skip to next event

            # decide which is the companion phase to the main phase
            #companiontiw = abs(ampl).argmax()
            companiontiws = abs(ampl).argsort()[::-1] # decreasing order
            companiontiws = companiontiws[0:2] # biggest two (or one)
            # calc updated absolute lockout wrt start of wave while we're at it,
            # don't apply it until the very end
            lockoutti = np.concatenate([[tiw], companiontiws]).max()
            lockout = t0i + lockoutti
            phasetis = np.sort([tiw, companiontiws[0]]) # keep them in temporal order
            Vs = window[phasetis] # maintain sign
            # temporarily make phasetis absolute indices into wave
            phasetis += t0i

            # align spikes by their min phase by default
            aligni = Vs.argmin()
            ti = phasetis[aligni] # overwrite ti, absolute time index to align to
            if lockouts[chani] >= ti:
                import pdb; pdb.set_trace() # this shouldn't happen, since t0i should respect lockout
                continue # this event is locked out when aligned to min

            # get full-sized window wrt ti, finalized in time and in chans
            #t0i = max(ti+twi[0], lockouts[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            t0i = ti+twi[0] # make final window always be full width, lockouts be damned
            #tendi = min(ti+twi[1]+1, maxti) # +1 makes it end inclusive, don't go further than last wave timepoint
            tendi = ti+twi[1]+1 # make window always be full width, lockouts be damned
            # update channel neighbourhood
            # find all enabled chanis within inclnbhd of chani, exclude those locked-out at ti
            #inclchanis = self.inclnbhdi[chani]
            #chanis = inclchanis[lockouts[inclchanis] < ti]
            chanis = self.inclnbhdi[chani] # give final waveform the full suite of chans, lockouts be damned
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous

            # make phasetis relative to new t0i
            phasetis -= t0i
            '''
            # find indices into chanis that give you only the chanis that exceed
            # 1/4 the spike amplitude at the spike time, with the correct sign
            V = Vs[aligni]
            Vi = phasetis[aligni]
            chaniis = window[:, Vi] / V >= 1/6 # excludes chans with wrong sign
            '''
            '''
            # find indices into chanis that give you only the chanis that exceed
            # half the current per-chan thresh
            inclthresh = self.thresh[chanis]/3 # 1D vector
            inclthresh.shape = len(inclthresh), 1 # 2D column vector
            chaniis = np.unique(np.where((abs(window) > inclthresh))[0])
            # update chanis
            chanis = chanis[chaniis]
            # update multichan window
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous
            '''
            try:
                assert cutrange[0] <= wave.ts[ti] <= cutrange[1], 'spike time %r falls outside cutrange for this searchblock call, discarding' % wave.ts[ti] # use %r since s['t'] is np.int64 and %d gives TypeError if > 2**31
            except AssertionError, message: # doesn't qualify as a spike, don't change lockouts
                if DEBUG: debug(message)
                continue # skip to next event
            if DEBUG: debug('final window params: t0=%d, tend=%d, phasets=%r, Vs=%r'
                            % (wave.ts[t0i], wave.ts[tendi], wave.ts[t0i+phasetis], AD2uV(Vs)))
            Vpp = abs(Vs[1]-Vs[0]) # don't maintain sign
            if Vpp < self.ppthresh[chani]:
                if DEBUG: debug("matched companion peak to extremum at "
                                "t=%d chan=%d gives only %d Vpp" % (wave.ts[ti], self.chans[chani], AD2uV(Vpp)))
                continue # skip to next event

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

            # TODO: add a sharpi field to designate which of the 2 phases is the sharpest (main) phase - use this for extractor.get_Vp_weights()

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

            if DEBUG: debug('*** found new spike: %d @ (%d, %d)' % (s['t'], self.siteloc[chani, 0], self.siteloc[chani, 1]))

            lockchanis = self.locknbhdi[chani]

            # update lockouts to just past the last phase of this spike
            # TODO: lock out to the latest of the 3 sharpest extrema. Some spikes are more
            # than biphasic, with a significant 3rd phase (see ptc18.14.24570980),
            # but never more than 3 significant phases
            #import pdb; pdb.set_trace()
            '''
            for lockchani, row in zip(lockchanis, window):
                i = np.where(abs(row) > self.thresh[lockchani])[0]
                if i.any(): lockouts[lockchani] = t0i + i.max()
            '''
            #dphaseti = phasetis[1] - phasetis[0]
            #lockout = t0i + phasetis[1] + dphaseti

            # TODO: lock out only those chans that exceed thresh within -dphase/2
            # of V0 and +dphase/2 of V1. Lock such chans out up to the latest
            # of the thresh exceeding peaks that are found. This should fix double
            # trigger at ptc18.14.7012560
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
