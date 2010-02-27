"""Spike detection and modelling"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
import sys
import time
import string
import logging
import datetime
import multiprocessing as mp
from copy import copy

import wx
import pylab
import matplotlib as mpl

import numpy as np
from scipy.weave import inline
#from scipy import ndimage
#from scipy.optimize import leastsq, fmin_slsqp
#import openopt
#import nmpfit

import spyke.surf
from spyke.core import eucd
from spyke import threadpool
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
    ri = sort.spikes['id'].searchsorted(s['id']) # returns an array
    ri = int(ri)
    wave = sort.get_wave(ri)
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
    """Return indices of all local extrema in 1D int signal
    Also return array designating each extremum's type (max or min)
    and its amplitude."""
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
    """Return indices of all local extrema in 2D int signal
    Also return array designating each extremum's type (max or min)
    and its "amplitude", which is actually its sharpness defined as
    rise**2/run"""
    assert len(signal.shape) == 2
    # it's possible to have a local extremum at every timepoint on every chan
    ampl = np.zeros(signal.shape, dtype=np.float64) # +ve: max, -ve: min, abs: peak sharpness
    itemsize = signal.dtype.itemsize
    aitemsize = ampl.dtype.itemsize
    nchans = int(signal.shape[0])
    nt = int(signal.shape[1])
    stride0 = int(signal.strides[0] // itemsize)
    stride1 = int(signal.strides[1] // itemsize)
    astride0 = int(ampl.strides[0] // aitemsize)
    astride1 = int(ampl.strides[1] // aitemsize)
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
                ampl[alasti] = 1; // +ve peak
                n_ext++;
            }
            else if (last2 >= last && last < now) {
                // last is a min
                extiw[n_ext] = ti-1; // save previous time index
                alasti = ci*astride0 + (ti-1)*astride1;
                ampl[alasti] = -1; // -ve peak
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
            ampl[ai] *= rise1*rise1/run1 + rise2*rise2/run2; // preserve existing sign in ampl
        }
    }
    """)
    inline(code, ['signal', 'nchans', 'nt', 'stride0', 'stride1', 'astride0', 'astride1',
           'extiw', 'ampl'])
    return ampl

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

'''
# no longer necessary, now that spikes struct array is being used
class Spike(object):
    """A Spike"""
    def __eq__(self, other):
        """Compare Spikes according to their hashes"""
        if self.__class__ != other.__class__: # might be comparing a Spike with a Neuron
            return False
        return hash(self) == hash(other) # good enough for just simple detection

    def __hash__(self):
        """Unique hash value for self, based on spike time and location.
        Required for effectively using Spikes in a Set, and for testing equality"""
        return hash((self.t, self.chani)) # hash of their tuple, should guarantee uniqueness

    def __repr__(self):
        chan = self.detection.detector.chans[self.chani] # dereference
        return str((self.t, chan))

    def __getstate__(self):
        """Get object state for pickling"""
        d = self.__dict__.copy() # this doesn't seem to be a slow step
        # when deleting a dict entry, the strategy here is to use
        # d.pop(entry, None) to not raise an error if the entry doesn't exist
        if not self.detection.sort.SAVEWAVES:
            d.pop('wave', None) # clear wave (if any) to save space and time during pickling
        if 'neuron' in d and d['neuron'] == None:
            del d['neuron']
        # TODO: do spikes really need a .neuron attribute at all? How about just a .nid attribute? .nid would be invalid
        # after a renumbering of the neurons
        d.pop('plt', None) # clear plot (if any) self is assigned to, since that'll have changed anyway on unpickle
        d.pop('itemID', None) # clear tree item ID (if any) since that'll have changed anyway on unpickle
        return d

    # TODO: Could potentially define __setstate___ to reset .wave, .neuron,
    # .plt, and .itemID back to None if they don't exist in the d returned
    # from the pickle. This might make it easier to work with other
    # code that relies on all of these attribs exsting all the time"""

    def get_chan(self):
        return self.detection.detector.chans[self.chani] # dereference max chani

    def set_chan(self, chans):
        raise RuntimeError("spike.chan isn't settable, set spike.chani instead")

    chan = property(get_chan, set_chan)

    def get_chans(self):
        return self.detection.detector.chans[self.chanis] # dereference

    def set_chans(self, chans):
        raise RuntimeError("spike.chans isn't settable, set spike.chanis instead")

    chans = property(get_chans, set_chans)

    def update_wave(self, stream, tw=None):
        """Load/update self's waveform taken from the given stream.
        Optionally slice it according to tw around self's spike time"""
        if stream == None:
            raise RuntimeError("No stream open, can't update waveform for spike %d" % self.id)
        if self.detection.detector.srffname != stream.srffname:
            msg = ("Spike %d was extracted from .srf file %s.\n"
                   "The currently opened .srf file is %s.\n"
                   "Can't update spike %d's waveform." %
                   (self.id, self.detection.detector.srffname, stream.srffname, self.id))
            wx.MessageBox(msg, caption="Error", style=wx.OK|wx.ICON_EXCLAMATION)
            raise RuntimeError(msg)
        wave = stream[self.t0 : self.tend]
        ts = np.arange(self.t0, self.tend, stream.tres) # build them up
        # can't do this cuz chanis indexes only into enabled chans,
        # not into all stream chans represented in data array:
        #data = wave.data[self.chanis]
        chans = self.detection.detector.chans[self.chanis] # dereference
        data = wave[chans].data # maybe a bit slower, but correct
        #assert data.shape[1] == len(np.arange(s.t0, s.tend, stream.tres)) # make sure I know what I'm doing
        self.wave = WaveForm(data=data, ts=ts, chans=chans)
        if tw != None:
            self.wave = self[self.t+tw[0] : self.t+tw[1]]
        return self.wave


class SpikeModel(Spike):
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
        return np.all(self.p == other.p) # disable for now while not modelling

    def __getitem__(self, key):
        """Return WaveForm for this spike given slice key"""
        assert type(key) == slice
        if self.wave != None:
            return self.wave[key] # slice existing .wave
        else: # existing .wave unavailable
            return WaveForm() # return empty waveform

    def __getstate__(self):
        """Get object state for pickling"""
        d = Spike.__getstate__(self)
        d['errs'] = None
        return d

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
        """
        c = mpl.patches.Circle((0, yrange-15), radius=15, # for calibrating aspect ratio of display
                                ec='#ffffff', fill=False, ls='dotted')
        a.add_patch(c)
        """
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
'''

class Detector(object):
    """Spike detector base class"""
    DEFTHRESHMETHOD = 'ChanFixed' # GlobalFixed, ChanFixed, or Dynamic
    DEFNOISEMETHOD = 'median' # median or stdev
    DEFNOISEMULT = 5
    DEFFIXEDTHRESH = 50 # uV, used by GlobalFixed, and as min thresh for ChanFixed
    DEFPPTHRESHMULT = 1.5 # peak-to-peak threshold is this times thresh
    DEFFIXEDNOISEWIN = 30000000 # 30s, used by ChanFixed - this should really be a % of self.trange
    DEFDYNAMICNOISEWIN = 10000 # 10ms, used by Dynamic
    DEFMAXNSPIKES = 0
    DEFMAXNCHANSPERSPIKE = 25 # overrides spatial lockout, 9 seems to give greatest clusterability for uMap54_2b probe.
                             # when using spatial mean extraction, setting this to a bad value
                             # can give artificially segregated clusters in space
    DEFBLOCKSIZE = 10000000 # 10s, waveform data block size
    DEFSLOCK = 150 # spatial lockout radius, um
    DEFDT = 370 # max time between phases of a single spike, us
    DEFRANDOMSAMPLE = False
    #DEFKEEPSPIKEWAVESONDETECT = False # turn this off is to save memory during detection, or during multiprocessing
    DEFEXTRACTPARAMSONDETECT = True

    # us, extra data as buffer at start and end of a block while detecting spikes.
    # Only useful for ensuring spike times within the actual block time range are
    # accurate. Spikes detected in the excess are discarded
    BLOCKEXCESS = 1000

    def __init__(self, sort, chans=None,
                 threshmethod=None, noisemethod=None, noisemult=None, fixedthresh=None,
                 ppthreshmult=None, fixednoisewin=None, dynamicnoisewin=None,
                 trange=None, maxnspikes=None, maxnchansperspike=None,
                 blocksize=None, slock=None, dt=None, randomsample=None,
                 #keepspikewavesondetect=None,
                 extractparamsondetect=None):
        """Takes a parent Sort session and sets various parameters"""
        self.sort = sort
        self.srffname = sort.stream.srffname # for reference, store which .srf file this Detector is run on
        self.chans = np.asarray(chans) or np.arange(sort.stream.nchans) # None means search all channels
        self.threshmethod = threshmethod or self.DEFTHRESHMETHOD
        self.noisemethod = noisemethod or self.DEFNOISEMETHOD
        self.noisemult = noisemult or self.DEFNOISEMULT
        self.fixedthresh = fixedthresh or self.DEFFIXEDTHRESH
        self.ppthreshmult = ppthreshmult or self.DEFPPTHRESHMULT
        self.fixednoisewin = fixednoisewin or self.DEFFIXEDNOISEWIN # us
        self.dynamicnoisewin = dynamicnoisewin or self.DEFDYNAMICNOISEWIN # us
        self.trange = trange or (sort.stream.t0, sort.stream.tend)
        self.maxnspikes = maxnspikes or self.DEFMAXNSPIKES # return at most this many spikes
        self.maxnchansperspike = maxnchansperspike or self.DEFMAXNCHANSPERSPIKE
        self.blocksize = blocksize or self.DEFBLOCKSIZE
        self.slock = slock or self.DEFSLOCK
        self.dt = dt or self.DEFDT
        self.randomsample = randomsample or self.DEFRANDOMSAMPLE
        #self.keepspikewavesondetect = keepspikewavesondetect or self.DEFKEEPSPIKEWAVESONDETECT
        self.extractparamsondetect = extractparamsondetect or self.DEFEXTRACTPARAMSONDETECT

        #self.dmurange = DMURANGE # allowed time difference between peaks of modelled spike

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
            ncpus = mp.cpu_count() # 1 per core
            pool = mp.Pool(ncpus, initializer, (self, stream, stream.srff)) # sends pickled copies to each process
            directions = [direction]*len(wavetranges)
            args = zip(wavetranges, directions)
            # TODO: FoundEnoughSpikesError is no longer being caught in multiprocessor code
            results = pool.map(callsearchblock, args, chunksize=1)
            pool.close()
            #pool.join() # unnecessary, I think
            blockspikes, blockwavedata = zip(*results) # results is a list of (spikes, wavedata) tuples, and needs to be unzipped
            spikes = np.concatenate(blockspikes)
            wavedata = np.concatenate(blockwavedata) # along spikei axis, all other dims are identical
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
        sort.wavedata = wavedata
        self.nspikes = len(spikes)
        assert len(wavedata) == self.nspikes
        # default -1 indicates no nid or detid is set as of yet, reserve 0 for actual ids
        spikes['nid'] = -1
        spikes['detid'] = -1
        info('\nfound %d spikes in total' % self.nspikes)
        info('inside .detect() took %.3f sec' % (time.time()-t0))
        return spikes

    def calc_chans(self):
        """Calculate max number of chans to use, and related stuff"""
        sort = self.sort
        self.enabledSiteLoc = {}
        for chan in self.chans: # for all enabled chans
            self.enabledSiteLoc[chan] = sort.stream.probe.SiteLoc[chan] # grab its (x, y) coordinate
        self.dm = DistanceMatrix(self.enabledSiteLoc) # distance matrix for the chans enabled for this search, sorted by chans
        #self.nbhd = {} # dict of neighbourhood of chans for each chan, as defined by self.slock, each sorted by ascending distance
        self.nbhdi = {} # corresponding dict of neighbourhood of chanis for each chani
        maxnchansperspike = 0
        for chani, distances in enumerate(self.dm.data): # iterate over rows of distances
            chanis, = np.uint8(np.where(distances <= self.slock)) # at what col indices does the returned row fall within slock?
            if len(chanis) > self.maxnchansperspike: # exceeds the hard upper limit
                ds = distances[chanis] # pick out relevant distances
                chaniis = ds.argsort() # indices that sort chanis by distance
                chanis = chanis[chaniis] # chanis sorted by distance
                chanis = chanis[:self.maxnchansperspike] # pick out closest self.maxnchansperspike chanis
                chanis.sort() # sorted numerical order assumed later on!
            maxnchansperspike = max(maxnchansperspike, len(chanis))
            self.nbhdi[chani] = chanis
            #chan = self.dm.chans[chani]
            #chans = self.dm.chans[chanis]
            #self.nbhd[chan] = chans
        self.maxnchansperspike = min(self.maxnchansperspike, maxnchansperspike)

        for detection in sort.detections.values():
            det = detection.detector
            if self.maxnchansperspike != det.maxnchansperspike:
                raise RuntimeError("Can't have multiple detections generating spikes struct arrays with "
                                   "different width 'chans' fields")

        self.SPIKEDTYPE = [('id', np.int32), ('nid', np.int16), ('detid', np.uint8),
                           ('chan', np.uint8), ('chans', np.uint8, self.maxnchansperspike), ('nchans', np.uint8),
                           # TODO: maybe it would be more efficient to store ti, t0i,
                           # and tendi wrt start of surf file instead of times in us?
                           ('t', np.int64), ('t0', np.int64), ('tend', np.int64),
                           ('Vs', np.float32, 2), ('Vpp', np.float32),
                           ('phasetis', np.uint8, 2), ('aligni', np.uint8),
                           ('x0', np.float32), ('y0', np.float32), ('dphase', np.int16), # in us
                           ('IC0', np.float32), ('IC1', np.float32)
                           ]

    def searchblock(self, wavetrange, direction):
        """Search a block of data, return a list of valid Spikes"""
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
            - take mean of sets of chans (say one set per chan, slock of chans
            around it), check when they exceed thresh, find max chan within
            that set at that time and report it as a threshold event
            - or slide some filter across the data in space and time that not
            only checks for thresh, but ppthresh as well

        TODO: make lockout in space and time proportional to the size (and slope?) of signal
        on each chan at the 2nd phase on the maxchan
            - on the maxchan, lockout for some determined time after 2nd phase (say one dphase),
            on others lock out a proportionally less amount in time (say V2/V2maxchan*dphase)
            - should help with nearly overlapping spikes, such as at ptc15.87.89740
        - or more crudely?: for chans within slock radius, lockout only those that
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
            extract = sort.extractor.extract
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

            # find all enabled chanis within nbhd of chani, exclude those locked-out at threshold xing
            #nbhdchanis = self.nbhdi[chani]
            #chanis = nbhdchanis[lockouts[nbhdchanis] < ti]
            # no need to check lockouts, lockouts are checked in amplis loop below
            chanis = self.nbhdi[chani]

            # get data window wrt threshold crossing
            # clip t0i to 0 since don't need full width wave just yet
            t0i = max(ti+twi[0], 0) # check for lockouts in amplis loop below,
            # not true: only need to search forward from thresh xing for the peak
            # actually: need to search forward and backward for sharpest peak, not biggest
            #t0i = ti # check for lockouts in amplis loop below
            tendi = ti+twi[1]+1 # +1 makes it end inclusive, don't worry about slicing past end
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous

            # do spatiotemporal search for all local extrema in window,
            # decide which extremum is sharpest
            ampl = argextrema2Dsharpness(window)
            # find max abs(amplitude) that isn't locked out
            amplis = abs(ampl.ravel()).argsort() # to get chani and ti of each sort index, reshape to ampl.shape
            amplis = amplis[::-1] # reverse for highest to lowest abs(amplitude)
            ncols = window.shape[1]
            for ampli in amplis:
                rowi = ampli // ncols
                coli = ampli % ncols
                chani = chanis[rowi]
                ti = t0i + coli
                if ti > lockouts[chani]:# and abs(window[rowi, coli]) > self.thresh[chani]:
                    # extremum is not locked out
                    if DEBUG: debug('found peak at t=%d chan=%d' % (wave.ts[ti], self.chans[chani]))
                    break # found valid extremum with biggest relative amplitude
                else: # extremum is locked out (rare)
                    if DEBUG: debug('extremum at t=%d chan=%d is locked out' % (wave.ts[ti], self.chans[chani]))
            else:
                if DEBUG: debug('all extrema are locked out')
                continue # skip to next event

            # get window +/- dti+1 around ti on chani, look for the other spike phase
            t0i = max(ti-dti-1, lockouts[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            #t0i = ti-dti-1 # don't worry about lockout for companion phase
            #if t0i < 0:
            #    continue # too close to start of wave to get full width window, abort
            tendi = ti+dti+1 # +1 makes it end inclusive, don't worry about slicing past end
            window = wave.data[chani, t0i:tendi] # single chan window of data, not necessarily contiguous
            '''
            # check if window has global max or min at end of window,
            # if so, extend by dti/2 before searching for extrema
            if len(window)-1 in [window.argmax(), window.argmin()]:
                tendi = min(ti+dti+1+dti//2, maxti) # don't go further than last wave timepoint
                window = wave.data[chani, t0i:tendi] # single chan window of data, not necessarily contiguous
            '''
            tiw = int(ti - t0i) # time index where ti falls wrt the window
            ampl = argextrema1Damplitude(window) # this is 1D
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
            # find all enabled chanis within nbhd of chani, exclude those locked-out at ti
            #nbhdchanis = self.nbhdi[chani]
            #chanis = nbhdchanis[lockouts[nbhdchanis] < ti]
            chanis = self.nbhdi[chani] # give final waveform full suite of chans, lockouts be damned
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous

            # make phasetis relative to new t0i
            phasetis -= t0i

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
            # leave each spike's chanis in sorted order, as they are in self.nbhdi, important
            # assumption used later on, like in sort.get_wave() and Neuron.update_wave()
            ts = wave.ts[t0i:tendi]
            # use ts = np.arange(s['t0'], s['tend'], stream.tres) to reconstruct
            s['t0'], s['tend'] = wave.ts[t0i], wave.ts[tendi]
            s['phasetis'][:] = phasetis # wrt t0i, not sure why the [:] is necessary
            s['aligni'] = aligni # 0 or 1

            # TODO: add a sharpi field to designate which of the 2 phases is the sharpest (main) phase - use this for extractor.get_Vp_weights()

            s['dphase'] = ts[phasetis[1]] - ts[phasetis[0]] # in us
            s['Vs'][:] = AD2uV(Vs) # in uV, not sure why the [:] is necessary
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
                maxchani = int(np.where(chans == chan)[0])
                s['x0'], s['y0'] = extract(window, maxchani, phasetis, aligni, x, y)
            if DEBUG: debug('*** found new spike: %d @ (%d, %d)' % (s['t'], self.siteloc[chani, 0], self.siteloc[chani, 1]))

            # update lockouts to just past the last phase of this spike
            # TODO: lock out to the latest of the 3 sharpest extrema. Some spikes are more
            # than biphasic, with a significant 3rd phase (see ptc18.14.24570980),
            # but never more than 3 significant phases
            #import pdb; pdb.set_trace()
            '''
            for chani, row in zip(chanis, window):
                i = np.where(abs(row) > self.thresh[chani])[0]
                if i.any(): lockouts[chani] = t0i + i.max()
            '''
            #dphaseti = phasetis[1] - phasetis[0]
            #lockout = t0i + phasetis[1] + dphaseti

            # TODO: lock out only those chans that exceed thresh within -dphase/2
            # of V0 and +dphase/2 of V1. Lock such chans out up to the latest
            # of the thresh exceeding peaks that are found. This should fix double
            # trigger at ptc18.14.7012560
            lockouts[chanis] = lockout # same for all chans in this spike
            if DEBUG:
                lockoutt = wave.ts[lockout]
                debug('lockout=%d for chans=%s' % (lockoutt, chans))

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

    def get_thresh(self):
        """Return array of thresholds in AD units, one per chan in self.chans,
        according to threshmethod and noisemethod"""
        fixedthresh = self.sort.converter.uV2AD(self.fixedthresh) # convert to AD units
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            thresh = np.tile(fixedthresh, len(self.chans))
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
            thresh = thresh.clip(fixedthresh, thresh.max()) # clip so that all threshes are at least fixedthresh
        elif self.threshmethod == 'Dynamic':
            # dynamic threshes are calculated on the fly during the search, so leave as zero for now
            # or at least they were, in the Cython code
            #thresh = np.zeros(len(self.chans), dtype=np.float32)
            raise NotImplementedError
        else:
            raise ValueError
        #assert len(thresh) == len(self.chans)
        #assert thresh.dtype == np.float32
        return thresh

    def get_noise(self, data):
        """Calculates noise over last dim in data (time), using .noisemethod"""
        print('calculating noise')
        ncpus = mp.cpu_count()
        pool = threadpool.Pool(ncpus)
        if self.noisemethod == 'median':
            noise = pool.map(self.get_median, data) # multithreads over rows in data
            #noise = np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            noise = pool.map(self.get_stdev, data) # multithreads over rows in data
            #noise = np.stdev(data, axis=-1)
        else:
            raise ValueError
        pool.terminate() # pool.close() doesn't allow Python to exit when spyke is closed
        #pool.join() # unnecessary, hangs
        return np.asarray(noise)

    def get_median(self, data):
        """Return median value of multichan data, scaled according to Quiroga2004"""
        return np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004

    def get_stdev(self, data):
        """Return stdev of multichan data"""
        return np.stdev(data, axis=-1)




class Detection(object):
    """A spike detection run, which happens every time Detect is pressed.
    When you're merely searching for the previous/next spike with
    F2/F3, that's not considered a detection run"""
    def __init__(self, sort, detector, id=None, datetime=None):
        self.sort = sort
        self.detector = detector # Detector object used in this Detection run
        self.id = id
        self.datetime = datetime

    def get_spikeis(self):
        return np.arange(self.spikei0, self.spikei0+self.nspikes)

    def set_spikeis(self, spikes):
        """Give each spike an ID, inc sort's _sid spike ID counter, and save
        array of spikeis to self"""
        # don't iterate over spikes struct array, since that generates
        # a bunch of np.void objects, which is slow?
        self.spikei0 = self.sort._sid
        self.nspikes = len(spikes)
        # generate IDs in one shot and assign them to spikes struct array
        spikes['id'] = np.arange(self.spikei0, self.spikei0+self.nspikes)
        spikes['detid'] = self.id
        self.sort._sid += self.nspikes # inc for next unique Detection

    # array of spike IDs that came from this detection
    spikeis = property(get_spikeis, set_spikeis)
