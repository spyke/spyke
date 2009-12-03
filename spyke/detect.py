"""Spike detection and modelling"""

from __future__ import division

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
import sys
import time
import string
import logging
import datetime

import wx
import pylab
import matplotlib as mpl

import numpy as np
from scipy.weave import inline
from scipy import ndimage
#from scipy.optimize import leastsq, fmin_slsqp
#import openopt
#import nmpfit

import spyke.surf
from spyke.core import eucd
#from spyke.core import WaveForm, toiter, argcut, intround, g, g2, RM
#from text import SimpleTable


#DMURANGE = 0, 500 # allowed time difference between peaks of modelled spike


def get_wave(obj, sort=None):
    """Return object's waveform, whether a spike record or a neuron,
    taken from sort.wavedatas or sort.stream"""
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


def minmax_filter(x, width=3):
    """An alternative to arglocalextrema, works on 2D arrays. Is about 10X slower
    than arglocalextrema for identical 1D arrays. Returns a boolean index array"""
    maxi = x == ndimage.maximum_filter(x, size=(1,width), mode='constant', cval=100000)
    mini = x == ndimage.minimum_filter(x, size=(1,width), mode='constant', cval=-100000)
    return maxi+mini

def arglocalextrema(signal):
    """Return indices of all local extrema in 1D int signal"""
    nt = len(signal) # it's possible to have a local extremum at every point
    stride = signal.strides[0] // signal.dtype.itemsize # signal might not be contiguous
    exti = np.zeros(nt, dtype=int)
    code = ("""
    #line 91 "detect.py"
    int n_ext = 0;
    int now, last, last2;
    // let's start from timepoint 2 (0-based)
    last = signal[stride]; // signal[(2-1)*stride] // signal 1 timepoint ago
    last2 = signal[0]; // signal[(2-2)*stride] // signal 2 timepoints ago
    for (int ti=2; ti<nt; ti++) {
        now = signal[ti*stride]; // signal at current timepoint
        // Two methods, equally fast. First one isn't quite correct, 2nd one is.
        // Test with signal = np.array([0, -5, -5, -5, -2]) and signal = -signal
        // should get 3 as an answer in both cases
        // Method 1: not quite right
        // if ((last2 < last) == (last > now)) {
        // Method 2: gives correct answer for consecutive identical points, both +ve and -ve:
        if ((last2 <= last && last > now) || (last2 >= last && last < now)) {
            exti[n_ext] = ti-1; // save previous time index
            n_ext++;
        }
        // update for next loop, reuse 'now' value instead of constantly indexing into signal
        last2 = last;
        last = now;
    }
    return_val = n_ext;""")
    n_ext = inline(code, ['signal', 'nt', 'stride', 'exti'], compiler='gcc')
    return exti[:n_ext]

def arg2ndpeak(signal, exti, peak1i, dir2, dti, ppthresh):
    """Return signal's biggest local extremum of opposite sign,
    in direction dir2, and within dti indices of signal at peak1i"""
    #assert type(peak1i) == int
    #assert type(dti) == int
    stride = signal.strides[0] // signal.dtype.itemsize # signal might not be contiguous
    if dir2 == 'left':
        exti = exti[exti < peak1i] # keep only left half of exti
    elif dir2 == 'right':
        exti = exti[exti > peak1i] # keep only right half of exti
    elif dir2 == 'both':
        pass # keep all of exti
    else:
        raise ValueError('unknown dir2 %r' % dir2)
    # abs(now - peak1) converts to Python int, so ppthresh has to be same type for >= comparison
    ppthresh = int(ppthresh) # convert from np.int16 type to Python int
    n_ext = len(exti)
    code = ("""
    #line 103 "detect.py"
    // index into signal to get voltages
    int peak1 = signal[peak1i*stride]; // int or short, doesn't matter
    int peak2i = -1; // 2nd peak index, -1 indicates suitable 2nd peak not yet found
    int peak2 = 0; // 2nd peak value
    int i, now;
    // test all extrema in exti
    for (int ei=0; ei<n_ext; ei++) {
        i = exti[ei]; // ei'th extremum's index into signal
        now = signal[i*stride]; // signal value at i
        if ((abs(i-peak1i) <= dti) && // if extremum's index is within dti of peak1i
            (now * peak1 < 0) && // and is of opposite sign
            (abs(now) > abs(peak2)) && // and is bigger than last one found
            (abs(now - peak1) >= ppthresh)) { // and resulting Vpp exceeds ppthresh
                peak2i = i; // save it
                peak2 = now;
        }
    }
    return_val = peak2i;""")
    peak2i = inline(code, ['signal', 'stride', 'exti', 'n_ext', 'peak1i', 'dti', 'ppthresh'],
                    compiler='gcc')
    if peak2i == -1:
        raise NoPeakError("can't find suitable 2nd peak")
    return peak2i

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
    stride0 = data.strides[0] // itemsize
    stride1 = data.strides[1] // itemsize
    #assert (thresh >= 0).all() # assume it's passed as +ve
    # NOTE: taking abs(data) in advance doesn't seem faster than constantly calling abs() in the loop
    nchans, nt = data.shape
    #assert nchans == len(thresh)
    # TODO: this could be sped up by declaring a pointer to data and calculating byte
    # offsets directly in the C code, instead of relying on weave to do it for you
    code = (r"""
    #line 149 "detect.py"
    int nd = 2; // num dimensions of output edgeis array
    npy_intp dimsarr[nd];
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
    edgeis = inline(code, ['data', 'nchans', 'nt', 'stride0', 'stride1', 'thresh'],
                    compiler='gcc')
    print("found %d edges" % len(edgeis))
    return edgeis


class FoundEnoughSpikesError(ValueError):
    pass

class NoPeakError(ValueError):
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
    DEFTHRESHMETHOD = 'GlobalFixed' # GlobalFixed, ChanFixed, or Dynamic
    DEFNOISEMETHOD = 'median' # median or stdev
    DEFNOISEMULT = 4.0
    DEFFIXEDTHRESH = 60 # uV, used by GlobalFixed
    DEFPPTHRESHMULT = 1.5 # peak-to-peak threshold is this times thresh
    DEFFIXEDNOISEWIN = 10000000 # 10s, used by ChanFixed - this should really be a % of self.trange
    DEFDYNAMICNOISEWIN = 10000 # 10ms, used by Dynamic
    DEFMAXNSPIKES = 0
    DEFMAXNCHANSPERSPIKE = 9 # overrides spatial lockout, 9 seems to give greatest clusterability for uMap54_2b probe.
                             # when using spatial mean extraction, setting this to a bad value
                             # can give artificially segregated clusters in space
    DEFBLOCKSIZE = 10000000 # 10s, waveform data block size
    DEFSLOCK = 150 # spatial lockout radius, um
    DEFDT = 350 # max time between phases of a single spike, us
    DEFRANDOMSAMPLE = False
    DEFKEEPSPIKEWAVESONDETECT = True # only reason to turn this off is to save memory during detection
    DEFEXTRACTPARAMSONDETECT = False

    # us, extra data as buffer at start and end of a block while detecting spikes.
    # Only useful for ensuring spike times within the actual block time range are
    # accurate. Spikes detected in the excess are discarded
    BLOCKEXCESS = 1000

    def __init__(self, sort, chans=None,
                 threshmethod=None, noisemethod=None, noisemult=None, fixedthresh=None,
                 ppthreshmult=None, fixednoisewin=None, dynamicnoisewin=None,
                 trange=None, maxnspikes=None, maxnchansperspike=None,
                 blocksize=None, slock=None, dt=None, randomsample=None,
                 keepspikewavesondetect=None,
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
        self.keepspikewavesondetect = keepspikewavesondetect or self.DEFKEEPSPIKEWAVESONDETECT
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
        nt = int(sort.stream.sampfreq * spikewidth) # num timepoints to allocate per spike
        if self.keepspikewavesondetect:
            try: sort.wavedatas
            except AttributeError: sort.init_wavedata(nchans=self.maxnchansperspike, nt=nt)

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

        nchans = len(self.chans) # number of enabled chans
        self.lockouts = np.zeros(nchans, dtype=np.int64) # holds time indices until which each enabled chani is locked out, updated on every found spike
        self.lockouts_us = np.zeros(nchans, dtype=np.int64) # holds times in us until which each enabled chani is locked out, updated only at end of each searchblock call
        self.nspikes = 0 # total num spikes found across all chans so far by this Detector, reset at start of every search
        spikes = np.zeros(0, self.SPIKEDTYPE) # init

        t0 = time.time()
        for wavetrange in wavetranges:
            try:
                blockspikes = self.searchblock(wavetrange, direction)
            except FoundEnoughSpikesError:
                break
            nblockspikes = len(blockspikes)
            shape = list(spikes.shape)
            shape[0] += nblockspikes
            spikes.resize(shape, refcheck=False)
            spikes[self.nspikes:self.nspikes+nblockspikes] = blockspikes
            self.nspikes += nblockspikes
        #spikets = [ s.t for s in spikes ]
        #spikeis = np.argsort(spikets, kind='mergesort') # indices into spikes, ordered by spike time
        #spikes = [ spikes[si] for si in spikeis ] # now guaranteed to be in temporal order
        # default -1 indicates no nid or detid is set as of yet, reserve 0 for actual ids
        spikes['nid'] = -1
        spikes['detid'] = -1
        info('\nfound %d spikes in total' % len(spikes))
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
                           ('t', np.int64), ('t0', np.int64), ('tend', np.int64), ('nt', np.uint8),
                           ('phase1ti', np.uint8), ('phase2ti', np.uint8),
                           ('Vpp', np.float32), ('x0', np.float32), ('y0', np.float32), ('dphase', np.int16)]

    def searchblock(self, wavetrange, direction):
        """Search a block of data, return a list of valid Spikes"""
        #info('searchblock():')
        stream = self.sort.stream
        info('self.nspikes=%d, self.maxnspikes=%d, wavetrange=%s, direction=%d' %
             (self.nspikes, self.maxnspikes, wavetrange, direction))
        if self.nspikes >= self.maxnspikes:
            raise FoundEnoughSpikesError # skip this iteration
        tlo, thi = wavetrange # tlo could be > thi
        bx = self.BLOCKEXCESS
        cutrange = (tlo+bx, thi-bx) # range without the excess, ie time range of spikes to actually keep
        info('wavetrange: %s, cutrange: %s' % (wavetrange, cutrange))
        tslice = time.time()
        wave = stream[tlo:thi:direction] # a block (WaveForm) of multichan data, possibly reversed, ignores out of range data requests, returns up to stream limits
        print('Stream slice took %.3f sec' % (time.time()-tslice))
        # TODO: simplify the whole channel deselection and indexing approach, maybe
        # make all chanis always index into the full probe chan layout instead of the self.chans
        # that represent which chans are enabled for this detector. Also, maybe do away with
        # the whole slicing/indexing into a WaveForm object - just work directly on the .data
        # and .chans and .ts, convenience be damned
        tres = stream.tres
        self.lockouts = np.int64((self.lockouts_us - wave.ts[0]) / tres)
        self.lockouts[self.lockouts < 0] = 0 # don't allow -ve lockout indices
        #info('at start of searchblock:\n new wave.ts[0, end] = %s\n new lockouts = %s' %
        #     ((wave.ts[0], wave.ts[-1]), self.lockouts))

        if self.randomsample:
            maxnspikes = 1 # how many more we're looking for in the next block
        else:
            maxnspikes = self.maxnspikes - self.nspikes

        # want an nchan*2 array of [chani, x/ycoord]
        # TODO: why is this done on every searchblock call? Doesn't take any noticeable amount of time
        xycoords = [ self.enabledSiteLoc[chan] for chan in self.chans ] # (x, y) coords in chan order
        xcoords = np.asarray([ xycoord[0] for xycoord in xycoords ])
        ycoords = np.asarray([ xycoord[1] for xycoord in xycoords ])
        self.siteloc = np.asarray([xcoords, ycoords]).T # index into with chani to get (x, y)
        tget_edges = time.time()
        edgeis = get_edges(wave, self.thresh)
        info('get_edges() took %.3f sec' % (time.time()-tget_edges))
        tcheck_edges = time.time()
        spikes = self.check_edges(wave, edgeis, cutrange)
        info('checking edges took %.3f sec' % (time.time()-tcheck_edges))
        print('found %d spikes' % len(spikes))
        #import cProfile
        #cProfile.runctx('spikes = self.check_edges(wave, edgeis, cutrange)', globals(), locals())
        #spikes = []
        self.lockouts_us = wave.ts[self.lockouts] # lockouts in us, use this to propagate lockouts to next searchblock call
        #info('at end of searchblock:\n lockouts = %s\n new lockouts_us = %s' %
        #     (self.lockouts, self.lockouts_us))
        return spikes

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
            extractXY = sort.extractor.extractXY
        lockouts = self.lockouts
        dti = self.dti
        twi = sort.twi
        nspikes = 0
        spikes = np.zeros(len(edgeis), self.SPIKEDTYPE) # nspikes will always be far less than nedgeis
        # check each edge for validity
        for ti, chani in edgeis: # ti begins life as the threshold xing time index
            chan = self.chans[chani]
            if DEBUG: debug('*** trying thresh event at t=%d chan=%d' % (wave.ts[ti], chan))
            if ti <= lockouts[chani]: # is this thresh crossing timepoint locked out?
                if DEBUG: debug('thresh event is locked out')
                continue # skip to next event

            # find all enabled chanis within nbhd of chani, exclude those locked-out at threshold xing
            nbhdchanis = self.nbhdi[chani]
            chanis = nbhdchanis[lockouts[nbhdchanis] < ti]

            # get data window wrt threshold crossing
            t0i = max(ti+twi[0], lockouts[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            tendi = min(ti+twi[1]+1, len(wave.ts)-1) # +1 makes it end inclusive, don't go further than last wave timepoint
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous

            # do spatiotemporal search for all local extrema in window
            exti = np.tile(False, window.shape)
            for rowi, row in enumerate(window):
                exti[rowi, arglocalextrema(row)] = True

            # sort extrema by decreasing abs(amplitude)
            ext = np.zeros(window.shape, dtype=int)
            ext[exti] = abs(window[exti]) # dereference
            # get indices into flattened ext
            sortis = ext.ravel().argsort() # to get chani and ti of each sort index, reshape to ext.shape
            sortis = sortis[::-1] # reverse for highest to lowest abs(amplitude)
            # find biggest single extremum that isn't locked out
            ncols = window.shape[1]
            for sorti in sortis:
                rowi = sorti // ncols
                coli = sorti % ncols
                chani = chanis[rowi] # NOTE: this overwrites the original value!
                ti = t0i + coli # NOTE: this overwrites the original value!
                if ti > lockouts[chani]: # is this extremum not locked out?
                    break # found the biggest non-locked out extremum
                else:
                    if DEBUG: debug('extremum at chani, ti = %d, %d is locked out' % (chani, ti))
            else:
                if DEBUG: debug('all extrema are locked out')
                continue # skip to next event

            # get window +/- dti+1 around ti on chani, look for the other spike phase
            t0i = max(ti-dti-1, lockouts[chani]+1) # make sure any timepoints included prior to ti aren't locked out
            tendi = min(ti+dti+1, len(wave.ts)-1) # don't go further than last wave timepoint
            window = wave.data[chani, t0i:tendi] # single chan window of data, not necessarily contiguous
            # find spike phases
            tiw = ti - t0i # time index where ti falls wrt the window
            exti = arglocalextrema(window)
            if window[tiw] >= 0: # got a +ve peak, keep only -ve extrema as potential matching peaks
                exti = exti[window[exti] < 0]
            else: # window[tiw] < 0, got a -ve peak, keep only +ve extrema as potential matching peaks
                exti = exti[window[exti] >= 0]
            if len(exti) == 0:
                if DEBUG: debug("couldn't find a matching peak to extremum at "
                                "chani, ti = %d, %d" % (chani, ti))
                continue # skip to next event

            # looks like a spike, now decide if it's biphasic or triphasic
            # split exti up into those before and after the main phase
            preexti  = exti[exti < tiw] # vector
            postexti = exti[exti > tiw] # vector
            # exti will become a scalar denoting the index into window of the spike
            # phase that's the companion to the main phase at tiw
            if len(preexti) == 0 or len(postexti) == 0:
                # it's definitely biphasic
                exti = exti[abs(window[exti]).argmax()]
            else:
                # it's potentially triphasic
                # check if either phase is 2X or more than the other
                preexti = preexti[abs(window[preexti]).argmax()] # now a scalar
                postexti = postexti[abs(window[postexti]).argmax()] # now a scalar
                prepostext = np.abs([window[preexti], window[postexti]])
                smallexti, bigexti = np.argsort(prepostext)
                if prepostext[bigexti] > 2*prepostext[smallexti]:
                    # it's biphasic
                    if bigexti == 0: exti = preexti
                    else: exti = postexti # bigexti == 1
                else: # the two phases on either side of the main one are close in amplitude
                    # it's triphasic
                    if window[tiw] < 0: # main phase is -ve
                        exti = postexti # choose last phase as the companion
                    else: # window[tiw] > 0 # main phase is +ve
                        exti = preexti # choose first phase as the companion
            assert exti != tiw # debugging
            if exti < tiw: # label the phases according to their temporal order
                phase1ti, phase2ti = exti, tiw
            else: # tiw < exti
                phase1ti, phase2ti = tiw, exti
            V1, V2 = window[phase1ti], window[phase2ti]

            # make phase1ti and phase2ti absolute indices into wave
            phase1ti += t0i
            phase2ti += t0i

            # get full-sized window wrt phase1ti
            t0i = max(phase1ti+twi[0], lockouts[chani]+1) # make sure any timepoints included prior to phase1ti aren't locked out
            tendi = min(phase1ti+twi[1]+1, len(wave.ts)-1) # +1 makes it end inclusive, don't go further than last wave timepoint
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous

            # make phase1ti and phase2ti relative to new t0i
            phase1ti -= t0i
            phase2ti -= t0i

            if DEBUG: debug('final window params: t0=%d, tend=%d, phase1t=%d, phase2t=%d, V1=%d, V2=%d'
                            % (wave.ts[t0i], wave.ts[tendi],
                               wave.ts[ti], wave.ts[t0i+phase2ti],
                               V1, V2))
            if abs(V2-V1) < self.ppthresh[chani]:
                if DEBUG: debug("matched peak to extremum at "
                                "chani, ti = %d, %d gives only %d Vpp" % (chani, ti, AD2uV(abs(V2-V1))))
                continue # skip to next event

            ti = t0i + phase1ti # overwrite ti, make it phase1ti wrt 0th time index
            s = spikes[nspikes]
            s['t'] = wave.ts[ti]
            try:
                assert cutrange[0] <= s['t'] <= cutrange[1], 'spike time %r falls outside cutrange for this searchblock call, discarding' % s['t'] # use %r since s['t'] is np.int64 and %d gives TypeError if > 2**31
            except AssertionError, message: # doesn't qualify as a spike, don't change lockouts
                if DEBUG: debug(message)
                continue # skip to next event
            # leave each spike's chanis in sorted order, as they are in self.nbhdi, important
            # assumption used later on, like in sort.get_wave() and Neuron.update_wave()
            ts = wave.ts[t0i:tendi]
            # use ts = np.arange(s['t0'], s['tend'], stream.tres) to reconstruct
            s['t0'], s['tend'] = wave.ts[t0i], wave.ts[tendi]
            s['phase1ti'], s['phase2ti'] = phase1ti, phase2ti # wrt t0i
            s['dphase'] = ts[phase2ti] - ts[phase1ti] # in us
            #s.V1, s.V2 = V1, V2
            # maintain polarity, first convert from int16 to float to prevent overflow
            s['Vpp'] = AD2uV(np.float32(V2) - np.float32(V1))

            # update channel neighbourhood
            # find all enabled chanis within nbhd of chani, exclude those locked-out at threshold xing
            nbhdchanis = self.nbhdi[chani]
            chanis = nbhdchanis[lockouts[nbhdchanis] < ti]
            chan = self.chans[chani]
            chans = self.chans[chanis]
            nchans = len(chans)
            s['chan'], s['chans'][:nchans], s['nchans'] = chan, chans, nchans
            # get new window corresponding to new chans
            window = wave.data[chanis, t0i:tendi] # multichan window of data, not necessarily contiguous
            if self.keepspikewavesondetect: # keep spike waveform for later use
                totalnspikes = sort.nspikes + self.nspikes + nspikes
                sort.set_wavedata(totalnspikes, window, phase1ti)
            if self.extractparamsondetect:
                # just x and y params for now
                x = self.siteloc[chanis, 0] # 1D array (row)
                y = self.siteloc[chanis, 1]
                maxchani = int(np.where(chans == chan))
                s['x0'], s['y0'] = extractXY(window, x, y, phase1ti, phase2ti, maxchani)

            if DEBUG: debug('*** found new spike: %d @ (%d, %d)' % (s['t'], self.siteloc[chani, 0], self.siteloc[chani, 1]))

            # update lockouts to 2nd phase of this spike
            #dphaseti = phase2ti - phase1ti
            lockout = t0i + phase2ti #+ dphaseti / 2
            lockouts[chanis] = lockout # same for all chans in this spike
            if DEBUG:
                lockoutt = wave.ts[lockout]
                #lockoutt = wave.ts[max(lockout, len(wave.ts)-1)] # stay inbounds
                debug('lockout = %d for chans = %s' % (lockoutt, chans))

            nspikes += 1

        spikes.resize(nspikes, refcheck=False) # shrink down to actual needed size
        return spikes

    def find_spike_phases(self, window, tiw, ppthresh, reftype='trigger'):
        """Find spike phases within window of data: search from tiw in direction
        (which might be, say, a threshold xing point) for 1st peak, then within
        self.dti of that for a 2nd peak of opposite phase. Decide which peak
        comes first, return window indices of 1st and 2nd spike phases. reftype
        describes what tiw represents: a 'trigger' point or previously found
        spike 'phase'"""
        exti = arglocalextrema(window) # indices of local extrema, wrt window
        if len(exti) == 0:
            raise NoPeakError("can't find any extrema within window")
        if reftype == 'trigger':
            dir1 = 'right'
            dir2 = 'both'
        elif reftype == 'phase':
            dir1 = 'nearest'
            dir2 = 'right' # assume the tiw given for reftype='phase' is for the 1st phase
        else:
            raise ValueError('unknown reftype %r' % reftype)
        '''
        if dir1 == 'left':
            try:
                peak1i = exti[(exti <= tiw)][-1] # index of first extremum left of tiw, wrt window
            except IndexError:
                raise NoPeakError("can't find 1st peak within window")'''
        if dir1 == 'right':
            try:
                #peak1i = exti[(exti >= tiw)][0] # index of first extremum right of tiw, wrt window
                rightexti = exti[(exti >= tiw)]
                peak1i = exti[abs(window[rightexti]).argmax()]
            except IndexError:
                raise NoPeakError("can't find 1st peak within window")
        else: # dir1 == 'nearest'
            # find not just the nearest peak, but the nearest of the
            # same phase as at tiw (on the previous maxchan)
            if window[tiw] < 0:
                samephaseexti = exti[window[exti] < 0]
            else: # window[tiw] >= 0:
                samephaseexti = exti[window[exti] >= 0]
            if len(samephaseexti) == 0:
                raise NoPeakError("can't find 1st peak of matching phase within window")
            peak1i = samephaseexti[abs(samephaseexti - tiw).argmin()]
        peak1i = int(peak1i)
        #if window[peak1i] < 0: # peak1i is -ve, look right for corresponding +ve peak
        #    dir2 = 'right'
        #else: # peak1i is +ve, look left for corresponding -ve peak
        #    dir2 = 'left'
        # find biggest 2nd extremum of opposite sign in dir2 within self.dti
        peak2i = arg2ndpeak(window, exti, peak1i, dir2, self.dti, ppthresh)
        # check which comes first
        if peak1i < peak2i:
            return peak1i, peak2i
        else: # peak2i < peak1i
            return peak2i, peak1i

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
        if self.threshmethod == 'GlobalFixed': # all chans have the same fixed thresh
            fixedthresh = self.sort.converter.uV2AD(self.fixedthresh) # convert to AD units
            thresh = np.tile(fixedthresh, len(self.chans))
        elif self.threshmethod == 'ChanFixed': # each chan has its own fixed thresh
            """randomly sample self.fixednoisewin's worth of data from self.trange in
            blocks of self.blocksize. NOTE: this samples with replacement, so it's
            possible, though unlikely, that some parts of the data will contribute
            more than once to the noise calculation
            """
            print('loading data to calculate noise')
            if self.fixednoisewin >= abs(self.trange[1] - self.trange[0]): # sample width exceeds search trange
                wavetranges = [self.trange] # use a single block of data, as defined by trange
            else:
                nblocks = int(round(self.fixednoisewin / self.blocksize))
                wavetranges = RandomWaveTranges(self.trange, bs=self.blocksize, bx=0, maxntranges=nblocks)
            data = []
            for wavetrange in wavetranges:
                wave = self.sort.stream[wavetrange[0]:wavetrange[1]]
                wave = wave[self.chans] # keep just the enabled chans
                data.append(wave.data)
            data = np.concatenate(data, axis=1) # int16 AD units
            noise = self.get_noise(data) # float AD units
            thresh = noise * self.noisemult # float AD units
            thresh = np.int16(np.round(thresh)) # int16 AD units
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
        if self.noisemethod == 'median':
            return np.median(np.abs(data), axis=-1) / 0.6745 # see Quiroga2004
        elif self.noisemethod == 'stdev':
            return np.stdev(data, axis=-1)
        else:
            raise ValueError


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
