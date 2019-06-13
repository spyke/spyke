"""Stream objects for requesting arbitrary time ranges of data from various file types
in a common way.

To test streams thoroughly:

* open up all 3 kinds of windows (spike, chart, LFP) and scroll around
* scroll to the very start and end of a stream. Do the start and end timepoint
  buttons correspond to the current displayed time position when at start and end of stream
  respectively?
* try zooming in and out in time on all the windows
* change between all the different settings for each of the pre-processing options:
  Filtering, CAR and Sampling
* export raw data to .dat, then open the .dat and compare with original using two instances
  of spyke. They should be identical, down to the last pixel. Compare them while doing all
  of the above
* also export high and low-pass data to .dat and compare with original

First do the above for single Streams (DATStream, NSXStream, SurfStream, SimpleStream),
then repeat for MultiStreams of all the above types. For simplicity, work on a
MultiStream that is made up of only two short single Streams.

Additional MultiStream tests:

* compare the original single Streams with their appropriate spots in the MultiStream. Use
  the MultiStream's .tranges array to tell you where to scroll to in the MultiStream to
  find the equivalent start and end positions of the all of its constituent Streams.
  Do this while varying the pre-processing settings.
* in addition to exporting raw data to .dat and comparing to the MultiStream, also compare to
  the constituent Streams

"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek']

import os
import time
from datetime import timedelta
from collections import OrderedDict as odict

import numpy as np

from . import core, probes
from .core import (WaveForm, EmptyClass, intround, lrstrip,
                   hamming, filterord, WMLDR, td2fusec)
from .core import (DEFHPRESAMPLEX, DEFLPSAMPLFREQ, DEFHPSRFSHCORRECT,
                   DEFHPDATSHCORRECT, DEFDATFILTMETH, DEFHPNSXSHCORRECT, DEFNSXFILTMETH,
                   DEFCAR, BWHPF0, BWLPF1, BWHPORDER, BWLPORDER, LOWPASSFILTERLPSTREAM,
                   SRFNCHANSPERBOARD, KERNELSIZE, XSWIDEBANDPOINTS, DATSAMPLEERRPCT)


class FakeStream(object):
    def __init__(self):
        self.fname = ''

    def close(self):
        pass


class Stream(object):
    """Base class for all (single) streams"""
    def is_multi(self):
        """Convenience method to specify if self is a MultiStream"""
        return False

    def is_open(self):
        return self.f.is_open()

    def open(self):
        self.f.open()

    def close(self):
        self.f.close()

    def get_dt(self):
        """Get self's duration"""
        return self.t1 - self.t0

    dt = property(get_dt)

    def get_fname(self):
        return self.f.fname

    fname = property(get_fname)

    def get_fnames(self):
        return [self.fname]

    fnames = property(get_fnames)

    def get_ext(self):
        """Get file extension of data source"""
        return os.path.splitext(self.fnames[0])[-1] # take extension of first fname

    ext = property(get_ext)

    def get_srcfnameroot(self):
        """Get root of filename of source data"""
        srcfnameroot = lrstrip(self.fname, '../', self.ext)
        return srcfnameroot

    srcfnameroot = property(get_srcfnameroot)

    def get_nchans(self):
        return len(self.chans)

    nchans = property(get_nchans)

    def get_sampfreq(self):
        return self._sampfreq

    def set_sampfreq(self, sampfreq):
        """On .sampfreq change, delete .kernels (if set), and update .tres"""
        self._sampfreq = sampfreq
        try:
            del self.kernels
        except AttributeError:
            pass
        self.tres = 1 / self.sampfreq * 1e6 # float us
        #print('Stream.tres = %g' % self.tres)

    sampfreq = property(get_sampfreq, set_sampfreq)

    def get_shcorrect(self):
        return self._shcorrect

    def set_shcorrect(self, shcorrect):
        """On .shcorrect change, delete .kernels (if set)"""
        if shcorrect == True and self.masterclockfreq == None:
            raise ValueError("Can't sample & hold correct data stream with no master "
                             "clock frequency")
        self._shcorrect = shcorrect
        try:
            del self.kernels
        except AttributeError:
            pass

    shcorrect = property(get_shcorrect, set_shcorrect)

    def get_israw(self):
        """Return if self is currently set to return raw data"""
        return (not self.filtmeth and
                not self.car and
                self.sampfreq == self.rawsampfreq and
                not self.shcorrect)

    israw = property(get_israw)

    def get_datetime(self):
        return self.f.datetime

    datetime = property(get_datetime)

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us wrt t=0. Return the corresponding WaveForm
        object with the full set of chans"""
        if key.step not in [None, 1]:
            raise ValueError('Unsupported slice step size: %s' % key.step)
        return self(key.start, key.stop, self.chans)

    def resample(self, rawdata, rawts, chans):
        """Return potentially sample-and-hold corrected and Nyquist interpolated
        data and timepoints. See Blanche & Swindale, 2006"""
        #print('sampfreq, rawsampfreq, shcorrect = (%r, %r, %r)' %
        #      (self.sampfreq, self.rawsampfreq, self.shcorrect))
        rawtres = self.rawtres # float us
        tres = self.tres # float us
        if self.sampfreq % self.rawsampfreq != 0:
            raise ValueError('Only integer multiples of rawsampfreq allowed for interpolated '
                             'sampfreq')
        # resample factor: n output resampled points per input raw point:
        resamplex = intround(self.sampfreq / self.rawsampfreq)
        assert resamplex >= 1, 'No decimation allowed'
        N = KERNELSIZE
        #print('N = %d' % N)

        # generate kernels if necessary:
        try:
            self.kernels
        except AttributeError:
            ADchans = None
            if self.shcorrect:
                ADchans = self.layout.ADchanlist
            self.kernels = self.get_kernels(resamplex, N, chans, ADchans=ADchans)

        # convolve the data with each kernel
        nrawts = len(rawts)
        nchans = len(chans)
        # all the interpolated points have to fit in between the existing raw
        # points, so there are nrawts - 1 interpolated points:
        #nt = nrawts + (resamplex - 1) * (nrawts - 1)
        # the above can be simplified to:
        nt = nrawts*resamplex - resamplex + 1
        tstart = rawts[0]
        # generate interpolated timepoints, use intfloor in case tres is a float, otherwise
        # arange might give one too many timepoints:
        #ts = np.arange(tstart, intfloor(tstart+tres*nt), tres)
        # safer to use linspace than arange in case of float tres, deals with endpoints
        # better and gives slightly more accurate output float timestamps:
        ts = np.linspace(tstart, tstart+(nt-1)*tres, nt) # end inclusive
        assert len(ts) == nt
        # resampled data, leave as int32 for convolution, then convert to int16:
        data = np.empty((nchans, nt), dtype=np.int32)
        #print('data.shape = %r' % (data.shape,))
        #tconvolve = time.time()
        #tconvolvesum = 0
        # Only the chans that are actually needed are resampled and returned.
        # Assume that chans index into ADchans. Normally they should map 1 to 1, ie chan 0
        # taps off of ADchan 0, but for probes like pt16a_HS27 and pt16b_HS27, it seems
        # ADchans start at 4.
        for chani, chan in enumerate(chans):
            try:
                kernels = self.kernels[chan]
            except KeyError: # all channels have the same kernels
                kernels = self.kernels[None]
            for point, kernel in enumerate(kernels):
                """np.convolve(a, v, mode)
                for mode='same', only the K middle values are returned starting at n = (M-1)/2
                where K = len(a)-1 and M = len(v) - 1 and K >= M
                for mode='valid', you get the middle len(a) - len(v) + 1 number of values"""
                #tconvolveonce = time.time()
                row = np.convolve(rawdata[chani], kernel, mode='same')
                #tconvolvesum += (time.time()-tconvolveonce)
                #print('len(rawdata[chani]) = %r' % len(rawdata[chani]))
                #print('len(kernel) = %r' % len(kernel))
                #print('len(row): %r' % len(row))
                # interleave by assigning from point to end in steps of resamplex
                # index to start filling data from for this kernel's points:
                ti0 = (resamplex - point) % resamplex
                # index of first data point to use from convolution result 'row':
                rowti0 = int(point > 0)
                # discard the first data point from interpolant's convolutions, but not for
                # raw data's convolutions, since interpolated values have to be bounded on both
                # sides by raw values?
                data[chani, ti0::resamplex] = row[rowti0:]
        #print('Convolve loop took %.3f sec' % (time.time()-tconvolve))
        #print('Convolve calls took %.3f sec total' % (tconvolvesum))
        #tundoscaling = time.time()
        # undo kernel scaling, shift 16 bits right in place, same as //= 2**16, leave as int32
        data >>= 16
        #print('Undo kernel scaling took %.3f sec total' % (time.time()-tundoscaling))
        return data, ts

    def get_kernels(self, resamplex, N, chans, ADchans=None):
        """Return dict of kernels, one per channel in chans, to convolve with raw data to get
        interpolated signal. Return a potentially different kernel for each ADchan to correct
        each ADchan's s+h delay. chans and ADchans may not always be contiguous from 0, but are
        assumed to always be in corresponding order.

        TODO: when resamplex > 1 and shcorrect == False, you only need resamplex - 1 kernels.
        You don't need a kernel for the original raw data points. Those won't be shifted,
        so you can just interleave appropriately.

        TODO: take DIN channel into account, might need to shift all highpass ADchans
        by 1us, see line 2412 in SurfBawdMain.pas. I think the layout.sh_delay_offset field
        may tell you if and by how much you should take this into account

        WARNING! TODO: not sure if say ADchan 4 will always have a delay of 4us, or only if
        it's preceded by AD chans 0, 1, 2 and 3 in the channel gain list - I suspect the latter
        is the case, but right now I'm coding the former. Note that there's a
        f.layout.sh_delay_offset field that describes the sh delay for first chan of probe.
        Should probably take this into account, although it doesn't affect relative delays
        between chans, I think. I think it's usually 1us.
        """
        if ADchans is None: # no per-channel delay:
            assert self.shcorrect == False
            chans = [None] # use same kernels for all channels
            dis = np.array([0])
        else:
            assert self.shcorrect == True
            # ordinal position of each ADchan in the hold queue of its ADC board:
            i = ADchans % SRFNCHANSPERBOARD
            ## TODO: stop hard-coding 1 masterclockfreq tick delay per ordinal position
            # per channel delays, us, usually 1 us/chan:
            dis = 1000000 / self.masterclockfreq * i
        ds = dis / self.rawtres # normalized per channel delays, float us
        wh = hamming # window function
        h = np.sinc # sin(pi*t) / pi*t
        kernels = {} # dict of array of kernels, indexed by [chan][resample point]
        for chan, d in zip(chans, ds): # chans and corresponding delays
            kernelrow = []
            for point in range(resamplex): # iterate over resampled points per raw point
                t0 = point/resamplex # some fraction of 1
                tstart = -N/2 - t0 - d
                tend = tstart + (N+1)
                # kernel sample timepoints, all of length N+1, float32s to match voltage
                # data type
                t = np.arange(tstart, tend, 1, dtype=np.float32)
                kernel = wh(t, N) * h(t) # windowed sinc, sums to 1.0, max val is 1.0
                # rescale to get values up to 2**16, convert to int32
                kernel = np.int32(np.round(kernel * 2**16))
                kernelrow.append(kernel)
            kernels[chan] = kernelrow
        return kernels

    def get_block_tranges(self, bs=10000000):
        """Get time ranges spanning self, in block sizes of bs us"""
        tranges = []
        for start in np.arange(self.t0, self.t1, bs):
            stop = start + bs
            tranges.append((start, stop))
        tranges = np.asarray(tranges)
        # don't exceed self.t1:
        tranges[-1, -1] = self.t1
        return tranges

    def get_block_data(self, bs=10000000, step=None, chans=None, units='uV'):
        """Get blocks of data in block sizes of bs us, keeping every step'th data point
        (default keeps all), on specified chans, in units"""
        tranges = self.get_block_tranges(bs=bs)
        data = []
        for (start, stop) in tranges:
            blockwave = self(start, stop, chans=chans)
            data.append(blockwave.data[::step]) # decimate
        data = np.concatenate(data, axis=1) # concatenate horizontally
        if units is None:
            return data
        elif units == 'uV':
            return self.converter.AD2uV(data)
        else:
            raise ValueError("Unknown units %r" % units)


class DATStream(Stream):
    """Stream interface for .dat files"""
    def __init__(self, f, kind='highpass', filtmeth=None, car=None, sampfreq=None,
                 shcorrect=None):
        self.f = f
        self.kind = kind
        if kind not in ['highpass', 'lowpass']:
            raise ValueError('Unknown stream kind %r' % kind)

        self.filtmeth = filtmeth or DEFDATFILTMETH
        self.car = car or DEFCAR

        self.converter = core.DatConverter(f.fileheader.AD2uVx)

        self.probe = f.fileheader.probe
        self.adapter = f.fileheader.adapter

        self.rawsampfreq = f.fileheader.sampfreq # Hz
        self.rawtres = 1 / self.rawsampfreq * 1e6 # float us

        if kind == 'highpass':
            self.sampfreq = sampfreq or DEFHPRESAMPLEX * self.rawsampfreq
            self.shcorrect = shcorrect or DEFHPDATSHCORRECT
        else: # kind == 'lowpass'
            self.sampfreq = sampfreq or DEFLPSAMPLFREQ
            # make sure that after low-pass filtering the raw data, we can easily decimate
            # the result to get the desired lowpass sampfreq:
            assert self.rawsampfreq % self.sampfreq == 0
            self.shcorrect = shcorrect or False
            # for simplicity, don't allow s+h correction of lowpass data, no reason to anyway:
            assert self.shcorrect == False

        # always keep probe chans sorted, even if they aren't sorted in the data file
        # (such as when using an adapter). Actual data order is handled by self.__call__
        self.chans = np.sort(f.fileheader.chans)

        self.contiguous = f.contiguous

        self.t0, self.t1 = f.t0, f.t1
        self.tranges = np.asarray([[self.t0, self.t1]])

    def get_filtering(self):
        """Get filtering settings in an odict, based on self.kind and self.filtmeth"""
        if not self.filtmeth:
            return None # nothing to report, don't even bother with odict
        od = odict()
        od['meth'] = self.filtmeth
        if self.kind == 'highpass':
            if self.filtmeth.startswith('BW'):
                od['f0'] = BWHPF0
                od['f1'] = None
                od['order'] = BWHPORDER
        elif self.kind == 'lowpass':
            if self.filtmeth.startswith('BW'):
                od['f0'] = None
                od['f1'] = BWLPF1
                od['order'] = BWLPORDER
        else:
            raise ValueError
        return od

    filtering = property(get_filtering)

    def __call__(self, start=None, stop=None, chans=None, checksat=False):
        """Called when Stream object is called using (). start and stop are timepoints in us
        wrt t=0. Returns the corresponding WaveForm object with just the specified chans.
        If checksat, also returns boolean array of data points of the same shape designating
        which data points reached saturation in the raw data.

        As of 2017-10-24 I'm not sure if this behaviour qualifies as end-inclusive or not,
        but I suspect not. See how these single Streams are called in MultiStream.__call__
        """
        if start is None:
            start = self.t0
        if stop is None:
            stop = self.t1
        if chans is None:
            chans = self.chans
        kind = self.kind
        try:
            # where to find requested chans in enabled self.chans:
            chanis = core.argmatch(self.chans, chans)
        except ValueError:
            raise IndexError("Requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, kind))
        # NOTE: because CAR needs to average across as many channels as possible, keep
        # the full self.chans (which are the chans enabled in the stream) until the very end,
        # and only then slice out what is potentially a subset of self.chans using `chans`
        tres, rawtres = self.tres, self.rawtres # float us

        # Only allow start and stop time requests that deviate up to DATSAMPLEERRPCT from the
        # nearest actual sample timepoint. This avoids roundoff errors for time requests that
        # fall exactly in between (50%) sample timepoints:
        # convert to multiples of sample timepoints:
        floatstarti = start / tres
        floatstopi = stop / tres
        starterrpct = abs(intround(floatstarti) - floatstarti) * 100
        stoperrpct = abs(intround(floatstopi) - floatstopi) * 100
        if starterrpct > DATSAMPLEERRPCT:
            raise ValueError("Requested start time %g us falls too far (%g%%) between "
                             "timepoints at %g us resolution" % (start, starterrpct, tres))
        if stoperrpct > DATSAMPLEERRPCT:
            raise ValueError("Requested stop time %g us falls too far (%g%%) between "
                             "timepoints at %g us resolution" % (stop, stoperrpct, tres))

        # mintres handles both high-pass data where tres < rawtres, and low-pass decimated
        # data where tres > rawtres:
        mintres = min(tres, rawtres)
        if kind == 'highpass':
            resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
            decimate = False
        elif kind == 'lowpass':
            resample = False # which also means no s+h correction allowed
            decimate = True
            assert self.rawsampfreq % self.sampfreq == 0
            decimatex = intround(self.rawsampfreq / self.sampfreq)
        else:
            raise ValueError('Unknown stream kind %r' % kind)

        # excess data to get at either end, to eliminate filtering and interpolation
        # edge effects:
        #print('XSWIDEBANDPOINTS: %d' % XSWIDEBANDPOINTS)
        xs = XSWIDEBANDPOINTS * rawtres # us
        #print('xs: %d, rawtres: %g' % (xs, rawtres))

        nt = self.f.nt
        # stream limits in sample indices, wrt sample=0:
        t0i, t1i = self.f.t0i, self.f.t1i
        # calculate *slice* indices t0xsi and t1xsi, for a greater range of
        # raw data (with xs) than requested:
        t0xsi = intround((start - xs) / rawtres) # round to nearest mult of rawtres
        t1xsi = intround((stop + xs) / rawtres) # round to nearest mult of rawtres
        # stay within stream *slice* limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, t0i)
        t1xsi = min(t1xsi, t1i+1)
        # convert slice indices back to nearest float us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        # these are slice indices, so don't add 1 when calculating ntxs:
        ntxs = t1xsi - t0xsi # int
        tsxs = np.linspace(t0xs, t0xs+(ntxs-1)*rawtres, ntxs) # end inclusive
        #print('t0xs, t1xs, ntxs: %f, %f, %d' % (t0xs, t1xs, ntxs))

        # Init dataxs, sized to hold all enabled channels.
        # Unlike for .srf files, int32 dataxs array isn't necessary for
        # int16 .dat or .nsx files, since there's no need to zero or rescale
        dataxs = np.zeros((self.nchans, ntxs), dtype=np.int16) # any gaps will have zeros

        '''
        Load up data+excess. The same raw data is used for high and low pass streams,
        the only difference being the subsequent filtering. It would be convenient to
        immediately subsample to get lowpass, but that would result in aliasing. We can
        only subsample after filtering.
        '''
        #tload = time.time()
        # source slice indices:
        st0i = max(t0xsi - t0i, 0)
        st1i = min(t1xsi - t0i, nt)
        assert st1i-st0i == ntxs
        # destination slice indices:
        dt0i = max(t0i - t0xsi, 0)
        dt1i = min(t1i + 1 - t0xsi, ntxs)
        allchanis = core.argmatch(self.f.fileheader.chans, self.chans)
        dataxs[:, dt0i:dt1i] = self.f.data[allchanis, st0i:st1i]
        #print('data load took %.3f sec' % (time.time()-tload))

        satis = None
        if checksat:
            # also return boolean indices of saturated rawdata points:
            tsxsi = intround(tsxs / rawtres)
            starti, stopi = intround(start / rawtres), intround(stop / rawtres)
            lo, hi = tsxsi.searchsorted([starti, stopi])
            rawdata = dataxs[:, lo:hi]
            dtinf = np.iinfo(rawdata.dtype)
            minval, maxval = dtinf.min, dtinf.max
            satis = (rawdata == minval) | (rawdata == maxval) # saturation bool indices

        #print('filtmeth: %s' % self.filtmeth)
        if self.filtmeth == None:
            pass
        elif self.filtmeth in ['BW', 'BWNC']:
            # High- or low-pass filter the raw data using a Butterworth filter.
            # Default to causal filtering to get high-pass data with added phase lag,
            # in order to preserve expected spike shape for spike detection. Use noncausal
            # (forward-backward) filtering to get low-pass LFP data without phase lag:
            if self.filtmeth == 'BW':
                hpcausal = True
            else: # self.filtmeth == 'BWNC'
                hpcausal = False
            f = self.filtering
            if kind == 'highpass':
                btype, order, f0, f1 = kind, f['order'], f['f0'], f['f1']
                dataxs, b, a = filterord(dataxs, sampfreq=self.rawsampfreq, f0=f0, f1=f1,
                                         order=order, rp=None, rs=None, btype=btype,
                                         ftype='butter', causal=hpcausal) # float64
            else: # kind == 'lowpass'
                if LOWPASSFILTERLPSTREAM:
                    btype, order, f0, f1 = kind, f['order'], f['f0'], f['f1']
                    dataxs, b, a = filterord(dataxs, sampfreq=self.rawsampfreq, f0=f0, f1=f1,
                                             order=order, rp=None, rs=None, btype=btype,
                                             ftype='butter', causal=False) # float64
        elif self.filtmeth == 'WMLDR':
            # high pass filter using wavelet multi-level decomposition and reconstruction,
            # can't directly use this for low pass filtering, but it might be possible to
            # approximate low pass filtering with WMLDR by subtracting the high pass data
            # from the raw data...:
            assert kind == 'highpass' # for now
            ## NOTE: WMLDR seems to leave behind some very low frequencies, resulting
            ## in weird slow wobbling of amplitude as a function of exactly what
            ## the WMLDR filtering time range happens to be. Setting a much bigger xs
            ## helps, but only until you move xs amount of time away from the start of
            ## the recording. I think WMLDR doesn't quite remove all the low freqs the way
            ## Butterworth filtering does. Applying a Butterworth filter non-causally
            ## results in exactly the same very low level of spike shape distortion as does
            ## WMLDR, but without leaving behind the very low frequency component, so there's
            ## really no reason to use WMLDR, unless you need online filtering with very
            ## low spike distortion
            dataxs = WMLDR(dataxs)
        else:
            raise ValueError('Unknown filter method %s' % self.filtmeth)

        # do common average reference (CAR): remove correlated noise by subtracting the
        # average across all channels (Ludwig et al, 2009, Pachitariu et al, 2016):
        if self.car and kind == 'highpass': # for now, only apply CAR to highpass stream
            if self.car == 'Median':
                avg = np.median
            elif self.car == 'Mean':
                avg = np.mean
            else:
                raise ValueError('Unknown CAR method %r' % self.car)
            # at each timepoint, find average across all chans:
            car = avg(dataxs, axis=0) # float64
            # at each timepoint, subtract average across all chans, don't do in place
            # because dataxs might be int16 or float64, depending on filtering:
            dataxs = dataxs - car

        # do any resampling if necessary:
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs, self.chans)
            #print('resample took %.3f sec' % (time.time()-tresample))

        nresampletxs = len(tsxs)
        #print('Stream ntxs, nresampletxs: %d, %d' % (ntxs, nresampletxs))
        #assert ntxs == len(tsxs)

        # Trim down to just the requested time range and chans, and optionally decimate.
        # For trimming time range, use integer multiple of mintres to prevent floating point
        # round-off error (when mintres is non-integer us) that can occasionally
        # cause searchsorted to produce off-by-one indices:
        tsxsi = intround(tsxs / mintres)
        starti, stopi = intround(start/mintres), intround(stop/mintres)
        lo, hi = tsxsi.searchsorted([starti, stopi])

        # Slice out chanis here only at the very end, because we want to use all
        # enabled chans up to this point for CAR, even those that we ultimately don't
        # need to return, because any extra chans that are enabled but aren't requested
        # will nevertheless affect the mean/median:
        if decimate:
            data = dataxs[chanis, lo:hi:decimatex]
            ts = tsxs[lo:hi:decimatex]
        else:
            data = dataxs[chanis, lo:hi]
            ts = tsxs[lo:hi]

        #print('Stream start, stop, tres, shape:\n', start, stop, self.tres, data.shape)
        # should be safe to convert back down to int16 now:
        data = np.int16(data)
        if checksat:
            assert data.shape == satis.shape
        return WaveForm(data=data, ts=ts, chans=chans, tres=tres, satis=satis)


class NSXStream(DATStream):
    """Stream interface for .nsx files"""
    def __init__(self, f, kind='highpass', filtmeth=None, car=None, sampfreq=None,
                 shcorrect=None):
        self.f = f
        self.kind = kind
        if kind not in ['highpass', 'lowpass']:
            raise ValueError('Unknown stream kind %r' % kind)

        self.filtmeth = filtmeth or DEFNSXFILTMETH
        self.car = car or DEFCAR

        self.converter = core.NSXConverter(f.fileheader.AD2uVx)

        self.probe = f.fileheader.probe
        self.adapter = f.fileheader.adapter

        self.rawsampfreq = f.fileheader.sampfreq # Hz
        self.rawtres = 1 / self.rawsampfreq * 1e6 # float us

        if kind == 'highpass':
            self.sampfreq = sampfreq or DEFHPRESAMPLEX * self.rawsampfreq
            self.shcorrect = shcorrect or DEFHPNSXSHCORRECT
        else: # kind == 'lowpass'
            self.sampfreq = sampfreq or DEFLPSAMPLFREQ
            # make sure that after low-pass filtering the raw data, we can easily decimate
            # the result to get the desired lowpass sampfreq:
            assert self.rawsampfreq % self.sampfreq == 0
            self.shcorrect = shcorrect or False
            # for simplicity, don't allow s+h correction of lowpass data, no reason to anyway:
            assert self.shcorrect == False

        # no need for shcorrect for .nsx because the Blackrock Cerebrus NSP system has a
        # 1 GHz multiplexer for every bank of 32 channels, according to Kian Torab
        # <support@blackrock.com>
        assert self.shcorrect == False

        # always keep chans sorted, even if they aren't in the underlying data in the file
        # (such as when using an adapter). Actual data order is handled by self.__call__
        self.chans = np.sort(f.fileheader.chans)

        self.contiguous = f.contiguous

        self.t0, self.t1 = f.t0, f.t1
        self.tranges = np.asarray([[self.t0, self.t1]])


class SurfStream(Stream):
    """Stream interface for .srf files. Maps from timestamps to record index
    of stream data to retrieve the approriate range of waveform data from disk"""
    def __init__(self, f, kind='highpass', filtmeth=None, car=None, sampfreq=None,
                 shcorrect=None):
        """Takes a sorted temporal (not necessarily evenly-spaced, due to pauses in recording)
        sequence of ContinuousRecords: either HighPassRecords or LowPassMultiChanRecords.
        sampfreq arg is useful for interpolation. Assumes that all HighPassRecords belong
        to the same probe. f must be open and parsed"""
        self.f = f
        self.kind = kind
        if kind == 'highpass':
            self.records = f.highpassrecords
        elif kind == 'lowpass':
            self.records = f.lowpassmultichanrecords
        else: raise ValueError('Unknown stream kind %r' % kind)

        self.filtmeth = filtmeth
        self.car = car

        # assume same layout for all records of type "kind"
        self.layout = self.f.layoutrecords[self.records['Probe'][0]]
        intgain = self.layout.intgain
        extgain = int(self.layout.extgain[0]) # assume same extgain for all chans in layout
        self.converter = core.Converter(intgain, extgain)
        self.nADchans = self.layout.nchans # always constant
        self.rawsampfreq = self.layout.sampfreqperchan
        self.rawtres = 1 / self.rawsampfreq * 1e6 # float us
        if kind == 'highpass':
            if list(self.layout.ADchanlist) != range(self.nADchans):
                print("WARNING: ADchans aren't contiguous from 0, highpass recordings are "
                      "nonstandard. Sample and hold delay correction in self.resample() "
                      "may not be exactly correct")
            # probe chans, as opposed to AD chans. Most probe types are contiguous from 0,
            # but there are some exceptions (like pt16a_HS27 and pt16b_HS27):
            self.chans = np.arange(self.nADchans)
            self.sampfreq = sampfreq or DEFHPRESAMPLEX * self.rawsampfreq
            self.shcorrect = shcorrect or DEFHPSRFSHCORRECT
        else: # kind == 'lowpass'
            # probe chan values are already parsed from LFP probe description
            self.chans = self.layout.chans
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default

        probename = self.layout.electrode_name
        self.probe = probes.getprobe(probename)
        self.adapter = None

        rts = self.records['TimeStamp'] # array of record timestamps
        NumSamples = np.unique(self.records['NumSamples'])
        if len(NumSamples) > 1:
            raise RuntimeError("Not all continuous records are of the same length. "
                               "NumSamples = %r" % NumSamples)
        rtlen = NumSamples / self.nADchans * self.rawtres # record time length, float us
        # Check whether rts values are all equally spaced, indicating there were no
        # pauses in recording
        diffrts = np.diff(rts)
        self.contiguous = (np.diff(diffrts) == 0).all() # could also call diff(rts, n=2)
        if self.contiguous:
            try: assert np.unique(diffrts) == rtlen
            except AssertionError: import pdb; pdb.set_trace()
            self.tranges = np.int64([[rts[0], rts[-1]+rtlen]]) # int us, keep it 2D
        else:
            if kind == 'highpass': # don't bother reporting again for lowpass
                print('NOTE: time gaps exist in %s, possibly due to pauses' % self.fname)
            # build up self.tranges
            splitis = np.where(diffrts != rtlen)[0] + 1
            splits = np.split(rts, splitis) # list of arrays of contiguous rts
            tranges = []
            for split in splits: # for each array of contiguous rts
                tranges.append([split[0], split[-1]+rtlen]) # float us
            self.tranges = np.int64(tranges) # int us, 2D
        self.t0 = self.tranges[0, 0] # int us
        self.t1 = self.tranges[-1, 1] # int us

    def get_fname(self):
        try:
            return self.f.fname
        except AttributeError:
            return self.srff.fname # for SurfStream in older sorts

    fname = property(get_fname)

    def get_srcfnameroot(self):
        """Get root of filename of source data. Also filter it to make recording
        names from older .srf files more succint"""
        srcfnameroot = lrstrip(self.fname, '../', '.srf')
        srcfnameroot = srcfnameroot.replace(' - track 5 ', '-tr5-')
        srcfnameroot = srcfnameroot.replace(' - track 6 ', '-tr6-')
        srcfnameroot = srcfnameroot.replace(' - track 7c ', '-tr7c-')
        # replace any remaining spaces with underscores
        srcfnameroot = srcfnameroot.replace(' ', '_')
        return srcfnameroot

    srcfnameroot = property(get_srcfnameroot)

    def get_masterclockfreq(self):
        return self.f.layoutrecords[0].MasterClockFreq

    masterclockfreq = property(get_masterclockfreq)

    def get_filtering(self):
        """Get filtering settings. For .srf files, these are fixed hardware settings"""
        od = odict()
        if self.kind == 'highpass':
            od['meth'] = 'high-pass hardware analog filter'
            od['f0'], od['f1'] = 500, 6000 # Hz
        elif self.kind == 'lowpass':
            od['meth'] = 'low-pass hardware analog filter'
            od['f0'], od['f1'] = 0.1, 150 # Hz
        else:
            raise ValueError
        return od

    filtering = property(get_filtering)

    def pickle(self):
        self.f.pickle()

    def __call__(self, start=None, stop=None, chans=None):
        """Called when Stream object is called using (). start and stop are timepoints in us
        wrt t=0. Returns the corresponding WaveForm object with just the specified chans.

        As of 2017-10-24 I'm not sure if this behaviour qualifies as end-inclusive or not,
        but I suspect not. See how these single Streams are called in MultiStream.__call__
        """
        if start is None:
            start = self.t0
        if stop is None:
            stop = self.t1
        if chans is None:
            chans = self.chans
        kind = self.kind
        if not set(chans).issubset(self.chans):
            raise ValueError("Requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, self.kind))
        nchans = len(chans)
        tres, rawtres = self.tres, self.rawtres # float us
        # mintres handles both high-pass data where tres < rawtres (40 us), and low-pass
        # data where tres = rawtres (1000 us):
        mintres = min(tres, rawtres)
        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at
            # key.start and key.stop
            xs = KERNELSIZE * rawtres # float us
        else:
            xs = 0.0
        # stream limits, in sample indices:
        t0i = intround(self.t0 / rawtres)
        t1i = intround(self.t1 / rawtres)
        # get a slightly greater range of raw data (with xs) than might be needed:
        t0xsi = intround((start - xs) / rawtres) # round to nearest mult of rawtres
        t1xsi = intround((stop + xs) / rawtres) # round to nearest mult of rawtres
        # stay within stream *slice* limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, t0i)
        t1xsi = min(t1xsi, t1i+1)
        # convert back to nearest float us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        # these are slice indices, so don't add 1:
        ntxs = t1xsi - t0xsi # int
        tsxs = np.linspace(t0xs, t0xs+(ntxs-1)*rawtres, ntxs) # end inclusive
        #print('t0xs, t1xs, ntxs: %f, %f, %d' % (t0xs, t1xs, ntxs))

        # init data as int32 so we have bitwidth to rescale and zero, convert to int16 later
        dataxs = np.zeros((nchans, ntxs), dtype=np.int32) # any gaps will have zeros

        # Find all contiguous tranges that t0xs and t1xs span, if any. Note that this
        # can now deal with case where len(trangeis) > 1. Test by asking for a slice
        # longer than any one trange or gap between tranges, like by calling:
        # >>> self.hpstream(201900000, 336700000)
        # on file ptc15.74.
        trangeis, = np.where((self.tranges[:, 0] <= t1xs) & (t0xs < self.tranges[:, 1]))
        tranges = []
        if len(trangeis) > 0:
            tranges = self.tranges[trangeis]
        #print('tranges:'); print(tranges)
        # collect relevant records from spanned tranges, if any:
        records = []
        for trange in tranges:
            trrec0i, trrec1i = self.records['TimeStamp'].searchsorted(trange)
            trrecis = np.arange(trrec0i, trrec1i)
            trrts = self.records['TimeStamp'][trrecis]
            trrecs = self.records[trrecis]
            rec0i, rec1i = trrts.searchsorted([t0xs, t1xs])
            rec0i = max(rec0i-1, 0)
            recis = np.arange(rec0i, rec1i)
            records.append(trrecs[recis])
        if len(records) > 0:
            records = np.concatenate(records)

        # load up data+excess, from all relevant records
        # TODO: fix code duplication
        #tload = time.time()
        if kind == 'highpass': # straightforward
            chanis = self.layout.ADchanlist.searchsorted(chans)
            for record in records: # iterating over highpass records
                d = self.f.loadContinuousRecord(record)[chanis] # record's data on chans
                nt = d.shape[1]
                t0i = intround(record['TimeStamp'] / rawtres)
                t1i = t0i + nt
                # source indices
                st0i = max(t0xsi - t0i, 0)
                st1i = min(t1xsi - t0i, nt)
                # destination indices
                dt0i = max(t0i - t0xsi, 0)
                dt1i = min(t1i - t0xsi, ntxs)
                dataxs[:, dt0i:dt1i] = d[:, st0i:st1i]
        else: # kind == 'lowpass', need to load chans from subsequent records
            chanis = [ int(np.where(chan == self.layout.chans)[0]) for chan in chans ]
            """NOTE: if the above raises an error it may be because this particular
            combination of LFP chans was incorrectly parsed due to a bug in the .srf file,
            and a manual remapping needs to be added to Surf.File.fixLFPlabels()"""
            # assume all lpmc records are same length:
            nt = intround(records[0]['NumSamples'] / self.nADchans)
            d = np.zeros((nchans, nt), dtype=np.int32)
            for record in records: # iterating over lowpassmultichan records
                for i, chani in enumerate(chanis):
                    lprec = self.f.lowpassrecords[record['lpreci']+chani]
                    d[i] = self.f.loadContinuousRecord(lprec)
                t0i = intround(record['TimeStamp'] / rawtres)
                t1i = t0i + nt
                # source indices
                st0i = max(t0xsi - t0i, 0)
                st1i = min(t1xsi - t0i, nt)
                # destination indices
                dt0i = max(t0i - t0xsi, 0)
                dt1i = min(t1i - t0xsi, ntxs)
                dataxs[:, dt0i:dt1i] = d[:, st0i:st1i]
        #print('record.load() took %.3f sec' % (time.time()-tload))

        # bitshift left to scale 12 bit values to use full 16 bit dynamic range, same as
        # * 2**(16-12) == 16. This provides more fidelity for interpolation, reduces uV per
        # AD to about 0.02
        dataxs <<= 4 # data is still int32 at this point

        ## TODO: add highpass filtering, although it probably won't make much difference
        if self.filtmeth:
            raise NotImplementedError("SurfStream doesn't support filtering yet")

        # do any resampling if necessary:
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs, chans)
            #print('resample took %.3f sec' % (time.time()-tresample))

        ## TODO: add CAR here, after S+H correction (in self.resample) rather than before it,
        ## because CAR assumes simultaneous timepoints across chans. Also need to slice out
        ## chans only at the very end, as in DATStream, and size and maintain dataxs with
        ## enabled self.chans as rows, not requested chans as rows, which might be only a
        ## subset of self.chans
        if self.car:
            raise NotImplementedError("SurfStream doesn't support CAR yet")

        # Trim down to just the requested time range.
        # For trimming time range, use integer multiple of mintres to prevent floating point
        # round-off error (when mintres is non-integer us) that can occasionally
        # cause searchsorted to produce off-by-one indices:
        tsxsi = intround(tsxs / mintres)
        starti, stopi = intround(start/mintres), intround(stop/mintres)
        lo, hi = tsxsi.searchsorted([starti, stopi])

        data = dataxs[:, lo:hi]
        ts = tsxs[lo:hi]

        #print('Stream start, stop, tres, shape:\n', start, stop, self.tres, data.shape)
        # should be safe to convert back down to int16 now:
        data = np.int16(data)
        return WaveForm(data=data, ts=ts, chans=chans, tres=tres)


class SimpleStream(Stream):
    """Simple Stream loaded fully in advance"""
    def __init__(self, fname, wavedata, siteloc, rawsampfreq, masterclockfreq,
                 intgain, extgain, filtmeth=None, car=None, sampfreq=None,
                 shcorrect=None, bitshift=4, tsfversion=None):
        self._fname = fname
        self.wavedata = wavedata
        self.filtmeth = filtmeth
        self.car = car
        nchans, nt = wavedata.shape
        self.chans = np.arange(nchans) # this sets self.nchans
        self.nt = nt
        self.nADchans = self.nchans
        # assumes contiguous 0-based channel IDs, used in self.__call__():
        self.ADchans = np.arange(self.nADchans)
        self.layout = EmptyClass()
        self.layout.ADchanlist = self.ADchans # for the sake of self.resample()
        self.probe = probes.findprobe(siteloc) # deduce probe type from siteloc array
        self.rawsampfreq = rawsampfreq
        self.rawtres = 1 / self.rawsampfreq * 1e6 # float us
        self.masterclockfreq = masterclockfreq
        self.extgain = extgain
        self.intgain = intgain
        if tsfversion == 1002:
            self.converter = core.Converter_TSF_1002(intgain, extgain)
        else:
            self.converter = core.Converter(intgain, extgain)
        self.sampfreq = sampfreq or DEFHPRESAMPLEX * self.rawsampfreq
        self.shcorrect = shcorrect
        self.bitshift = bitshift
        self.t0 = 0.0 # float us
        self.t1 = nt * self.rawtres # float us
        self.tranges = np.int64([[self.t0, self.t1]]) # int us, 2D

    def open(self):
        pass

    def is_open(self):
        return True

    def close(self):
        pass

    def get_fname(self):
        return self._fname

    fname = property(get_fname)

    def get_fnames(self):
        return [self._fname]

    fnames = property(get_fnames)

    def get_datetime(self):
        """.tsf files don't currently have a datetime stamp, return Unix epoch instead"""
        return UNIXEPOCH

    datetime = property(get_datetime)

    def get_masterclockfreq(self):
        return self._masterclockfreq

    def set_masterclockfreq(self, masterclockfreq):
        self._masterclockfreq = masterclockfreq

    masterclockfreq = property(get_masterclockfreq, set_masterclockfreq)
    
    def __getstate__(self):
        """Get object state for pickling"""
        # copy it cuz we'll be making changes, this is fast because it's just a shallow copy
        d = self.__dict__.copy()
        try: del d['wavedata'] # takes up way too much space
        except KeyError: pass
        return d

    def __call__(self, start=None, stop=None, chans=None):
        """Called when Stream object is called using (). start and stop are timepoints in us
        wrt t=0. Returns the corresponding WaveForm object with just the specified chans.

        As of 2017-10-24 I'm not sure if this behaviour qualifies as end-inclusive or not,
        but I suspect not. See how these single Streams are called in MultiStream.__call__
        """
        if start is None:
            start = self.t0
        if stop is None:
            stop = self.t1
        if chans is None:
            chans = self.chans
        kind = self.kind
        if not set(chans).issubset(self.chans):
            raise ValueError("Requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, kind))
        nchans = len(chans)
        chanis = self.ADchans.searchsorted(chans)
        tres, rawtres = self.tres, self.rawtres # float us
        # mintres handles both high-pass data where tres < rawtres, and low-pass decimated
        # data where tres > rawtres:
        mintres = min(tres, rawtres)
        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at
            # key.start and key.stop
            xs = KERNELSIZE * rawtres # float us
        else:
            xs = 0.0
        # stream limits, in sample indices:
        t0i = intround(self.t0 / rawtres)
        t1i = intround(self.t1 / rawtres)
        # get a slightly greater range of raw data (with xs) than might be needed:
        t0xsi = intround((start - xs) / rawtres) # round to nearest mult of rawtres
        t1xsi = intround((stop + xs) / rawtres) # round to nearest mult of rawtres
        # stay within stream *slice* limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, t0i)
        t1xsi = min(t1xsi, t1i+1)
        # convert back to nearest float us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        # these are slice indices, so don't add 1:
        ntxs = t1xsi - t0xsi # int
        tsxs = np.linspace(t0xs, t0xs+(ntxs-1)*rawtres, ntxs) # end inclusive
        #print('t0xs, t1xs, ntxs: %f, %f, %d' % (t0xs, t1xs, ntxs))

        # slice out excess data on requested channels, init as int32 so we have bitwidth
        # to rescale and zero, convert to int16 later:
        dataxs = np.int32(self.wavedata[chanis, t0xsi:t1xsi])

        # bitshift left by self.bitshift to use full 16 bit dynamic range, same as
        # * 2**(16-12) == 16. This provides more fidelity for interpolation, reduces uV per
        # AD to about 0.02
        if self.bitshift:
            dataxs <<= self.bitshift # data is still int32 at this point

        ## TODO: add highpass filtering
        if self.filtmeth:
            raise NotImplementedError("SimpleStream doesn't support filtering yet")

        # do any resampling if necessary:
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs, chans)
            #print('resample took %.3f sec' % (time.time()-tresample))

        ## TODO: add CAR here, after S+H correction (in self.resample) rather than before it,
        ## because CAR assumes simultaneous timepoints across chans. Also need to slice out
        ## chans only at the very end, as in DATStream, and size and maintain dataxs with
        ## enabled self.chans as rows, not requested chans as rows, which might be only a
        ## subset of self.chans
        if self.car:
            raise NotImplementedError("SimpleStream doesn't support CAR yet")

        # Trim down to just the requested time range.
        # For trimming time range, use integer multiple of mintres to prevent floating point
        # round-off error (when mintres is non-integer us) that can occasionally
        # cause searchsorted to produce off-by-one indices:
        tsxsi = intround(tsxs / mintres)
        starti, stopi = intround(start/mintres), intround(stop/mintres)
        lo, hi = tsxsi.searchsorted([starti, stopi])

        data = dataxs[:, lo:hi]
        ts = tsxs[lo:hi]

        #print('Stream start, stop, tres, shape:\n', start, stop, self.tres, data.shape)
        # should be safe to convert back down to int16 now:
        data = np.int16(data)
        return WaveForm(data=data, ts=ts, chans=chans, tres=tres)


class MultiStream(object):
    """A collection of multiple streams, all from the same track/insertion/series. This is
    used to simultaneously cluster all spikes from many (or all) recordings from the same
    track. Designed to have as similar an interface as possible to a normal Stream. fs
    needs to be a list of open and parsed data file objects, in temporal order"""
    def __init__(self, fs, trackfname, kind='highpass', filtmeth=None, car=None,
                 sampfreq=None, shcorrect=None):
        # to prevent pickling problems, don't bind fs
        self.fname = trackfname
        self.kind = kind
        streams = []
        self.streams = streams # bind right away so setting sampfreq and shcorrect will work
        # collect appropriate streams from fs
        if kind == 'highpass':
            for f in fs:
                streams.append(f.hpstream)
        elif kind == 'lowpass':
            for f in fs:
                streams.append(f.lpstream)
        else: raise ValueError('Unknown stream kind %r' % kind)

        datetimes = [stream.datetime for stream in streams]
        if not (np.diff(datetimes) >= timedelta(0)).all():
            raise RuntimeError("Files aren't in temporal order")

        try: self.layout = streams[0].layout # assume they're identical
        except AttributeError: pass
        try:
            gains = np.asarray([ stream.converter.intgain for stream in streams ])
        except AttributeError:
            gains = np.asarray([ stream.converter.AD2uVx for stream in streams ])
        if max(gains) != min(gains):
            import pdb; pdb.set_trace() # investigate which are the deviant files
            raise NotImplementedError("Not all files have the same gain")
            # TODO: find recording with biggest intgain, call that value maxintgain. For each
            # recording, scale its AD values by its intgain/maxintgain when returning a slice
            # from its stream. Note that this ratio should always be a factor of 2, so all you
            # have to do is bitshift, I think. Then, have a single converter for the
            # MultiStream whose intgain value is set to maxintgain
        self.converter = streams[0].converter # they're identical
        self.fnames = [f.fname for f in fs]
        self.rawsampfreq = streams[0].rawsampfreq # assume they're identical
        self.rawtres = streams[0].rawtres # float us, assume they're identical
        contiguous = np.asarray([stream.contiguous for stream in streams])
        if not contiguous.all() and kind == 'highpass':
            # don't bother reporting again for lowpass
            fnames = [ s.fname for s, c in zip(streams, contiguous) if not c ]
            print("Some files are non contiguous:")
            for fname in fnames:
                print(fname)
        probe = streams[0].probe
        adapter = streams[0].adapter
        if not np.all([type(probe) == type(stream.probe) for stream in streams]):
            raise RuntimeError("Some files have different probe types")
        if not np.all([type(adapter) == type(stream.adapter) for stream in streams]):
            raise RuntimeError("Some files have different adapter types")
        self.probe = probe # they're identical
        self.adapter = adapter # they're identical

        """Generate tranges, an array of all the contiguous data ranges in all the
        streams in self. These are relative to the start of acquisition (t=0) in the first
        stream. Round the time deltas between neighbouring streams to the nearest multiple
        of rawtres to avoid problems indexing across streams. This works for both high-pass
        data whose tres < rawtres, and decimated low-pass data whose tres > rawtres"""
        tranges = []
        rawtres = self.rawtres # float us
        for stream in streams:
            # time delta between this stream and first stream:
            dt = td2fusec(stream.datetime - datetimes[0]) # float us
            dt = intround(dt / rawtres) * rawtres # round to nearest raw timepoint
            # stream.tranges is only ever 1 row for a regular non-multi stream:
            assert len(stream.tranges) == 1
            t0, t1 = stream.tranges[0]
            streamnt = (t1 - t0) / rawtres
            assert streamnt % 1 == 0 # ensure integer num of timepoints between t0 and t1
            streamnt = int(streamnt)
            t0 = intround(t0 / rawtres) * rawtres
            t1 = t0 + streamnt * rawtres
            tranges.append([dt+t0, dt+t1])
        self.tranges = np.asarray(tranges)
        self.t0 = self.tranges[0, 0] # float us
        self.t1 = self.tranges[-1, 1] # float us

        # set filtmeth, car, sampfreq, and shcorrect for all streams:
        streamtype = type(streams[0])
        if streamtype == DATStream:
            if kind == 'highpass':
                self.filtmeth = filtmeth or DEFDATFILTMETH
                self.car = car or DEFCAR
                self.sampfreq = sampfreq or self.rawsampfreq * DEFHPRESAMPLEX
                self.shcorrect = shcorrect or DEFHPDATSHCORRECT
            else: # kind == 'lowpass'
                return None
        elif streamtype == NSXStream:
            if kind == 'highpass':
                self.filtmeth = filtmeth or DEFNSXFILTMETH
                self.car = car or DEFCAR
                self.sampfreq = sampfreq or self.rawsampfreq * DEFHPRESAMPLEX
                self.shcorrect = shcorrect or DEFHPNSXSHCORRECT
            else: # kind == 'lowpass'
                return None
        elif streamtype == SurfStream:
            if kind == 'highpass':
                self.filtmeth = filtmeth
                self.car = car
                self.sampfreq = sampfreq or self.rawsampfreq * DEFHPRESAMPLEX
                self.shcorrect = shcorrect or DEFHPSRFSHCORRECT
            else: # kind == 'lowpass'
                self.filtmeth = filtmeth
                self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
                self.shcorrect = shcorrect or False # don't s+h correct by default

    def is_multi(self):
        """Convenience method to specify if self is a MultiStream"""
        return True

    def is_open(self):
        return np.all([stream.is_open() for stream in self.streams])

    def open(self):
        for stream in self.streams:
            stream.open()

    def close(self):
        for stream in self.streams:
            stream.close()

    def get_dt(self):
        """Get self's duration"""
        return self.t1 - self.t0

    dt = property(get_dt)

    def get_chans(self):
        return self.streams[0].chans # assume they're identical

    def set_chans(self, chans):
        for stream in self.streams:
            stream.chans = chans

    chans = property(get_chans, set_chans)

    def get_ext(self):
        """Get file extension of data source"""
        return os.path.splitext(self.fnames[0])[-1] # take extension of first fname

    ext = property(get_ext)

    def get_srcfnameroot(self):
        """Get root of filename of source data"""
        srcfnameroot = lrstrip(self.fname, '../', '.track')
        return srcfnameroot

    srcfnameroot = property(get_srcfnameroot)

    def get_nchans(self):
        return len(self.chans)

    nchans = property(get_nchans)

    def get_filtmeth(self):
        return self.streams[0].filtmeth # they're identical

    def set_filtmeth(self, filtmeth):
        for stream in self.streams:
            stream.filtmeth = filtmeth

    filtmeth = property(get_filtmeth, set_filtmeth)

    def get_car(self):
        return self.streams[0].car # they're identical

    def set_car(self, car):
        for stream in self.streams:
            stream.car = car

    car = property(get_car, set_car)

    def get_sampfreq(self):
        return self.streams[0].sampfreq # they're identical

    def set_sampfreq(self, sampfreq):
        for stream in self.streams:
            stream.sampfreq = sampfreq

    sampfreq = property(get_sampfreq, set_sampfreq)

    def get_tres(self):
        return self.streams[0].tres # they're identical

    tres = property(get_tres)

    def get_shcorrect(self):
        return self.streams[0].shcorrect # they're identical

    def set_shcorrect(self, shcorrect):
        for stream in self.streams:
            stream.shcorrect = shcorrect

    shcorrect = property(get_shcorrect, set_shcorrect)
    '''
    # having this would make sense, but it isn't currently needed:
    def get_datetime(self):
        return self.streams[0].datetime # datetime of first stream

    datetime = property(get_datetime)
    '''

    def get_filtering(self):
        """Get filtering settings of first stream in self"""
        return self.streams[0].filtering

    filtering = property(get_filtering)

    def get_israw(self):
        """Return if self is currently set to return raw data"""
        return self.streams[0].israw

    israw = property(get_israw)

    def pickle(self):
        """Just a way to pickle all the files associated with self"""
        for stream in self.streams:
            stream.pickle()

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us. Returns the corresponding WaveForm
        object with the full set of chans"""
        if key.step not in [None, 1]:
            raise ValueError('Unsupported slice step size: %s' % key.step)
        return self(key.start, key.stop, self.chans)

    def __call__(self, start=None, stop=None, chans=None, checksat=False):
        """Called when Stream object is called using (). start and stop are
        timepoints in us wrt t=0. Returns the corresponding WaveForm object with just the
        specified chans. Figure out which stream(s) the slice spans (usually just one,
        sometimes 0 or 2), send the request to the stream(s), generate the appropriate
        timestamps, and return the waveform"""
        if start is None:
            start = self.t0
        if stop is None:
            stop = self.t1
        if chans is None:
            chans = self.chans
        if not set(chans).issubset(self.chans):
            raise ValueError("Requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, self.kind))

        #print('*** new MultiStream.__call__()')
        nchans = len(chans)
        tres, rawtres = self.tres, self.rawtres # float us
        #print('Multi start, stop', start, stop)
        start, stop = max(self.t0, start), min(stop, self.t1+tres) # stay within stream limits
        #print('Multi limit start, stop', start, stop)
        nt = intround((stop - start) / tres) # in units of tres
        stop = start + nt * tres # in units of tres
        #print('Multi nearest rawtres start, stop', start, stop)
        streamis = []
        ## TODO: this could probably be more efficient by not iterating over all streams:
        for streami, trange in enumerate(self.tranges):
            if (trange[0] <= start < trange[1]) or (trange[0] <= stop < trange[1]):
                streamis.append(streami)
        # safer to use linspace than arange in case of float tres, deals with endpoints
        # better and gives slightly more accurate output float timestamps:
        ts = np.linspace(start, start+(nt-1)*tres, nt) # end inclusive
        assert len(ts) == nt
        data = np.zeros((nchans, nt), dtype=np.int16) # any gaps will have zeros
        satis = None
        if checksat:
            satis = np.zeros((nchans, nt), dtype=bool)
        # iterate over all relevant streams:
        for streami in streamis:
            stream = self.streams[streami]
            streamt0 = intround(stream.t0 / rawtres) * rawtres
            streamt1 = intround(stream.t1 / rawtres) * rawtres
            abst0 = self.tranges[streami, 0] # absolute start time of this stream
            # find start and end offsets relative to abst0, while observing lower and upper
            # stream limits:
            relt0 = max(0, start - abst0)
            relt1 = min(stop - abst0, streamt1 - streamt0 + rawtres) # end inclusive at
                                                                     # upper stream limit?
            # source slice times:
            st0 = relt0 + streamt0
            st1 = relt1 + streamt0
            #print('Multi abst0, relt0, st0, st1:', abst0, relt0, st0, st1)
            wave = stream(st0, st1, chans, checksat) # source WaveForm
            sdata, ssatis = wave.data, wave.satis # source data and satis, in units of tres
            # destination slice indices:
            dt0i = intround((abst0 + relt0 - start) / tres) # absolute index
            dt1i = dt0i + sdata.shape[1]
            #print('Multi dt0i, dt1i', dt0i, dt1i)
            #print('Multi start, stop, tres, sdata, data:\n',
            #      start, stop, tres, sdata.shape, data.shape)
            data[:, dt0i:dt1i] = sdata # destination data, in units of tres
            if checksat:
                satis[:, dt0i:dt1i] = ssatis # destination satis, in units of tres
            assert data.shape[1] == len(ts)
        return WaveForm(data=data, ts=ts, chans=chans, tres=tres, satis=satis)

    def get_block_tranges(self, bs=10000000):
        """Get time ranges spanning self, in block sizes of bs us
        HACK: this was copied from Stream class"""
        tranges = []
        for start in np.arange(self.t0, self.t1, bs):
            stop = start + bs
            tranges.append((start, stop))
        tranges = np.asarray(tranges)
        # don't exceed self.t1:
        tranges[-1, -1] = self.t1
        return tranges

    def get_block_data(self, bs=10000000, step=None, chans=None, units='uV'):
        """Get blocks of data in block sizes of bs us, keeping every step'th data point
        (default keeps all), on specified chans, in units.
        HACK: this was copied from Stream class"""
        tranges = self.get_block_tranges(bs=bs)
        data = []
        for (start, stop) in tranges:
            blockwave = self(start, stop, chans=chans)
            data.append(blockwave.data[::step]) # decimate
        data = np.concatenate(data, axis=1) # concatenate horizontally
        if units is None:
            return data
        elif units == 'uV':
            return self.converter.AD2uV(data)
        else:
            raise ValueError("Unknown units %r" % units)
