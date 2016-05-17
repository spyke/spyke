"""Stream objects for requesting arbitrary time ranges of data from various file types
in a common way"""

from __future__ import division

__authors__ = ['Martin Spacek']

import numpy as np

import core
from core import (WaveForm, EmptyClass, intround, lrstrip, hamming, MU)
from core import DEFHPSRFSAMPFREQ, DEFHPNSXSAMPFREQ, DEFHPSHCORRECT, NCHANSPERBOARD, KERNELSIZE
import probes

## TODO: rename Stream to SurfStream, factor out stuff common to all the single streams
## into a Stream class. Also, rename TrackStream to Streams? or MultiStream? Note that this
## kind of renaming might cause problems with pickled streams in the .sort files


class Stream(object):


class NSXStream(Stream):
    pass


class SurfStream(Stream):
    """Data stream object - provides stream interface to .srf files.
    Maps from timestamps to record index of stream data to retrieve the
    approriate range of waveform data from disk"""
    def __init__(self, f, kind='highpass', sampfreq=None, shcorrect=None):
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

        # assume same layout for all records of type "kind"
        self.layout = self.f.layoutrecords[self.records['Probe'][0]]
        intgain = self.layout.intgain
        extgain = int(self.layout.extgain[0]) # assume same extgain for all chans in layout
        self.converter = Converter(intgain, extgain)
        self.nADchans = self.layout.nchans # always constant
        self.rawsampfreq = self.layout.sampfreqperchan
        self.rawtres = intround(1 / self.rawsampfreq * 1e6) # us
        if kind == 'highpass':
            if list(self.layout.ADchanlist) != range(self.nADchans):
                print("WARNING: ADchans aren't contiguous from 0, highpass recordings are "
                      "nonstandard. Sample and hold delay correction in self.resample() "
                      "may not be exactly correct")
            # probe chans, as opposed to AD chans. Most probe types are contiguous from 0,
            # but there are some exceptions (like pt16a_HS27 and pt16b_HS27):
            self.chans = np.arange(self.nADchans)
            self.sampfreq = sampfreq or DEFHPSRFSAMPFREQ # desired sampling frequency
            self.shcorrect = shcorrect or DEFHPSHCORRECT
        else: # kind == 'lowpass'
            # probe chan values are already parsed from LFP probe description
            self.chans = self.layout.chans
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default
        probename = self.layout.electrode_name
        probename = probename.replace(MU, 'u') # replace any 'micro' symbols with 'u'
        probetype = eval('probes.' + probename) # yucky. TODO: switch to a dict with keywords?
        self.probe = probetype() # instantiate it

        rts = self.records['TimeStamp'] # array of record timestamps
        NumSamples = np.unique(self.records['NumSamples'])
        if len(NumSamples) > 1:
            raise RuntimeError("Not all continuous records are of the same length. "
                               "NumSamples = %r" % NumSamples)
        rtlen = NumSamples / self.nADchans * self.rawtres
        # Check whether rts values are all equally spaced, indicating there were no
        # pauses in recording
        diffrts = np.diff(rts)
        self.contiguous = (np.diff(diffrts) == 0).all() # could also call diff(rts, n=2)
        if self.contiguous:
            try: assert np.unique(diffrts) == rtlen
            except AssertionError: import pdb; pdb.set_trace()
            self.tranges = np.int64([[rts[0], rts[-1]+rtlen]]) # keep it 2D
        else:
            if kind == 'highpass': # don't bother reporting again for lowpass
                print('NOTE: time gaps exist in %s, possibly due to pauses' % self.fname)
            # build up self.tranges
            splitis = np.where(diffrts != rtlen)[0] + 1
            splits = np.split(rts, splitis) # list of arrays of contiguous rts
            tranges = []
            for split in splits: # for each array of contiguous rts
                tranges.append([split[0], split[-1]+rtlen])
            self.tranges = np.int64(tranges)
        self.t0 = self.tranges[0, 0]
        self.t1 = self.tranges[-1, 1]

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
        return [self.f.fname]

    fnames = property(get_fnames)

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
        self.tres = intround(1 / self.sampfreq * 1e6) # us, for convenience

    sampfreq = property(get_sampfreq, set_sampfreq)

    def get_masterclockfreq(self):
        return self.f.layoutrecords[0].MasterClockFreq
        
    masterclockfreq = property(get_masterclockfreq)

    def get_shcorrect(self):
        return self._shcorrect

    def set_shcorrect(self, shcorrect):
        """On .shcorrect change, deletes .kernels (if set)"""
        self._shcorrect = shcorrect
        try:
            del self.kernels
        except AttributeError:
            pass

    shcorrect = property(get_shcorrect, set_shcorrect)

    def get_datetime(self):
        return self.f.datetime

    datetime = property(get_datetime)

    def pickle(self):
        self.f.pickle()

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us. Returns the corresponding WaveForm
        object with the full set of chans"""
        if key.step not in [None, 1]:
            raise ValueError('unsupported slice step size: %s' % key.step)
        return self(key.start, key.stop, self.chans)

    def __call__(self, start, stop, chans=None):
        """Called when Stream object is called using (). start and stop indicate start
        and end timepoints in us. Returns the corresponding WaveForm object with just the
        specified chans"""
        if chans == None:
            chans = self.chans
        if not set(chans).issubset(self.chans):
            raise ValueError("requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, self.kind))
        nchans = len(chans)
        rawtres = self.rawtres
        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at
            # key.start and key.stop
            xs = KERNELSIZE * rawtres
        else:
            xs = 0
        # get a slightly greater range of raw data (with xs) than might be needed:
        t0xsi = (start - xs) // rawtres # round down to nearest mult of rawtres
        t1xsi = ((stop + xs) // rawtres) + 1 # round up to nearest mult of rawtres
        # stay within stream limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, self.t0 // rawtres)
        t1xsi = min(t1xsi, self.t1 // rawtres)
        # convert back to us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        tsxs = np.arange(t0xs, t1xs, rawtres)
        ntxs = len(tsxs)
        # init data as int32 so we have bitwidth to rescale and zero, then convert to int16
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
        if self.kind == 'highpass': # straightforward
            chanis = self.layout.ADchanlist.searchsorted(chans)
            for record in records: # iterating over highpass records
                d = self.f.loadContinuousRecord(record)[chanis] # record's data on chans
                nt = d.shape[1]
                t0i = record['TimeStamp'] // rawtres
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
            nt = records[0]['NumSamples'] / self.nADchans
            d = np.zeros((nchans, nt), dtype=np.int32)
            for record in records: # iterating over lowpassmultichan records
                for i, chani in enumerate(chanis):
                    lprec = self.f.lowpassrecords[record['lpreci']+chani]
                    d[i] = self.f.loadContinuousRecord(lprec)
                t0i = record['TimeStamp'] // rawtres
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

        # do any resampling if necessary:
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs, chans)
            #print('resample took %.3f sec' % (time.time()-tresample))

        # now trim down to just the requested time range:
        lo, hi = tsxs.searchsorted([start, stop])
        data = dataxs[:, lo:hi]
        ts = tsxs[lo:hi]

        # should be safe to convert back down to int16 now:
        data = np.int16(data)
        return WaveForm(data=data, ts=ts, chans=chans)

    def resample(self, rawdata, rawts, chans):
        """Return potentially sample-and-hold corrected and Nyquist interpolated
        data and timepoints. See Blanche & Swindale, 2006"""
        #print('sampfreq, rawsampfreq, shcorrect = (%r, %r, %r)' %
        #      (self.sampfreq, self.rawsampfreq, self.shcorrect))
        rawtres = self.rawtres # us
        tres = self.tres # us
        # resample factor: n output resampled points per input raw point:
        resamplex = intround(self.sampfreq / self.rawsampfreq)
        assert resamplex >= 1, 'no decimation allowed'
        N = KERNELSIZE

        # check if kernels have been generated already
        try:
            self.kernels
        except AttributeError:
            self.kernels = self.get_kernels(self.layout.ADchanlist, resamplex, N)

        # convolve the data with each kernel
        nrawts = len(rawts)
        nchans = len(chans)
        # all the interpolated points have to fit in between the existing raw
        # points, so there's nrawts - 1 of each of the interpolated points:
        #nt = nrawts + (resamplex-1) * (nrawts - 1)
        # the above can be simplified to:
        nt = nrawts*resamplex - resamplex + 1
        tstart = rawts[0]
        ts = np.arange(tstart, tstart+tres*nt, tres) # generate interpolated timepoints
        #print 'len(ts) is %r' % len(ts)
        assert len(ts) == nt
        # resampled data, leave as int32 for convolution, then convert to int16:
        data = np.empty((nchans, nt), dtype=np.int32)
        #print 'data.shape = %r' % (data.shape,)
        #tconvolve = time.time()
        #tconvolvesum = 0
        # Only the chans that are actually needed are resampled and returned.
        # Assume that chans index into ADchans. Normally they should map 1 to 1, ie chan 0
        # taps off of ADchan 0, but for probes like pt16a_HS27 and pt16b_HS27, it seems
        # ADchans start at 4. However, because self.kernels is indexed into using chans, and
        # chans are always assumed to be contiguous from 0, this shouldn't cause a problem
        for chani, chan in enumerate(chans):
            for point, kernel in enumerate(self.kernels[chan]):
                """np.convolve(a, v, mode)
                for mode='same', only the K middle values are returned starting at n = (M-1)/2
                where K = len(a)-1 and M = len(v) - 1 and K >= M
                for mode='valid', you get the middle len(a) - len(v) + 1 number of values"""
                #tconvolveonce = time.time()
                row = np.convolve(rawdata[chani], kernel, mode='same')
                #tconvolvesum += (time.time()-tconvolveonce)
                #print 'len(rawdata[chani]) = %r' % len(rawdata[chani])
                #print 'len(kernel) = %r' % len(kernel)
                #print 'len(row): %r' % len(row)
                # interleave by assigning from point to end in steps of resamplex
                # index to start filling data from for this kernel's points:
                ti0 = (resamplex - point) % resamplex
                # index of first data point to use from convolution result 'row':
                rowti0 = int(point > 0)
                # discard the first data point from interpolant's convolutions, but not for
                # raw data's convolutions, since interpolated values have to be bounded on both
                # sides by raw values?
                data[chani, ti0::resamplex] = row[rowti0:]
        #print('convolve loop took %.3f sec' % (time.time()-tconvolve))
        #print('convolve calls took %.3f sec total' % (tconvolvesum))
        #tundoscaling = time.time()
        data >>= 16 # undo kernel scaling, shift 16 bits right in place, same as //= 2**16
        #print('undo kernel scaling took %.3f sec total' % (time.time()-tundoscaling))
        return data, ts

    def get_kernels(self, ADchans, resamplex, N):
        """Generate a different set of kernels for each ADchan to correct each ADchan's
        s+h delay. ADchans may not always be contiguous from 0, but chans are assumed
        to always be, and to always be in same order as ADchans.

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
        # ordinal position of each ADchan in the hold queue of its ADC board:
        i = ADchans % NCHANSPERBOARD
        if self.shcorrect:
            ## TODO: stop hard-coding 1 masterclockfreq tick delay per ordinal position
            # per channel delays, us, usually 1 us/chan:
            dis = 1000000 / self.masterclockfreq * i
        else:
            dis = 0 * i
        ds = dis / self.rawtres # normalized per channel delays
        wh = hamming # window function
        h = np.sinc # sin(pi*t) / pi*t
        kernels = [] # list of array of kernels, indexed by [chan][resample point]
        for d in ds: # delay for this ADchan
            kernelrow = []
            for point in xrange(resamplex): # iterate over resampled points per raw point
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
            kernels.append(kernelrow)
        return kernels


class SimpleStream(SurfStream):
    """Simple Stream loaded fully in advance"""
    def __init__(self, fname, wavedata, siteloc, rawsampfreq, masterclockfreq,
                 intgain, extgain, sampfreq=None, shcorrect=None, bitshift=4,
                 tsfversion=None):
        self._fname = fname
        self.wavedata = wavedata
        nchans, nt = wavedata.shape
        self.chans = np.arange(nchans) # this sets self.nchans
        self.nt = nt
        self.nADchans = self.nchans
        # assumes contiguous 0-based channel IDs, used in self.__call__():
        self.ADchans = np.arange(self.nADchans)
        self.layout = EmptyClass()
        self.layout.ADchanlist = self.ADchans # for the sake of self.resample()
        probematch = False
        for probetype in probes.TYPES:
            probe = probetype()
            if (probe.siteloc_arr().shape == siteloc.shape and
                (probe.siteloc_arr() == siteloc).all()):
                self.probe = probe
                probematch = True
                break
        if not probematch:
            raise ValueError("siteloc in %s doesn't match known probe type" % fname)
        self.rawsampfreq = rawsampfreq
        self.rawtres = intround(1 / self.rawsampfreq * 1e6) # us
        self.masterclockfreq = masterclockfreq
        self.extgain = extgain
        self.intgain = intgain
        if tsfversion == 1002:
            self.converter = Converter_TSF_1002(intgain, extgain)
        else:
            self.converter = Converter(intgain, extgain)
        self.sampfreq = sampfreq or DEFHPSRFSAMPFREQ # desired sampling frequency
        self.shcorrect = shcorrect
        self.bitshift = bitshift
        self.t0 = 0 # us
        self.t1 = nt * self.rawtres
        self.tranges = np.int64([[self.t0, self.t1]])

    def open(self):
        pass

    def is_open(self):
        return True

    def close(self):
        pass

    def get_fname(self):
        return self._fname

    fname = property(get_fname)

    def get_datetime(self):
        """.tsf files don't currently have a datetime stamp, return Unix epoch instead"""
        return UNIXEPOCH

    datetime = property(get_datetime)

    # override masterclockfreq property in base Stream class, make it behave like a simple
    # attribute:
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

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us. Returns the corresponding WaveForm
        object with the full set of chans"""
        if key.step not in [None, 1]:
            raise ValueError('unsupported slice step size: %s' % key.step)
        return self(key.start, key.stop, self.chans)

    def __call__(self, start, stop, chans=None):
        """Called when Stream object is called using (). start and stop indicate start
        and end timepoints in us. Returns the corresponding WaveForm object with just the
        specified chans"""
        if chans == None:
            chans = self.chans
        if not set(chans).issubset(self.chans):
            raise ValueError("requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, self.kind))
        nchans = len(chans)
        chanis = self.ADchans.searchsorted(chans)
        rawtres = self.rawtres
        resample = self.sampfreq != self.rawsampfreq or self.shcorrect == True
        if resample:
            # excess data in us at either end, to eliminate interpolation distortion at
            # key.start and key.stop
            xs = KERNELSIZE * rawtres
        else:
            xs = 0
        # get a slightly greater range of raw data (with xs) than might be needed:
        t0xsi = (start - xs) // rawtres # round down to nearest mult of rawtres
        t1xsi = ((stop + xs) // rawtres) + 1 # round up to nearest mult of rawtres
        # stay within stream limits, thereby avoiding interpolation edge effects:
        t0xsi = max(t0xsi, self.t0 // rawtres)
        t1xsi = min(t1xsi, self.t1 // rawtres)
        # convert back to us:
        t0xs = t0xsi * rawtres
        t1xs = t1xsi * rawtres
        tsxs = np.arange(t0xs, t1xs, rawtres)
        ntxs = len(tsxs)
        # slice out excess data on requested channels, init as int32 so we have bitwidth
        # to rescale and zero, convert to int16 later:
        dataxs = np.int32(self.wavedata[chanis, t0xsi:t1xsi])

        # bitshift left by 4 to scale 12 bit values to use full 16 bit dynamic range, same as
        # * 2**(16-12) == 16. This provides more fidelity for interpolation, reduces uV per
        # AD to about 0.02
        if self.bitshift:
            dataxs <<= self.bitshift # data is still int32 at this point

        # do any resampling if necessary:
        if resample:
            #tresample = time.time()
            dataxs, tsxs = self.resample(dataxs, tsxs, chans)
            #print('resample took %.3f sec' % (time.time()-tresample))

        # now trim down to just the requested time range:
        lo, hi = tsxs.searchsorted([start, stop])
        data = dataxs[:, lo:hi]
        ts = tsxs[lo:hi]

        # should be safe to convert back down to int16 now:
        data = np.int16(data)
        return WaveForm(data=data, ts=ts, chans=chans)


class MultiStream(object):
    """A collection of multiple streams, all from the same track/insertion/series. This is
    used to simultaneously cluster all spikes from many (or all) recordings from the same
    track. Designed to have as similar an interface as possible to a normal Stream. fs
    needs to be a list of open and parsed data file objects, in temporal order"""
    def __init__(self, fs, trackfname, kind='highpass', sampfreq=None, shcorrect=None):
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
            raise RuntimeError("files aren't in temporal order")

        """Generate tranges, an array of all the contiguous data ranges in all the
        streams in self. These are relative to the start of acquisition (t=0) in the first
        stream. Also generate streamtranges, an array of each stream's t0 and t1"""
        tranges = []
        streamtranges = []
        for stream in streams:
            td = stream.datetime - datetimes[0] # time delta between stream i and stream 0
            for trange in stream.tranges:
                t0 = td2usec(td + timedelta(microseconds=int(trange[0])))
                t1 = td2usec(td + timedelta(microseconds=int(trange[1])))
                tranges.append([t0, t1])
            streamt0 = td2usec(td + timedelta(microseconds=int(stream.t0)))
            streamt1 = td2usec(td + timedelta(microseconds=int(stream.t1)))
            streamtranges.append([streamt0, streamt1])
        self.tranges = np.int64(tranges)
        self.streamtranges = np.int64(streamtranges)
        self.t0 = self.streamtranges[0, 0]
        self.t1 = self.streamtranges[-1, 1]

        self.layout = streams[0].layout # assume they're identical
        intgains = np.asarray([ stream.converter.intgain for stream in streams ])
        if max(intgains) != min(intgains):
            import pdb; pdb.set_trace() # investigate which are the deviant files
            raise NotImplementedError("not all files have the same intgain")
            # TODO: find recording with biggest intgain, call that value maxintgain. For each
            # recording, scale its AD values by its intgain/maxintgain when returning a slice
            # from its stream. Note that this ratio should always be a factor of 2, so all you
            # have to do is bitshift, I think. Then, have a single converter for the
            # MultiStream whose intgain value is set to maxintgain
        self.converter = streams[0].converter # they're identical
        self.fnames = [f.fname for f in fs]
        self.rawsampfreq = streams[0].rawsampfreq # assume they're identical
        self.rawtres = streams[0].rawtres # assume they're identical
        contiguous = np.asarray([stream.contiguous for stream in streams])
        if not contiguous.all() and kind == 'highpass':
            # don't bother reporting again for lowpass
            fnames = [ s.fname for s, c in zip(streams, contiguous) if not c ]
            print("some files are non contiguous:")
            for fname in fnames:
                print(fname)
        probe = streams[0].probe
        if not np.all([type(probe) == type(stream.probe) for stream in streams]):
            raise RuntimeError("some files have different probe types")
        self.probe = probe # they're identical

        # set sampfreq and shcorrect for all streams
        if kind == 'highpass':
            self.sampfreq = sampfreq or DEFHPSRFSAMPFREQ # desired sampling frequency
            self.shcorrect = shcorrect or DEFHPSHCORRECT
        else: # kind == 'lowpass'
            self.sampfreq = sampfreq or self.rawsampfreq # don't resample by default
            self.shcorrect = shcorrect or False # don't s+h correct by default

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

    def get_nchans(self):
        return len(self.chans)

    nchans = property(get_nchans)

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
    def pickle(self):
        """Just a way to pickle all the files associated with self"""
        for stream in self.streams:
            stream.pickle()

    def __getitem__(self, key):
        """Called when Stream object is indexed into using [] or with a slice object,
        indicating start and end timepoints in us. Returns the corresponding WaveForm
        object with the full set of chans"""
        if key.step not in [None, 1]:
            raise ValueError('unsupported slice step size: %s' % key.step)
        return self(key.start, key.stop, self.chans)

    def __call__(self, start, stop, chans=None):
        """Figure out which stream(s) the slice spans (usually just one, sometimes 0 or
        2), send the request to the stream(s), generate the appropriate timestamps, and
        return the waveform"""
        if chans == None:
            chans = self.chans
        if not set(chans).issubset(self.chans):
            raise ValueError("requested chans %r are not a subset of available enabled "
                             "chans %r in %s stream" % (chans, self.chans, self.kind))
        nchans = len(chans)
        start, stop = max(start, self.t0), min(stop, self.t1) # stay in bounds
        streamis = []
        ## TODO: this could probably be more efficient by not iterating over all streams:
        for streami, trange in enumerate(self.streamtranges):
            if (trange[0] <= start < trange[1]) or (trange[0] <= stop < trange[1]):
                streamis.append(streami)
        tres = self.tres
        ts = np.arange(start, stop, tres)
        data = np.zeros((nchans, len(ts)), dtype=np.int16) # any gaps will have zeros
        for streami in streamis:
            stream = self.streams[streami]
            abst0 = self.streamtranges[streami, 0] # absolute start time of stream
            # find start and end offsets relative to abst0
            relt0 = max(start - abst0, 0) # stay within stream's lower limit
            relt1 = min(stop - abst0, stream.t1 - stream.t0) # stay within stream's upper limit
            # source slice times:
            st0 = relt0 + stream.t0
            st1 = relt1 + stream.t0
            sdata = stream(st0, st1, chans).data # source data
            # destination time indices:
            dt0i = (abst0 + relt0 - start) // tres # absolute index
            dt1i = dt0i + sdata.shape[1]
            data[:, dt0i:dt1i] = sdata
        return WaveForm(data=data, ts=ts, chans=chans)

