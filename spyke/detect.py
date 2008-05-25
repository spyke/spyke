"""Spike detection algorithms

TODO: use median based noise estimation instead of std based
      - estimate noise level dynamically with sliding window
        and independently for each channel

TODO: spatiotemporal lockout:
      - do spatial lock out only during first 1/2 phase of trigger spike
      - phases are part of same spike if less than 250us between each other

TODO: for speed (esp comparing signal to thresh), consider converting all uV data
      from 64bit float to 16 bit integer (actually, just keep it as 16 bit to begin
      with)

DONE: Might need to use scipy.weave or some other low-level code
      (cython?) to make these algorithms fast enough, while allowing you to
      step through one timepoint at a time, which numpy might not let you do
      easily...

DONE: add a method to search forward until you find the next spike on any or on a
      specific set of channels, as opposed to searching forward over a known fixed
      trange. This would make testing easier too
"""

from __future__ import division

__authors__ = ['Reza Lotun, Martin Spacek']

import itertools
import sys
import time

import numpy as np
from numpy import where
from scipy import weave

import spyke.surf
from spyke.core import WaveForm, toiter, cut, intround, eucd
#from spyke import Spike, Template, Collection



class Detector(object):
    """Spike detector base class"""
    DEFAULTTHRESHMETHOD = 'median'
    DEFTLOCK = 250 # us
    DEFSLOCK = 175 # um

    MAXAVGFIRINGRATE = 1000 # Hz, assume no chan will trigger more than this rate of events on average within a block. TODO: should be a property
    BLOCKSIZE = 1000000 # waveform data block size, us. TODO: should be a property
    MAXNSPIKETISPERCHAN = BLOCKSIZE/1000000 * MAXAVGFIRINGRATE # num elements per chan to preallocate before searching a block
    BLOCKEXCESS = 1000 # us, extra data as buffer at start and end of a block while searching for spikes. Only useful for ensuring spike times within the actual block time range are accurate. Spikes detected in the excess are discarded


    def __init__(self, stream, chans=None, trange=None, maxnspikes=None,
                 tlock=None, slock=None):
        """Takes a data stream and sets various parameters"""
        self.stream = stream
        if chans == None:
            chans = range(self.stream.nchans) # search all channels
        self.chans = toiter(chans) # need not be contiguous. TODO: this should really be a property!!!! Update self.dm on change
        self.chans.sort() # make sure they're in order
        self.dm = self.get_distance_matrix() # Euclidean channel distance matrix, in self.chans order
        if trange == None:
            trange = (stream.t0, stream.tend)
        self.trange = trange
        if maxnspikes == None:
            maxnspikes = sys.maxint
        self.maxnspikes = maxnspikes # return at most this many spikes, applies across chans
        if tlock == None:
            tlock = self.DEFTLOCK
        self.tlock = tlock
        if slock == None:
            slock = self.DEFSLOCK
        self.slock = slock

    def get_distance_matrix(self):
        """Get channel distance matrix, in um"""
        sl = self.stream.probe.SiteLoc
        coords = []
        for chan in self.chans:
            coords.append(sl[chan])
        return eucd(coords)
    '''
    def __iter__(self):
        """Returns an iterator object. Called in for loops and in 'in' statements, and by the iter() f'n?.
        This is here to allow you to treat any of the detection classes directly as iterators"""
        spikes = iter(self.find())
        # why not just return the iterator, and let the outside world iterate over it, like this:?
        return (chan, spiketime), or maybe a Spike object with .chan and .ts attribs
        #while True:
        #    try:
        #        yield spikes.next()
        #    except StopIteration:
        #        break
    '''
    def get_threshold(self, chan, kind=DEFAULTTHRESHMETHOD):
        """Calculate either median or stdev based threshold for a given chan"""
        if kind == 'median':
            self.get_median_threshold(chan)
        elif  kind == 'stdev':
            self.get_stdev_threshold(chan)

    def get_median_threshold(self, chan):
        return self.get_median_noise(chan) * self.MEDIAN_MULT

    def get_stdev_threshold(self, chan):
        return self.get_stdev_noise(chan) * self.STDEV_MULT

    def get_median_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def get_stdev_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def us2nt(self, us):
        """Convert time in us to nearest number of eq'v timepoints in stream"""
        nt = intround(self.stream.layout.sampfreqperchan * us / 1000000)
        # prevent rounding nt down to 0. This way, even the smallest
        # non-zero us will get you at least 1 timepoint
        if nt == 0 and us != 0:
            nt = 1
        return nt


class FixedThresh(Detector):
    """Base class for fixed threshold spike detection,
    Uses the same single static threshold throughout the entire file,
    with an independent threshold for every channel"""

    STDEV_WINDOW = 10000000 # 10 sec
    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    #SEARCH_SPAN = 1000

    def __init__(self, *args, **kwargs):
        Detector.__init__(self, *args, **kwargs)
        self.thresh = 50 # uV, TODO: calculate this from noise level

    '''
    def setup(self):
        """Used to determine threshold and set initial state"""
        # get stdev for each channel along a STDEV_WINDOW window
        wave = self.stream[self.t0:self.t0 + STDEV_WINDOW]
        self.std = {}
        for chan, d in enumerate(wave.data):
            self.std[chan] = wave.data[chan].std()

        # set the threshold to be STDEV_MULT * standard deviation
        # each chan has a separate thresh
        self.thresholds = {}
        for chan, stdev in self.std.iteritems():
            self.thresholds[chan] = stdev * self.STDEV_MULT

        # spike window: -SPIKE_PRE and +SPIKE_POST around spike, search window will be 1ms
        self.search_span = self.SEARCH_SPAN
        self.curr = self.t0 + self.SEARCH_SPAN # XXX: add an initial jump: TODO: why?
        self.window = self.stream[self.t0:self.t0 + self.search_span]

        self.lockout = self.LOCKOUT

    def yield_events(self, chan_events):
        """TODO: what does this do? need description here"""
        # sort event indices
        chan_events.sort()
        for event_index, chan in chan_events:
            # if the event is firing before our current location
            # then we're in lockout mode and should just continue
            if self.window.ts[event_index] < self.curr:
                continue
            # reposition window for each event
            self.curr = self.window.ts[event_index] - self.SPIKE_PRE
            spike = self.stream[self.curr:self.curr + \
                                self.SPIKE_PRE + self.SPIKE_POST]
            self.curr = self.curr + self.SPIKE_PRE + \
                            self.SPIKE_POST + self.lockout
            #self.window = self.stream[self.curr:self.curr + self.search_span]
            yield Spike(spike, chan, self.window.ts[event_index])
    '''

class DynamicThresh(Detector):
    """Base class for dynamic threshold spike detection,
    Uses varying thresholds throughout the entire file,
    depending on the local noise level

    Calculate noise level using, say, a 50ms sliding window centered on the
    timepoint you're currently testing for a spike. Or, use fixed pos
    windows, pre calc noise for each of them, and take noise level from whichever
    window you happen to be in while checking a timepoint for thresh xing.
    """

    def get_median_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass

    def get_stdev_noise(self, chan):
        """Overriden by FixedThresh and DynamicThresh classes"""
        pass


class BipolarAmplitudeFixedThresh(FixedThresh):
    """Bipolar amplitude fixed threshold detector,
    with fixed temporal lockout on all channels, plus a spatial lockout"""

    def search(self):
        """Search for spikes. Divides large searches into more manageable
        blocks of (slightly overlapping) multichannel waveform data, and
        then combines the results. method = 'all' or 'indep' treats chans
        together or independently"""

        t0 = time.clock()

        # reset this at the start of every search
        self.totalnspikes = 0 # total num spikes found across all chans so far by this Detector

        # holds a channel's spike times, passed by assignment to C code.
        # no need for more than one max every other timepoint, can get away with less to save memory.
        # recordings not likely to have more than 2**32 timestamps, even when interpolated to 50 kHz,
        # so uint32 allows us at least 23 hour long recordings, don't think int64 is needed here
        self.spiketis = np.zeros((2, len(self.chans)*self.MAXNSPIKETISPERCHAN), dtype=np.uint32) # row0: chanii, row1: ti
        print 'init .spiketis took %.3f sec' % (time.clock()-t0)
        self.tilock = self.us2nt(self.tlock) # TODO: this should be a property, or maybe tlock should be

        # generate time ranges for slightly overlapping blocks of data
        wavetranges = []
        cutranges = []
        BS = self.BLOCKSIZE
        BX = self.BLOCKEXCESS
        les = range(self.trange[0], self.trange[1], BS)  # left edges of data blocks
        for le in les:
            wavetranges.append((le-BX, le+BS+BX)) # time range of waveform to give to .searchblock
            cutranges.append((le, le+BS)) # time range of spikes to keep from those returned by .searchblock
        # last wavetrange and cutrange surpass self.trange[1], fix that here:
        wavetranges[-1] = (wavetranges[-1][0], self.trange[1]+BX)
        cutranges[-1] = (cutranges[-1][0], self.trange[1])

        spikeslist = [] # list of 2D spike arrays returned by searchblock, one array per block
        for (lo, hi), cutrange in zip(wavetranges, cutranges): # iterate over blocks
            if self.totalnspikes < self.maxnspikes:
                wave = self.stream[lo:hi] # a block (Waveform) of multichan data
                maxchans, spiketimes = self.searchblock(wave)
                cutspiketimes = cut(spiketimes, cutrange) # remove excess
                # TODO: remove any spikes that happen right at the last timepoint in the file,
                # since we can't say when an interrupted rising edge would've reached peak
                spikeslist.append(np.asarray([maxchans, cutspiketimes]))

        print 'inside .search() took %.3f sec' % (time.clock()-t0)

        return spikeslist

    def searchblock(self, wave):
        """Search across all chans in a manageable block of waveform
        data and return a tuple of maxchan and spike time arrays.
        Apply both temporal and spatial lockouts

        DONE: implement channel(s) specific searches, ie make use of self.chans"""
        t0 = time.clock()
        absdata = np.abs(wave.data[self.chans]) # pull our chans of data out, this assumes wave.data has all possible chans in it
        print 'abs took %.3f sec' % (time.clock()-t0)
        chans = np.asarray(self.chans)
        nchans = len(self.chans)
        nt = wave.data.shape[1]
        totalnspikes = self.totalnspikes # total num of spikes found so far in this Detector.search()
        maxnspikes = self.maxnspikes
        thresh = self.thresh

        xthresh = np.zeros(nchans, dtype=np.int32) # thresh xing flags
        last = np.zeros(nchans) # holds last signal value per chan, floats in uV
        lock = np.zeros(nchans, dtype=np.int32) # holds number of lockout timepoints left per chan

        tilock = self.tilock # temporal lockout, in num timepoints
        slock = self.slock # spatial lockout, in um
        dm = self.dm # Euclidean channel distance matrix
        spiketis = self.spiketis # init'd in self.search()

        """
        TODO: after searching for maxchan, search again for peak on that maxchan,
              maybe iterate a few times until stop seeing changes in result for
              that spike

        TODO: when searching for maxchan, limit search to chans within spatial lockout.
              This will be faster, and will also prevent unrelated distant cells firing
              at the same time from triggering each other, or something

        TODO: instead of searching forward indefinitely for a peak, maybe search
              forward and backwards say 0.5ms. Ah, but then searching backwards would
              ignore any tlock you may have had on that chan in the past. Maybe that means
              having to change how tlock is implemented
        """

        code = r"""
        #line 295 "detect.py" // for debugging
        int spikei, maxchanii, maxchani, chanj;
        double v; // current signal voltage, uV (Python float), using a pointer doesn't seem faster
        for ( int ti=0; ti<nt; ti++ ) { // iterate over all timepoints
            // TODO: chan loop should go in random order on each ti, to prevent a chan from
            // dominating with its spatial lockout or something like that
            for ( int chanii=0; chanii<nchans; chanii++ ) { // iterate over indices into chans
                if ( lock(chanii) > 0 ) // if this chan is still locked out
                    lock(chanii)--; // decr this chan's temporal lockout
                else { // search for a thresh xing or a peak
                    v = absdata(chanii, ti);
                    if ( xthresh(chanii) == 0 ) { // we're looking for a thresh xing
                        if ( v >= thresh ) { // met or exceeded threshold
                            xthresh(chanii) = 1; // set crossed threshold flag for this chan
                            last(chanii) = v; // update last value for this chan, wait til next ti to decide if this is a peak
                        }
                    }
                    else { // xthresh(chanii)==1, in crossed thresh state, now we're look for a peak
                        if ( v > last(chanii) ) // if signal is still increasing
                            last(chanii) = v; // update last value for this chan, wait til next ti to decide if this is a peak
                        else { // signal is decreasing, declare previous ti as a spike timepoint
                            spikei = totalnspikes++; // 0-based spike index. assign, then increment
                            // find max chan, ie chan spike is centered on
                            maxchanii = chanii; // start with assumption that current chan is max chan
                            for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all indices into chans again
                                if ( absdata(chanjj, ti) > absdata(maxchanii, ti) )
                                    maxchanii = chanjj; // update maxchanii
                            }
                            spiketis(0, spikei) = maxchanii; // store chan spike is centered on
                            spiketis(1, spikei) = ti-1; // save previous time index as that of the spikei'th spike
                            // apply spatial and temporal lockouts
                            for ( int chanjj=0; chanjj<nchans; chanjj++ ) { // iterate over all indices into chans again
                                maxchani = chans(maxchanii);
                                chanj = chans(chanjj);
                                if ( dm(maxchani, chanj) <= slock ) { // chanjj is within spatial lockout in um
                                    xthresh(chanjj) = 0; // clear its threshx flag
                                    last(chanjj) = 0.0; // reset last so it's ready when it comes out of lockout
                                    // apply its temporal lockout
                                    if ( chanjj > chanii ) // we haven't encountered chanjj yet in the chanii loop
                                        lock(chanjj) = tilock+1; // lockout by one extra timepoint which it'll
                                                                 // then promptly decr before we leave this ti
                                    else
                                        lock(chanjj) = tilock;
                                }
                            }
                            if ( totalnspikes >= maxnspikes ) {
                                return_val = totalnspikes;
                                return return_val; // exit here, don't search any more timepoints
                            }
                            // don't break out of chanii loop to move to next ti: there may be other
                            // chans with spikes that are outside the spatial lockout.
                        }
                    }
                }
            }
        }
        return_val = totalnspikes;
        """
        tCloop = time.clock()
        totalnspikes = weave.inline(code, ['absdata', 'chans', 'nchans', 'nt',
                                           'totalnspikes', 'maxnspikes',
                                           'thresh', 'xthresh', 'last', 'lock',
                                           'tilock', 'slock', 'dm', 'spiketis'],
                                    type_converters=weave.converters.blitz,
                                    compiler='gcc')
        print 'C loop took %.3f sec' % (time.clock()-tCloop)
        nnewspikes = totalnspikes - self.totalnspikes
        self.totalnspikes = totalnspikes # update
        spiketis = spiketis[:, :nnewspikes] # keep only the entries that were filled
        maxchaniis = spiketis[0] # chanii that each spike occurred on
        maxchans = np.asarray(self.chans)[maxchaniis] # convert from indices to actual chan ids
        spiketimes = wave.ts[spiketis[1]] # convert from indices to actual spike times, wave.ts is int64 array
        print 'inside .searchblock() took %.3f sec' % (time.clock()-t0)
        return (maxchans, spiketimes)

    def searchblock_indepchans(self, wave):
        """Search across all chans in a manageable block of waveform
        data, searching each chan independently in its own C loop.
        Return a dict of arrays of spike times, one entry per chan.
        This is slightly faster than searchblock only because the searchchan
        C loop used here skips over timepoints to do its temporal lockout, while
        searchblock checks every timepoint. The latter method is better for
        doing spatial lockout at the same time"""
        spikes = {}
        for chan in self.chans:
            if self.totalnspikes < self.maxnspikes:
                abschan = np.abs(wave[chan])
                tis = self.searchchan(abschan)
                spikes[chan] = wave.ts[tis] # spike times in us
        # TODO: apply spatial lockout here. Would have to iterate over all
        # timepoints again, which would be slow
        return spikes

    def searchchan(self, abschan):
        """Search a single chan of absval data for thresh xings, apply temporal lockout.
        If this crashes, it might be possible that self.spiketis was init'd too small"""
        nt = len(abschan)
        totalnspikes = self.totalnspikes # total num of spikes found by this Detector so far
        maxnspikes = self.maxnspikes
        thresh = self.thresh
        tilock = self.tilock
        spiketis = self.spiketis # init'd in self.search()

        code = r"""
        #line 399 "detect.py" // for debugging
        double last=0.0; // last signal value, uV
        int ti=0; // current time index
        while ( ti<nt && totalnspikes < maxnspikes ) { // enforce maxnspikes across single chan
            if ( abschan(ti) >= thresh ) { // if we've exceeded threshold
                while ( abschan(ti) > last ) { // while signal is still increasing
                    last = abschan(ti); // update last
                    ti++; // go to next timepoint
                }
                // now signal is decreasing, save last timepoint as spike
                spiketis(totalnspikes++ - 1) = ti-1; // 0-based index into spiketis
                last = 0.0; // reset for search for next spike
                ti += tilock+1; // skip forward one temporal lockout, and go to next timepoint
            }
            else
                ti++; // no thresh xing, go to next timepoint
        }
        return_val = totalnspikes;
        """
        totalnspikes = weave.inline(code, ['abschan', 'nt', 'totalnspikes', 'maxnspikes', 'thresh', 'tilock', 'spiketis'],
                                    type_converters=weave.converters.blitz,
                                    compiler='gcc')
        nnewspikes = totalnspikes - self.totalnspikes
        self.totalnspikes = totalnspikes # update
        return spiketis[:nnewspikes]


class MultiPhasic(FixedThresh):
    """Multiphasic filter - spikes triggered only when consecutive
    thresholds of opposite polarity occur on a given channel within
    a specified time window delta_t

    That is, either:

        1) s_i(t) > f and s_i(t + t') < -f
        2) s_i(t) < -f and s_it(t + t') > f

    for 0 < t' <= delta_t
    """

    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000
    delta_t = 300

    def find(self):
        """Maintain state and search forward for a spike"""

        # keep on sliding our search window forward to find spikes
        while True:

            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = where(numpy.abs(self.window.data[chan]) > thresh)[0]

                if len(_ev) <= 0:
                    continue

                thresh_vals = [(self.window.data[chan][ind], ind) \
                                    for ind in _ev.tolist()]
                # for each threshold value, scan forwrd in time delta_t
                # to see if an opposite threshold crossing occurred
                for i, tup in enumerate(thresh_vals):
                    val, ind = tup
                    sgn = numpy.sign(val)
                    t = self.window.ts[ind]
                    for cand_val, t_ind in thresh_vals[i + 1:]:
                        # check ahead only with delt_t
                        if self.window.ts[t_ind] - t > self.delta_t:
                            break
                        cand_sgn = numpy.sign(cand_val)
                        # check if threshold crossings are opposite
                        # polarity
                        if cand_sgn != sgn:
                            chan_events.append((ind, chan))
                            break

                for evt in self.yield_events(chan_events):
                    yield evt

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]


class DynamicMultiPhasic(FixedThresh):
    """Dynamic Multiphasic filter - spikes triggered only when consecutive
    thresholds of opposite polarity occured on a given channel within
    a specified time window delta_t, where the second threshold level is
    determined relative to the amplitude of the waveform peak/valley
    following initial phase trigger

    That is, either:

        1) s_i(t) > f and s_i(t + t') < f_pk - f'
    or  2) s_i(t) < -f and s_it(t + t') > f_val + f'

    for -delta_t < t' <= delta_t
    and where f' is the minimum amplitdude inflection in delta_t
    """

    STDEV_MULT = 4
    SPIKE_PRE = 250
    SPIKE_POST = 750
    SEARCH_SPAN = 1000
    LOCKOUT = 1000
    delta_t = 300

    def setup(self):
        FixedThreshold.setup(self)
        self.f_inflect = {}
        # set f' to be 3.5 * standard deviation (see paper)
        for chan, val in self.std.iteritems():
            self.f_inflect[chan] = 3.5 * val

    def find(self):
        """Maintain state and search forward for a spike"""

        # keep on sliding our search window forward to find spikes
        while True:

            # check if we have a channel firing above threshold
            chan_events = []
            for chan, thresh in self.thresholds.iteritems():
                # this will only be along one dimension
                _ev = where(numpy.abs(self.window.data[chan]) > thresh)[0]

                if len(_ev) <= 0:
                    continue

                thresh_vals = [(self.window.data[chan][ind], ind) \
                                    for ind in _ev.tolist()]

                # for each threshold value, scan forwrd in time delta_t
                # to see if an opposite threshold crossing occurred
                for val, ind in thresh_vals:

                    # scan forward to find local max or local min
                    extremal_ind = ind
                    extremal_val = val
                    #while True:
                    #    next_ind = extremal_ind + 1
                    #    next_val = self.window.data[chan][next_ind]
                    #    if abs(next_val) < abs(extremal_val):
                    #        break
                    #    extremal_val, extremal_ind = next_val, next_ind

                    # calculate our dynamic threshold
                    # TODO: make this more compact
                    if extremal_val < 0:
                        # a valley
                        dyn_thresh = extremal_val + self.f_inflect[chan]
                        dyn_events = where(self.window.data[chan] \
                                                        > dyn_thresh)[0]
                    else:
                        # a peak
                        dyn_thresh = extremal_val - self.f_inflect[chan]
                        dyn_events = where(self.window.data[chan] \
                                                        < dyn_thresh)[0]

                    dyn_vals = [(self.window.data[chan][_ind], _ind) \
                                    for _ind in dyn_events.tolist()]
                    t = self.window.ts[extremal_ind]
                    # check for next inflection
                    for dyn_val, t_ind in dyn_vals:
                        # check ahead only within +/- delta_t
                        t_prime = self.window.ts[t_ind]
                        if (t_prime > t - self.delta_t) and \
                                (t_prime <= t + self.delta_t):
                            break

                        event_val = extremal_val
                        event_ind = extremal_ind
                        if abs(dyn_val) > abs(extremal_val):
                            event_val = dyn_val
                            event_ind = t_ind
                        chan_events.append((event_ind, chan))
                        break

                # yield all the events we've found
                for evt in self.yield_events(chan_events):
                    yield evt

            self.curr += self.search_span
            self.window = self.stream[self.curr:self.curr + self.search_span]
