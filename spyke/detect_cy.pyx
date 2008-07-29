"""Event detection loops implemented in Cython for speed"""

from __future__ import division

__author__ = 'Martin Spacek'

cdef extern from "stdio.h":
    int printf(char *, ...)

include "Python.pxi" # include from the python headers
include "numpy.pxi" # include the Numpy C API for use via Cython extension code
import_array() # initialize numpy - this MUST be done before any other code is executed.

import numpy as np
import time
import wx


DEF INF = float('+inf')


# struct of relevant detector vars that need to be passed around to functions
ctypedef struct det_vars_t:
    int maxchanii
    int *chansp
    int nchans
    int ndmchans
    double *dmp
    double slock
    int tilock
    float *absdatap
    long long *tsp
    int *xthreshp
    float *lastp
    int *lockp
    int nt
    int ti


cpdef class BipolarAmplitudeFixedThresh_Cy:

    cpdef searchblock(self, wave, cutrange, int maxnevents):
        """Search one timepoint at a time across chans in a manageable
        block of waveform data and return a 2D array of event times and maxchans.
        Apply both temporal and spatial lockouts.
        cutrange: determines which events are saved and which are discarded as excess
        maxnevents: maximum number of events to return while searching this block

        TODO: after searching for maxchan, search again for peak on that maxchan,
              maybe iterate a few times until stop seeing changes in result for
              that event. But iteration isn't really possible, because we have our
              big ti loop to obey and its associated t and s locks.
              Maybe this is just an inherent limit to simple thresh detector?

        TODO: instead of searching forward indefinitely for a peak, maybe search
              forward and backwards say 0.5ms. Ah, but then searching backwards would
              ignore any tlock you may have had on that chan in the past. Maybe that means
              having to change how tlock is implemented

        TODO: (maybe): chanii loop should go in random order on each ti, to prevent a chan from
              dominating with its spatial lockout or something like that. So, shuffle chans in-place
              on every ti, or have a second level of chan indices, shuffle those every ti, and iterate over chaniii

        TODO: replace lockp with binary bitmask of length 1ms say,
              shift bits on every ti, might let you search for events into future
              semi-independently on each chan

        """
        cdef det_vars_t s # s for struct (or search, or settings)

        if not wave.data.flags.contiguous:
            #print "wave.data ain't contig, strides:", wave.data.strides
            wave.data = wave.data.copy() # make it contiguous for easy pointer indexing
        if not wave.ts.flags.contiguous:
            #print "wave.ts ain't contig, strides:", wave.ts.strides
            wave.ts = wave.ts.copy()

        s.nchans = len(self.chans)
        cdef ndarray chans = np.asarray(self.chans)
        s.chansp = <int *>chans.data # int pointer to .data field

        if self.chans != range(s.nchans): # if self.chans is not contiguous
            #ttake = time.clock()
            # pull our chans of data out, this assumes wave.data has all possible chans in it, which it should
            data = wave.data.take(chans, axis=0) # returns a contiguous copy of data
            #print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # save time by avoiding an unnecessary .take
            data = wave.data
        cdef ndarray absdata = data # name it absdata
        arrabs(absdata) # now do the actual abs, in-place, about 2x faster than np.abs
        s.absdatap = <float *>absdata.data # float pointer to absdata .data field, rows correspond to chans in chansp

        cdef ndarray ts = wave.ts
        s.tsp = <long long *>ts.data # long long pointer to timestamp .data
        cdef long long eventt # holds current event timestamp

        #assert s.nchans == absdata.dimensions[0] # yup
        s.nt = absdata.dimensions[1]
        cdef float fixedthresh = self.fixedthresh

        # cut times, these are for testing whether to inc nevents
        cdef long long cut0 = cutrange[0]
        cdef long long cut1 = cutrange[1]
        if cut0 > cut1: # swap 'em for the test
            cut0, cut1 = cut1, cut0

        cdef ndarray xthresh = np.zeros(s.nchans, dtype=int) # per-channel thresh xing flags (0 or 1)
        s.xthreshp = <int *>xthresh.data # int pointer to .data field
        cdef ndarray lock = np.zeros(s.nchans, dtype=int) # per-channel lockout timepoint counters, >= 0 indicates locked out
        s.lockp = <int *>lock.data # int pointer to .data field
        cdef ndarray last = np.zeros(s.nchans, dtype=np.float32) # holds last signal value per chan, floats in uV
        s.lastp = <float *>last.data # float pointer to .data field

        s.tilock = self.tilock # temporal index lockout, in num timepoints
        s.slock = self.slock # spatial lockout, in um
        cdef ndarray dm = self.dm # full Euclidean channel distance matrix for all possible chanis, floats in um
        s.dmp = <double *>dm.data # double pointer to .data field
        s.ndmchans = len(self.dm) # number of all possible chans in dm

        cdef ndarray eventtimes = self._eventtimes # init'd in self.search()
        cdef ndarray maxchans = self._maxchans # init'd in self.search()
        cdef long long *eventtimesp = <long long *>eventtimes.data # long long pointer to .data field
        cdef int *maxchansp = <int *>maxchans.data # int pointer to .data field

        cdef int ti, chanii
        cdef int nevents = 0 # num non-excess events found so far while searching this block
        cdef int eventi = -1 # index into eventtimes
        cdef float v # current signal voltage, uV

        #tcyloop = time.clock()
        for ti from 0 <= ti < s.nt: # iterate over all timepoints
            # TODO: shuffle chans in-place here
            for chanii from 0 <= chanii < s.nchans:
                s.lockp[chanii] -= 1 # decr all chans' lockout counters
            for chanii from 0 <= chanii < s.nchans: # iterate over indices into chans
                if s.lockp[chanii] < 0: # if this chan isn't locked out, search for a thresh xing or a peak
                    v = s.absdatap[chanii*s.nt + ti] # (absdata[chanii, ti])
                    if s.xthreshp[chanii] == 0: # we're looking for a thresh xing
                        if v >= fixedthresh: # met or exceeded threshold
                            #print 't: %d, thresh xing, chan: %d' % (s.tsp[ti], chanii)
                            s.xthreshp[chanii] = 1 # set maxchan's crossed threshold flag
                            s.lastp[chanii] = v # update maxchan's last value
                    else: # s.xthresh[chanii] == 1, in crossed thresh state, now we're look for a peak
                        if v > s.lastp[chanii]: # if signal is still increasing
                            s.lastp[chanii] = v # update last value for this chan, continue searching for peak
                        else: # signal is decreasing, declare previous ti as an event timepoint
                            eventt = s.tsp[ti-1] # event time is timestamp of previous time index
                            # find maxchanii within slock of chanii, start with current chanii as maxchanii, pass previous ti
                            find_maxchanii(s, chanii, ti-1)
                            #print 't: %d, found event at t=%d on chanii %d' % (s.tsp[ti], eventt, s.maxchanii)
                            # apply spatiotemporal lockout now that event has been found, pass tilock relative to previous ti
                            set_lockout(s, s.tilock-1)
                            if cut0 <= eventt and eventt <= cut1: # event falls within cutrange, save it
                                eventi += 1
                                eventtimesp[eventi] = eventt # save event time
                                maxchansp[eventi] = s.chansp[s.maxchanii] # save maxchani that event is centered on
                                nevents += 1 # this event has been saved, so inc
                            if nevents >= maxnevents: # exit here, don't search any more chans
                                ti = s.nt # TODO: nasty hack to get out of outer ti loop
                                break # out of inner chanii loop
        #print 'cy loop took %.3f sec' % (time.clock()-tcyloop)
        eventtimes = eventtimes[:nevents] # keep only the entries that were filled
        maxchans = maxchans[:nevents] # keep only the entries that were filled
        return np.asarray([eventtimes, maxchans])


cpdef class DynamicMultiphasicFixedThresh_Cy:

    cpdef searchblock(self, wave, cutrange, int maxnevents):
        """Search one timepoint at a time across chans in a manageable
        block of waveform data and return a 2D array of event times and maxchans.
        Apply both temporal and spatial lockouts.
        cutrange: determines which events are saved and which are discarded as excess
        maxnevents: maximum number of events to return while searching this block
        """
        cdef det_vars_t s # s for struct (or search, or settings)

        if not wave.data.flags.contiguous:
            #print "wave.data ain't contig, strides:", wave.data.strides
            wave.data = wave.data.copy() # make it contiguous for easy pointer indexing
        if not wave.ts.flags.contiguous:
            #print "wave.ts ain't contig, strides:", wave.ts.strides
            wave.ts = wave.ts.copy()

        s.nchans = len(self.chans)
        cdef ndarray chans = np.asarray(self.chans)
        s.chansp = <int *>chans.data # int pointer to .data field

        if self.chans != range(s.nchans): # if self.chans is not contiguous
            #ttake = time.clock()
            # pull our chans of data out, this assumes wave.data has all possible chans in it, which it should
            data = wave.data.take(chans, axis=0) # returns a contiguous copy of data
            #print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # save time by avoiding an unnecessary .take
            data = wave.data
        cdef ndarray absdata = data # name it absdata
        arrabs(absdata) # now do the actual abs, in-place, about 2x faster than np.abs
        s.absdatap = <float *>absdata.data # float pointer to absdata .data field, rows correspond to chans in chansp

        cdef ndarray ts = wave.ts
        s.tsp = <long long *>ts.data # long long pointer to timestamp .data
        cdef long long eventt # holds current event timestamp

        #assert s.nchans == absdata.dimensions[0] # yup
        s.nt = absdata.dimensions[1]
        cdef float fixedthresh = self.fixedthresh

        # cut times, these are for testing whether to inc nevents
        cdef long long cut0 = cutrange[0]
        cdef long long cut1 = cutrange[1]
        if cut0 > cut1: # swap 'em for the test
            cut0, cut1 = cut1, cut0

        cdef ndarray xthresh = np.zeros(s.nchans, dtype=int) # per-channel thresh xing flags (0 or 1)
        s.xthreshp = <int *>xthresh.data # int pointer to .data field
        cdef ndarray lock = np.zeros(s.nchans, dtype=int) # per-channel lockout timepoints (end inclusive?)
        s.lockp = <int *>lock.data # int pointer to .data field
        cdef ndarray last = np.zeros(s.nchans, dtype=np.float32) # holds last signal value per chan, floats in uV
        s.lastp = <float *>last.data # float pointer to .data field

        s.tilock = self.tilock # temporal index lockout, in num timepoints
        s.slock = self.slock # spatial lockout, in um
        cdef ndarray dm = self.dm # full Euclidean channel distance matrix for all possible chanis, floats in um
        s.dmp = <double *>dm.data # double pointer to .data field
        s.ndmchans = len(self.dm) # number of all possible chans in dm

        cdef ndarray eventtimes = self._eventtimes # init'd in self.search()
        cdef ndarray maxchans = self._maxchans # init'd in self.search()
        cdef long long *eventtimesp = <long long *>eventtimes.data # long long pointer to .data field
        cdef int *maxchansp = <int *>maxchans.data # int pointer to .data field

        cdef int ti, chanii
        cdef int nevents = 0 # num non-excess events found so far while searching this block
        cdef int eventi = -1 # index into eventtimes
        cdef float v # current signal voltage, uV

        #tcyloop = time.clock()
        for ti from 0 <= ti < s.nt: # iterate over all timepoints
            # TODO: shuffle chans in-place here
            for chanii from 0 <= chanii < s.nchans: # iterate over indices into chans
                # check if chan is eligible and we've met or exceeded threshold
                if ti <= s.lockp[chanii] or s.absdatap[chanii*s.nt + ti] < fixedthresh:
                    continue # no, skip to next chan in chan loop
                # find maxchan at timepoint ti by searching across eligible chans, center search on chanii
                find_maxchanii2(s, chanii, ti)
                # search forward indefinitely for the first peak, this will be used as the event time
                sign = get_sign(data[s.maxchanii*s.nt + ti]) # find whether thresh xing was +ve or -ve
                eventti = find_peak(s, ti, 2**31, sign) # search forward indefinitely for peak of correct phase
                if eventti == -1: # couldn't find a peak
                    continue # skip to next chan in chan loop
                find_maxchanii2(s, s.maxchanii, eventti) # update maxchan one last time for this putative event
                # search forward one tlock on the maxchan for another peak of opposite phase
                # that is 2*thresh greater (in the right direction) than the previous peak
                sign = -sign
                peak2ti = find_peak(s, eventti, s.tilock, sign)
                if peak2ti == -1 # couldn't find a 2nd peak
                    continue # skip to next chan in chan loop
                if fabs(data[s.maxchanii*s.nt + eventi] - data[s.maxchanii*s.nt + peak2ti]) < 2*fixedthresh:
                    continue # 2nd peak isn't big enough, skip to next chan in chan loop
                # if we get this far, it's a valid event
                eventt = s.tsp[eventti] # event time
                if cut0 <= eventt and eventt <= cut1: # event falls within cutrange, save it
                    eventi += 1
                    eventtimesp[eventi] = eventt # save event time
                    maxchansp[eventi] = s.chansp[s.maxchanii] # save maxchani that event is centered on
                    nevents += 1 # this event has been saved, so inc
                if nevents >= maxnevents: # exit here, don't search any more chans
                    ti = s.nt # TODO: nasty hack to get out of outer ti loop
                    break # out of inner chanii loop
                # to set lockout, continue searching forward for consecutive peaks of
                # alternating phase within tlock of each other
                lastpeakti = peak2ti
                while True: # do Python booleans translate to fast C?
                    sign = -sign
                    peakti = find_peak(s, lastpeakti, s.tilock, sign)
                    if peakti == -1:
                        break # out of while loop
                    lastpeakti = peakti # update
                set_lockout2(s, lastpeakti)
        #print 'cy loop took %.3f sec' % (time.clock()-tcyloop)
        eventtimes = eventtimes[:nevents] # keep only the entries that were filled
        maxchans = maxchans[:nevents] # keep only the entries that were filled
        return np.asarray([eventtimes, maxchans])


cdef int find_maxchanii(det_vars_t s, int maxchanii, int ti):
    """Update s.maxchanii at timepoint ti over non locked-out chans within slock of maxchanii
    TODO: this could be applied over and over, recentering the search, until maxchanii stops changing,
          although that would probably be a very minor refinement at a rather high cost, and might
          start to creep onto other events nearby in space and time
    """
    cdef int chanjj, chanj, maxchani
    for chanjj from 0 <= chanjj < s.nchans: # iterate over all chan indices
        maxchani = s.chansp[maxchanii] # dereference to index into dmp
        chanj = s.chansp[chanjj] # dereference to index into dmp
        # if chanj is within slock of maxchani and has higher signal and isn't locked out:
        if s.dmp[maxchani*s.ndmchans + chanj] <= s.slock and \
            s.absdatap[chanjj*s.nt + ti] > s.absdatap[maxchanii*s.nt + ti] and \
            s.lockp[chanjj] < 0:
            maxchanii = chanjj # update
    s.maxchanii = maxchanii # store it

cdef int find_maxchanii2(det_vars_t s, int maxchanii, int ti):
    """Update s.maxchanii at timepoint ti over non locked-out chans within slock of maxchanii
    TODO: this could be applied over and over, recentering the search, until maxchanii stops changing,
          although that would probably be a very minor refinement at a rather high cost, and might
          start to creep onto other events nearby in space and time
    """
    cdef int chanjj, chanj, maxchani
    for chanjj from 0 <= chanjj < s.nchans: # iterate over all chan indices
        maxchani = s.chansp[maxchanii] # dereference to index into dmp
        chanj = s.chansp[chanjj] # dereference to index into dmp
        # if chanj is within slock of maxchani and has higher signal and isn't locked out:
        if s.dmp[maxchani*s.ndmchans + chanj] <= s.slock and \
            s.absdatap[chanjj*s.nt + ti] > s.absdatap[maxchanii*s.nt + ti] and \
            s.lockp[chanjj] < ti:
            maxchanii = chanjj # update
    s.maxchanii = maxchanii # store it

cdef int get_sign(float val):
    """Return the sign of the float input"""
    if val < 0.0:
        return -1
    else
        return 1

cdef int find_peak(det_vars_t s, int ti, int tirange, int sign):
    """Return timepoint index of first peak (if sign == 1) or valley (if sign == -1)
    on s.maxchanii searching forward up to tirange from passed ti.
    Return -1 if no peak is found"""
    peaktj = -1
    offset = s.maxchanii*s.nt # this is a constant during the loop
    last = -INF * sign # uV
    for tj from ti <= tj < (ti + tirange):
        # if signal is becoming more -ve (sign == -1) or +ve (sign == 1)
        if data[offset + tj] * sign > last * sign:
            peaktj = tj # update
            last = data[offset + tj] # update
        else: # signal is becoming less -ve (sign == -1) or +ve (sign == 1)
            return peaktj
    return -1 # a peak was never found
    #return peaktj # a peak was never found, or return last tj in loop as the peaktj

cdef set_lockout(det_vars_t s, int tilock):
    """Apply spatiotemporal lockout centered on s.maxchanii from current ti forward"""
    cdef int chanjj, chanj, maxchani
    for chanjj from 0 <= chanjj < s.nchans: # iterate over all chan indices
        maxchani = s.chansp[s.maxchanii] # dereference to index into dmp
        chanj = s.chansp[chanjj] # dereference to index into dmp
        if s.dmp[maxchani*s.ndmchans + chanj] <= s.slock: # (dm[maxchani, chanj]) chanjj is within spatial lockout in um
            s.xthreshp[chanjj] = 0 # clear its threshx flag
            s.lastp[chanjj] = 0.0 # reset last so it's ready when it comes out of lockout
            s.lockp[chanjj] = tilock # apply its temporal lockout, use passed tilock, not s.tilock

cdef set_lockout2(det_vars_t s, int ti):
    """Apply spatiotemporal lockout centered on s.maxchanii, lock out all chans
    within slock up to timepoint ti"""
    cdef int chanjj, chanj, maxchani
    for chanjj from 0 <= chanjj < s.nchans: # iterate over all chan indices
        maxchani = s.chansp[s.maxchanii] # dereference to index into dmp
        chanj = s.chansp[chanjj] # dereference to index into dmp
        if s.dmp[maxchani*s.ndmchans + chanj] <= s.slock: # chanjj is within spatial lockout in um
            s.lockp[chanjj] = ti # lock it out up to ti

cdef arrabs(ndarray a):
    """In-place absolute value of 2D float32 array.
    Or, might be better to just use fabs from C math library?"""
    cdef int nrows = a.dimensions[0]
    cdef int ncols = a.dimensions[1]
    cdef float *datap = <float *>a.data
    cdef int i, j
    for i from 0 <= i < nrows:
        for j from 0 <= j < ncols:
            if datap[i*ncols + j] < 0.0:
                datap[i*ncols + j] *= -1.0 # modify in-place

cdef abs(int x):
    """Absolute value of an integer"""
    if x < 0:
        x *= -1
    return x

cdef fabs(float x):
    """Absolute value of a float
    Untested"""
    if x < 0.0:
        x *= -1.0
    return x
