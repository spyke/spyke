"""Event detection loops implemented in Cython for speed"""

from __future__ import division

__author__ = 'Martin Spacek'

cdef extern from "stdio.h":
    int printf(char *, ...)

include "Python.pxi" # include from the python headers
include "numpy.pxi" # include the Numpy C API for use via Cython extension code
import_array() # initialize numpy - this MUST be done before any other code is executed.

#import sys
import time
import numpy as np

#DEF INF = float('+inf') # doesn't work on Windows until Python 2.6+
DEF INF = float(2**128)
#DEF MAXINT = int(sys.maxint) # declare it here to prevent Python name lookups, doesn't work
#DEF MAXINT = int(2**31-1)
DEF DEFTIRANGE = 1000 # more reasonable to use this than MAXINT, won't overflow when added to an int32

# struct of relevant detector vars that need to be passed around to functions
cdef struct Settings:
    int maxchanii
    int *chansp
    int nchans
    int ndmchans
    double *dmp
    double slock
    int tilock
    float *datap
    long long *tsp
    int *lockp
    int nt


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
        cdef Settings s # has to be passed by reference

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
            contigdata = wave.data.take(chans, axis=0) # returns a contiguous copy of data
            #print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # save time by avoiding an unnecessary .take
            contigdata = wave.data
        cdef ndarray data = contigdata
        s.datap = <float *>data.data # float pointer to data's .data field, rows correspond to chans in chansp

        cdef ndarray ts = wave.ts
        s.tsp = <long long *>ts.data # long long pointer to timestamp .data
        cdef long long eventt # holds current event timestamp

        assert s.nchans == data.dimensions[0] # yup
        s.nt = data.dimensions[1]
        cdef float fixedthresh = self.fixedthresh

        # cut times, these are for testing whether to inc nevents
        cdef long long cut0 = cutrange[0]
        cdef long long cut1 = cutrange[1]
        if cut0 > cut1: # swap 'em for the test
            cut0, cut1 = cut1, cut0

        cdef ndarray lock = np.zeros(s.nchans, dtype=int) # per-channel lockout timepoint indices (end inclusive?)
        s.lockp = <int *>lock.data # int pointer to .data field

        s.tilock = self.tilock # temporal index lockout, in num timepoints
        s.slock = self.slock # spatial lockout, in um
        cdef ndarray dm = self.dm # full Euclidean channel distance matrix for all possible chanis, floats in um
        s.dmp = <double *>dm.data # double pointer to .data field
        s.ndmchans = len(self.dm) # number of all possible chans in dm

        cdef ndarray eventtimes = self._eventtimes # init'd in self.search()
        cdef long long *eventtimesp = <long long *>eventtimes.data # long long pointer to .data field
        cdef ndarray maxchans = self._maxchans # init'd in self.search()
        cdef int *maxchansp = <int *>maxchans.data # int pointer to .data field

        cdef int ti, prevti, chanii, peakti, lastpeakti
        cdef float sign
        cdef int nevents = 0 # num non-excess events found so far while searching this block
        cdef int eventi = -1 # event index (indexes into .eventtimes)
        cdef int eventti # event time index (indexes into .data and .ts)

        #tcyloop = time.clock()
        for ti from 0 <= ti < s.nt: # iterate over all timepoints
            prevti = max(ti-1, 0) # previous timepoint, ensuring to go no earlier than first timepoint
            # TODO: shuffle chans in-place here
            for chanii from 0 <= chanii < s.nchans: # iterate over indices into chans
                # check that chan is free at previous timepoint and that we've gone from below to at or above threshold
                if not ( s.lockp[chanii] < prevti and abs(s.datap[chanii*s.nt + prevti]) < fixedthresh \
                         and abs(s.datap[chanii*s.nt + ti]) >= fixedthresh ):
                    continue # no, skip to next chan in chan loop
                sign = get_sign(s.datap[chanii*s.nt + ti]) # find whether thresh xing was +ve or -ve
                # find maxchan at timepoint ti by searching across eligible chans, center search on chanii
                find_maxchanii(&s, chanii, ti, sign)
                # search forward indefinitely for the first peak, this will be used as the event time
                eventti = find_peak(&s, ti, DEFTIRANGE, sign) # search forward almost indefinitely for peak of correct phase
                if eventti == -1: # couldn't find a peak
                    continue # skip to next chan in chan loop
                # if we get this far, it's a valid event
                find_maxchanii(&s, s.maxchanii, eventti, sign) # update maxchan one last time for this event
                eventti = find_peak(&s, eventti, s.tilock, sign) # update eventti for this maxchan
                eventt = s.tsp[eventti] # event time
                #print 'ti=%s, t=%s, chanii=%s, sign=%d, eventt=%s, maxchanii=%s' % (ti, s.tsp[ti], chanii, sign, eventt, s.maxchanii)
                if cut0 <= eventt and eventt <= cut1: # event falls within cutrange, save it
                    eventi += 1
                    eventtimesp[eventi] = eventt # save event time
                    maxchansp[eventi] = s.chansp[s.maxchanii] # save maxchani that event is centered on
                    nevents += 1 # this event has been saved, so inc
                if nevents >= maxnevents: # exit here, don't search any more chans
                    ti = s.nt # TODO: nasty hack to get out of outer ti loop
                    break # out of inner chanii loop
                # to set lockout, continue searching forward for consecutive threshold-exceeding peaks
                # of alternating phase within tlock of each other that are conceivably
                # part of the same spike, yet large enough and far enough away in time to trigger an
                # unwanted threshold crossing
                # ch31 t=40880 is a corner case where this fails badly, can probably only be fixed by using multiphasic detection
                # ch46 t=68420 is another tough case centered on a smaller chan with a premature peak due to distorted signal
                lastpeakti = eventti
                while True:
                    sign = -sign
                    peakti = find_peak(&s, lastpeakti, s.tilock, sign)
                    if peakti == -1 or abs(s.datap[s.maxchanii*s.nt + peakti]) < fixedthresh:
                        # no peak found, or found peak doesn't meet or exceed thresh
                        break # out of while loop
                    lastpeakti = peakti # update
                set_lockout(&s, lastpeakti+s.tilock) # lock out up to peak of last spike phase plus another tlock
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
        cdef Settings s # has to be passed by reference

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
            contigdata = wave.data.take(chans, axis=0) # returns a contiguous copy of data
            #print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # save time by avoiding an unnecessary .take
            contigdata = wave.data
        cdef ndarray data = contigdata
        s.datap = <float *>data.data # float pointer to data's .data field, rows correspond to chans in chansp

        cdef ndarray ts = wave.ts
        s.tsp = <long long *>ts.data # long long pointer to timestamp .data
        cdef long long eventt # holds current event timestamp

        assert s.nchans == data.dimensions[0] # yup
        s.nt = data.dimensions[1]
        cdef float fixedthresh = self.fixedthresh

        # cut times, these are for testing whether to inc nevents
        cdef long long cut0 = cutrange[0]
        cdef long long cut1 = cutrange[1]
        if cut0 > cut1: # swap 'em for the test
            cut0, cut1 = cut1, cut0

        cdef ndarray lock = np.zeros(s.nchans, dtype=int) # per-channel lockout timepoint indices (end inclusive?)
        s.lockp = <int *>lock.data # int pointer to .data field

        s.tilock = self.tilock # temporal index lockout, in num timepoints
        s.slock = self.slock # spatial lockout, in um
        cdef ndarray dm = self.dm # full Euclidean channel distance matrix for all possible chanis, floats in um
        s.dmp = <double *>dm.data # double pointer to .data field
        s.ndmchans = len(self.dm) # number of all possible chans in dm

        cdef ndarray eventtimes = self._eventtimes # init'd in self.search()
        cdef long long *eventtimesp = <long long *>eventtimes.data # long long pointer to .data field
        cdef ndarray maxchans = self._maxchans # init'd in self.search()
        cdef int *maxchansp = <int *>maxchans.data # int pointer to .data field

        cdef int ti, prevti, chanii, peakti, peak2ti, lastpeakti
        cdef float sign
        cdef int nevents = 0 # num non-excess events found so far while searching this block
        cdef int eventi = -1 # event index (indexes into .eventtimes)
        cdef int eventti # event time index (indexes into .data and .ts)

        '''procedure according to paper (indented stuff are my additions/speculations):
            - for all methods, get thresh xing, make sure it's a +ve thresh xing, ie make sure the
              signal at the preceding timepoint isn't locked out and is below threshold!
                - center on non locked-out maxchan within slock of chan with thresh xing?
            - search forward indefinitely to find waveform peak/valley? this is your spike time?
                - again, center on non locked-out maxchan within slock of chan with thresh xing?
                - again, search forward a bit to ensure you're at the waveform peak/valley? this is your spike time?
            - search forward and back by one tlock relative to timepoint of thresh xing
                - or should this search be relative to time of the peak/valley?
            - within that search, look for signal 2f greater than valley/2f less than peak
            - if you find such signal, you've got a spike
                - respect previously set lockouts during this search?

            - general lockout procedure (applies to all): only those chans within slock of maxchan that are over
              threshold at spike time are locked out, and are locked out only until the end of the first half of the
              first phase of the spike, although I'm not too clear why. Why not until the end of the first half of
              the last phase of the spike?


        '''
        #tcyloop = time.clock()
        for ti from 0 <= ti < s.nt: # iterate over all timepoints
            prevti = max(ti-1, 0) # previous timepoint, ensuring to go no earlier than first timepoint
            # TODO: shuffle chans in-place here
            for chanii from 0 <= chanii < s.nchans: # iterate over indices into chans
                # check that chan is free at previous timepoint and that we've gone from below to at or above threshold
                if not ( s.lockp[chanii] < prevti and abs(s.datap[chanii*s.nt + prevti]) < fixedthresh \
                         and abs(s.datap[chanii*s.nt + ti]) >= fixedthresh ):
                    continue # no, skip to next chan in chan loop
                sign = get_sign(s.datap[chanii*s.nt + ti]) # find whether thresh xing was +ve or -ve
                # find maxchan at timepoint ti by searching across eligible chans, center search on chanii
                find_maxchanii(&s, chanii, ti, sign)
                # search forward indefinitely for the first peak, this will be used as the event time
                eventti = find_peak(&s, ti, DEFTIRANGE, sign) # search forward almost indefinitely for peak of correct phase
                if eventti == -1: # couldn't find a peak
                    continue # skip to next chan in chan loop
                find_maxchanii(&s, s.maxchanii, eventti, sign) # update maxchan one last time for this putative event
                eventti = find_peak(&s, eventti, s.tilock, sign) # update eventti for this maxchan
                print 'ti=%s, t=%s, chanii=%s, sign=%d, eventt=%s, maxchanii=%s' % (ti, s.tsp[ti], chanii, sign, s.tsp[eventti], s.maxchanii)
                # search forward one tlock on the maxchan for another peak of opposite phase
                # that is 2*thresh greater (in the right direction) than the previous peak
                sign = -sign
                print 'sign', sign
                peak2ti = find_peak(&s, eventti, s.tilock, sign)
                if peak2ti == -1: # couldn't find a 2nd peak of opposite phase
                    continue # skip to next chan in chan loop
                print 'peak2ti, peak2t', peak2ti, s.tsp[peak2ti]
                if abs(s.datap[s.maxchanii*s.nt + eventti] - s.datap[s.maxchanii*s.nt + peak2ti]) < 2*fixedthresh:
                    print 'peak2ti isnt big enough'
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
                # to set lockout, continue searching forward for consecutive threshold-exceeding
                # peaks of alternating phase within tlock of each other that are conceivably
                # part of the same spike, yet large enough and far enough away in time to trigger an
                # unwanted threshold crossing
                lastpeakti = peak2ti
                print 'lastpeakti, lastpeakt', lastpeakti, s.tsp[lastpeakti]
                while True:
                    sign = -sign
                    print 'sign', sign
                    peakti = find_peak(&s, lastpeakti, s.tilock, sign)
                    print 'peakti, peakt', peakti, s.tsp[peakti]
                    if peakti == -1 or abs(s.datap[s.maxchanii*s.nt + peakti]) < fixedthresh:
                        # no peak found, or found peak doesn't exceed thresh
                        break # out of while loop
                    lastpeakti = peakti # update
                    print 'lastpeakti, lastpeakt', lastpeakti, s.tsp[lastpeakti]
                set_lockout(&s, lastpeakti) # lock out up to and including peak of last spike phase, and no further
        #print 'cy loop took %.3f sec' % (time.clock()-tcyloop)
        eventtimes = eventtimes[:nevents] # keep only the entries that were filled
        maxchans = maxchans[:nevents] # keep only the entries that were filled
        return np.asarray([eventtimes, maxchans])

cdef find_maxchanii(Settings *s, int maxchanii, int ti, float sign):
    """Update s.maxchanii at timepoint ti over non locked-out chans within slock of maxchanii.
    Finds most negative (sign == -1) or most positive (sign == 1) chan"""
    cdef int chanjj, chanj, maxchani
    s.maxchanii = maxchanii # init
    maxchani = s.chansp[maxchanii] # dereference to index into dmp
    for chanjj from 0 <= chanjj < s.nchans: # iterate over all chan indices
        chanj = s.chansp[chanjj] # dereference to index into dmp
        # if chanj is within slock of original maxchani, and has higher signal of correct sign than the biggest maxchan found so far, and isn't locked out:
        if s.dmp[maxchani*s.ndmchans + chanj] <= s.slock and \
            s.datap[chanjj*s.nt + ti]*sign > s.datap[s.maxchanii*s.nt + ti]*sign and \
            s.lockp[chanjj] < ti:
            s.maxchanii = chanjj # update

cdef float get_sign(float val):
    """Return the sign of the float input"""
    if val < 0.0:
        return -1.0
    else:
        return 1.0

cdef int find_peak(Settings *s, int ti, int tirange, float sign):
    """Return timepoint index of first peak (if sign == 1) or valley (if sign == -1)
    on s.maxchanii searching forward up to tirange from passed ti.
    Return -1 if no peak is found

    TODO: perhaps find_peak should ensure the maxchanii isn't locked out over any of
    the tirange it's searching, although this is implicit when it's called, ie,
    don't call find_peak without first checking for lockout up to ti

    NOTE: right now, this only finds the first peak, not the biggest peak within tirange!!!!!!!!!!!!!!!!!!!!

    """
    cdef int tj, peaktj = -1
    cdef long long offset = s.maxchanii*s.nt # this is a constant during the loop
    cdef float last = -INF * sign # uV
    for tj from ti <= tj < (ti + tirange):
        # if signal is becoming more -ve (sign == -1) or +ve (sign == 1)
        if s.datap[offset + tj]*sign > last*sign:
            peaktj = tj # update
            last = s.datap[offset + tj] # update
        else: # signal is becoming less -ve (sign == -1) or +ve (sign == 1)
            return peaktj
    return -1 # a peak was never found

cdef set_lockout(Settings *s, int ti):
    """Apply spatiotemporal lockout centered on s.maxchanii, lock out all chans
    within slock up to timepoint ti"""
    cdef int chanjj, chanj, maxchani
    for chanjj from 0 <= chanjj < s.nchans: # iterate over all chan indices
        maxchani = s.chansp[s.maxchanii] # dereference to index into dmp
        chanj = s.chansp[chanjj] # dereference to index into dmp
        if s.dmp[maxchani*s.ndmchans + chanj] <= s.slock: # chanjj is within spatial lockout in um
            s.lockp[chanjj] = ti # lock it out up to ti

cdef int max(int x, int y):
    """Return maximum of two ints"""
    if x >= y:
        return x
    else:
        return y

cdef float abs(float x):
    """Absolute value of a float"""
    if x < 0.0:
        x *= -1.0
    return x
'''
cdef int iabs(int x):
    """Absolute value of an integer"""
    if x < 0:
        x *= -1
    return x

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
'''
