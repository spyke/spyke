"""Spike detection loops implemented in Cython for speed"""

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


cpdef class BipolarAmplitudeFixedThresh_Cy:

    cpdef searchblock(self, wave, cutrange):
        """Search across all chans in a manageable block of waveform
        data and return a 2D array of spike times and maxchans.
        Apply both temporal and spatial lockouts.
        cutrange is required to correctly count number of spikes that
        won't be cut away as excess

        TODO: get_maxchan and set_lockout both have awkwardly long arg lists, maybe make up a struct type
              whose fields point to all the variables used in this method, and pass the struct to get_maxchan
              and set_lockout
                - or, just make most of the variables globals, using global keyword

        TODO: after searching for maxchan, search again for peak on that maxchan,
              maybe iterate a few times until stop seeing changes in result for
              that spike. But iteration isn't really possible, because we have our
              big ti loop to obey and its associated t and s locks.
              Maybe this is just an inherent limit to simple thresh detector?
              Maybe compromise is to search for maxchan immed on thresh xing (within slock),
              then wait til find peak on that chan, then search for maxchan (within slock)
              again upon finding peak.

              - find thresh xing
              - search immed for maxchan within slock
              - apply slock immed around maxchan so you stop getting more thresh xings on other chans for the same spike
              - wait for peak on maxchan
              - once peak found, search again for maxchan within slock, mark that as a spike
              - reapply slock around maxchan

        TODO: instead of searching forward indefinitely for a peak, maybe search
              forward and backwards say 0.5ms. Ah, but then searching backwards would
              ignore any tlock you may have had on that chan in the past. Maybe that means
              having to change how tlock is implemented

        TODO: take weighted spatial average of say the top 3 channels at peak time of the
              maxchan, and use that point as the center of the spatial lockout
                - this would mean you couldn't rely in the distance matrix dm anymore
                - expensive refinement for little payoff

        TODO: add option to search backwards in time:
                - maybe just slice data in reverse order, Cy loop won't know the difference?
                - would also have to reverse order of block loop, can prolly do this by
                  simply making blocksize BS -ve

        TODO: (maybe): chanii loop should go in random order on each ti, to prevent a chan from
              dominating with its spatial lockout or something like that

        DONE: when searching for maxchan, limit search to chans within spatial lockout.
              This will be faster, and will also prevent unrelated distant cells firing
              at the same time from triggering each other, or something
        """
        if not wave.data.flags.contiguous:
            print "wave.data ain't contig, strides:", wave.data.strides
            wave.data = wave.data.copy() # make it contiguous for easy pointer indexing
        if not wave.ts.flags.contiguous:
            print "wave.ts ain't contig, strides:", wave.ts.strides
            wave.ts = wave.ts.copy()

        cdef int nchans = len(self.chans)
        cdef ndarray chans = np.asarray(self.chans)
        cdef int *chansp = <int *>chans.data # int pointer to .data field

        if self.chans != range(nchans): # if self.chans is non contiguous
            ttake = time.clock()
            data = wave.data.take(chans, axis=0) # pull our chans of data out, this assumes wave.data has all possible chans in it
            print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # save time by avoiding an unnecessary .take
            data = wave.data
        cdef ndarray absdata = data # name it absdata
        arrabs(absdata) # now do the actual abs, in-place, about 2x faster than np.abs
        cdef float *absdatap = <float *>absdata.data # float pointer to .data field

        cdef ndarray ts = wave.ts
        cdef long long *tsp = <long long *>ts.data # long long pointer to timestamp .data
        cdef long long spiket # holds current spike timestamp

        cdef int nt = absdata.dimensions[1]
        cdef int nkeptspikes = self.totalnspikes # num non-excess spikes found so far in this Detector.search()
        cdef int nexcessspikes = 0
        cdef int maxnspikes = self.maxnspikes
        cdef float fixedthresh = self.fixedthresh

        # cut times, these are for testing whether to inc nkeptspikes
        cdef long long cut0 = cutrange[0]
        cdef long long cut1 = cutrange[1]
        cdef long long tmp
        if cut0 > cut1: # swap 'em for the test
            tmp = cut0
            cut0 = cut1
            cut1 = tmp

        cdef ndarray xthresh = np.zeros(nchans, dtype=int) # per-channel thresh xing flags (0 or 1)
        cdef int *xthreshp = <int *>xthresh.data # int pointer to .data field
        cdef ndarray lock = np.zeros(nchans, dtype=int) # per-channel lockout timepoint counters, >= 0 indicates locked out
        cdef int *lockp = <int *>lock.data # int pointer to .data field
        cdef ndarray last = np.zeros(nchans, dtype=np.float32) # holds last signal value per chan, floats in uV
        cdef float *lastp = <float *>last.data # float pointer to .data field

        cdef int tilock = self.tilock # temporal index lockout, in num timepoints
        cdef double slock = self.slock # spatial lockout, in um
        cdef ndarray dm = self.dm # Euclidean channel distance matrix, floats in um
        cdef double *dmp = <double *>dm.data # double pointer to .data field

        cdef ndarray spiketimes = self._spiketimes # init'd in self.search()
        cdef ndarray maxchans = self._maxchans # init'd in self.search()
        cdef long long *spiketimesp = <long long *>spiketimes.data # long long pointer to .data field
        cdef int *maxchansp = <int *>maxchans.data # int pointer to .data field

        cdef int ti, chanii, maxchanii
        cdef int spikei = -1 # index into spiketimes
        cdef float v # current signal voltage, uV

        #tcyloop = time.clock()
        for ti from 0 <= ti < nt: # iterate over all timepoints
            # TODO: shuffle chans in-place here, or just have a second level of chan indices,
            # shuffle those, and iterate over chaniii
            # TODO: replace lockp with binary bitmask, shift bits on every ti, might let you search for spikes into future
            # semi-independently on each chan
            for chanii from 0 <= chanii < nchans:
                lockp[chanii] -= 1 # decr all chans' lockout counters
                #if chanii == 33:
                #    print 't: %d, decr ch33 lock to: %d' % (tsp[ti], lockp[chanii])
            for chanii from 0 <= chanii < nchans: # iterate over indices into chans
                if lockp[chanii] < 0: # if this chan isn't locked out, search for a thresh xing or a peak
                    v = absdatap[chanii*nt + ti] # (absdata[chanii, ti])
                    if xthreshp[chanii] == 0: # we're looking for a thresh xing
                        if v >= fixedthresh: # met or exceeded threshold
                            #print 't: %d, thresh xing, chan: %d' % (tsp[ti], chanii)
                            xthreshp[chanii] = 1 # set maxchan's crossed threshold flag
                            lastp[chanii] = v # update maxchan's last value
                            lockp[chanii] = -1 # ensure maxchan's lockout is off
                    else: # xthresh[chanii] == 1, in crossed thresh state, now we're look for a peak
                        if v > lastp[chanii]: # if signal is still increasing
                            lastp[chanii] = v # update last value for this chan, continue searching for peak
                        else: # signal is decreasing, declare previous ti as a spike timepoint
                            # find maxchanii within slock of chanii, start with current chan as max chan, pass previous ti
                            maxchanii = self.get_maxchanii(chanii, nchans, chansp, dmp, slock, absdatap, nt, ti-1)
                            #print 't: %d, found spike at t=%d on chan %d' % (tsp[ti], tsp[ti-1], maxchanii)
                            # apply spatiotemporal lockout now that spike has been found, pass tilock relative to previous ti
                            self.set_lockout(chanii, maxchanii, nchans, chansp, dmp, slock, tilock-1, xthreshp, lastp, lockp)
                            spikei += 1
                            spiket = tsp[ti-1] # spike time is timestamp of previous time index
                            if cut0 <= spiket and spiket <= cut1:
                                nkeptspikes += 1 # this spike won't be cut away as excess in .search()
                            else:
                                nexcessspikes += 1 # this spike will be cut away as excess in .search()
                            spiketimesp[spikei] = spiket # save spike time
                            maxchansp[spikei] = chansp[maxchanii] # store chan spike is centered on
                            if nkeptspikes >= maxnspikes: # exit here, don't search any more chans
                                ti = nt # TODO: nasty hack to get out of outer ti loop
                                break # out of inner chanii loop
        #print 'cy loop took %.3f sec' % (time.clock()-tcyloop)
        nnewspikes = nkeptspikes + nexcessspikes - self.totalnspikes # num spikes added to spiketimes and maxchans
        self.totalnspikes = nkeptspikes # update
        spiketimes = spiketimes[:nnewspikes] # keep only the entries that were filled
        maxchans = maxchans[:nnewspikes] # keep only the entries that were filled
        return np.asarray([spiketimes, maxchans])

    cdef int get_maxchanii(self, int maxchanii, int nchans, int *chansp,
                           double *dmp, double slock, float *absdatap,
                           int nt, int ti):
        """Finds max chanii at timepoint ti

        TODO: this should really be applied over and over, recentering the search radius, until maxchanii stops changing,
              although that would probably be a very minor refinement at a rather high cost
        TODO: might be able to speed this up by pulling out the maxchanii'th row (or column, whatever your fancy) from dm,
              sorting it from smallest to largest distance while keeping track of associated chaniis, doing a searchsorted on it
              to find index of first distance that's outside slock, and then only checking those chans that fall less than that
              index for larger signal
        """
        cdef int chanj, chanjj
        maxchani = chansp[maxchanii]
        for chanjj from 0 <= chanjj < nchans: # iterate over all chan indices
            chanj = chansp[chanjj]
            # if chanjj within slock of maxchani has higher signal:
            if dmp[maxchani*nchans + chanj] <= slock and absdatap[chanjj*nt + ti] > absdatap[maxchanii*nt + ti]:
                maxchanii = chanjj # update maxchanii
        # recursive call goes here to search with newly centered slock radius
        return maxchanii

    cdef set_lockout(self, int chanii, int maxchanii, int nchans, int *chansp,
                     double *dmp, double slock, int tilock,
                     int *xthreshp, float *lastp, int *lockp):
        """Applies spatiotemporal lockout centered on current maxchanii from current ti forward"""
        for chanjj from 0 <= chanjj < nchans: # iterate over all chan indices
            maxchani = chansp[maxchanii]
            chanj = chansp[chanjj]
            if dmp[maxchani*nchans + chanj] <= slock: # (== dm[maxchani, chanj]) chanjj is within spatial lockout in um
                xthreshp[chanjj] = 0 # clear its threshx flag
                lastp[chanjj] = 0.0 # reset last so it's ready when it comes out of lockout
                lockp[chanjj] = tilock # apply its temporal lockout
                #if chanjj == 33:
                #    print 'locked ch33:', lockp[chanjj]


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
