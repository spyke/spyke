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

    cpdef searchblock(self, wave): # TODO: convert to cpdef
        """Search across all chans in a manageable block of waveform
        data and return a tuple of spike time and maxchan arrays.
        Apply both temporal and spatial lockouts

        TODO: take transpose of absdata, so that ti are in rows and chanii are in cols. This might
              be more efficient, since iterating over the chans loop (which is what we're doing most
              here) would iterate over adjacent points in memory, instead of points nt away from each
                - tried this in a simple test using cython_test.cy_setmat, didn't see any speed difference

        TODO: init lock, last, and xthresh in .search(), and bind them to self
                - nope, can't do, cuz we need to ensure they're all reset to zero on every .searchblock() call

        TODO: is there an abs f'n in C I can use? what's the most optimized way of taking abs?

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
        cdef int nchans = len(self.chans)
        cdef ndarray chans = np.asarray(self.chans)
        cdef int *chansp = <int *>chans.data # int pointer to .data field

        if self.chans != range(nchans): # if self.chans is non contiguous
            ttake = time.clock()
            data = wave.data.take(chans, axis=0) # pull our chans of data out, this assumes wave.data has all possible chans in it
            print 'data take in searchblock() took %.3f sec' % (time.clock()-ttake)
        else: # save time by avoiding an unnecessary .take
            data = wave.data
        tabs = time.clock()
        cdef ndarray absdata = data # name it absdata
        abs(absdata) # now do the actual abs, in-place, about 2x faster than np.abs
        print 'abs took %.3f sec' % (time.clock()-tabs)
        cdef float *absdatap = <float *>absdata.data # float pointer to .data field

        #cdef int nt = wave.data.shape[1]
        cdef int nt = absdata.dimensions[1] # same thing, maybe faster (not!)
        cdef int totalnspikes = self.totalnspikes # total num of spikes found so far in this Detector.search()
        cdef int maxnspikes = self.maxnspikes
        cdef float thresh = self.thresh

        cdef ndarray xthresh = np.zeros(nchans, dtype=int) # thresh xing flags (0 or 1)
        cdef int *xthreshp = <int *>xthresh.data # int pointer to .data field
        cdef ndarray lock = np.zeros(nchans, dtype=int) # holds number of lockout timepoints left per chan
        cdef int *lockp = <int *>lock.data # int pointer to .data field
        cdef ndarray last = np.zeros(nchans, dtype=np.float32) # holds last signal value per chan, floats in uV
        cdef float *lastp = <float *>last.data # float pointer to .data field

        cdef int tilock = self.tilock # temporal index lockout, in num timepoints
        cdef double slock = self.slock # spatial lockout, in um
        cdef ndarray dm = self.dm # Euclidean channel distance matrix, floats in um
        cdef double *dmp = <double *>dm.data # double pointer to .data field
        cdef ndarray spiketis = self.spiketis # init'd in self.search()
        cdef int *spiketisp = <int *>spiketis.data # int pointer to .data field
        cdef int nti = spiketis.dimensions[1] # width

        cdef int ti, chanii, maxchanii
        cdef int spikei = -1 # index into spiketis
        cdef float v # current signal voltage, uV

        tcyloop = time.clock()

        for ti from 0 <= ti < nt: # iterate over all timepoints
            for chanii from 0 <= chanii < nchans: # iterate over indices into chans
                if lockp[chanii] > 0: # if this chan is locked out
                    lockp[chanii] -= 1 # do nothing other than decr this chan's temporal lockout
                else: # search for a thresh xing or a peak
                    v = absdatap[chanii*nt + ti] # (absdata[chanii, ti])
                    if xthreshp[chanii] == 0: # we're looking for a thresh xing
                        if v >= thresh: # met or exceeded threshold
                            # find maxchanii within slock of chanii, start with current chan as max chan
                            maxchanii = self.get_maxchanii(chanii, nchans, chansp, dmp, slock, lockp, absdatap, nt, ti)
                            # apply spatiotemporal lockout to prevent extra thresh xings for this developing spike
                            self.set_lockout(chanii, maxchanii, nchans, chansp, dmp, slock, tilock, xthreshp, lastp, lockp)
                            xthreshp[maxchanii] = 1 # set maxchan's crossed threshold flag
                            lastp[maxchanii] = v # update maxchan's last value
                            lockp[maxchanii] = 0 # clear maxchan's lockout set inadvertently by set_lockout
                    else: # xthresh[chanii] == 1, in crossed thresh state, now we're look for a peak
                        if v > lastp[chanii]: # if signal is still increasing
                            lastp[chanii] = v # update last value for this chan, wait til next ti to decide if this is a peak
                        else: # signal is decreasing, declare previous ti as a spike timepoint
                            # find maxchanii within slock of chanii, start with current chan as max chan, pass previous ti
                            maxchanii = self.get_maxchanii(chanii, nchans, chansp, dmp, slock, lockp, absdatap, nt, ti-1)
                            # apply spatiotemporal lockout to prevent extra thresh xings for this developing spike
                            self.set_lockout(chanii, maxchanii, nchans, chansp, dmp, slock, tilock, xthreshp, lastp, lockp)
                            spikei += 1;
                            totalnspikes += 1 # inc
                            spiketisp[spikei] = ti-1 # (spiketisp[0, spikei]) save previous time index
                            spiketisp[nti + spikei] = chansp[maxchanii] # (spiketisp[1, spikei]) store chan spike is centered on
                            if totalnspikes >= maxnspikes: # exit here, don't search any more chans
                                ti = nt # TODO: nasty hack to get out of outer ti loop
                                break # out of inner chanii loop

        print 'final ti is', ti
        print 'cy loop took %.3f sec' % (time.clock()-tcyloop)
        nnewspikes = totalnspikes - self.totalnspikes
        self.totalnspikes = totalnspikes # update
        spiketis = spiketis[:, :nnewspikes] # keep only the entries that were filled
        spiketimes = wave.ts.take(spiketis[0]) # convert from indices to actual spike times, wave.ts is int64 array
        maxchans = spiketis[1]
        return spiketimes, maxchans

    cdef int get_maxchanii(self, int maxchanii, int nchans, int *chansp,
                           double *dmp, double slock, int *lockp, float *absdatap,
                           int nt, int ti):
        """Finds max chanii at current ti

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
            # chanjj within slock of maxchani has higher signal:
            if dmp[maxchani*nchans + chanj] <= slock and absdatap[chanjj*nt + ti] > absdatap[maxchanii*nt + ti]:
                maxchanii = chanjj # update maxchanii
        # recursive call goes here to search with newly centered slock radius
        return maxchanii # return maxchani

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
                # apply its temporal lockout
                if chanjj > chanii: # we haven't encountered chanjj yet in the outer chanii loop
                    lockp[chanjj] = tilock+1 # lockout by one extra timepoint which it'll decr before we leave this ti
                else:
                    lockp[chanjj] = tilock


cdef abs(ndarray a):
    """In-place absolute value of 2D float32 array.
    Or, might be better to just use fabs from C math library?"""
    cdef int nrows = a.dimensions[0]
    cdef int ncols = a.dimensions[1]
    cdef float *datap = <float *>a.data
    cdef int i, j
    for i from 0 <= i < nrows: #( i=0; ti<nt; ti++ ) { // iterate over all timepoint indices
        for j from 0 <= j < ncols: #( int chanii=0; chanii<nchans; chanii++ ) { // iterate over all chan indices
            if datap[i*ncols + j] < 0.0:
                datap[i*ncols + j] *= -1 # modify in-place
