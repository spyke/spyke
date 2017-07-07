# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""Some functions written in Cython for max performance"""

cimport cython
from cython.parallel import prange#, parallel
import numpy as np
cimport numpy as np
from numpy cimport uint8_t, int8_t, int16_t, int32_t, int64_t, float32_t, float64_t

import time

cdef extern from "math.h":
    int abs(int x)
    float fabs(float x)
    double ceil(double x) nogil

cdef extern from "limits.h":
    int INT_MAX

cdef extern from "float.h":
    double DBL_MAX

cdef extern from "stdio.h":
    int printf(char *, ...)

cdef extern from "string.h":
    cdef void *memset(void *, int, size_t) nogil # sets n bytes in memory to constant


cdef short select_short(short *a, int l, int r, int k):
    """Returns the k'th (0-based) ranked entry from float array a within left
    and right pointers l and r. This is quicksort partitioning based
    selection, taken from Sedgewick (Algorithms, 2ed 1988, p128).
    Note that this modifies a in-place"""
    cdef int i, j
    cdef short v, temp
    if r < l:
        raise ValueError('bad pointer range in select()')
    while r > l:
        v = a[r]
        i = l-1
        j = r
        while True:
            while True:
                i += 1
                if a[i] >= v: break
            while True:
                j -= 1
                if a[j] <= v: break
            temp = a[i] # swap a[i] and a[j]
            a[i] = a[j]
            a[j] = temp
            if j <= i: break
        a[j] = a[i]
        a[i] = a[r]
        a[r] = temp # temp was old a[j]
        if i >= k: r = i-1
        if i <= k: l = i+1
    return a[k] # return kth in 0-based


def median_inplace_2Dshort(np.ndarray[int16_t, ndim=2, mode='c'] arr):
    """Assumes C-contig 2D input array. arr will probably be from a copy anyway,
    since it modifies in-place"""
    cdef Py_ssize_t nchans, nt, k, i
    cdef np.ndarray[int16_t, ndim=1] result
    cdef short *a
    nchans = arr.shape[0]
    nt = arr.shape[1]
    result = np.zeros(nchans, dtype=np.int16)
    k = (nt-1) // 2
    a = <short *>arr.data # short pointer to arr's .data field
    for i in range(nchans):
        result[i] = select_short(a, i*nt, i*nt+nt-1, i*nt+k) # this won't work for strided stuff
    return result


cdef double mean_short(short *a, int N):
    cdef Py_ssize_t i # recommended type for looping
    cdef double s=0.0
    for i in range(N):
        s += a[i]
    s /= N
    return s


def mean(np.ndarray[float64_t, ndim=1] a):
    """Uses new simpler numpy type notation for fast indexing, but is still a
    bit slower than the classical way, because you currently can't
    use the new notation with cdefs. (This may no longer be true...)"""
    cdef Py_ssize_t i, N = len(a)
    cdef double s=0
    for i in range(N):
        s += a[i]
    s /= N
    return s


def mean_2Dshort(np.ndarray[int16_t, ndim=2] a):
    """Uses new simpler numpy type notation for fast indexing, but is still a
    bit slower than the classical way, because you currently can't
    use the new notation with cdefs"""
    cdef Py_ssize_t i, j, nchans, nt
    nchans = a.shape[0]
    nt = a.shape[1]
    cdef np.ndarray[float64_t, ndim=1] s = np.zeros(nchans)
    for i in range(nchans):
        for j in range(nt):
            s[i] += a[i, j]
        s[i] /= nt # normalize
    return s


cpdef dostuff(np.ndarray[float64_t, ndim=1] a):
    """Just a f'n to do some stuff in place with the GIL released"""
    cdef Py_ssize_t i, N = len(a)
    cdef float b = 1.2345
    with nogil:
        for i in range(N):
            a[i] += a[i] * b / (a[i]**2 + b)
            #a[i] *= 2.0


def dostuffthreads(np.ndarray[float64_t, ndim=1] a):
    """Demo use of multithreading pool from within Cython"""
    from spyke import threadpool_alt
    from multiprocessing import cpu_count
    ncpus = cpu_count()
    pool = threadpool_alt.Pool(ncpus)
    units = np.split(a, ncpus)
    pool.map(dostuff, units)
    pool.terminate()


def testrange(np.ndarray[int32_t, ndim=1] a,
              int start, int end):
    """Testing cython range f'n"""
    cdef Py_ssize_t i, N = len(a)
    for i in range(start, end):
        printf('%d\n', i)


def sharpness2D(np.ndarray[int16_t, ndim=2] signal):
    """Spike peak sharpness measure which takes (height)**2 / width
    for each peak, and relies on zero crossings to demarcate borders between peaks.
    First, update npoints, check for extremum and update ext. Then, look one step ahead
    for 0-crossing and calc sharpness if one is about to occur.

    Return array of same size as signal, filled mostly with zeros, with signed
    sharpness values at the points corresponding to spike peaks.

    TODO: test if double math is faster than float math. They're probably identical.

    TODO: might also try adding mode='c' kwarg to signal arg, if you know it's C contig,
    reduces need to do stride calc on each access. Actually, might try adding mode='c'
    to all locally declared np arrays as well.

    TODO: use FWHM or FW 1/10 max or something instead of npoints of each segment to more
    accurately determine width. This will require at least some linear interpolation between
    points straddling whatever fraction of max level on either side of each extremum.

    DONE: do I really need to check for an extremum between each 0 crossing? I think not.
    Just find the max abs between zero crossings. Also, don't need to check sign, since sign
    will always alternate anyway.

    DONE: stop checking for weird corner cases, since end bits of results are now thrown away
    after the call, and we won't be running sharpness2D on short bits of waveforms any more.
    """
    cdef Py_ssize_t nchans, nt, ci, ti, extti, npoints
    cdef bint cross=False, crossedonce=False
    cdef short now, next
    cdef float ext

    nchans = signal.shape[0]
    nt = signal.shape[1]
    cdef np.ndarray[float32_t, ndim=2] sharp = np.zeros((nchans, nt), dtype=np.float32)

    assert nt < 2**31 # make sure time indices don't overflow

    for ci in range(nchans):
        ext = 0.0 # val of biggest extremum so far for current segment
        extti = 0 # ti of biggest extremum so far for current segment
        npoints = 0 # npoints in current segment
        next = signal[ci, 0] # init
        for ti in range(nt-1):
            now = next # now = signal[ci, ti]
            next = signal[ci, ti+1]
            cross = (now > 0) != (next > 0) # 0-crossing coming up?
            if not crossedonce: # haven't crossed 0 yet...
                if cross: # ...but about to
                    crossedonce = True # for next iter
                continue # nothing to do until we cross 0 at least once
            npoints += 1 # inc for this segment, corresponds to "now" point in segment
            #print('ci=%d, ti=%d, npoints=%d' % (ci, ti, npoints))
            if abs(now) > fabs(ext): # found new biggest extremum so far for this segment
                extti = ti # store its timepoint
                ext = now # update for this segment
                #print('found new biggest local ext=%f at ci=%d, ti=%d' % (ext, ci, extti))
            if cross:
                # 0-cross coming up, calculate sharpness of extremum in this segment
                #print('reached end of segment')
                #print('using npoints=%d for sharpness calc' % npoints)
                # square height, normalize by peak width
                ext *= fabs(ext) # maintain extremum sign
                ext /= npoints
                sharp[ci, extti] = ext # store
                #print('sharpness is %f at ci=%d, extti=%d' % (ext, ci, extti))
                ext = 0.0 # reset biggest max/min so far for new segment
                # increment extti so that even if next segment only has ext=0 and npoints=1,
                # sharp[ci, extti] of the current segment isn't overwritten with a 0:
                extti += 1
                npoints = 0 # reset for new segment

    return sharp


def argthreshsharp(np.ndarray[int16_t, ndim=2] signal,
                   np.ndarray[int16_t, ndim=1] thresh,
                   np.ndarray[float32_t, ndim=2] sharp):
    """Given original signal, threshold array, and sharpness array,
    return a temporally sorted n x 2 (ti, ci) array of peak indices that exceed
    thresh for the appropriate chan"""

    cdef Py_ssize_t nt, nchans, ti, ci, npeaks = 0

    assert signal.shape[1] < 2**31 # stick to int32 time indices
    nchans = signal.shape[0]
    nt = signal.shape[1]
    assert sharp.shape[0] == nchans
    assert sharp.shape[1] == nt
    assert thresh.shape[0] == nchans

    # worst case scenario: we find as many thresh exceeding peaks as nt
    cdef np.ndarray[int32_t, ndim=2] peakis = np.empty((nt, 2), dtype=np.int32)

    for ti in range(nt):
        for ci in range(nchans):
            #if ci == 28 and (ti == 543331 or ti == 543359):
            #    printf('ti=%d, ci=%d, sharp=%f, signal=%d, thresh=%d\n',
            #           ti, ci, sharp[ci, ti], abs(signal[ci, ti]), thresh[ci])
            if sharp[ci, ti] != 0.0 and abs(signal[ci, ti]) >= thresh[ci]:
                peakis[npeaks, 0] = ti
                peakis[npeaks, 1] = ci
                npeaks += 1

    return peakis[:npeaks]

'''
def argsharp(np.ndarray[float32_t, ndim=2] sharp):
    """Given sharpness array, return a temporally sorted n x 2 (ti, ci) array
    of peak indices"""

    cdef Py_ssize_t nt, chans, ti, ci, npeaks = 0

    assert sharp.shape[1] < 2**31 # stick to int32 time indices
    nchans = sharp.shape[0]
    nt = sharp.shape[1]

    # worst case scenario: we find as many thresh exceeding peaks as nt
    cdef np.ndarray[int32_t, ndim=2] peakis = np.empty((nt, 2), dtype=np.int32)

    for ti in range(nt):
        for ci in range(nchans):
            if sharp[ci, ti] != 0.0:
                peakis[npeaks, 0] = ti
                peakis[npeaks, 1] = ci
                npeaks += 1

    return peakis[:npeaks]
'''

def rowtake_cy(np.ndarray[int32_t, ndim=2] a,
               np.ndarray[int32_t, ndim=2] i):
    """For each row in a, return values according to column indices in the
    corresponding row in i. Returned shape == i.shape"""

    cdef Py_ssize_t nrows, ncols, rowi, coli
    cdef np.ndarray[int32_t, ndim=2] out

    nrows = i.shape[0]
    ncols = i.shape[1] # num cols to take for each row
    #assert a.shape[0] == nrows
    #assert i.max() < a.shape[1] # make sure col indices into a aren't out of range
    out = np.empty((nrows, ncols), dtype=np.int32)

    for rowi in range(nrows):
        for coli in range(ncols):
            out[rowi, coli] = a[rowi, i[rowi, coli]]

    return out


def xcorr(int64_t[::1] x,
          int64_t[::1] y,
          int64_t[::1] trange):
    """Calculate cross-correlation of timepoints in x with y, constrained to lower
    and upper bounds in trange. Assume timepoints in x and y are sorted. Return spike times
    of y relative to x."""
    cdef int64_t ntx, nty, loti, dtsi, xti, yti, maxxti, maxyti, t, dt
    cdef int64_t low = trange[0]
    cdef int64_t high = trange[1]
    cdef int64_t DTSALLOCSIZE = 1000000
    ntx = x.shape[0]
    nty = y.shape[0]
    maxxti = ntx - 1
    maxyti = nty - 1
    cdef int64_t[::1] dts = np.zeros(DTSALLOCSIZE, dtype=np.int64)
    cdef int64_t maxdtsi = dts.shape[0] - 1

    loti = 0
    dtsi = 0
    for xti in range(ntx):
        # t is current timepoint in x to compare to all timepoints in y:
        t = x[xti]
        while y[loti] - t < low: # keep checking lower trange bound
            loti += 1
            if loti > maxyti: # no y timepoints fall within trange of t
                break
        # start collecting dt values:
        if loti > maxyti: # no y timepoints fall within trange of t
            continue # to next xti
        yti = loti
        dt = y[yti] - t # dt is y relative to x
        while dt < high: # keep checking upper trange bound
            if dtsi > maxdtsi:
                # when growing an array, pretty much need to allocate a new one,
                # can't very often do it in place:
                dts = np.resize(dts, (dts.shape[0] + DTSALLOCSIZE,))
                maxdtsi = dts.shape[0] - 1
                printf('resized dts array to %d entries\n', dts.shape[0])
            dts[dtsi] = dt
            #printf('%d ', dtsi)
            dtsi += 1 # inc for next loop iter
            yti += 1
            if yti > maxyti: # don't exceed maxyti when indexing into y
                break
            dt = y[yti] - t # update for next loop iter
    return np.asarray(dts[:dtsi]) # trim it down, convert memory view slice to array


## TODO: it may be that np.ndarray[float32_t, ndim=2, mode='c'] definitions run faster
## than float32_t[:, :] definitions. Or at least they seem to in 1D in alignbest_cy.
def NDsepmetric(float32_t[:, :] C0,
                float32_t[:, :] C1,
                int Nmax=INT_MAX):
    """Calculate N-dimensional cluster seperation metric, for a pair of clusters. This is
    based on nearest neighbour membership: assuming cluster 0 is smaller than cluster 1,
    calculate fraction of points in cluster 0 whose nearest neighbour is another point in
    cluster 0. Each row in the arrays is a point, each column a dimension.
    This returns 1 - overlap index in Swindale & Spacek, 2014"""
    
    cdef int N, N0, N1, ndim, ci, i, j, k, nself
    cdef double f0, O, S
    assert C0.shape[1] == C1.shape[1]
    ndim = C0.shape[1]

    # ensure cluster 0 is smaller than cluster 1:
    N0 = C0.shape[0]
    N1 = C1.shape[0]
    if not N0 <= N1:
        C0, C1 = C1, C0 # swap them
        N0, N1 = N1, N0

    # for speed, limit to up to Nmax points in each cluster, keeping only every
    # step'th point
    if N0 > Nmax:
        step = <int> ceil(<double>(N0) / Nmax) # round up
        #print('Nmax: %d, N0: %d, step: %d' % (Nmax, N0, step))
        C0 = C0[::step, :]
        N0 = C0.shape[0] # update
        #print('new N0: %d' % N0)
    if N1 > Nmax:
        step = <int> ceil(<double>(N1) / Nmax) # round up
        #print('Nmax: %d, N1: %d, step: %d' % (Nmax, N1, step))
        C1 = C1[::step, :]
        N1 = C1.shape[0] # update
        #print('new N1: %d' % N1)
    N = N0 + N1 # total npoints across clusters

    # check nearest neighbour membership of each point in C0:
    #to use prange, might need to have data in 2D float array instead of 2d numpy array,
    #to prevent segfaults. Actually, that no longer seems to be the case:
    nself = 0
    for i in prange(N0, nogil=True, schedule='dynamic'):
        # how is it you define variables as private to a thread, vs shared between threads?
        # Cython does it implicitly
        nself += NNmembership(i, ndim, N0, N1, C0, C1)

    f0 = <double>nself / <double>N0 # nearest neighbour fraction belonging to same cluster
    O = (1 - f0) / (1 - <double>N0/<double>N) # overlap index
    S = 1 - O # separation metric
    #print('nself=%d, N0=%d, N1=%d'  % (nself, N0, N1))
    #print('f0=%.3f, O=%.3f, S=%.3f' % (f0, O, S))
    return S


## TODO: it may be that np.ndarray[float32_t, ndim=2, mode='c'] definitions run faster
## than float32_t[:, :] definitions. Or at least they seem to in 1D in alignbest_cy.
## However, Cython says memoryview slices in arg def are necessary in order to allow nogil.
cdef int NNmembership(int i, int ndim, int N0, int N1,
                      float32_t[:, :] C0,
                      float32_t[:, :] C1) nogil:
    """Determine membership of nearest neighbour of point i, assumed to be a point
    in cluster C0. Return 1 if nearest neighbour is in C0, 0 otherwise"""
    cdef int j, k
    cdef bint continuei, continuej
    cdef double d, d02, d12, min_d02=DBL_MAX, min_d12=DBL_MAX
    for j in range(N0):
        if i == j:
            continue # to next j
        d02 = 0.0
        for k in range(ndim):
            d = C0[i, k] - C0[j, k]
            d02 += d * d # faster than calling **2
            if d02 > min_d02: # break out of k loop, continue to next j
                continuej = True
                break # out of k loop
        if continuej:
            continuej = False
            continue # to next j
        if d02 < min_d02:
            min_d02 = d02 # update
            
    for j in range(N1):
        d12 = 0.0
        for k in range(ndim):
            d = C0[i, k] - C1[j, k]
            d12 += d * d
            if d12 > min_d12: # break out of k loop, continue to next j
                continuej = True
                break # out of k loop
        if continuej:
            continuej = False
            continue # to next j
        if d12 < min_d12:
            min_d12 = d12 # update
            if min_d12 < min_d02: # nearest point is not in cluster 0
                return 0

    # we have min_d02 <= min_d12, so point i's closest neighbour is also in
    # cluster 0, count it as having the same membership
    return 1


#def alignbest_cy(sort, int64_t[:] sids, int64_t[:] tis, int64_t[:] chans):
def alignbest_cy(sort,
                 np.ndarray[int64_t, ndim=1, mode='c'] sids,
                 np.ndarray[int64_t, ndim=1, mode='c'] tis,
                 np.ndarray[int64_t, ndim=1, mode='c'] chans):
    """Align all sids between tis on chans by best fit according to mean squared error.
    chans are assumed to be a subset of channels of sids. Return sids
    that were actually moved and therefore need to be marked as dirty"""
    # TODO: make maxshift a f'n of interpolation factor
    DEF MAXSHIFT = 2 # constant, shift +/- this many timepoints, keep within int8 bounds
    spikes = sort.spikes
    # copy needed fields from spikes rect array as simple arrays, should come out as contig:
    cdef np.ndarray[uint8_t, ndim=1, mode='c'] spikes_nchans = spikes['nchans'][sids]
    cdef np.ndarray[uint8_t, ndim=2, mode='c'] spikes_chans = spikes['chans'][sids]
    cdef int nspikes = sids.shape[0]
    cdef int nchans = chans.shape[0]
    cdef np.ndarray[int16_t, ndim=3, mode='c'] wd = sort.wavedata
    cdef int nt = wd.shape[2] # num timepoints in each waveform
    cdef int ti0 = tis[0]
    cdef int ti1 = tis[1]
    cdef int subnt = ti1 - ti0 # num timepoints to slice from each waveform
    cdef int subntdiv2 = subnt // 2
    #print('subntdiv2 on either side of t=0: %d' % subntdiv2)
    if subntdiv2 < MAXSHIFT:
        raise ValueError("Selected waveform duration too short")
    #maxshiftus = MAXSHIFT * self.stream.tres
    # from -MAXSHIFT to MAXSHIFT, inclusive, use int8 because spikes['tis'] is int8:
    cdef np.ndarray[int8_t, ndim=1, mode='c'] shifts = \
        np.arange(-MAXSHIFT, MAXSHIFT+1, dtype=np.int8)
    cdef int nshifts = shifts.shape[0]
    print("Padding waveforms with up to +/- %d points of edge data" % MAXSHIFT)

    # not worth subsampling here while calculating meandata, since all this
    # stuff in this loop is needed in the shift loop below
    cdef np.ndarray[int16_t, ndim=3, mode='c'] subsd
    subsd = np.zeros((nspikes, nchans, subnt), dtype=wd.dtype) # subset of spike data
    cdef np.ndarray[int64_t, ndim=2, mode='c'] spikechanis
    spikechanis = np.zeros((nspikes, nchans), dtype=np.int64)
    #t0 = time.time()
    cdef int sidi, sid
    cdef int shifti, chani, ti, spikechani=0
    cdef long long chansubsd
    for sidi in range(nspikes):
        sid = sids[sidi]
        for chani in range(nchans):
            chan = chans[chani]
            for spikechani in range(spikes_nchans[sidi]):
                if spikes_chans[sidi, spikechani] == chan:
                    spikechanis[sidi, chani] = spikechani
                    break # out of spikechani loop
            for ti in range(subnt):
                subsd[sidi, chani, ti] = wd[sid, spikechani, ti0+ti]
    #print('mean prep loop for best shift took %.3f sec' % (time.time()-t0))
    #t0 = time.time()
    cdef np.ndarray[float64_t, ndim=2, mode='c'] meandata = subsd.mean(axis=0) # float64
    #print('mean for best shift took %.3f sec' % (time.time()-t0))

    # choose best shifted waveform for each spike
    # widesd holds current spike data plus padding on either side
    # to allow for full width slicing for all time shifts:
    cdef int maxnchans = spikes_nchans.max() # of all sids
    cdef int wident = MAXSHIFT+nt+MAXSHIFT
    cdef int maxwidesdi = wident - 1
    cdef np.ndarray[int16_t, ndim=2, mode='c'] sd
    sd = np.zeros((maxnchans, nt), dtype=wd.dtype)
    cdef np.ndarray[int16_t, ndim=2, mode='c'] widesd
    widesd = np.zeros((maxnchans, wident), dtype=wd.dtype)
    cdef np.ndarray[int16_t, ndim=3, mode='c'] shiftedsubsd = subsd.copy() # init
    cdef np.ndarray[int16_t, ndim=3, mode='c'] tempsubshifts
    tempsubshifts = np.zeros((nshifts, nchans, subnt), dtype=wd.dtype)
    cdef int bestshift, bestshifti=0, dt, ndirty=0
    cdef double error
    cdef np.ndarray[float64_t, ndim=1, mode='c'] sserrors
    sserrors = np.zeros(nshifts, dtype=np.float64) # sum of squared errors
    cdef int nbytessserrors = nshifts*sizeof(float64_t)
    cdef double tres = sort.tres
    cdef np.ndarray[int64_t, ndim=1, mode='c'] dirtysids = np.empty(nspikes, dtype=np.int64)
    #t0 = time.time()
    for sidi in range(nspikes):
        # pad start and end with first and last points per chan:
        sid = sids[sidi]
        for chani in range(maxnchans):
            for ti in range(nt):
                sd[chani, ti] = wd[sid, chani, ti] # sid's spike data
                widesd[chani, ti+MAXSHIFT] = sd[chani, ti] # 2D
            for ti in range(MAXSHIFT):
                widesd[chani, ti] = sd[chani, 0] # pad start with first point per chan
                widesd[chani, maxwidesdi-ti] = sd[chani, -1] # pad end with last point per chan

        # calculate sum of squared errors for all possible shifts of each spike:
        memset(&sserrors[0], 0, nbytessserrors) # clear sserrors
        for shifti in range(nshifts):
            for chani in range(nchans):
                spikechani = spikechanis[sidi, chani]
                for ti in range(subnt):
                    tempsubshifts[shifti, chani, ti] = widesd[spikechani, ti+ti0+shifti]
                    error = <double>tempsubshifts[shifti, chani, ti] - meandata[chani, ti]
                    sserrors[shifti] += error*error

        # find shift with smallest error:
        error = DBL_MAX
        for shifti in range(nshifts):
            if sserrors[shifti] < error:
                error = sserrors[shifti]
                bestshifti = shifti
        bestshift = shifts[bestshifti]
        if bestshift != 0: # no need to update sort.wavedata[sid] if there's no shift
            # update time values:
            dt = <int>round(bestshift * tres) # time to shift by, signed, in us
            ## TODO: update spikes array in pure C:
            spikes['t'][sid] += dt # should remain as ints between t0 and t1
            spikes['t0'][sid] += dt
            spikes['t1'][sid] += dt
            # might give out of bounds tis because the original peaks have shifted off the
            # ends. Use opposite sign because we're referencing within wavedata:
            spikes['tis'][sid] = spikes['tis'][sid] - bestshift
            # this in-place operation raises a TypeError in numpy 1.11.2, something related to
            # subtracting an int from an unsigned int:
            #spikes['tis'][sid] -= bestshift
            # update sort.wavedata
            for chani in range(maxnchans):
                for ti in range(nt):
                    wd[sid, chani, ti] = widesd[chani, ti+bestshifti]
            for chani in range(nchans):
                for ti in range(subnt):
                    shiftedsubsd[sidi, chani, ti] = tempsubshifts[bestshifti, chani, ti]
            dirtysids[ndirty] = sid # mark sid as dirty
            ndirty += 1
    #print('shifting loop took %.3f sec' % (time.time()-t0))
    AD2uV = sort.converter.AD2uV
    stdevbefore = AD2uV(subsd.std(axis=0).mean())
    stdevafter = AD2uV(shiftedsubsd.std(axis=0).mean())
    print('stdev went from %.3f to %.3f uV' % (stdevbefore, stdevafter))
    return dirtysids[:ndirty]


def intersect1d_uint8(arrs):
    """Find the intersection of any number of 1D arrays in arrs list.
    Return the sorted, unique values that are in all of the input arrays.
    This is a much faster (at least for many arrs) but type-specific version of
    core.intersect1d()"""
    cdef np.ndarray[uint8_t, ndim=1, mode='c'] arr
    cdef np.ndarray[uint8_t, ndim=1, mode='c'] common = np.unique(arrs[0])
    cdef int ncommon=common.shape[0]
    cdef int lenarr
    cdef int i, j, k
    cdef bint continuei=False
    #print('common: %s' % common)
    #print('ncommon: %d' % ncommon)
    for arr in arrs[1:]:
        lenarr = arr.shape[0]
        #print('arr: %s' % arr)
        i = 0 # reset
        while i < ncommon:
            #print('i = %d' % i)
            ## TODO: this naive search could be sped up by assuming sorted arrs, or by
            ## first sorting each arr with the quicksort algorithm, although that would
            ## modify each array in place, and so the caller would need to send a copy
            ## to prevent modifying the originals:
            for j in range(lenarr):
                #print('j = %d' % j)
                if common[i] == arr[j]:
                    #print('breaking out of j')
                    continuei = True
                    break # out of j loop
            if continuei:
                continuei = False # reset
                i += 1
                continue # to next i
            # never broke out of j loop, didn't find a value in current arr that matches
            # common[i], common[i] is no longer common:
            ncommon -= 1
            for k in range(i, ncommon):
                common[k] = common[k+1] # shift values above i down by 1
            #print('new common: %s' % common)
            #print('new ncommon: %d' % ncommon)
            # don't inc i, new value at common[i] has just shifted into view
    return common[:ncommon]
