"""Some functions written in Cython for max performance"""
cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    int abs(int x)
    float fabs(float x)

#cdef extern from "stdio.h":
#    int printf(char *, ...)


cdef short select_short(short *a, int l, int r, int k):
    """Returns the k'th (0-based) ranked entry from float array a within left
    and right pointers l and r. This is quicksort partitioning based
    selection, taken from Sedgewick (Algorithms, 2ed 1988, p128).
    Note that this modifies a in-place"""
    cdef int i, j
    cdef short v, temp
    if r < l:
        raise ValueError, 'bad pointer range in select()'
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

@cython.boundscheck(False)
@cython.wraparound(False)
def median_inplace_2Dshort(np.ndarray[np.int16_t, ndim=2, mode='c'] arr):
    """Assumes C-contig 2D input array. arr will probably be from a copy anyway,
    since it modifies in-place"""
    cdef Py_ssize_t nchans, nt, k, i
    cdef np.ndarray[np.int16_t, ndim=1] result
    cdef short *a
    nchans = arr.shape[0]
    nt = arr.shape[1]
    result = np.zeros(nchans, dtype=np.int16)
    k = (nt-1) // 2
    a = <short *>arr.data # short pointer to arr's .data field
    for i in range(nchans):
        result[i] = select_short(a, i*nt, i*nt+nt-1, i*nt+k) # this won't work for strided stuff
    return result


'''
cdef double mean(short *a, int N):
    cdef Py_ssize_t i # recommended type for looping
    cdef double s=0
    for i in range(N):
        s += a[i]
    s /= N
    return s


def mean2(np.ndarray[np.int16_t, ndim=1] a):
    """Uses new simpler numpy type notation for fast indexing, but is still a
    bit slower than the classical way, because you currently can't
    use the new notation with cdefs"""
    cdef Py_ssize_t i
    cdef double s=0
    for i in range(a.shape[0]):
        s += a[i]
    s /= a.shape[0]
    return s
'''

def mean_2Dshort(np.ndarray[np.int16_t, ndim=2] a):
    """Uses new simpler numpy type notation for fast indexing, but is still a
    bit slower than the classical way, because you currently can't
    use the new notation with cdefs"""
    cdef Py_ssize_t i, j, nchans, nt
    nchans = a.shape[0]
    nt = a.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] s = np.zeros(nchans)
    for i in range(nchans):
        for j in range(nt):
            s[i] += a[i, j]
        s[i] /= nt # normalize
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # might be necessary to release the GIL?
def sharpness2D(np.ndarray[np.int16_t, ndim=2] signal):
    """Spike phase sharpness measure which takes (accumulated height)**2 / width
    for each phase, and relies on zero crossings to demarcate borders between phases.
    First, update npoints, check for extremum and update ext. Then, then look forward
    for 0-crossing or end of signal, and calc sharpness if you find either is the case.

    TODO: test if double math is faster than float math. They're probably identical.

    TODO: might also try adding mode='c' kwarg to signal arg, if you know it's C contig,
    reduces need to do stride calc on each access. Actually, might try adding mode='c'
    to all locally declared np arrays as well.

    TODO: use FWHM instead of npoints of each segment to more accurately determine width.
    This will require at least some linear interpolation though between points straddling
    the half max level on either side of each extremum.

    TODO: do I really need to check for an extremum between each 0 crossing? I think not.
    Just find the max abs between zero crossings. Also, don't need to check sign, since sign
    will always alternate anyway. Trouble is at the endpoints, where you don't
    have a 0 crossing, and you need to actually check if an extremum was found between the
    last 0 crossing and the end of the signal (or vice versa). But that can be done easily
    enough by checking to the left and right of the max abs found in that last segment,
    and deciding if it represents an extremum.

    DONE: instead of simply taking extremum value and dividing by width, accumulate the
    change in signal in the correct direction on either side of the extremum. This way,
    an extremum with a long amount of signal leading up to it and away from it on either
    side is deemed sharper than one that only has a lead up on one side (say due to falling
    near the border of the signal)

    DONE: you only really need to do this accumulation thing for segments that fall at
    the ends of the signal. For all the rest, you can just take 2*(extremum value). Duh.
    This will be more accurate too, since you won't be relying on getting segment edge
    points that are really close to crossing 0.

    DONE: weight each accumulation of phase height value by the abs(extremum), so when you have
    big and small phases of similar shape, the big ones are considered sharper

    """
    cdef Py_ssize_t nchans, nt, ci, ti, extti, npoints, maxti
    cdef bint seg0=True, cross=False
    cdef short last, now, next, sig0
    cdef float ext

    nchans = signal.shape[0]
    nt = signal.shape[1]
    maxti = nt-2
    cdef np.ndarray[np.float32_t, ndim=2] sharp = np.zeros((nchans, nt), dtype=np.float32)

    assert nt < 2**31 # make sure time indices don't overflow

    for ci in range(nchans):
        ext = 0.0 # val of biggest extremum so far for current segment
        extti = 0 # ti of biggest extremum so far for current segment
        npoints = 0 # npoints in current segment
        sig0 = signal[ci, 0]
        now = sig0 # init
        next = sig0 # init
        for ti in range(nt-1):
            last = now # last = signal[ci, ti-1], except when ti==0: last = signal[ci, 0]
            now = next # now = signal[ci, ti]
            next = signal[ci, ti+1]
            npoints += 1 # inc for this segment, corresponds to "now" point in segment
            #print('ti=%d, npoints=%d' % (ti, npoints))
            if (last < now > next and now > 0) or (last > now < next and now < 0):
                #print('found a local extremum of appropriate sign')
                if abs(now) > fabs(ext): # found new biggest extremum so far for this segment
                    extti = ti # store its timepoint
                    ext = now # update for this segment
                    #print('found new biggest local ext=%f at ti=%d' % (ext, extti))
            cross = (now > 0) != (next > 0) # 0-crossing coming up?
            if cross or ti == maxti: # both might happen simultaneously
                # 0-cross coming up, or at end of signal. ti is last timepoint in segment,
                # but if we're at end of signal, ti+1 is last timepoint in segment, and
                # needs to be counted in npoints.
                # calculate sharpness of extremum in this segment
                #print('reached end of segment')
                if seg0: # we're on first segment
                    # left segment edge == left signal edge, left side is shorter than usual
                    if ext == 0.0: # leave untouched if 0, don't have extremum to store
                        extti = 0 # harmlessly write 0 to first entry in sharp
                    else:
                        ext -= <float>sig0 / 2 # penalize
                    seg0 = False
                elif not cross: # we're on last segment, bound only by signal, not true 0-cross
                    # right segment edge == right signal edge, right side is shorter than usual
                    if ext == 0.0: # leave untouched if 0, don't have extremum to store
                        extti = nt-1 # harmlessly write 0 to last entry in sharp
                    else:
                        ext -= <float>signal[ci, nt-1] / 2 # penalize
                        npoints += 1 # count next'th point as well for this last segment
                #print('using npoints=%d for sharpness calc' % npoints)
                # square height, normalize by phase width
                ext *= fabs(ext) # maintain extremum sign
                ext /= npoints
                sharp[ci, extti] = ext # store
                ext = 0.0 # reset biggest max/min so far for new segment
                npoints = 0 # reset for new segment

    return sharp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # might be necessary to release the GIL?
def argthreshsharp(np.ndarray[np.int16_t, ndim=2] signal,
                   np.ndarray[np.int16_t, ndim=1] thresh,
                   np.ndarray[np.float32_t, ndim=2] sharp):
    """Given original signal, threshold array, and sharpness array,
    returns a temporally sorted n x 2 (ti, ci) array of peaks that exceed
    thresh for the appropriate chan"""

    cdef Py_ssize_t nt, chans, ti, ci, npeaks = 0

    assert signal.shape[1] < 2**31 # stick to int32 time indices
    nchans = signal.shape[0]
    nt = signal.shape[1]
    assert sharp.shape[0] == nchans
    assert sharp.shape[1] == nt
    assert len(thresh) == nchans

    # worst case scenario: we find as many thresh exceeding peaks as nt
    cdef np.ndarray[np.int32_t, ndim=2] peakis = np.empty((nt, 2), dtype=np.int32)

    for ti in range(nt):
        for ci in range(nchans):
            if sharp[ci, ti] != 0.0 and abs(signal[ci, ti]) >= thresh[ci]:
                peakis[npeaks, 0] = ti
                peakis[npeaks, 1] = ci
                npeaks += 1

    return peakis[:npeaks]
