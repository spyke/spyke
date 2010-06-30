"""Some functions written in Cython for max performance"""
cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double abs(int x)
    double fabs(float x)

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
def argsharpness2D(np.ndarray[np.int16_t, ndim=2] signal):
    """Spike phase sharpness measure which takes (accumulated height)**2 / width
    for each phase, and relies on zero crossings to demarcate borders between phases.

    TODO: might also try adding mode='c' kwarg to signal arg, if you know it's C contig,
    reduces need to do stride calc on each access. Actually, might try adding mode='c'
    to all locally declared np arrays as well.

    DONE: instead of simply taking extremum value and dividing by width, accumulate the
    change in signal in the correct direction on either side of the extremum. This way,
    an extremum with a long amount of signal leading up to it and away from it on either
    side is deemed sharper than one that only has a lead up on one side (say due to falling
    near the border of the waveform)

    DONE: you only really need to do this accumulation thing for segments that fall at
    the ends of the waveform. For all the rest, you can just take 2*(extremum value). Duh. This will be more accurate too, since you won't be relying on getting segment edge points that
    are really close to crossing 0

    TODO: use FWHM instead of npoints of each segment to more accurately determine width.
    This will require at least some linear interpolation though between points straddling
    the half max level on either side of each extremum.

    DONE: weight each accumulation of phase height value by the abs(extremum), so when you have
    big and small phases of similar shape, the big ones are considered sharper

    """
    cdef Py_ssize_t ci, ti, nchans, nt, segi, nseg
    cdef short last, now, next, ext
    nchans = signal.shape[0]
    nt = signal.shape[1]
    cdef np.ndarray[np.int16_t, ndim=2] extti = np.zeros((nchans, nt), dtype=np.int16)
    cdef np.ndarray[np.int16_t, ndim=2] npoints = np.zeros((nchans, nt), dtype=np.int16)
    cdef np.ndarray[np.int16_t, ndim=1] nsegments = np.zeros(nchans, dtype=np.int16)
    cdef np.ndarray[np.float64_t, ndim=2] sharp = np.zeros((nchans, nt))

    for ci in range(nchans):
        segi = 0 # segment index, segments bound by 0 crossings, endpoints count as 0 crossings
        ext = 0 # biggest max/min so far for current segment
        npoints[ci, segi] = 1 # count signal startpoint of first segment
        for ti in range(1, nt-1): # start at 2nd ti, go to 2nd last ti
            now = signal[ci, ti]
            last = signal[ci, ti-1]
            next = signal[ci, ti+1]
            if (last > 0) != (now > 0): # crossed 0 between last and now
                segi += 1 # each segment between 0 crossings has 1 extremum
                ext = 0 # reset biggest max/min so far for new segment
            npoints[ci, segi] += 1 # inc npoints in current segment
            if (last < now > next and now > 0) or (last > now < next and now < 0):
                # found a local max or min
                if abs(now) > abs(ext): # found new biggest max/min so far for this segment
                    extti[ci, segi] = ti # store its timepoint
                    ext = now # update for this segment
        npoints[ci, segi] += 1 # count signal endpoint of last segment
        nseg = segi + 1
        nsegments[ci] = nseg
        # calc sharpness for each extremum (one per segment), maintain extremum sign
        for segi in range(nseg):
            ti = extti[ci, segi] # ti for extremum in this segment
            if ti == 0 or ti == nt-1: # "extremum" is on signal edge, don't consider it
                continue
            # store sum of heights of left and right sides of each phase
            sharp[ci, segi] = 2*signal[ci, ti]
            if segi == 0: # first segment
                # left segment edge == left waveform edge, left side is shorter than usual
                sharp[ci, segi] -= signal[ci, 0]
            elif segi == nseg-1: # last segment
                # right segment edge == right waveform edge, right side is shorter than usual
                sharp[ci, segi] -= signal[ci, nt-1]
            # weight by sum of heights value, normalize by phase width
            sharp[ci, segi] *= fabs(sharp[ci, segi]) # maintain extremum sign
            sharp[ci, segi] /= npoints[ci, segi]

    return extti, npoints, sharp, nsegments
