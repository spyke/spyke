import numpy as np
cimport numpy as np

import time

cdef short select(short *a, int l, int r, int k):
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


cdef double mean(short *a, int N):
    cdef Py_ssize_t i
    cdef double s=0
    for i in range(N):
        s += a[i]
    s /= N
    return s


def mean2(np.ndarray[np.int16_t, ndim=1] a):
    cdef Py_ssize_t i
    cdef double s=0
    for i in range(a.shape[0]):
        s += a[i]
    s /= a.shape[0]
    return s


#def selectpy(ndarray data, int k):
def selectpy(np.ndarray[np.int16_t, ndim=1] data):
    """Test Cython select() from Python

    data = np.float32(np.random.rand(10000000))
    data = np.array([1., 2., 2., 3., 1., 3., 5.], dtype=np.float32)
    """
    print 'data:', data
    cdef int N = len(data)
    cdef int k = (N-1) // 2 # this gives us exact median for odd N

    tsort = time.clock()
    sorteddata = np.sort(data) # copy, do a full quicksort
    print 'np.sort:', sorteddata[k]
    print 'np.sort took %.3f sec' % (time.clock()-tsort)

    tmed = time.clock()
    result = np.median(data) # doesn't modify data, does internal copy
    print 'np.median:', result
    print 'np.median took %.3f sec' % (time.clock()-tmed)

    tmean = time.clock()
    result = np.mean(data) # doesn't modify data
    print 'np.mean:', result
    print 'np.mean took %.3f sec' % (time.clock()-tmean)

    tmean = time.clock()
    cdef short *d = <short *>data.data
    result = mean(d, N) # doesn't modify data
    print 'Cython mean:', result
    print 'Cython mean took %.3f sec' % (time.clock()-tmean)

    tmean = time.clock()
    result = mean2(data) # doesn't modify data
    print 'Cython mean2:', result
    print 'Cython mean2 took %.3f sec' % (time.clock()-tmean)

    tcy = time.clock()
    cdef np.ndarray[np.int16_t, ndim=1] newdata = data.copy() # copy, select modifies in-place
    print 'copy took %.3f sec' % (time.clock()-tcy)
    cdef short *a = <short *>newdata.data # int pointer to newdata's .data field
    print 'Cython median:', select(a, 0, N-1, k)
    print 'Cython median took %.3f sec' % (time.clock()-tcy)

    print 'data:', data
    print 'newdata:', newdata
