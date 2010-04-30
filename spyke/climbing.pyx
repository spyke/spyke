"""Some functions written in Cython for max performance"""
cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    #double sqrt(double x)
    #double abs(double x)
    double exp(double x)

cdef extern from "stdio.h":
    int printf(char *, ...)


@cython.boundscheck(False)
@cython.wraparound(False)
def gradient_ascent(np.ndarray[np.float64_t, ndim=2] data,
                    double sigma, double alpha):
    """Implement Nick's gradient ascent (mountain climbing) algorithm
    TODO: keep track of max movement on each iter, use consistently low max movement as
          automatic exit criteria
    TODO: add some of Nick's optimizations
    TODO: add freezing of points, for speed and also to leave noise points unclustered
    TODO: reverse annealing by starting with small sigma, and gradually increasing it over iters
    TODO: maybe annealing of alpha (decreasing it over time)?
    TODO: multithreading/multiprocessing
    """
    cdef int N = len(data) # total num data points
    cdef int ndims = data.shape[1] # num cols in data
    cdef np.ndarray[np.float64_t, ndim=2] scouts = data.copy() # scouts will be modified
    cdef int M = N # current num scout points (clusters)
    cdef np.ndarray[np.int32_t, ndim=1] clusteris = np.arange(N) # cluster indices into data
    cdef double sigma2 = sigma**2
    cdef double twosigma2 = 2 * sigma2
    cdef double r = sigma / 2.0 # radius within which scout points are merged
    cdef double r2 = r**2
    cdef int nneighs # num points in vicinity of scout point
    cdef double rneigh = 5 * sigma # radius around scout to include data for gradient calc
    cdef double rneigh2 = rneigh**2
    cdef double diff2sum = 0
    cdef np.ndarray[np.float64_t, ndim=1] diffs = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] diffs2 = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros(ndims)
    cdef Py_ssize_t i, j, k, iteri, scouti, clustii
    cdef int continuej = 0

    #while True:
    for iteri in range(2500):

        # merge those scout points sufficiently close to each other
        for i in range(M):
            # M may be decr in this loop, so this condition may
            # be reached before this loop completes
            if i >= M: break # out of for loop
            for j in range(i+1, M):
                if j >= M: break # out of for loop
                # for each pair of scouts, check if any pair is within r of each other
                diff2sum = 0 # reset
                for k in range(ndims):
                    diff2sum += (scouts[i, k] - scouts[j, k])**2
                if diff2sum <= r2:
                    # merge the scouts: keep scout i, ditch scout j
                    # shift all entries at j and above in scouts array down by one
                    for scouti in range(j, M-1):
                        for k in range(ndims):
                            scouts[scouti, k] = scouts[scouti+1, k]
                    # update cluster indices
                    for clustii in range(N):
                        if clusteris[clustii] == j:
                            clusteris[clustii] = i # overwrite all occurences of j with i
                        elif clusteris[clustii] > j:
                            clusteris[clustii] -= 1 # decr all clust indices above j
                    M -= 1 # decr num of scouts (clusters)
                    printf(' %d<-%d ', i, j)

        # move each scout point up its local gradient
        for i in range(M): # iterate over all scout points
            # measure gradient
            nneighs = 0 # reset
            for k in range(ndims):
                v[k] = 0 # reset
            for j in range(N): # iterate over all data points, check if they're within rneigh
                diff2sum = 0 # reset
                for k in range(ndims): # iterate over dims for each point
                    diffs[k] = data[j, k] - scouts[i, k]
                    diffs2[k] = diffs[k]**2 # square it instead of absing it
                    if diffs2[k] > rneigh2: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    diff2sum += diffs2[k]
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next j loop
                if diff2sum <= rneigh2: # do the calculation
                    for k in range(ndims):
                        # v is ndim vector of sum of g-weighted distances between
                        # current scout point and all data within rneigh
                        v[k] += diffs[k] * exp(-diffs2[k] / twosigma2)
                    nneighs += 1
            # update scout position in direction of v, normalize by nneighs
            if nneighs != 0:
                for k in range(ndims):
                    scouts[i, k] += alpha / nneighs * v[k]
            # else: freeze this scout point, since it'll never move again?

        printf('.')

    return clusteris, scouts[:M]

