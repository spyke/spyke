"""Some functions written in Cython for max performance"""
cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    #double sqrt(double x)
    double abs(double x)
    double exp(double x)

cdef extern from "stdio.h":
    int printf(char *, ...)


@cython.boundscheck(False)
@cython.wraparound(False)
def climb(np.ndarray[np.float64_t, ndim=2] data, double sigma, double alpha):
    """Implement Nick's gradient ascent (mountain climbing) clustering algorithm
    TODO:
        - keep track of max movement on each iter, use consistently low max movement as
          automatic exit criteria
        - add freezing of points, for speed and also to leave noise points unclustered?
            - when a scout point has moved less than some distance per iteration for all of the last n iterations, freeze it. Then, in the position update loop, check for frozen scout points
        - delete scouts that have fewer than n points (at any point during iteration?)
        - reverse annealing sigma: starting small, and gradually increase it over iters
        - maybe some way of making sigma dynamic for each scout and for each iteration?
        - maybe annealing of alpha (decreasing it over time)? NVS sounds skeptical
        - classify obvious wide flat areas as noise points that shouldn't be clustered:
            - track the distance each point has travelled during the course of the algorithm. When done, plot the distribution of travel distances, and maybe you'll get something bimodal, and choose a cutoff travel distance past which any point that travelled further is considered a noise point
            - or maybe plot distribution of travel times

        - visualize algorithm in real time to see what exactly it's doing, and why some clusters are split while others are merged

        - try using simplex algorithm for scout position update step, though that might miss local maxima
        - multithreading/multiprocessing
        - turn off checks for ZeroDivisionError, though I doubt that slows things down much

        - add subsampling to reduce initial number of scout points
        - instead of subsampling or duplicating all data as scouts, try starting with a regularly spaced grid of scout points, spaced at intervals slightly less than the smallest cluster. This will especially speed things up for really big data sets, where you'd otherwise start with an enormous number of scouts

        - rescale all data by 2*sigma so you can get rid of the div by twosigma2 operation?
        - try using the n nearest neighbours to calculate gradient, instead of a guassian with a sigma. This makes it scale free, but NVS says this often results in situations where the gradient is 0 for some reason
    """
    cdef int N = len(data) # total num data points
    cdef int ndims = data.shape[1] # num cols in data
    cdef np.ndarray[np.float64_t, ndim=2] scouts = data.copy() # scouts will be modified
    cdef int M = N # current num scout points (clusters)
    cdef np.ndarray[np.int32_t, ndim=1] clusteris = np.arange(N) # cluster indices into data
    cdef np.ndarray[np.int32_t, ndim=1] frozen = np.zeros(N, dtype=np.int32) # frozen states
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
    for iteri in range(500):

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
                    '''
                    diff = abs(scouts[i, k] - scouts[j, k])
                    if diff > r: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    else:
                        diff2sum += diff**2
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next j loop
                    '''
                if diff2sum <= r2:
                    # merge the scouts: keep scout i, ditch scout j
                    # shift all entries at j and above in scouts array down by one
                    for scouti in range(j, M-1):
                        for k in range(ndims):
                            scouts[scouti, k] = scouts[scouti+1, k]
                        #frozen[scouti] = frozen[scouti+1] # ditto for frozen array
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
            # skip frozen scout points
            #if frozen[i] == 1:
            #    continue
            # measure gradient
            nneighs = 0 # reset
            for k in range(ndims):
                v[k] = 0 # reset
            for j in range(N): # iterate over all data points, check if they're within rneigh
                diff2sum = 0 # reset
                for k in range(ndims): # iterate over dims for each point
                    diffs[k] = data[j, k] - scouts[i, k]
                    if abs(diffs[k]) > rneigh: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    diffs2[k] = diffs[k]**2 # used twice, so calc it only once
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
            # nneighs will never be 0, because each scout point starts as a data point
            for k in range(ndims):
                scouts[i, k] += alpha / nneighs * v[k]

        printf('.')

    return clusteris, scouts[:M]

