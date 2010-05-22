"""Nick's gradient-ascent (mountain-climbing) algorithm"""

cimport cython
import numpy as np
cimport numpy as np

import random, time

cdef extern from "math.h":
    #double sqrt(double x)
    double fabs(double x)
    double exp(double x)

cdef extern from "stdio.h":
    int printf(char *, ...)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # might be necessary to release the GIL?
def climb(np.ndarray[np.float32_t, ndim=2] data,
          double sigma, double alpha, double rneighx=4, int subsample=1, int maxstill=100):
    """Implement Nick's gradient ascent (mountain climbing) clustering algorithm
    TODO:
        - delete scouts that have fewer than n points (at any point during iteration?)
        - reverse annealing sigma: starting small, and gradually increase it over iters
        - maybe some way of making sigma dynamic for each scout and for each iteration?
        - maybe annealing of alpha (decreasing it over time)? NVS sounds skeptical

        - to speed up the first massive merge step, maybe sort all data according to y position, and then take advantage of that somehow when checking distances - check the distance along y first...

        - classify obvious wide flat areas as noise points that shouldn't be clustered:
            - track the distance each point has travelled during the course of the algorithm. When done, plot the distribution of travel distances, and maybe you'll get something bimodal, and choose a cutoff travel distance past which any point that travelled further is considered a noise point
            - or maybe plot distribution of travel times
            - use some cutoff of local density to specify what's noise and what isn't? skeptical..
        - NVS - to weed out potential noise spikes, for each cluster, find the local density of the scout point at the max, then reject all other data points in that cluster whose local density falls below, say, 1% of the max. Apply it as a mask, so you can tweak that 1% value as you wish, without having to run the whole algorithm all over again

        - visualize algorithm in real time to see what exactly it's doing, and why some clusters are split while others are merged
        - instead of merging the higher indexed scout into the lower indexed one, you should really merge the one with the lower density estimate into the one with the higher density estimate - otherwise you potentially end up deleting the scout that's closer to the local max density, which probably sets you back several iterations
            - this would require storing all the density calculations - not a big deal
            - find whichever has the biggest density estimate - if it's not the lowest indexed scout (which will be the case 50% of the time), swap the entries in all the arrays (scouts, densities, still) except for the clusteris array, then proceed as usual
        - maybe to deal with clusters that are oversplit, look for pairs of scouts that are fairly close to each other, but most importantly, have lots and lots of points that butt up against those of the other scout

        - try using simplex algorithm for scout position update step, though that might miss local maxima
        - multithreading/multiprocessing
            - NVS thinks you could leave the merging step as a thread running in parallel with the gradient step (which itself could be split up easily into multiple threads) - although now with subsampling, the slow steps are mostly the gradient calcs, since there are far fewer scouts to merge to begin with

        - rescale all data by 2*sigma so you can get rid of the div by twosigma2 operation?
        - try using the n nearest neighbours to calculate gradient, instead of a guassian with a sigma. This makes it scale free, but NVS says this often results in situations where the gradient is 0 for some reason

        - scale x not just by its std, but also according to some absolute multiple of space (say 1.0 is 50 um), such that recordings with wider or narrower x locations (2 or 3 column probes) will cluster roughly as well with a constant sigma value (like 0.25, which really means you can expect up to 4 clusters along the x axis)

    DONE:
        - turn off checks for ZeroDivisionError, though I doubt that slows things down much
        - keep track of max movement on each iter, use consistently low max movement as
          automatic exit criteria
            - alternative: keep track of how long it's been since the last scout merger, and exit based on that
        - add freezing of points, for speed?
            - when a scout point has moved less than some distance per iteration for all of the last n iterations, freeze it. Then, in the position update loop, check for frozen scout points
        - add subsampling to reduce initial number of scout points


    """
    cdef int N = len(data) # total num data points
    cdef int ndims = data.shape[1] # num cols in data
    cdef np.ndarray[np.float32_t, ndim=2] alldata # used to store all data in case of sampling
    cdef int M # current num scout points (clusters)
    cdef int nsamples
    cdef np.ndarray[np.float32_t, ndim=2] scouts # stores scout positions
    cdef np.ndarray[np.int32_t, ndim=1] sampleis # potentially, indices sumsampling data
    cdef np.ndarray[np.int32_t, ndim=1] clusteris = np.zeros(N, dtype=np.int32) # cluster indices into data
    cdef np.ndarray[np.uint8_t, ndim=1] still = np.zeros(N, dtype=np.uint8) # for each scout, num consecutive iters without significant movement
    cdef double sigma2 = sigma**2
    cdef double twosigma2 = 2 * sigma2
    cdef double rmerge = sigma # radius within which scout points are merged
    cdef double rmerge2 = rmerge**2
    cdef int nneighs # num points in vicinity of scout point
    cdef double rneigh = rneighx * sigma # radius around scout to include data for gradient calc
    cdef double rneigh2 = rneigh**2
    cdef double diff, diff2sum, mindiff2sum, move, movesum
    cdef double minmovesum = 0.00001 * sigma * ndims # maybe this should depend on alpha too, and if proper sum of squares distance was calculated, it wouldn't have to depend on ndims
    cdef np.ndarray[np.float64_t, ndim=1] diffs = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] diffs2 = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros(ndims)
    cdef Py_ssize_t i, j, k, samplei, scouti, clustii
    cdef int iteri = 0, continuej = 0, merged = 0, nnomerges = 0, maxnnomerges = 1000

    if subsample > 1:
        # subsample to get a reasonable number of scouts
        nsamples = N / subsample # this will trunc
        sampleis = np.asarray(random.sample(xrange(N), nsamples))
        M = nsamples # initially, but M will decrease over time
        alldata = data.copy()
        data = alldata[sampleis]
        scouts = data.copy() # scouts will be modified
        clusteris.fill(-1) # -ve number indicates an unclustered data point
        clusteris[sampleis] = np.arange(M)
    else:
        nsamples = N
        M = N
        scouts = data.copy() # scouts will be modified
        clusteris = np.arange(N)


    while True:

        if nnomerges == maxnnomerges:
            break

        # merge pairs of scout points sufficiently close to each other
        merged = 0
        for i in range(M):
            # M may be decr in this loop, so this condition may
            # be reached before this loop completes
            if i >= M: break # out of for loop
            for j in range(i+1, M):
                if j >= M: break # out of for loop
                if still[i] > maxstill and still[j] > maxstill: # both scouts are frozen
                    continue
                # for each pair of scouts, check if any pair is within rmerge of each other
                diff2sum = 0.0 # reset
                for k in range(ndims):
                    diff = fabs(scouts[i, k] - scouts[j, k])
                    if diff > rmerge: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    else:
                        diff2sum += diff * diff
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next j loop
                if diff2sum <= rmerge2:
                    # merge the scouts: keep scout i, ditch scout j
                    # shift all entries at j and above in scouts array down by one
                    for scouti in range(j, M-1):
                        for k in range(ndims):
                            scouts[scouti, k] = scouts[scouti+1, k]
                        still[scouti] = still[scouti+1] # ditto for still array
                    # update cluster indices
                    for clustii in range(N):
                        if clusteris[clustii] == j:
                            clusteris[clustii] = i # overwrite all occurences of j with i
                        elif clusteris[clustii] > j:
                            clusteris[clustii] -= 1 # decr all clust indices above j
                    M -= 1 # decr num of scouts (clusters)
                    #printf(' %d<-%d ', i, j)
                    printf('M')
                    merged = 1
        if merged == 0:
            nnomerges += 1
        else:
            nnomerges = 0

        # move each scout point up its local gradient
        for i in range(M): # iterate over all scout points
            # skip frozen scout points
            if still[i] == maxstill:
                continue
            # measure gradient
            nneighs = 0 # reset
            for k in range(ndims):
                v[k] = 0.0 # reset
            for j in range(nsamples): # iterate over sampled data, check if they're within rneigh
                diff2sum = 0.0 # reset
                for k in range(ndims): # iterate over dims for each point
                    diffs[k] = data[j, k] - scouts[i, k]
                    if fabs(diffs[k]) > rneigh: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    else:
                        diffs2[k] = diffs[k] * diffs[k] # used twice, so calc it only once
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
            movesum = 0.0 # reset
            for k in range(ndims):
                move = alpha / nneighs * v[k]
                movesum += fabs(move) # to save time, just take sum instead of sum of squares
                scouts[i, k] += move
            #printf(',%f', movesum)
            if movesum < minmovesum:
                still[i] += 1
            else:
                still[i] = 0 # reset stillness counter for this scout

        printf('.')
        iteri += 1


    if subsample > 1:
        # for each unclusterd point, find the closest clustered point, and assign
        # it to the same cluster
        # TODO: this seems quite slow. Optimize somehow? Maybe swap inner and outer loops?
        print('Finding nearest clustered points for each unclustered point')
        t0 = time.clock()
        for j in range(N): # iterate over all data points
            if clusteris[j] > -1: # point already has a valid cluster index
                continue
            # point is unclustered, find nearest clustered point
            mindiff2sum = 100e99
            for i in range(nsamples): # iterate over all clustered points
                # sampleis is an array of nsamples indices into data that were used as scouts,
                # and therefore have been clustered
                samplei = sampleis[i]
                diff2sum = 0.0 # reset
                for k in range(ndims):
                    diff = alldata[j, k] - alldata[samplei, k]
                    diff2sum += diff * diff
                if diff2sum < mindiff2sum:
                    # update this unclustered point's cluster index
                    mindiff2sum = diff2sum
                    clusteris[j] = clusteris[samplei]
        print('Assigning unclustered points took %.3f sec' % (time.clock()-t0))


    nmoving = (still[:M] < maxstill).sum()
    print('\nniters: %d' % iteri)
    print('nscouts: %d' % M)
    print('nmoving: %d, minmovesum: %f' % (nmoving, minmovesum))
    print('sigma: %.2f, rneigh: %.2f, rmerge: %.2f, alpha: %.2f' % (sigma, rneigh, rmerge, alpha))
    print('still array:')
    print still[:M]
    return clusteris, scouts[:M]

