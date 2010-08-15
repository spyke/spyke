"""Nick's gradient-ascent (mountain-climbing) clustering algorithm"""

cimport cython
import numpy as np
cimport numpy as np

import random, time
#from scipy.spatial import KDTree

cdef extern from "math.h":
    double sqrt(double x)
    double fabs(double x)
    double exp(double x)

cdef extern from "stdio.h":
    int printf(char *, ...)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # might be necessary to release the GIL?
def climb(np.ndarray[np.float32_t, ndim=2] data,
          np.ndarray[np.int32_t, ndim=1] sampleis=np.zeros(0, dtype=np.int32),
          double sigma=0.25, double alpha=1.0, double rmergex=1.0,
          double rneighx=4, int nsamples=0,
          bint calcpointdensities=True, bint calcscoutdensities=True,
          double minmove=-1.0, int maxstill=100, int maxnnomerges=1000,
          int minpoints=10):
    """Implement Nick's gradient ascent (mountain climbing) clustering algorithm
    TODO:
        - reverse annealing sigma: starting small, and gradually increase it over iters
            - increase it a bit every time you get an iteration with no mergers?
        - maybe some way of making sigma dynamic for each scout and for each iteration?
        - maybe annealing of alpha (decreasing it over time)? NVS sounds skeptical

        - to speed up the first massive merge step, maybe sort all data according to y position, and then take advantage of that somehow when checking distances - check the distance along y first...

        - classify obvious wide flat areas as noise points that shouldn't be clustered:
            - track the distance each point has travelled during the course of the algorithm. When done, plot the distribution of travel distances, and maybe you'll get something bimodal, and choose a cutoff travel distance past which any point that travelled further is considered a noise point
            - or maybe plot distribution of travel times
            - use some cutoff of local density to specify what's noise and what isn't? skeptical..

        - visualize algorithm in real time to see what exactly it's doing, and why some clusters are split while others are merged

        - instead of merging the higher indexed scout into the lower indexed one, you should really merge the one with the lower density estimate into the one with the higher density estimate - otherwise you potentially end up deleting the scout that's closer to the local max density, which probably sets you back several iterations
            - this would require calc'ing and storing the density for each cluster, and updating it every time it moves
                - is local density and local gradient calc sufficiently similar that this won't be expensive?
            - find whichever has the biggest density estimate - if it's not the lowest indexed scout (which will be the case 50% of the time), swap the entries in all the arrays (scouts, densities, still) except for the cids array, then proceed as usual. Then update the density for the newly merged cluster

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
        - NVS - to weed out potential noise spikes, for each cluster, find the local density of the scout point at the max, then reject all other data points in that cluster whose local density falls below, say, 1% of the max. Apply it as a mask, so you can tweak that 1% value as you wish, without having to run the whole algorithm all over again
        - delete scouts that have fewer than n points (at any point during iteration?)
    """
    cdef int N = len(data) # total num data points
    cdef int ndims = data.shape[1] # num cols in data
    cdef int M # current num scout points (clusters)
    cdef int npoints, nremoved
    cdef np.ndarray[np.float32_t, ndim=2] scouts # stores scout positions
    cdef np.ndarray[np.int32_t, ndim=1] cids = np.zeros(N, dtype=np.int32) # cluster indices into data
    cdef np.ndarray[np.uint8_t, ndim=1] still = np.zeros(N, dtype=np.uint8) # for each scout, num consecutive iters without significant movement
    cdef double sigma2 = sigma * sigma
    cdef double twosigma2 = 2 * sigma2
    cdef double rmerge = rmergex * sigma # radius within which scout points are merged
    cdef double rmerge2 = rmerge * rmerge
    cdef int nneighs # num points in vicinity of scout point
    cdef double rneigh = rneighx * sigma # radius around scout to include data for gradient calc
    cdef double rneigh2 = rneigh * rneigh
    cdef double d, d2, min_d2, move
    cdef np.ndarray[np.float64_t, ndim=1] ds = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] d2s = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros(ndims)
    cdef np.ndarray[np.float64_t, ndim=1] densities = np.zeros(0), scoutdensities = np.zeros(0)
    cdef Py_ssize_t i, j, k, samplei, scouti, clustii
    cdef int iteri = 0, continuej = 0, merged = 0, nnomerges = 0
    cdef bint incstill

    if len(sampleis) != 0: # sampleis arg trumps nsamples arg
        nsamples = len(sampleis)
    elif 0 < nsamples < N: # nsamples == 0 means use all points
        # subsample with nsamples to get a reasonable number of scouts
        sampleis = np.asarray(random.sample(xrange(N), nsamples))
    else: # nsamples == 0, or nsamples >= N, use all N points
        nsamples = N
        sampleis = np.arange(nsamples)
    M = nsamples # initially, but M will decrease over time
    scouts = data[sampleis].copy() # scouts will be modified
    cids.fill(-1) # -ve number indicates an unclustered data point
    cids[sampleis] = np.arange(M)

    if minmove == -1.0:
        minmove = 0.000001 * sigma * alpha # along a single dimension

    while True:

        if nnomerges == maxnnomerges:
            break

        # merge pairs of scout points sufficiently close to each other
        merged = 0
        i = 0
        while i < M:
            j = i+1
            while j < M:
                if still[i] > maxstill and still[j] > maxstill: # both scouts are frozen
                    j += 1
                    continue
                # for each pair of scouts, check if any pair is within rmerge of each other
                d2 = 0.0 # reset
                for k in range(ndims):
                    d = fabs(scouts[i, k] - scouts[j, k])
                    if d > rmerge: # break out of k loop, continue to next j
                        continuej = 1
                        break # out of k loop
                    d2 += d * d
                if continuej == 1:
                    continuej = 0 # reset
                    j += 1
                    continue # to next j loop
                if d2 <= rmerge2:
                    # merge the scouts: keep scout i, ditch scout j
                    # shift all entries at j and above in scouts array down by one
                    for scouti in range(j, M-1):
                        for k in range(ndims):
                            scouts[scouti, k] = scouts[scouti+1, k]
                        still[scouti] = still[scouti+1] # ditto for still array
                    # update cluster indices
                    for clustii in range(N):
                        if cids[clustii] == j:
                            cids[clustii] = i # overwrite all occurences of j with i
                        elif cids[clustii] > j:
                            cids[clustii] -= 1 # decr all clust indices above j
                    M -= 1 # decr num of scouts, don't inc j, new value at j has just slid into view
                    #printf(' %d<-%d ', i, j)
                    printf('M')
                    merged = 1
                else:
                    j += 1
            i += 1
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
                samplei = sampleis[j]
                d2 = 0.0 # reset
                for k in range(ndims): # iterate over dims for each point
                    ds[k] = data[samplei, k] - scouts[i, k]
                    if fabs(ds[k]) > rneigh: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    d2s[k] = ds[k] * ds[k] # used twice, so calc it only once
                    d2 += d2s[k]
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next j
                if d2 <= rneigh2: # do the calculation
                    for k in range(ndims):
                        # v is ndim vector of sum of g-weighted distances between
                        # current scout point and all data within rneigh
                        v[k] += ds[k] * exp(-d2s[k] / twosigma2)
                    nneighs += 1
            # update scout position in direction of v, normalize by nneighs
            # nneighs will never be 0, because each scout point starts as a data point
            incstill = True # reset
            for k in range(ndims):
                move = alpha / nneighs * v[k]
                scouts[i, k] += move
                if incstill and fabs(move) > minmove:
                    incstill = False
            if incstill:
                still[i] += 1
            else:
                still[i] = 0 # reset stillness counter for this scout

        printf('.')
        iteri += 1

    printf('\n')

    if nsamples != N: # if we're subsampling
        # for each unclusterd point, find the closest clustered point, and assign
        # it to the same cluster
        print('Finding nearest clustered point for each unclustered point')
        t0 = time.time()
        '''
        # Tried using kdtree for speed, results aren't right, and it's much slower:
        kdtree = KDTree(data) # tree of clustered data points
        # pick out unclustered data points
        print('picking out unclustered data')
        uciis = cids == -1
        ucdata = data[uciis]
        # query the tree for the closest clustered point for each unclustered data point
        print('calling query')
        nnd, nni = kdtree.query(ucdata)
        print('assigning clusters')
        cids[uciis] = cids[nni]
        '''
        for i in range(N): # iterate over all data points
            if cids[i] != -1: # point already has a valid cluster index
                continue
            # point is unclustered, find nearest clustered point
            min_d2 = 100e99
            for j in range(nsamples): # iterate over all clustered points
                # sampleis is an array of nsamples indices into data that were used as scouts,
                # and therefore have been clustered
                samplei = sampleis[j]
                for k in range(ndims):
                    ds[k] = data[i, k] - data[samplei, k]
                    if fabs(ds[k]) > min_d2: # break out of k loop, continue to next i
                        continuej = 1
                        break # out of k loop
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next i
                d2 = 0.0 # reset
                for k in range(ndims):
                    d2 += ds[k] * ds[k]
                if d2 < min_d2:
                    # update this unclustered point's cluster index
                    min_d2 = d2
                    cids[i] = cids[samplei]
        print('Assigning unclustered points took %.3f sec' % (time.time()-t0))

    # remove clusters with less than minpoints number of points
    nremoved = 0
    i = 0
    while i < M:
        npoints = 0 # reset
        for j in range(N):
            if cids[j] == i:
                npoints += 1
        if npoints < minpoints:
            #print('cluster %d has only %d points' % (i, npoints))
            # remove cluster i
            # shift all entries at i and above in scouts array down by one
            for scouti in range(i, M-1):
                for k in range(ndims):
                    scouts[scouti, k] = scouts[scouti+1, k]
                still[scouti] = still[scouti+1] # ditto for still array
            # update cluster indices
            for clustii in range(N):
                if cids[clustii] == i:
                    cids[clustii] = -1 # overwrite all occurences of i with -1
                elif cids[clustii] > i:
                    cids[clustii] -= 1 # decr all clust indices above i
            M -= 1 # decr num of scouts, don't inc i, new value at i has just slid into view
            nremoved += 1
        else:
            i += 1
    print('%d clusters deleted for having less than %d points' % (nremoved, minpoints))

    if calcpointdensities:
        # calculate the local density for each point, using potentially just subsampled data
        # from this cluster. This does g weighted sum of distances, almost like just counting
        # the number of points in a volume, and then dividing by the volume. Except you don't
        # really need to divide by the volume of the gaussian, cuz that's just a constant for
        # a given sigma
        # TODO: normalize by volume anyway, since you might run climb() with different values
        # of sigma (like during a cluster split), and you want to keep all your density
        # values consistent and comparable. Calculate the volume once per call, and divide
        # all density values by it. Shouldn't be expensive
        print('Calculating density around each data point, based on sampled data')
        t0 = time.time()
        densities = np.zeros(N)
        for i in range(N):
            for j in range(nsamples): # iterate over sampled data, check if they're within rneigh
                samplei = sampleis[j]
                if cids[i] != cids[samplei] or samplei == i:
                    continue # don't include points from different clusters, or the point itself
                d2 = 0.0 # reset
                for k in range(ndims): # iterate over dims for each point
                    ds[k] = data[i, k] - data[samplei, k]
                    if fabs(ds[k]) > rneigh: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    d2 += ds[k] * ds[k] # add to sum of squares for this sample
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next j
                if d2 <= rneigh2: # include this point in the density calculation
                    d = sqrt(d2) # Euclidean distance
                    densities[i] += d * exp(-d2 / twosigma2)
        print('Point density calculations took %.3f sec' % (time.time()-t0))
    else: densities = np.zeros(0)
    if calcscoutdensities:
        print('Calculating density around each scout, based on sampled data')
        t0 = time.time()
        scoutdensities = np.zeros(M)
        for i in range(M):
            for j in range(nsamples): # iterate over sampled data, check if they're within rneigh
                samplei = sampleis[j]
                if i != cids[samplei]:
                    continue # don't include points from different clusters. TODO: maybe I should anyway?
                d2 = 0.0 # reset
                for k in range(ndims): # iterate over dims for each point
                    ds[k] = scouts[i, k] - data[samplei, k]
                    if fabs(ds[k]) > rneigh: # break out of k loop, continue to next j loop
                        continuej = 1
                        break # out of k loop
                    d2 += ds[k] * ds[k] # add to sum of squares for this sample
                if continuej == 1:
                    continuej = 0 # reset
                    continue # to next j
                if d2 <= rneigh2: # include this point in the density calculation
                    d = sqrt(d2) # Euclidean distance
                    scoutdensities[i] += d * exp(-d2 / twosigma2)
        print('Scout density calculations took %.3f sec' % (time.time()-t0))
    else:
        scoutdensities = np.zeros(0)


    moving = still[:M] < maxstill
    nmoving = moving.sum()
    print('\nniters: %d' % iteri)
    print('nscouts: %d' % M)
    print('sigma: %.2f, rneigh: %.2f, rmerge: %.2f, alpha: %.2f' % (sigma, rneigh, rmerge, alpha))
    print('nmoving: %d, minmove: %f' % (nmoving, minmove))
    print('moving scouts: %r' % np.where(moving)[0])
    print('still array:')
    print still[:M]
    return cids, scouts[:M], densities, scoutdensities, sampleis

