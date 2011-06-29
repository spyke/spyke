# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""Nick's gradient-ascent (mountain-climbing) clustering algorithm"""

cimport cython
import numpy as np
cimport numpy as np

import random, time
#from extlib import threadpool
## TODO: consider switching to built in ThreadPool in multiprocessing.pool:
##from multiprocessing.pool import ThreadPool
## Better yet, use C level threads via OpenMP by using prange in latest Cython!
from multiprocessing import cpu_count

cdef extern from "math.h":
    double sqrt(double x) nogil
    double fabs(double x) nogil
    double exp(double x) nogil
    double ceil(double x) nogil

cdef extern from "stdio.h":
    int printf(char *, ...) nogil
    cdef void *malloc(size_t) nogil # allocates without clearing to 0
    cdef void *calloc(size_t, size_t) nogil # allocates with clearing to 0
    cdef void free(void *) nogil

cdef extern from "string.h":
    cdef void *memset(void *, int, size_t) nogil # sets n bytes in memory to constant

# NOTE: stdout is buffered by default in linux. This means anything printed to screen from
# within C code won't show up until it gets a newline, or until you call fflush(stdout).
# Unbuffered output can be forced by running Python with the "-u" switch

cdef unsigned short MAXUINT16 = 2**16 - 1
cdef unsigned int MAXUINT32 = 2**32 - 1

def climb(np.ndarray[np.float32_t, ndim=2, mode='c'] data,
          double sigma=0.05, double alpha=2.0,
          double rmergex=1.0, double rneighx=4,
          double minmove=-1.0, int maxstill=100, int maxnnomerges=1000,
          int minpoints=10):
    """Implement Nick's gradient ascent (mountain climbing) clustering algorithm
    TODO:
        - test if datah and scoutspace can be allocated - if not (too many dimensions,
        data too sparse, bin size too small), do normal climb() using data and scout
        tables, or make the bins coarser
            - might not be possible to do sparse clustering on full space of all spikes,
            might only work when subclustering, which will be a denser space. Could tie
            in well with Nick's one channel at a time clustering
            - could also use a histogram cutoff: along each dimension discretized data,
            check how many points fall below npoints threshold. Set threshold
            to 10 say. Check from both ends of each dimension. If npoints < 10 at given
            level, then truncate the space at that level, and check the next level. Ie, instead
            of using absolute min and max of data to set dimensionality of datah, be
            flexible enough to throw away a tiny bit of data for the sake of a potentially
            much smaller (and allocatable) space
            - might need to look into sparse matrix implementations that will save memory,
            but still provide the speed advantage over a table with its O(N**2) problem
                - scipy.sparse looks interesting - nope, that's just sparse matrices. Apparently
                there's no such thing as sparse ndimensional arrays anywhere - such problems
                are typically mapped down to 2D sparse matrices according to Travis Oliphant:
                http://mail.scipy.org/pipermail/numpy-discussion/2003-January/014212.html
        
        - get rid of all 1D temporary numpy arrays. Use alloc() instead
            - in the same vein, try using cdef inline instead of just cdef to reduce
            a fn's call overhead - think it won't do anything for fn's with numpy array
            args
        
        - reverse annealing sigma: starting small, and gradually increase it over iters
            - increase it a bit every time you get an iteration with no mergers?
        - maybe some way of making sigma dynamic for each scout and for each iteration?
        - maybe annealing of alpha (decreasing it over time)? NVS sounds skeptical

        - classify obvious wide flat areas as noise points that shouldn't be clustered:
            - track the distance each point has travelled during the course of the algorithm.
            When done, plot the distribution of travel distances, and maybe you'll get something
            bimodal, and choose a cutoff travel distance past which any point that travelled
            further is considered a noise point
            - or maybe plot distribution of travel times
            - use some cutoff of local density to specify what's noise and what isn't? skeptical..

        - visualize algorithm in real time to see what exactly it's doing, and why some clusters
        are split while others are merged

        - instead of merging the higher indexed scout into the lower indexed one, you should
        really merge the one with the lower density estimate into the one with the higher
        density estimate - otherwise you potentially end up deleting the scout that's closer
        to the local max density, which probably sets you back several iterations
            - this would require calc'ing and storing the density for each cluster, and updating
            it every time it moves
                - is local density and local gradient calc sufficiently similar that this won't
                be expensive?
            - find whichever has the biggest density estimate - if it's not the lowest indexed
            scout (which will be the case 50% of the time), swap the entries in all the arrays
            (scouts, still, etc) except for the cids array, then proceed as usual. Then update
            the density for the newly merged cluster

        - try using simplex algorithm for scout position update step, though that might miss
        local maxima

        - rescale all data by 2*sigma so you can get rid of the div by twosigma2 operation?
            - only applies to Gaussian kernel, not Cauchy

        - try using the n nearest neighbours to calculate gradient, instead of a guassian with
        a sigma. This makes it scale free, but NVS says this often results in situations where
        the gradient is 0 for some reason

        - scale x not just by its std, but also according to some absolute multiple of space
        (say 1.0 is 50 um), such that recordings with wider or narrower x locations (2 or 3
        column probes) will cluster roughly as well with a constant sigma value (like 0.25,
        which really means you can expect up to 4 clusters along the x axis)

    DONE:
        - turn off checks for ZeroDivisionError, though I doubt that slows things down much
        - keep track of max movement on each iter, use consistently low max movement as
          automatic exit criteria
            - alternative: keep track of how long it's been since the last scout merger, and
            exit based on that
        - add freezing of points, for speed?
            - when a scout point has moved less than some distance per iteration for all of
            the last n iterations, freeze it. Then, in the position update loop, check for
            frozen scout points
        - add subsampling to reduce initial number of scout points
        - NVS - to weed out potential noise spikes, for each cluster, find the local density
        of the scout point at the max, then reject all other data points in that cluster whose
        ocal density falls below, say, 1% of the max. Apply it as a mask, so you can tweak that
        1% value as you wish, without having to run the whole algorithm all over again
        - delete scouts that have fewer than n points (at any point during iteration?)
        - multithread scout update and assignment of unclustered points step
    """
    cdef Py_ssize_t i, j, k, scouti, clustii
    cdef bint incstill, merged=False, continuej=False
    cdef int iteri=0, nnomerges=0
    cdef int N = data.shape[0] # total num data points
    cdef int ndims = data.shape[1] # num cols in data
    # dimension sizes:
    cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] dims = np.zeros(ndims, dtype=np.uint32)
    # n-dimensional index working array:
    cdef unsigned int *ndi = <unsigned int *> malloc(ndims*sizeof(unsigned int))
    cdef int M = N # current num scout points (clusters), each data point starts as its own scout
    cdef int npoints, npointsremoved, nclustsremoved
    cdef long long li, proddims
    cdef double binx = 0.25 # some fraction of sigma to bin data by
    cdef double binsize = binx * sigma
    sigma /= binsize # scale sigma the same way data will be scaled, ie sigma = 1/binx
    cdef double sigma2 = sigma * sigma
    #cdef double twosigma2 = 2 * sigma2
    cdef double rmerge = rmergex * sigma # radius within which scout points are merged
    cdef double rmerge2 = rmerge * rmerge
    cdef double rneigh = rneighx * sigma # radius around scout to include data for gradient calc
    cdef double rneigh2 = rneigh * rneigh
    cdef double d, d2, minmove2

    # store scout positions in a float table:
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] scouts
    # store indices into rows of scouts float table:
    cdef unsigned int *sr = <unsigned int *> malloc(M*sizeof(unsigned int))
    uirange(sr, M) # init sr to consecutive int values
    # for each scout, num consecutive iters without significant movement:
    ## TODO: should check that (maxstill < 256).all(), or use uint16 instead:
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] still = np.zeros(N, dtype=np.uint8)
    # cluster indices into data:
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] cids = np.zeros(N, dtype=np.int32)
    
    
    # need to convert data table to something suitable for truncing to easily get
    # ints. First have to add some offset to make everything +ve. Then, divide by your fraction
    # of sigma that you want to discretize by. Then, when returning scout positions, need to do 
    # the inverse
    for k in range(ndims):
        data[:, k] -= data[:, k].min() # offset data in each dimension to be +ve starting from 0
    data /= binsize # scale data, same for all dims since sigma apples to all dims
    assert data.max() < 2**32
    assert data.min() == 0
    ## TODO: use scaled versions of sigma, and everything else that depends on sigma too,
    ## when calculating ranges in datah and scoutpsace
    scouts = data.copy()

    # get dimensions of sparse matrices
    for k in range(ndims):
        dims[k] = <unsigned int> data[:, k].max() + 1 # dim size = max index + 1
    print 'dims = ', dims
    proddims = prod(dims)

    # datah: ndim static histogram of point positions in data, bins of size binsize
    # use uint16, since not likely to have more than 65k points in a single bin.
    # Hell, maybe uint8 would work too
    print('creating %d MB datah array' % (proddims * 2 / 1e6))
    cdef unsigned short *datah = <unsigned short *> calloc(proddims, sizeof(unsigned short))
    if not datah:
        raise MemoryError("can't allocate datah")
    # build up histogram in datah
    for i in range(N):
        # trunc float data point position to int nd index:
        for k in range(ndims):
            ndi[k] = <unsigned int> data[i, k]
        li = ndi2li(ndi, dims)
        #print('li = %d, datah[li] = %d' % (li, datah[li]))
        datah[li] += 1
        if datah[li] == MAXUINT16:
            raise RuntimeError("uint16 isn't enough for datah!")
    print('done initing datah')
    print('max(datah) = %d' % usmax(datah, proddims))

    # scoutspace: ndim dynamic histogram of scout positions in scouts table
    # use dynamic scoutspace sparse matrix to approximate each scout's position and
    # calculate gradient according to same sized datah, but then update the scout's
    # actual position in separate scouts table in float, per usual. Otherwise, if you
    # stored scout positions quantized, you could easily get stuck in a bin and never
    # get out, because you could never accumulate less than bin sized changes in position
    print('creating %d MB scoutspace array' % (proddims * 4 / 1e6))
    #cdef np.ndarray[np.uint32_t, ndim=ndims, mode='c'] scoutspace = np.zeros(dims, dtype=np.uint32)
    cdef unsigned int *scoutspace = <unsigned int *> malloc(proddims*sizeof(unsigned int))
    if not scoutspace:
        raise MemoryError("can't allocate scoutspace")
    print('initing scoutspace, M=%d' % M)
    M = update_scoutspace(M, proddims, dims, scoutspace, sr, scouts, still, cids)
    print('done initing scoutspace, M=%d' % M)
    return
    # for merging scouts, clear scoutspace, and start writing their indices to it.
    # While writing, if you find the position in the matrix is already occupied,
    # then obviously you need to merge the current scout into the one that's already
    # there. Once you're done filling the matrix, for every non-zero entry (which you can
    # quickly find by truncing scout position in scouts array to get its index)
    # take slice corresponding to rmerge, then maybe do sum of squared discrete distances,
    # and merge if < rmerge

    #Mthresh = 3000000 / N / ndims
    #print("Mthresh = %d" % Mthresh)
    cids = np.arange(M, dtype=np.int32)

    if minmove == -1.0:
        # TODO: should minmove also depend on sqrt(ndims)? it already does via sigma
        minmove = 0.000001 * sigma * alpha # in any direction in ndims space
    minmove2 = minmove * minmove

    #ncpus = cpu_count()
    #cdef long *lohi = <long *> malloc((ncpus+1)*sizeof(long))
    #pool = threadpool.ThreadPool(ncpus)

    while True:

        if nnomerges == maxnnomerges:
            break

        # merge pairs of scout points sufficiently close to each other
        i = 0
        while i < M:
            j = i+1
            while j < M:
                if still[i] == maxstill and still[j] == maxstill: # both scouts are frozen
                    j += 1
                    continue
                # for each pair of scouts, check if any pair is within rmerge of each other
                d2 = 0.0 # reset
                for k in range(ndims):
                    d = fabs(scouts[sr[i], k] - scouts[sr[j], k])
                    if d > rmerge: # break out of k loop, continue to next j
                        continuej = True
                        break # out of k loop
                    d2 += d * d
                if continuej:
                    continuej = False # reset
                    j += 1
                    continue # to next j loop
                if d2 <= rmerge2:
                    # merge the scouts: keep scout i, ditch scout j
                    M = merge_scouts(i, j, M, sr, still, cids)
                    # don't inc j, new value at j has just slid into view
                    merged = True
                else:
                    j += 1
            i += 1
        if merged: # at least one merger happened on this iter
            printf('M')
            nnomerges = 0 # reset
            merged = False # reset
        else: # no mergers happened on this iter
            nnomerges += 1 # inc

        # move scouts up their local density gradient
        #if M < Mthresh: # use a single thread
        move_scouts(0, M, sr, scouts, data, still,
                    N, ndims, sigma2, alpha,
                    rneigh, rneigh2, minmove2, maxstill)
        '''
        else: # use multiple threads
            span(lohi, 0, M, ncpus) # modify lohi in place
            for i in range(ncpus):
                args = (lohi[i], lohi[i+1], sr, scouts, data, still,
                        N, ndims, sigma2, alpha,
                        rneigh, rneigh2, minmove2, maxstill)
                req = threadpool.WorkRequest(move_scouts, args)
                pool.putRequest(req)
            pool.wait()
        '''
        printf('.')

        iteri += 1

    printf('\n')

    #pool.terminate()

    # remove clusters with less than minpoints
    npointsremoved = 0
    nclustsremoved = 0
    i = 0
    while i < M:
        npoints = 0 # reset
        for j in range(N):
            if cids[j] == i:
                npoints += 1
        if npoints < minpoints:
            #print('cluster %d has only %d points' % (i, npoints))
            # remove cluster i by merging it into "cluster" -1
            M = merge_scouts(-1, i, M, sr, still, cids)
            # don't inc i, new value at i has just slid into view
            npointsremoved += npoints
            nclustsremoved += 1
        else:
            i += 1
    print('%d points (%.1f%%) and %d clusters deleted for having less than %d points each' %
         (npointsremoved, npointsremoved/float(N)*100, nclustsremoved, minpoints))

    moving = still[:M] < maxstill
    nmoving = moving.sum()
    print('\nniters: %d' % iteri)
    print('nscouts: %d' % M)
    print('sigma: %.3f, rneigh: %.3f, rmerge: %.3f, alpha: %.3f' % (sigma, rneigh, rmerge, alpha))
    print('nmoving: %d, minmove: %f' % (nmoving, minmove))
    print('moving scouts: %r' % np.where(moving)[0])
    print('still array:')
    print still[:M]
    ## TODO: need to rebuild scouts so it's contiguous on return:
    ## UNTESTED:
    for i in range(M):
        for k in range(ndims):
            scouts[i, k] = scouts[sr[i], k]
    # now sr is no longer valid, if still needed it, would have to reinit with:
    #uirange(sr, M)
    free(sr)
    free(datah)
    free(scoutspace)
    #free(lohi)
    return cids, scouts[:M]


cdef void move_scouts(int lo, int hi, unsigned int *sr,
                      np.ndarray[np.float32_t, ndim=2, mode='c'] scouts,
                      np.ndarray[np.float32_t, ndim=2, mode='c'] data,
                      np.ndarray[np.uint8_t, ndim=1, mode='c'] still,
                      int N, int ndims, double sigma2, double alpha,
                      double rneigh, double rneigh2, double minmove2, int maxstill) nogil:
    """Move scouts up their local density gradient"""
    # use much faster C allocation for temporary 1D arrays instead of numpy:
    cdef double *ds = <double *> malloc(ndims*sizeof(double))
    cdef double *d2s = <double *> malloc(ndims*sizeof(double))
    cdef double *kernel = <double *> malloc(ndims*sizeof(double))
    cdef double *v = <double *> malloc(ndims*sizeof(double))
    cdef Py_ssize_t i, j, k
    #cdef int nneighs
    cdef bint continuej=False
    cdef double d2, kern, move, move2#, maxmove = 0.0
    for i in range(lo, hi): # iterate over lo to hi scout points
        # skip frozen scout points
        if still[i] == maxstill:
            continue
        # measure gradient
        #nneighs = 0 # reset
        #for k in range(ndims):
        #    kernel[k] = 0.0 # reset
        #    v[k] = 0.0 # reset
        # slightly faster, though not guaranteed to be valid thing to do for non-int array:
        memset(kernel, 0, ndims*sizeof(double)) # reset
        memset(v, 0, ndims*sizeof(double)) # reset
        for j in range(N): # iterate over data, check if they're within rneigh
            d2 = 0.0 # reset
            for k in range(ndims): # iterate over dims for each point
                ds[k] = data[j, k] - scouts[sr[i], k]
                if fabs(ds[k]) > rneigh: # break out of k loop, continue to next j loop
                    continuej = True
                    break # out of k loop
                d2s[k] = ds[k] * ds[k] # used twice, so calc it only once
                d2 += d2s[k]
            if continuej:
                continuej = False # reset
                continue # to next j
            if d2 <= rneigh2: # do the calculation
                for k in range(ndims):
                    # v is ndim vector of sum of kernel-weighted distances between
                    # current scout point and all data within rneigh
                    #kern = exp(-d2s[k] / twosigma2) # Gaussian kernel
                    kern = sigma2 / (d2s[k] + sigma2) # Cauchy kernel, faster
                    #printf('%.3f ', kern)
                    kernel[k] += kern
                    v[k] += ds[k] * kern
                #nneighs += 1
        # update scout position in direction of v, normalize by kernel
        # nneighs (and kernel?) will never be 0, because each scout point starts as a data point
        move2 = 0.0 # reset
        for k in range(ndims):
            move = alpha / kernel[k] * v[k] # normalize by kernel, not just nneighs
            scouts[sr[i], k] += move
            move2 += move * move
            #if fabs(move) > fabs(maxmove):
            #    maxmove = move
        if move2 < minmove2:
            still[i] += 1 # count scout as still during this iter
        else:
            still[i] = 0 # reset stillness counter for this scout
        # wanted to see if points move faster when normalized by kernel vs nneighs:
        #printf('%f ', maxmove)
        free(ds)
        free(d2s)
        free(kernel)
        free(v)

'''
cdef void span(long *lohi, int start, int end, int N) nogil:
    """Fill len(N) lohi array with fairly equally spaced int
    values, from start to end"""
    cdef Py_ssize_t i
    cdef int step
    step = <int> ceil(<double> (end - start) / N) # round up
    for i in range(N):
        lohi[i] = start + step*i
    lohi[N] = end
'''
cdef long long prod(np.ndarray[np.uint32_t, ndim=1, mode='c'] a) nogil:
    """Return product of entries in uint32 array a"""
    cdef long long result
    cdef Py_ssize_t i, n
    n = a.shape[0]
    result = 1
    for i in range(n):
        result *= a[i] # should I upcast to long long here?
    return result

cdef void uifill(unsigned int *a, unsigned int val, long long n) nogil:
    """Fill unsigned int array with n values"""
    cdef unsigned long long i
    for i in range(n):
        a[i] = val

cdef void uirange(unsigned int *a, n):
    """Fill unsigned int array with n increasing values"""
    a[0] = 0
    for i in range(1, n):
        a[i] = a[i-1] + 1 

cdef unsigned short usmax(unsigned short *a, unsigned long long n) nogil:
    """Return maximum value in array"""
    cdef unsigned short result=0
    cdef unsigned long long i
    for i in range(n):
        if a[i] > result:
            result = a[i]
    return result

cdef long long ndi2li(unsigned int *ndi,
                      np.ndarray[np.uint32_t, ndim=1, mode='c'] dims) nogil:
    """Convert n dimensional index in array ndi to linear index. ndi
    and dims should be the same length, and each entry in ndi should be
    less than its corresponding dimension size in dims"""
    ## NOTE: np.unravel_index() and np.ravel_multi_index() are useful!
    cdef long long li, pr=1
    cdef Py_ssize_t k, ndims=dims.shape[0]
    li = ndi[ndims-1] # init with index of deepest dimension
    # iterate from ndims-1 to 0, from 2nd deepest to shallowest dimension
    # either syntax works, and both seem to be C optimized:
    #for k from ndims-1 >= k > 0:
    for k in range(ndims-1, 0, -1):
        pr *= dims[k] # running product of dimensions
        li += ndi[k-1] * pr # accum sum of products of next ndi and all deeper dimensions
    return li

cdef int merge_scouts(Py_ssize_t scouti, Py_ssize_t scoutj, int M,
                      unsigned int *sr,
                      np.ndarray[np.uint8_t, ndim=1, mode='c'] still,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] cids) nogil:
    """Merge scoutj into scouti, where scouti < scoutj"""
    ## TODO: maybe make an array of pointers (scoutps**) or just indices (scoutis*)
    ## that index into scouts table, and only modify those when merging. That way
    ## you don't have to do the nested k loop iteration of dims on all scouts > i
    ## on every merge. Should speed this function up quite a bit. Tradeoff is that
    ## then you need a level of indirection when indexing into scouts table: need
    ## to first index into scoutis*, and then use that result to index into scouts
    ## table
    if not scouti < scoutj: # can only merge higher id into lower id!
        printf('ERROR: scouti >= scoutj: %d >= %d', scouti, scoutj)
    cdef Py_ssize_t i, cii
    cdef int N = cids.shape[0]
    # shift all entries at j and above in scouts and still arrays down by one
    for i in range(scoutj, M-1):
        sr[i] = sr[i+1]
        still[i] = still[i+1]
    # update cluster indices
    for cii in range(N):
        if cids[cii] == scoutj:
            cids[cii] = scouti # replace all scoutj entries with scouti
        elif cids[cii] > scoutj:
            cids[cii] -= 1 # decr all clust indices above scout j
    M -= 1 # decr num scouts
    #printf(' %d<-%d ', scouti, scoutj)
    return M

cdef int update_scoutspace(int M, long long proddims,
                           np.ndarray[np.uint32_t, ndim=1, mode='c'] dims,
                           unsigned int *scoutspace,
                           unsigned int *sr,
                           np.ndarray[np.float32_t, ndim=2, mode='c'] scouts,
                           np.ndarray[np.uint8_t, ndim=1, mode='c'] still,
                           np.ndarray[np.int32_t, ndim=1, mode='c'] cids) nogil:
    """Refill scoutspace based on current scout positions in scouts table"""
    cdef Py_ssize_t i=0, k, ndims=dims.shape[0]
    # reset, use MAXUINT32 to indicate empty slot:
    uifill(scoutspace, MAXUINT32, proddims)
    # nd index working array:
    cdef unsigned int *ndi = <unsigned int *> malloc(ndims*sizeof(unsigned int))
    #print('scouts.max() = %f, scouts.min() = %f' % (scouts.max(), scouts.min()))
    #printf('trunc max() = %d, trunc min() = %d\n', <unsigned int> scouts.max(), <unsigned int> scouts.min())
    while i < M: # iterate over all scouts
        # trunc float scout position to int nd index:
        for k in range(ndims):
            ndi[k] = <unsigned int> scouts[sr[i], k]
        li = ndi2li(ndi, dims)
        #printf('li = %d\n', li)
        if scoutspace[li] == MAXUINT32: # bin is unoccupied
            scoutspace[li] = i # occupy the bin with scout i
            i += 1
        else: # there's already a scout there, merge into it
            M = merge_scouts(scoutspace[li], i, M, sr, still, cids)
            # don't inc i, new value at i has just slid into view
    free(ndi)
    return M
