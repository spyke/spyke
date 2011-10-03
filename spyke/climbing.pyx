# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""Nick's gradient-ascent (mountain-climbing) clustering algorithm"""

#cimport cython # not sure why this was needed before
from cython.parallel import prange#, parallel
import numpy as np
cimport numpy as np
#import time

cdef extern from "math.h":
    double fabs(double x) nogil
    double exp(double x) nogil
    double sqrt(double x) nogil
    #double ceil(double x) nogil

cdef extern from "stdio.h":
    int printf(char *, ...) nogil
    cdef void *malloc(size_t) nogil # allocates without clearing to 0
    cdef void *calloc(size_t, size_t) nogil # allocates with clearing to 0
    cdef void free(void *) nogil
'''
cdef extern from "string.h":
    cdef void *memset(void *, int, size_t) nogil # sets n bytes in memory to constant
'''
# NOTE: stdout is buffered by default in linux. This means anything printed to screen from
# within C code won't show up until it gets a newline, or until you call fflush(stdout).
# Unbuffered output can be forced by running Python with the "-u" switch

cdef short MAXUINT16 = 2**16 - 1

def climb(np.ndarray[np.float32_t, ndim=2, mode='c'] data,
          double sigma=0.25, double alpha=2.0,
          double rmergex=0.25, double rneighx=4,
          double minmovex=0.00001, int maxnnomerges=1000,
          int minpoints=5):
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
        
        - try using cdef inline instead of just cdef to reduce
        a fn's call overhead - think it won't do anything for fn's with numpy array
        args
            - doesn't seem to make any difference, even for purely C arg f'ns. Perhaps
            GCC automatically finds f'ns that are macro candidates and uses them as such?
        
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
        density estimate - otherwise shortyou potentially end up deleting the scout that's closer
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

        - rescale all data by sqrt(2)*sigma so you can get rid of the div by twosigma2 operation?
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
        - get rid of all 1D temporary numpy arrays. Use alloc() instead
    """
    cdef Py_ssize_t i, j, k, scouti
    cdef bint merged=False
    cdef int iteri=0, nnomerges=0
    cdef int N = data.shape[0] # total num rows (points) in data table
    cdef int ndims = data.shape[1] # num cols in data table
    cdef int *dims = <int *> malloc(ndims*sizeof(int)) # dimension sizes
    cdef int *ndi = <int *> malloc(ndims*sizeof(int)) # n-dimensional index working array
    cdef int *cids = <int *> malloc(N*sizeof(int)) # cluster indices into data
    if not cids: raise MemoryError("can't allocate cids")
    irange(cids, N) # init cids to consecutive int values
    cdef int M = N # current num scout points (clusters), each data point starts as its own scout
    cdef int npoints, npointsremoved, nclustsremoved
    #cdef long long li, proddims
    '''
    cdef double binx = 0.25 # some fraction of sigma to bin data by
    cdef double binsize = binx * sigma
    sigma /= binsize # scale sigma the same way data will be scaled, ie sigma = 1/binx
    '''
    # normalize all data related variables by norm to avoid having to
    # do so in move_scout() loop. Note that all of these are also scaled
    # by sqrt(ndims) via sigma scaling in caller:
    cdef int lenexps = 1000000
    cdef double norm0 = sqrt(2) * sigma
    cdef double rneigh0 = rneighx * sigma / norm0
    cdef double rneigh02 = rneigh0 * rneigh0
    cdef double norm = norm0 * rneigh0 / sqrt(lenexps)
    # radius around scout to include data for gradient calc:
    cdef double rneigh = sqrt(lenexps)
    cdef double rneigh2 = lenexps # rneigh * rneigh
    #printf('norm: %f, rneigh: %.1f, rneigh2: %.1f\n', norm, rneigh, rneigh2)
    # radius within which scout points are merged:
    cdef double rmerge = rmergex * sigma / norm
    cdef double rmerge2 = rmerge * rmerge
    # min motion in any direction in ndims space req'd for scout to be considered moving:
    cdef double minmove = minmovex * sigma * alpha / norm
    cdef double minmove2 = minmove * minmove

    # pre-calc exp function:
    #t0 = time.time()
    cdef double *exps = <double *> malloc(lenexps*sizeof(double)) # pre-calced exp function
    if not exps: raise MemoryError("can't allocate exps")
    for i in range(lenexps):
        exps[i] = exp(-<double>i / lenexps * rneigh02) # watch out for int div
    #print('exps malloc took %.3f sec' % (time.time()-t0))
    
    # store point positions in a 2D C float array, since handling numpy data array directly
    # causes segfaults in prange() loops:
    cdef float **points = <float **> malloc(N*sizeof(float *))
    if not points: raise MemoryError("can't allocate points")
    for i in range(N):
        points[i] = <float *> malloc(ndims*sizeof(float))
    # store scout positions in a 2D C float array:
    cdef float **scouts = <float **> malloc(M*sizeof(float *))
    if not scouts: raise MemoryError("can't allocate scouts")
    for i in range(M):
        scouts[i] = <float *> malloc(ndims*sizeof(float))
    # store indices into rows of scouts float table:
    cdef int *sr = <int *> malloc(M*sizeof(int))
    if not sr: raise MemoryError("can't allocate sr")
    irange(sr, M) # init sr to consecutive int values
    # for each scout, num consecutive iters without significant movement:
    cdef bint *still = <bint *> calloc(M, sizeof(bint))
    if not still: raise MemoryError("can't allocate still")
    '''
    # need to convert data table to something suitable for truncing to easily get
    # ints. First have to add some offset to make everything +ve. Then, divide by your fraction
    # of sigma that you want to discretize by. Then, when returning scout positions, need to do 
    # the inverse
    for k in range(ndims):
        data[:, k] -= data[:, k].min() # offset data in each dimension to be +ve starting from 0
    data /= binsize # scale data, same for all dims since sigma apples to all dims
    assert data.max() < 2**32
    assert data.min() == 0
    '''
    # init points and scouts at data point positions, normalize by norm
    # to reduce math in move_scout() nested loops, and allow use of exps lookup
    for i in range(N): # M == N
        for k in range(ndims):
            points[i][k] = data[i, k] / norm
            scouts[i][k] = data[i, k] / norm
    '''
    # get dimensions of sparse matrices
    printf('dims = ')
    for k in range(ndims):
        dims[k] = <int> data[:, k].max() + 1 # dim size = max index + 1
        printf('%d ', dims[k])
    printf('\n')
    proddims = prod(dims, ndims)
    '''
    # pointh: ndim static histogram of point positions in points, bins of size binsize
    # use uint16, since not likely to have more than 65k points in a single bin.
    # Hell, maybe uint8 would work too
    '''
    print('creating %d MB pointh array' % (proddims * 2 / 1e6))
    cdef unsigned short *pointh = <unsigned short *> calloc(proddims, sizeof(unsigned short))
    if not pointh: raise MemoryError("can't allocate pointh")
    # build up histogram in pointh
    for i in range(N):
        # trunc float point position to int nd index:
        for k in range(ndims):
            ndi[k] = <int> points[i][k]
        li = ndi2li(ndi, dims, ndims)
        #print('li = %d, pointh[li] = %d' % (li, pointh[li]))
        pointh[li] += 1
        if pointh[li] == MAXUINT16:
            raise RuntimeError("uint16 isn't enough for pointh!")
    print('done initing pointh')
    print('max(pointh) = %d' % usmax(pointh, proddims))

    # scoutspace: ndim dynamic histogram of scout positions in scouts table
    # use dynamic scoutspace sparse matrix to approximate each scout's position and
    # calculate gradient according to same sized pointh, but then update the scout's
    # actual position in separate scouts table in float, per usual. Otherwise, if you
    # stored scout positions quantized, you could easily get stuck in a bin and never
    # get out, because you could never accumulate less than bin sized changes in position
    print('creating %d MB scoutspace array' % (proddims * 4 / 1e6))
    cdef int *scoutspace = <int *> malloc(proddims*sizeof(int))
    if not scoutspace: raise MemoryError("can't allocate scoutspace")
    print('initing scoutspace, M=%d' % M)
    M = update_scoutspace(M, proddims, ndims, dims, scoutspace, sr, scouts, still, N, cids)
    print('done initing scoutspace, M=%d' % M)
    '''
    # for merging scouts, clear scoutspace, and start writing their indices to it.
    # While writing, if you find the position in the matrix is already occupied,
    # then obviously you need to merge the current scout into the one that's already
    # there. Once you're done filling the matrix, for every non-zero entry (which you can
    # quickly find by truncing scout position in scouts array to get its index)
    # take slice corresponding to rmerge, then maybe do sum of squared discrete distances,
    # and merge if < rmerge

    while True:

        if nnomerges == maxnnomerges:
            break

        # merge scouts within rmerge of each other
        M = merge_scouts(M, sr, scouts, rmerge, rmerge2, still,
                         N, cids, ndims, &merged)
        if merged: # at least one merger happened on this iter
            printf('M=%d', M)
            nnomerges = 0 # reset
            merged = False # reset
        else: # no mergers happened on this iter
            nnomerges += 1 # inc

        # move scouts up their local density gradient
        for scouti in prange(M, nogil=True, schedule='dynamic'):
            if still[scouti]: continue # only move scout points that aren't frozen:
            move_scout(scouti, sr, scouts, points, exps, still,
                       N, ndims, alpha, rneigh, rneigh2, minmove2)

        printf('.')

        iteri += 1

    printf('\n')

    # remove clusters with less than minpoints
    npointsremoved = 0
    nclustsremoved = 0
    i = 0
    while i < M:
        npoints = 0 # reset
        # tally up npoints in cluster i
        for j in range(N): # TODO: this could maybe be prange, inc'ing concurrently won't cause races?
            if cids[j] == i:
                npoints += 1
        if npoints < minpoints:
            #printf('cluster %d has only %d points', i, npoints)
            # remove cluster i by merging it into "cluster" -1
            M = merge(-1, i, M, sr, still, N, cids)
            # don't inc i, new value at i has just slid into view
            npointsremoved += npoints
            nclustsremoved += 1
        else:
            i += 1
    printf('%d points (%.1f%%) and %d clusters deleted for having less than %d points each\n',
           npointsremoved, npointsremoved/(<double>N)*100, nclustsremoved, minpoints)

    # for display, restore sigma dependent params to be unnormalized by norm:
    rmerge *= norm
    rneigh *= norm
    minmove *= norm

    cdef int nmoving=0
    for i in range(M):
        if not still[i]:
            nmoving += 1
    printf('nniters: %d\n',iteri)
    printf('nclusters: %d\n', M)
    printf('sigma: %.3f, rneigh: %.3f, rmerge: %.3f, alpha: %.3f\n', sigma, rneigh, rmerge, alpha)
    printf('nmoving: %d, minmove: %f\n', nmoving, minmove)
    printf('still array:\n[')
    for i in range(M):
        printf('%d, ', still[i])
    printf(']\n')

    # build returnable numpy ndarray for cids
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] np_cids = np.empty(N, dtype=np.int32)
    for i in range(N):
        np_cids[i] = cids[i]

    # generate contiguous numpy scouts array for return, scale up by norm again
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] np_scouts = np.empty((M, ndims), dtype=np.float32)
    for i in range(M):
        for k in range(ndims):
            np_scouts[i, k] = scouts[sr[i]][k] * norm # undo previous normalization

    free(dims)
    free(ndi)
    free(cids)
    free(exps)
    free(points)
    free(scouts)
    free(sr)
    free(still)
    #free(pointh)
    #free(scoutspace)
    return np_cids, np_scouts


cdef int merge_scouts(int M, int *sr, float **scouts, double rmerge, double rmerge2,
                      bint *still, int N, int *cids, int ndims, int *merged):
    """Merge pairs of scout points sufficiently close to each other"""
    cdef Py_ssize_t i=0, j, k
    cdef double d, d2
    cdef bint continuej=False
    while i < M:
        j = i+1
        while j < M:
            if still[i] and still[j]: # both scouts are frozen
                j += 1
                continue
            # for each pair of scouts, check if any pair is within rmerge of each other
            d2 = 0.0 # reset
            for k in range(ndims):
                d = fabs(scouts[sr[i]][k] - scouts[sr[j]][k])
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
                M = merge(i, j, M, sr, still, N, cids)
                # don't inc j, new value at j has just slid into view
                merged[0] = True
            else:
                j += 1
        i += 1
    return M


cdef void move_scout(int i, int *sr, float **scouts, float **points,
                     double *exps, bint *still,
                     int N, int ndims, double alpha,
                     double rneigh, double rneigh2, double minmove2) nogil:
    """Move a scout up its local density gradient"""
    cdef Py_ssize_t j, k
    #cdef int nneighs
    cdef bint continuej=False
    cdef double d2, kern, move, move2
    cdef double *ds = <double *> malloc(ndims*sizeof(double))
    cdef double *d2s = <double *> malloc(ndims*sizeof(double))
    cdef double *kernel = <double *> malloc(ndims*sizeof(double))
    cdef double *v = <double *> malloc(ndims*sizeof(double))

    # reset some local vars:
    #nneighs = 0
    dfill(kernel, 0, ndims)
    dfill(v, 0, ndims)
    # measure gradient v:
    for j in range(N): # iterate over points, check if any are within rneigh
        d2 = 0.0 # reset
        for k in range(ndims): # iterate over dims for each point
            ds[k] = points[j][k] - scouts[sr[i]][k]
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
                # current scout and all points within rneigh
                #kern = exp(-d2s[k] / (2 * sigma2)) # Gaussian kernel
                kern = exps[<int>(d2s[k])] # data rescaled for Gaussian lookup table
                #kern = sigma2 / (d2s[k] + sigma2) # Cauchy kernel
                kernel[k] += kern
                v[k] += ds[k] * kern # this is why you can't store fabs of ds[k]!
    # update scout position in direction of v, normalize by kernel
    # nneighs (and kernel?) will never be 0, because each scout starts as a point
    move2 = 0.0 # reset
    for k in range(ndims):
        move = alpha * v[k] / kernel[k] # normalize by kernel, not just nneighs
        scouts[sr[i]][k] += move
        move2 += move * move
    if move2 < minmove2:
        still[i] = True # freeze scout
            
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
cdef long long prod(int *a, int n) nogil:
    """Return product of entries in int array a"""
    cdef long long result
    cdef Py_ssize_t i
    result = 1
    for i in range(n):
        result *= a[i] # should I upcast to long long here?
    return result

cdef void ifill(int *a, int val, long long n) nogil:
    """Fill int array with n values"""
    cdef long long i
    for i in range(n):
        a[i] = val

cdef void dfill(double *a, double val, long long n) nogil:
    """Fill double array with n values"""
    cdef long long i
    for i in range(n):
        a[i] = val

cdef void irange(int *a, int n) nogil:
    """Fill int array with n increasing values"""
    cdef Py_ssize_t i
    a[0] = 0
    for i in range(1, n):
        a[i] = a[i-1] + 1 

cdef unsigned short usmax(unsigned short *a, long long n) nogil:
    """Return maximum value in array"""
    cdef unsigned short result=0
    cdef long long i
    for i in range(n):
        if a[i] > result:
            result = a[i]
    return result

cdef long long ndi2li(int *ndi, int *dims, int ndims) nogil:
    """Convert n dimensional index in array ndi to linear index. ndi
    and dims should be of length ndims, and each entry in ndi should be
    less than its corresponding dimension size in dims"""
    ## NOTE: np.unravel_index() and np.ravel_multi_index() are useful!
    cdef long long li, pr=1
    cdef Py_ssize_t k
    li = ndi[ndims-1] # init with index of deepest dimension
    # iterate from ndims-1 to 0, from 2nd deepest to shallowest dimension
    # either syntax works, and both seem to be C optimized:
    #for k from ndims-1 >= k > 0:
    for k in range(ndims-1, 0, -1):
        pr *= dims[k] # running product of dimensions
        li += ndi[k-1] * pr # accum sum of products of next ndi and all deeper dimensions
    return li

cdef int merge(Py_ssize_t scouti, Py_ssize_t scoutj, int M, int *sr,
               bint *still, int N, int *cids) nogil:
    """Merge scoutj into scouti, where scouti < scoutj"""
    if not scouti < scoutj: # can only merge higher id into lower id!
        printf('ERROR: scouti >= scoutj: %d >= %d', scouti, scoutj)
    cdef Py_ssize_t i, cii
    # shift all entries at j and above in sr and still arrays down by one,
    # needs to be done in succession, can't use prange
    for i in range(scoutj, M-1):
        sr[i] = sr[i+1]
        still[i] = still[i+1]
    # update cluster indices, doesn't need to be done in succession, can use prange,
    # but runs slower than a single thread - operations are too simple?
    #for cii in prange(N, nogil=True, schedule='static'):
    for cii in range(N):
        if cids[cii] == scoutj:
            cids[cii] = scouti # replace all scoutj entries with scouti
        elif cids[cii] > scoutj:
            cids[cii] -= 1 # decr all clust indices above scout j
    M -= 1 # decr num scouts
    #printf(' %d<-%d ', scouti, scoutj)
    return M
'''
cdef int update_scoutspace(int M, long long proddims, int ndims, int *dims,
                           int *scoutspace, int *sr, float **scouts,
                           bint *still, int N, int *cids) nogil:
    """Refill scoutspace based on current scout positions in scouts table"""
    cdef Py_ssize_t i=0, k
    # reset, use -1 to indicate empty slot:
    ifill(scoutspace, -1, proddims)
    # nd index working array:
    cdef int *ndi = <int *> malloc(ndims*sizeof(int))
    while i < M: # iterate over all scouts
        # trunc float scout position to int nd index:
        for k in range(ndims):
            ndi[k] = <int> scouts[sr[i]][k]
        li = ndi2li(ndi, dims, ndims)
        #printf('li = %d\n', li)
        if scoutspace[li] == -1: # bin is unoccupied
            scoutspace[li] = i # occupy the bin with scout i
            i += 1
        else: # there's already a scout there, merge into it
            M = merge(scoutspace[li], i, M, sr, still, N, cids)
            # don't inc i, new value at i has just slid into view
    free(ndi)
    return M
'''
