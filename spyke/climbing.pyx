# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""Nick Swindale's gradient-ascent (mountain-climbing) clustering algorithm"""

#cimport cython # this was only needed for decorators like @cython.boundscheck(False)
from cython.parallel import prange#, parallel
import numpy as np
cimport numpy as np
import time

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

cdef extern from "string.h":
    cdef void *memset(void *, int, size_t) nogil # sets n bytes in memory to constant
    cdef void *memcpy(void *, void *, size_t) nogil # copy to *dest from *src n bytes

# NOTE: stdout is buffered by default in linux. This means anything printed to screen from
# within C code won't show up until it gets a newline, or until you call fflush(stdout).
# Unbuffered output can be forced by running Python with the "-u" switch

#DEF MAXUINT16 = 2**16 - 1
#DEF MAXINT32 = 2**31 - 1

def climb(np.ndarray[np.float32_t, ndim=2, mode='c'] data,
          double sigma=0.25, double rmergex=0.25, double rneighx=4,
          double alpha=2.0, int maxgrad=1000,
          double minmovex=0.00001, int maxnnomerges=1000, int minpoints=5):
    """Implement Nick's gradient ascent (mountain climbing) clustering algorithm
    TODO:
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
        - rescale all data by sqrt(2)*sigma so you can get rid of the div by twosigma2 operation?
            - only applies to Gaussian kernel, not Cauchy
    """
    cdef Py_ssize_t i, j, k, scouti
    cdef bint allstill
    cdef int iteri=0, nnomerges=0
    cdef int N = data.shape[0] # total num rows (points) in data table
    cdef int ndims = data.shape[1] # num cols in data table
    cdef int *dims = <int *> malloc(ndims*sizeof(int)) # dimension sizes
    cdef int *ndi = <int *> malloc(ndims*sizeof(int)) # n-dimensional index working array
    cdef int *cids = <int *> malloc(N*sizeof(int)) # cluster indices of data points
    if not cids: raise MemoryError("can't allocate cids")
    irange(cids, N) # init cids to consecutive int values
    cdef int M = N # current num scout points (clusters), each data point starts as its own scout
    cdef int nm, newM, npoints, npointsremoved

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
    # working list for keeping track of pending scout merges
    cdef int *mlist = <int *> malloc(M*sizeof(int))
    if not mlist: raise MemoryError("can't allocate mlist")

    # shuffle rows in data (spike ids) to prevent temporal bias using maxgrad:
    randis = np.arange(N)
    np.random.shuffle(randis) # in place
    data = data[randis]
    sortis = randis.argsort()
    # init points and scouts at data point positions, normalize by norm
    # to reduce math in move_scout() nested loops, and allow use of exps lookup
    for i in range(N): # M == N
        for k in range(ndims):
            points[i][k] = data[i, k] / norm
            scouts[i][k] = data[i, k] / norm

    while True:

        #t0 = time.time()
        # merge scouts within rmerge of each other
        newM = merge_scouts(M, sr, scouts, mlist, rmerge, rmerge2, still, N, cids, ndims)
        #print('merge_scouts took %.3f sec' % (time.time()-t0))
        #break

        if newM != M: # at least one merger happened on this iter
            M = newM
            printf('M=%d\n', M) # print the value of M
            nnomerges = 0 # reset
        else: # no mergers happened on this iter
            nnomerges += 1 # inc

        if nnomerges == maxnnomerges:
            break

        # move scouts up their local density gradient
        for i in prange(M, nogil=True, schedule='dynamic'):
            scouti = sr[i]
            if not still[scouti]: # only move scout points that aren't frozen
                move_scout(scouti, sr, scouts, points, exps, still, maxgrad,
                           N, ndims, alpha, rneigh, rneigh2, minmove2)

        printf('.\n')

        iteri += 1

        allstill = True
        for i in range(M):
            if still[sr[i]] == False:
                allstill = False
                break
        if allstill:
            break

    printf('\n')
    
    # remove clusters with less than minpoints
    nm = 0
    npointsremoved = 0
    for i in range(M):
        npoints = 0 # reset
        # tally up npoints in cluster sr[i]
        scouti = sr[i]
        for j in range(N): # TODO: could this maybe be prange?
            if cids[j] == scouti:
                npoints += 1
        if npoints < minpoints:
            #printf('cluster %d has only %d points', i, npoints)
            # remove cluster i by merging it into "cluster" -1
            mlist[nm] = i
            nm += 1
            npointsremoved += npoints

    if nm > 0:
        M = merge(-1, mlist, nm, M, sr, still, N, cids)

    printf('%d points (%.1f%%) and %d clusters deleted for having less than %d points each\n',
           npointsremoved, npointsremoved/(<double>N)*100, nm, minpoints)

    # for display, restore sigma dependent params to be unnormalized by norm:
    rmerge *= norm
    rneigh *= norm
    minmove *= norm

    cdef int nmoving=0
    for i in range(M):
        if not still[sr[i]]:
            nmoving += 1
    printf('nniters: %d\n',iteri)
    printf('nclusters: %d\n', M)
    printf('sigma: %.3f, rneigh: %.3f, rmerge: %.3f, alpha: %.3f, maxgrad: %d\n',
           sigma, rneigh, rmerge, alpha, maxgrad)
    printf('nmoving: %d, minmove: %f\n', nmoving, minmove)
    printf('still array:\n[')
    for i in range(M):
        printf('%d, ', still[sr[i]])
    printf(']\n')

    # build returnable numpy ndarray for cids
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] np_cids = np.empty(N, dtype=np.int32)
    for i in range(N):
        np_cids[i] = cids[i]
    np_cids = np_cids[sortis] # undo shuffling

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
    free(mlist)
    return np_cids, np_scouts


cdef int merge_scouts(int M, int *sr, float **scouts, int *mlist, double rmerge,
                      double rmerge2, bint *still, int N, int *cids, int ndims):
    """Merge pairs of scout points within rmerge of each other"""
    cdef Py_ssize_t i=0, j, k, scouti, scoutj
    cdef int nm # number of inner loop mergers
    cdef double d, d2
    cdef bint continuej=False
    
    '''
    memcpy(sr2, sr, M*sizeof(int))
    cdef bint *still2 = <bint *> calloc(M, sizeof(bint))
    if not still2: raise MemoryError("can't allocate still2")
    memcpy(still2, still, M*sizeof(bint))
    '''
    #M = 10000
    while i < M: # iterate i over all (not necessarily consecutively numbered) scouts
        scouti = sr[i]
        nm = 0 # reset
        for j in range(i+1, M): # this inner loop could prolly be prange!!!!
            scoutj = sr[j]
            if still[scouti] and still[scoutj]: # both scouts are frozen
                continue # to next j
            # for each pair of scouts, check if they're within rmerge of each other
            d2 = 0.0 # reset
            for k in range(ndims):
                d = scouts[scouti][k] - scouts[scoutj][k]
                if fabs(d) > rmerge: # break out of k loop, continue to next j
                    continuej = True
                    break # out of k loop
                d2 += d * d
                #if d2 > rmerge2: # no apparent speedup
                #    continuej = True
                #    break # out of k loop
            if continuej:
                continuej = False # reset
                continue # to next j
            if d2 <= rmerge2:
                # queue scoutj to be merged into scouti
                mlist[nm] = j # store sr index of scoutj, not its value
                nm += 1
        # merge all queued scouts into scouti
        if nm > 0:
            M = merge(scouti, mlist, nm, M, sr, still, N, cids)
        i += 1
    return M


cdef void move_scout(int scouti, int *sr, float **scouts, float **points,
                     double *exps, bint *still, int maxgrad,
                     int N, int ndims, double alpha,
                     double rneigh, double rneigh2, double minmove2) nogil:
    """Move a scout up its local density gradient"""
    cdef Py_ssize_t j, k
    cdef int npoints=0
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
            ## TODO: points[j] should be dereferenced as point
            ## outside of this loop:
            ds[k] = points[j][k] - scouts[scouti][k]
            if fabs(ds[k]) > rneigh: # break out of k loop, continue to next j
                continuej = True
                break # out of k loop
            d2s[k] = ds[k] * ds[k] # used twice, so calc it only once
            d2 += d2s[k]
            #if d2 > rneigh2: # no apparent speedup
            #    continuej = True
            #    break # out of k loop
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
            npoints += 1
            if npoints == maxgrad: # this is kinda like doing nearest neighbours though...
                break # out of j loop
    # update scout position in direction of v, normalize by kernel
    # nneighs (and kernel?) will never be 0, because each scout starts as a point
    move2 = 0.0 # reset
    for k in range(ndims):
        move = alpha * v[k] / kernel[k] # normalize by kernel, not just nneighs
        scouts[scouti][k] += move
        move2 += move * move
    if move2 < minmove2:
        still[scouti] = True # freeze scout
            
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
'''
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
'''
cdef int merge(Py_ssize_t scouti, int *mlist, int nm, int M, int *sr,
               bint *still, int N, int *cids) nogil:
    """Merge scouts represented by ordered indices into sr in mlist into scouti, where i < j for scouti = sr[i] and scoutj = sr[j]"""
    cdef Py_ssize_t mi, j, scoutj, src, dst, cii, decr
    #assert nm > 0
    printf('cids before cids loop: ')
    for cii in range(N):
        printf('%d, ', cids[cii])
    printf('\n')
    printf('M: %d\n', M)
    printf('scouti: %d\n', scouti)
    printf('mlist: ')
    for mi in range(nm):
        printf('%d, ', mlist[mi])
    printf('\n')

    # cids loop: do this before sr loop, which shifts contents of sr, thereby messing up
    # dereferencing values in mlist
    for mi in range(nm):
        scoutj = sr[mlist[mi]]
        for cii in range(N): ## TODO: try prange here?
            if cids[cii] == scoutj:
                cids[cii] = scouti # replace all scoutj entries with scouti

    printf('cids after cids loop: ')
    for cii in range(N):
        printf('%d, ', cids[cii])
    printf('\n')

    printf('sr before sr loop: ')
    for j in range(M):
        printf('%d, ', sr[j])
    printf('\n')

    # init mi and j for sr loop
    mi = 1
    printf('mi: %d\n', mi)
    if mi < nm:
        j = mlist[mi] # update to mi'th adjusted j value in mlist
    else:
        j = -1
    printf('j1: %d\n', j)
   
    ## TODO: if say M > 10000, might be more efficient to call memcpy for every contig
    ## block of memory between scouts in mlist, but have to be careful about overlapping
    ## mem ranges - I think there's an alternative to memcpy in that case...

    # various cases:
    # - consecutive values in mlist
    # - consecutive values at start of mlist
    # - mlist of length 1

    # sr loop: delete all mlist entries from sr
    dst = mlist[0]
    for src in range(mlist[0]+1, M): # start just after the first j value in mlist
        printf('src: %d\n', src)
        if src == j:
            # read pointer has reached the next value in mlist, skip this value,
            # ie don't shift contents of sr down at this value of src
            mi += 1
            printf('mi: %d\n', mi)
            if mi < nm:
                j = mlist[mi] # update to mi'th adjusted j value in mlist
            else:
                j = -1
            printf('j: %d\n', j)
        else: # shift contents of sr at src down to dst
            printf('dst: %d\n', dst)
            sr[dst] = sr[src]
            dst += 1
    
    M -= nm # decr num scouts

    printf('sr after sr loop: ')
    for j in range(M):
        printf('%d, ', sr[j])
    printf('\n')

    return M
