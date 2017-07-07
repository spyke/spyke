# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

"""Nicholas Swindale's gradient ascent clustering (GAC) algorithm, a variant of the
mean-shift algorithm

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
            - use some cutoff of local density to specify what's noise and what isn't? skeptical

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
            scout (which will be the case 50% of the time), swap the scout's IDs, then
            proceed as usual? Then update the density for the newly merged cluster

        - try using simplex algorithm for scout position update step, though that might miss
        local maxima

        - try using the n nearest neighbours to calculate gradient, instead of a guassian with
        a sigma. This makes it scale free, but NVS says this often results in situations where
        the gradient is 0 for some reason

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

#cimport cython # this was only needed for decorators like @cython.boundscheck(False)
from cython.parallel import prange #, parallel
import numpy as np
cimport numpy as np
import time

cdef double MERGESCOUTSTIME, MERGETIME, MOVESCOUTSTIME

cdef extern from "math.h" nogil:
    double fabs(double x)
    double exp(double x)
    double sqrt(double x)
    #double ceil(double x)

cdef extern from "stdio.h" nogil:
    int printf(char *, ...)
    cdef void *malloc(size_t) # allocates without clearing to 0
    #cdef void *calloc(size_t, size_t) # allocates with clearing to 0
    cdef void free(void *)

cdef extern from "stdlib.h" nogil:
    ctypedef void const_void "const void"
    void qsort(void *ARRAY, size_t COUNT, size_t SIZE,
               int (*COMPARE)(const_void *, const_void *))
'''
cdef extern from "string.h" nogil:
    cdef void *memset(void *, int, size_t) # sets n bytes in memory to constant
    cdef void *memcpy(void *, void *, size_t) # copy to *dest from *src n bytes
    cdef void *memmove(void *, void *, size_t) # copy to *dest from *src n bytes
'''
# despite GNU C manual, an initial empty declaration doesn't seem necessary for the
# recursive struct definition below, maybe not required in Cython:
#cdef struct Scout

cdef struct Scout:
    int id
    Scout *next # pointer to scout that this scout merged into, if any
    float *pos0 # original position
    float *pos # current position
    ## TODO: this should probably be a counter, not boolean:
    bint still # has scout stopped moving?
    int size # number of points belonging to this scout


# NOTE: stdout is buffered by default in linux. This means anything printed to screen from
# within C code won't show up until it gets a newline, or until you call fflush(stdout).
# Unbuffered output can be forced by running Python with the "-u" switch

#DEF MAXUINT16 = 2**16 - 1
#DEF MAXINT32 = 2**31 - 1
#DEF DEBUG = 0 # could use this for different levels of debug messages
DEF MANHATTAN = False # use Manhattan distance for estimating density? if not, use Euclidean
DEF RETURNCPOSHIST = False # return cluster position history on every iteration
DEF PROFILE = False # print timing information
DEF SORTDIMSBYVARIANCE = False

def gac(np.ndarray[np.float32_t, ndim=2, mode='c'] data,
        double sigma=0.25, double rmergex=0.25, double rneighx=4,
        double alpha=2.0, int maxgrad=1000,
        double minmovex=0.00001, int maxnnomerges=1000, int minpoints=5,
        bint returncpos=False):
    """Nicholas Swindale's gradient ascent clustering (GAC) algorithm, a variant
    of the mean-shift algorithm"""
    cdef Py_ssize_t i, j, k, cid
    cdef bint allstill
    cdef int iteri=0, nnomerges=0
    cdef int N = data.shape[0] # total num rows (points) in data table
    cdef int ndims = data.shape[1] # num cols in data table
    cdef int *dims = <int *>malloc(ndims*sizeof(int)) # dimension sizes
    cdef int *ndi = <int *>malloc(ndims*sizeof(int)) # n-dimensional index working array
    cdef int M = N # current num scout points, each data point starts as its own scout
    cdef int nm, newM, npoints, npointsremoved
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] cids # cluster IDs to return for each point
    #cdef np.ndarray[np.float32_t, ndim=2, mode='c'] cpos # cluster positions

    IF PROFILE: 
        global MERGESCOUTSTIME, MERGETIME, MOVESCOUTSTIME
        MERGESCOUTSTIME = 0.0
        MERGETIME = 0.0
        MOVESCOUTSTIME = 0.0
   
    # Normalize all data-related variables by norm to avoid having to
    # do so in move_scout() loop. The data itself is also normalized in the scouti
    # declaration and initialization loop below:
    cdef int lenexps = 1000000
    cdef double norm0 = sqrt(2) * sigma
    cdef double rneigh0 = rneighx * sigma / norm0
    cdef double rneigh02 = rneigh0 * rneigh0
    cdef double norm = norm0 * rneigh0 / sqrt(lenexps)
    # radius around scout to include data for gradient calc:
    cdef double rneigh = sqrt(lenexps)
    cdef double rneigh2 = lenexps # rneigh * rneigh
    #printf('norm: %f, rneigh: %.1f, rneigh2: %.1f\n', norm, rneigh, rneigh2)
    # radius within which scouts are merged:
    cdef double rmerge = rmergex * sigma / norm
    cdef double rmerge2 = rmerge * rmerge
    # min motion in any direction in ndims space req'd for scout to be considered moving:
    cdef double minmove = minmovex * sigma * alpha / norm
    cdef double minmove2 = minmove * minmove

    # pre-calc exp function:
    #t0 = time.time()
    cdef double *exps = <double *>malloc(lenexps*sizeof(double)) # pre-calced exp function
    if not exps: raise MemoryError("can't allocate exps\n")
    for i in range(lenexps): ## TODO: could use prange here
        exps[i] = exp(-<double>i / lenexps * rneigh02) # watch out for int div
    #print('exps malloc took %.3f sec' % (time.time()-t0))
    
    # working list for keeping track of pending scout merges
    cdef int *mlist = <int *>malloc((M+1)*sizeof(int))
    if not mlist: raise MemoryError("can't allocate mlist\n")

    IF SORTDIMSBYVARIANCE:
        # sort data along dimension of max variance, ostensibly the one with most information
        # about distance between points:
        stds = data.var(axis=0)
        sortdimis = stds.argsort()[::-1] # dim indices in order of decreasing variance
        data = data[:, sortdimis].copy() # reorder dims by dec variance, copy for C contig
        print('Reordered dimensions of data by variance: %s' % sortdimis)
    ELSE:
        # always sort along dimension 0. Due to subsampling, sorting along various dimensions
        # can affect clustering results. This can be mitigated by increasing maxgrad.
        # Sorting along a fixed dimension allows the user to choose which is most important
        # to split data along
        pass
    sortis = data[:, 0].argsort()
    data = data[sortis]
    sortis = sortis.argsort() # sorting of points (not dimensions) needs to be undone later
    IF SORTDIMSBYVARIANCE:
        print('Sorted data along dimension %d' % sortdimis[0])
    #ELSE:
    #    print('Sorted data along dimension 0')
    
    # declare placeholder scout:
    cdef Scout *scouti

    # declare and init "junk" scout for points that are thrown out:
    cdef Scout *junk = <Scout *>malloc(sizeof(Scout))
    junk.id = -1
    junk.next = NULL
    junk.pos0 = NULL
    junk.pos = NULL
    junk.still = False

    # declare and init scouts, normalize data:
    cdef Scout *scouts = <Scout *>malloc(N*sizeof(Scout))
    if not scouts: raise MemoryError("can't allocate scouts\n")
    for i in range(N): ## TODO: could use prange here
        scouti = scouts+i
        scouti.id = i
        scouti.next = NULL
        scouti.pos0 = <float *>malloc(ndims*sizeof(float))
        scouti.pos = <float *>malloc(ndims*sizeof(float))
        scouti.still = False
        scouti.size = 1
        for k in range(ndims):
            scouti.pos0[k] = data[i, k] / norm
            scouti.pos[k] = data[i, k] / norm

    # init a shrinking view of scouts, s:
    cdef Scout **s = <Scout **>malloc(N*sizeof(Scout *)) # array of pointers to Scout pointers
    if not s: raise MemoryError("can't allocate s\n")
    for i in range(M): ## TODO: could use prange here
        s[i] = scouts+i

    IF RETURNCPOSHIST:
        cposhist = [] # cluster position history list, one entry per timepoint

    while True:

        IF RETURNCPOSHIST:
            # store history of cluster positions, scale up by norm again:
            cpos = np.zeros((M, ndims), dtype=np.float32)
            ## TODO: for clarity, maybe exclude scouts with < minpoints
            for i in range(M): ## TODO: could use prange here
                scouti = s[i]
                for k in range(ndims):
                    cpos[i, k] = scouti.pos[k] * norm # undo previous normalization
            cposhist.append(cpos)

        # merge current scouts within rmerge of each other:
        IF PROFILE: t0 = <double>time.time()
        newM = merge_scouts(M, s, mlist, rmerge, rmerge2, ndims)
        IF PROFILE: MERGESCOUTSTIME += (<double>time.time() - t0)

        if newM != M: # at least one merger happened on this iter
            M = newM
            printf('%d', M) # print the value of M
            nnomerges = 0 # reset
        else: # no mergers happened on this iter
            nnomerges += 1 # inc

        if nnomerges == maxnnomerges:
            break

        # move current scouts up their local density gradient:
        IF PROFILE: t0 = <double>time.time()
        for i in prange(M, nogil=True, schedule='dynamic'):
            scouti = s[i]
            if not scouti.still: # only move scouts that aren't frozen
                move_scout(scouti, scouts, exps, maxgrad,
                           N, ndims, alpha, rneigh, rneigh2, minmove2)
        IF PROFILE: MOVESCOUTSTIME += (<double>time.time() - t0)
        printf('.')

        iteri += 1

        allstill = True
        for i in range(M): ## TODO: could use prange here
            if s[i].still == False:
                allstill = False
                break
        if allstill:
            break
        '''
        printf('before qsort:\n')
        for i in range(M):
            printf('%.3f, ', s[i].pos[0])
        printf('\n')
        '''
        # resort scouts along dimension 0:
        qsort(s, M, sizeof(Scout *), cmp_scouts_k0)
        '''
        for i in range(1, M):
            if s[i].pos[0] < s[i-1].pos[0]:
                printf('error: scouts aren't sorted along dimension 0\n')
                return

        printf('after qsort:\n')
        for i in range(M):
            printf('%.3f, ', s[i].pos[0])
        printf('\n')
        '''
        
    printf('\n')
    IF PROFILE:
        printf('merge scouts: %.9f sec\n', MERGESCOUTSTIME)
        printf('     merge(): %.9f sec\n', MERGETIME)
        printf(' move scouts: %.9f sec\n', MOVESCOUTSTIME)
        printf('       total: %.9f sec\n', MERGESCOUTSTIME+MOVESCOUTSTIME)

    cids = walkscouts(scouts, N) # build cids from merge history in scouts

    # remove clusters with less than minpoints
    nm = 0
    npointsremoved = 0
    for i in range(M):
        if s[i].size < minpoints:
            #printf('cluster %d has only %d points', cid, npoints)
            # remove cluster cid by merging it into "cluster" -1
            mlist[nm] = i
            nm += 1
            npointsremoved += s[i].size
    if nm > 0:
        M = merge(junk, mlist, nm, s, M)
 
    printf('%d points (%.1f%%) and %d clusters deleted for having less than %d points each\n',
           npointsremoved, npointsremoved/(<double>N)*100, nm, minpoints)

    # sort s by decreasing cluster size and relabel scouts in that order
    qsort(s, M, sizeof(Scout *), cmp_scouts_decsize)
    for i in range(M):
        s[i].id = i

    # rebuild cids after labelling small clusters as junk and sorting by decreasing
    # cluster size
    cids = walkscouts(scouts, N)
    cids = cids[sortis] # restore original point ordering

    if returncpos:
        # build returnable numpy array of cluster positions, scale up by norm again:
        cpos = np.zeros((M, ndims), dtype=np.float32)
        for i in range(M): ## TODO: could use prange here
            scouti = s[i]
            for k in range(ndims):
                cpos[i, k] = scouti.pos[k] * norm # undo previous normalization

    # for display, restore sigma dependent params to be unnormalized by norm:
    rmerge *= norm
    rneigh *= norm
    minmove *= norm

    cdef int nmoving=0
    for i in range(M): ## TODO: could use prange here
        if not s[i].still:
            nmoving += 1
    printf('niters: %d\n', iteri)
    printf('nclusters: %d\n', M)
    printf('sigma: %.3f, rneigh: %.3f, rmerge: %.3f, alpha: %.3f, maxgrad: %d\n',
           sigma, rneigh, rmerge, alpha, maxgrad)
    printf('nmoving: %d, minmove: %f\n', nmoving, minmove)
    printf('still flags:\n[')
    for i in range(M):
        printf('%d, ', s[i].still)
    printf(']\n')

    free(dims)
    free(ndi)
    free(exps)
    free(mlist)
    ## TODO: do the pos and pos0 fields of each scout need to be freed explicitly?
    free(scouts)
    free(s)
    
    IF RETURNCPOSHIST:
        return cids, cposhist
    if returncpos:
        return cids, cpos
    else:
        return cids


cdef inline int merge_scouts(int M, Scout **s, int *mlist, double rmerge,
                             double rmerge2, int ndims):
    """Merge pairs of scouts within rmerge of each other"""
    cdef Py_ssize_t i=0, j, k
    cdef Scout *scouti
    cdef Scout *scoutj
    cdef int nm # number of inner loop mergers
    cdef int lo, hi, step
    cdef double d, d2
    cdef bint continuej=False
    IF PROFILE: global MERGETIME

    while i < M: # iterate i over current scouts
        scouti = s[i]
        lo = bsearch_scouts_k0(s, i+1, M, scouti.pos[0]-rmerge)
        hi = bsearch_scouts_k0(s, i+1, M, scouti.pos[0]+rmerge)
        nm = 0 # reset
        for j in range(lo, hi): ## TODO: could maybe use prange here, except for nm inc?
            scoutj = s[j]
            if scouti.still and scoutj.still: # both scouts are frozen
                continue # to next j
            # for each pair of scouts, check if they're within rmerge of each other
            d2 = 0.0 # reset
            for k in range(ndims):
                d = scouti.pos[k] - scoutj.pos[k]
                if fabs(d) > rmerge: # break out of k loop, continue to next j
                    continuej = True
                    break # out of k loop
                d2 += d * d
                #if d2 > rmerge2: # no apparent speedup
                #    continuej = True
                #    break # out of k loop
            if continuej:
                # got fabs(d) > rmerge along k'th dimension
                continuej = False # reset
                continue # to next j
            if d2 <= rmerge2:
                # queue scoutj to be merged into scouti
                mlist[nm] = j # store s index of scoutj, not its scoutj.id
                nm += 1
        # merge all queued scouts into scouti
        if nm > 0:
            IF PROFILE: t0 = <double>time.time()
            M = merge(scouti, mlist, nm, s, M)
            IF PROFILE: MERGETIME += (<double>time.time() - t0)
        i += 1
    return M


cdef inline void move_scout(Scout *scouti, Scout *scouts, double *exps,
                            int maxgrad, int N, int ndims, double alpha,
                            double rneigh, double rneigh2, double minmove2) nogil:
    """Move a scout up its local density gradient"""
    cdef Py_ssize_t j, k
    cdef float *pos
    cdef float *pos0
    cdef int npoints = 0
    cdef bint continuej = False
    cdef double d2, kern, move, move2
    cdef double *ds = <double *>malloc(ndims*sizeof(double))
    IF MANHATTAN:
        cdef double *d2s = <double *>malloc(ndims*sizeof(double))
        cdef double *kernel = <double *>malloc(ndims*sizeof(double))
    ELSE:
        cdef double kernel = 0
    cdef double *v = <double *>malloc(ndims*sizeof(double))
    
    # set some local vars:
    IF MANHATTAN:
        dfill(kernel, 0, ndims)
    dfill(v, 0, ndims)
    
    # measure gradient v:
    pos = scouti.pos
    # search along sorted dim 0 for 1st and last data point within rneigh of scouti:
    cdef int lo = bsearch_points_k0(scouts, N, pos[0]-rneigh)
    cdef int hi = bsearch_points_k0(scouts, N, pos[0]+rneigh)
    """Subsample, though unfortunately, the actual number of points used to calculate the
    gradient will depend on what fraction of those within rneigh along 0'th dimension
    actually fall within rneigh2 in Euclidean space. To roughly make up for this, maybe
    maxgrad should be scaled by ndims? Or by ndim equivalent of the volume of a sphere? Or
    maybe nicer solution would be to re-sort and do binary split search along each
    successive dimension, though that might not pay off"""
    cdef int step = (hi - lo) // maxgrad
    if step == 0:
        step = 1
    # iterate over evenly subsampled original data points within rneigh of scouti:
    for j from lo <= j < hi by step:
        pos0 = scouts[j].pos0
        d2 = 0.0 # reset
        # for speed, iterate starting from dim 1, then do dim 0 last on its own,
        # outside the loop. This increases chance of skipping j, because
        # we already know that scouti is <= rneigh along dim 0, and the rest of the
        # dims are (maybe) in order of decreasing variance:
        for k in range(1, ndims): # iterate over dims for each point
            ds[k] = pos0[k] - pos[k]
            if fabs(ds[k]) > rneigh: # break out of k loop, continue to next j
                continuej = True
                break # out of k loop
            IF MANHATTAN:
                d2s[k] = ds[k] * ds[k] # used twice, so calc it only once
                d2 += d2s[k]
            ELSE:
                d2 += ds[k] * ds[k]
            #if d2 > rneigh2: # no apparent speedup for 3D
            #    continuej = True
            #    break # out of k loop
        if continuej:
            continuej = False # reset
            continue # to next j
        # now fill in ds[0], d2s[0], and top up d2:
        ds[0] = pos0[0] - pos[0]
        IF MANHATTAN:
            d2s[0] = ds[0] * ds[0] # used twice, so calc it only once
            d2 += d2s[0]
        ELSE:
            d2 += ds[0] * ds[0]
        if d2 <= rneigh2: # do the calculation
            IF not MANHATTAN:
                kern = exps[<int>(d2)] # data rescaled for Gaussian lookup table
            for k in range(ndims):
                # v is ndim vector of sum of kernel-weighted distances between
                # scouti and all points within rneigh
                #kern = exp(-d2s[k] / (2 * sigma2)) # Gaussian kernel
                IF MANHATTAN:
                    kern = exps[<int>(d2s[k])] # data rescaled for Gaussian lookup table
                    #kern = sigma2 / (d2s[k] + sigma2) # Cauchy kernel
                    kernel[k] += kern
                v[k] += ds[k] * kern # this is why you can't store fabs of ds[k]!
            IF not MANHATTAN:
                kernel += kern
            npoints += 1

    # update scouti position in direction of v, normalize by kernel.
    # kernel will never be 0, because each scout starts at a point:
    move2 = 0.0 # reset
    if npoints > 0:
        for k in range(ndims):
            IF MANHATTAN:
                move = alpha * v[k] / kernel[k] # normalize by kernel, not just nneighs
            ELSE:
                move = alpha * v[k] / kernel # normalize by kernel, not just nneighs
            pos[k] += move
            move2 += move * move
    #printf('%d,', npoints)
    if move2 < minmove2:
        scouti.still = True # freeze it
            
    free(ds)
    IF MANHATTAN:
        free(d2s)
        free(kernel)
    free(v)


cdef inline int bsearch_points_k0(Scout *scouts, int N, float pos0k0) nogil:
    """Use bisection of point positions in scouts sorted along k=0 to find first
    entry >= pos0k0. In other words, find the first index i such that
    pos0k0 <= scouts[i].pos0[0]. When there is no such index i, set i = N = len(scouts).

    Adapted from numpy/core/src/multiarray/item_selection.c/local_search_left(),
    because C's bsearch only returns an exact hit, not the insertion index of key into arr.
    """
    cdef int mini=0, maxi=N, midi
    while mini < maxi:
        midi = mini + ((maxi - mini) >> 1) # bitshift right 1 is same as div 2
        if scouts[midi].pos0[0] < pos0k0:
            mini = midi + 1
        else:
            maxi = midi
    return mini


cdef inline int bsearch_scouts_k0(Scout **s, int i, int M, float posk0) nogil:
    """Use bisection of scout positions in s sorted along k=0 to find first
    entry >= posk0. In other words, find the first index j >= i such that
    posk0 <= s[j].pos[0]. When there is no such index j, set j = M = len(s).

    Adapted from numpy/core/src/multiarray/item_selection.c/local_search_left(),
    because C's bsearch only returns an exact hit, not the insertion index of key into arr.
    """
    cdef int mini=i, maxi=M, midi
    while mini < maxi:
        midi = mini + ((maxi - mini) >> 1) # bitshift right 1 is same as div 2
        if s[midi].pos[0] < posk0:
            mini = midi + 1
        else:
            maxi = midi
    return mini


cdef inline int merge(Scout *scouti, int *mlist, int nm, Scout **s, int M) nogil:
    """Take scouts represented by ordered s indices in mlist and merge them into scouti"""
    cdef Py_ssize_t mi, src, dst, n, j
    cdef Scout **dstp
    cdef Scout **srcp
    #assert nm > 0
    '''
    printf('M: %d\n', M)
    printf('scouti.id: %d\n', scouti.id)
    printf('mlist: ')
    for mi in range(nm):
        printf('%d, ', mlist[mi])
    printf('\n')
    printf('mlist s.ids: ')
    for mi in range(nm):
        printf('%d, ', s[mlist[mi]].id)
    printf('\n')
    '''
    dst = mlist[0]
    ## maybe, should this last value be decr by nm? M is constantly changing throughout
    ## this loop. Maybe M should be decr by 1 on the fly instead of all at once at the end?
    mlist[nm] = M # make final mlist[mi+1] give correct value
    for mi in range(nm):
        j = mlist[mi]
        s[j].next = scouti # point s[j] to scouti to record merge history
        scouti.size += s[j].size # grow scouti's size
        src = j + 1
        n = mlist[mi+1] - src
        #printf('src: %d, dst: %d, n: %d\n', src, dst, n)
        if n == 0: # nothing to copy, probably consecutive values in mlist
            continue
        # shift contents of s at src down to dst:
        dstp = s+dst
        srcp = s+src
        dst += n # inc dst here for next loop, before n gets decr in while loop below
        #nbytes = n * sizeof(Scout)
        #memmove(s+dst, s+src, nbytes) # allows src and dst to overlap
        #memcpy(s+dst, s+src, nbytes) # doesn't allow src and dst to overlap

        # Cython translation of what memcpy and memove probably do. Can get away with this
        # pointer method because dst < src always, and srcp and dstp reference the same
        # object (s). The fastest implementation is cited as:
        #
        #void *memcpy(void *dest, const void *src, size_t n)
        #{
        #    char *dp = dest;
        #    const char *sp = src;
        #    while (n--)
        #        *dp++ = *sp++;
        #    return dest;
        #}
        # on the web, e.g. http://clc-wiki.net/wiki/memcpy. Although the above is a byte
        # level copy, and word level copy should be faster. The following Cython code seems
        # to run as fast as memmove. Hand optimizing the generated .c file to use -- and *
        # and ++ didn't seem to help:
        while n:
            dstp[0] = srcp[0]
            dstp += 1
            srcp += 1
            n -= 1
        '''
        printf('s.ids after shift: ')
        for j in range(M):
            printf('%d, ', s[j].id)
        printf('\n')
        '''
    M -= nm # decr num scouts
    return M


cdef inline int cmp_scouts_k0(const_void *dps0, const_void *dps1) nogil:
    """Return relative positional order of a pair of scouts along their 0'th dimension.
    In this case, dps0 and dps1 are actually double pointers, each of which is an
    entry in the s array"""
    # typecast to Scout double pointers, deref to single pointers, then access .pos:
    if (<Scout **>dps0)[0].pos[0] < (<Scout **>dps1)[0].pos[0]:
        return -1 # report that s0 should come before s1
    return 1 # report that s1 should come before s0
    # don't worry about case when they're equal
    
cdef inline int cmp_scouts_decsize(const_void *dps0, const_void *dps1) nogil:
    """Return *decreasing* relative size of a pair of scouts.
    In this case, dps0 and dps1 are actually double pointers, each of which is an
    entry in the s array"""
    # typecast to Scout double pointers, deref to single pointers, then access .size:
    cdef int d = (<Scout **>dps1)[0].size - (<Scout **>dps0)[0].size
    # if s1 is bigger than s0, return +ve value; if s1 is smaller than s0, return -ve value;
    # if equal, return 0. +ve value means they're in the wrong order and need to be swapped
    return d

cdef inline np.ndarray[np.int32_t, ndim=1, mode='c'] walkscouts(Scout *scouts, int N):
    """Return numpy array of cluster ids by walking each scout's merge path
    to its final destination scout"""
    cdef int i
    cdef Scout *scouti
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] cids
    cids = np.zeros(N, dtype=np.int32)
    # dereference old scout IDs to new ones after all merging is done:
    for i in range(N): ## TODO: this could be prange? maybe not if cids is ndarray?
        scouti = scouts+i
        #printf('%d\n', <void *>scouti.next)
        while scouti.next != NULL:
            scouti = scouti.next # walk scouts[i]'s merge path to its final destination
        cids[i] = scouti.id
    return cids


'''
cdef void span(long *lohi, int start, int end, int N) nogil:
    """Fill len(N) lohi array with fairly equally spaced int
    values, from start to end"""
    cdef Py_ssize_t i
    cdef int step
    step = <int>ceil(<double>(end - start) / N) # round up
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

cdef inline void dfill(double *a, double val, long long n) nogil:
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
