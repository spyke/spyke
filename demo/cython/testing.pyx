# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

#cimport cython
from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
import time

cdef extern from "stdio.h":
    int printf(char *, ...) nogil
    cdef void *malloc(size_t) nogil # allocates without clearing to 0
    cdef void *calloc(size_t, size_t) nogil # allocates with clearing to 0
    cdef void free(void *) nogil


def testing():#np.ndarray[np.uint32_t, ndim=1, mode='c'] ndi,
              #np.ndarray[np.uint32_t, ndim=1, mode='c'] dims):
    '''
    printf('float: %d\n', sizeof(float))
    printf('double: %d\n', sizeof(double))

    printf('char: %d\n', sizeof(char))
    printf('short: %d\n', sizeof(short))
    printf('int: %d\n', sizeof(int))
    printf('long: %d\n', sizeof(long))
    printf('long long: %d\n', sizeof(long long))
    
    print 'dims=', dims
    print 'prod(dims)=', prod(dims)
    print 'ndi=', ndi
    print 'li=', ndi2li(ndi, dims)

    cdef double *ds = <double *>calloc(10, sizeof(double))
    print ds[0]
    ds[0] = 5
    print ds[0]
    free(ds)
    print 'i =', looping()

    cdef long long i=0, n=1000000000, sum=0
    cdef long long *a = <long long *>malloc(n*sizeof(long long))
    if not a:
        raise MemoryError("can't allocate a")
    print 'starting loop:'
    t0 = time.time()
    # schedule: 'dynamic' and 'runtime' suck, 'auto' raises an error. Leaving as default None,
    # or setting to 'static' or 'guided' all perform the same. static is probably most
    # appropriate for loops where each iteration takes the same amount of time
    for i in prange(n, nogil=True):
    #for i in range(n):
        sum += 1
        a[i] = 2*i + i/(i+1)
    print('loop took %.3f sec' % (time.time()-t0))
    printf('sum = %lld\n', sum)
    printf('a[:100] = ')
    for i in range(100):
        printf('%lld, ', a[i])
    printf('\n')
    printf('a[-100:] = ')
    for i in range(n-100, n):
        printf('%lld, ', a[i])
    printf('\n')
    '''
    cdef int i, j
    cdef int blarg=999, temp, ndims=4
    cdef int *a
    # test creating thread-local variables to avoid race conditions.
    # Looks like there aren't any when you use a parallel block!:
    printf('blarg before parallel = %d\n', blarg)
    with nogil, parallel():
        printf('blarg at start of parallel = %d\n', blarg)
        a = <int *>malloc(ndims*sizeof(int))
        for j in range(ndims):
            a[j] = 2*j
        for i in prange(4):
            blarg = i
            if i == 2:
                blarg = 222
                # do something slowish:
                for j in range(1000000000):
                    temp = j*j
            printf('end of loop, i, blarg = %d, %d\n', i, blarg)
            printf('end of loop %d, a = ', i)
            for j in range(ndims):
                printf('%d ', a[j])
            printf('\n')
        printf('end of parallel, i, blarg = %d, %d\n', i, blarg)
    printf('outside parallel, i, blarg = %d, %d\n', i, blarg)


cdef long long prod(np.ndarray[np.uint32_t, ndim=1, mode='c'] a) nogil:
    """Return product of entries in uint32 array a"""
    cdef long result, n
    cdef Py_ssize_t i
    n = a.shape[0] # this doesn't invoke Python apparently
    result = 1
    for i in range(n):
        result *= a[i]
    return result

cdef long long ndi2li(np.ndarray[np.uint32_t, ndim=1, mode='c'] ndi,
                      np.ndarray[np.uint32_t, ndim=1, mode='c'] dims) nogil:
    """Convert n dimensional index in array ndi to linear index. ndi
    and dims should be the same length, and each entry in ndi should be
    less than its corresponding dimension size in dims"""
    cdef long long li, pr=1
    cdef Py_ssize_t di, ndims
    ndims = ndi.shape[0]
    li = ndi[ndims-1] # init with index of deepest dimension
    # iterate from ndims-1 to 0, from 2nd deepest to shallowest dimension
    # either syntax works, and both seem to be C optimized:
    #for di in range(ndims-1, 0, -1):
    for di from ndims-1 >= di > 0:
        pr *= dims[di] # running product of dimensions
        li += ndi[di-1] * pr # accum sum of products of current ndi and all deeper dimensions
    return li

cdef long long nothing() nogil:
    cdef long long result
    result = 10
    return result

cdef int looping() nogil:
    cdef Py_ssize_t i, M=10
    for i in range(M):
        if i == 4:
            M = 5
    return i
