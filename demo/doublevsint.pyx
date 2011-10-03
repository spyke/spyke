# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

import numpy as np
cimport numpy as np

import time

cdef extern from "stdio.h":
    int printf(char *, ...)


def doublevsint(np.ndarray[np.float64_t, ndim=2, mode='c'] ddata,
                np.ndarray[np.int64_t, ndim=2, mode='c'] idata):
    cdef int N = ddata.shape[0] # total num rows (points) in data table
    cdef int ndims = ddata.shape[1] # num cols in data table
    cdef int i, j
    cdef double dd, dd2
    cdef int id, id2
    t0 = time.time()
    for i in range(N):
        for j in range(N-1):
            for k in range(ndims): # iterate over dims for each point
                dd = ddata[i][k] - ddata[j][k]
                dd2 += dd * dd
    printf('dd2=%f, double took %f sec\n', dd2, <double>(time.time()-t0))
    t0 = time.time()
    for i in range(N):
        for j in range(N-1):
            for k in range(ndims): # iterate over dims for each point
                id = idata[i][k] - idata[j][k]
                id2 += id * id
    printf('id2=%lld, int took %f sec\n', id2, <double>(time.time()-t0))
