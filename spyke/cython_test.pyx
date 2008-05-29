"""
compile using something like:
python setup.py build_ext --compiler=mingw32 --inplace
"""

#a = np.zeros((54,25000), dtype=int)

include "Python.pxi" # Includes from the python headers
include "numpy.pxi" # Include the Numpy C API for use via Cython extension code
import_array() # Initialize numpy - this MUST be done before any other code is executed.


def cy_setmat(ndarray a, int val):
    """This is how to index into np arrays quickly,
    just as fast as weave.inline. Assumes 2D array"""
    cdef int i, j
    cdef int *ap = <int *>a.data # int pointer to .data field
    cdef int nrows=a.dimensions[0]
    cdef int ncols=a.dimensions[1]
    for i from 0 <= i < nrows:
        for j from 0 <= j < ncols:
            ap[i*ncols+j] = val # a[i][j] also works, but is just as slow as a[i,j]

cpdef cy_recurse(int val):
    """Recursion works in Cython!"""
    if val > 0:
        val -= 1
        print 'recurse...'
        cy_recurse(val)
    print 'done'


cpdef cy_inc(int N=1000000000): # with def: 2.1 sec, width cpdef: 1.4 sec. weave.inline is also 1.4 sec
    """In place incrementation is just as fast as in weave.inline"""
    cdef int i=0
    while i < N:
        i += 1
    return i


cpdef point(ndarray a):
    """Print out all the values in a 1D array, index directly into
    array data quickly with a pointer of the right size"""
    cdef int *ap = <int *>a.data # int pointer to .data field
    cdef int n = a.dimensions[0]
    for i from 0 <= i < n:
        print ap[i]



#timeit.Timer('cy_inc(1000000000)', 'from __main__ import cy_inc').timeit(1)
