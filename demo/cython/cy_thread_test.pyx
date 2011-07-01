cdef square(int x, int N):
    """Square x N number of times before returning the result.
    Not accessible from Python code"""
    cdef int i, y
    for i from 0 <= i < N:
        y = x*x
    return y

cpdef cy_square(int x, int N=1000000000):
    """Accessible from Python code"""
    return square(x, N)
