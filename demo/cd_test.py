import numpy as np

def concatenate_destroy(arrs):
    """Concatenate list of arrays along 0th axis, destroying them in the process.
    Doesn't duplicate everything in arrays, as does numpy.concatenate. Only
    temporarily duplicates one array at a time, saving memory"""
    if type(arrs) != list:
        raise TypeError('arrays must be in a list')
    #arrs = list(arrs) # don't do this! this prevents destruction of the original arrs
    nrows = 0
    subshape = arrs[0].shape[1::] # dims excluding concatenation dim
    dtype = arrs[0].dtype
    # ensure all arrays in arrs are compatible:
    for i, a in enumerate(arrs):
        nrows += len(a)
        if a.shape[1::] != subshape:
            raise TypeError("array %d has subshape %r instead of %r" %
                           (i, a.shape[1::], subshape))
        if a.dtype != dtype:
            raise TypeError("array %d has dtype %r instead of %r" % (i, a.dtype, dtype))
    subshape = list(subshape)
    shape = [nrows] + subshape

    # unlike np.zeros, it seems np.empty doesn't allocate real memory, but does temporarily
    # allocate virtual memory, which is then converted to real memory as 'a' is filled:
    try:
        a = np.empty(shape, dtype=dtype) # empty only allocates virtual memory
    except MemoryError:
        raise MemoryError("concatenate_destroy: not enough virtual memory to allocate "
                          "destination array. Create/grow swap file?")
        
    rowi = 0
    for i in range(len(arrs)):
        arr = arrs.pop(0)
        nrows = len(arr)
        a[rowi:rowi+nrows] = arr # concatenate along 0th axis
        rowi += nrows
    return a


arrs=[]
shape = [1000, 9, 50]
size = np.prod(shape)
for i in range(10000):
    #arr = np.int16(np.random.randint(0, 2**16-1, size=size))
    #arr.shape = shape
    arr = np.zeros(shape, dtype=np.int16)
    arrs.append(arr)
print('done creating test list of arrays')
a = concatenate_destroy(arrs)


