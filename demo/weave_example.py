"""For numpy arrays declared in Python code, you can access
the array data by going arr[i*ncols + j] without having to
take data type into account. You can also use py_arr to get the
python reference to the array, Narr to get an array of arr's
dimensions, Narr[0] to get nrows, Narr[1] to get ncols, and
I think weave has some other handy shortcuts... see the
scipy/weave/doc/ folder for the tutorial."""


import numpy as np
from scipy import weave

def weave_pure_inc(N):
    code = r"""
           int i=0;
           while (i<N)
               i+=1;
           return_val = i;
           """
    return weave.inline(code, ['N'], compiler='gcc')



def weave_setmat(a, val):
    code = r"""
            int i, j;
            int nrows = Na[0];
            int ncols = Na[1];
            for (i=0; i<nrows; i++)
                for (j=0; j<ncols; j++)
                    a(i,j) = val;
           """
    return weave.inline(code, ['a', 'val'],
                        type_converters=weave.converters.blitz,
                        compiler='gcc')

def weave_pure_setmat(a, val):
    code = r"""
            int i, j;
            int nrows = Na[0];
            int ncols = Na[1];
            for (i=0; i<nrows; i++)
                for (j=0; j<ncols; j++)
                    a[i*ncols + j] = val;
           """
    return weave.inline(code, ['a', 'val'], compiler='gcc')


#timeit.Timer('weave_pure_setmat(a, 5)', 'from __main__ import weave_pure_setmat, a').timeit(100)



def resize_cdef_arr():
    """Demos in weave.inline how to allocate a numpy array directly in C,
    fill it up with values, and then dynamically resize it if it's
    full, allowing you to continue adding values indefinitely. I can't
    get resizing to work if the array was originally declared in Python"""

    #a = np.empty((np.random.randint(50, 150), 2), dtype=np.int64)
    leninc = np.random.randint(50, 150) # inc arr len this much each time
    code = r"""
    #line 53 "weave_example.py"
    npy_intp dimsarr[2];
    dimsarr[0] = leninc; // nrows
    dimsarr[1] = 2;      // ncols
    PyArrayObject *a;
    a = (PyArrayObject *) PyArray_SimpleNew(2, dimsarr, NPY_LONGLONG);

    PyArray_Dims dims;
    int nd = 2;
    dims.len = nd;
    dims.ptr = dimsarr;
    PyObject *dummy;

    printf("a started with length %d\n", PyArray_DIM(a, 0));
    for (int i=0; i<1000; i++) {
        if (i == PyArray_DIM(a, 0)) { // resize to prevent going out of bounds
            dims.ptr[0] += leninc;
            dummy = PyArray_Resize(a, &dims, 0, NPY_ANYORDER);
            if (dummy == NULL) {
                PyErr_Format(PyExc_TypeError, "can't resize a");
                return NULL;
            }
            // don't need dummy anymore I guess, see
            // http://www.mail-archive.com/numpy-discussion@scipy.org/msg13013.html
            Py_DECREF(dummy);
            printf("a is now %d long\n", dims.ptr[0]);
        }
        // get pointer to i,jth entry in data, typecast appropriately,
        // then dereference the whole thing so you can assign
        // a value to it. Using PyArray_GETPTR2 macro is easier than
        // manually doing pointer math using strides
        *((long long *) PyArray_GETPTR2(a, i, 0)) = i;   // assign to ith row, col 0
        *((long long *) PyArray_GETPTR2(a, i, 1)) = 2*i; // assign to ith row, col 1
    }
    //return_val = Na[0];
    //return_val = (PyObject *) a;  // these two both
    return_val = PyArray_Return(a); // seem to work
    """
    a = weave.inline(code, ['leninc'], compiler='gcc')
    return a


if __name__ == '__main__':
    for i in range(20):
        print resize_cdef_arr()

