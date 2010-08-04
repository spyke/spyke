"""Demonstrate use of a threadpool.ThreadPool

Hmm, unsure about this. Seem to be able to get a numpy call to multithread nicely
(100% of a 2 core machine), but calls to my own C code (compiled from Cython code)
doesn't multithread. This suggests that my Cython code is calling the interpreter
somewhere/somehow and touching some Python objects, which engages the GIL???

update: probably need to use "with nogil" in cython code, see cython docs
"""

from threadpool import ThreadPool, WorkRequest
import processing
#import cython_test
import cy_thread_test
import numpy as np
import time

global outputs
outputs = []

'''
def f(x):
    for i in xrange(10000000): # to slow things down a bit
        output = x*x
    return output

#f = cython_test.cy_inc

def f():
    return np.sort(np.random.random(10000000))
'''
def f(x, N):
    return cy_thread_test.cy_square(x, N)

def handleOutput(request, output):
    """This f'n, as a callback, is blocking.
    It blocks the whole program, regardless of number of processes or CPUs/cores"""
    print 'handleOutput got: %r, %r' % (request, output)
    outputs.append(output)
    #print 'pausing'
    #for i in xrange(100000000):
    #    pass

if __name__ == '__main__':
    ncpus = processing.cpuCount()
    nthreads = ncpus + 1
    print 'ncpus: %d, nthreads: %d' % (ncpus, nthreads)
    pool = ThreadPool(nthreads) # create a threading pool
    t0 = time.time()
    #arr = np.random.random(10000000)
    #for i, val in enumerate([1000000000]*10):#range(10):
    for i in range(10):
        args = (i, 1000000000)
        print 'queueing task %d' % i
        request = WorkRequest(f, args=args, callback=handleOutput)
        # these requests will only multithread if f is a C extension call?? definitely don't multithread if f is pure Python
        pool.putRequest(request)
    print 'done queueing tasks'
    pool.wait()
    print 'tasks took %.3f sec' % time.time()
    print 'outputs: %r' % outputs
    time.sleep(2) # pause so you can watch the parent thread in taskman hang around after worker threads exit
