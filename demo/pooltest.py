"""Demonstrates use of a processing.Pool"""

from processing import Pool
import time

global outputs
outputs = []

def f(x):
    for i in xrange(2000000): # to slow things down a bit
        a = x*x
    return a

def handleOutput(output):
    """This f'n, as a callback, is blocking.
    I think it blocks the whole program,
    regardless of number of processes or cores??"""
    print 'handleOutput got: %r' % output
    outputs.append(output)

if __name__ == '__main__':
    pool = Pool() # create a processing pool with as many processes as there are CPUs/cores on this machine
    results = []
    t0 = time.clock()
    for i in range(10):
        print 'queueing task %d' % i
        result = pool.applyAsync(f, args=(i,), callback=handleOutput) # evaluate f(i) asynchronously
        results.append(result)
    print 'done queueing tasks, result objects are: %r' % results
    pool.close()
    pool.join()
    print 'tasks took %.3f sec' % time.clock()
    print 'outputs: %r' % outputs

    #import pdb; pdb.set_trace()
    #time.sleep(10)
    #result = pool.applyAsync(f, [10])     # evaluate "f(10)" asynchronously
    #print result.get(timeout=1)           # prints "100" unless your computer is *very* slow
    #print pool.map(f, range(10))          # prints "[0, 1, 4,..., 81]"
