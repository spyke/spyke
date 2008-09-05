"""Demonstrate use of a processing.Pool"""

from processing import Pool
import time

global outputs
outputs = []


def f(x):
    for i in xrange(10000000): # to slow things down a bit
        a = x*x
    return a

def handleOutput(output):
    """This f'n, as a callback, is blocking.
    It blocks the whole program, regardless of number of processes or CPUs/cores"""
    print 'handleOutput got: %r' % output
    outputs.append(output)
    #print 'pausing'
    #for i in xrange(100000000):
    #    pass

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
    time.sleep(2) # pause so you can watch the parent process in taskman hang around after worker processes exit
