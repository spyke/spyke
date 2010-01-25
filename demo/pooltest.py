"""Demonstrate use of a processing.Pool"""

from multiprocessing import Pool
import time

#global outputs
#outputs = []


def f(x):
    for i in xrange(100000000): # to slow things down a bit
        output = x*x
    return [x, output]
'''
def handleOutput(output):
    """This f'n, as a callback, is blocking.
    It blocks the whole program, regardless of number of processes or CPUs/cores"""
    print 'handleOutput got: %r' % output
    outputs.append(output)
    #print 'pausing'
    #for i in xrange(100000000):
    #    pass
'''
if __name__ == '__main__':
    pool = Pool() # create a processing pool with as many processes as there are CPUs/cores
                  # on this machine, or set arg to n to use exactly n processes
    #results = []
    #for i in range(10):
    #    print 'queueing task %d' % i
    #    result = pool.apply_async(f, args=(i,), callback=handleOutput) # evaluate f(i) asynchronously
    #    results.append(result)
    ncpus = multiprocessing.cpu_count()
    t0 = time.clock()
    results = pool.map(f, range(2*ncpus)) # make it int multiple of ncpus for efficiency
    print('tasks took %.3f sec' % time.clock())
    #print 'done queueing tasks, result objects are: %r' % results
    print(results)
    pool.close()
    pool.join()
    #print 'outputs: %r' % outputs
    time.sleep(10) # pause so you can watch the parent process in taskman hang around after worker processes exit
