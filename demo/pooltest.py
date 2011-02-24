"""Demonstrate use of a processing.Pool"""

import multiprocessing as mp
ps = mp.current_process
import time
import numpy as np

#global outputs
#outputs = []


def f(x):
    n = int(np.random.random(1)[0] * 100000000)
    for i in xrange(n): # to slow things down a bit
        output = x*x
    return [x, output]

def g(x, y):
    n = int(np.random.random(1)[0] * 100000000)
    for i in xrange(n): # to slow things down a bit
        output = x*y
    return [x, y, output]


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

def callsearchblock(blockrange):
    """Run current process' Detector on blockrange"""
    detector = ps().detector
    result = detector.searchblock(blockrange)
    print("%s done" % ps().name)
    return result

def initializer(detector):
    """Save pickled copy of the Detector to the current process"""
    ps().detector = detector


class Detector(object):
    def detect(self):
        blockranges = np.asarray([[0, 10000000], [10000000, 20000000], [20000000, 30000000]])

        ncores = mp.cpu_count() # 1 per core
        nprocesses = min(ncores, len(blockranges))
        pool = mp.Pool(nprocesses, initializer, (self,))
        results = pool.map(callsearchblock, blockranges, chunksize=1)
        pool.close()
        # results is a list of (spikes, wavedata) tuples, and needs to be unzipped
        spikes, wavedata = zip(*results)
        print spikes, wavedata

    def searchblock(self, blockrange):
        #t = np.random.random(1) * 10 # sec
        #time.sleep(t[0])
        return f(blockrange[0]), g(blockrange[0], blockrange[1])

if __name__ == '__main__':
    det = Detector()
    det.detect()
