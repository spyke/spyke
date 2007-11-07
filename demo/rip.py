""" rip demo """

import cPickle

import numpy

import spyke.surf
from spyke.detect import Collection

class Ripper(object):
    def __init__(self, fobj, surf_name=None, op=None):
        self.collection = cPickle.load(fobj)
        self.surf = surf_name and spyke.surf(surf_name) or op.surf_file
        self.dstream = surf_name and self.surf.parse() or op.dstream

    def rip(self):
        # start ripping process
        pass

if __name__ == '__main__':
    import sys
    from spyke.gui.plot import Opener
    if len(sys.argv) > 2:
        surf_name = sys.argv[1]
        collection_name = sys.argv[2]
        op = None
    else:
        op = Opener()
        collection_name = 'collection.pickle'
        surf_name = None

    try:
        fcol = file(collection_name)
        rip = Ripper(fcol, surf_name, op)
    finally:
        fcol.close()

