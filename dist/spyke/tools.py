from __future__ import with_statement   # 2.5 only
""" spyke.tools

Various classes and functions used in multiple places within spyke.
"""

__author__ = 'Reza Lotun'

import cPickle
import gzip
import hashlib


class SpykeError(Exception):
    """ Base spyke error. """
    pass


class CollectionError(SpykeError):
    """ Problem with collection file. """
    pass


def get_sha1(filename, blocksize=2**20):
    """ Gets the sha1 hash of filename (with full path)."""
    m = hashlib.sha1()
    # automagically clean up after ourselves
    with file(filename, 'rb') as f:

        # continually update hash until EOF
        while True:
            block = f.read(blocksize)
            if not block: break
            m.update(block)

    return m.hexdigest()


def load_collection(filename):
    """ Loads a collection file. Returns None if filename is not a collection.
    """
    with file(filename, 'rb') as f:
        try:
            g = gzip.GzipFile(fileobj=f, mode = 'rb')
            col = cPickle.load(g)
        except Exception, e:
            raise CollectionError(str(e))
        g.close()
    return col


def write_collection(collection, filename):
    """ Writes a collection to filename. """
    with file(filename, 'wb') as f:
        try:
            g = gzip.GzipFile(fileobj=f, mode='wb')
            cPickle.dump(collection, g, 2)
        except Exception, e:
            raise CollectionError(str(e))
        g.close()

