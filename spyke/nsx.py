"""Load Blackrock Neural Signal Processing System-generated .nsx files."""

from __future__ import division

__authors__ = ['Martin Spacek']

import numpy as np
import os
import cPickle
from struct import Struct, unpack
import datetime

from core import Stream, NULL


class File(object):
    """Open an .nsx file and expose all of its headers and records as attribs"""
    def __init__(self, fname, path):
        self.fname = fname
        self.path = path
        self.fileSize = os.stat(self.join(fname))[6]
        self.open()
        self._parseFileHeader()

    def join(self, fname):
        return os.path.join(self.path, fname)

    def open(self):
        """(Re)open previously closed .nsx file"""
        self.f = open(self.join(self.fname), 'rb')

    def close(self):
        """Close the .nsx file"""
        self.f.close()

    def is_open(self):
        try:
            return not self.f.closed
        except AttributeError: # self.f unbound
            return False

    def get_datetime(self):
        """Return datetime stamp corresponding to t=0us timestamp"""
        return self.fileheader.datetime

    datetime = property(get_datetime)

    def __getstate__(self):
        """Don't pickle open .nsx file handle on pickle. Also, save space (for .sort files)
        by not pickling all records unless explicitly signalled to do so (for .parse files)
        """
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        if 'f' in d:
            del d['f'] # exclude open .nsx file handle, if any
        return d

    def _parseFileHeader(self):
        """Parse the .nsx file header"""
        self.fileheader = FileHeader()
        self.fileheader.parse(self.f)
        #print('Parsed fileheader')

    def parse(self, force=False, save=True):
        """Parse the .nsx file"""
        pass
        

class FileHeader(object):
    """.nsx file header. Takes an open file, parses in from current file
    pointer position, stores header fields as attribs"""

    #def __len__(self):
    #    return self.nbytes

    def parse(self, f):
        # basic header:
        self.offset = f.tell()
        self.filetype = f.read(8)
        assert self.filetype == 'NEURALCD'
        self.version = unpack('BB', f.read(2))# aka "File Spec", major and minor versions
        self.nbytes, = unpack('I', f.read(4))
        self.label = f.read(16).rstrip(NULL) # sampling group label, null terminated
        self.comment = f.read(256).rstrip(NULL).lstrip(NULL)
        self.decimation, = unpack('I', f.read(4)) # aka "Period", wrt 30 kHz sampling rate
        self.sampfreq, = unpack('I', f.read(4)) # Hz
        # date and time corresponding to t=0
        year, month, dow, day, hour, m, s, ms = unpack('HHHHHHHH', f.read(16))
        self.datetime = datetime.datetime(year, month, day, hour, m, s, ms)
        self.nchans, = unpack('I', f.read(4))

        # extended header:
