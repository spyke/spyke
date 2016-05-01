"""Load Blackrock Neural Signal Processing System .nsx files.

Based on file documentation at:

http://support.blackrockmicro.com/KB/View/166838-file-specifications-packet-details-headers-etc
"""

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
        self.filesize = os.stat(self.join(fname))[6] # in bytes
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

    def __len__(self):
        return self.nbytes

    def parse(self, f):
        # "basic" header:
        self.offset = f.tell()
        self.filetype = f.read(8)
        assert self.filetype == 'NEURALCD'
        self.version = unpack('BB', f.read(2)) # aka "File Spec", major and minor versions
        self.nbytes, = unpack('I', f.read(4)) # length of full header, in bytes
        self.label = f.read(16).rstrip(NULL) # sampling group label, null terminated
        self.comment = f.read(256).rstrip(NULL)
        # "Period", wrt 30 kHz sampling freq; sampling freq in Hz:
        self.decimation, self.sampfreq = unpack('II', f.read(8))
        # date and time corresponding to t=0
        year, month, dow, day, hour, m, s, ms = unpack('HHHHHHHH', f.read(16))
        self.datetime = datetime.datetime(year, month, day, hour, m, s, ms)
        self.nchans, = unpack('I', f.read(4))

        # "extended" headers, each one describing a channel
        self.chans = {}
        for chani in range(self.nchans):
            chanheader = ChanHeader()
            chanheader.parse(f)
            self.chans[chanheader.id] = chanheader
        assert len(self.chans) == self.nchans # make sure each chan ID is unique
        assert len(self) == f.tell() # header should be of expected length


class ChanHeader(object):
    """.nsx header information for a single channel"""

    def parse(self, f):
        self.type = f.read(2)
        assert self.type == 'CC' # for "continuous channel"
        self.id, = unpack('H', f.read(2)) # aka "electrode ID"
        self.label = f.read(16).rstrip(NULL)
        self.connector, self.pin = unpack('BB', f.read(2)) # physical connector and pin
        # max and min digital and analog values:
        self.mindval, self.maxdval, self.minaval, self.maxaval = unpack('hhhh', f.read(8))
        self.units = f.read(16).rstrip(NULL) # analog value units: "mV" or "uV"
        ## TODO: given the corner freq values, perhaps the order of hi and lo fields
        ## is reversed in the Blackrock documentation?
        # hi corner freq: mHz; hi filt order: 0=None; hi filt type: 0=None, 1=Butterworth
        self.hfcorner, self.hforder, self.hfilttype = unpack("IIH", f.read(10))
        # lo corner freq: mHz; lo filt order: 0=None; lo filt type: 0=None, 1=Butterworth
        self.lfcorner, self.lforder, self.lfilttype = unpack("IIH", f.read(10))

