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

from core import NULL, rstripnonascii
from stream import Stream


class File(object):
    """Open an .nsx file and expose its header fields and data as attribs"""
    def __init__(self, fname, path):
        self.fname = fname
        self.path = path
        self.filesize = os.stat(self.join(fname))[6] # in bytes
        self.open()
        self._parseFileHeader()
        self.load()

    def join(self, fname):
        return os.path.join(self.path, fname)

    def open(self):
        """(Re)open previously closed .nsx file"""
        # the 'b' for binary is only necessary for MS Windows:
        self.f = open(self.join(self.fname), 'rb')

    def close(self):
        """Close the .nsx file"""
        # the only way to close a np.memmap is to close its underlying mmap and make sure
        # there aren't any remaining handles to it
        self.datapacket._data._mmap.close()
        del self.datapacket._data
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
        ## TODO: does datapacket need its own __getstate__ and remove _data?
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        if 'f' in d:
            del d['f'] # exclude open .nsx file handle, if any
        return d

    def _parseFileHeader(self):
        """Parse the .nsx file header"""
        self.fileheader = FileHeader()
        self.fileheader.parse(self.f)
        #print('Parsed fileheader')

    def parse(self):
        pass # nothing to do here, everything happens in __init__

    def load(self):
        """Load the waveform data. Data are stored in packets. Normally, there is only one
        long contiguous data packet, but if there are pauses during the recording, the
        data is broken up into multiple packets, with a time gap between each one"""
        datapacket = DataPacket(self.f, self.fileheader.nchans)
        if self.f.tell() != self.filesize: # make sure we're at EOF
            raise NotImplementedError("Can't handle pauses in recording yet")
        self.datapacket = datapacket
        self.t0i, self.nt = datapacket.t0i, datapacket.nt # copy for convenience

    def get_data(self):
        try:
            return self.datapacket._data
        except AttributeError:
            raise RuntimeError('waveform data not available, file is closed/mmap deleted?')

    data = property(get_data)


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
        self.comment = rstripnonascii(f.read(256)) # null terminated, junk bytes if empty (bug)
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
        # high and low pass hardware filter settings? Blackrock docs are a bit vague:
        # corner freq (mHz); filt order (0=None); filter type (0=None, 1=Butterworth)
        self.hpcorner, self.hporder, self.hpfilttype = unpack("IIH", f.read(10))
        self.lpcorner, self.lporder, self.lpfilttype = unpack("IIH", f.read(10))


class DataPacket(object):
    """.nsx data packet"""
    
    def __init__(self, f, nchans):
        self.nchans = nchans
        header, = unpack('B', f.read(1))
        assert header == 1
        # nsamples offset of first timepoint from t=0; number of timepoints:
        self.t0i, self.nt = unpack('II', f.read(8))
        self.dataoffset = f.tell()

        # load all data into memory using np.fromfile. Time is MSB, chan is LSB:
        #self._data = np.fromfile(f, dtype=np.int16, count=self.nt*nchans)
        #self._data.shape = -1, self.nchans # reshape, t in rows, chans in columns
        #self._data = self._data.T # reshape, chans in columns, t in rows

        # load data on demand using np.memmap, numpy always assumes binary mode.
        # Time is the outer loop, chan is the inner loop, so load in column-major (Fortran)
        # order to get contiguous (chani, ti) array:
        self._data = np.memmap(f, dtype=np.int16, mode='r', offset=self.dataoffset,
                               shape=(self.nchans, self.nt), order='F')
