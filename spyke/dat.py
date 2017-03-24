"""Load flat .dat files with accompanying metadata. Inherits from nsx.File, which
also stores its waveform data in flat .dat format"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek']

import numpy as np
import os
import datetime
import json

from stream import DatStream
import nsx # for inheritance
import probes


class File(nsx.File):
    """Open a .dat file"""
    def __init__(self, fname, path):
        self.fname = fname
        self.path = path
        self.filesize = os.stat(self.join(fname))[6] # in bytes
        self.open() # calls parse() and load()

        self.datapacketoffset = self.datapacket.offset # save for unpickling
        self.t0i, self.nt = self.datapacket.t0i, self.datapacket.nt # copy for convenience
        self.t1i = self.t0i + self.nt - 1
        self.t0 = self.t0i * self.fileheader.tres # us
        self.t1 = self.t1i * self.fileheader.tres # us
        self.hpstream = DatStream(self, kind='highpass')
        self.lpstream = DatStream(self, kind='lowpass')

    def open(self):
        """(Re)open previously closed file"""
        # the 'b' for binary is only necessary for MS Windows:
        self.f = open(self.join(self.fname), 'rb')
        # parse file and load datapacket here instead of in __init__, because during
        # multiprocess detection, __init__ isn't called on unpickle, but open() is:
        try:
            self.f.seek(self.datapacketoffset) # skip over FileHeader
        except AttributeError: # hasn't been parsed before, self.datapacketoffset is missing
            self.parse()
        self.load()

    def parse(self):
        self._parseFileHeader()

    def _parseFileHeader(self):
        """Parse the file header"""
        self.fileheader = FileHeader()
        self.fileheader.parse(self.join(self.fname))
        #print('Parsed fileheader')

    def load(self):
        """Load the waveform data. Treat the whole .dat file the same way nsx.File treats a
        single nsx "packet". Need to step over all chans, including aux chans, so pass
        nchanstotal instead of nchans"""
        assert self.filesize % 2 == 0 # 2 bytes per sample, make sure it divides evenly
        nsamples = int(self.filesize / 2) # total number of samples in file
        nchanstotal = self.fileheader.nchanstotal
        if nsamples % nchanstotal != 0:
            raise ValueError('%d samples in self.fname is not an integer multiple of %d total '
                             'channels specified in .json file' % (nsamples, nchanstotal))
        nt = int(nsamples / nchanstotal) # total number of timepoints in file

        datapacket = DataPacket(self.f, self.fileheader.nchanstotal, nt)
        assert self.f.tell() == self.filesize # make sure we're at EOF
        self.datapacket = datapacket
        self.contiguous = True

    def export_dat(self, dt=None):
        """Irrelevant feature"""
        pass


class FileHeader(object):
    """.dat 'file header', derived from metadata file with same basename"""

    def parse(self, datfname):
        self.parse_json(datfname)

    def parse_json(self, datfname):
        """Parse metadata from .dat.json file"""
        metafname = datfname + '.json' # assume that .dat meta files are named *.dat.json
        with open(metafname, 'rb') as mf:
            j = json.load(mf) # should return a dict of key:val pairs
        assert type(j) == dict

        self.nchanstotal = j['n_chans'] # ephys + aux chans
        self.sampfreq = j['sample_rate'] # Hz
        self.tres = 1 / self.sampfreq * 1e6 # float us
        self.dtype = j['dtype']
        if self.dtype != 'int16':
            raise ValueError('only int16 sample data type is supported, got %r' % dtype)
        self.AD2uVx = j['uV_per_AD']
        self.probename = j['chan_layout_name']
        probe = probes.getprobe(self.probename) # make sure probename is recognized
        chan0 = probe.chan0 # 0-based or 1-based

        # the rest are optional fields:
        self.chans = j.get('chans') # ephys chan IDs, 0-based, ordered by row in .dat file
        self.auxchans = j.get('aux_chans') # aux chan IDs, 0-based, order by row in .dat file
        if self.chans:
            self.chans = np.asarray(self.chans) # convert list to array
            self.nchans = len(self.chans) # number of ephys chans
        else: # chans == [] or None
            assert not auxchans # make sure auxchans aren't specified
            self.chans = np.arange(chan0, chan0+self.nchans)
            self.nchans = self.nchanstotal
        # for simplicity, require that all ephys chans are included, and are always sorted:
        assert (self.chans == np.arange(chan0, chan0+self.nchans)).all()
        if self.auxchans:
            self.auxchans = np.asarray(self.auxchans) # convert list to array
            self.nauxchans = len(self.auxchans) # number of aux chans
            # all auxiliary channels must follow all ephys channels:
            assert max(self.chans) < min(self.auxchans)
        else: # auxchans == [] or None
            self.auxchans = np.array([])
            self.nauxchans = 0
        assert self.nchans + self.nauxchans == self.nchanstotal
        self.datetimestr = j.get('datetime') # ISO time string of first sample point
        if self.datetimestr:
            self.datetime = datetime.datetime.strptime(self.datetimestr, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            self.datetime = None
        self.author = j.get('author')
        self.version = j.get('version')
        self.notes = j.get('notes')

    comment = property(lambda self: '') # for parent NSXStream


class DataPacket(object):
    """.dat data packet"""
    
    def __init__(self, f, nchans, nt):
        self.offset = f.tell()
        self.nchans, self.nt = nchans, nt
        # nsamples offset of first timepoint from t=0; number of timepoints:
        ## TODO: this should probably come from the .json file!!! Not sure if this is
        ## actually 0 for open-ephys:
        self.t0i = 0
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
