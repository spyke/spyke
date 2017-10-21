"""Load flat .dat files with accompanying metadata. See notes/dat_metadata*.json for
example metadata files"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek']

import numpy as np
import os
import datetime
import json

from . import probes
from .stream import DATStream


class File(object):
    """Open a .dat file"""
    def __init__(self, fname, path=None):
        if path is None:
            path, fname = os.path.split(fname) # separate path from fname
        self.fname, self.path = fname, path
        self.filesize = os.stat(self.join(fname))[6] # in bytes
        self.open() # calls parse() and load()

        self.datapacketoffset = self.datapacket.offset # save for unpickling
        self.t0i, self.nt = self.datapacket.t0i, self.datapacket.nt # copy for convenience
        self.t1i = self.t0i + self.nt - 1
        self.t0 = self.t0i * self.fileheader.tres # us
        self.t1 = self.t1i * self.fileheader.tres # us
        self.hpstream = DATStream(self, kind='highpass')
        self.lpstream = DATStream(self, kind='lowpass')

    def join(self, fname):
        """Return fname joined to self.path"""
        return os.path.abspath(os.path.expanduser(os.path.join(self.path, fname)))

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

    def close(self):
        """Close the file, don't do anything if already closed"""
        if self.is_open():
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
        fh = self.fileheader
        assert self.filesize % 2 == 0 # 2 bytes per sample, make sure it divides evenly
        nsamples = int(self.filesize / 2) # total number of samples in file
        nchanstotal, t0i = fh.nchanstotal, fh.t0i
        if nsamples % nchanstotal != 0:
            raise ValueError('%d total samples in self.fname is not an integer multiple of '
                             '%d total channels specified in .json file'
                             % (nsamples, nchanstotal))
        nt = int(nsamples / nchanstotal) # total number of timepoints in file

        datapacket = DataPacket(self.f, nchanstotal, nt, t0i)
        assert self.f.tell() == self.filesize # make sure we're at EOF
        self.datapacket = datapacket
        self.contiguous = True

    def get_tsec(self):
        """Return all timestamps for data in self, in sec"""
        start, stop, step = self.t0, self.t1, self.fileheader.tres
        ts = np.arange(start, stop+step, step) # timestamps in us
        assert len(ts) == self.data.shape[1]
        return ts / 1000000 # timestamps in sec

    tsec = property(get_tsec)

    def get_data(self):
        """Return all ephys data"""
        try:
            _data = self.datapacket._data
        except AttributeError:
            raise RuntimeError('Waveform data not available, file is closed/mmap deleted?')
        return _data[:self.fileheader.nchans] # only ephys data

    data = property(get_data)

    def get_auxdata(self):
        """Return all auxiliary data"""
        try:
            _data = self.datapacket._data
        except AttributeError:
            raise RuntimeError('Waveform data not available, file is closed/mmap deleted?')
        return _data[self.fileheader.nchans:] # only aux data

    auxdata = property(get_auxdata)

    def __getstate__(self):
        """Don't pickle open file handle or datapacket with open mmap"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        try: del d['f'] # exclude open file handle, if any
        except KeyError: pass
        try: del d['datapacket'] # avoid pickling datapacket._data mmap
        except KeyError: pass
        return d
    '''
    def export_raw_dat(self, path, dt=None):
        """Redundant feature"""
        raise RuntimeError("export_raw_dat() is redundant for a .dat file")
    '''

class FileHeader(object):
    """.dat 'file header', derived from metadata file with same basename"""
    def parse(self, datfname):
        self.parse_json(datfname)

    def parse_json(self, datfname):
        """Parse metadata from .dat.json file"""
        jsonfname = datfname + '.json' # assume that .dat meta files are named *.dat.json
        print('Parsing file %r' % jsonfname)
        with open(jsonfname, 'r') as jf:
            j = json.load(jf) # should return a dict of key:val pairs
        assert type(j) == dict

        # required fields:
        self.nchanstotal = j['nchans'] # ephys + aux chans
        self.sampfreq = j['sample_rate'] # Hz
        self.tres = 1 / self.sampfreq * 1e6 # float us
        self.dtype = j['dtype']
        if self.dtype != 'int16':
            raise ValueError('only int16 sample data type is supported, got %r' % dtype)
        self.AD2uVx = j['uV_per_AD']
        try:
            self.probename = j['probe_name'] # new name
        except KeyError:
            self.probename = j['chan_layout_name'] # old name
        self.set_probe() # initialize probe

        # optional fields:
        self.adaptername = j.get('adapter_name')
        self.set_adapter() # initialize adapter
        chan0 = self.probe.chan0 # base of channel indices: either 0-based or 1-based
        # ephys channel indices, ordered by row in the .dat file, can be 0- or 1-based,
        # depending on how they're defined for the above self.probename:
        self.chans = j.get('chans')
        # auxiliary channel indices, e.g. optogenetic/LED channels:
        self.auxchans = j.get('aux_chans')
        if self.chans:
            self.chans = np.asarray(self.chans) # convert list to array
            self.nchans = len(self.chans) # number of ephys chans
        else: # chans == [] or None
            assert not auxchans # make sure auxchans aren't specified
            self.nchans = self.nchanstotal
            self.chans = np.arange(chan0, chan0+self.nchans)
        if self.auxchans:
            self.auxchans = np.asarray(self.auxchans) # convert list to array
            self.nauxchans = len(self.auxchans) # number of aux chans
            # all auxiliary channels must follow all ephys channels:
            assert max(self.chans) < min(self.auxchans)
        else: # auxchans == [] or None
            self.auxchans = np.array([])
            self.nauxchans = 0
        assert self.nchans + self.nauxchans == self.nchanstotal

        # now that all the chan fields are done, check probe and adapter:
        self.check_probe()
        self.check_adapter()

        # offset of first timepoint wrt t=0, in number of samples:
        self.t0i = j.get('nsamples_offset', 0)
        assert type(self.t0i) == int
        self.datetimestr = j.get('datetime') # ISO 8601 datetime wrt t=0, local time, no TZ
        if self.datetimestr:
            self.datetime = datetime.datetime.strptime(self.datetimestr, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            self.datetime = None
        self.author = j.get('author') # software that generated the .dat file
        self.version = j.get('version') # version of software that generated the .dat file
        self.notes = j.get('notes') # notes

    def get_allchans(self):
        return np.concatenate([self.chans, self.auxchans])

    allchans = property(get_allchans)

    def set_probe(self):
        """Set probe from self.probename"""
        print('Setting probe type: %r' % self.probename)
        self.probe = probes.getprobe(self.probename) # check for valid probe name

    def set_adapter(self):
        """Set adapter from self.adaptername, if any"""
        self.adapter = None
        if self.adaptername:
            print('Setting adapter type: %r' % self.adaptername)
            self.adapter = probes.getadapter(self.adaptername) # checks for valid name

    def check_probe(self):
        if not np.isin(self.chans, self.probe.chans).all():
            raise ValueError("Data chans are not a subset of probe chans. Wrong probe "
                             "specified in .json file?\n"
                             "Data chans:\n%s\n"
                             "Probe %r chans:\n%s"
                             % (self.chans, self.probename, self.probe.chans))

    def check_adapter(self):
        if not self.adapter:
            return
        if not np.isin(self.chans, self.adapter.ADchans).all():
            raise ValueError("Data chans are not a subset of adapter ADchans. Wrong adapter "
                             "specified in .json file?\n"
                             "Data chans:\n%s\n"
                             "Adapter %r ADchans:\n%s"
                             % (self.chans, self.adaptername, self.adapter.ADchans))
        # if an adapter is present, self.chans actually represent ADchans,
        # which are not necessarily equal to probe chans. Since the convention throughout
        # spyke is that self.chans represent probe chans, replace the ADchans in self.chans
        # to be the equivalent probe chans:
        self.chans = self.adapter.probechans[self.adapter.ADchansortis]


class DataPacket(object):
    """.dat data packet"""
    def __init__(self, f, nchans, nt, t0i):
        self.offset = f.tell()
        self.nchans, self.nt, self.t0i = nchans, nt, t0i
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
