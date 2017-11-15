"""Load Blackrock Neural Signal Processing System .nsx files. Inherits from dat.File, because
.nsx also stores its waveform data in flat .dat format

Based on file documentation at:

http://support.blackrockmicro.com/KB/View/166838-file-specifications-packet-details-headers-etc
"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek']

import numpy as np
import os
from struct import unpack
import datetime
import json

from .core import NULL, rstripnonascii, intround, write_dat_json
from . import dat # for inheritance
from .stream import NSXStream
from . import probes


class File(dat.File):
    """Open an .nsx file and expose its header fields and data as attribs"""
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
        self.hpstream = NSXStream(self, kind='highpass')
        self.lpstream = NSXStream(self, kind='lowpass')

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

    def _parseFileHeader(self):
        """Parse the file header"""
        self.fileheader = FileHeader()
        self.fileheader.parse(self.f)
        self.fileheader.parse_json(self.f)
        #print('Parsed fileheader')

    def load(self):
        """Load the waveform data. Data are stored in packets. Normally, there is only one
        long contiguous data packet, but if there are pauses during the recording, the
        data is broken up into multiple packets, with a time gap between each one. Need
        to step over all chans, including aux chans, so pass nchanstotal instead of nchans"""
        datapacket = DataPacket(self.f, self.fileheader.nchanstotal)
        if self.f.tell() != self.filesize: # make sure we're at EOF
            raise NotImplementedError("Can't handle pauses in recording yet")
        self.datapacket = datapacket
        self.contiguous = True

    def chan2datarowi(self, chan):
        """Find row in self.datapacket._data corresponding to chan.
        chan can be either an integer id or a string label"""
        allchansrowis, = np.where(chan == self.fileheader.allchans)
        alllabelrowis, = np.where(chan == self.fileheader.alllabels)
        datarowis = np.concatenate([allchansrowis, alllabelrowis])
        if len(datarowis) == 0:
            raise ValueError("Can't find chan %r" % chan)
        elif len(datarowis) > 1:
            raise ValueError("Found multiple occurences of chan %r at data rows %r"
                             % (chan, datarowis))
        return datarowis[0]

    def getchantype(self, chan):
        """Return the type ('ephys' or 'aux') of chan. chan can be either an integer id
        or a string label."""
        chantypes = []
        fh = self.fileheader
        if chan in fh.chans or chan in fh.ephyschanlabels:
            chantypes.append('ephys')
        if chan in fh.auxchans or chan in fh.auxchanlabels:
            chantypes.append('aux')
        if len(chantypes) == 0:
            raise ValueError("Can't find chan %r" % chan)
        elif len(chantypes) > 1:
            raise ValueError("Found mutliple types for chan %r: %r" % (chan, chantypes))
        return chantypes[0]

    def getchanAD(self, chan):
        """Return AD data for a single chan. chan can be either an integer id
        or a string label. To convert to voltage, use the appropriate multiplier
        (AD2uVx for ephys chans, AD2mVx for aux chans)"""
        datarowi = self.chan2datarowi(chan)
        return self.datapacket._data[datarowi]

    def getchanV(self, chan):
        """Return data for a single chan, in volts. chan can be either an integer id
        or a string label"""
        AD = self.getchanAD(chan)
        chantype = self.getchantype(chan)
        if chantype == 'ephys':
            return AD * self.fileheader.AD2uVx / 1000000 # convert uV to V
        elif chantype == 'aux':
            return AD * self.fileheader.AD2mVx / 1000 # convert mV to V
        else:
            raise ValueError('Unknown chantype %r' % chantype)


class FileHeader(dat.FileHeader):
    """.nsx file header. Takes an open file, parses in from current file
    pointer position, stores header fields as attribs"""
    def __len__(self):
        return self.nbytes

    def parse(self, f):
        # "basic" header:
        self.offset = f.tell()
        self.filetype = f.read(8).decode()
        assert self.filetype == 'NEURALCD'
        self.version = unpack('BB', f.read(2)) # aka "File Spec", major and minor versions
        self.nbytes, = unpack('I', f.read(4)) # length of full header, in bytes
        self.label = f.read(16).rstrip(NULL).decode() # sampling group label, null terminated
        # null terminated, trailing junk bytes (bug):
        self.comment = rstripnonascii(f.read(256)).decode()
        # "Period" wrt sampling freq; sampling freq in Hz:
        self.decimation, self.sampfreq = unpack('II', f.read(8))
        if self.decimation != 1: # doesn't have to be, but probably should for neural data
            print('WARNING: data is decimated by a factor of %d' % self.decimation)
        self.tres = self.decimation / self.sampfreq * 1e6 # float us
        #print('FileHeader.tres = %f' % self.tres)

        # date and time corresponding to t=0:
        year, month, dow, day, hour, m, s, ms = unpack('HHHHHHHH', f.read(16))
        self.datetime = datetime.datetime(year, month, day, hour, m, s, ms)
        self.nchanstotal, = unpack('I', f.read(4)) # ephys and aux chans

        # "extended" headers, each one describing a channel. Use the channel label
        # to distinguish ephys chans from auxiliary channels. Note that seeking through
        # the DataPacket won't work if ephys and aux channels are intermingled. The current
        # assumption is that all ephys chans come before any aux chans:
        self.chanheaders = {} # for ephys signals
        self.auxchanheaders = {} # for auxiliary signals, such as opto/LED signals
        for chani in range(self.nchanstotal):
            chanheader = ChanHeader()
            chanheader.parse(f)
            label, id = chanheader.label, chanheader.id
            if label != ('chan%d' % id):
                print('Treating chan%d (%r) as auxiliary channel' % (id, label))
                self.auxchanheaders[id] = chanheader
            else: # save ephys channel
                self.chanheaders[id] = chanheader
        self.nchans = len(self.chanheaders) # number of ephys chans
        self.nauxchans = len(self.auxchanheaders) # number of aux chans
        assert self.nchans + self.nauxchans == self.nchanstotal
        if self.nauxchans > 0: # some chans were aux chans
            print('Found %d auxiliary channels' % (self.nauxchans))
        assert len(self) == f.tell() # header should be of expected length

        # if there's no adapter, AD ephys chans == probe chans:
        self.chans = np.asarray(sorted(self.chanheaders)) # sorted array of keys
        self.auxchans = np.asarray(sorted(self.auxchanheaders)) # sorted array of keys
        if len(self.chans) > 0 and len(self.auxchans) > 0:
            # ensure that the last ephys chan comes before the first aux chan:
            assert self.chans[-1] < self.auxchans[0]

        # check AD2uV params of all ephys and aux chans:
        for chantype, chanheaders in (('ephys', self.chanheaders),
                                      ('aux', self.auxchanheaders)):
            chans = {'ephys': self.chans, 'aux': self.auxchans}[chantype]
            # all ephys should be in uV, all aux in mV:
            units = {'ephys': 'uV', 'aux': 'mV'}[chantype]
            try:
                c0 = chanheaders[chans[0]] # ref channel for comparing AD2V params
            except IndexError:
                continue # no channels of this type (ephys or aux)
            assert c0.units == units # assumed later during AD2V conversion
            assert c0.maxaval == abs(c0.minaval) # not strictly necessary, but check anyway
            assert c0.maxdval == abs(c0.mindval)
            ref = c0.units, c0.maxaval, c0.minaval, c0.maxdval, c0.mindval
            for c in chanheaders.values():
                if (c.units, c.maxaval, c.minaval, c.maxdval, c.mindval) != ref:
                    raise ValueError('not all chans have the same AD2V params')
            # calculate AD2uV/AD2mV conversion factor:
            if chantype == 'ephys':
                self.AD2uVx = (c0.maxaval-c0.minaval) / float(c0.maxdval-c0.mindval)
            else: # chantype == 'aux'
                self.AD2mVx = (c0.maxaval-c0.minaval) / float(c0.maxdval-c0.mindval)


    def parse_json(self, f):
        """Parse potential .nsx.json file for probe name and optional adapter name"""
        fname = os.path.realpath(f.name) # make sure we have the full fname with path
        path = os.path.dirname(fname)
        ext = os.path.splitext(fname)[1] # e.g., '.ns6'
        # check if there is a file named exactly fname.json:
        jsonfname = fname + '.json'
        jsonbasefname = os.path.split(jsonfname)[-1]
        print('Checking for metadata file %r' % jsonbasefname)
        if os.path.exists(jsonfname):
            print('Found metadata file %r' % jsonbasefname)
        else:
            jsonext = '%s.json' % ext # e.g. '.ns6.json'
            print('No file named %r, checking for a single %s file of any name'
                  % (jsonbasefname, jsonext))
            jsonbasefnames = [ fname for fname in os.listdir(path) if fname.endswith(jsonext)
                               and not fname.startswith('.') ]
            njsonfiles = len(jsonbasefnames)
            if njsonfiles == 1:
                jsonbasefname = jsonbasefnames[0]
                jsonfname = os.path.join(path, jsonbasefname) # full fname with path
                print('Using metadata file %r' % jsonbasefname)
            else:
                jsonfname = None
                print('Found %d %s files, ignoring them' % (njsonfiles, jsonext))

        # get probe name and optional adapter name:
        if jsonfname:
            with open(jsonfname, 'r') as jf:
                j = json.load(jf) # should return a dict of key:val pairs
            assert type(j) == dict
            # check field validity:
            validkeys = ['chan_layout_name', # old name
                         'probe_name', # new name
                         'adapter_name']
            keys = list(j)
            for key in keys:
                if key not in validkeys:
                    raise ValueError("Found invalid field %r in %r\n"
                                     "Fields currently allowed in .nsx.json files: %r"
                                     % (key, jsonfname, validkeys))
            try:
                self.probename = j['probe_name'] # new name
            except KeyError:
                self.probename = j['chan_layout_name'] # old name
            # make sure probename is valid probe.name or probe.layout,
            # potentially rename any old probe names to new ones:
            probe = probes.getprobe(self.probename)
            self.probename = probe.name
            self.adaptername = j.get('adapter_name')
        else: # no .json file, maybe the .nsx comment specifies the probe type?
            self.probename = self.comment.replace(' ', '_')
            if self.probename != '':
                print('Using %r in .nsx comment as probe name' % self.probename)
            else:
                self.probename = probes.DEFNSXPROBETYPE # A1x32
                print('WARNING: assuming probe %s was used in this recording' % self.probename)
            self.adaptername = None

        # initialize probe and adapter:
        self.set_probe()
        self.set_adapter()
        self.check_probe()
        self.check_adapter()

    def get_ephyschanlabels(self):
        return np.asarray([ self.chanheaders[chan].label for chan in self.chans ])

    ephyschanlabels = property(get_ephyschanlabels)

    def get_auxchanlabels(self):
        return np.asarray([ self.auxchanheaders[chan].label for chan in self.auxchans ])

    auxchanlabels = property(get_auxchanlabels)

    def get_alllabels(self):
        return np.concatenate([self.ephyschanlabels, self.auxchanlabels])

    alllabels = property(get_alllabels)


class ChanHeader(object):
    """.nsx header information for a single channel"""
    def parse(self, f):
        self.type = f.read(2).decode()
        assert self.type == 'CC' # for "continuous channel"
        self.id, = unpack('H', f.read(2)) # AD channel, usually == probe channel if no adapter
        self.label = f.read(16).rstrip(NULL).decode()
        self.connector, self.pin = unpack('BB', f.read(2)) # physical connector and pin
        # max and min digital and analog values:
        self.mindval, self.maxdval, self.minaval, self.maxaval = unpack('hhhh', f.read(8))
        self.units = f.read(16).rstrip(NULL).decode() # analog value units: "mV" or "uV"
        # high and low pass hardware filter settings? Blackrock docs are a bit vague:
        # corner freq (mHz); filt order (0=None); filter type (0=None, 1=Butterworth)
        self.hpcorner, self.hporder, self.hpfilttype = unpack("IIH", f.read(10))
        self.lpcorner, self.lporder, self.lpfilttype = unpack("IIH", f.read(10))


class DataPacket(object):
    """.nsx data packet"""
    def __init__(self, f, nchans):
        self.offset = f.tell()
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
