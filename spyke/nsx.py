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
from collections import OrderedDict as odict
import json

from core import NULL, rstripnonascii, intround
import dat # for inheritance
from stream import NSXStream


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

    def parse(self):
        self._parseFileHeader()

    def _parseFileHeader(self):
        """Parse the file header"""
        self.fileheader = FileHeader()
        self.fileheader.parse(self.f)
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

    def export_dat(self, dt=None):
        """Export contiguous data packet to .dat file, in the original (ti, chani) order
        using same base file name in the same folder. Also export companion .json metadata
        file. dt is duration to export from start of recording, in sec"""
        if dt == None:
            nt = self.nt
            dtstr = ''
        else:
            nt = intround(dt * self.fileheader.sampfreq)
            dtstr = str(dt)
        assert self.is_open()
        fh = self.fileheader
        nbytes = nt * fh.nchanstotal * 2 # number of bytes requested, 2 bytes per datapoint
        offset = self.datapacket.dataoffset
        self.f.seek(offset)
        basefname = os.path.splitext(self.fname)[0]
        if dtstr:
            basefname = '%s_%ss' % (basefname, dtstr)
        datfname = basefname + '.dat'
        jsonfname = datfname + '.json'
        fulldatfname = self.join(datfname)
        fulljsonfname = self.join(jsonfname)

        # export .dat file:
        print('writing raw ephys data to %r' % fulldatfname)
        print('starting from dataoffset at %d bytes' % offset)
        with open(fulldatfname, 'wb') as datf:
            datf.write(self.f.read(nbytes))
        nbyteswritten = self.f.tell() - offset
        print('%d bytes written' % nbyteswritten)
        print('%d attempted, %d actual timepoints written'
              % (nt, nbyteswritten / fh.nchanstotal / 2))

        # export companion .json metadata file:
        od = odict()
        fh = self.fileheader
        od['nchans'] = fh.nchanstotal
        od['sample_rate'] = fh.sampfreq
        od['dtype'] = 'int16' # hard-coded, only dtype supported for now
        od['uV_per_AD'] = fh.AD2uVx
        od['chan_layout_name'] = self.hpstream.probe.name
        od['chans'] = list(fh.chans)
        od['aux_chans'] = list(fh.auxchans)
        od['nsamples_offset'] = self.t0i
        od['datetime'] = fh.datetime.isoformat()
        od['author'] = ''
        od['version'] = '' # no way to extract Blackrock NSP version from .nsx?
        od['notes'] = fh.comment
        with open(fulljsonfname, 'w') as jsonf:
            ## TODO: make list fields not have a newline for each entry
            json.dump(od, jsonf, indent=0) # write contents of odict to file
            jsonf.write('\n') # end with a blank line
        print('wrote metadata file %r' % fulljsonfname)

        # print the important metadata:
        print('total chans: %d' % fh.nchanstotal)
        print('ephys chans: %d' % fh.nchans)
        print('sample rate: %d Hz' % fh.sampfreq)
        print('voltage gain: %g uV/AD' % fh.AD2uVx)
        print('chan layout: %s' % self.hpstream.probe.name)


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
        self.comment = rstripnonascii(f.read(256)) # null terminated, trailing junk bytes (bug)
        # "Period" wrt sampling freq; sampling freq in Hz:
        self.decimation, self.sampfreq = unpack('II', f.read(8))
        assert self.decimation == 1 # doesn't have to be, but probably should for neural data
        self.tres = 1 / self.sampfreq * 1e6 # float us
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
                print('excluding chan%d (%r) as auxiliary channel' % (id, label))
                self.auxchanheaders[id] = chanheader
            else: # save ephys channel
                self.chanheaders[id] = chanheader
        self.nchans = len(self.chanheaders) # number of ephys chans
        self.nauxchans = len(self.auxchanheaders) # number of aux chans
        assert self.nchans + self.nauxchans == self.nchanstotal
        if self.nauxchans > 0: # some chans were aux chans
            print('excluded %d auxiliary channels' % (self.nauxchans))
        assert len(self) == f.tell() # header should be of expected length
        self.chans = np.asarray(sorted(self.chanheaders)) # sorted array of keys
        self.auxchans = np.asarray(sorted(self.auxchanheaders)) # sorted array of keys
        if len(self.auxchans) > 0:
            # ensure that the last ephys chan comes before the first aux chan:
            assert self.chans[-1] < self.auxchans[0]

        # check AD2uV params of all ephys chans:
        c0 = self.chanheaders[self.chans[0]] # reference channel for comparing AD2uV params
        assert c0.units == 'uV' # assumed later during AD2uV conversion
        assert c0.maxaval == abs(c0.minaval) # not strictly necessary, but check anyway
        assert c0.maxdval == abs(c0.mindval)
        ref = c0.units, c0.maxaval, c0.minaval, c0.maxdval, c0.mindval
        for c in self.chanheaders.values():
            if (c.units, c.maxaval, c.minaval, c.maxdval, c.mindval) != ref:
                raise ValueError('not all chans have the same AD2uV params')
        # calculate AD2uV conversion factor:
        self.AD2uVx = (c0.maxaval-c0.minaval) / float(c0.maxdval-c0.mindval)


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
