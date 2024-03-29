"""Load Blackrock Neural Signal Processing System .nsx files. Inherits from dat.File, because
.nsx also stores its waveform data in flat .dat format

Based on file documentation at:

http://support.blackrockmicro.com/KB/View/166838-file-specifications-packet-details-headers-etc
"""

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek']


import os
from struct import unpack
import datetime
import json

import numpy as np
import matplotlib.pyplot as plt

from .core import NULL, rstripnonascii
from . import dat # for inheritance
from . import probes
from .probes import TEMPLATEJSONFNAMES
from .stream import NSXStream


class File(dat.File):
    """Open an .nsx file and expose its header fields and data as attribs"""

    def _bind_streams(self):
        self.hpstream = NSXStream(self, kind='highpass')
        self.lpstream = NSXStream(self, kind='lowpass')

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
        if type(chan) in [int, np.int_]:
            datarowis, = np.where(chan == self.fileheader.allchans)
        elif type(chan) in [str, np.str_]:
            datarowis, = np.where(chan == self.fileheader.alllabels)
        else:
            raise ValueError("Unexpected type %s for chan %r" % (type(chan), chan))
        if len(datarowis) == 0:
            raise ValueError("Can't find chan %r" % chan)
        elif len(datarowis) > 1:
            raise ValueError("Found multiple occurences of chan %r at data rows %s"
                             % (chan, datarowis))
        return datarowis[0]

    def get_chantype(self, chan):
        """Return the type ('ephys' or 'aux') of chan. chan can be either an integer id
        or a string label."""
        chantypes = []
        fh = self.fileheader
        if chan in list(fh.chans) or chan in list(fh.ephyschanlabels):
            chantypes.append('ephys')
        if chan in list(fh.auxchans) or chan in list(fh.auxchanlabels):
            chantypes.append('aux')
        if len(chantypes) == 0:
            raise ValueError("Can't find chan %r" % chan)
        elif len(chantypes) > 1:
            raise ValueError("Found multiple types for chan %r: %r" % (chan, chantypes))
        return chantypes[0]

    def get_chanAD(self, chan):
        """Return AD data for a single chan. chan can be either an integer id
        or a string label. To convert to voltage, use the appropriate multiplier
        (AD2uVx for ephys chans, AD2mVx for aux chans)"""
        datarowi = self.chan2datarowi(chan)
        return self.datapacket._data[datarowi]

    def get_chanV(self, chan):
        """Return data for a single chan, in volts. chan can be either an integer id
        or a string label"""
        AD = self.get_chanAD(chan)
        chantype = self.get_chantype(chan)
        if chantype == 'ephys':
            return AD * self.fileheader.AD2uVx / 1000000 # convert uV to V
        elif chantype == 'aux':
            return AD * self.fileheader.AD2mVx / 1000 # convert mV to V
        else:
            raise ValueError('Unknown chantype %r' % chantype)

    def plot_chanV(self, chan, trange=None, figsize=(16, 10), maximize=True, fmt='.-',
                   linewidth=1, markersize=3, color='k', alpha=0.2):
        """Plot chan voltage as a function of time. chan can be either an integer id
        or a string label. trange is window time range, in sec"""
        f, a = plt.subplots(figsize=figsize)
        t = self.tsec
        V = self.get_chanV(chan)
        if trange:
            assert len(trange) == 2
            t0i, t1i = t.searchsorted(trange)
            t = t[t0i:t1i]
            V = V[t0i:t1i]
        a.plot(t, V, fmt, linewidth=linewidth, markersize=markersize,
               color=color, alpha=alpha)
        a.set_xlabel('Time (s)')
        a.set_ylabel('Voltage (V)')
        f.canvas.set_window_title(self.fname)
        if maximize:
            win = f.canvas.window() # possibly a Qt window
            try:
                win.showMaximized() # Qt specific method
            except AttributeError:
                pass # not using Qt backend?
        return a


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
                #print('Treating chan%d (%r) as auxiliary channel' % (id, label))
                self.auxchanheaders[id] = chanheader
            else: # save ephys channel
                self.chanheaders[id] = chanheader
        self.nchans = len(self.chanheaders) # number of ephys chans
        self.nauxchans = len(self.auxchanheaders) # number of aux chans
        assert self.nchans + self.nauxchans == self.nchanstotal
        #if self.nauxchans > 0: # some chans were aux chans
        #    print('Found %d auxiliary channels' % (self.nauxchans))
        assert len(self) == f.tell() # header should be of expected length

        # if there's no adapter, AD ephys chans == probe chans:
        self.chans = np.int64(sorted(self.chanheaders)) # sorted array of keys
        self.auxchans = np.int64(sorted(self.auxchanheaders)) # sorted array of keys
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
                    raise ValueError('Not all chans have the same AD2V params')
            # calculate AD2uV/AD2mV conversion factor:
            if chantype == 'ephys':
                self.AD2uVx = (c0.maxaval-c0.minaval) / float(c0.maxdval-c0.mindval)
            else: # chantype == 'aux'
                self.AD2mVx = (c0.maxaval-c0.minaval) / float(c0.maxdval-c0.mindval)

    def parse_json(self, f):
        """Parse potential .nsx.json file for probe name and optional adapter name"""
        fname = os.path.realpath(f.name) # make sure we have the full fname with path
        path, basefname = os.path.split(fname)
        base, ext = os.path.splitext(basefname) # ext is e.g. '.ns2', '.ns3', '.ns6'...
        jsonbasefname = base + '.ns6' + '.json'
        # Check if there is a file named exactly base.ns6.json, since we only
        # ever copy over *.ns6.json files, though they apply equally well to .ns2 and .ns3,
        # because they only ever indicate the probe and adapter names, which are identical
        # for all nsx files with the same base name:
        exactjsonfname = os.path.join(path, jsonbasefname) # full fname with path
        if os.path.exists(exactjsonfname):
            jsonfname = exactjsonfname # full fname with path
        else: # choose first entry in TEMPLATEJSONFNAMES to match a .ns6.json file in path
            #print('No file named %r, checking for a generic .ns6.json template file'
            #      % jsonbasefname)
            jsonfname = None
            jsonext = 'ns6.json'
            jsonbasefnames = [ fname for fname in os.listdir(path) if fname.endswith(jsonext)
                               and not fname.startswith('.') ]
            for templatejsonfname in TEMPLATEJSONFNAMES:
                if templatejsonfname in jsonbasefnames:
                    jsonbasefname = templatejsonfname # found a match
                    jsonfname = os.path.join(path, jsonbasefname) # full fname with path
                    #print('Found metadata file %r' % jsonbasefname)
                    break

            if jsonfname == None:
                errmsg = (
                    "%s:\n"
                    "Couldn't find a .ns6.json file whose name exactly matches %r\n"
                    "Couldn't find a generic file that matches any of these template names:\n"
                    "%r\n"
                    "*** ERROR: unknown probe type"
                    % (fname, exactjsonfname, TEMPLATEJSONFNAMES)
                    )
                raise RuntimeError(errmsg)
                '''
                # This was always just a fantasy:
                # maybe the comment field in the .nsx file specifies the probe type?
                self.probename = self.comment.replace(' ', '_')
                if self.probename != '':
                    print('Using %r in .nsx comment as probe name' % self.probename)
                else:
                    raise RuntimeError('No probe type specified for recording %s' % fname)
                self.adaptername = None
                '''

        # get probe name and optional adapter name:
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

        # initialize probe and adapter:
        self.set_probe()
        self.set_adapter()
        self.check_probe()
        self.check_adapter()

        print('%s: json=%r, probe=%s, adapter=%s'
              % (fname, jsonbasefname, self.probename, self.adaptername))

    def get_ephyschanlabels(self):
        return np.array([ self.chanheaders[chan].label for chan in self.chans ], dtype=str)

    ephyschanlabels = property(get_ephyschanlabels)

    def get_auxchanlabels(self):
        return np.array([ self.auxchanheaders[chan].label for chan in self.auxchans ], dtype=str)

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
        filesize = os.stat(f.name).st_size # in bytes
        datasize = filesize - self.dataoffset
        expectednt = datasize / 2 / self.nchans
        if self.nt != int(expectednt):
            print("*** WARNING: Header in file %s says nt=%d, "
                  "but file size suggests nt=%.3f. Perhaps the nt header field "
                  "is incorrect/corrupt, but the rest of the file is fine."
                  % (f.name, self.nt, expectednt))
            if expectednt % 1 != 0.0: # remaining file size doesn't divide evenly
                print("*** WARNING: Expected nt=%.3f is non-integer. "
                      "This means there was an unknown amount of actual data loss, "
                      "which could have occurred any time during the recording, including "
                      "somewhere in the middle which would create offset errors with respect "
                      "to e.g. stimulus data. Use extreme caution!" % expectednt)
            response = input("Try and continue anyway? (y/[n]) >> ")
            if response != 'y':
                raise RuntimeError('Stopping')
            else:
                self.nt = int(np.floor(expectednt)) # round down
                ndroppedbytes = datasize - 2*self.nchans*self.nt
                print('Using only the first %d timepoints, discarding last %d bytes'
                      % (self.nt, ndroppedbytes))
        # load all data into memory using np.fromfile. Time is MSB, chan is LSB:
        #self._data = np.fromfile(f, dtype=np.int16, count=self.nt*nchans)
        #self._data.shape = -1, self.nchans # reshape, t in rows, chans in columns
        #self._data = self._data.T # reshape, chans in columns, t in rows

        # load data on demand using np.memmap, numpy always assumes binary mode.
        # Time is the outer loop, chan is the inner loop, so load in column-major (Fortran)
        # order to get contiguous (chani, ti) array:
        self._data = np.memmap(f, dtype=np.int16, mode='r', offset=self.dataoffset,
                               shape=(self.nchans, self.nt), order='F')
