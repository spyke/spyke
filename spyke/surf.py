"""Handles parsing of Surf-generated .srf files.
Some field names and comments are copied from Tim Blanche's Delphi program "SurfBawd".
"""

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
import numpy as np
import os
import cPickle
import cProfile
from struct import Struct, unpack
import unittest
from copy import copy
import re
import time
import weakref

import wx

from spyke.core import Stream, iterable, toiter
from spyke.filelock import FileLock

NULL = '\x00'

DEFNHIGHPASSRECORDS = 50000
DEFNLOWPASSRECORDS = 300000
DEFNDIGITALSVALRECORDS = 600000
CTSRECORDDTYPE = [('TimeStamp', '<i8'), ('Probe', '<i2'), ('NumSamples', '<i4'), ('dataoffset', '<i8')]
DIGITALSVALDTYPE = [('TimeStamp', np.int64), ('SVal', np.uint16)]

# create a couple of Struct objects with compiled format strings
# and call them as needed for the most common record types.
# Using np.fromfile to load one dtype record at a time without itermediate string isn't any faster
# Without little endian < symbol, it struct.unpack uses 4 byte native size and alignment
ctsstruct = Struct('qqhhii')
dsvalstruct = Struct('qqhhi')
unpackctsrec = ctsstruct.unpack
unpackdsvalrec = dsvalstruct.unpack


class DRDBError(ValueError):
    """Used to indicate when you've passed the last DRDB at the start of the .srf file"""


class File(object):
    """Open a .srf file and, after parsing, expose all of its headers and
    records as attribs:
        - Surf file header
        - Surf data record descriptor blocks
        - electrode layout records
        - message records
        - high and low pass continuous waveform records
        - stimulus display header records
        - stimulus digital single val records"""
    def __init__(self, fname):
        # TODO: ensure fname is a full path name, so that there won't be issues finding the file if self is ever unpickled
        self.fname = fname
        self.fileSize = os.stat(fname)[6]
        self.filelock = FileLock(fname, timeout=3600, delay=0.01)
        self.open()
        self._parseFileHeader()
        self.parsefname = fname + '.parse'
        # init struct ndarrays for high volume record types
        self.highpassrecords = np.empty(DEFNHIGHPASSRECORDS, dtype=CTSRECORDDTYPE)
        self.lowpassrecords = np.empty(DEFNLOWPASSRECORDS, dtype=CTSRECORDDTYPE)
        self.nhighpassrecords = 0
        self.nlowpassrecords = 0
        self.digitalsvalrecords = np.empty(DEFNDIGITALSVALRECORDS, dtype=DIGITALSVALDTYPE)
        self.ndigitalsvalrecords = 0

    def open(self):
        """(Re)open previously closed .srf file"""
        self.filelock.acquire()
        self.f = open(self.fname, 'rb')

    def close(self):
        """Close the .srf file"""
        self.f.close()
        self.filelock.release()

    def _parseFileHeader(self):
        """Parse the Surf file header"""
        self.fileheader = FileHeader()
        self.fileheader.parse(self.f)
        #print('Parsed fileheader')

    def _parseDRDBS(self):
        """Parse the DRDBs (Data Record Descriptor Blocks)"""
        self.drdbs = []
        f = self.f
        while True:
            drdb = DRDB()
            try:
                drdb.parse(f)
                self.drdbs.append(drdb)
            except DRDBError:
                # we've gone past the last DRDB
                # set file pointer back to where we were
                f.seek(drdb.offset)
                break

    def parse(self, force=False, save=True):
        """Parse the .srf file, potentially unpickling parse info from
        a .parse file. If doing a new parsing, optionally save parse info
        to a .parse file"""
        t0 = time.time()
        try: # recover self pickled in .parse file
            if force: # force a new parsing
                raise IOError # make the try fail, skip to the except block
            self.unpickle()
            print('unpickling took %.3f sec' % (time.time()-t0))
        # parsing is being forced, or .parse file doesn't exist, or something's
        # wrong with it. Parse the .srf file
        except IOError:
            print('Parsing %r' % self.fname)
            self._parseDRDBS()
            #cProfile.runctx('self._parseRecords()', globals(), locals())
            self._parseRecords()
            print('Done parsing %r' % self.fname)
            print('parsing took %.3f sec' % (time.time()-t0))
            self._connectRecords()
            self._verifyParsing()

            if hasattr(self, 'highpassrecords'):
                self.hpstream = Stream(self, kind='highpass') # highpass record (spike) stream
            else:
                self.hpstream = None
            if hasattr(self, 'lowpassmultichanrecords'):
                self.lpstream = Stream(self, kind='lowpass') # lowpassmultichan record (LFP) stream
            else:
                self.lpstream = None

            if save:
                tsave = time.time()
                self.pickle()
                print('pickling took %.3f sec' % (time.time()-tsave))

    def _parseRecords(self):
        """Parse all the records in the file, but don't load any waveforms"""
        # dict of parse methods for most common records that will be stored in structured arrays
        FLAG2METHOD = {'PS' : self.parseHighPassRecord,
                       'PC' : self.parseLowPassRecord,
                       'VD' : self.parseDigitalSValRecord}
        # dict of (record type, listname to store it in) tuples
        FLAG2REC = {'L'  : (LayoutRecord, 'layoutrecords'),
                    'MS' : (SurfMessageRecord, 'messagerecords'),
                    'MU' : (UserMessageRecord, 'messagerecords'),
                    'PE' : (EpochRecord, 'epochrecords'),
                    'D'  : (DisplayRecord, 'displayrecords'),
                    'VA' : (AnalogSValRecord, 'analogsvalrecords')}
        f = self.f
        while True:
            # returns an empty string when EOF is reached
            flag = f.read(2).rstrip(NULL) # TODO: should this strip NULL as defined above?
            if flag == '':
                break
            # put file pointer back to start of flag
            f.seek(-2, 1) # TODO: unnecessary - doesn't this slow down parsing quite a bit?
            if flag in FLAG2METHOD: # these are the most common
                FLAG2METHOD[flag](f) # call the method
            elif flag in FLAG2REC:
                rectype, reclistname = FLAG2REC[flag]
                rec = rectype()
                rec.parse(f)
                #wx.Yield() # allow wx GUI event processing during parsing
                self._appendRecord(rec, reclistname)
            else:
                raise ValueError('Unexpected flag %r at offset %d' % (flag, f.tell()))
            #self.percentParsed = f.tell() / self.fileSize * 100

    def parseContinuousRecord(self, f, r):
        """Parse a continuous record (high or low pass).
        Its length in the file in bytes is 26 + self.NumSamples*2, not including the
        2 byte flag at the beginning"""

        #junk, junk, r['TimeStamp'], r['Probe'], junk, junk, NumSamples = unpack('<hiqhhii', f.read(26))
        #junk, r['TimeStamp'], r['Probe'], junk, junk, NumSamples = unpack('qqhhii', f.read(28))
        junk, r['TimeStamp'], r['Probe'], junk, junk, NumSamples = unpackctsrec(f.read(28))
        r['NumSamples'] = NumSamples
        r['dataoffset'] = f.tell()
        # skip the waveform data for now
        f.seek(NumSamples*2, 1) # each sample is 2 bytes long

    def parseHighPassRecord(self, f):
        try:
            r = self.highpassrecords[self.nhighpassrecords] # gives a np.void with named fields
        except IndexError:
            newsize = len(self.highpassrecords) + DEFNHIGHPASSRECORDS
            self.highpassrecords.resize(newsize, refcheck=False)
            r = self.highpassrecords[self.nhighpassrecords] # gives a np.void with named fields
        self.parseContinuousRecord(f, r) # writes to the np.void, and therefore to the struct array
        self.nhighpassrecords += 1

    def parseLowPassRecord(self, f):
        try:
            r = self.lowpassrecords[self.nlowpassrecords] # gives a np.void with named fields
        except IndexError:
            newsize = len(self.lowpassrecords) + DEFNLOWPASSRECORDS
            self.lowpassrecords.resize(newsize, refcheck=False)
            r = self.lowpassrecords[self.nlowpassrecords] # gives a np.void with named fields
        self.parseContinuousRecord(f, r) # writes to the np.void, and therefore to the struct array
        self.nlowpassrecords += 1

    def parseDigitalSValRecord(self, f):
        """Parse a digital SVal record. Its length in the file in bytes is 22,
        not including the 2 byte flag at the beginning"""
        try:
            r = self.digitalsvalrecords[self.ndigitalsvalrecords] # gives a np.void with named fields
        except IndexError:
            newsize = len(self.digitalsvalrecords) + DEFNDIGITALSVALRECORDS
            self.digitalsvalrecords.resize(newsize, refcheck=False)
            r = self.digitalsvalrecords[self.ndigitalsvalrecords] # gives a np.void with named fields
        #junk, junk, r['TimeStamp'], r['SVal'], junk, junk = unpack('<hiqhhi', f.read(22))
        #junk, r['TimeStamp'], r['SVal'], junk, junk = unpack('qqhhi', f.read(24))
        junk, r['TimeStamp'], r['SVal'], junk, junk = unpackdsvalrec(f.read(24))
        self.ndigitalsvalrecords += 1

    def loadContinuousRecord(self, record):
        """Load waveform data for this continuous record"""
        # TODO: add chans arg to pull out only certain chans, and maybe a ti arg
        # to pull out less than the full set of sample points for this record
        self.f.seek(record['dataoffset'])
        # {ADC Waveform type; dynamic array of SHRT (signed 16 bit)} - converted to an ndarray
        # Using stuct.unpack for this is very slow:
        #self.data = np.asarray(unpack(str(record['NumSamples'])+'h', f.read(2*record['NumSamples'])), dtype=np.int16)
        data = np.fromfile(self.f, dtype=np.int16, count=record['NumSamples']) # load directly using numpy
        nchans = self.layoutrecords[record['Probe']].nchans
        data.shape = (nchans, -1) # reshape to have nchans rows, as indicated in layout
        return data

    '''
    class LowPassMultiChanRecord(object):
        """Low-pass multichannel (usually 10) continuous waveform record"""
        def __init__(self, lowpassrecords):
            """Takes several low pass records, all at the same timestamp"""
            self.lowpassrecords = toiter(lowpassrecords) # len of this is nchans
            self.TimeStamp = self.lowpassrecords[0].TimeStamp
            self.layout = self.lowpassrecords[0].layout
            self.NumSamples = self.lowpassrecords[0].NumSamples
            # why is this commented out? I guess because the checks it does are somewhat redundant
            # and slow things down a little...
            """
            self.tres = self.lowpassrecords[0].layout.tres
            self.chanis = []
            self.dataoffsets = []
            for recordi, record in enumerate(self.lowpassrecords): # typically 10 of these records
                # make sure all passed lowpassrecords have the same timestamp
                assert record.TimeStamp == self.TimeStamp
                assert record.layout.tres == self.tres # ditto
                # make sure each lowpassrecord in this batch of them at this timestamp all have unique channels
                newchanis = [ chani for chani in record.layout.ADchanlist if chani not in self.chanis ]
                assert newchanis != []
                # assigning this to each and every record might be taking up a lot of space,
                # better to assign it higher up, say to the stream?
                self.chanis.extend(newchanis)
            """

        def load(self, f):
            """Load waveform data for each lowpass record, appending it as
            channel(s) to a single 2D data array"""
            data = []
            for record in self.lowpassrecords:
                try:
                    recorddata = record.data
                except AttributeError:
                    recorddata = record.load(f) # to save time, only load the waveform if it's not already loaded
                # shouldn't matter if record.data is one channel (row) or several
                data.append(recorddata)
            # save as array, removing singleton dimensions
            data = np.squeeze(data)
            return data
    '''

    def _appendRecord(self, rec, reclistname):
        """Append record to reclistname"""
        if reclistname not in self.__dict__: # if not already an attrib
            self.__dict__[reclistname] = [] # init it
        self.__dict__[reclistname].append(rec) # append this record to its list

    def _connectRecords(self):
        """Connect the appropriate probe layout to each high and lowpass record"""
        #print('Connecting probe layouts to waveform records')
        # trim struct ndarrays to their final size
        self.highpassrecords.resize(self.nhighpassrecords, refcheck=False)
        self.lowpassrecords.resize(self.nlowpassrecords, refcheck=False)
        self.digitalsvalrecords.resize(self.ndigitalsvalrecords, refcheck=False)
        # cleanup by deleting any struct arrays of len 0
        for recname in ('highpassrecords', 'lowpassrecords', 'digitalsvalrecords'):
            if len(self.__getattribute__(recname)) == 0:
                self.__delattr__(recname)

        return # skip lowpassmultichan stuff for now

        try: # check if lowpass records are present in this .srf file
            self.lowpassrecords
        except AttributeError:
            return

        # Rearrange single channel lowpass records into
        # multichannel lowpass records
        #print('Rearranging single lowpass records into multichannel lowpass records')
        # get array of lowpass record timestamps
        rts = np.asarray([record.TimeStamp for record in self.lowpassrecords])
        # find at which records the timestamps change
        rtsis, = np.diff(rts).nonzero()
        # convert to edge values appropriate for getting slices of records
        # with the same timestamp
        rtsis = np.concatenate([[0], rtsis+1, [len(rts)]])
        self.lowpassmultichanrecords = []
        for rtsii in xrange(1, len(rtsis)): # start with the second rtsi
            lo = rtsis[rtsii-1]
            hi = rtsis[rtsii]
            lpass = LowPassMultiChanRecord(self.lowpassrecords[lo:hi])
            self.lowpassmultichanrecords.append(lpass)

        lpmclayout = self.get_LowPassMultiChanLayout()
        for lpmcr in self.lowpassmultichanrecords:
            lpmcr.layout = lpmclayout # overwrite each lpmc record's layout attrib

    def _verifyParsing(self):
        """Make sure timestamps of all records are in causal (increasing)
        order. If not, sort them I guess?"""
        #print('Asserting increasing record order')
        for attrname, attr in self.__dict__.items():
            if attrname.endswith('records') and iterable(attr):
                #print('Asserting %s are in causal order' % attrname)
                assert causalorder(attr)

    def get_LowPassMultiChanLayout(self):
        """Creates sort of a fake lowpassmultichan layout record, based on
        a lowpass single chan record, with some fields copied/modified from the
        highpass layout record in the file"""
        hplayout = self.highpassrecords[0].layout
        lpmclayout = copy(self.lowpassrecords[0].layout) # start with the layout of a lp single chan record
        lowpassrecords_t0 = self.lowpassmultichanrecords[0].lowpassrecords # lowpass records at first timestamp
        lpmclayout.nchans = len(lowpassrecords_t0)
        chans = [] # probe chans that were tapped off of the MCS patch board
                   # assume the mapping between AD chans and probe chans (if not 1 to 1) was done correctly before recording
        ADchanlist = [] # corresponding A/D chans
        PROBEDESCRIPRE = re.compile(r'ch(?P<tappedchan>[0-9]+)') # find 'ch' followed by at least 1 digit
        for lowpassrecord in self.lowpassmultichanrecords[0].lowpassrecords: # should be one per LFP channel
            layout = lowpassrecord.layout
            mo = PROBEDESCRIPRE.search(layout.probe_descrip) # match object
            if mo != None:
                chan = int(mo.groupdict()['tappedchan'])
                chans.append(chan)
                assert len(layout.ADchanlist) == 1 # shouldn't have more than one ADchan per lowpassrecord
                ADchan = layout.ADchanlist[0]
                ADchanlist.append(ADchan)
            else:
                raise ValueError, 'cannot parse LFP chan from probe description: %r' % layout.probe_descrip
        lpmclayout.chans = np.asarray(chans)
        lpmclayout.ADchanlist = np.asarray(ADchanlist) # replace single chan A/D chanlist with our new multichan highpass probe based one
        lpmclayout.probe_descrip = "LFP probe chans: %r; A/D chans: %r" % (lpmclayout.chans, lpmclayout.ADchanlist)
        lpmclayout.electrode_name = hplayout.electrode_name
        lpmclayout.probewinlayout = hplayout.probewinlayout
        return lpmclayout

    def pickle(self):
        """Pickle self to a .parse file"""
        print('Saving parse info to %r' % self.parsefname)
        pf = open(self.parsefname, 'wb') # can also compress pickle with gzip
        cPickle.dump(self, pf, protocol=-1) # pickle self to .parse file, use most efficient (least human readable) protocol
        pf.close()
        print('Saved parse info to %r' % self.parsefname)

    def unpickle(self):
        """Unpickle self from a .parse file"""
        print('Trying to recover parse info from %r' % self.parsefname)
        pf = open(self.parsefname, 'rb') # can also uncompress pickle with gzip
        #self = cPickle.load(pf) # NOTE: this doesn't work as intended
        other = cPickle.load(pf)
        pf.close()
        #other.fname = self.fname # seems unnecessary
        #other.parsefname = self.parsefname # ditto
        other.f = self.f # restore open .srf file on unpickle
        other.filelock = self.filelock # ditto for the filelock object
        for stream in [other.hpstream, other.lpstream]:
            if stream: stream.srff = self # rebind self to other's non-None streams
        self.__dict__ = other.__dict__ # set other's attribs to self
        print('Recovered parse info from %r' % self.parsefname)

    def __getstate__(self):
        """Don't pickle open .srf file on pickle"""
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['f'] # exclude open .srf file
        del d['filelock'] # ditto for the filelock object
        # leave the streams be, since you need them for their enabled chans info
        return d
    '''
    def __setstate__(self, d):
        """Do the stuff you might normally do here in self.unpickle() instead"""
    '''

class FileHeader(object):
    """Surf file header. Takes an open file, parses in from current file
    pointer position, stores header fields as attribs"""
    # Surf file header constants, field sizes in bytes
    UFF_FILEHEADER_LEN = 2048 # 'UFF' == 'Universal File Format'
    UFF_NAME_LEN = 10
    UFF_OSNAME_LEN = 12
    UFF_NODENAME_LEN = 32
    UFF_DEVICE_LEN = 32
    UFF_PATH_LEN = 160
    UFF_FILENAME_LEN = 32
    # pad area to bring uff area to 512, this really should be calculated, not hard-coded
    UFF_FH_PAD_LEN = 76
    UFF_APPINFO_LEN = 32
    UFF_OWNER_LEN = 14
    UFF_FILEDESC_LEN = 64
    UFF_FH_USERAREA_LEN = UFF_FILEHEADER_LEN - 512 # 1536

    def __len__(self):
        return 2050

    def parse(self, f):
        self.offset = f.tell()
        self.FH_rec_type, = unpack('B', f.read(1)) # must be 1
        assert self.FH_rec_type == 1
        self.FH_rec_type_ext, = unpack('B', f.read(1)) # must be 0
        assert self.FH_rec_type_ext == 0
        self.UFF_name = f.read(self.UFF_NAME_LEN).rstrip(NULL) # must be 'UFF'
        assert self.UFF_name == 'UFF'
        # major UFF ver
        self.UFF_major, = unpack('B', f.read(1))
        # minor UFF ver
        self.UFF_minor, = unpack('B', f.read(1))
        # FH record length in bytes
        self.FH_rec_len, = unpack('H', f.read(2))
        # DBRD record length in bytes
        self.DRDB_rec_len, = unpack('H', f.read(2))
        # 2 bi-directional seeks format
        self.bi_di_seeks, = unpack('H', f.read(2))
        # OS name, ie "WINDOWS 2000"
        self.OS_name = f.read(self.UFF_OSNAME_LEN).rstrip(NULL)
        self.OS_major, = unpack('B', f.read(1)) # OS major rev
        self.OS_minor, = unpack('B', f.read(1)) # OS minor rev
        # creation time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes
        self.create = TimeDate()
        self.create.parse(f)
        # last append time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes,
        # although this tends to be identical to creation time for some reason
        self.append = TimeDate()
        self.append.parse(f)
        # system node name - same as BDT
        self.node = f.read(self.UFF_NODENAME_LEN).rstrip(NULL)
        # device name - same as BDT
        self.device = f.read(self.UFF_DEVICE_LEN).rstrip(NULL)
        # path name
        self.path = f.read(self.UFF_PATH_LEN).rstrip(NULL)
        # original file name at creation
        self.filename = f.read(self.UFF_FILENAME_LEN).rstrip(NULL)
        # pad area to bring uff area to 512, no need to save it, skip it instead
        #self.pad = f.read(self.UFF_FH_PAD_LEN).replace(NULL, ' ')
        f.seek(self.UFF_FH_PAD_LEN, 1)
        # application task name & version
        self.app_info = f.read(self.UFF_APPINFO_LEN).rstrip(NULL)
        # user's name as owner of file
        self.user_name = f.read(self.UFF_OWNER_LEN).rstrip(NULL)
        # description of file/exp
        self.file_desc = f.read(self.UFF_FILEDESC_LEN).rstrip(NULL)
        # non user area of UFF header should be 512 bytes
        assert f.tell() - self.offset == 512
        # additional user area, no need to save it, skip it instead
        #self.user_area = unpack('B'*self.UFF_FH_USERAREA_LEN, f.read(self.UFF_FH_USERAREA_LEN))
        f.seek(self.UFF_FH_USERAREA_LEN, 1)
        assert f.tell() - self.offset == self.UFF_FILEHEADER_LEN

        # this is the end of the original UFF header I think,
        # the next two fields seem to have been added on to the end by Tim:

        # record type, must be 1 BIDIRECTIONAL SUPPORT
        self.bd_FH_rec_type, = unpack('B', f.read(1))
        assert self.bd_FH_rec_type == 1
        # record type extension, must be 0 BIDIRECTIONAL SUPPORT
        self.bd_FH_rec_type_ext, = unpack('B', f.read(1))
        assert self.bd_FH_rec_type_ext == 0
        # total length is 2050 bytes
        self.length = f.tell() - self.offset
        assert self.length == 2050


class TimeDate(object):
    """TimeDate record, reverse of C'S DateTime"""
    def __len__(self):
        return 18

    def parse(self, f):
        # not really necessary, comment out to save memory
        #self.offset = f.tell()
        self.sec, = unpack('H', f.read(2))
        self.min, = unpack('H', f.read(2))
        self.hour, = unpack('H', f.read(2))
        self.day, = unpack('H', f.read(2))
        self.month, = unpack('H', f.read(2))
        self.year, = unpack('H', f.read(2))
        # hack to skip 6 bytes
        f.seek(6, 1)


class DRDB(object):
    """Data Record Descriptor Block, aka UFF_DATA_REC_DESC_BLOCK in SurfBawd"""
    # Surf DRDB constants
    UFF_DRDB_BLOCK_LEN = 2048
    UFF_DRDB_NAME_LEN = 20
    UFF_DRDB_PAD_LEN = 16
    UFF_RSFD_PER_DRDB = 77

    def __init__(self):
        self.DR_subfields = []

    def __len__(self):
        return 2208

    def __str__(self):
        info = "Record type: %s size: %s name: %s" % \
               (self.DR_rec_type, self.DR_size, self.DR_name)
        #for sub in self.DR_subfields:
        #    info += "\t" + str(sub) + "\n"
        return info

    def parse(self, f):
        self.offset = f.tell()
        # record type; must be 2
        self.DRDB_rec_type, = unpack('B', f.read(1))
        # SurfBawd uses this to detect that it's passed the last DRDB, not exactly failsafe...
        if self.DRDB_rec_type != 2:
            raise DRDBError
        # record type extension
        self.DRDB_rec_type_ext, = unpack('B', f.read(1))
        # Data Record type for DRDB 3-255, ie 'P' or 'V' or 'M', etc..
        # don't know quite why SurfBawd required these byte values, this is
        # more than the normal set of ASCII chars
        self.DR_rec_type = f.read(1)
        assert int(unpack('B', self.DR_rec_type)[0]) in range(3, 256)
        # Data Record type ext; ignored
        self.DR_rec_type_ext, = unpack('B', f.read(1))
        # Data Record size in bytes, signed, -1 means dynamic
        self.DR_size, = unpack('i', f.read(4))
        # Data Record name
        self.DR_name = f.read(self.UFF_DRDB_NAME_LEN).rstrip(NULL)
        # number of sub-fields in Data Record
        self.DR_num_fields, = unpack('H', f.read(2))
        # pad bytes for expansion, no need to save it, skip it instead
        #self.DR_pad = unpack('B'*self.UFF_DRDB_PAD_LEN, f.read(self.UFF_DRDB_PAD_LEN))
        f.seek(self.UFF_DRDB_PAD_LEN, 1)
        # sub fields desc. RSFD = Record Subfield Descriptor
        for rsfdi in xrange(self.UFF_RSFD_PER_DRDB):
            rsfd = RSFD()
            rsfd.parse(f)
            self.DR_subfields.append(rsfd)
        assert f.tell() - self.offset == self.UFF_DRDB_BLOCK_LEN

        # this is the end of the original DRDB I think, the next two fields
        # seem to have been added on to the end by Tim:

        # hack to skip past unexplained extra 156 bytes (happens to equal 6*RSFD.length)
        f.seek(156, 1)
        # record type; must be 2 BIDIRECTIONAL SUPPORT
        self.bd_DRDB_rec_type, = unpack('B', f.read(1))
        assert self.bd_DRDB_rec_type == 2
        # record type extension; must be 0 BIDIRECTIONAL SUPPORT
        self.bd_DRDB_rec_type_ext, = unpack('B', f.read(1))
        assert self.bd_DRDB_rec_type_ext == 0
        # hack to skip past unexplained extra 2 bytes, sometimes they're 0s, sometimes not
        f.seek(2, 1)
        # total length should be 2050 bytes, but with above skip hacks, it's 2208 bytes
        self.length = f.tell() - self.offset
        assert self.length == 2208


class RSFD(object):
    """Record Subfield Descriptor for Data Record Descriptor Block"""
    UFF_DRDB_RSFD_NAME_LEN = 20

    def __len__(self):
        return 26

    def __str__(self):
        return "%s of type: %s with field size: %s" \
                % (self.subfield_name, self.subfield_data_type, self.subfield_size)

    def parse(self, f):
        self.offset = f.tell()
        # DRDB subfield name
        self.subfield_name = f.read(self.UFF_DRDB_RSFD_NAME_LEN).rstrip(NULL)
        # underlying data type
        self.subfield_data_type, = unpack('H', f.read(2))
        # sub field size in bytes, signed
        self.subfield_size, = unpack('i', f.read(4))
        self.length = f.tell() - self.offset
        assert self.length == 26


class LayoutRecord(object):
    """Polytrode layout record"""
    # Surf layout record constants
    SURF_MAX_CHANNELS = 64 # currently supports one or two DT3010 boards, could be higher

    def __len__(self):
        return 1725

    def parse(self, f):
        # not really necessary, comment out to save memory
        #self.offset = f.tell()
        # Record type 'L'
        self.UffType = f.read(1)
        # hack to skip next 7 bytes
        f.seek(7, 1)
        # Time stamp, 64 bit signed int
        self.TimeStamp, = unpack('q', f.read(8))
        # SURF major version number (2)
        self.SurfMajor, = unpack('B', f.read(1))
        # SURF minor version number (1)
        self.SurfMinor, = unpack('B', f.read(1))
        # hack to skip next 2 bytes
        f.seek(2, 1)
        # ADC/precision CT master clock frequency (1Mhz for DT3010)
        self.MasterClockFreq, = unpack('i', f.read(4))
        # undecimated base sample frequency per channel (25kHz)
        self.BaseSampleFreq, = unpack('i', f.read(4))
        # true (1) if Stimulus DIN acquired
        self.DINAcquired, = unpack('B', f.read(1))
        # hack to skip next byte
        f.seek(1, 1)
        # probe number
        self.Probe, = unpack('h', f.read(2))
        # =E,S,C for epochspike, spikestream, or continuoustype
        self.ProbeSubType = f.read(1)
        # hack to skip next byte
        f.seek(1, 1)
        # number of channels in the probe (54, 1)
        self.nchans, = unpack('h', f.read(2))
        # number of samples displayed per waveform per channel (25, 100)
        self.pts_per_chan, = unpack('h', f.read(2))
        # hack to skip next 2 bytes
        f.seek(2, 1)
        # {n/a to cat9} total number of samples per file buffer for this probe
        # (redundant with SS_REC.NumSamples) (135000, 100)
        self.pts_per_buffer, = unpack('i', f.read(4))
        # pts before trigger (7)
        self.trigpt, = unpack('h', f.read(2))
        # Lockout in pts (2)
        self.lockout, = unpack('h', f.read(2))
        # A/D board threshold for trigger (0-4096)
        self.threshold, = unpack('h', f.read(2))
        # A/D sampling decimation factor (1, 25)
        self.skippts, = unpack('h', f.read(2))
        # S:H delay offset for first channel of this probe (1)
        self.sh_delay_offset, = unpack('h', f.read(2))
        # hack to skip next 2 bytes
        f.seek(2, 1)
        # A/D sampling frequency specific to this probe (ie. after decimation,
        # if any) (25000, 1000)
        self.sampfreqperchan, = unpack('i', f.read(4))
        # us, store it here for convenience
        self.tres = int(round(1 / float(self.sampfreqperchan) * 1e6)) # us
        # MOVE BACK TO AFTER SHOFFSET WHEN FINISHED WITH CAT 9!!! added May 21, 1999
        # only the first self.nchans are filled (5000), the rest are junk values that pad to 64 channels
        self.extgain = np.asarray(unpack('H'*self.SURF_MAX_CHANNELS, f.read(2*self.SURF_MAX_CHANNELS)))
        # throw away the junk values
        self.extgain = self.extgain[:self.nchans]
        # A/D board internal gain (1,2,4,8) <--MOVE BELOW extgain after finished with CAT9!!!!!
        self.intgain, = unpack('h', f.read(2))
        # (0 to 53 for highpass, 54 to 63 for lowpass, + junk values that pad
        # to 64 channels) v1.0 had ADchanlist to be an array of 32 ints.  Now it
        # is an array of 64, so delete 32*4=128 bytes from end
        self.ADchanlist = unpack('h'*self.SURF_MAX_CHANNELS, f.read(2 * self.SURF_MAX_CHANNELS))
        # throw away the junk values
        self.ADchanlist = np.asarray(self.ADchanlist[:self.nchans])
        # hack to skip next byte
        f.seek(1, 1)
        # ShortString (uMap54_2a, 65um spacing)
        self.probe_descrip = f.read(255).rstrip(NULL)
        # hack to skip next byte
        f.seek(1, 1)
        # ShortString (uMap54_2a)
        self.electrode_name = f.read(255).rstrip(NULL)
        # hack to skip next 2 bytes
        f.seek(2, 1)
        # MOVE BELOW ADCHANLIST FOR CAT 9
        # v1.0 had ProbeWinLayout to be 4*32*2=256 bytes, now only 4*4=16 bytes, so add 240 bytes of pad
        self.probewinlayout = ProbeWinLayout()
        self.probewinlayout.parse(f)
        # array[0..879 {remove for cat 9!!!-->}- 4{pts_per_buffer} - 2{SHOffset}] of BYTE;
        # {pad for future expansion/modification}, no need to save it, skip it instead
        f.seek(880-4-2, 1)
        # hack to skip next 6 bytes, or perhaps pad should be 4+2 longer
        f.seek(6, 1)


class ProbeWinLayout(object):
    """Probe window layout"""
    def __len__(self):
        return 16

    def parse(self, f):
        # not really necessary, comment out to save memory
        #self.offset = f.tell()
        self.left, = unpack('i', f.read(4))
        self.top, = unpack('i', f.read(4))
        self.width, = unpack('i', f.read(4))
        self.height, = unpack('i', f.read(4))


class EpochRecord(object):
    def __init__(self):
        raise NotImplementedError('Spike epoch (non-continous) recordings currently unsupported')


class AnalogSValRecord(object):
    def __init__(self):
        raise NotImplementedError('Analog single value recordings currently unsupported')


class MessageRecord(object):
    """Message record"""
    def __len__(self):
        return 28 + self.MsgLength

    def parse(self, f):
        # not really necessary, comment out to save memory
        #self.offset = f.tell()
        # 1 byte -- SURF_MSG_REC_UFFTYPE: 'M'
        self.UffType = f.read(1)
        # 1 byte -- 'U' user or 'S' Surf-generated
        self.SubType = f.read(1)
        # hack to skip next 6 bytes
        f.seek(6, 1)
        # Time stamp, 64 bit signed int
        self.TimeStamp, = unpack('q', f.read(8))
        # 8 bytes -- double - number of days (integral and fractional) since 30 Dec 1899
        self.DateTime, = unpack('d', f.read(8))
        # 4 bytes -- length of the msg string
        self.MsgLength, = unpack('i', f.read(4))
        # any length message {shortstring - for cat9!!!}
        self.Msg = f.read(self.MsgLength)


class SurfMessageRecord(MessageRecord):
    """Surf generated message record"""


class UserMessageRecord(MessageRecord):
    """User generated message record"""


class DisplayRecord(object):
    """Stimulus display header record"""
    def __len__(self):
        return 24 + len(self.Header) + 4

    def parse(self, f):
        #self.offset = f.tell() # not really necessary, comment out to save memory
        # 1 byte -- SURF_DSP_REC_UFFTYPE = 'D'
        self.UffType = f.read(1)
        # hack to skip next 7 bytes
        f.seek(7, 1)
        # Cardinal, 64 bit signed int
        self.TimeStamp, = unpack('q', f.read(8))
        # double, 8 bytes - number of days (integral and fractional) since 30 Dec 1899
        self.DateTime, = unpack('d', f.read(8))
        self.Header = StimulusHeader()
        self.Header.parse(f)
        # hack to skip next 4 bytes
        f.seek(4, 1)


class StimulusHeader(object):
    """Stimulus display header"""
    # Stimulus header constants
    OLD_STIMULUS_HEADER_FILENAME_LEN = 16
    STIMULUS_HEADER_FILENAME_LEN = 64
    NVS_PARAM_LEN = 749
    PYTHON_TBL_LEN = 50000

    def __len__(self):
        if self.version == 100: # Cat < 15
            return 4 + self.OLD_STIMULUS_HEADER_FILENAME_LEN + self.NVS_PARAM_LEN*4 + 28
        elif self.version == 110: # Cat >= 15
            return 4 + self.STIMULUS_HEADER_FILENAME_LEN + self.NVS_PARAM_LEN*4 + self.PYTHON_TBL_LEN + 28

    def parse(self, f):
        #self.offset = f.tell() # not really necessary, comment out to save memory
        self.header = f.read(2).rstrip(NULL) # always 'DS'?
        self.version, = unpack('H', f.read(2))
        if self.version not in (100, 110): # Cat < 15, Cat >= 15
            raise ValueError, 'Unknown stimulus header version %d' % self.version
        if self.version == 100: # Cat < 15 has filename field length == 16
            # ends with a NULL followed by spaces for some reason, at least in Cat 13 file 03 - ptumap#751a_track5_m-seq.srf
            self.filename = f.read(self.OLD_STIMULUS_HEADER_FILENAME_LEN).rstrip().rstrip(NULL)
        elif self.version == 110: # Cat >= 15 has filename field length == 64
            self.filename = f.read(self.STIMULUS_HEADER_FILENAME_LEN).rstrip(NULL)
        # NVS binary header, array of single floats
        self.parameter_tbl = list(unpack('f'*self.NVS_PARAM_LEN, f.read(4*self.NVS_PARAM_LEN)))
        for parami, param in enumerate(self.parameter_tbl):
            if str(param) == '1.#QNAN':
                # replace 'Quiet NAN' floats with Nones. This won't work for Cat < 15
                # because NVS display left empty fields as NULL instead of NAN
                self.parameter_tbl[parami] = None
        # dimstim's text header
        if self.version == 110: # only Cat >= 15 has the text header
            self.python_tbl = f.read(self.PYTHON_TBL_LEN).rstrip()
        self.screen_width, = unpack('f', f.read(4)) # cm, single float
        self.screen_height, = unpack('f', f.read(4)) # cm
        self.view_distance, = unpack('f', f.read(4)) # cm
        self.frame_rate, = unpack('f', f.read(4)) # Hz
        self.gamma_correct, = unpack('f', f.read(4))
        self.gamma_offset, = unpack('f', f.read(4))
        self.est_runtime, = unpack('H', f.read(2)) # in seconds
        self.checksum, = unpack('H', f.read(2))


def causalorder(records):
    """Checks to see if the timestamps of all the records are in
    causal (increasing) order. Returns True or False"""
    '''
    for record1, record2 in itertools.izip(records[:-1], records[1:]):
        if record1.TimeStamp > record2.TimeStamp:
            return False
    return True
    '''
    # more straightforward using numpy:
    try:
        ts = np.asarray([ record.TimeStamp for record in records ])
    except AttributeError:
        ts = np.asarray([ record['TimeStamp'] for record in records ])
    # is ts in increasing order, ie is difference between subsequent entries >= 0?
    return (np.diff(ts) >= 0).all()
