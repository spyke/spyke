"""Handles parsing of Surf-generated .srf files"""

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
import numpy as np
import os
import cPickle
import struct
import unittest

NULL = '\x00'
DEFAULTINTERPSAMPFREQ = 50000 # default interpolated sample rate, in Hz


class DRDBError(ValueError):
    """ Used to indicate when you've passed the last DRDB at the start of the
    .srf file
    """
    pass


class Section(object):
    """ Represents a block of bytes denoting a 'section' of the Surf file. """
    def __init__(self, sec):
        self.id = sec.__class__.__name__
        self.len = len(sec)

    def __len__(self):
        return self.len


class Record(object):
    """ Used to explicity represent the endianness of the surf file and that
    of the machine. The machine endianness doesn't strictly have to be known
    but it might come in useful.
    """
    SURF_ENDIANNESS = '<'
    def __init__(self):
        if struct.pack('i', 1)[0] == '\x01':
            self.endianness = '<'
        else:
            self.endianness = '>'

    def unpack(self, format, bytes):
        return struct.unpack(self.SURF_ENDIANNESS + format, bytes)


class File(Record):
    """ Open a .srf file and, after parsing, expose all of its headers and
    records as attribs.
    Disabled: If no synonymous .parse file exists, parses
              the .srf file and saves the parsing in a .parse file.
    Stores as attribs:
        - Surf file header
        - Surf data record descriptor blocks
        - electrode layout records
        - message records
        - high and low pass continuous waveform records
        - stimulus display header records
        - stimulus digital single val records
    """

    def __init__(self, name):
        Record.__init__(self)

        self.name = name
        self.fileSize = os.stat(self.name)[6]
        self.sections = []
        self.sectionSummary = {}
        self.open()
        self.parsefname = os.path.splitext(self.f.name)[0] + '.parse'

    def __str__(self):
        if len(self.sections) == 0:
            return self.name + ' has not been parsed yet.'
        else:
            # we assume the file has been parsed correctly
            if '_cached_' not in self.sectionSummary:
                self.sectionSummary['_cached_'] = True

                for sec in self.sections:
                    num = self.sectionSummary.setdefault(sec.id, 0) + 1
                    self.sectionSummary[sec.id] = num
            return str(self.sectionSummary)

    def open(self):
        """ Open the .srf file """
        self.f = file(self.name, 'rb')
        self._parseFileHeader()

    def close(self):
        """ Close the .srf file """
        self.f.close()

    def _deserialize(self):
        """ Old deserialization code factored out as a method. Potentially
        on its way to deprecation.
        """
        print 'Trying to recover parse info from %r' % self.parsefname

        pf = file(self.parsefname, 'rb')
        u = cPickle.Unpickler(pf)

        def persistent_load(persid):
            """ required to restore the .srf file Record as an existing
            open file for reading
            """
            if persid == self.f.name:
                return self.f
            else:
                raise cPickle.UnpicklingError,  \
                            'Invalid persistent id: %r' % persid

        # add this method to the unpickler
        u.persistent_load = persistent_load
        fat = u.load()
        pf.close()

        # Grab all normal attribs of fat and assign them to self
        for key, val in fat.__dict__.items():
            self.__setattr__(key, val)

        print 'Recovered parse info from %r' % self.parsefname

    def _parseFileHeader(self):
        # Parse the Surf file header
        self.fileheader = FileHeader(self.f)
        self.fileheader.parse()
        self.sections.append(Section(self.fileheader))
        print 'Parsed fileheader'

    def _parseDRDBS(self):
        # Parse the DRDBs (Data Record Descriptor Blocks)
        self.drdbs = []
        while True:
            drdb = DRDB(self.f)
            try:
                drdb.parse()
                self.drdbs.append(drdb)
                self.sections.append(Section(drdb))

            except DRDBError:
                # we've gone past the last DRDB
                # set file pointer back to where we were
                self.f.seek(drdb.offset)
                break

    def _verifyParsing(self):
        # Make sure timestamps of all records are in causal (increasing)
        # order. If not, sort them I guess?
        print 'Asserting increasing record order'
        for item in self.__dict__:
            if item.endswith('records'):
                print 'Asserting ' + item + ' is in causal order'
                assert causalorder(self.__dict__[item])

    def parse(self, force=True, save=False):
        """ Parse the .srf file """

        # recover Fat Record pickled in .parse file
        try:
            # force a new parsing
            if force:
                # make the try fail, skip to the except block
                raise IOError

            self._deserialize()

        # parsing is being forced, or .parse file doesn't exist, or something's
        # wrong with it. Parse the .srf file
        except IOError:

            print 'Parsing %r' % self.f.name
            f = self.f # abbrev

            self._parseDRDBS()
            self._parserecords()


            print 'Done parsing %r' % self.f.name

            self._verifyParsing()
            self._connectRecords()

            if save:
                self.serialize()

    def _parserecords(self):
        """ Parse all the records in the file, but don't load any waveforms """

        f = self.f

        # mapping of flag values to appropriate records
        records = {
                    'L'     :   LayoutRecord,
                    'M'     :   MessageRecord,
                    'MS'    :   MessageRecord,
                    'PS'    :   HighPassRecord,
                    'PC'    :   LowPassRecord,
                    'PE'    :   EpochRecord,
                    'D'     :   DisplayRecord,
                    'VD'    :   DigitalSValRecord,
                    'VA'    :   AnalogSingleValRecord,
                  }

        while True:
            # returns an empty string when EOF is reached
            flag = f.read(2).strip('\0')
            if flag == '':
                break

            # put file pointer back to start of flag
            f.seek(-2, 1)

            if flag in records:
                rec = records[flag](f)
                rec.parse()

                # our list of the collected records
                lis = rec.__class__.__name__.lower() + 's'
                if lis not in self.__dict__:
                    self.__dict__[lis] = []
                self.__dict__[lis].append(rec)

                self.sections.append(Section(rec))

            else:
                raise ValueError, \
                        'Unexpected flag %r at offset %d' % (flag, f.tell())

            self.percentParsed = f.tell() * 100 / self.fileSize

    def _connectRecords(self):
        # Connect the appropriate probe layout to each high
        # and lowpass record

        print 'Connecting probe layouts to waveform records'
        for record in self.highpassrecords:
            record.layout = self.layoutrecords[record.Probe]
        for record in self.lowpassrecords:
            record.layout = self.layoutrecords[record.Probe]

        # Rearrange single channel lowpass records into
        # multichannel lowpass records
        print 'Rearranging single lowpass records into ' \
                'multichannel lowpass records'
        rts = np.asarray([record.TimeStamp for record in \
                self.lowpassrecords]) # array of lowpass record timestamps

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

        #self.highpassstream = Stream(records=self.highpassrecords,
        #                                sampfreq=DEFAULTINTERPSAMPFREQ)
        #self.lowpassstream = Stream(records=self.lowpassrecords)

    def serialize(self):
        """ Creates a Fat Record, saves all the parsed headers and records to
        it, and pickles it to a file
        """
        print 'Saving parse info to %r' % self.parsefname
        fat = Fat()
        fat.fileheader = self.fileheader
        fat.drdbs = self.drdbs
        fat.layoutrecords = self.layoutrecords
        fat.messagerecords = self.messagerecords
        fat.highpassrecords = self.highpassrecords
        fat.lowpassrecords = self.lowpassrecords
        fat.lowpassmultichanrecords = self.lowpassmultichanrecords
        fat.displayrecords = self.displayrecords
        fat.digitalsvalrecords = self.digitalsvalrecords
        pf = file(self.parsefname, 'wb')

        # make a Pickler, use most efficient (least human readable) protocol
        p = cPickle.Pickler(pf, protocol=-1)

        # required to make the .srf file Record persistent and remain open for
        # reading when unpickled
        def persistent_id(obj):
            if hasattr(obj, 'name'):
                # the file Record's filename defines its persistent id for
                # pickling purposes
                return obj.name
            else:
                return None

        # assign this method to the pickler
        p.persistent_id = persistent_id

        # pickle fat to .parse file
        p.dump(fat)
        pf.close()
        print 'Saved parse info to %r' % self.parsefname


class FileHeader(Record):
    """ Surf file header. Takes an open file, parses in from current file
    pointer position, stores header fields as attribs
    """
    # Surf file header constants, field sizes in bytes
    UFF_FILEHEADER_LEN = 2048 # 'UFF' == 'Universal File Format'
    UFF_NAME_LEN = 10
    UFF_OSNAME_LEN = 12
    UFF_NODENAME_LEN = 32
    UFF_DEVICE_LEN = 32
    UFF_PATH_LEN = 160
    UFF_FILENAME_LEN = 32
    UFF_FH_PAD_LEN = 76  # pad area to bring uff area to 512,
                         # this really should be calculated, not hard-coded
    UFF_APPINFO_LEN = 32
    UFF_OWNER_LEN = 14
    UFF_FILEDESC_LEN = 64
    UFF_FH_USERAREA_LEN = UFF_FILEHEADER_LEN - 512 # 1536

    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 2050

    def parse(self):
        f = self.f
        self.offset = f.tell()
        self.FH_rec_type, = self.unpack('B', f.read(1)) # must be 1
        assert self.FH_rec_type == 1
        self.FH_rec_type_ext, = self.unpack('B', f.read(1)) # must be 0
        assert self.FH_rec_type_ext == 0
        self.UFF_name = f.read(self.UFF_NAME_LEN).rstrip(NULL) # must be 'UFF'
        assert self.UFF_name == 'UFF'

        # major UFF ver
        self.UFF_major, = self.unpack('B', f.read(1))

        # minor UFF ver
        self.UFF_minor, = self.unpack('B', f.read(1))

        # FH record length in bytes
        self.FH_rec_len, = self.unpack('H', f.read(2))

        # DBRD record length in bytes
        self.DRDB_rec_len, = self.unpack('H', f.read(2))

        # 2 bi-directional seeks format
        self.bi_di_seeks, = self.unpack('H', f.read(2))

        # OS name, ie "WINDOWS 2000"
        self.OS_name = f.read(self.UFF_OSNAME_LEN).rstrip(NULL)
        self.OS_major, = self.unpack('B', f.read(1)) # OS major rev
        self.OS_minor, = self.unpack('B', f.read(1)) # OS minor rev

        # creation time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes
        self.create = TimeDate(f)
        self.create.parse()

        # last append time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes,
        # although this tends to be identical to creation time for some reason
        self.append = TimeDate(f)
        self.append.parse()

        # system node name - same as BDT
        self.node = f.read(self.UFF_NODENAME_LEN).rstrip(NULL)

        # device name - same as BDT
        self.device = f.read(self.UFF_DEVICE_LEN).rstrip(NULL)

        # path name
        self.path = f.read(self.UFF_PATH_LEN).rstrip(NULL)

        # original file name at creation
        self.filename = f.read(self.UFF_FILENAME_LEN).rstrip(NULL)

        # pad area to bring uff area to 512
        self.pad = f.read(self.UFF_FH_PAD_LEN).replace(NULL, ' ')

        # application task name & version
        self.app_info = f.read(self.UFF_APPINFO_LEN).rstrip(NULL)

        # user's name as owner of file
        self.user_name = f.read(self.UFF_OWNER_LEN).rstrip(NULL)

        # description of file/exp
        self.file_desc = f.read(self.UFF_FILEDESC_LEN).rstrip(NULL)

        # non user area of UFF header should be 512 bytes
        assert f.tell() - self.offset == 512

        # additional user area
        self.user_area = self.unpack('B'*self.UFF_FH_USERAREA_LEN,
                                        f.read(self.UFF_FH_USERAREA_LEN))
        assert f.tell() - self.offset == self.UFF_FILEHEADER_LEN

        # this is the end of the original UFF header I think,
        # the next two fields seem to have been added on to the end by Tim:
        # record type, must be 1 BIDIRECTIONAL SUPPORT
        self.bd_FH_rec_type, = self.unpack('B', f.read(1))
        assert self.bd_FH_rec_type == 1

        # record type extension, must be 0 BIDIRECTIONAL SUPPORT
        self.bd_FH_rec_type_ext, = self.unpack('B', f.read(1))
        assert self.bd_FH_rec_type_ext == 0

        # total length is 2050 bytes
        self.length = f.tell() - self.offset
        assert self.length == 2050


class TimeDate(Record):
    """ TimeDate record, reverse of C'S DateTime """

    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 18

    def parse(self):
        f = self.f
        # not really necessary, comment out to save memory
        #self.offset = f.tell()
        self.sec, = self.unpack('H', f.read(2))
        self.min, = self.unpack('H', f.read(2))
        self.hour, = self.unpack('H', f.read(2))
        self.day, = self.unpack('H', f.read(2))
        self.month, = self.unpack('H', f.read(2))
        self.year, = self.unpack('H', f.read(2))
        self.junk = self.unpack('B'*6, f.read(6)) # junk data at end


class DRDB(Record):
    """ Data Record Descriptor Block, aka UFF_DATA_REC_DESC_BLOCK
    in SurfBawd
    """
    # Surf DRDB constants
    UFF_DRDB_BLOCK_LEN = 2048
    UFF_DRDB_NAME_LEN = 20
    UFF_DRDB_PAD_LEN = 16
    UFF_RSFD_PER_DRDB = 77

    def __init__(self, f):
        Record.__init__(self)
        self.DR_subfields = []
        self.f = f

    def __len__(self):
        return 2208

    def __str__(self):
        info = "Record type: %s size: %s name: %s" % \
                            (self.DR_rec_type, self.DR_size, self.DR_name)
        #for sub in self.DR_subfields:
        #    info += "\t" + str(sub) + "\n"
        return info

    def parse(self):
        f = self.f
        self.offset = f.tell()

        # record type; must be 2
        self.DRDB_rec_type, = self.unpack('B', f.read(1))

        # SurfBawd uses this to detect that it's passed the last DRDB,
        # not exactly failsafe...
        if self.DRDB_rec_type != 2:
            raise DRDBError

        # record type extension
        self.DRDB_rec_type_ext, = self.unpack('B', f.read(1))

        # Data Record type for DRDB 3-255, ie 'P' or 'V' or 'M', etc..
        # don't know quite why SurfBawd required these byte values, this is
        # more than the normal set of ASCII chars
        self.DR_rec_type = f.read(1)
        assert int(self.unpack('B', self.DR_rec_type)[0]) in range(3, 256)

        # Data Record type ext; ignored
        self.DR_rec_type_ext, = self.unpack('B', f.read(1))

        # Data Record size in bytes, signed, -1 means dynamic
        self.DR_size, = self.unpack('i', f.read(4))

        # Data Record name
        self.DR_name = f.read(self.UFF_DRDB_NAME_LEN).rstrip(NULL)

        # number of sub-fields in Data Record
        self.DR_num_fields, = self.unpack('H', f.read(2))

        # pad bytes for expansion
        self.DR_pad = self.unpack('B'*self.UFF_DRDB_PAD_LEN,
                        f.read(self.UFF_DRDB_PAD_LEN))

        # sub fields desc. RSFD = Record Subfield Descriptor
        for rsfdi in xrange(self.UFF_RSFD_PER_DRDB):
            rsfd = RSFD(f)
            rsfd.parse()
            self.DR_subfields.append(rsfd)
        assert f.tell() - self.offset == self.UFF_DRDB_BLOCK_LEN

        # this is the end of the original DRDB I think, the next two fields
        # seem to have been added on to the end by Tim:

        # hack to skip past unexplained extra 156 bytes (happens to equal
        # 6*RSFD.length)
        f.seek(156, 1)

        # record type; must be 2 BIDIRECTIONAL SUPPORT
        self.bd_DRDB_rec_type, = self.unpack('B', f.read(1))
        assert self.bd_DRDB_rec_type == 2

        # record type extension; must be 0 BIDIRECTIONAL SUPPORT
        self.bd_DRDB_rec_type_ext, = self.unpack('B', f.read(1))
        assert self.bd_DRDB_rec_type_ext == 0

        # hack to skip past unexplained extra 2 bytes, sometimes they're 0s,
        # sometimes not
        f.seek(2, 1)

        # total length should be 2050 bytes, but with above skip hacks,
        # it's 2208 bytes
        self.length = f.tell() - self.offset
        assert self.length == 2208


class RSFD(Record):
    """ Record Subfield Descriptor for Data Record Descriptor Block """
    UFF_DRDB_RSFD_NAME_LEN = 20

    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 26

    def __str__(self):
        return "%s of type: %s with field size: %s" % (self.subfield_name,
                                            self.subfield_data_type,
                                            self.subfield_size)

    def parse(self):
        f = self.f
        self.offset = f.tell()

        # DRDB subfield name
        self.subfield_name = f.read(self.UFF_DRDB_RSFD_NAME_LEN).rstrip(NULL)

        # underlying data type
        self.subfield_data_type, = self.unpack('H', f.read(2))

        # sub field size in bytes, signed
        self.subfield_size, = self.unpack('i', f.read(4))

        self.length = f.tell() - self.offset
        assert self.length == 26


class LayoutRecord(Record):
    """ Polytrode layout record """

    # Surf layout record constants
    SURF_MAX_CHANNELS = 64 # currently supports one or two
                           # DT3010 boards, could be higher

    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 1725

    def parse(self):
        f = self.f

        # not really necessary, comment out to save memory
        #self.offset = f.tell()

        # Record type 'L'
        self.UffType = f.read(1)

        # hack to skip next 7 bytes
        f.seek(7, 1)

        # Time stamp, 64 bit signed int
        self.TimeStamp, = self.unpack('q', f.read(8))

        # SURF major version number (2)
        self.SurfMajor, = self.unpack('B', f.read(1))
        # SURF minor version number (1)
        self.SurfMinor, = self.unpack('B', f.read(1))

        # hack to skip next 2 bytes
        f.seek(2, 1)

        # ADC/precision CT master clock frequency (1Mhz for DT3010)
        self.MasterClockFreq, = self.unpack('i', f.read(4))

        # undecimated base sample frequency per channel (25kHz)
        self.BaseSampleFreq, = self.unpack('i', f.read(4))

        # true (1) if Stimulus DIN acquired
        self.DINAcquired, = self.unpack('B', f.read(1))

        # hack to skip next byte
        f.seek(1, 1)

        # probe number
        self.Probe, = self.unpack('h', f.read(2))

        # =E,S,C for epochspike, spikestream, or continuoustype
        self.ProbeSubType = f.read(1)

        # hack to skip next byte
        f.seek(1, 1)

        # number of channels in the probe (54, 1)
        self.nchans, = self.unpack('h', f.read(2))

        # number of samples displayed per waveform per channel (25, 100)
        self.pts_per_chan, = self.unpack('h', f.read(2))

        # hack to skip next 2 bytes
        f.seek(2, 1)

        # {n/a to cat9} total number of samples per file buffer for this probe
        # (redundant with SS_REC.NumSamples) (135000, 100)
        self.pts_per_buffer, = self.unpack('i', f.read(4))

        # pts before trigger (7)
        self.trigpt, = self.unpack('h', f.read(2))

        # Lockout in pts (2)
        self.lockout, = self.unpack('h', f.read(2))

        # A/D board threshold for trigger (0-4096)
        self.threshold, = self.unpack('h', f.read(2))

        # A/D sampling decimation factor (1, 25)
        self.skippts, = self.unpack('h', f.read(2))

        # S:H delay offset for first channel of this probe (1)
        self.sh_delay_offset, = self.unpack('h', f.read(2))

        # hack to skip next 2 bytes
        f.seek(2, 1)

        # A/D sampling frequency specific to this probe (ie. after decimation,
        # if any) (25000, 1000)
        self.sampfreqperchan, = self.unpack('i', f.read(4))

        # us, store it here for convenience
        self.tres = int(round(1 / float(self.sampfreqperchan) * 1e6))

        # MOVE BACK TO AFTER SHOFFSET WHEN FINISHED WITH CAT 9!!! added May 21,
        # 1999 - only the first self.nchans are filled (5000), the rest are
        # junk values that pad to 64 channels
        self.extgain = self.unpack('H'*self.SURF_MAX_CHANNELS,
                                        f.read(2*self.SURF_MAX_CHANNELS))
        # throw away the junk values
        self.extgain = self.extgain[:self.nchans]

        # A/D board internal gain (1,2,4,8) <--MOVE BELOW extgain after
        # finished with CAT9!!!!!
        self.intgain, = self.unpack('h', f.read(2))

        # (0 to 53 for highpass, 54 to 63 for lowpass, + junk values that pad
        # to 64 channels) v1.0 had chanlist to be an array of 32 ints.  Now it
        # is an array of 64, so delete 32*4=128 bytes from end
        self.chanlist = self.unpack('h'*self.SURF_MAX_CHANNELS,
                                        f.read(2 * self.SURF_MAX_CHANNELS))

        # throw away the junk values
        self.chanlist = self.chanlist[:self.nchans]

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

        # MOVE BELOW CHANLIST FOR CAT 9 v1.0 had ProbeWinLayout to be
        # 4*32*2=256 bytes, now only 4*4=16 bytes, so add 240 bytes of pad
        self.probewinlayout = ProbeWinLayout(f)
        self.probewinlayout.parse()

        # array[0..879 {remove for cat 9!!!-->}- 4{pts_per_buffer} -
        # 2{SHOffset}] of BYTE; {pad for future expansion/modification}
        self.pad = self.unpack(str(880-4-2)+'B', f.read(880-4-2))

        # hack to skip next 6 bytes, or perhaps self.pad should be 4+2 longer
        f.seek(6, 1)

class ProbeWinLayout(Record):
    """ Probe window layout """
    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 16

    def parse(self):
        f = self.f
        # not really necessary, comment out to save memory
        #self.offset = f.tell()

        self.left, = self.unpack('i', f.read(4))
        self.top, = self.unpack('i', f.read(4))
        self.width, = self.unpack('i', f.read(4))
        self.height, = self.unpack('i', f.read(4))

class EpochRecord(Record):
    def __init__(self):
        raise NotImplementedError('Spike epochs (non-continous)'  \
                                  ' recordings currently unsupported')

class AnalogSingleValRecord(Record):
    def __init__(self):
        raise NotImplementedError('Analog single value recordings' \
                                  ' currently unsupported')

class MessageRecord(Record):
    """Message record"""
    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 28 + self.MsgLength

    def parse(self):
        f = self.f # abbrev

        # not really necessary, comment out to save memory
        #self.offset = f.tell()

        # 1 byte -- SURF_MSG_REC_UFFTYPE: 'M'
        self.UffType = f.read(1)

        # 1 byte -- 'U' user or 'S' Surf-generated
        self.SubType = f.read(1)

        # hack to skip next 6 bytes
        f.seek(6, 1)

        # Time stamp, 64 bit signed int
        self.TimeStamp, = self.unpack('q', f.read(8))

        # 8 bytes -- double - number of days (integral and fractional) since 30
        # Dec 1899
        self.DateTime, = self.unpack('d', f.read(8))

        # 4 bytes -- length of the msg string
        self.MsgLength, = self.unpack('i', f.read(4))

        # any length message {shortstring - for cat9!!!}
        self.Msg = f.read(self.MsgLength)

class ContinuousRecord(Record):
    """ Continuous waveform record """
    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 28 + self.NumSamples*2

    def parse(self):
        f = self.f

        # for speed and memory, read all 28 bytes at a time, skip reading
        # UffType, SubType, and CRC32 (which is always 0 anyway?)

        junk, self.TimeStamp, self.Probe, junk, junk, self.NumSamples = \
                                            self.unpack('qqhhii', f.read(28))
        self.waveformoffset = f.tell()

        # skip the waveform data for now
        f.seek(self.NumSamples*2, 1)

    def load(self):
        """ Loads waveform data for this continuous record, assumes that the
        appropriate probe layout record has been assigned as a .layout attrib
        """
        f = self.f # abbrev
        f.seek(self.waveformoffset)

        # {ADC Waveform type; dynamic array of SHRT (signed 16 bit)} -
        # converted to an ndarray
        self.waveform = np.asarray(self.unpack(str(self.NumSamples)+'h',
                        f.read(2*self.NumSamples)), dtype=np.int16)

        # reshape to have nchans rows, as indicated in layout
        self.waveform = self.waveform.reshape(self.layout.nchans, -1)

class HighPassRecord(ContinuousRecord):
    """High-pass continuous waveform record"""
    pass

class LowPassRecord(ContinuousRecord):
    """Low-pass continuous waveform record"""
    pass

class LowPassMultiChanRecord(Record):
    """ Low-pass multichannel (usually 10) continuous waveform record """
    def __init__(self, lowpassrecords):
        """ Takes several low pass records, all at the same timestamp """
        Record.__init__(self)
        self.lowpassrecords = toiter(lowpassrecords)
        self.TimeStamp = self.lowpassrecords[0].TimeStamp
        self.tres = self.lowpassrecords[0].layout.tres
        self.chanis = []
        #self.waveformoffsets = []

        for recordi, record in enumerate(self.lowpassrecords):

            # make sure all passed lowpassrecords have the same timestamp
            assert record.TimeStamp == self.TimeStamp
            assert record.layout.tres == self.tres # ditto

            newchanis = [chani for chani in record.layout.chanlist \
                                            if chani not in self.chanis ]

            # make sure there aren't any duplicate channels
            assert newchanis != []
            self.chanis.extend(newchanis)

    def load(self):
        """ Load waveform data for each lowpass record, appending it as
        channel(s) to a single 2D waveform array
        """
        self.waveform = []

        for record in self.lowpassrecords:
            record.load()
            # shouldn't matter if record.waveform is one
            # channel (row) or several
            self.waveform.append(record.waveform)

        # save as array, removing singleton dimensions
        self.waveform = np.squeeze(self.waveform)


class DisplayRecord(Record):
    """ Stimulus display header record """
    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 53092 + 28

    def parse(self):
        f = self.f
        # not really necessary, comment out to save memory
        #self.offset = f.tell()

        # 1 byte -- SURF_DSP_REC_UFFTYPE = 'D'
        self.UffType = f.read(1)

        # hack to skip next 7 bytes
        f.seek(7, 1)

        # Cardinal, 64 bit signed int
        self.TimeStamp, = self.unpack('q', f.read(8))

        # double, 8 bytes - number of days (integral and fractional) since 30
        # Dec 1899
        self.DateTime, = self.unpack('d', f.read(8))
        self.Header = StimulusHeader(f)
        self.Header.parse()

        # hack to skip next 4 bytes
        f.seek(4, 1)

class StimulusHeader(Record):
    """ Stimulus display header """

    # Stimulus header constants
    STIMULUS_HEADER_FILENAME_LEN = 64
    NVS_PARAM_LEN = 749
    PYTHON_TBL_LEN = 50000

    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def parse(self):
        f = self.f

        # not really necessary, comment out to save memory
        #self.offset = f.tell()

        # always 'DS'?
        self.header = f.read(2).rstrip(NULL)
        self.version, = self.unpack('H', f.read(2))
        self.filename = f.read(self.STIMULUS_HEADER_FILENAME_LEN).rstrip(NULL)

        # NVS binary header, array of single floats
        self.parameter_tbl = list(self.unpack('f'*self.NVS_PARAM_LEN,
                                                    f.read(4*self.NVS_PARAM_LEN)))

        for parami, param in enumerate(self.parameter_tbl):
            if str(param) == '1.#QNAN':
                # replace 'Quiet NAN' floats with Nones
                self.parameter_tbl[parami] = None

        # dimstim's text header
        self.python_tbl = f.read(self.PYTHON_TBL_LEN).rstrip()

        # cm, single float
        self.screen_width, = self.unpack('f', f.read(4))

        # cm
        self.screen_height, = self.unpack('f', f.read(4))

        # cm
        self.view_distance, = self.unpack('f', f.read(4))

        # Hz
        self.frame_rate, = self.unpack('f', f.read(4))
        self.gamma_correct, = self.unpack('f', f.read(4))
        self.gamma_offset, = self.unpack('f', f.read(4))

        # in seconds
        self.est_runtime, = self.unpack('H', f.read(2))
        self.checksum, = self.unpack('H', f.read(2))


class DigitalSValRecord(Record):
    """ Digital single value record """
    def __init__(self, f):
        Record.__init__(self)
        self.f = f

    def __len__(self):
        return 24

    def parse(self):
        f = self.f

        # for speed and memory, read all 24 bytes at a time, skip UffType and
        # SubType
        # Cardinal, 64 bit signed int; 16 bit single value
        junk, self.TimeStamp, self.SVal, junk, junk = \
                self.unpack('QQHHI', f.read(24))


class Fat(Record):
    """Stores all the stuff to be pickled into a .parse file and then
    unpickled as saved parse info
    """
    # TODO
    pass


def causalorder(records):
    """Checks to see if the timestamps of all the records are in
    causal (increasing) order. Returns True or False"""
    for record1, record2 in itertools.izip(records[:-1], records[1:]):
        if record1.TimeStamp > record2.TimeStamp:
            return False
    return True

def iterable(x):
    """Check if the input is iterable, stolen from numpy.iterable()"""
    try:
        iter(x)
        return True
    except:
        return False

def toiter(x):
    """Convert to iterable. If input is iterable, returns it. Otherwise returns it in a list.
    Useful when you want to iterate over an Record (like in a for loop),
    and you don't want to have to do type checking or handle exceptions
    when the Record isn't a sequence"""
    if iterable(x):
        return x
    else:
        return [x]


if __name__ == '__main__':
    # insert unittests here
    pass
