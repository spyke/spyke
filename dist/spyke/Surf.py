"""Handles parsing and streaming of Surf-generated .srf files"""

__authors__ = ['Martin Spacek', 'Reza Lotun']

import itertools
import numpy as np
import os
import cPickle
import struct
import unittest

NULL = '\x00'
DEFAULTSRFFNAME = '/data/Cat 15/81 - track 7c mseq16exps.srf'
DEFAULTINTERPSAMPFREQ = 50000 # default interpolated sample rate, in Hz

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

# Surf DRDB constants
UFF_DRDB_BLOCK_LEN = 2048
UFF_DRDB_NAME_LEN = 20
UFF_DRDB_PAD_LEN = 16
UFF_RSFD_PER_DRDB = 77
UFF_DRDB_RSFD_NAME_LEN = 20

# Surf layout record constants
SURF_MAX_CHANNELS = 64 # currently supports one or two 
                       # DT3010 boards, could be higher

# Stimulus header constants
STIMULUS_HEADER_FILENAME_LEN = 64
NVS_PARAM_LEN = 749
PYTHON_TBL_LEN = 50000


class DRDBError(ValueError):
    """ Used to indicate when you've passed the last DRDB at the start of the 
    .srf file
    """
    pass


class File(object):
    """ Open a .srf file and, after parsing, expose all of its headers and
    records as attribs. 
    Disabled: If no synonymous .parse file exists, parses
              the .srf file and saves the parsing in a .parse file.
    Stores as attribs:
        - Surf file header
        - Surf data record descriptor blocks
        - electrode layout
        - records
        - message records
        - high and low pass continuous waveform records
        - stimulus display header records
        - stimulus digital single val records
    """ 
    def __init__(self, name=DEFAULTSRFFNAME): 
        self.name = name 
        self.open()
        self.parsefname = os.path.splitext(self.f.name)[0] + '.parse'

    def open(self):
        """ Open the .srf file """
        self.f = file(self.name, 'rb')

    def close(self):
        """ Close the .srf file """
        self.f.close()

    def parse(self, force=True, save=False):
        """ Parse the .srf file """

        # recover Fat object pickled in .parse file
        try:             
            # force a new parsing
            if force:                 
                # make the try fail, skip to the except block
                raise IOError

            print 'Trying to recover parse info from %r' % self.parsefname

            pf = file(self.parsefname, 'rb')
            u = cPickle.Unpickler(pf)

            def persistent_load(persid):
                """ required to restore the .srf file object as an existing 
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

        # parsing is being forced, or .parse file doesn't exist, or something's
        # wrong with it. Parse the .srf file
        except IOError:

            print 'Parsing %r' % self.f.name
            f = self.f # abbrev

            # make sure we're at the start of the srf file before trying to
            # parse it
            f.seek(0) 

            # Parse the Surf file header
            self.fileheader = FileHeader(f)
            self.fileheader.parse()

            # Parse the DRDBs (Data Record Descriptor Blocks)
            self.drdbs = []
            while True:
                drdb = DRDB(f)
                try:
                    drdb.parse()
                    self.drdbs.append(drdb)
                
                except DRDBError: 
                    # we've gone past the last DRDB
                    # set file pointer back to where we were
                    f.seek(drdb.offset)                     
                    break

            self._parserecords()
            print 'Done parsing %r' % self.f.name

            # Make sure timestamps of all records are in causal (increasing) 
            # order. If not, sort them I guess?
            print 'Asserting increasing record order'
            assert causalorder(self.layoutrecords)
            assert causalorder(self.messagerecords)
            assert causalorder(self.highpassrecords)
            assert causalorder(self.lowpassrecords)
            assert causalorder(self.displayrecords)
            assert causalorder(self.digitalsvalrecords)

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
            for rtsii in range(1, len(rtsis)): # start with the second rtsi
                lo = rtsis[rtsii-1]
                hi = rtsis[rtsii]
                lpass = LowPassMultiChanRecord(self.lowpassrecords[lo:hi])
                self.lowpassmultichanrecords.append(lpass)

            self.highpassstream = Stream(records=self.highpassrecords, 
                                            sampfreq=DEFAULTINTERPSAMPFREQ)
            self.lowpassstream = Stream(records=self.lowpassrecords)

            if save:
                self.save()

    def _parserecords(self):
        """ Parse all the records in the file, but don't load any waveforms """

        f = self.f

        self.layoutrecords = []
        self.messagerecords = []
        self.highpassrecords = []
        self.lowpassrecords = []
        self.lowpassmultichanrecords = []
        self.displayrecords = []
        self.digitalsvalrecords = []

        while True:

            # returns an empty string when EOF is reached
            flag = f.read(2) 
            if flag == '':
                break

            # put file pointer back to start of flag
            f.seek(-2, 1)

            if flag[0] == 'L': 
                # polytrode layout record
                layoutrecord = LayoutRecord(f)
                layoutrecord.parse()
                self.layoutrecords.append(layoutrecord)

            elif flag[0] == 'M': 
                # message record
                messagerecord = MessageRecord(f)
                messagerecord.parse()
                self.messagerecords.append(messagerecord)

            elif flag[0] == 'P': 
                # polytrode waveform record
                if flag[1] == 'S': 
                    # spike stream (highpass) record
                    highpassrecord = HighPassRecord(f)
                    highpassrecord.parse()
                    self.highpassrecords.append(highpassrecord)
                elif flag[1] == 'C': 
                    # continuous (lowpass) record
                    lowpassrecord = LowPassRecord(f)
                    lowpassrecord.parse()
                    self.lowpassrecords.append(lowpassrecord)
                elif flag[1] == 'E': 
                    # spike epoch record
                    raise NotImplementedError, 'Spike epochs (non-continous)' \
                                            ' recordings currently unsupported'
                else:
                    raise ValueError, 'Unknown polytrode waveform' \
                                        'record type %r' % flag[1]

            elif flag[0] == 'D': 
                # stimulus display header record
                displayrecord = DisplayRecord(f)
                displayrecord.parse()
                self.displayrecords.append(displayrecord)
            elif flag[0] == 'V': 
                # single value record
                if flag[1] == 'D': 
                    # digital single value record
                    digitalsvalrecord = DigitalSValRecord(f)
                    digitalsvalrecord.parse()
                    self.digitalsvalrecords.append(digitalsvalrecord)
                elif flag[1] == 'A': 
                    # analog single value record
                    raise NotImplementedError, \
                        'Analog single value recordings currently unsupported'
                else:
                    raise ValueError, \
                            'Unknown single value record type %r' % flag[1]
            else:
                raise ValueError, \
                        'Unexpected flag %r at offset %d' % (flag, f.tell())

    def save(self):
        """ Creates a Fat object, saves all the parsed headers and records to
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

        # required to make the .srf file object persistent and remain open for
        # reading when unpickled
        def persistent_id(obj): 
            if hasattr(obj, 'name'):
                # the file object's filename defines its persistent id for
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

class FileHeader(object):
    """ Surf file header. Takes an open file, parses in from current file
    pointer position, stores header fields as attribs
    """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 
        self.offset = f.tell()
        self.FH_rec_type, = struct.unpack('B', f.read(1)) # must be 1
        assert self.FH_rec_type == 1
        self.FH_rec_type_ext, = struct.unpack('B', f.read(1)) # must be 0
        assert self.FH_rec_type_ext == 0
        self.UFF_name = f.read(UFF_NAME_LEN).rstrip(NULL) # must be 'UFF'
        assert self.UFF_name == 'UFF'


        # major UFF ver
        self.UFF_major, = struct.unpack('B', f.read(1))
        
        # minor UFF ver
        self.UFF_minor, = struct.unpack('B', f.read(1))
       
        # FH record length in bytes
        self.FH_rec_len, = struct.unpack('H', f.read(2))
        
        # DBRD record length in bytes
        self.DRDB_rec_len, = struct.unpack('H', f.read(2))
        
        # 2 bi-directional seeks format
        self.bi_di_seeks, = struct.unpack('H', f.read(2))
        
        # OS name, ie "WINDOWS 2000"
        self.OS_name = f.read(UFF_OSNAME_LEN).rstrip(NULL)
        self.OS_major, = struct.unpack('B', f.read(1)) # OS major rev
        self.OS_minor, = struct.unpack('B', f.read(1)) # OS minor rev


        # creation time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes
        self.create = TimeDate(f)
        self.create.parse()
        
        # last append time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes,
        # although this tends to be identical to creation time for some reason
        self.append = TimeDate(f)
        self.append.parse()         

        # system node name - same as BDT
        self.node = f.read(UFF_NODENAME_LEN).rstrip(NULL)

        # device name - same as BDT
        self.device = f.read(UFF_DEVICE_LEN).rstrip(NULL)

        # path name
        self.path = f.read(UFF_PATH_LEN).rstrip(NULL)

        # original file name at creation
        self.filename = f.read(UFF_FILENAME_LEN).rstrip(NULL)         
        
        # pad area to bring uff area to 512
        self.pad = f.read(UFF_FH_PAD_LEN).replace(NULL, ' ')
        
        # application task name & version
        self.app_info = f.read(UFF_APPINFO_LEN).rstrip(NULL)         
        
        # user's name as owner of file
        self.user_name = f.read(UFF_OWNER_LEN).rstrip(NULL)
        
        # description of file/exp
        self.file_desc = f.read(UFF_FILEDESC_LEN).rstrip(NULL)         

        # non user area of UFF header should be 512 bytes
        assert f.tell() - self.offset == 512 
        
        # additional user area
        self.user_area = struct.unpack('B'*UFF_FH_USERAREA_LEN, 
                                        f.read(UFF_FH_USERAREA_LEN))         
        assert f.tell() - self.offset == UFF_FILEHEADER_LEN
        
        # this is the end of the original UFF header I think, 
        # the next two fields seem to have been added on to the end by Tim:
        # record type, must be 1 BIDIRECTIONAL SUPPORT
        self.bd_FH_rec_type, = struct.unpack('B', f.read(1))
        assert self.bd_FH_rec_type == 1

        # record type extension, must be 0 BIDIRECTIONAL SUPPORT
        self.bd_FH_rec_type_ext, = struct.unpack('B', f.read(1))
        assert self.bd_FH_rec_type_ext == 0
        
        # total length is 2050 bytes
        self.length = f.tell() - self.offset
        assert self.length == 2050


class TimeDate(object):
    """ TimeDate record, reverse of C'S DateTime """

    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f
        # not really necessary, comment out to save memory
        #self.offset = f.tell()
        self.sec, = struct.unpack('H', f.read(2))
        self.min, = struct.unpack('H', f.read(2))
        self.hour, = struct.unpack('H', f.read(2))
        self.day, = struct.unpack('H', f.read(2))
        self.month, = struct.unpack('H', f.read(2))
        self.year, = struct.unpack('H', f.read(2))
        self.junk = struct.unpack('B'*6, f.read(6)) # junk data at end


class DRDB(object):
    """ Data Record Descriptor Block, aka UFF_DATA_REC_DESC_BLOCK 
    in SurfBawd 
    """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 
        self.offset = f.tell()
        
        # record type; must be 2
        self.DRDB_rec_type, = struct.unpack('B', f.read(1))

        # SurfBawd uses this to detect that it's passed the last DRDB, 
        # not exactly failsafe...
        if self.DRDB_rec_type != 2: 
            raise DRDBError

        # record type extension
        self.DRDB_rec_type_ext, = struct.unpack('B', f.read(1))

        # Data Record type for DBRD 3-255, ie 'P' or 'V' or 'M', etc..
        # don't know quite why SurfBawd required these byte values, this is 
        # more than the normal set of ASCII chars
        self.DR_rec_type = f.read(1) 
        assert int(struct.unpack('B', self.DR_rec_type)[0]) in range(3, 256)

        # Data Record type ext; ignored
        self.DR_rec_type_ext, = struct.unpack('B', f.read(1)) 

        # Data Record size in bytes, signed, -1 means dynamic
        self.DR_size, = struct.unpack('l', f.read(4)) 

        # Data Record name
        self.DR_name = f.read(UFF_DRDB_NAME_LEN).rstrip(NULL) 

        # number of sub-fields in Data Record
        self.DR_num_fields, = struct.unpack('H', f.read(2)) 

        # pad bytes for expansion
        self.DR_pad = struct.unpack('B'*UFF_DRDB_PAD_LEN, 
                        f.read(UFF_DRDB_PAD_LEN)) 

        # sub fields desc. RSFD = Record Subfield Descriptor
        self.DR_subfields = [] 
        for rsfdi in xrange(UFF_RSFD_PER_DRDB):
            rsfd = RSFD(f)
            rsfd.parse()
            self.DR_subfields.append(rsfd)
        assert f.tell() - self.offset == UFF_DRDB_BLOCK_LEN

        # this is the end of the original DRDB I think, the next two fields
        # seem to have been added on to the end by Tim:
        
        # hack to skip past unexplained extra 156 bytes (happens to equal
        # 6*RSFD.length)
        f.seek(156, 1) 

        # record type; must be 2 BIDIRECTIONAL SUPPORT
        self.bd_DRDB_rec_type, = struct.unpack('B', f.read(1)) 
        assert self.bd_DRDB_rec_type == 2

        # record type extension; must be 0 BIDIRECTIONAL SUPPORT
        self.bd_DRDB_rec_type_ext, = struct.unpack('B', f.read(1)) 
        assert self.bd_DRDB_rec_type_ext == 0

        # hack to skip past unexplained extra 2 bytes, sometimes they're 0s,
        # sometimes not
        f.seek(2, 1)

        # total length should be 2050 bytes, but with above skip hacks, 
        # it's 2208 bytes
        self.length = f.tell() - self.offset 
        assert self.length == 2208


class RSFD(object):
    """ Record Subfield Descriptor for Data Record Descriptor Block """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 
        self.offset = f.tell()

        # DRDB subfield name
        self.subfield_name = f.read(UFF_DRDB_RSFD_NAME_LEN).rstrip(NULL) 

        # underlying data type
        self.subfield_data_type, = struct.unpack('H', f.read(2)) 
        
        # sub field size in bytes, signed
        self.subfield_size, = struct.unpack('l', f.read(4)) 

        self.length = f.tell() - self.offset
        assert self.length == 26


class LayoutRecord(object):
    """ Polytrode layout record """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 

        # not really necessary, comment out to save memory
        #self.offset = f.tell() 

        # Record type 'L'
        self.UffType = f.read(1) 

        # hack to skip next 7 bytes
        f.seek(7, 1) 

        # Time stamp, 64 bit signed int
        self.TimeStamp, = struct.unpack('q', f.read(8)) 

        # SURF major version number (2)
        self.SurfMajor, = struct.unpack('B', f.read(1)) 
        # SURF minor version number (1)
        self.SurfMinor, = struct.unpack('B', f.read(1)) 

        # hack to skip next 2 bytes
        f.seek(2, 1)

        # ADC/precision CT master clock frequency (1Mhz for DT3010)
        self.MasterClockFreq, = struct.unpack('l', f.read(4)) 

        # undecimated base sample frequency per channel (25kHz)
        self.BaseSampleFreq, = struct.unpack('l', f.read(4)) 

        # true (1) if Stimulus DIN acquired
        self.DINAcquired, = struct.unpack('B', f.read(1)) 

        # hack to skip next byte
        f.seek(1, 1) 

        # probe number
        self.Probe, = struct.unpack('h', f.read(2)) 

        # =E,S,C for epochspike, spikestream, or continuoustype
        self.ProbeSubType = f.read(1) 

        # hack to skip next byte
        f.seek(1, 1) 

        # number of channels in the probe (54, 1)
        self.nchans, = struct.unpack('h', f.read(2)) 

        # number of samples displayed per waveform per channel (25, 100)
        self.pts_per_chan, = struct.unpack('h', f.read(2)) 

        # hack to skip next 2 bytes
        f.seek(2, 1) 

        # {n/a to cat9} total number of samples per file buffer for this probe
        # (redundant with SS_REC.NumSamples) (135000, 100)
        self.pts_per_buffer, = struct.unpack('l', f.read(4)) 

        # pts before trigger (7)
        self.trigpt, = struct.unpack('h', f.read(2)) 

        # Lockout in pts (2)
        self.lockout, = struct.unpack('h', f.read(2)) 

        # A/D board threshold for trigger (0-4096)
        self.threshold, = struct.unpack('h', f.read(2)) 

        # A/D sampling decimation factor (1, 25)
        self.skippts, = struct.unpack('h', f.read(2)) 

        # S:H delay offset for first channel of this probe (1)
        self.sh_delay_offset, = struct.unpack('h', f.read(2)) 

        # hack to skip next 2 bytes
        f.seek(2, 1) 

        # A/D sampling frequency specific to this probe (ie. after decimation,
        # if any) (25000, 1000)
        self.sampfreqperchan, = struct.unpack('l', f.read(4)) 

        # us, store it here for convenience
        self.tres = int(round(1 / float(self.sampfreqperchan) * 1e6)) 

        # MOVE BACK TO AFTER SHOFFSET WHEN FINISHED WITH CAT 9!!! added May 21,
        # 1999 - only the first self.nchans are filled (5000), the rest are
        # junk values that pad to 64 channels
        self.extgain = struct.unpack('H'*SURF_MAX_CHANNELS, 
                                        f.read(2*SURF_MAX_CHANNELS)) 
        # throw away the junk values
        self.extgain = self.extgain[:self.nchans] 

        # A/D board internal gain (1,2,4,8) <--MOVE BELOW extgain after
        # finished with CAT9!!!!!
        self.intgain, = struct.unpack('h', f.read(2)) 

        # (0 to 53 for highpass, 54 to 63 for lowpass, + junk values that pad
        # to 64 channels) v1.0 had chanlist to be an array of 32 ints.  Now it
        # is an array of 64, so delete 32*4=128 bytes from end
        self.chanlist = struct.unpack('h'*SURF_MAX_CHANNELS, 
                                        f.read(2 * SURF_MAX_CHANNELS)) 

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
        self.pad = struct.unpack(str(880-4-2)+'B', f.read(880-4-2)) 
        
        # hack to skip next 6 bytes, or perhaps self.pad should be 4+2 longer
        f.seek(6, 1) 

class ProbeWinLayout(object):
    """ Probe window layout """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 
        # not really necessary, comment out to save memory 
        #self.offset = f.tell() 

        self.left, = struct.unpack('l', f.read(4))
        self.top, = struct.unpack('l', f.read(4))
        self.width, = struct.unpack('l', f.read(4))
        self.height, = struct.unpack('l', f.read(4))

class MessageRecord(object):
    """Message record"""
    def __init__(self, f):
        self.f = f
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
        self.TimeStamp, = struct.unpack('q', f.read(8)) 

        # 8 bytes -- double - number of days (integral and fractional) since 30
        # Dec 1899
        self.DateTime, = struct.unpack('d', f.read(8)) 

        # 4 bytes -- length of the msg string
        self.MsgLength, = struct.unpack('l', f.read(4)) 
        
        # any length message {shortstring - for cat9!!!}
        self.Msg = f.read(self.MsgLength) 

'''
class PolytrodeRecord(object):
    """Base Polytrode record"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        #self.offset = f.tell() # not really necessary, comment out to save memory
        f.seek(8, 1) # for speed and memory, skip reading the UffType and SubType
        #self.UffType = f.read(1) # {1 byte} {SURF_PT_REC_UFFTYPE}: 'P'

class ContinuousRecord(PolytrodeRecord):
    """Continuous waveform record"""
    def __init__(self, f):
        super(ContinuousRecord, self).__init__(f)
    def parse(self):
        super(ContinuousRecord, self).parse()
        f = self.f # abbrev
        # for speed and memory, skip reading the UffType and SubType
        #self.SubType = f.read(1) # {1 byte} {SURF_PT_REC_UFFTYPE}: 'S' for spikestream (highpass), 'C' for other continuous (lowpass)}
        #f.seek(6, 1) # hack to skip next 6 bytes
        self.TimeStamp, = struct.unpack('q', f.read(8)) # Time stamp, 64 bit signed int
        self.Probe, = struct.unpack('h', f.read(2)) # {2 bytes -- the probe number}
        f.seek(2, 1) # hack to skip next 2 bytes - guessing this should be before the CRC????????????????????????????????????
        self.CRC32, = struct.unpack('l', f.read(4)) # {4 bytes -- PKZIP-compatible CRC} - is this always 0??????????????????????
        self.NumSamples, = struct.unpack('l', f.read(4)) # {4 bytes -- the # of samples in this file buffer record}
        self.waveformoffset = f.tell()
        f.seek(self.NumSamples*2, 1) # skip the waveform data for now
    def load(self):
        """Loads waveform data for this continuous record, assumes that the
        appropriate probe layout record has been assigned as a .layout attrib"""
        f = self.f # abbrev
        f.seek(self.waveformoffset)
        self.waveform = np.asarray(struct.unpack(str(self.NumSamples)+'h', f.read(2*self.NumSamples)), dtype=np.int16) # {ADC Waveform type; dynamic array of SHRT (signed 16 bit)} - converted to an ndarray
        self.waveform = self.waveform.reshape(self.layout.nchans, -1) # reshape to have nchans rows, as indicated in layout
'''

class ContinuousRecord(object):
    """ Continuous waveform record """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 

        # for speed and memory, read all 28 bytes at a time, skip reading
        # UffType, SubType, and CRC32 (which is always 0 anyway?)

        junk, self.TimeStamp, self.Probe, junk, junk, self.NumSamples = \
                                            struct.unpack('qqhhll', f.read(28))
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
        self.waveform = np.asarray(struct.unpack(str(self.NumSamples)+'h', 
                        f.read(2*self.NumSamples)), dtype=np.int16) 

        # reshape to have nchans rows, as indicated in layout
        self.waveform = self.waveform.reshape(self.layout.nchans, -1) 

class HighPassRecord(ContinuousRecord):
    """High-pass continuous waveform record"""
    pass

class LowPassRecord(ContinuousRecord):
    """Low-pass continuous waveform record"""
    pass

class LowPassMultiChanRecord(object):
    """ Low-pass multichannel (usually 10) continuous waveform record """
    def __init__(self, lowpassrecords):
        """ Takes several low pass records, all at the same timestamp """
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


'''
class EpochRecord(PolytrodeRecord):
    """Epoch waveform record, currently unsupported"""
    def __init__(self, f):
        super(EpochRecord, self).__init__(f)
'''


class DisplayRecord(object):
    """ Stimulus display header record """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 
        # not really necessary, comment out to save memory
        #self.offset = f.tell() 

        # 1 byte -- SURF_DSP_REC_UFFTYPE = 'D'
        self.UffType = f.read(1) 

        # hack to skip next 7 bytes
        f.seek(7, 1) 

        # Cardinal, 64 bit signed int
        self.TimeStamp, = struct.unpack('q', f.read(8)) 
    
        # double, 8 bytes - number of days (integral and fractional) since 30
        # Dec 1899
        self.DateTime, = struct.unpack('d', f.read(8)) 
        self.Header = StimulusHeader(f)
        self.Header.parse()

        # hack to skip next 4 bytes
        f.seek(4, 1) 

class StimulusHeader(object):
    """ Stimulus display header """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 
        
        # not really necessary, comment out to save memory
        #self.offset = f.tell() 

        # always 'DS'?
        self.header = f.read(2).rstrip(NULL) 
        self.version, = struct.unpack('H', f.read(2))
        self.filename = f.read(STIMULUS_HEADER_FILENAME_LEN).rstrip(NULL)

        # NVS binary header, array of single floats
        self.parameter_tbl = list(struct.unpack('f'*NVS_PARAM_LEN, 
                                                    f.read(4*NVS_PARAM_LEN))) 

        for parami, param in enumerate(self.parameter_tbl):
            if str(param) == '1.#QNAN':
                # replace 'Quiet NAN' floats with Nones
                self.parameter_tbl[parami] = None 

        # dimstim's text header
        self.python_tbl = f.read(PYTHON_TBL_LEN).rstrip() 

        # cm, single float
        self.screen_width, = struct.unpack('f', f.read(4)) 
        
        # cm
        self.screen_height, = struct.unpack('f', f.read(4)) 

        # cm
        self.view_distance, = struct.unpack('f', f.read(4)) 
    
        # Hz
        self.frame_rate, = struct.unpack('f', f.read(4)) 
        self.gamma_correct, = struct.unpack('f', f.read(4))
        self.gamma_offset, = struct.unpack('f', f.read(4))

        # in seconds
        self.est_runtime, = struct.unpack('H', f.read(2)) 
        self.checksum, = struct.unpack('H', f.read(2))

'''
class SValRecord(object):
    """Single value record"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        #self.offset = f.tell() # not really necessary, comment out to save memory
        f.seek(8, 1) # for speed and memory, skip reading the UffType and SubType
        #self.UffType = f.read(1) # 1 byte -- SURF_SV_REC_UFFTYPE: 'V'
        #self.SubType = f.read(1) # 1 byte -- 'D' digital or 'A' analog
        #f.seek(6, 1) # hack to skip next 6 bytes
        self.TimeStamp, = struct.unpack('q', f.read(8)) # Cardinal, 64 bit signed int

class DigitalSValRecord(SValRecord):
    """Digital single value record"""
    def __init__(self, f):
        super(DigitalSValRecord, self).__init__(f)
    def parse(self):
        super(DigitalSValRecord, self).parse()
        f = self.f # abbrev
        # read 8 bytes at a time for speed:
        self.SVal = struct.unpack('HHL', f.read(8))[0] # 2 bytes -- 16 bit single value
        #self.SVal, = struct.unpack('H', f.read(2)) # 2 bytes -- 16 bit single value
        #f.seek(6, 1) # hack to skip next 6 bytes
'''

class DigitalSValRecord(object):
    """ Digital single value record """
    def __init__(self, f):
        self.f = f

    def parse(self):
        f = self.f 

        # for speed and memory, read all 24 bytes at a time, skip UffType and
        # SubType 
        # Cardinal, 64 bit signed int; 16 bit single value
        junk, self.TimeStamp, self.SVal, junk, junk = \
                struct.unpack('QQHHL', f.read(24)) 


class Stream(object):
    """ Streaming object. Maps from timestamps to record index of stream data to
    retrieve the approriate range of waveform data from disk. 
    """
    def __init__(self, records=None, sampfreq=None):
        """ Takes a sorted temporal (not necessarily evenly-spaced, due to
        pauses in recording) sequence of ContinuousRecords: either 
        HighPassRecords or LowPassMultiChanRecords 
        """
        self.records = records

        # array of record timestamps
        self.rts = np.asarray([record.TimeStamp for record in self.records]) 

        if sampfreq == None:
            # use sampfreq of the raw data
            self.sampfreq = records[0].layout.sampfreqperchan 
        else:
            self.sampfreq = sampfreq

    def __len__(self):
        """ Total number of timepoints? Length in time? Interp'd or raw? """
        pass

    def __getitem__(self, key, endinclusive=False):
        """ Called when Stream object is indexed into using [] or with a slice
        object, indicating start and end timepoints in us. Returns the
        corresponding WaveForm object, which has as its attribs the 2D
        multichannel waveform array as well as the timepoints,
        potentially spanning multiple ContinuousRecords
        """

        # for now, accept only slice objects as keys
        assert key.__class__ == slice 

        # find the first and last records corresponding to the slice. If the
        # start of the slice matches a record's timestamp, start with that
        # record. If the end of the slice matches a record's timestamp, end
        # with that record (even though you'll only potentially use the one
        # timepoint from that record, depending on the value of 'endinclusive')
        lorec, hirec = self.rts.searchsorted([key.start, key.stop],
                                                                side='right')

        # we always want to get back at least 1 record (ie records[0:1]). When
        # slicing, we need to do lower bounds checking (don't go less than 0),
        # but not upper bounds checking
        cutrecords = self.records[max(lorec-1, 0):max(hirec, 1)]
        for record in cutrecords:
            try:
                record.waveform
            except AttributeError:
                # to save time, only load the waveform if not already loaded
                record.load() 

        # join all waveforms, returns a copy
        data = np.concatenate([record.waveform for record in cutrecords], 
                                axis=1) 
        try:
            
            # all highpass records should be using the same layout, 
            # use tres from the first one
            tres = cutrecords[0].layout.tres 

        except AttributeError:
            # records are lowpassmulti
            tres = cutrecords[0].tres 

        # build up waveform timepoints, taking into account any time gaps in
        # between records due to pauses in recording
        ts = []
        for record in cutrecords:
            tstart = record.TimeStamp
            # number of timepoints (columns) in this record's waveform
            nt = record.waveform.shape[-1] 
            ts.extend(range(tstart, tstart + nt*tres, tres))

            #del record.waveform 
            # save memory by unloading waveform data from records that 
            # aren't needed anymore
        ts = np.asarray(ts)
        lo, hi = ts.searchsorted([key.start, key.stop])
        data = data[:, lo:hi+endinclusive]
        ts = ts[lo:hi+endinclusive]

        # interp and s+h correct here
        data, ts = self.interp(data, ts, self.sampfreq) 

        # return a WaveForm object
        return WaveForm(data=data, ts=ts, sampfreq=self.sampfreq) 

    def interp(self, data, ts, sampfreq=None, kind='nyquist'):
        """ Returns interpolated and sample-and-hold corrected data and 
        timepoints, at the given sample frequency
        """
        if kind == 'nyquist':
            # do Nyquist interpolation and S+H correction here
            # find a scipy function that'll do Nyquist interpolation?
            # TODO: Implement this!
            return data, ts
        else:
            raise ValueError, 'Unknown kind of interpolation %r' % kind

    def plot(self, chanis=None, trange=None):
        """ Creates a simple matplotlib plot of the specified chanis over 
        trange
        """
        # wouldn't otherwise need this
        import pylab as pl 
        from pylab import get_current_fig_manager as gcfm

        try: 
            # see if neuropy is available
            from neuropy.Core import lastcmd, neuropyScalarFormatter, \
                                                        neuropyAutoLocator
        except ImportError:
            pass

        if chanis == None:
            # all high pass records should have the same chanlist
            if self.records[0].__class__ == HighPassRecord: 
                chanis = self.records[0].layout.chanlist
                # same goes for lowpassmultichanrecords, each has its own 
                # set of chanis, derived previously from multiple 
                # single layout.chanlists
            elif self.records[0].__class__ == LowPassMultiChanRecord: 
                chanis = self.records[0].chanis
            else:
                raise ValueError, 'unknown record type %s in self.records' % \
                                                    self.records[0].__class__
        nchans = len(chanis)

        if trange == None:
            trange = (self.rts[0], self.rts[0]+100000)

        # make a waveform object
        wf = self[trange[0]:trange[1]] 
        figheight = 1.25+0.2*nchans
        self.f = pl.figure(figsize=(16, figheight))
        self.a = self.f.add_subplot(111)

        try:
            gcfm().frame.SetTitle(lastcmd())
        except NameError:
            pass

        try:

            # better behaved tick label formatter
            self.formatter = neuropyScalarFormatter() 

            # use a thousands separator
            self.formatter.thousandsSep = ',' 

            # better behaved tick locator
            self.a.xaxis.set_major_locator(neuropyAutoLocator()) 
            self.a.xaxis.set_major_formatter(self.formatter)

        except NameError:
            pass
        for chanii, chani in enumerate(chanis):
            # upcast to int32 to prevent int16 overflow
            self.a.plot(wf.ts/1e3, 
                    (np.int32(wf.data[chanii])-2048+500*chani)/500., '-', 
                                                            label=str(chani)) 
        #self.a.legend()
        self.a.set_xlabel('time (ms)')
        self.a.set_ylabel('channel id')

        # assumes chanis are sorted
        self.a.set_ylim(chanis[0]-1, chanis[-1]+1) 
        bottominches = 0.75
        heightinches = 0.15+0.2*nchans
        bottom = bottominches / figheight
        height = heightinches / figheight
        self.a.set_position([0.035, bottom, 0.94, height])


class WaveForm(object):
    """ Waveform object, has data, timestamps and sample frequency attribs """
    def __init__(self, data=None, ts=None, sampfreq=None):
        self.data = data
        # array of timestamps, one for each sample (column) in self.data
        self.ts = ts 
        self.sampfreq = sampfreq


class Fat(object):
    """Stores all the stuff to be pickled into a .parse file and then 
    unpickled as saved parse info
    """
    # TODO
    pass

'''
class HighPassStream(Stream): # or call this SpikeStream?
    def __init__(self, records=None, sampfreq=DEFAULTINTERPSAMPFREQ):
        super(HighPassStream, self).__init__(records=records, sampfreq=sampfreq)


class LowPassStream(Stream): # or call this LFPStream?
    def __init__(self, records=None), sampfreq=None:
        super(LowPassStream, self).__init__(records=records, sampfreq=sampfreq)
'''

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
    Useful when you want to iterate over an object (like in a for loop),
    and you don't want to have to do type checking or handle exceptions
    when the object isn't a sequence"""
    if iterable(x):
        return x
    else:
        return [x]


if __name__ == '__main__':
    # insert unittests here
    pass

