"""Handles parsing and streaming of .srf files"""

'''from SurfPublicTypes.pas:
  SHRT   = SmallInt;{2 bytes} // signed short (from DTxPascal.pas)
  LNG    = LongInt;{4 bytes}  // signed long  (from DTxPascal.pas)

LongInt is guaranteed to be 4 bytes

'''


'''from UFFTYPES.pas:

(* File Header definitions - denotes the size of fields in BYTES *)
   UFF_FH_REC_TYPE      = 1;
   UFF_DRDB_REC_TYPE    = 2;
   UFF_MIN_REC_TYPE     = 3;

   UFF_FILEHEADER_LEN   = 2048;
   UFF_NAME_LEN         = 10;
   UFF_OSNAME_LEN       = 12;
   UFF_NODENAME_LEN     = 32;
   UFF_DEVICE_LEN       = 32;
   UFF_PATH_LEN         = 160;
   UFF_FILENAME_LEN     = 32;
   UFF_FH_PAD_LEN       = 76;
   UFF_APPINFO_LEN      = 32;
   UFF_OWNER_LEN        = 14;
   UFF_FILEDESC_LEN     = 64;
   UFF_FH_USERAREA_LEN  = UFF_FILEHEADER_LEN - 512;

(* Data Record Descriptor Block definitions - field sizes in BYTES *)
   UFF_DRDB_BLOCK_LEN    = 2048;
   UFF_DRDB_NAME_LEN     = 20;
   UFF_DRDB_PAD_LEN  = 16;
   UFF_DRDB_RSFD_NAME_LEN= 20;
   UFF_RSFD_PER_DRDB     = 77;

TYPE
   TIMEDATE = record   (* reverse of C'S DATETIME *)
     Sec,Min,Hour,Day,Month,Year : WORD;
     junk : array[0..5] of BYTE;
   end;

   UFF_FILE_HEADER = record
      FH_rec_type                       : BYTE;  // 1 must be 1
      FH_rec_type_ext                   : BYTE;  // 1 must be 0
      UFF_name : array[0..UFF_NAME_LEN-1] of CHAR; // 10 must be "UFF" sz
      UFF_major                         : BYTE;  // 1 major UFF ver
      UFF_minor                         : BYTE;  // 1 minor UFF ver
      FH_rec_len                        : WORD;  // 2 FH record length in bytes
      DRDB_rec_len                      : WORD;  // 2 DBRD record length in bytes
      bi_di_seeks                       : WORDBOOL; // 2 bi-directional seeks format
      OS_name : array[0..UFF_OSNAME_LEN-1] of CHAR;  // 12 OS name ie "MS-DOS"
      OS_major                          : BYTE;  // 1 OS major rev
      OS_minor                          : BYTE;  // 1 OS minor rev
      create                            : TIMEDATE; // 18 creation time & date
      append                            : TIMEDATE; // 18 last append time & date
      node      : array[0..UFF_NODENAME_LEN-1]    of CHAR;  // 32 system node name - same as BDT
      device    : array[0..UFF_DEVICE_LEN-1]      of CHAR;  // 32 device name - same as BDT
      path      : array[0..UFF_PATH_LEN-1]        of CHAR;  // 160 path name
      filename  : array[0..UFF_FILENAME_LEN-1]    of CHAR;  // 32 original file name at creation
      pad       : array[0..UFF_FH_PAD_LEN-1]      of CHAR;  // 76 pad area to bring uff area to 512
      app_info  : array[0..UFF_APPINFO_LEN-1]     of CHAR;  // 32 application task name & version
      user_name : array[0..UFF_OWNER_LEN-1]       of CHAR;  // 14 user's name as owner of file
      file_desc : array[0..UFF_FILEDESC_LEN-1]    of CHAR;  // 64 description of file/exp
      user_area : array[0..UFF_FH_USERAREA_LEN-1] of BYTE;  // 1536 additional user area
      bd_FH_rec_type                    : BYTE;     // record type; must be 1 BIDIRECTIONAL SUPPORT
      bd_FH_rec_type_ext                : BYTE;     // record type extension; must be 0 BIDIRECTIONAL SUPPORT
                                          {total = 2048 bytes}
   end;

   drdbrsfname = array[1..UFF_DRDB_RSFD_NAME_LEN] of CHAR;

   UFF_DRDB_RSFD = record
{20}  subfield_name        : drdbrsfname; (* sz DRDB subfield name*)
{22}  subfield_data_type   : WORD;    (* underlying data type *)
{26}  subfield_size        : LONGINT; (* sub field size in bytes *)
   end;
{77*26=2002}
   UFF_DATA_REC_DESC_BLOCK = record
{1}   DRDB_rec_type                       : BYTE;    (* record type; must be 2 *)
{2}   DRDB_rec_type_ext                   : BYTE;    (* record type extension *)
{3}   DR_rec_type                         : CHAR;    (* Data Record type for DBRD 3-255 *)
{4}   DR_rec_type_ext                     : BYTE;    (* Data Record type ext; ignored *)
{8}   DR_size                             : LONGINT; (* Data Record size in bytes *)
{28}  DR_name:array[0..UFF_DRDB_NAME_LEN-1] of CHAR; (* Data Record name *)
{30}  DR_num_fields                       : WORD;    (* number of sub-fields in Data Record*)
{46}  DR_pad : array[0..UFF_DRDB_PAD_LEN-1] of BYTE; (* pad bytes for expansion *)
{2048}DR_subfields : array[1..UFF_RSFD_PER_DRDB] of UFF_DRDB_RSFD; (* sub fields desc *)
{2049}bd_DRDB_rec_type                     : BYTE;   (* record type; must be 2 BIDIRECTIONAL SUPPORT*)
{2050}bd_DRDB_rec_type_ext                 : BYTE;   (* record type extension; must be 0 BIDIRECTIONAL SUPPORT*)
   end;
'''

'''from SurfPublicTypes.pas:
   SURF_PT_REC_UFFTYPE      = 'P'; //Polytrode records for spike, continuous spike & continuous recordings
     SPIKEEPOCH             = 'E'; //was 'P', original SURF type, changed from 'S' to 'E' June 2002 tjb
     SPIKESTREAM            = 'S'; //continuous stream to disk
     CONTINUOUS             = 'C'; //all other, non-spike continuous records (eg. EEG)
   SURF_SV_REC_UFFTYPE      = 'V'; //Single value record...
     SURF_DIGITAL           = 'D'; //...from the digital ports
     SURF_ANALOG            = 'A'; //...from an analog channel
   SURF_PL_REC_UFFTYPE      = 'L'; //Polytrode layout record
   SURF_MSG_REC_UFFTYPE     = 'M'; //Message record...
     USER_MESSAGE           = 'U'; //...generated by user
     SURF_MESSAGE           = 'S'; //...generated by Surf
   SURF_DSP_REC_UFFTYPE     = 'D'; //Stimulus display parameter header record

   NVS_PARAM_LEN            = 749;
   PYTHON_TBL_LEN           = 50000;

  TChanList = array[0..SURF_MAX_CHANNELS-1] of SHRT;

  TProbeWinLayout = record
    left,top,width,height : integer;
  end;


'''
'''from SurfTypes.pas:
  SURF_LAYOUT_REC = record { Type for all probe layout records }
    UffType         : CHAR; // Record type 'L'
    TimeStamp       : INT64;// Time stamp, 64 bit signed int
    SurfMajor       : BYTE; // SURF major version number
    SurfMinor       : BYTE; // SURF minor version number
    MasterClockFreq : LNG;  // ADC/precision CT master clock frequency (1Mhz for DT3010)
    BaseSampleFreq  : LNG;  // undecimated base sample frequency per channel
    DINAcquired     : Boolean; //true if Stimulus DIN acquired

    Probe          : SHRT; // probe number
    ProbeSubType   : CHAR; // =E,S,C for epochspike, spikestream, or continuoustype
    nchans         : SHRT; // number of channels in the probe
    pts_per_chan   : SHRT; // number of samples per waveform per channel (display)
    pts_per_buffer : LNG;  // {n/a to cat9} total number of samples per file buffer for this probe (redundant with SS_REC.NumSamples)
    trigpt         : SHRT; // pts before trigger
    lockout        : SHRT; // Lockout in pts
    threshold      : SHRT; // A/D board threshold for trigger
    skippts        : SHRT; // A/D sampling decimation factor
    sh_delay_offset: SHRT; // S:H delay offset for first channel of this probe
    sampfreqperchan: LNG;  // A/D sampling frequency specific to this probe (ie. after decimation, if any)
    extgain        : array[0..SURF_MAX_CHANNELS-1] of WORD; // MOVE BACK TO AFTER SHOFFSET WHEN FINISHED WITH CAT 9!!! added May 21'99
    intgain        : SHRT; // A/D board internal gain <--MOVE BELOW extgain after finished with CAT9!!!!!
    chanlist       : TChanList; //v1.0 had chanlist to be an array of 32 ints.  Now it is an array of 64, so delete 32*4=128 bytes from end
    probe_descrip  : ShortString;
    electrode_name : ShortString;
    ProbeWinLayout : TProbeWinLayout; //MOVE BELOW CHANLIST FOR CAT 9 v1.0 had ProbeWinLayout to be 4*32*2=256 bytes, now only 4*4=16 bytes, so add 240 bytes of pad
    pad            : array[0..879 {remove for cat 9!!!-->}- 4{pts_per_buffer} - 2{SHOffset}] of BYTE; {pad for future expansion/modification}
  end;

  SURF_MSG_REC = record // Message record
    UffType    : char; //1 byte -- SURF_MSG_REC_UFFTYPE
    SubType    : char; //1 byte -- 'U' user or 'S' Surf-generated
    TimeStamp  : INT64; //Cardinal, 64 bit signed int
    DateTime   : TDateTime; //8 bytes -- double
    MsgLength  : integer;//4 bytes -- length of the msg string
    Msg        : string{shortstring - for cat9!!!}; //any length message
  end;

  SURF_SS_REC    = record // SpikeStream record
    UffType      : char;    {1 byte} {SURF_PT_REC_UFFTYPE}
    SubType      : char;    {1 byte} {=E,S,C for spike epoch, continuous spike or other continuous }
    TimeStamp    : INT64;   {Cardinal, 64 bit signed int}
    Probe        : shrt;    {2 bytes -- the probe number}
    CRC32        : {u}LNG;  {4 bytes -- PKZIP-compatible CRC}
    NumSamples   : integer; {4 bytes -- the # of samples in this file buffer record}
    ADCWaveform  : TWaveForm{ADC Waveform type; dynamic array of SHRT (signed 16 bit)}
  end;



'''
'''
Tim's .fat format:
    first the surf file header?
    for buffer in buffers:
        stream flag (PS or PC or VD or MS or MU) (2 bytes)
        stream id (int16, 0 for 1st stream, 1 for 2nd, etc)
        buffer id (int32) or value id if flag==VD
        timestamp (int64)
        == total of 16 bytes per buffer
'''

'''
ad-hoc delphi packing rules:
    - single CHARs: pack 8 bytes at a time? if several consecutive, pack within the 8 bytes?
    - arrays of chars are the length you'd expect
'''

import numpy as np
import os
import cPickle
import struct

# Surf file header constants, field sizes in bytes
UFF_FILEHEADER_LEN = 2048 # 'UFF' == 'Universal File Format'
UFF_NAME_LEN = 10
UFF_OSNAME_LEN = 12
UFF_NODENAME_LEN = 32
UFF_DEVICE_LEN = 32
UFF_PATH_LEN = 160
UFF_FILENAME_LEN = 32
UFF_FH_PAD_LEN = 76  # pad area to bring uff area to 512, this really should be calculated, not hard-coded
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
SURF_MAX_CHANNELS = 64 # currently supports one or two DT3010 boards, could be higher


class SurfFile(object): # or should I inherit from file?
    """Opens a .srf file and exposes all of its data as attribs.
    If no synonymous .parse file exists, parses the .srf file, stores parsing in a .parse file.
    Stores as attribs: srf header, electrode layout records, high and low pass stream data, stimulus display headers, stimulus digital vals, message data"""
    def __init__(self, fname='C:/data/Cat 15/81 - track 7c mseq16exps.srf'):
        self.f = file(fname, 'rb')

    def close(self):
        self.f.close()

    def parse(self):
        fatfname = os.path.splitext(self.f.name)[0] + '.pickled.fat'
        try: # recover SurfFat object pickled in .fat file
            ff = file(fatfname, 'rb')
            u = cPickle.Unpickler(ff)
            fat = u.load()
            ff.close()
            self.fat = fat
            print 'Recovered fat from %r' % fatfname
        except: # .fat file doesn't exist, or something's wrong with it. Parse the .srf file
            print 'Parsing %r' % self.f.name
            f = self.f # abbrev
            f.seek(0) # make sure we're at the start of the srf file before trying to parse it

            # Parse the Surf header
            self.surfheader = SurfHeader(f)
            self.surfheader.parse()

            # Parse the DRDBs (Data Record Descriptor Blocks)
            self.drdbs = []
            while 1:
                drdb = DRDB(f)
                try:
                    drdb.parse()
                    self.drdbs.append(drdb)
                except DRDBError: # we've gone past the last DRDB
                    f.seek(drdb.offset) # set file pointer back to where we were
                    break

            # Parse all the records in the file, but don't load any waveforms
            self.layoutrecords = []
            self.messagerecords = []
            self.highpassrecords = []
            self.lowpassrecords = []
            while 1:
                flag = f.read(2) # returns empty string when EOF is reached
                if flag == '':
                    break
                f.seek(-2, 1) # go back
                if flag[0] == 'L': # polytrode layout record
                    layoutrecord = LayoutRecord(f)
                    layoutrecord.parse()
                    self.layoutrecords.append(layoutrecord)
                elif flag[0] == 'M': # message record
                    messagerecord = MessageRecord(f)
                    messagerecord.parse()
                    self.messagerecords.append(messagerecord)
                elif flag[0] == 'P': # polytrode waveform record
                    if flag[1] == 'S': # spike stream (highpass) record
                        highpassrecord = HighPassRecord(f)
                        highpassrecord.parse()
                        self.highpassrecords.append(highpassrecord)
                    elif flag[1] == 'C': # continuous (lowpass) record
                        lowpassrecord = LowPassRecord(f)
                        lowpassrecord.parse()
                        self.lowpassrecords.append(lowpassrecord)
                    elif flag[1] == 'E': # spike epoch record
                        raise ValueError, 'Spike epochs (non-continous) recordings currently unsupported'
                else:
                    raise ValueError, 'Unexpected flag %r at offset %d' % (flag, f.tell())


            # Make sure to sort each record type list according to timestamp...

            # Create a fat for mapping between stream position and file position
            fat = SurfFat()
            fat.surfheader = self.surfheader
            fat.drdbs = self.drdbs
            # Don't save to .fat for now, to make testing easier...
            #ff = file(fatfname, 'wb')
            #p = cPickle.Pickler(ff, protocol=-1) # make a Pickler, use most efficient (least readable) protocol
            #p.dump(fat) # pickle fat to file
            #ff.close()
        self.fat = fat

class SurfHeader(object):
    """Surf file header. Takes an open file, parses in from current file pointer position,
    stores header fields as attibs"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        self.offset = f.tell()
        self.FH_rec_type, = struct.unpack('B', f.read(1)) # must be 1
        assert self.FH_rec_type == 1
        self.FH_rec_type_ext, = struct.unpack('B', f.read(1)) # must be 0
        assert self.FH_rec_type_ext == 0
        self.UFF_name = f.read(UFF_NAME_LEN).rstrip('\x00') # must be 'UFF'
        assert self.UFF_name == 'UFF'
        self.UFF_major, = struct.unpack('B', f.read(1)) # major UFF ver
        self.UFF_minor, = struct.unpack('B', f.read(1)) # minor UFF ver
        self.FH_rec_len, = struct.unpack('H', f.read(2)) # FH record length in bytes
        self.DRDB_rec_len, = struct.unpack('H', f.read(2)) # DBRD record length in bytes
        self.bi_di_seeks, = struct.unpack('H', f.read(2)) # 2 bi-directional seeks format
        self.OS_name = f.read(UFF_OSNAME_LEN).rstrip('\x00') # OS name, ie "WINDOWS 2000"
        self.OS_major, = struct.unpack('B', f.read(1)) # OS major rev
        self.OS_minor, = struct.unpack('B', f.read(1)) # OS minor rev
        self.create = TimeDate(f)
        self.create.parse() # creation time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes
        self.append = TimeDate(f)
        self.append.parse() # last append time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes, although this tends to be identical to creation time for some reason
        self.node = f.read(UFF_NODENAME_LEN).rstrip('\x00') # system node name - same as BDT
        self.device = f.read(UFF_DEVICE_LEN).rstrip('\x00') # device name - same as BDT
        self.path = f.read(UFF_PATH_LEN).rstrip('\x00') # path name
        # this has some kinda problem in .srf file '81 - track 7c mseq16exps.srfcula'
        self.filename = f.read(UFF_FILENAME_LEN).rstrip('\x00') # original file name at creation
        self.pad = f.read(UFF_FH_PAD_LEN).replace('\x00', ' ') # pad area to bring uff area to 512
        self.app_info = f.read(UFF_APPINFO_LEN).rstrip('\x00') # application task name & version
        self.user_name = f.read(UFF_OWNER_LEN).rstrip('\x00') # user's name as owner of file
        self.file_desc = f.read(UFF_FILEDESC_LEN).rstrip('\x00') # description of file/exp
        assert f.tell() - self.offset == 512 # non user area of UFF header should be 512 bytes
        self.user_area = struct.unpack('B'*UFF_FH_USERAREA_LEN, f.read(UFF_FH_USERAREA_LEN)) # additional user area
        assert f.tell() - self.offset == UFF_FILEHEADER_LEN
        # this is the end of the original UFF header I think, the next two fields seem to have been added on to the end by Tim:
        self.bd_FH_rec_type, = struct.unpack('B', f.read(1)) # record type, must be 1 BIDIRECTIONAL SUPPORT
        assert self.bd_FH_rec_type == 1
        self.bd_FH_rec_type_ext, = struct.unpack('B', f.read(1)) # record type extension, must be 0 BIDIRECTIONAL SUPPORT
        assert self.bd_FH_rec_type_ext == 0
        self.length = f.tell() - self.offset # total length is 2050 bytes
        assert self.length == 2050

class TimeDate(object):
    """TimeDate record, reverse of C'S DateTime"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f
        self.offset = f.tell()
        self.sec, = struct.unpack('H', f.read(2))
        self.min, = struct.unpack('H', f.read(2))
        self.hour, = struct.unpack('H', f.read(2))
        self.day, = struct.unpack('H', f.read(2))
        self.month, = struct.unpack('H', f.read(2))
        self.year, = struct.unpack('H', f.read(2))
        self.junk = struct.unpack('B'*6, f.read(6)) # junk data at end

class DRDB(object):
    """Data Record Descriptor Block, aka UFF_DATA_REC_DESC_BLOCK in SurfBawd"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        self.offset = f.tell()
        self.DRDB_rec_type, = struct.unpack('B', f.read(1)) # record type; must be 2
        if self.DRDB_rec_type != 2: # SurfBawd uses this to detect that it's passed the last DRDB, not exactly failsafe...
            raise DRDBError
        self.DRDB_rec_type_ext, = struct.unpack('B', f.read(1)) # record type extension
        self.DR_rec_type = f.read(1) # Data Record type for DBRD 3-255, ie 'P' or 'V' or 'M', etc..
        assert int(struct.unpack('B', self.DR_rec_type)[0]) in range(3, 256) # don't know quite why SurfBawd required these byte values, this is more than the normal set of ASCII chars
        self.DR_rec_type_ext, = struct.unpack('B', f.read(1)) # Data Record type ext; ignored
        self.DR_size, = struct.unpack('l', f.read(4)) # Data Record size in bytes, signed, -1 means dynamic
        self.DR_name = f.read(UFF_DRDB_NAME_LEN).rstrip('\x00') # Data Record name
        self.DR_num_fields, = struct.unpack('H', f.read(2)) # number of sub-fields in Data Record
        self.DR_pad = struct.unpack('B'*UFF_DRDB_PAD_LEN, f.read(UFF_DRDB_PAD_LEN)) # pad bytes for expansion
        self.DR_subfields = [] # sub fields desc. RSFD = Record Subfield Descriptor
        for rsfdi in range(UFF_RSFD_PER_DRDB):
            rsfd = RSFD(f)
            rsfd.parse()
            self.DR_subfields.append(rsfd)
        assert f.tell() - self.offset == UFF_DRDB_BLOCK_LEN
        # this is the end of the original DRDB I think, the next two fields seem to have been added on to the end by Tim:
        f.seek(156, 1) # hack to skip past unexplained extra 156 bytes (happens to equal 6*RSFD.length)
        self.bd_DRDB_rec_type, = struct.unpack('B', f.read(1)) # record type; must be 2 BIDIRECTIONAL SUPPORT
        assert self.bd_DRDB_rec_type == 2
        self.bd_DRDB_rec_type_ext, = struct.unpack('B', f.read(1)) # record type extension; must be 0 BIDIRECTIONAL SUPPORT
        assert self.bd_DRDB_rec_type_ext == 0
        f.seek(2, 1) # hack to skip past unexplained extra 2 bytes, sometimes they're 0s, sometimes not
        self.length = f.tell() - self.offset # total length should be 2050 bytes, but with above skip hacks, it's 2208 bytes
        assert self.length == 2208

class DRDBError(ValueError):
    """Used to indicate when you've passed the last DRDB at the start of the .srf file"""
    pass

class RSFD(object):
    """Record Subfield Descriptor for Data Record Descriptor Block"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        self.offset = f.tell()
        self.subfield_name = f.read(UFF_DRDB_RSFD_NAME_LEN).rstrip('\x00') # DRDB subfield name
        self.subfield_data_type, = struct.unpack('H', f.read(2)) # underlying data type
        self.subfield_size, = struct.unpack('l', f.read(4)) # sub field size in bytes, signed
        self.length = f.tell() - self.offset
        assert self.length == 26

class LayoutRecord(object):
    """Polytrode layout record"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        self.offset = f.tell()
        self.UffType = f.read(1) # Record type 'L'
        f.seek(7, 1) # hack to skip next 7 bytes
        self.TimeStamp, = struct.unpack('q', f.read(8)) # Time stamp, 64 bit signed int
        self.SurfMajor, = struct.unpack('B', f.read(1)) # SURF major version number (2)
        self.SurfMinor, = struct.unpack('B', f.read(1)) # SURF minor version number (1)
        f.seek(2, 1) # hack to skip next 2 bytes
        self.MasterClockFreq, = struct.unpack('l', f.read(4)) # ADC/precision CT master clock frequency (1Mhz for DT3010)
        self.BaseSampleFreq, = struct.unpack('l', f.read(4)) # undecimated base sample frequency per channel (25kHz)
        self.DINAcquired, = struct.unpack('B', f.read(1)) # true (1) if Stimulus DIN acquired
        f.seek(1, 1) # hack to skip next byte

        self.Probe, = struct.unpack('h', f.read(2)) # probe number
        self.ProbeSubType = f.read(1) # =E,S,C for epochspike, spikestream, or continuoustype
        f.seek(1, 1) # hack to skip next byte
        self.nchans, = struct.unpack('h', f.read(2)) # number of channels in the probe (54)
        self.pts_per_chan, = struct.unpack('h', f.read(2)) # number of samples per waveform per channel (display) (25, 100)
        f.seek(2, 1) # hack to skip next 2 bytes
        self.pts_per_buffer, = struct.unpack('l', f.read(4)) # {n/a to cat9} total number of samples per file buffer for this probe (redundant with SS_REC.NumSamples) (135000, 100)
        self.trigpt, = struct.unpack('h', f.read(2)) # pts before trigger (7)
        self.lockout, = struct.unpack('h', f.read(2)) # Lockout in pts (2)
        self.threshold, = struct.unpack('h', f.read(2)) # A/D board threshold for trigger (0-4096)
        self.skippts, = struct.unpack('h', f.read(2)) # A/D sampling decimation factor (1, 25)
        self.sh_delay_offset, = struct.unpack('h', f.read(2)) # S:H delay offset for first channel of this probe (1)
        f.seek(2, 1) # hack to skip next 2 bytes
        self.sampfreqperchan, = struct.unpack('l', f.read(4)) # A/D sampling frequency specific to this probe (ie. after decimation, if any) (25000, 1000)
        self.extgain = struct.unpack('H'*SURF_MAX_CHANNELS, f.read(2*SURF_MAX_CHANNELS)) # MOVE BACK TO AFTER SHOFFSET WHEN FINISHED WITH CAT 9!!! added May 21, 1999 - only the first self.nchans are filled (5000), the rest are junk values that pad to 64 channels
        self.extgain = self.extgain[:self.nchans] # throw away the junk values
        self.intgain, = struct.unpack('h', f.read(2)) # A/D board internal gain (1,2,4,8) <--MOVE BELOW extgain after finished with CAT9!!!!!
        self.chanlist = struct.unpack('h'*SURF_MAX_CHANNELS, f.read(2*SURF_MAX_CHANNELS)) # (0 to 53 for highpass, 54 to 63 for lowpass, + junk values that pad to 64 channels) v1.0 had chanlist to be an array of 32 ints.  Now it is an array of 64, so delete 32*4=128 bytes from end
        self.chanlist = self.chanlist[:self.nchans] # throw away the junk values
        f.seek(1, 1) # hack to skip next byte
        self.probe_descrip = f.read(255).rstrip('\x00') # ShortString (uMap54_2a, 65um spacing)
        f.seek(1, 1) # hack to skip next byte
        self.electrode_name = f.read(255).rstrip('\x00') # ShortString (uMap54_2a)
        f.seek(2, 1) # hack to skip next 2 bytes
        self.probewinlayout = ProbeWinLayout(f)
        self.probewinlayout.parse() # MOVE BELOW CHANLIST FOR CAT 9 v1.0 had ProbeWinLayout to be 4*32*2=256 bytes, now only 4*4=16 bytes, so add 240 bytes of pad
        self.pad = struct.unpack('B'*(880-4-2), f.read(880-4-2)) # array[0..879 {remove for cat 9!!!-->}- 4{pts_per_buffer} - 2{SHOffset}] of BYTE; {pad for future expansion/modification}
        f.seek(6, 1) # hack to skip next 6 bytes, or perhaps self.pad should be 4+2 longer

    """
    with SurfRecord do
      m_ProbeWaveFormLength[probe]:= nchans * pts_per_chan; //necessary, others here too?
    """

class ProbeWinLayout(object):
    """Probe window layout"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        self.offset = f.tell()
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
        self.offset = f.tell()
        self.UffType = f.read(1) # 1 byte -- SURF_MSG_REC_UFFTYPE: 'M'
        self.SubType = f.read(1) # 1 byte -- 'U' user or 'S' Surf-generated
        f.seek(6, 1) # hack to skip next 6 bytes
        self.TimeStamp, = struct.unpack('q', f.read(8)) # Time stamp, 64 bit signed int
        self.DateTime, = struct.unpack('d', f.read(8)) # 8 bytes -- double - number of days (integral and fractional) since 30 Dec 1899, I think
        self.MsgLength, = struct.unpack('l', f.read(4)) # 4 bytes -- length of the msg string
        self.Msg = f.read(self.MsgLength) # any length message {shortstring - for cat9!!!}

class PolytrodeRecord(object):
    """Base Polytrode record"""
    def __init__(self, f):
        self.f = f
    def parse(self):
        f = self.f # abbrev
        self.offset = f.tell()
        self.UffType = f.read(1) # {1 byte} {SURF_PT_REC_UFFTYPE}: 'P'

class ContinuousRecord(PolytrodeRecord):
    """Continuous waveform recording"""
    def __init__(self, f):
        super(ContinuousRecord, self).__init__(f)
    def parse(self):
        super(ContinuousRecord, self).parse()
        f = self.f # abbrev
        self.SubType = f.read(1) # {1 byte} {SURF_PT_REC_UFFTYPE}: 'S' for spikestream (highpass), 'C' for other continuous (lowpass)}
        f.seek(6, 1) # hack to skip next 6 bytes
        self.TimeStamp, = struct.unpack('q', f.read(8)) # Time stamp, 64 bit signed int
        self.Probe, = struct.unpack('h', f.read(2)) # {2 bytes -- the probe number}
        f.seek(2, 1) # hack to skip next 2 bytes - guessing this should be before the CRC????????????????????????????????????
        self.CRC32, = struct.unpack('l', f.read(4)) # {4 bytes -- PKZIP-compatible CRC} - is this always 0??????????????????????
        self.NumSamples, = struct.unpack('l', f.read(4)) # {4 bytes -- the # of samples in this file buffer record}
        self.waveformoffset = f.tell()
        f.seek(self.NumSamples*2, 1) # skip the waveform data for now
    def load(self):
        """Loads waveform data for this continuous record"""
        f = self.f # abbrev
        f.seek(self.waveformoffset)
        self.waveform = np.asarray(struct.unpack('h'*self.NumSamples, f.read(2*self.NumSamples))) # {ADC Waveform type; dynamic array of SHRT (signed 16 bit)} - convert this to an ndarray - this needs to be reshaped according to sf.layoutrecords[self.Probe].nchans

class HighPassRecord(ContinuousRecord):
    pass

class LowPassRecord(ContinuousRecord):
    pass

class EpochRecord(PolytrodeRecord):
    def __init__(self, f):
        super(EpochRecord, self).__init__(f)





class SurfFat(object): # stores all the stuff to be pickled into a .fat and then unpickled as saved parse info
    def __init__(self):
        pass
        # initing these to None ain't really necessary:
        #self.surfheader = None # srf header
        #self.drdbs = None # list of DRDBs
        # fill in other stuff here

class SurfStream(object): # or should I inherit from np.ndarray?
    """Returns stream data based on mapping between stream index and file position.
    Returns either raw or interpolated, depending on interp attrib"""
    def __init__(self, interp=False):
        self.data = np.random.randint(0, 9, 10)
        self.interp = interp

    def __len__(self):
        """Should this return length in time? Number of data points? Interp'd or raw?"""
        return len(self.data)

    def __getitem__(self, key):
        """Use the fat to decide where the item is in the file, return it according to interp"""
        return self.data.__getitem__(key)
    '''
    def __getslice__(self, i, j): # not necessary? __getitem__ already supports slice objects as keys
        """Use the fat to decide where the ith and j-1th items are in the file, return everything between them"""
        return self.data.__getslice__(i, j)
    '''

class HighPass(SurfStream): # or call this SpikeStream?
    def __init__(self, interp=50000):
        super(HighPass, self).__init__(interp=interp)




class LowPass(SurfStream): # or call this LFPStream?
    def __init__(self, interp=False):
        super(LowPass, self).__init__(interp=interp)

