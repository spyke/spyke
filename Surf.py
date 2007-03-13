"""Handles parsing and streaming of .srf files"""

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
import numpy as np
import os
import cPickle
import struct

# Surf header constants, field sizes in bytes
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
            # starts at offset 2050
            self.drdbs = []
            while 1:
                drdb = DRDB(f)
                try:
                    drdb.parse() # 158 too many extra bytes in file
                    self.drdbs.append(drdb)
                except DRDBError: # we've gone past the last DRDB
                    f.seek(drdb.offset) # set file pointer back to where we were
                    break

            # Now parse whatever comes next...

            # Create a fat for mapping between stream position and file position
            fat = SurfFat()
            fat.surfheader = self.surfheader
            fat.drdbs = self.drdbs
            #ff = file(fatfname, 'wb')
            #p = cPickle.Pickler(ff, protocol=-1) # make a Pickler, use most efficient (least readable) protocol
            #p.dump(fat) # pickle fat to file
            #ff.close()
        self.fat = fat

class SurfHeader(object):
    """Surf file header.
    Takes an open file, parses in from current file pointer position,
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
        self.create = struct.unpack('H'*(18/2), f.read(18)) # creation time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes
        self.append = struct.unpack('H'*(18/2), f.read(18)) # last append time & date: Sec,Min,Hour,Day,Month,Year + 6 junk bytes
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

class DRDB(object):
    """Data Record Descriptor Block, aka UFF_DATA_REC_DESC_BLOCK in SurfBawd.
    Takes an open file, parses in from current file pointer position,
    stores header fields as attibs"""
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
        f.seek(156, 1) # hack to skip past unexplained extra 156 bytes (happens to be 6*RSFD.length)
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
    """Record Subfield Descriptor for Data Record Descriptor Block
    Takes an open file, parses in from current file pointer position,
    stores header fields as attibs"""
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

