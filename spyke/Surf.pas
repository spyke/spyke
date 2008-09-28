//Relevant Delphi records taken from Surf

//from SurfPublicTypes.pas:
  SHRT   = SmallInt;{2 bytes} // signed short (from DTxPascal.pas)
  LNG    = LongInt;{4 bytes}  // signed long  (from DTxPascal.pas)

//LongInt is guaranteed to be 4 bytes




//from UFFTYPES.pas:

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


///from SurfPublicTypes.pas:
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

  TStimulusHeader = {packed}record
    header        : array[0..1] of char;
    version       : word;
    filename      : array[0..63] of char;
    parameter_tbl : array[0..NVS_PARAM_LEN - 1]  of single;
    python_tbl    : array[0..PYTHON_TBL_LEN - 1] of char;  //added by MAS
    screen_width  : single;
    screen_height : single;
    view_distance : single;
    frame_rate    : single;
    gamma_correct : single;
    gamma_offset  : single;
    est_runtime   : word;
    checksum      : word;
  end;



///from SurfTypes.pas:
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

  SURF_DSP_REC = record // Stimulus display header record
    UffType    : char;  //1 byte -- SURF_DSP_REC_UFFTYPE = 'D'
    TimeStamp  : INT64;  //Cardinal, 64 bit signed int
    DateTime   : TDateTime; //double, 8 bytes
    Header     : TStimulusHeader;
  end;

  SURF_SV_REC = record // Single value record
    UffType   : char; //1 byte -- SURF_SV_REC_UFFTYPE
    SubType   : char; //1 byte -- 'D' digital or 'A' analog
    TimeStamp : INT64;//Cardinal, 64 bit signed int
    SVal      : word; //2 bytes -- 16 bit single value
  end;



{Tim's .fat format:
    first the surf file header?
    for buffer in buffers:
        stream flag (PS or PC or VD or MS or MU) (2 bytes)
        stream id (int16, 0 for 1st stream, 1 for 2nd, etc)
        buffer id (int32) or value id if flag==VD
        timestamp (int64)
        == total of 16 bytes per buffer
}

{
ad-hoc delphi packing rules:
    - single CHARs: pack 8 bytes at a time? if several consecutive, pack within the 8 bytes?
    - arrays of chars are the length you'd expect
}

