CONST
   SURF_PT_REC_UFFTYPE      = 'N';  //Polytrode Records for spikes and continuous recordings
     SPIKETYPE      = 'S'; //was 'P'
     CONTINUOUSTYPE = 'C';
   SURF_SV_REC_UFFTYPE      = 'V';  //Single Value Record
     SURF_DIGITAL    = 'D';
   SURF_PL_REC_UFFTYPE      = 'L';  //Probe Layout Record
   SURF_MSG_REC_UFFTYPE     = 'M';  //Message Record

   SURF_MAX_CHANNELS        = 32;
   SURF_MAX_PROBES          = 32;
   SURF_MAX_WAVEFORM_PTS    = 1024;

   SURF_OUT_HANDLE          = 1000;
   SURF_OUT_PROBE           = 1001;
   SURF_OUT_SPIKE           = 1002;
   SURF_OUT_SPIKE_ARRAY     = 1003;
   SURF_OUT_CR              = 1004;
   SURF_OUT_CR_ARRAY        = 1005;
   SURF_OUT_SV              = 1006;
   SURF_OUT_SV_ARRAY        = 1007;
   SURF_OUT_MSG             = 1008;
   SURF_OUT_MSG_ARRAY       = 1009;
   SURF_OUT_SURFEVENT       = 1010;
   SURF_OUT_SURFEVENT_ARRAY = 1011;
   SURF_OUT_FILESTART       = 1012;
   SURF_OUT_FILEEND         = 1013;

   SURF_IN_HANDLE    = 2000;
   SURF_IN_SPIKE     = 2001;
   SURF_IN_CR        = 2002;
   SURF_IN_SV        = 2003;
   SURF_IN_D2A       = 2004;
   SURF_IN_READFILE  = 2005;
   SURF_IN_SAVEFILE  = 2006;

TYPE
  TBufDesc = record
    d1,d2,d3,d4,d5 : integer;
  end;
  //end for SurfBridge only--------------------

  TPoint = record  //graphics
    x,y : integer;
  end;

  SHRT   = SmallInt;{2 bytes} // short (from DTxPascal.pas)
  LNG    = LongInt;{4 bytes}  // long  (from DTxPascal.pas)

  TWaveform = array of SHRT;
  TScreenLayout = array[0..SURF_MAX_CHANNELS-1] of TPoint;
  TChanList = array[0..SURF_MAX_CHANNELS-1] of SHRT;

  //SPIKE DATA ANALYSIS RECORD-----------------------------
  TSpike = record
    time_stamp    : LNG;{4 bytes}
    cluster       : SHRT;{2 bytes}
    EventNum      : Integer;{4 bytes}
    waveform      : array{nchans}of TWaveform;
    param         : array{nparams}of SHRT;
  end;
  TSpikeArray = array of TSpike;

  //CONTINUOUS RECORD ANALYSIS RECORD----------------------
  TCr = record
    time_stamp    : LNG;{4 bytes}
    EventNum      : LNG;{4 bytes}
    waveform      : TWaveform;
  end;
  TCrArray = array of TCr;

  //THE MAIN PROBE RECORD----------------------------------
  TProbe = record
    ProbeSubType   : CHAR; {=S,C for spike or continuous }
    numchans       : SHRT; //number of channels in this spike waveform
    pts_per_chan   : SHRT; //number of pts per waveform
    trigpt         : SHRT; // pts before trigger
    lockout        : SHRT; // Lockout in pts
    intgain        : SHRT; // A/D board internal gain
    threshold      : SHRT; // A/D board threshold for trigger
    skippts        : SHRT;
    sampfreqperchan: LNG;  // A/D sampling frequency
    chanlist       : TChanList;//= array[0..SURF_MAX_CHANNELS-1] of SHRT;
    screenlayout   : TScreenLayout;//= array[0..SURF_MAX_CHANNELS-1] of TPoint;
    probe_descrip  : ShortString;
    numparams      : LNG;
    paramname      : array{numparams}of String[32];
    numspikes      : LNG;
    numcr          : LNG;
    Spike          : TSpikeArray;
    CR             : TCrArray;
    extgain        : array[0..SURF_MAX_CHANNELS-1] of WORD;//added May21'99
  end;
  TProbeArray = array of TProbe;

  //SINGLE VALUE ANALYSIS RECORD-----------------------------
  TSVal = record
    time_stamp    : LNG; //4 bytes
    subtype       : CHAR;//1 byte -- can be used for different values
    EventNum      : LNG;//4 bytes
    sval          : WORD;//2 bytes -- 16 bit unsigned value
  end;
  TSValArray = array of TSVal;

  //D2A ANALYSIS RECORD-----------------------------
  TD2A = record
    channel       : Byte;//1 or 2
    val         : WORD;//2 bytes -- 16 bit unsigned value
  end;
  //TD2AArray = array of TD2A;

  //Message record-------------------------------------------
  TSurfMsg = record
    time_stamp    : LNG; //4 bytes
    EventNum      : LNG;{4 bytes}
    msg           : ShortString;//256 bytes
  end;
  TSurfMsgArray = array of TSurfMsg;

  TStim = record
    posx,posy,len,wid,sfreq,contrast : SHRT;
  end;
  TStimulus = record
    timediv : double;
    time : array of TStim;
  end;

  TSurfEvent = record
    Time_Stamp    : LNG; //4 bytes
    EventType     : CHAR;//e.g., POLYTRODE, SINGLE VALUE, MESSAGE...
    SubType       : CHAR;//e.g., S,C for spike or continuous
    Probe         : SHRT;//if used, the probe number
    Index         : LNG;//the index into the data array-- e.g., prb[probe].spike[index]....
  end;
  TSurfEventArray = array of TSurfEvent;

  TSurfFileInfo = Record
    NEvents        : Integer;
    SurfEventArray : TSurfEventArray;
    ProbeArray     : TProbeArray;
    SValArray      : TSValArray;
    SurfMsgArray   : TSurfMsgArray;
  end;
