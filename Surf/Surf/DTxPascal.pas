UNIT DTxPascal;
//This unit contains all constants, types, and calls to the dtz dlls

INTERFACE

uses Dialogs,Windows;
{================================ C O N S T A N T S ===========================}
CONST
//------------------------------ system error codes-------------------------
  DTACQ32_ERRORBASE = 22010;
  OLNOERROR = 0;
  OLBADCAP = 1;
  OLBADELEMENT = 2;
  OLBADSUBSYSTEM = 3;
  OLNOTENOUGHDMACHANS = 4;
  OLBADLISTSIZE = 5;
  OLBADLISTENTRY = 6;
  OLBADCHANNEL = 7;
  OLBADCHANNELTYPE = 8;
  OLBADENCODING = 9;
  OLBADTRIGGER = 10;
  OLBADRESOLUTION = 11;
  OLBADCLOCKSOURCE = 12;
  OLBADFREQUENCY = 13;
  OLBADPULSETYPE = 14;
  OLBADPULSEWIDTH = 15;
  OLBADCOUNTERMODE = 16;
  OLBADCASCADEMODE = 17;
  OLBADDATAFLOW = 18;
  OLBADWINDOWHANDLE = 19;
  OLSUBSYSINUSE = 20;
  OLSUBSYSNOTINUSE = 21;
  OLALREADYRUNNING = 22;
  OLNOCHANNELLIST = 23;
  OLNOGAINLIST = 24;
  OLNOFILTERLIST = 25;
  OLNOTCONFIGURED = 26;
  OLDATAFLOWMISMATCH = 27;
  OLNOTRUNNING = 28;
  OLBADRANGE = 29;
  OLBADSSCAP = 30;
  OLBADDEVCAP = 31;
  OLBADRANGEINDEX = 32;
  OLBADFILTERINDEX = 33;
  OLBADGAININDEX = 34;
  OLBADWRAPMODE = 35;
  OLNOTSUPPORTED = 36;
  OLBADDIVIDER = 37;
  OLBADGATE = 38;

  OLBADDEVHANDLE = 39;
  OLBADSSHANDLE = 40;
  OLCANNOTALLOCDASS = 41;
  OLCANNOTDEALLOCDASS = 42;
  OLBUFFERSLEFT = 43;

  OLBOARDRUNNING = 44;       // another subsystem on board is already running
  OLINVALIDCHANNELLIST = 45; // channel list has been filled incorrectly
  OLINVALIDCLKTRIGCOMBO = 46; // selected clock & trigger source may not be used together
  OLCANNOTALLOCLUSERDATA = 47; // driver could not allocate needed memory

  OLCANTSVTRIGSCAN = 48;
  OLCANTSVEXTCLOCK = 49;
  OLBADRESOLUTIONINDEX = 50;

  OLADTRIGERR = 60;
  OLADOVRRN = 61;
  OLDATRIGERR = 62;
  OLDAUNDRRN = 63;

  OLNOREADYBUFFERS = 64;

  OLBADCPU = 65;
  OLBADWINMODE = 66;
  OLCANNOTOPENDRIVER = 67;
  OLBADENUMCAP = 68;
  OLBADDASSPROC = 69;
  OLBADENUMPROC = 70;
  OLNOWINDOWHANDLE = 71;
  OLCANTCASCADE = 72;

  OLINVALIDCONFIGURATION = 73;
  OLCANNOTALLOCJUMPERS = 74;
  OLCANNOTALLOCCHANNELLIST = 75;
  OLCANNOTALLOCGAINLIST = 76;
  OLCANNOTALLOCFILTERLIST = 77;
  OLNOBOARDSINSTALLED = 78;
  OLINVALIDDMASCANCOMBO = 79;
  OLINVALIDPULSETYPE = 80;
  OLINVALIDGAINLIST = 81;
  OLWRONGCOUNTERMODE = 82;
  OLLPSTRNULL = 83;
  OLINVALIDPIODFCOMBO = 84;           // Invalid Polled I/O combination
  OLINVALIDSCANTRIGCOMBO = 85;        // Invalid Scan / Trigger combo
  OLBADGAIN = 86;                     // Invalid Gain
  OLNOMEASURENOTIFY = 87;             // No window handle specified for frequency measurement
  OLBADCOUNTDURATION = 88;            // Invalid count duration for frequency measurement
  OLBADQUEUE = 89;                    // Invalid queue type specified
  OLBADRETRIGGERRATE = 90;            // Invalid Retrigger Rate for channel list size
  OLCOMMANDTIMEOUT = 91;              // No Command Response from Hardware
  OLCOMMANDOVERRUN = 92;              // Hardware Command Sequence Error
  OLDATAOVERRUN = 93;                 // Hardware Data Sequence Error

  OLCANNOTALLOCTIMERDATA = 94;        // Cannot allocate timer data for driver
  OLBADHTIMER = 95;                   // Invalid Timer handle
  OLBADTIMERMODE = 96;                // Invalid Timer mode
  OLBADTIMERFREQUENCY = 97;           // Invalid Timer frequency
  OLBADTIMERPROC = 98;                // Invalid Timer callback procedure
  OLBADDMABUFFERSIZE = 99;            // Invalid Timer DMA buffer size

  OLBADDIGITALIOLISTVALUE = 100;      // Illegal synchronous digital I/O value requested.
  OLCANNOTALLOCSSLIST = 101;          // Cannot allocate simultaneous start list
  OLBADSSLISTHANDLE = 102;            // Illegal simultaneous start list handle specified.
  OLBADSSHANDLEONLIST = 103;          // Invalid subsystem handle on simultaneous start list.
  OLNOSSHANDLEONLIST = 104;           // No subsystem handles on simultaneous start list.
  OLNOCHANNELINHIBITLIST = 105;       // The subsystem has no channel inhibit list.
  OLNODIGITALIOLIST = 106;            // The subsystem has no digital I/O list.
  OLNOTPRESTARTED = 107;              // The subsystem has not been prestarted.
  OLBADNOTIFYPROC = 108;              // Invalid notification procedure
  OLBADTRANSFERCOUNT = 109;           // Invalid DT-Connect transfer count
  OLBADTRANSFERSIZE = 110;            // Invalid DT-Connect transfer size
  OLCANNOTALLOCINHIBITLIST = 111;     // Cannot allocate channel inhibit list
  OLCANNOTALLOCDIGITALIOLIST = 112;   // Cannot allocate digital I/O list
  OLINVALIDINHIBITLIST = 113;         // Channel inhibit list has been filled incorrectly
  OLSSHANDLEALREADYONLIST = 114;      // Subsystem is already on simultaneous start list
  OLCANNOTALLOCRANGELIST = 115;   //Cannot allocate range list
  OLNORANGELIST = 116;            //No Range List
  OLNOBUFFERINPROCESS = 117;      //No buffers in process
  OLREQUIREDSUBSYSINUSE = 118;    //Additional required subsystem in use

  OLBMBASE                    ={ULNG, or ECODE}Cardinal(200);

  OLCANNOTALLOCBCB            = OLBMBASE+0;   // Cannot allocate a buffer control block for the requested data buffer.
  OLCANNOTALLOCBUFFER         = OLBMBASE+1;   // Cannot allocate the requested data buffer.
  OLBADBUFFERHANDLE           = OLBMBASE+2;   // Invalid buffer handle (HBUF) passed to a library from an application.
  OLBUFFERLOCKFAILED          = OLBMBASE+3;   // The data buffer cannot be put to a section because it cannot be properly locked.
  OLBUFFERLOCKED              = OLBMBASE+4;   // Buffer Locked
  OLBUFFERONLIST              = OLBMBASE+5;   // Buffer on List
  OLCANNOTREALLOCBCB          = OLBMBASE+6;   // Reallocation of a buffer control block was unsuccessful.
  OLCANNOTREALLOCBUFFER       = OLBMBASE+7;   // Reallocation of the data buffer was unsuccessful.
  OLBADSAMPLESIZE             = OLBMBASE+8;   // Bad Sample Size
  OLCANNOTALLOCLIST           = OLBMBASE+9;   // Cannot Allocate List
  OLBADLISTHANDLE             = OLBMBASE+10;  // Bad List Handle
  OLLISTNOTEMPTY              = OLBMBASE+11;  // List Not Empty
  OLBUFFERNOTLOCKED           = OLBMBASE+12;  // Bufffer Not Locked
  OLBADDMACHANNEL             = OLBMBASE+13;  // Invalid DMA Channel specified
  OLDMACHANNELINUSE           = OLBMBASE+14;  // Specified DMA Channel in use
  OLBADIRQ                    = OLBMBASE+15;  // Invalid IRQ specified
  OLIRQINUSE                  = OLBMBASE+16;  // Specififed IRQ in use
  OLNOSAMPLES                 = OLBMBASE+17;  // Buffer has no valid samples
  OLTOOMANYSAMPLES            = OLBMBASE+18;  // Valid Samples cannot be larger than buffer
  OLBUFFERTOOSMALL            = OLBMBASE+19;  // Specified buffer too small for requested copy operation

  //VBMAXERROR = 32767;
//------------------------------- Device Capabilities --------------------------
  OLDC_ADELEMENTS = 0;
  OLDC_DAELEMENTS = 1;
  OLDC_DINELEMENTS = 2;
  OLDC_DOUTELEMENTS = 3;
  OLDC_SRLELEMENTS = 4;
  OLDC_CTELEMENTS = 5;
  OLDC_TOTALELEMENTS = Not 0;
//------------------------------- SubSystem Capabilities -----------------------
  OLSSC_MAXSECHANS = 0;
  OLSSC_MAXDICHANS = 1;
  OLSSC_CGLDEPTH = 2;
  OLSSC_NUMFILTERS = 3;
  OLSSC_NUMGAINS = 4;
  OLSSC_NUMRANGES = 5;
  OLSSC_NUMDMACHANS = 6;
  OLSSC_NUMCHANNELS = 7;
  OLSSC_NUMEXTRACLOCKS = 8;
  OLSSC_NUMEXTRATRIGGERS = 9;
  OLSSC_NUMRESOLUTIONS = 10;
  OLSSC_SUP_INTERRUPT = 11;
  OLSSC_SUP_SINGLEENDED = 12;
  OLSSC_SUP_DIFFERENTIAL = 13;
  OLSSC_SUP_BINARY = 14;
  OLSSC_SUP_2SCOMP = 15;
  OLSSC_SUP_SOFTTRIG = 16;
  OLSSC_SUP_EXTERNTRIG = 17;
  OLSSC_SUP_THRESHTRIGPOS = 18;
  OLSSC_SUP_THRESHTRIGNEG = 19;
  OLSSC_SUP_ANALOGEVENTTRIG = 20;
  OLSSC_SUP_DIGITALEVENTTRIG = 21;
  OLSSC_SUP_TIMEREVENTTRIG = 22;
  OLSSC_SUP_TRIGSCAN = 23;
  OLSSC_SUP_INTCLOCK = 24;
  OLSSC_SUP_EXTCLOCK = 25;
  OLSSC_SUP_SWCAL = 26;
  OLSSC_SUP_EXP2896 = 27;
  OLSSC_SUP_EXP727 = 28;
  OLSSC_SUP_FILTERPERCHAN = 29;
  OLSSC_SUP_DTCONNECT = 30;
  OLSSC_SUP_FIFO = 31;
  OLSSC_SUP_PROGRAMGAIN = 32;
  OLSSC_SUP_PROCESSOR = 33;
  OLSSC_SUP_SWRESOLUTION = 34;
  OLSSC_SUP_CONTINUOUS = 35;
  OLSSC_SUP_SINGLEVALUE = 36;
  OLSSC_SUP_PAUSE = 37;
  OLSSC_SUP_WRPMULTIPLE = 38;
  OLSSC_SUP_WRPSINGLE = 39;
  OLSSC_SUP_POSTMESSAGE = 40;
  OLSSC_SUP_CASCADING = 41;
  OLSSC_SUP_CTMODE_COUNT = 42;
  OLSSC_SUP_CTMODE_RATE = 43;
  OLSSC_SUP_CTMODE_ONESHOT = 44;
  OLSSC_SUP_CTMODE_ONESHOT_RPT = 45;

  OLSSC_MAX_DIGITALIOLIST_VALUE = 46;
  OLSSC_SUP_DTCONNECT_CONTINUOUS = 47;
  OLSSC_SUP_DTCONNECT_BURST = 48;
  OLSSC_SUP_CHANNELLIST_INHIBIT = 49;
  OLSSC_SUP_SYNCHRONOUS_DIGITALIO = 50;
  OLSSC_SUP_SIMULTANEOUS_START = 51;
  OLSSC_SUP_INPROCESSFLUSH = 52;

  OLSSC_SUP_RANGEPERCHANNEL = 53;
  OLSSC_SUP_SIMULTANEOUS_SH = 54;
  OLSSC_SUP_RANDOM_CGL = 55;
  OLSSC_SUP_SEQUENTIAL_CGL = 56;
  OLSSC_SUP_ZEROSEQUENTIAL_CGL = 57;

  OLSSC_SUP_GAPFREE_NODMA = 58;
  OLSSC_SUP_GAPFREE_SINGLEDMA = 59;
  OLSSC_SUP_GAPFREE_DUALDMA = 60;

  OLSSCE_MAXTHROUGHPUT = 61;
  OLSSCE_MINTHROUGHPUT = 62;
  OLSSCE_MAXRETRIGGER = 63;
  OLSSCE_MINRETRIGGER = 64;
  OLSSCE_MAXCLOCKDIVIDER = 65;
  OLSSCE_MINCLOCKDIVIDER = 66;
  OLSSCE_BASECLOCK = 67;
  OLSSCE_RANGELOW = 68;
  OLSSCE_RANGEHIGH = 69;
  OLSSCE_FILTER = 70;
  OLSSCE_GAIN = 71;
  OLSSCE_RESOLUTION = 72;

  OLSSC_SUP_PLS_HIGH2LOW = 73;
  OLSSC_SUP_PLS_LOW2HIGH = 74;

  OLSSC_SUP_GATE_NONE = 75;
  OLSSC_SUP_GATE_HIGH_LEVEL = 76;
  OLSSC_SUP_GATE_LOW_LEVEL = 77;
  OLSSC_SUP_GATE_HIGH_EDGE = 78;
  OLSSC_SUP_GATE_LOW_EDGE = 79;
  OLSSC_SUP_GATE_LEVEL = 80;
  OLSSC_SUP_GATE_HIGH_LEVEL_DEBOUNCE = 81;
  OLSSC_SUP_GATE_LOW_LEVEL_DEBOUNCE = 82;
  OLSSC_SUP_GATE_HIGH_EDGE_DEBOUNCE = 83;
  OLSSC_SUP_GATE_LOW_EDGE_DEBOUNCE = 84;
  OLSSC_SUP_GATE_LEVEL_DEBOUNCE = 85;

  OLSS_SUP_RETRIGGER_INTERNAL = 86;
  OLSS_SUP_RETRIGGER_SCAN_PER_TRIGGER = 87;
  OLSSC_MAXMULTISCAN = 88;
  OLSSC_SUP_CONTINUOUS_PRETRIG = 89;
  OLSSC_SUP_CONTINUOUS_ABOUTTRIG = 90;
  OLSSC_SUP_BUFFERING = 91;
  OLSSC_SUP_RETRIGGER_EXTRA = 92;
//------------------------------- Configuration Settings -----------------------
// for SubSysType property;
  OLSS_AD = 0;
  OLSS_DA = 1;
  OLSS_DIN = 2;
  OLSS_DOUT = 3;
  OLSS_SRL = 4;
  OLSS_CT = 5;

// for ChannelType property;
  OL_CHNT_SINGLEENDED = 0;
  OL_CHNT_DIFFERENTIAL = 1;

// for Encoding property;
  OL_ENC_BINARY = 0;
  OL_ENC_2SCOMP = 1;

// for trigger property;
  OL_TRG_SOFT = 0;
  OL_TRG_EXTERN = 1;
  OL_TRG_THRESHPOS = 2;
  OL_TRG_THRESHNEG = 3;
  OL_TRG_ANALOGEVENT = 4;
  OL_TRG_DIGITALEVENT = 5;
  OL_TRG_TIMEREVENT = 6;
  OL_TRG_EXTRA = 7;

// for ClockSource property;
  OL_CLK_INTERNAL = 0;
  OL_CLK_EXTERNAL = 1;
  OL_CLK_EXTRA = 2;

// for GateType property;
  OL_GATE_NONE = 0;
  OL_GATE_HIGH_LEVEL = 1;
  OL_GATE_LOW_LEVEL = 2;
  OL_GATE_HIGH_EDGE = 3;
  OL_GATE_LOW_EDGE = 4;
  OL_GATE_LEVEL               =5;
  OL_GATE_HIGH_LEVEL_DEBOUNCE =6;
  OL_GATE_LOW_LEVEL_DEBOUNCE  =7;
  OL_GATE_HIGH_EDGE_DEBOUNCE  =8;
  OL_GATE_LOW_EDGE_DEBOUNCE   =9;
  OL_GATE_LEVEL_DEBOUNCE      =10;

// for PulseType property;
  OL_PLS_HIGH2LOW = 0;
  OL_PLS_LOW2HIGH = 1;

// for CTMode property;
  OL_CTMODE_COUNT = 0;
  OL_CTMODE_RATE = 1;
  OL_CTMODE_ONESHOT = 2;
  OL_CTMODE_ONESHOT_RPT = 3;

// for DataFlow property;
  OL_DF_CONTINUOUS = 0;
  OL_DF_SINGLEVALUE = 1;
  OL_DF_DTCONNECT_CONTINUOUS = 2;
  OL_DF_DTCONNECT_BURST = 3;
  OL_DF_CONTINUOUS_PRETRIG = 4;
  OL_DF_CONTINUOUS_ABOUTTRIG = 5;

// for CascadeMode Property;
  OL_CT_CASCADE = 0;
  OL_CT_SINGLE = 1;

// for WrapMode Property;
  OL_WRP_NONE = 0;
  OL_WRP_MULTIPLE = 1;
  OL_WRP_SINGLE = 2;

// for QueueSize Property;
  OL_QUE_READY = 0;
  OL_QUE_DONE = 1;
  OL_QUE_INPROCESS = 2;

// for RetriggerMode Property;
  OL_RETRIGGER_INTERNAL = 0;
  OL_RETRIGGER_SCAN_PER_TRIGGER = 1;
  OL_RETRIGGER_EXTRA = 2;

{================================== T Y P E S =================================}
TYPE                 // In C:
  CHR    = char;     // signed char
  LPCHR  = ^char;    // CHR FAR*
  UCHR   = byte;     // unsigned char
  LUPCHR = ^byte;    // UCHR FAR*
  SHRT   = smallint; // short
  LPSHRT = ^smallint;// SHRT FAR*
  USHRT  = word;     // unsigned short
  LPUSHRT= ^word;    // USHRT FAR*
  LNG    = LongInt;  // long
  LPLNG  = ^LongInt; // LNG FAR*
  ULNG   = LongWord; // unsigned long
  LPULNG = ^LongWord;// ULNG FAR*
  FLT    = single;   // float
  LPFLT  = ^single;  // FLT FAR*
  DBL    = double;   // double
  LPDBL  = ^double;  // DBL FAR*

  UINT   = LongWord; //unsigned int
  LPUINT = ^LongWord;//UINT FAR*
  LPBOOL = ^boolean; //BOOL FAR*

  ECODETYPE = ULNG;
  LPECODE = ^ECODETYPE;//ECODE FAR*

  HBUFTYPE = ULNG;
  HBUFPTR = ^HBUFTYPE;

  TDS = record
    year,month,day,hour,min,sec : Integer;
  end;

VAR
  ecode : ECODETYPE; //unsigned long
  //hbuf : HBUFTYPE; //unsigned long

{============================  F O R W A R D S ================================}
Function ErrorMsg(Ecode : ECODETYPE) : boolean;
//Handle all error messages and return true if there is an error

function olDmAllocBuffer (usWinFlags : UINT; ulBufferSize : ULNG; hBuf : HBUFPTR): ECODETYPE;
  stdcall; external 'OLMEM32.DLL';

function olDmCallocBuffer (uiWinFlags : UINT; uiExFlags : UINT; dwSamples : DWORD; uiSampleSize : UINT; hBuf : HBUFPTR): ECODETYPE;
  stdcall; external 'OLMEM32.DLL';

function olDmFreeBuffer (hBuf : HBUFTYPE): ECODETYPE;
  stdcall; external 'OLMEM32.DLL';
//Function olDmFreeBuffer& "OLMEM32.DLL" (ByVal hbuf&)

Function olDmGetBufferSize (hbuf : HBUFTYPE; var UlBufferSize : ULNG): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//Declare Function olDmGetBufferSize& Lib "OLMEM32.DLL" (ByVal hbuf&, lpUlBufferSize&)
//ECODE WINAPI olDmGetBufferSize (HBUF, DWORD FAR*);

Function olDmGetValidSamples (hbuf : HBUFTYPE; var size : ULNG): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//Declare Function olDmGetValidSamples& Lib "OLMEM32.DLL" (ByVal hbuf&, size&)
//ECODE WINAPI olDmGetValidSamples (HBUF, DWORD FAR*);

Function olDmGetDataBits (hbuf : HBUFTYPE; var size : ULNG): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//Declare Function olDmGetDataBits& Lib "OLMEM32.DLL" (ByVal hbuf&, size&)
//ECODE WINAPI olDmGetDataBits (HBUF, UINT FAR*);

Function olDmCopyFromBuffer (hbuf : HBUFTYPE; lpAppBuffer : LPSHRT; ulMaxSamples : ULNG): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//Declare Function olDmCopyFromBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&, vbarray As Integer, ByVal maxsamples&)
//ECODE olDmCopyFromBuffer (HBUF hBuf, LPVOID lpAppBuffer, ULNG ulMaxSamples)

Function olDmCopyToBuffer (hbuf : HBUFTYPE; lpAppBuffer : LPSHRT; ulMaxSamples : ULNG): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//Declare Function olDmCopyToBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&, vbarray As Integer, ByVal Numsamples&)
//ECODE WINAPI olDmCopyToBuffer (HBUF hBuf, LPVOID lpAppBuffer, ULNG ulNumSamples);

Function olDmGetErrorString (ecode : ECODETYPE; lpStr : LPSTR; uiMaxSize : UINT): LPSTR;
 stdcall; external 'OLMEM32.DLL';
//Declare Sub olDmGetErrorString Lib "OLMEM32.DLL" (ByVal ecode&, ByVal ErrMsg As String, ByVal maxsize%)
//LPSTR WINAPI olDmGetErrorString (ECODE eCode, LPSTR lpStr, );

Function olDmGetBufferPtr (hbuf : HBUFTYPE; var lpVoid : LPUSHRT): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';

Function olDspInputToVolts (hbufIn : HBUFTYPE; vbufOut : HBUFTYPE; Datatype: byte;
                            MinRange : FLT; MaxRange : FLT; Bits : WORD): ECODETYPE;
 stdcall; external 'OLDSP32.DLL';

Function olDspMagtoDB (hbufIn : HBUFTYPE; hbufOut : HBUFTYPE) : ECODETYPE;
 stdcall; external 'OLDSP32.DLL';

Function olDspRealFFT (hbufIn : HBUFTYPE; hbufMag : HBUFTYPE; hbufPhase: HBUFTYPE;
                            FFTSize : SHRT): ECODETYPE;
 stdcall; external 'OLDSP32.DLL';

Function olDmCopyChannelFromBuffer (hbuf : HBUFTYPE; Channel, NumChannels : integer;
                                 lpArrayBuffer : LPSHRT; var NumSamples : ULNG): ECODETYPE;
 stdcall; external 'OLMEMSUP.DLL';  {var/const declarations?}


 //ECODE WINAPI olDmGetBufferPtr (HBUF, LPVOID FAR*);

(*   //these don't work
Function olDaSetChannelRange (hdass : HBUFTYPE; Channel : UINT; dMaxVoltage, dMinVoltage : DBL): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//ECODE olDaSetChannelRange (HDASS hDass, UINT uiChan, DBL dMaxVoltage. DBL dMinVoltage)
Function olDaGetChannelRange (hdass : HBUFTYPE; Channel : UINT; dMaxVoltage,dMinVoltage : LPDBL): ECODETYPE;
 stdcall; external 'OLMEM32.DLL';
//ECODE olDaGetChannelRange (HDASS hDass, UINT uiChan, LPDBL lpdMaxVoltage. LPDBL lpdMinVoltage)
*)

{ C olmem routines:
/*********************** BCB's **********************/

DECLARE_HANDLE(HBUF);

typedef HBUF FAR* LPHBUF;

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _OLACI_        // if oldaaci module is included - do NOT
                       // redefine these routines!!!  The names
                       // are the same as V0.9

ECODE WINAPI olDmAllocBuffer (UINT, DWORD, LPHBUF);
ECODE WINAPI olDmFreeBuffer (HBUF);
ECODE WINAPI olDmReAllocBuffer (UINT, DWORD, LPHBUF);
ECODE WINAPI olDmGetTimeDateStamp (HBUF, LPTDS);
ECODE WINAPI olDmGetBufferPtr (HBUF, LPVOID FAR*);
ECODE WINAPI olDmGetBufferECode (HBUF, LPECODE);
ECODE WINAPI olDmGetBufferSize (HBUF, DWORD FAR*);
ECODE WINAPI olDmGetVersion (LPSTR);
ECODE WINAPI olDmCopyBuffer (HBUF, LPVOID);
ECODE WINAPI olDmCopyToBuffer (HBUF hBuf, LPVOID lpAppBuffer, ULNG ulNumSamples);

#endif

ECODE WINAPI olDmCallocBuffer (UINT, UINT, DWORD, UINT, LPHBUF);
ECODE WINAPI olDmMallocBuffer (UINT, UINT, DWORD, LPHBUF);
ECODE WINAPI olDmLockBuffer (HBUF);
ECODE WINAPI olDmUnlockBuffer (HBUF);
ECODE WINAPI olDmReCallocBuffer (UINT, UINT, DWORD, UINT, LPHBUF);
ECODE WINAPI olDmReMallocBuffer (UINT, UINT, DWORD, LPHBUF);
ECODE WINAPI olDmGetDataBits (HBUF, UINT FAR*);
ECODE WINAPI olDmSetDataWidth (HBUF, UINT);
ECODE WINAPI olDmGetDataWidth (HBUF, UINT FAR*);
ECODE WINAPI olDmGetMaxSamples (HBUF, DWORD FAR*);
ECODE WINAPI olDmSetValidSamples (HBUF, DWORD);
ECODE WINAPI olDmGetValidSamples (HBUF, DWORD FAR*);
ECODE WINAPI olDmCopyFromBuffer(HBUF hBuf, LPVOID lpAppBuffer, ULNG ulMaxSamples);

ECODE WINAPI olDmGetExtraBytes (HBUF hBuf, ULNG FAR *lpulExtra1,ULNG FAR *lpulExtra2);
ECODE WINAPI olDmSetExtraBytes (HBUF hBuf, ULNG ulExtra1, ULNG ulExtra2);

ECODE WINAPI olDmLockBufferEx (HBUF hBuf, BOOL fEnableScatter);
ECODE WINAPI olDmUnlockBufferEx (HBUF hBuf, BOOL fEnableScatter);




/*********************** BTL's **********************/


DECLARE_HANDLE(HLIST);

typedef HLIST FAR* LPHLIST;
typedef BOOL (CALLBACK* BUFPROC)(HBUF, LPARAM);
typedef BOOL (CALLBACK* LISTPROC)(HLIST, LPARAM);
typedef BOOL (__far __pascal * INTPROC)(LPARAM);    // must be FIXED


ECODE WINAPI olDmCreateList (LPHLIST, UINT, LPCSTR, LPCSTR);
ECODE WINAPI olDmEnumLists (LISTPROC, LPARAM);
ECODE WINAPI olDmEnumBuffers (HLIST, BUFPROC, LPARAM);
ECODE WINAPI olDmFreeList (HLIST);
ECODE WINAPI olDmPutBufferToList (HLIST, HBUF);
ECODE WINAPI olDmGetBufferFromList (HLIST, LPHBUF);
ECODE WINAPI drvDmPutBufferToListForDriver (HLIST, HBUF);
ECODE WINAPI drvDmGetBufferFromListForDriver (HLIST, LPHBUF);
ECODE WINAPI olDmPeekBufferFromList (HLIST, LPHBUF);
ECODE WINAPI olDmGetListCount (HLIST, UINT FAR*);
ECODE WINAPI olDmGetListHandle (HBUF, LPHLIST);
ECODE WINAPI olDmGetListIds (HLIST, LPSTR, UINT, LPSTR, UINT);

ECODE WINAPI olDmLockBufferEx (HBUF hBuf, BOOL fEnableScatter);
ECODE WINAPI olDmUnLockBufferEx (HBUF hBuf, BOOL fEnableScatter);

LPSTR WINAPI olDmGetErrorString (ECODE eCode, LPSTR lpStr, UINT uiMaxSize);

}

(*
//------------- ol Memory Management Function prototypes -----------'
//get error string
Declare Sub olDmGetErrorString Lib "OLMEM32.DLL" (ByVal ecode&, ByVal ErrMsg As String, ByVal maxsize%)
//get version info
Declare Function olDmGetVersion& Lib "OLMEM32.DLL" (ByVal Version As String, ByVal maxsize%)

//Allocate buffers:
Declare Function olDmAllocBuffer& Lib "OLMEM32.DLL" (ByVal usWinFlags%, ByVal ulBufferSize&, hBuffer&)
Declare Function olDmCallocBuffer& Lib "OLMEM32.DLL" (ByVal usWinFlags%, ByVal ExFlags%, ByVal samples&, ByVal SampSize%, hBuffer&)

//Free Buffer
Declare Function olDmFreeBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&)

//Re-allocate (resize) buffer
Declare Function olDmReAllocBuffer& Lib "OLMEM32.DLL" (ByVal usWinFlags%, ByVal ulBufferSize&, hBuffer&)
Declare Function olDmReCallocBuffer& Lib "OLMEM32.DLL" (ByVal usWinFlags%, ByVal ExFlags%, ByVal samples&, ByVal SampSize%, hBuffer&)


//Get physical size of Buffer
Declare Function olDmGetBufferSize& Lib "OLMEM32.DLL" (ByVal hbuf&, lpUlBufferSize&)

//Copy VB array to buffer
Declare Function olDmCopyToBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&, vbarray As Integer, ByVal Numsamples&)
Declare Function olDmCopyLongToBuffer& Lib "OLMEM32.DLL" Alias "olDmCopyToBuffer" (ByVal hbuf&, vbarray As Long, ByVal Numsamples&)
Declare Function olDmCopySingleToBuffer& Lib "OLMEM32.DLL" Alias "olDmCopyToBuffer" (ByVal hbuf&, vbarray As Single, ByVal Numsamples&)


//Copy buffer data to VB array
Declare Function olDmCopyFromBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&, vbarray As Integer, ByVal maxsamples&)
Declare Function olDmCopyLongFromBuffer& Lib "OLMEM32.DLL" Alias "olDmCopyFromBuffer" (ByVal hbuf&, vbarray As Long, ByVal maxsamples&)
Declare Function olDmCopySingleFromBuffer& Lib "OLMEM32.DLL" Alias "olDmCopyFromBuffer" (ByVal hbuf&, vbarray As Single, ByVal maxsamples&)


//Get logical size of buffer
Declare Function olDmGetValidSamples& Lib "OLMEM32.DLL" (ByVal hbuf&, size&)
Declare Function olDmGetMaxSamples& Lib "OLMEM32.DLL" (ByVal hbuf&, size&)

//Get size of sample in bytes
Declare Function olDmGetDataWidth& Lib "OLMEM32.DLL" (ByVal hbuf&, size&)

//Get number of significant bits in each sample
Declare Function olDmGetDataBits& Lib "OLMEM32.DLL" (ByVal hbuf&, size&)

//Set logical size of buffer
Declare Function olDmSetValidSamples& Lib "OLMEM32.DLL" (ByVal hbuf&, ByVal size&)

//Get Time/Date Stamp of buffer
Declare Function olDmGetTimeDateStamp& Lib "OLMEM32.DLL" (ByVal hbuf&, stamp As TDS)

//Lock / Unlock buffer in memory
Declare Function olDmLockBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&)
Declare Function olDmUnlockBuffer& Lib "OLMEM32.DLL" (ByVal hbuf&)

//Copy VB array to buffer
Declare Function olDmCopyChannelToBuffer& Lib "OLMEMSUP.DLL" (ByVal hbuf&, ByVal StartSamp%, ByVal SampInc%, vbarray As Integer, Numsamples&)
Declare Function olDmCopyLongChannelToBuffer& Lib "OLMEMSUP.DLL" Alias "olDmCopyChannelToBuffer" (ByVal hbuf&, ByVal StartSamp%, ByVal SampInc%, vbarray As Long, Numsamples&)
Declare Function olDmCopySingleChannelToBuffer& Lib "OLMEMSUP.DLL" Alias "olDmCopyChannelToBuffer" (ByVal hbuf&, ByVal StartSamp%, ByVal SampInc%, vbarray As Single, Numsamples&)

//Copy buffer data to VB array

Declare Function olDmCopyChannelFromBuffer& Lib "OLMEMSUP.DLL" (ByVal hbuf&, ByVal StartSamp%, ByVal SampInc%, vbarray As Integer, Numsamples&)
Declare Function olDmCopyLongChannelFromBuffer& Lib "OLMEMSUP.DLL" Alias "olDmCopyChannelFromBuffer" (ByVal hbuf&, ByVal StartSamp%, ByVal SampInc%, vbarray As Long, Numsamples&)
Declare Function olDmCopySingleChannelFromBuffer& Lib "OLMEMSUP.DLL" Alias "olDmCopyChannelFromBuffer" (ByVal hbuf&, ByVal StartSamp%, ByVal SampInc%, vbarray As Single, Numsamples&)

BASIC CONVERSIONS:
Global Const GMEM_FIXED = 0
Global Const GMEM_MOVEABLE = 2
Global Const GMEM_NOCOMPACT = 16
Global Const GMEM_NODISCARD = 32
Global Const GMEM_ZEROINIT = 64
Global Const GMEM_MODIFY = 128
Global Const GMEM_DISCARDABLE = 256
Global Const GMEM_NOT_BANKED = 4096
Global Const GMEM_SHARE = 8192
Global Const GMEM_DDESHARE = 8192
Global Const GMEM_NOTIFY = 16384
Sub CopyChannelFromBuffer(ByVal hbuf As Long, ByVal StartSamp As Integer, ByVal SampleInc As Integer, vbarray As Integer, Numsamples As Long)
Dim ecode As Long

    ecode = olDmCopyChannelFromBuffer(hbuf, StartSamp, SampleInc, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyChannelFromBuffer", ecode
    End If


End Sub
Sub CopyChannelToBuffer(hbuf As Long, ByVal StartSamp As Integer, ByVal SampleInc As Integer, vbarray As Integer, Numsamples As Long)
Dim ecode As Long

    ecode = olDmCopyChannelToBuffer(hbuf, StartSamp, SampleInc, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyChannelToBuffer", ecode
    End If

End Sub
Sub CopyLongChannelFromBuffer(ByVal hbuf As Long, ByVal StartSamp As Integer, ByVal SampleInc As Integer, vbarray As Long, Numsamples As Long)
Dim ecode As Long
    
    ecode = olDmCopyLongChannelFromBuffer(hbuf, StartSamp, SampleInc, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyLongChannelFromBuffer", ecode
    End If


End Sub
Sub CopyLongChannelToBuffer(hbuf As Long, ByVal StartSamp As Integer, ByVal SampleInc As Integer, vbarray As Long, Numsamples As Long)
Dim ecode As Long

    ecode = olDmCopyLongChannelToBuffer(hbuf, StartSamp, SampleInc, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyLongChannelToBuffer", ecode
    End If

End Sub

Sub CopyLongFromBuffer(ByVal hbuf As Long, vbarray As Long, ByVal maxsamples As Long)
Dim ecode As Long

    ecode = olDmCopyLongFromBuffer(hbuf, vbarray, maxsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyLongFromBuffer", ecode
    End If

End Sub
Sub CopyLongToBuffer(ByVal hbuf As Long, vbarray As Long, ByVal Numsamples As Long)
Dim ecode As Long

    ecode = olDmCopyLongToBuffer(hbuf, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyLongToBuffer", ecode
    End If

End Sub
Sub CopySingleChannelFromBuffer(ByVal hbuf As Long, ByVal StartSamp As Integer, ByVal SampleInc As Integer, vbarray As Single, Numsamples As Long)
Dim ecode As Long
    
    ecode = olDmCopySingleChannelFromBuffer(hbuf, StartSamp, SampleInc, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopySingleChannelFromBuffer", ecode
    End If

End Sub

Sub CopySingleChannelToBuffer(ByVal hbuf As Long, ByVal StartSamp As Integer, ByVal SampleInc As Integer, vbarray As Single, Numsamples As Long)
Dim ecode As Long
    
    ecode = olDmCopySingleChannelToBuffer(hbuf, StartSamp, SampleInc, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopySingleChannelToBuffer", ecode
    End If

End Sub
Sub CopySingleFromBuffer(ByVal hbuf As Long, vbarray As Single, ByVal maxsamples As Long)
Dim ecode As Long
    
    ecode = olDmCopySingleFromBuffer(hbuf, vbarray, maxsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopySingleFromBuffer", ecode
    End If
End Sub
Sub CopySingleToBuffer(ByVal hbuf As Long, vbarray As Single, ByVal Numsamples As Long)
Dim ecode As Long
    
    ecode = olDmCopySingleToBuffer(hbuf, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopySingleToBuffer", ecode
    End If

End Sub








Function AllocBuffer(ByVal WinFlags As Integer, ByVal buffersize As Long) As Long
Dim hbuf As Long
Dim ecode As Long

    ecode = olDmAllocBuffer(WinFlags, buffersize, hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "AllocBuffer", ecode
    Else
        AllocBuffer = hbuf
    End If

End Function

Sub OLMEMTrap(FuncName As String, ByVal Errnum As Integer)
Dim ErrStg As String * 255
Const MB_ICONSTOP = 16

    'Get DT-Open Layers Error string
    ErrStg = GetOLMEMErrorString(Errnum, 80)
    'MsgBox ErrStg, MB_ICONSTOP, FuncName & " Error"
    Err.Raise vbObjectError + Errnum, FuncName, ErrStg
    
End Sub
Function GetOLMEMErrorString(ByVal ecode As Integer, ByVal maxsize As Integer) As String
    Dim ErrMsg As String * 255

    If maxsize > 255 Then
        maxsize = 255
    End If

    olDmGetErrorString ecode, ErrMsg, maxsize

    GetOLMEMErrorString = ErrMsg

End Function


Function CallocBuffer(ByVal WinFlags As Integer, ByVal ExFlags As Integer, ByVal samples As Long, ByVal Samplesize As Integer) As Long
Dim hbuf As Long
Dim ecode As Long

    ecode = olDmCallocBuffer(WinFlags, ExFlags, samples, Samplesize, hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CallocBuffer", ecode
    Else
        CallocBuffer = hbuf
    End If

End Function

Sub CopyFromBuffer(ByVal hbuf As Long, vbarray As Integer, ByVal maxsamples As Long)
Dim ecode As Long

    ecode = olDmCopyFromBuffer(hbuf, vbarray, maxsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyFromBuffer", ecode
    End If

End Sub

Sub CopyToBuffer(ByVal hbuf As Long, vbarray As Integer, ByVal Numsamples As Long)
Dim ecode As Long

    ecode = olDmCopyToBuffer(hbuf, vbarray, Numsamples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "CopyToBuffer", ecode
    End If

End Sub

Sub FreeBuffer(ByVal hbuf As Long)
Dim ecode As Long

    ecode = olDmFreeBuffer(hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "FreeBuffer", ecode
    End If

End Sub

Function GetBufferSize(ByVal hbuf As Long) As Long
Dim ecode As Long
Dim size As Long

    ecode = olDmGetBufferSize(hbuf, size)
    If ecode <> OLNOERROR Then
        OLMEMTrap "GetBufferSize", ecode
    Else
        GetBufferSize = size
    End If

End Function
Function GetDataBits(ByVal hbuf As Long) As Integer
Dim ecode As Long
Dim size As Long

    ecode = olDmGetDataBits(hbuf, size)
    If ecode <> OLNOERROR Then
        OLMEMTrap "GetDataBits", ecode
    Else
        GetDataBits = size
    End If

End Function
Function GetDataWidth(ByVal hbuf As Long) As Integer
Dim ecode As Long
Dim size As Long

    ecode = olDmGetDataWidth(hbuf, size)
    If ecode <> OLNOERROR Then
        OLMEMTrap "GetDataWidth", ecode
    Else
        GetDataWidth = size
    End If

End Function
Function GetMaxSamples(ByVal hbuf As Long) As Long
Dim ecode As Long
Dim samples As Long

    ecode = olDmGetMaxSamples(hbuf, samples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "GetMaxSamples", ecode
    Else
        GetMaxSamples = samples
    End If

End Function

Function GetTimeDateStamp(ByVal hbuf As Long) As String
Dim ecode As Long
Dim TDSstamp As TDS
Dim stampstg As String * 255

    ecode = olDmGetTimeDateStamp(hbuf, TDSstamp)
    If ecode <> OLNOERROR Then
        OLMEMTrap "GetTimeDateStamp", ecode
    Else
        stampstg = TDSstamp.month & "/" & TDSstamp.day & "/" & TDSstamp.year & "  " & TDSstamp.hour & ":" & TDSstamp.min & ":" & TDSstamp.sec
        GetTimeDateStamp = stampstg
    End If

End Function

Function GetValidSamples(ByVal hbuf As Long) As Long
Dim ecode As Long
Dim samples As Long

    ecode = olDmGetValidSamples(hbuf, samples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "GetValidSamples", ecode
    Else
        GetValidSamples = samples
    End If

End Function
Sub LockBuffer(ByVal hbuf As Long)
Dim ecode As Long

    ecode = olDmLockBuffer(hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "LockBuffer", ecode
    End If

End Sub

Sub ReAllocBuffer(ByVal WinFlags As Integer, ByVal buffersize As Long, hbuf As Long)
Dim ecode As Long
    
    ecode = olDmReAllocBuffer(WinFlags, buffersize, hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "ReAllocBuffer", ecode
    End If

End Sub

Sub ReCallocBuffer(ByVal WinFlags As Integer, ByVal ExFlags As Integer, ByVal samples As Long, ByVal Samplesize As Integer, hbuf As Long)
Dim ecode As Long

    ecode = olDmReCallocBuffer(WinFlags, ExFlags, samples, Samplesize, hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "ReCallocBuffer", ecode
    End If

End Sub

Sub SetValidSamples(ByVal hbuf As Long, ByVal samples As Long)
Dim ecode As Long

    ecode = olDmSetValidSamples(hbuf, samples)
    If ecode <> OLNOERROR Then
        OLMEMTrap "SetValidSamples", ecode
    End If

End Sub

Sub UnlockBuffer(ByVal hbuf As Long)
Dim ecode As Long

    ecode = olDmUnlockBuffer(hbuf)
    If ecode <> OLNOERROR Then
        OLMEMTrap "UnLockBuffer", ecode
    End If

End Sub
*)


IMPLEMENTATION

function ErrorMsg(Ecode : ECODETYPE) : boolean;
var es : string;
begin
  ErrorMsg := FALSE;
  if(ecode <> OLNOERROR) then
  begin
    ShowMessage(olDmGetErrorString(ecode,@es,100)^);
    ErrorMsg := TRUE;
  end;
end;

END.
