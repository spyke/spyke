{ (c) 2002-2003 Tim Blanche, University of British Columbia }
unit SURFContAcq;

interface

uses
  About, Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs, ComCtrls, ExtCtrls,
  CommCtrl, Menus, ToolWin, ImgList, Buttons, StdCtrls, DTxPascal, DTAcq32Lib_TLB, SurfPublicTypes,
  ElectrodeTypes, ShellApi, ProbeSet, InfoWinUnit, EEGUnit, WaveFormPlotUnit, ChartFormUnit,
  SurfFile{temporary, incorporate into VCL object!}, SurfTypes, Mask {consolidate!}, PolytrodeGUI;

const {some of these should/are probably be declared in SurfTypes or SurfPublicTypes}
  MASTER_CLOCK_FREQUENCY = 1000000{1MHz};
  MASTER_CLOCK_PERIOD = 1 / MASTER_CLOCK_FREQUENCY * 1000000{µsec};
  DEFAULT_SAMPLE_FREQ_PER_CHAN = 25000{25KHz};
  EXP_TIMER_FREQ = 10{Hz};
  TIMER_TICKS_PER_MINUTE = 60 * EXP_TIMER_FREQ;
  TIMER_TICKS_PER_HOUR = 60 * TIMER_TICKS_PER_MINUTE;
  BUFFERS_PER_SECOND = 10{Hz};
  DEFAULT_NUM_BUFFS = 5 * BUFFERS_PER_SECOND{5s of A/D buffering};
  CLOCK_TICKS_PER_BUFFER = MASTER_CLOCK_FREQUENCY div BUFFERS_PER_SECOND;
  DISK_WRITES_PER_SECOND = 1{Hz};

  EXTERNAL_AMPLIFIER_GAIN = 5000; //default gain for MCS FA-I-64

  DIN_PROBE         = 32;
  DIN_DATA_STROBE   = $0100; //data strobe      bit 0
  DIN_DISP_SWEEP    = $0200; //display sweep    bit 1
  DIN_DISP_RUNNING  = $0400; //display running  bit 2
  DIN_DISP_FRAME    = $0800; //frame toggle     bit 3

  //SSRINGARRAYSIZE = 100; <-- dynamic, depends on # of save probes
  SVRINGARRAYSIZE  = 1000; //10 seconds ringbuffer assuming 100Hz display (ie. no SV analogue channels)
  MSGRINGARRAYSIZE = 50;

  MAX_EEG_WINDOWS  = 3;
  MAX_MUX_CHANNELS = 80;
  AUTOMON_INTERVAL = 5{sec};
  ADC_SATURATED_ERROR_INTERVAL = 10{sec};

  CSD_SEARCH_STRING = 'LFP ch';  //only probe descriptors with this substring will be incl. in CSDWins
  CSD_nDELTAY = 1;
  DEFAULT_NUM_CSD_SAMPLES = 600; //ie. 600ms for 1kHz EEG/LFP sample rate

  RED_LED = 10;
  YELLOW_LED = 21;

type {some of these should/are probably be declared in SurfTypes or SurfPublicTypes}
  TADC = record
    DTAcq            : TDTAcq32;
    hbuf             : HBUFTYPE;
    Configured,
    BuffersAllocated : Boolean;
    BuffSize         : Integer;
    hbufAddresses    : array [0..DEFAULT_NUM_BUFFS - 1] of LPUSHRT;
    hbufPtrIndex     : Integer;
    DINOffset        : Integer;
    BoardOffset      : Integer;
    ChansPerBuff     : Integer;
    BuffersCollected : Integer;
    SamplesCollected : Int64;
  end;

  TWindowLoc = record
    Left,Top,Width,Height : integer;
  end;

  TCGList = record
    ProbeId       : integer;
    Channel       : integer;
    Decimation    : integer;
  end;

  TProbeConfig = record {ver 1.3.1}
    NumProbes, NumAnalogChannels : integer;
    BaseSampFreqPerChan : integer;
    AcqDIN : boolean;
    {DT3010DIN : boolean; //true if DT3010 provides DIN display data word, false if DT340}
    CGList : array[0..SURF_MAX_CHANNELS-1] of TCGList; //analog (non-DIN) CGList
    WindowLoc : array[0..SURF_MAX_PROBES-1] of TWindowLoc;
    Setup : TProbeSetup; //local (global) repository for finalised ProbeSetupWin
    MainWinLeft, MainWinTop, MainWinHeight, MainWinWidth : integer;
    InfoWinVisible : boolean;
    InfoWinLeft, InfoWinTop, InfoWinHeight, InfoWinWidth : integer;
    MsgMemoHeight : integer;
    Empty : boolean;
  end;

  CProbeSetupWin = class(TProbeSetupWin)
  public
    function CalcActualFreqPerChan(DesiredSampFreq : integer) : integer; override;
  end;

  CProbeWin = class(TWaveFormPlotForm)
  public
    procedure ThreshChange(pid, seThreshold : integer); override;
    procedure ClickProbeChan(pid, ChanNum : byte); override;
    procedure CreatePolytrodeGUI(pid : integer); override;
  end;

  CAveragerWin = class(TChartWin)
  public
    //AvgRingBuffer : TWaveform; //now public var in base class
    //AvgBufferIndex, AvgTriggerOffset, n : integer;
    {SumRingBuffer, CSDRingBuffer : array of integer;}
    //BuffersEmpty : boolean;
    procedure MoveTimeMarker(MouseXFraction : Single); override;
    procedure OneShotFillBuffers; override;
    procedure RefreshChartPlot; override;
  end;

  TProbeWin = record
    Win : CProbeWin;
    DispTrigOffset, TrigChan, LastTrigChan : integer;
    DispTrigBipolar, DispTrigPositive : boolean;
    Exists : boolean;
  end;

  TContAcqForm = class(TForm)
    SaveConfigDialog: TSaveDialog;
    SURFMainMenu: TMainMenu;
    muData: TMenuItem;
    muConfig: TMenuItem;
    muAbout: TMenuItem;
    tb_acq_buttons: TToolBar;
    sb_play: TSpeedButton;
    sb_record: TSpeedButton;
    sb_stop: TSpeedButton;
    tb_main: TToolBar;
    tb_expinfo: TToolButton;
    tb_vitals: TToolButton;
    large_tb_images: TImageList;
    tb_EEG: TToolButton;
    tb_msg: TToolButton;
    tb_file: TToolBar;
    sml_tb_images: TImageList;
    tbNewConfig: TToolButton;
    tb_saveconfig: TToolButton;
    tbNewDataFile: TToolButton;
    tbCloseDataFile: TToolButton;
    ToolButton1: TToolButton;
    ToolButton4: TToolButton;
    tb_MUX: TToolButton;
    MUXChan: TUpDown;
    muNewFile: TMenuItem;
    TimePanel: TPanel;
    WaveFormPanel: TPanel;
    muCfgSave: TMenuItem;
    muCfgSaveAs: TMenuItem;
    muCfgNew: TMenuItem;
    OpenConfigDialog: TOpenDialog;
    muCfgOpen: TMenuItem;
    N1: TMenuItem;
    MsgPanel: TPanel;
    Splitter1: TSplitter;
    NewDataFileDialog: TSaveDialog;
    muCloseFile: TMenuItem;
    muCfgModify: TMenuItem;
    N2: TMenuItem;
    Exit1: TMenuItem;
    LEDs: TImageList;
    StatusBar: TStatusBar;
    tb_CSD: TToolButton;
    MsgMemo: TMemo;
    procedure PrecisionClockOverflow(Sender: TObject; var lStatus: Integer);
    //procedure ExpTimerTick(Sender: TObject; var lStatus: Integer);
    {procedure DINSSEventDone(Sender: TObject; var lStatus: Integer);}
    procedure tb_MUXClick(Sender: TObject);
    procedure MUXChanClick(Sender: TObject; Button: TUDBtnType);
    procedure FormCreate(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure muAboutClick(Sender: TObject);
    procedure muExitClick(Sender: TObject);
    procedure sb_playClick(Sender: TObject);
    procedure NewConfigClick(Sender: TObject);
    procedure SaveConfigClick(Sender: TObject);
    procedure SaveConfigAsClick(Sender: TObject);
    procedure OpenConfigClick(Sender: TObject);
    procedure tb_expinfoClick(Sender: TObject);
    procedure tb_EEGClick(Sender: TObject);
    procedure tb_msgClick(Sender: TObject);
    procedure MsgMemoChange(Sender: TObject);
    procedure sb_stopClick(Sender: TObject);
    procedure sb_recordMouseUp(Sender: TObject; Button: TMouseButton;
                               Shift: TShiftState; X, Y: Integer);
    procedure NewDatafileClick(Sender: TObject);
    procedure CloseDataFileClick(Sender: TObject);
    procedure ModifyConfigClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure sb_recordClick(Sender: TObject);
    procedure WaveFormPanelClick(Sender: TObject);
    procedure ReduceFlicker(var message : TWMEraseBkgnd); message WM_ERASEBKGND;
    procedure StatusBarDrawPanel(StatusBar: TStatusBar;
                Panel: TStatusPanel; const Rect: TRect);
    procedure MUXChanChangingEx(Sender: TObject; var AllowChange: Boolean;
      NewValue: Smallint; Direction: TUpDownDirection);
    procedure tb_CSDClick(Sender: TObject);
  private
    ProbeWin : array[0..SURF_MAX_PROBES - 1] of TProbeWin;
    EEGWin   : array[0..MAX_EEG_WINDOWS - 1] of TEEGWin;
    EEGWinsOpen : set of 0..SURF_MAX_CHANNELS;
    EEGWinIndex : integer; //redundant with EEGWinOpenset?
    SelectEEGChannel : boolean;
    InfoWin  : TInfoWin;
    AvgWin   : CAveragerWin;
    CSDChannelList : set of 0..SURF_MAX_CHANNELS;
    AvgWinCreated, AvgFillBuffer : boolean;
    ProbeSetupWinCreated : boolean;
    CfgFileName : string;
    GotCfgFileName : boolean;
    Config : TProbeConfig;
    SurfFile : TSurfFile;

    LastDINStatus, SweepChecksum : word;
    StimulusDisplayRunning, StimulusDisplayPaused, StopPauseRecordMode, GotValidDSHeader : boolean;
    StimulusHeader : TStimulusHeader;
    DSPRecord      : SURF_DSP_REC;
    DINHeaderBuffer : array[0..(SizeOf(TStimulusHeader) div 2)-1] of word;
    SSRingBuffer    : array{[0..SSRINGARRAYSIZE - 1]} of SURF_SS_REC; //dynamic, depends on #save probes
    SVRingBuffer    : array[0..SVRINGARRAYSIZE - 1] of SURF_SV_REC;
    MsgRingBuffer   : array[0..MSGRINGARRAYSIZE -1] of SURF_MSG_REC;
    DeMUXedDTBuffer : TWaveform; //array of SHRT{word};

    SSRINGARRAYSIZE : integer;
    SSRingBufferIndex, SSRingSaveIndex : integer;
    SVRingBufferIndex, SVRingSaveIndex : integer;
    MsgRingBufferIndex, MsgRingSaveIndex : integer;
    StimulusSweepsRemaining, StimulusTimeRemaining, NumDINHeader, NumDINFrames,
      TotalNumFrames, LastTotalFrames, VSyncLagUsec : integer;

    DT340Installed, MUXDOUTEnabled, ExpTimerEnabled, MasterClockEnabled, DT32BitCounterEnabled,
      StimulusDINEnabled, Acquiring, Recording, PauseRecordMode, RecordingPaused, DataFileOpen : boolean;
    NumADBoards, MaxADChannels, ADCFrequency{actual},
      MaxSampFreqPerChan, MinSampFreqPerChan, SampFreqPerChan{actual},
      SamplePeriod{usec}, SampPerChanPerBuff, DTBandwidth, NumSaveProbes, FileBandwidth : integer;
    ADCSaturated, DTUnderrrunError{, BoardsOutOfSync }: boolean;
    DTQuery, DTMUX, {DTExpTimer,} DTMasterClock, DT32BitCounter : TDTAcq32;
    WinExpTimer : TTimer;

    Time32BitOverflow, BufferStartTime : Int64;
    Old32BitTime : UINT;
    ExpTime, RecTime, TotalRecTime, ErrorTimeCount, MUXTimeCount : integer;
    NotBoardInQueue : integer;

    ADC : array of TADC;
    {DTDIN : array of TDTAcq32;}

    procedure StartStopAcquisition;
    procedure StartStopRecording;
    procedure ExpTimerTick(Sender: TObject);
    procedure DTAcqBufferDone(Sender: TObject);
    procedure PutPTRecordsOnFileBuffer(Board : integer);
    procedure DTAcqOverrun(Sender: TObject);
    procedure DTAcqQueueUnderrun(Sender: TObject);
    procedure DTBoardsOutOfSync;
    procedure CopyChannelsFromDTBuffer({const}hbufAddress : LPUSHRT;
                                        const Board : integer);
    procedure CopyChannelsToAverager(BoardTag : integer);
    function  FindThresholdXing(hbufAddress : LPUSHRT; const ProbeID : integer) : integer;
    procedure DecodeStimulusDIN(BoardTag : integer);
    function  ValidateStimulusHeader : boolean; //false if error
    procedure PostSurfMessage(const MessageString : string;
                              const SpecifiedTimeStamp : Int64 = 0{default};
                              const MsgType : char = SURF_MESSAGE{default});
    procedure FlagCriticalError;
    procedure RefreshFileInfo;
    procedure RefreshBufferInfo;
    procedure RefreshStimulusInfo;
    procedure ResetStimulusInfo;
    procedure SetupProbes;
    procedure FreeDTBuffers;
    procedure PutWindowLocationsInConfig(Probe : integer);
    procedure GetWindowLocationsFromConfig(Probe : integer);
    procedure FreeChanWindows;
    procedure SaveSurfConfig(Filename : string);
    procedure OpenSurfConfig(Filename : string);
    procedure WriteSurfLayoutRecords;
    function  CreateAProbeWindow(probenum,left,top,npts : integer) : boolean;
    function  ConfigBoard : boolean; //false if error
    function  GetPrecisionTime : Int64;
    procedure AddRemoveCSDWin;
    procedure CheckDisplayPaused;
    procedure CheckForPTRecordsToSave;
    procedure CheckForSVRecordsToSave;
    procedure CheckForMsgRecordsToSave;
    procedure CloseDataFile;
    { Private declarations }
  public
    function  Chan2MUX(Channel : byte): boolean; //false if error
    procedure AcceptFiles( var msg : TMessage ); message WM_DROPFILES;
    { Public declarations }
  end;

var
  ContAcqForm: TContAcqForm;

implementation

{$R *.DFM}
{-------------------------------------------------------------------------}
procedure TContAcqForm.FormCreate(Sender: TObject);
begin
  DragAcceptFiles(Handle, True);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.AcceptFiles( var msg : TMessage );
const
  cnMaxFileNameLen = 255;
var
  acFileName : array [0..cnMaxFileNameLen] of char;
begin
  if Acquiring or DataFileOpen then Exit;
  //returns how many files were dragged on form...
  DragQueryFile(msg.WParam, $FFFFFFFF, acFileName, cnMaxFileNameLen);
  Application.BringToFront;
  //...take the first one
  DragQueryFile(msg.WParam, 0, acFileName, cnMaxFileNameLen );
  DragFinish(msg.WParam);
  OpenSurfConfig(acfilename);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.FormShow(Sender: TObject);
var b, ADSSIndex : integer; EssentialResourceError : boolean;
begin
  //initialise global variables -- any missing here?
  //or is any of this even needed if form dynamically (re-)created at runtime?
  Acquiring := False;
  Recording := False;
  PauseRecordMode:= False;
  StopPauseRecordMode:= False;
  //BoardsOutOfSync:= False;
  DataFileOpen := False;
  DTUnderrrunError:= False;
  DT340Installed:= False;
  MUXDOUTEnabled:= False;
  ExpTimerEnabled:= False;
  StimulusDINEnabled:= False;
  MasterClockEnabled:= False;
  DT32BitCounterEnabled:= False;
  ProbeSetupWinCreated:= False;
  EssentialResourceError:= False;

  GotCfgFileName:= False;
  CfgFileName:= '';
  Config.NumProbes:= 0;
  Config.NumAnalogChannels:= 0;
  Config.AcqDIN:= False;
  Config.Setup.TotalChannels:= 0;
  Config.Empty:= True;

  ClientWidth:= 0;
  ClientHeight:= 0;
  InfoWin:= TInfoWin.CreateParented(WaveFormPanel.Handle);
  InfoWin.Top:= 100;
  InfoWin.Left:= ClientWidth - InfoWin.Width - 10;
  InfoWin.ProgressBar1.Max:= DEFAULT_NUM_BUFFS;
  InfoWin.ProgressBar2.Max:= DEFAULT_NUM_BUFFS;
  InfoWin.ProgressBar3.Max:= DEFAULT_NUM_BUFFS;
  InfoWin.ProgressBar4.Max:= DEFAULT_NUM_BUFFS;
  InfoWin.ProgressBar5.Max:= DEFAULT_NUM_BUFFS;
  InfoWin.ProgressBar6.Max:= DEFAULT_NUM_BUFFS;
  EEGWinIndex:= 0;

  SurfFile:= TSurfFile.Create; //SurfFile methods class

  TimePanel.ControlStyle:=TimePanel.ControlStyle + [csOpaque]; //reduce flicker
  WaveFormPanel.ControlStyle:=  WaveFormPanel.ControlStyle + [csOpaque];
  WaveFormPanel.DoubleBuffered:= False; //not necessary

  Randomize; //initialise random number generator

  {determine number of DT3010 & DT340 boards and assign DTxEz
   controls to all SURF-relevant subsystems; for ease of indexing,
   all A/D subsystems are grouped in a single dynamic array TADC}
  try
    DTQuery:= TDTAcq32.Create(Self);//create control for probing hardware resources
    if DTQuery.NumBoards > 0 then
    begin
      ADSSIndex:= 0;
      for b:= 0 to DTQuery.NumBoards - 1 do
      begin
        if Pos('3010', DTQuery.BoardList[b]) > 0 then
        begin //first, create and assign controls for all DT3010 A/D subsystems
          SetLength(ADC, ADSSIndex + 1);
          ADC[ADSSIndex].DTAcq:= TDTAcq32.Create(Self);
          with ADC[ADSSIndex].DTAcq do
          begin
            Board:= DTQuery.Boardlist[b];
            SubSystem := OLSS_AD;
            if GetSSCaps(OLSSC_SUP_CONTINUOUS) <> 0
              then DataFlow := OL_DF_CONTINUOUS //continuous mode operation
            else begin
              ShowMessage(Board + ' cannot support continuous acquisition');
              EssentialResourceError:= True;
              SetLength(ADC, ADSSIndex - 1);
              Break;
            end;
            if GetSSCaps(OLSSC_SUP_SINGLEENDED) <> 0 then
              ChannelType := OL_CHNT_SINGLEENDED //expect single-ended signals
            else begin
              ShowMessage(Board + ' cannot support single-ended acquisition');
              EssentialResourceError:= True;
              SetLength(ADC, ADSSIndex - 1);
              Break;
            end;
            if GetSSCaps(OLSSC_SUP_BINARY) <> 0 then
              Encoding := OL_ENC_BINARY //setup for binary encoding
            else begin
              ShowMessage(Board + ' cannot support binary encoding');
              EssentialResourceError:= True;
              SetLength(ADC, ADSSIndex - 1);
              Break;
            end;
            WrapMode := OL_WRP_NONE; //no buffer wrapping
            if ADSSIndex = 0 then //first DT3010 board   -- NO LONGER provides the master ADC clock...
            begin //... in order to synchronise timestamp clock with ADC, all ADCs use (external) user clock
              {if ADC[0].DTAcq.GetSSCaps(OLSSC_SUP_INTCLOCK) <> 0 then
              begin
                ADC[0].DTAcq.ClockSource := OL_CLK_INTERNAL;
                ADC[0].DTAcq.Frequency := 1e6; //query maximum ADC sample frequency
                ADCMaxFreq := Round(ADC[0].DTAcq.Frequency);
                ADC[0].DTAcq.Frequency := 0; //query minimum ADC sample frequency
                ADCMinFreq := Round(ADC[0].DTAcq.Frequency);
                if (MASTER_CLOCK_FREQUENCY >= ADCMinFreq) and (MASTER_CLOCK_FREQUENCY <= ADCMaxFreq)
                  then ADC[0].DTAcq.Frequency:= MASTER_CLOCK_FREQUENCY
                else begin
                  ShowMessage('Default ADC sample rate outside range of ' + ADC[0].DTAcq.Board + ' board');
                  Exit;
                end;
              end else
              begin
                ShowMessage(ADC[0].DTAcq.Board + ' cannot support internal sample clock');
                Exit;
              end;}
              if GetSSCaps(OLSSC_SUP_EXTCLOCK) <> 0 then
                ClockSource := OL_CLK_EXTERNAL else //gets input from master sample clock
              begin
                ShowMessage(Board + ' cannot support external sample clock');
                EssentialResourceError:= True;
                SetLength(ADC, ADSSIndex - 1);
                Break;
              end;
              ClockDivider:= 1; //same as the master precision clock/timer
              if (MASTER_CLOCK_FREQUENCY >= GetSSCapsEx(OLSSCE_MAXTHROUGHPUT))
                or (MASTER_CLOCK_FREQUENCY <= GetSSCapsEx(OLSSCE_MINTHROUGHPUT)) then
                begin
                  Showmessage('Default external ADC clock frequency outside range of ' + Board + ' board');
                  EssentialResourceError:= True;
                  SetLength(ADC, ADSSIndex - 1);
                  Break;
                end;
              if GetSSCaps(OLSS_SUP_RETRIGGER_INTERNAL) <> 0 then
              begin //first DT3010 board provides master (internal) retrigger clock
                ReTriggerMode:= OL_RETRIGGER_INTERNAL;
                TriggeredScan:= 1;
                MultiScanCount:= 1; //scan through CG list only once per trigger
                MinSampFreqPerChan:= Round(GetSSCapsEx(OLSSCE_MINRETRIGGER));
                MaxSampFreqPerChan:= Round(GetSSCapsEx(OLSSCE_MAXRETRIGGER));
                if (DEFAULT_SAMPLE_FREQ_PER_CHAN >= MinSampFreqPerChan)
                  and (DEFAULT_SAMPLE_FREQ_PER_CHAN <= MaxSampFreqPerChan) then
                  RetriggerFreq:= DEFAULT_SAMPLE_FREQ_PER_CHAN
                else begin
                  ShowMessage('Default sample rate per channel outside ' + Board + '''s retrigger frequency range');
                  EssentialResourceError:= True;
                  SetLength(ADC, ADSSIndex - 1);
                  Break;
                end;
              end else
              begin
                ShowMessage(Board + ' cannot support internal retriggered scan mode');
                EssentialResourceError:= True;
                SetLength(ADC, ADSSIndex - 1);
                Break;
              end;
              try
                Config; //although CGlist not yet defined, configuring allows the precise default...
                SampFreqPerChan:= Round(RetriggerFreq); //...sample rate per chan to be determined
              except
                ShowMessage(Board + ' could not be configured');
                EssentialResourceError:= True;
                SetLength(ADC, ADSSIndex - 1);
                Break;
              end;
            end{first DT3010} else
            begin //this is not the first DT3010 board, so both retrigger and sample clock are external
              if GetSSCaps(OLSSC_SUP_EXTCLOCK) <> 0 then
                ClockSource := OL_CLK_EXTERNAL else //gets input from master sample clock
              begin
                ShowMessage(Board + ' cannot support external sample clock');
                SetLength(ADC, ADSSIndex - 1);
                Continue;
              end;
              ClockDivider:= 1; //the same as the Master sample clock
              if GetSSCaps(OLSSC_SUP_RETRIGGER_EXTRA) <> 0 then
              begin
                TriggeredScan:= 1; //enable triggered scan mode
                ReTriggerMode:= OL_RETRIGGER_SCAN_PER_TRIGGER;
                Trigger:= OL_TRG_EXTRA{EXTERN}; //MUST use OL_TRG_EXTRA as trigger output from board 0 is -ve
                ReTrigger:= OL_TRG_EXTRA{EXTERN}; //MUST use OL_TRG_EXTRA as trigger output from board 0 is -ve
                MultiScanCount:= 1; //scan through CG list only once per trigger
              end else
              begin
                ShowMessage(Board + ' cannot support external (-ve) triggered scan mode');
                SetLength(ADC, ADSSIndex - 1);
                Continue;
              end;
              try
                Config; //final check for subsys availability, or any configuration errors
              except
                Showmessage(Board + ' could not be configured');
                SetLength(ADC, ADSSIndex - 1);
                Continue;
              end;
            end;
            inc(MaxADChannels, GetSSCaps(OLSSC_MAXSECHANS)); //tally A/D channels
            if MaxADChannels > SURF_MAX_CHANNELS then MaxADChannels := SURF_MAX_CHANNELS;
            inc(ADSSIndex);
          end{with ADC[ADSSIndex].DTAcq};
          if MUXDOUTEnabled = False then //default configuration assigns first available
          begin                          //DT3010 DOUT subsystem to MUX/monitor control
            DTMUX:= TDTAcq32.Create(Self);
            with DTMUX do
            begin
              Board:= DTQuery.BoardList[b];
              SubSysType := OLSS_DOUT;
              SubSysElement:= 0;
              Resolution:= 8;
              DataFlow:= OL_DF_SINGLEVALUE;
              try
                Config;
                MUXDOUTEnabled:= True;
                tb_MUX.Down:= True;
                Chan2Mux(MUXChan.Position); //reset MUX controller
              except
                Free;
                tb_MUX.Caption:='Disabled';
              end;
            end;
          end;
        end{DT3010} else
        if Pos('340', DTQuery.BoardList[b]) > 0 then
        begin //default assigns controls for 32bit timestamp counter & experiment timer, both with interrupts
          DT340Installed:= True;
          {the commented-out code for configuring the DT340's "interrupt on bit change" method
           for collecting stimulus display information is too unreliable:- the windows message
           stack can be delayed INDEFINITELY(!), even with SURF runtime priority HIGH),
           during which time all acquisition of stimulus DIN is halted/delayed. Instead, the
           stimulus display status byte will be read continuously at ~30KHz by the first
           available DT3010 board, and stimulus data words read continuously from the
           second DT3010 will be stored only when the frame/data bit is toggled}

          {DTDIN[0]:= TDTAcq32.Create(Self);
          with DTDIN[0] do
          begin
            Board:= DTQuery.Boardlist[b]; //use first available DT340:
            SubSysType:= OLSS_DIN; // ii) for stimulus status byte
            SubSysElement:= 3; //DT340 interrupt port D...
            Resolution:= 8; //...AOB6 port C
            DataFlow:= OL_DF_CONTINUOUS; //continuous, interrupt driven
            OnSSEventDone:= DINSSEventDone; //assign interrupt handle
          end;
          Setlength(DTDIN, 1);
          DTDIN[0]:= TDTAcq32.Create(Self); // i) for stimulus information
          with DTDIN[0] do
          begin
            Board:= DTQuery.Boardlist[b]; //use first available DT340
            SubSysType:= OLSS_DIN;
            SubSysElement:= 0; //DT340 ports A & B...
            Resolution:= 16;   //map to AOB6 ports A & B
            DataFlow:= OL_DF_SINGLEVALUE; //will use oldmGetSingleValue method
          end;
          try
            DTDIN[0].Config;
            DTDIN[1].Config;
            StimulusDINEnabled:= True;
          except
            Setlength(DTDIN, 0);
          end;}
          { no longer used either... using Delphi TTimer for exp timer as precision not critical...
          DTExpTimer:= TDTAcq32.Create(Self); //configure experiment timer (0.1 sec precision)
          with DTExpTimer do
          begin
            Board:= DTQuery.Boardlist[b];
            SubSysType:= OLSS_CT;
            SubSysElement:= 8; //first of the 24bit interval timers
            ClockSource := OL_CLK_INTERNAL;
            Frequency:= EXP_TIMER_FREQ; //1/10th second experiment timer
            CTMode:= OL_CTMODE_RATE;
            GateType:= OL_GATE_NONE; //software start/stop
            OnSSEventDone:= ExpTimerTick; //assign the the interrupt handle
            try
              Config;
              ExpTimerEnabled:= True;
            except
              Free;
            end;
          end; }

          {DT32BitCounter:= TDTAcq32.Create(Self); //32bit counter for precision timestamp clock...
          with DT32BitCounter do //...used in preference to DT3010 as DT340 generates interrupt upon overflow
          begin
            Board:= DTQuery.Boardlist[b];
            SubSysType:= OLSS_CT;
            SubSysElement:= 4;
            CascadeMode:= OL_CT_CASCADE; //internally cascaded clocks (CT4 & CT5)
            ClockSource:= OL_CLK_EXTERNAL;//output from DT3010 CT0 --> input of DT340 CT0
            CTMode:= OL_CTMODE_COUNT;
            GateType:= OL_GATE_NONE; //software start/stop
            OnSSEventDone:= PrecisionClockOverflow; //assign the interrupt handle
          {end;
          DTMasterClock:= TDTAcq32.Create(Self); //precision clock -- NOT USED AS NON-TTL OUTPUT!
          with DTMasterClock do
          begin
            Board:= DTQuery.Boardlist[b];
            SubSysType:= OLSS_CT;
            SubSysElement:= 0;
            ClockSource:= OL_CLK_INTERNAL;
            Frequency:= MASTER_CLOCK_FREQUENCY; //1MHz sample clock, 1usec timestamp
            CTMode:= OL_CTMODE_RATE;
            GateType:= OL_GATE_NONE; //software start/stop
          end;
            try
              Config;
              DT32BitCounterEnabled:= True;
            except
              Free;
            end;
          end{DT32BitCounter;}
        end{DT340};
      end{b};
      NumADBoards:= ADSSIndex;
      {the next section checks for default subsystems not yet assigned to
       a DT340 board, and attempts to assign them to available DT3010 subsystems}
      if NumADBoards > 0 then
      begin
        if StimulusDINEnabled = False then
          if NumADBoards > 1 then //two or more DT3010's needed for DT3010-based stimulus DIN
            StimulusDINEnabled:= True {continuous mode for DT3010 DINs -- no configuration needed here}
          else Showmessage('Two free DIN subsystems not available. Stimulus acquisition disabled.');
          {Setlength(DTDIN, 1);
          DTDIN[0]:= TDTAcq32.Create(Self);
          with DTDIN[0] do
          begin
            Board:= ADC[1].DTAcq.DeviceName; //no DT340, so use second available DT3010...
            SubSysType:= OLSS_DIN;           //for stimulus information
            SubSysElement:= 0; //ports A & B...
            Resolution:= 16; //...AOB6 ports A and B
            DataFlow:= OL_DF_SINGLEVALUE; //use get SV method
            //nb: stimulus status port acquired as CH32 of first A/D subsystem - CONFIGURED LATER
            try
              Config;
               StimulusDINEnabled:= True;
            except
              SetLength(DTDIN, 0);
              Showmessage('No available DIN subsystems. Stimulus acquisition disabled.');
            end;
          end;
        end;}
       {DTExpTimer:= TDTAcq32.Create(Self);
          with DTExpTimer do
          begin
            Board:= ADC[0].DTAcq.DeviceName; //use first DT3010
            SubSysType:= OLSS_CT;
            SubSysElement:= 3; //user timer 4
            ClockSource := OL_CLK_INTERNAL;
            Frequency:= EXP_TIMER_FREQ; //1/10th second experiment timer
            CTMode:= OL_CTMODE_RATE;
            GateType:= OL_GATE_NONE; //software start/stop
            try
              Config;
              ExpTimerEnabled:= True;
            except
              Showmessage('No suitable DT clocks available for experiment timer. Using Windows timer.');
              Free;
            end;
          end;
        //end{DTExpTimer};
        if DT32BitCounterEnabled = False then //use first DT3010 instead
        begin
          DT32BitCounter:= TDTAcq32.Create(Self); //32 bit counter for precision clock
          with DT32BitCounter do
          begin
            Board:= ADC[0].DTAcq.DeviceName;
            SubSysType:= OLSS_CT;
            SubSysElement:= 1; //user timers 2 & 3...
            CascadeMode:= OL_CT_CASCADE; //...internally cascaded counters
            ClockSource:= OL_CLK_EXTERNAL;//output from CT0 --> input of CT1
            //ClockDivider:= 10; //for some reason this has no effect
            CTMode:= OL_CTMODE_COUNT;
            GateType:= OL_GATE_NONE; //software start/stop
            try
              Config;
              DT32BitCounterEnabled:= True;
            except
              Showmessage('No suitable DT counters available for precision 32 bit timestamp. Unable to run SURF');
              Free;
              EssentialResourceError:= True;
            end;
          end;
        end{DT32BitCounter};
        if MasterClockEnabled = False then //use first DT3010 instead
        begin
          DTMasterClock:= TDTAcq32.Create(Self); //precision clock
          with DTMasterClock do
          begin
            Board:= ADC[0].DTAcq.DeviceName;
            SubSysType:= OLSS_CT;
            SubSysElement:= 0; //user timer 1
            ClockSource:= OL_CLK_INTERNAL;
            Frequency:= MASTER_CLOCK_FREQUENCY;
            CTMode:= OL_CTMODE_RATE;
            GateType:= OL_GATE_NONE; //software start/stop
            PulseType:= OL_PLS_HIGH2LOW;
            PulseWidth:= 60; //emulate duty cycle of internal ADC sample clock
            try
              Config;
              ADCFrequency:= Round(DTMasterClock.Frequency); //confirm if default A/D sample clock freq achievable
              MasterClockEnabled:= True;
              if Frequency <> MASTER_CLOCK_FREQUENCY then
                Showmessage('Warning: Master clock frequency only approximates ' +
                             inttostr(MASTER_CLOCK_FREQUENCY)+'Hz default specified.');
            except
              Showmessage('No suitable DT clocks available for precision master clock. Unable to run SURF');
              Free;
              EssentialResourceError:= True;
            end;
          end;
        end{DTMasterClock};
      end;
      if ExpTimerEnabled = False then //use Delphi TTimer for Exp Timer...
      begin
        WinExpTimer:= TTimer.Create(Self);
        with WinExpTimer do
        begin
          Enabled:= False;
          OnTimer:= ExpTimerTick;
          Interval:= 1000 div EXP_TIMER_FREQ;
        end;
        ExpTimerEnabled:= True;
      end;{ExpTimer}
    end else
    begin
      ShowMessage('No supported DT boards installed');
      EssentialResourceError:= True;
    end;
    if NumADBoards = 0 then
    begin
      ShowMessage('No supported A/D boards installed');
      EssentialResourceError:= True;
    end;
  finally
    DTQuery.Free; //free control created for probing hardware
    if EssentialResourceError then Application.Terminate;
  end;
end;

{-------------------------------------------------------------------------}
function TContAcqForm.ConfigBoard : boolean; //false if error
var i, b, BoardIndex, NumInhibitedChans : integer;
 ListIndex : array of integer;
begin
  ConfigBoard:= True;

  if Config.NumAnalogChannels > (DT3010_MAX_SE_CHANS * NumADBoards) then
  begin
    Showmessage('Available hardware resources cannot accommodate number of A/D channels.');
    ConfigBoard:= false;
    Exit;
  end;

  {program CGL, including display stimulus DIN, if so configured}
  SetLength(ListIndex, NumADBoards);
  for i:= 0 to NumADBoards - 1 do
    ListIndex[i]:= 0; //initialise local array

  if Config.AcqDIN then
  begin
    if StimulusDINEnabled then
    begin
      for b:= 0 to 1 do
      begin
        ADC[b].DTAcq.ChannelList[0]:= DIN_PROBE; //stimulus display ports are acquired...
        ADC[b].DTAcq.GainList[0]:= 1; //...as first CGL entrys of DT3010A/B A/D subsystems
        inc(ListIndex[b]);
      end;
      InfoWin.EnableStimulusInfo;
    end else
    begin
      Showmessage('Available hardware resources cannot accommodate stimulus display DIN.');
      ConfigBoard:= False;
      Exit;
    end;
  end else
    InfoWin.DisableStimulusInfo;

  for i := 0 to Config.NumAnalogChannels - 1 do
  begin
    BoardIndex:= Config.CGList[i].Channel div DT3010_MAX_SE_CHANS;
    ADC[BoardIndex].DTAcq.ChannelList[ListIndex[BoardIndex]]:=
      Config.CGList[i].Channel mod DT3010_MAX_SE_CHANS;
    ADC[BoardIndex].DTAcq.GainList[ListIndex[BoardIndex]]:=
      Config.Setup.Probe[Config.CGList[i].ProbeId].InternalGain;
    inc(ListIndex[BoardIndex]);
  end;

  {program inhibit list for Master board, if necessary i.e. if any Slave
   board has more channels configured than the Master board.  This ensures
   the Master A/D sample and retrigger clocks continue to pace/trigger any slave
   board(s) with channel list-sizes greater than that of the Master board}
  NumInhibitedChans:= 0;
  for b:= 1 to NumADBoards - 1 do
    if ListIndex[b] > ListIndex[0] then
      begin
        Showmessage('Number of probe channels on slave board(s) more than master board.');
        ConfigBoard:= false;
        Exit;
      end;

  {BUG with DT-Ez driver: the following code is replaced by the preceeding warning message
    because inhibiting (any number of) channels of the master board reduces the sample rate of
    the second (slave) board by half (irrespective of how many more channels are in second board).
    This is not due to 'inhibition' of clocks or triggers -- they still output according to
    board 0's listsize (this was checked on an oscilloscope).  Until this bug is figured out,
    the user is limited to configuring probes such that the number of channels on slave board(s)
    must be less than or equal to that of the master board.}
      {while ListIndex[0] < ListIndex[b] do
      begin
        ADC[0].DTAcq.ChannelList[ListIndex[0]]:= 1; //arbitrary, as channel will be inhibited
        ADC[0].DTAcq.GainList[ListIndex[0]]:= 1; //again, arbitrary, as channel will be inhibited
        ADC[0].DTAcq.InhibitList[ListIndex[0]]:= True;
        inc(ListIndex[0]);
        inc(NumInhibitedChans);
      end;}
  {Driver BUG! The following line is needed to circumvent "Invalid inhibit list" error bug,
   since it appears, although not documented, that all channels in a CGLIST cannot be inhibited.
   This line is not needed if the master DT3010 board uses DIN (ch32), but if it is needed, then
   'undesired' data from the un-inhibited channel 0 will fill buffers and needs to be discarded...}
  //ADC[0].DTAcq.InhibitList[0]:= False;

  FreeDTBuffers; //release any existing buffers from previous configs/setups

  {program final CG listsize for each board}
  for i:= 0 to NumADBoards - 1 do
    if ListIndex[i] <> 0 then ADC[i].DTAcq.ListSize:= ListIndex[i]
      else ADC[i].Configured:= False;

  {program sampling frequency -- master board only}
  ADC[0].DTAcq.RetriggerFreq:= SampFreqPerChan;
  {allocate buffers, put on ready queue, and finalise configuration of A/D subsystems}
  Dec(ListIndex[0], NumInhibitedChans); //adjust buffersize of master board for inhibited channels
  for i:= 0 to NumADBoards - 1 do
  begin
    if ListIndex[i] <> 0 then
      with ADC[i] do
      begin
        ChansPerBuff:= ListIndex[i];
        BuffSize:= SampFreqPerChan * ChansPerBuff div BUFFERS_PER_SECOND; //default: 10Hz bufferdone events/board
        for b := 0 to DEFAULT_NUM_BUFFS - 1 do
        begin
          if ErrorMsg(olDmAllocBuffer(0, BuffSize, @hbuf)) then
          begin
            ConfigBoard := False;
            BuffersAllocated := False;
            Showmessage('Error allocating buffers');
            Exit;
          end else
          begin
            DTAcq.Queue := {ULNG}hbuf; //put on Ready queue
            ErrorMsg(oldmGetBufferPtr(hbuf, hbufAddresses[b])); //add to buffer ptr list
          end;
        end;
        DTAcq.Config;
        Configured:= True;
        BuffersAllocated := True;
        DTAcq.OnBufferDone:= DTAcqBufferDone;
        DTAcq.OnQueueDone:= DTAcqQueueUnderrun;
        DTAcq.OnOverrunError:= DTAcqOverrun;
        DTAcq.Tag:= i; //tag is used to identify sender object for BufferDone events
        hbufPtrIndex:= 0;
      end{ADC[i]};
  end{i};
  SampFreqPerChan{undecimated, actual}:= Round(ADC[0].DTAcq.RetriggerFreq);
  SamplePeriod:= Round(1 / SampFreqPerChan * MASTER_CLOCK_FREQUENCY); //precompute ADC period, in usec...
  SampPerChanPerBuff:= SampFreqPerChan div BUFFERS_PER_SECOND; //ditto

  {free and reallocate SSRINGARRAY and deMUXBuffer, taking into account any decimated channels}
  for b:= 0 to High(SSRingBuffer) do //free any dynamic arrays in SSRingBuffer
    SSRingBuffer[b].ADCWaveform:= nil;
  SSRingBuffer:= nil;
  SSRINGARRAYSIZE:= 0; //SSRINGARRAY takes account of probes with 'save' unchecked
  for i:= 0 to Config.NumProbes - 1 do
    if Config.Setup.Probe[i].Save then inc(SSRINGARRAYSIZE, BUFFERS_PER_SECOND div DISK_WRITES_PER_SECOND);
  SetLength(SSRingBuffer, SSRINGARRAYSIZE);

  DeMUXedDTBuffer:= nil;
  b:= 0;
  for i:= 0 to Config.NumAnalogChannels - 1 do
  begin
    b:= b + (SampPerChanPerBuff div Config.CGLIST[i].Decimation);
    ADC[Config.CGList[i].Channel div DT3010_MAX_SE_CHANS].BoardOffset:= b;
  end;
  SetLength(DeMUXedDTBuffer, b * BUFFERS_PER_SECOND div DISK_WRITES_PER_SECOND);

  {pre-calculate commonly used indicies for CPU-intensive procedures}
  for b:= 1 to NumADBoards - 1 do
    ADC[b].BoardOffset:= ADC[b-1].BoardOffset;
  ADC[0].BoardOffset:= 0;
  for b:= 0 to NumADBoards - 1 do
    if Config.AcqDIN and (b < 2) then ADC[b].DINOffset:= 1
      else ADC[b].DINOffset:= 0;

  {finally, tally DT/file bandwidth, including estimate of stimulus DIN}
  DTBandwidth:= 0;
  NumSaveProbes:= 0;
  FileBandwidth:= 0;
  for i:= 0 to Config.NumAnalogChannels - 1 do
  begin
    inc(DTBandWidth, Config.Setup.Probe[Config.CGList[i].ProbeID].SampFreq);
    if Config.Setup.Probe[Config.CGList[i].ProbeID].Save then
      inc(FileBandWidth, Config.Setup.Probe[Config.CGList[i].ProbeID].SampFreq * 2{12 bit--> 2 bytes});
  end;
  if Config.AcqDIN then inc(FileBandWidth, SizeOf(SURF_SV_REC) * 100{assumes 100Hz continuous display});

  for i:= 0 to Config.NumProbes - 1 do
    if Config.Setup.Probe[i].Save then inc(NumSaveProbes);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.FreeDTBuffers;
var i, b : integer;
begin
  for i:= 0 to NumADBoards - 1 do
    with ADC[i] do
    begin
      if Configured then
      begin
        DTAcq.Flush; //transfer all buffers to done queue
        for b:= 0 to DEFAULT_NUM_BUFFS - 1 do
        begin
          hbuf := DTAcq.Queue; //get buffer from done queue...
          if hBuf <> null then {if ErrorMsg(}olDmFreeBuffer(hBuf);{)} //... and free it!
        end{b};
        Configured:= False;
        BuffersAllocated:= False;
      end;
    end{ADC[i]};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.tb_MUXClick(Sender: TObject);
begin
  MUXchan.Enabled:= not(MUXchan.Enabled);
  if MUXchan.Enabled then
  begin
    tb_MUX.Down:= True;  //re-select MUX channel
    if not (Chan2MUX(MUXChan.Position)) then tb_MUX.Caption:= '-- error -- '
      else tb_MUX.Caption:= 'CH ' + inttostr(MUXChan.Position);
  end else
  begin  //disable MUX
    tb_MUX.Down:= False;
    try
      DTMUX.PutSingleValue(0, 1, 1); //disables MUX by holding reset high
      tb_MUX.Caption:= 'MUX off';
    except
      tb_MUX.Caption:= '-- error -- ';
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.MUXChanChangingEx(Sender: TObject;
  var AllowChange: Boolean; NewValue: Smallint;  Direction: TUpDownDirection);
var p : integer;
begin
  AllowChange:= False;
  if MUXChan.Enabled = False then Exit;
  if (NewValue < MUXChan.Min) or (NewValue > MUXChan.Max) then Exit;
  for p:= 0 to Config.NumProbes - 1 do
  begin  //update polytrodeGUI (if present) for current MUX chan...
    with Probewin[p].win do
    begin
      if GUICreated then
      with Config.Setup.Probe[p] do
      begin
        if (MUXChan.Position >= ChanStart) and
           (MUXChan.Position <= ChanEnd) then
             GUIForm.ChangeSiteColor(MUXChan.Position - ChanStart); //erase old MUX chan...
        if (NewValue >= ChanStart) and
           (NewValue <= ChanEnd) then
             GUIForm.ChangeSiteColor(NewValue - ChanStart, clRed);  //...and highlight new
        GUIForm.Invalidate;
      end;
    end{with};
  end{p};
  AllowChange:= True;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.MUXChanClick(Sender: TObject; Button: TUDBtnType);
begin
  if not (Chan2MUX(MUXChan.Position)) then tb_MUX.Caption:= '-- error -- '
    else tb_MUX.Caption:= 'CH ' + inttostr(MUXChan.Position);
end;

{-------------------------------------------------------------------------}
function TContAcqForm.Chan2MUX(Channel : byte) : boolean;
var i : integer;
begin
  if (MUXDOUTEnabled = false) or (Channel > MAX_MUX_CHANNELS - 1{zero base})
    then Result:= False else
  try
    DTMUX.PutSingleValue(0, 1, 1); //reset...
    for i:= 0 to Channel do
    begin
      DTMUX.PutSingleValue(0, 1, 2); //...and count
      DTMUX.PutSingleValue(0, 1, 0);
    end;
    Result:= True;
  except
    Result:= False;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.muAboutClick(Sender: TObject);
begin
  //AboutBox.ShowModal;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.muExitClick(Sender: TObject);
begin
  //checks need to be here to see if files are saved/closed
  //also DT methods need to be stopped, buffers flushed, and object/memory allocation freed
  Close;
end;

{-------------------------------------------------------------------------}
{procedure TContAcqForm.DINSSEventDone(Sender: TObject; var lStatus: Integer);
begin
  DINDisplayStatus:= (lStatus and $00FF0000) shr 16; //retrieve AOB6 port C (display status)
  if (DINDisplayStatus and DIN_DISP_RUNNING) = 0 then Exit; //ignore interrupt if display not 'running'
  if (DINDisplayStatus and DIN_DISP_SWEEP) = 0 then
  begin //interrupt is data strobe for display parameter header...
    DINHeaderTime[NumDINHeader]:= DT32BitCounter.CTReadEvents;
    DINHeaderData[NumDINHeader]:= DTDIN[1].GetSingleValue(0,1) shr 8;
    inc(NumDINHeader);
  end else
  begin //...otherwise interrupt is frame toggle during stimulus sweep
    DINHeaderTime[NumDINFrames]:= DT32BitCounter.CTReadEvents;
    DINHeaderData[NumDINFrames]:= DTDIN[1].GetSingleValue(0,1) shr 8;
    DINFrameData[NumDINFrames]:= DTDIN[1].GetSingleValue(0,1) shr 8;//retrieve AOB6 ports A & B (data word)
    inc(NumDINFrames);
  end;
end;
}

{-------------------------------------------------------------------------}
procedure TContAcqForm.ExpTimerTick(Sender: TObject);// var lStatus: Integer);
var p, t, PrbIndex, NumWaveSamples : integer;
  DisplayPtr : LPUSHRT;
begin
  ExpTime:= GetPrecisionTime div (MASTER_CLOCK_FREQUENCY div EXP_TIMER_FREQ);//inc(ExpTime);
  t:= ExpTime mod EXP_TIMER_FREQ;

  {This section includes anything for display or processing at EXP_TIMER_FREQ (typ. 10Hz) }
  {update probe/EEG/CSD display(s)}
  PrbIndex:= (((ADC[NotBoardInQueue].BuffersCollected + (BUFFERS_PER_SECOND -1)) mod BUFFERS_PER_SECOND)
              * Length(DeMUXedDTBuffer) div BUFFERS_PER_SECOND); //point to first sample/channel of first probe
  for p:= 0 to Config.NumProbes - 1 do
  begin
    with Config.Setup.Probe[p], ProbeWin[p] do
    begin
      NumWaveSamples:= SampPerChanPerBuff div Skippts;
      if ProbeType = CONTINUOUS then
      begin
        DisplayPtr:= @DeMUXedDTBuffer[PrbIndex{ + NumWaveSamples - NPtsPerChan}];
        win.PlotWaveform(DisplayPtr, NumWaveSamples);
        (*if CSDWinCreated then //transfer to CSD buffer, update when full...
        with CSDWin do
        begin
          Move(DisplayPtr^, CSDRingBuffer[CSDRingBufferIndex], NumWaveSamples * 2{bytes});
          CSDRingBufferIndex:= (CSDRingBufferIndex + 300) mod 2900; //inc(CSDRingBufferIndex, NumWaveSamples);
          if CSDRingBufferIndex = 0 then PlotCSD(@CSDRingBuffer[0], 300);
        end{CSDWin};*)
      end else //SPIKESTREAM or SPIKEEPOCH probe...
      begin
        DisplayPtr:= @DeMUXedDTBuffer[PrbIndex + NumWaveSamples -1]; //point to last sample of probe's first chan
        if win.muContDisp.Checked then DispTrigOffset:= NumWaveSamples - NPtsPerChan {disp last epoch of buffer}
          else DispTrigOffset:= FindThresholdXing(DisplayPtr, p);
        if DispTrigOffset > -1 then //spike over threshold
        begin
          if (DispTrigOffset - TrigPt) < 1 then DispTrigOffset:= TrigPt + 1
            else if (DispTrigOffset + NPtsPerChan) > NumWaveSamples //ensure trigger in bounds...
              then DispTrigOffset:= NumWaveSamples - NPtsPerChan;   //...of current probe buffer
          DisplayPtr:= @DeMUXedDTBuffer[PrbIndex + DispTrigOffset - TrigPt -1{?}];
          win.PlotWaveform(DisplayPtr, SampPerChanPerBuff); //finally, plot the spike waveform
          if win.GUICreated and (win.muContDisp.Checked = False) then
          begin
            if (TrigChan + ChanStart) <> MUXChan.Position then //don't erase monitor channel
            with win.GUIForm do
            begin
              ChangeSiteColor(LastTrigChan);
              ChangeSiteColor(TrigChan, clYellow); //highlight trigger site on GUI
              Refresh;
              LastTrigChan:= TrigChan;
            end;
          end;
          DispTrigOffset:= -1;
        end;
      end;
      inc(PrbIndex, NumWaveSamples * NChannels);
    end{with};
  end{p};

  {old method, for displaying from raw DT buffers}
  {c:= Config.CGList[cg].Channel;
  nc:= Config.Setup.Probe[p].ChanEnd - c + 1;
  if ((c mod 32) + nc) > 32 then nc:= 32 - (c mod 32);
  b:= c div 32;
  currently, no check for win enabled, or probe type(ie. epoch probes also display continuously)
  with ADC[b] do
  begin
    DisplayPtr:= hbufAddress[hbufPtrIndex];
    inc(DisplayPtr, cg mod 32); {channel offset -- is this correct for all cases?
    ProbeWin[p].win.PlotDTBufferWaveforms(DisplayPtr, c-Config.Setup.Probe[p].ChanStart, nc, ChansPerBuff);
  end;
  inc(cg, Config.Setup.Probe[p].NChannels);
  end; }

  {update experiment timer display}
  with LEDs, TimePanel.Font do
    if ErrorTimeCount > 1 then
      Tag:= Tag xor $00000001 //...LEDs flash yellow
    else if Recording then
    begin //red for recording...
      Color:= clRed;
      Tag:= RED_LED + t;
    end else
    begin //green for acquiring...
      Color:= clLime;
      Tag:= t;
    end;
  StatusBar.Invalidate; {update statusbar LEDs}

  TimePanel.Caption:= IntToStr(ExpTime div TIMER_TICKS_PER_HOUR) //hours...
    + FormatFloat(':00:', ((ExpTime mod TIMER_TICKS_PER_HOUR) div TIMER_TICKS_PER_MINUTE)) //minutes
    + FormatFloat('00.0', ((ExpTime mod TIMER_TICKS_PER_MINUTE) / EXP_TIMER_FREQ)); //secs.10th/sec

  if InfoWin.Visible then RefreshBufferInfo;

  {end 10Hz section}

  {This section includes anything for display or processing at 1Hz}
  if t = 0 then
  begin
    if StimulusDisplayRunning and GotValidDSHeader and not StimulusDisplayPaused then
    begin
      RefreshStimulusInfo;
      Dec(StimulusTimeRemaining);
    end;

    if Recording then
    begin
      inc(RecTime);//RecTime:= TotalRecTime + ExpTime div EXP_TIMER_FREQ;
      CheckForMsgRecordsToSave;
      CheckForSVRecordsToSave;
    end;

    if InfoWin.Visible then RefreshFileInfo;
    if ErrorTimeCount > 1 then dec(ErrorTimeCount) else
      ADCSaturated:= False;

    if MUXTimeCount > 1 then dec(MUXTimeCount) else
    begin
      if MUXChan.Enabled then
        for p:= 0 to Config.NumProbes - 1 do //add random seed to allow multiple probes AutoMUX?
          with Config.Setup.Probe[p], ProbeWin[p] do
          begin
            if (ProbeType <> CONTINUOUS) and win.muAutoMUX.Checked then
            begin
              if (TrigChan + ChanStart) <> MUXChan.Position then
              begin
                MUXChan.Position:= TrigChan + ChanStart;
                MUXChanClick(Self, btNext);
              end;
              Break;
            end;
          end;
      MUXTimeCount:= AUTOMON_INTERVAL;
    end{AutoMUX};

    {This section includes anything for display or processing at 0.5Hz}
    if (ExpTime mod (EXP_TIMER_FREQ*5)) = 0 then
    begin
      if StimulusDisplayRunning and not StimulusDisplayPaused and GotValidDSHeader then
        CheckDisplayPaused;
    end{0.5Hz};

  end{1Hz};
end;

{-------------------------------------------------------------------------}
function TContAcqForm.FindThresholdXing(hbufAddress : LPUSHRT; const ProbeID : integer) : integer;
var hbufAddressCopy : LPUSHRT;
  s, c, ChanSeed, NumWaveSamples : integer;
  GlobalSearch : boolean;
begin
  hbufAddressCopy:= hbufAddress;
  with ProbeWin[ProbeID], Config.Setup.Probe[ProbeID] do
  begin
    if win.GUICreated and (win.GUIForm.m_iNSitesSelected <> 0) then GlobalSearch:= False
      else GlobalSearch:= True; //search all or subset of channels per polytrodeGUI selection
    NumWaveSamples:= SampPerChanPerBuff div Skippts;
    ChanSeed:= Random(NChannels); //start search at random channel (0 to NChan-1)...
    c:= ChanSeed; //...to avoid any single active channel from dominating display
    repeat
      if GlobalSearch or (win.GUIForm.SiteSelected[c]) then
      begin
        hbufaddress:= hbufAddressCopy; //points to last sample of channel 0
        inc(hbufAddress, c * NumWaveSamples); //points to last sample of channel 'c'
        for s:= 0 to NumWaveSamples - 1 do
        begin
          if DispTrigBipolar then //check for spike of either polarity...
          begin                   //room for a little optimisation?(takes ~1.3ms for 54chan probe)
            if (hbufAddress^ > Threshold) or (hbufAddress^ < (RESOLUTION_12_BIT - Threshold)) then
            begin
              Result:= NumWaveSamples - s;
              TrigChan:= c {+ ChanStart};
              Exit;
            end;
          end else
          if DispTrigPositive then //positive trigger...
          begin
            if hbufAddress^ > Threshold then
            begin
              Result:= NumWaveSamples - s;
              TrigChan:= c {+ ChanStart};
              Exit;
            end;
          end else //negative trigger...
            if hbufAddress^ < Threshold then
            begin
              Result:= NumWaveSamples - s;
              TrigChan:= c {+ ChanStart};
              Exit;
            end;
          dec(hbufAddress); //searches back in time through buffer
        end{s};
      end{siteselected};
      c:= (c + 1) mod NChannels;
    until c = ChanSeed; //back at start, so quit
  end;{with}
  Result:= -1;//no threshold crossing found
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.RefreshFileInfo;
begin
  if DataFileOpen then
    with InfoWin do
    begin
      case lbRecorded.Tag of
        0 : lbRecorded.Caption:= FormatFloat('#,', SurfFile.Get64BitFileSize) + ' bytes';
        1 : lbRecorded.Caption:= FormatFloat('#,;;0', SurfFile.Get64BitFileSize shr 20) + ' Mbytes';
        2 : lbRecorded.Caption:= FormatFloat('#,;;0', SurfFile.Get64BitFileSize shr 30) + ' Gbytes'
      else lbRecorded.Caption:= IntToStr(RecTime div 3600)  //hours...
           + FormatFloat(':00:', (RecTime mod 3600) div 60) //minutes
           + FormatFloat('00', RecTime mod 60);             //seconds
      end{case};
    end{infowin};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.RefreshBufferInfo;
//var b : integer;
begin  //currently assumes one or two boards installed (no more, no less)...
  //for b:= 0 to NumADBoards - 1 do
  with InfoWin{, ADC[b]} do
  begin
    ProgressBar1.Position:= ADC[0].DTAcq.QueueSize[OL_QUE_READY];
    ProgressBar2.Position:= ADC[0].DTAcq.QueueSize[OL_QUE_INPROCESS];
    ProgressBar3.Position:= ADC[0].DTAcq.QueueSize[OL_QUE_DONE];
    if ADC[1].Configured then
    begin
      ProgressBar4.Position:= ADC[1].DTAcq.QueueSize[OL_QUE_READY];
      ProgressBar5.Position:= ADC[1].DTAcq.QueueSize[OL_QUE_INPROCESS];
      ProgressBar6.Position:= ADC[1].DTAcq.QueueSize[OL_QUE_DONE];
    end;
    case lbAcquired.Tag of
      0 : begin
            Label2.Caption:= FormatFloat('#,', ADC[0].SamplesCollected);
            if ADC[1].Configured then
              lbAcquired.Caption:= FormatFloat('#,', ADC[1].SamplesCollected);
           end;
      1 : begin
            Label2.Caption:= FormatFloat('#,', ADC[0].BuffersCollected);
            if ADC[1].Configured then
              lbAcquired.Caption:= FormatFloat('#,', ADC[1].BuffersCollected);
          end else
     {2}  begin
            Label2.Caption:= FormatFloat('#,;;0', ADC[0].SamplesCollected div 1000000);
            if ADC[1].Configured then
              lbAcquired.Caption:= FormatFloat('#,;;0', ADC[1].SamplesCollected div 1000000);
          end;
    end{case};
  end{infowin};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.RefreshStimulusInfo;
var StimulusTimeElapsed : integer;
begin
  with InfoWin do
  begin
    Label3.Caption:= FormatFloat('#,;;0', NumDINFrames);
    GaugeDS.Progress:= GaugeDS.MaxValue - StimulusSweepsRemaining;
    Label7.Caption:= FormatFloat('#,;;0', GaugeDS.Progress);
    case lbStimTime.Tag of
      0 : begin //stimulus time remaining...
            Label11.Caption:= IntToStr(StimulusTimeRemaining div 3600)     //hours...
            + FormatFloat(':00:', (StimulusTimeRemaining mod 3600) div 60) //minutes
            + FormatFloat('00', StimulusTimeRemaining mod 60);             //seconds
          end;
     -1 : begin //stimulus time elapsed..
            StimulusTimeElapsed:= StimulusHeader.est_runtime - StimulusTimeRemaining;
            Label11.Caption:= IntToStr(StimulusTimeElapsed div 3600)     //hours...
            + FormatFloat(':00:', (StimulusTimeElapsed mod 3600) div 60) //minutes
            + FormatFloat('00', StimulusTimeElapsed mod 60);             //seconds
          end;
    end{case};
  end{infowin};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CheckDisplayPaused;
begin
  if NumDINFrames = LastTotalFrames then
  begin //run-bit still high, but user paused DS...
    StimulusDisplayPaused:= True;
    if PauseRecordMode and Recording then
    begin
      Recording:= False;
      RecordingPaused:= True;
      sb_record.Enabled:= False;
      PostSurfMessage('Stimulus paused. Recording paused.');
    end{pauserec}else
      PostSurfMessage('Stimulus paused.');
    StimulusTimeRemaining:= Round(StimulusSweepsRemaining / InfoWin.GaugeDS.MaxValue
                          * StimulusHeader.est_runtime); //corrects for 10 pause delay
    RefreshStimulusInfo;
  end else
    LastTotalFrames:= NumDINFrames;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.ResetStimulusInfo;
begin
  NumDINHeader:= 0; //reset stimulus variables
  NumDINFrames:= 0;

  TotalNumFrames:= 0;
  LastTotalFrames:= -1;
  StimulusTimeRemaining:= 0;
  StimulusSweepsRemaining:= 0;
  GotValidDSHeader:= False;
  SweepChecksum:= 0;
  with InfoWin do  //reset infowin
  begin
    lbStimTime.OnClick:= lbStimTimeClick; //restore time-mode click handle
    lbStimTime.Tag:= not(lbStimTime.Tag);
    lbStimTime.OnClick(Self); //display current user-selected time mode
    Label11.Caption:= '';
    Label8.Font.Color:= clBlue;
    Label8.Caption:= 'Filename:';
    Label5.Caption:= '';
    Label3.Caption:= '';
    Label7.Caption:= '';
    GaugeDS.Progress:= 0;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.NewConfigClick(Sender: TObject);
begin
  GotCfgFileName:= False;
  CfgFileName:= '';
  Config.NumProbes:= 0;
  Config.NumAnalogChannels:= 0;
  Config.AcqDIN:= False;
  MsgPanel.Height:= MsgMemo.Font.Height * 4 + 1;
  Config.Empty:= True;
  SetupProbes;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.ModifyConfigClick(Sender: TObject);
begin
  SetupProbes;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.sb_playClick(Sender: TObject);
begin
  StartStopAcquisition;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.sb_recordClick(Sender: TObject);
begin
  StartStopRecording;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.sb_recordMouseUp(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  if (ssShift in Shift) and (sb_Record.down = False)
    and Config.AcqDIN then PauseRecordMode:= True;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.sb_stopClick(Sender: TObject);
begin
  sb_stop.down:= False; //grouped, but stays up, what the $%^*?
  if PauseRecordMode then Recording:= True; //briefly, so StartStopRec finalization called...
  if Recording then StartStopRecording else
    if Acquiring then
    begin
      StartStopAcquisition;
      sb_play.Enabled:= True;
      sb_record.Enabled:= True;
    end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.StartStopAcquisition;
var b : integer;
begin
  if Acquiring then //stop acquisition
  begin
    Acquiring := False;
    Screen.Cursor:= crHourGlass;
    try
      {stop clocks/counters}
      if ExpTimerEnabled then WinExpTimer.Enabled:= False;//DTExpTimer.Stop;
      if DT32BitCounterEnabled and MasterClockEnabled then
      begin
        DT32BitCounter.Stop;
        DTMasterClock.Stop;
      end;
      {clean-up buffers}
      for b:= 0 to NumADBoards - 1 do
        with ADC[b] do
        begin
          if ADC[b].Configured then
          begin
            if not DTUnderrrunError then DTAcq.Abort;
            DTAcq.Flush; //flush ready/inprocess buffers to done queue
            hbuf:= DTAcq.Queue;
            while hbuf <> 0 do
            begin
              {save aborted/partial buffers here? -- oldmGetValidSamples)}
              DTAcq.Queue:= hbuf; //return buffer to Ready Queue
              hbuf:= DTAcq.Queue; //get next buffer from Done Queue
              {if hbufptrIndex = 0 then hbufptrIndex:= DEFAULT_NUM_BUFFS - 1
                else dec(hbufptrIndex); //and reset ptr index}
            end;
          end;
        end;
      RefreshBufferInfo;
      if StimulusDisplayRunning then
        with InfoWin.lbStimTime do
        begin
          Caption:= 'SURF halted                ';// before stimulus finished    ';
          StimulusDisplayRunning:= False;
          OnClick:= nil;
        end;
    except
      PostSurfMessage('Exception raised stopping subsystems!'); //for the moment...
    end{try};
    {enable user menus}
    for b:= 0 to SURFMainMenu.Items.Count-1 do SURFMainMenu.Items[b].Enabled := True;
    tb_file.Enabled:= True;
    Screen.Cursor:= crDefault;
    LEDs.Tag:= 0; //dim status LED
    StatusBar.Invalidate;
  end{stop}else //start acquisition...
  begin
    if Config.NumAnalogChannels = 0 then
    begin
      ShowMessage('No channels setup');
      sb_play.down:= False;
      Exit;
    end;

    {initialise acquisition variables}
    ExpTime:= 0;
    Old32BitTime:= 0;
    MUXTimeCount:= 0;
    ErrorTimeCount:= 0;
    Time32BitOverflow:= 0;
    if Config.AcqDIN then ResetStimulusInfo;
    for b:= 0 to Config.NumProbes - 1 do
      ProbeWin[b].DispTrigOffset:= -1; //reset display triggers
    {ringbuffer indexes are safer if reset to zero...
     eg. if an exception puts the index pairs out of sync}
    SSRingBufferIndex := 0;
    SSRingSaveIndex   := 0;
    SVRingBufferIndex := 0;
    SVRingSaveIndex   := 0;

    {pre-start A/D subsystems}
    DTUnderrrunError:= False;
    for b:= NumADBoards - 1 downto 0 do
    with ADC[b] do
    begin
      SamplesCollected:= 0;
      BuffersCollected:= 0;
      if Configured then
        begin
          DTAcq.Config; //re-configure ensures buffer indexing/alignment valid
          DTAcq.Start;
        end;
    end{ADC[b]};

    {(re-)configure then start clocks/counters}
    //DTExpTimer.Config; //using TTimer now
    if DT32BitCounterEnabled and MasterClockEnabled then
    begin
      DT32BitCounter.Config; //re-configure resets precision timer to zero
      DTMasterClock.Config;
      DT32BitCounter.Start; //start Master clock LAST to ensure perfect synchronisation...
      DTMasterClock.Start;  //of precision timestamp clock with A/D sample clock(s)
    end;
    if ExpTimerEnabled then WinExpTimer.Enabled:= True; //DTExpTimer.Start;
    {disable user menus}
    for b:= 0 to SURFMainMenu.Items.Count - 1 do SURFMainMenu.Items[b].Enabled := False;
    tb_file.Enabled:= False;
    Acquiring:= True;
  end{Start acquisition};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.StartStopRecording;
begin
  if Recording then //stop recording...
  begin
    TotalRecTime:= RecTime;
    StartStopAcquisition;
    PostSurfMessage('Recording stopped.'{, BufferStartTime + CLOCK_TICKS_PER_BUFFER - 1});
    TimePanel.Font.Color:= clLime;
    (*for i:= 0 to Screen.FormCount - 1 do {mark all open EEGWin}
      if Screen.Forms[i] is TEEGWin then
        EEGWin[i].DrawLabelMarker(clWhite, 'rec', ' stop ');*)
    {write any remaining buffered data to file}
    CheckForPTRecordsToSave;
    CheckForSVRecordsToSave;
    CheckForMsgRecordsToSave;
    Recording:= False;
    PauseRecordMode:= False;
    StopPauseRecordMode:= False;
    sb_play.Enabled:= True;
    sb_record.Enabled:= True;
    RefreshFileInfo;
    RefreshStimulusInfo;
  end{stop}else
  begin             //start recording...
    if DataFileOpen then
    begin
      StartStopAcquisition;
      if not Acquiring then StartStopAcquisition; {restart, with clocks reset, if direct play-->rec}
      if Acquiring then
      begin
        if not PauseRecordMode then
        begin
          Recording:= True;
          PostSurfMessage('Recording started. Data file: '''
                         + ExtractFileName(SurfFile.SurfFileName) + '''');
          (*for i:= 0 to Screen.FormCount - 1 do {mark all open EEGWin}
            if Screen.Forms[i] is TEEGWin then
              EEGWin[i].DrawLabelMarker(clWhite, 'rec', 'start');*)
        end else
        begin
          PostSurfMessage('Waiting for the stimulus to begin...');
          sb_record.Enabled:= False;
        end;
        sb_play.Enabled:= False;
      end;
    end else
    begin
      sb_record.down:= False;
      PauseRecordMode:= False;
      if Acquiring then sb_play.down:= True;
      ShowMessage('No data file open');
    end;
  end{start};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.WriteSurfLayoutRecords;
var p, c, prbIndex, SHOffset, BoardIndex : integer;
    slr : SURF_LAYOUT_REC;
begin
  for p := 0 to Config.NumProbes-1 do
    PutWindowLocationsInConfig(p);

  with slr do //common acquisition-wide settings
  begin
    UffType  := SURF_PL_REC_UFFTYPE; //record type - chr(234)
    TimeStamp:= GetPrecisionTime;
    SurfMajor:= 2; //SURF major version number
    SurfMinor:= 1; //SURF minor version number // ver 2.1 added python header to TStimulusHeader
    MasterClockFreq:= MASTER_CLOCK_FREQUENCY; //master ADC/CT clock rate (typ. 1Mhz)
    BaseSampleFreq := Config.BaseSampFreqPerChan; //undecimated A/D sample rate
    if Config.AcqDIN then DINAcquired:= True else DINAcquired:= False;
    FillChar(pad, SizeOf(pad), 0);
  end;

  prbIndex:= 0;
  BoardIndex:= -1;
  SHOffset:= 0;
  for p := 0 to Config.NumProbes -1 do
  begin
    {calculate S&H delay offset for this probe's first channel}
    if BoardIndex <> (Config.Setup.Probe[p].ChanStart div DT3010_MAX_SE_CHANS) then
    begin
      BoardIndex:= Config.Setup.Probe[p].ChanStart div DT3010_MAX_SE_CHANS;
      if Config.acqDIN and (BoardIndex < 2) then SHOffset:= 1 else
      SHOffset:= 0;
    end else
      inc(SHOffset, Config.Setup.Probe[p-1].NChannels);

    if not Config.Setup.Probe[p].Save then Continue; //only write slr for save-checked probes
                           //BUG ALERT -- if no save-checked, NO FILE SLR?!
    with slr do //probe-specific settings
    begin
      Probe          := prbIndex;                                //probe number
      ProbeSubType   := Config.Setup.probe[p].ProbeType;         //= S,E or C for spike, epoch or continuous
      nchans         := Config.Setup.probe[p].NChannels;         //number of channels in this spike waveform
      pts_per_chan   := Config.Setup.probe[p].NPtsPerChan;       //# samples per waveform per chan (display)
      skippts        := Config.Setup.probe[p].SkipPts;           //decimation factor (1 = none)
      pts_per_buffer := nchans * SampPerChanPerBuff div skippts; //# of samples per file buffer (all chans, this probe)
      trigpt         := Config.Setup.probe[p].TrigPt;            //pts before trigger
      lockout        := Config.Setup.probe[p].Lockout;           //trig lockout in pts
      threshold      := Config.Setup.probe[p].Threshold;         //A/D board threshold for trigger
      intgain        := Config.Setup.probe[p].InternalGain;      //A/D board internal gain
      sampfreqperchan:= Config.Setup.probe[p].SampFreq;          //A/D sampling frequency (incl. any decimation)
      sh_delay_offset:= SHOffset;
      probe_descrip  := Config.Setup.probe[p].Descrip;           //description of the electrode
      electrode_name := Config.Setup.probe[p].ElectrodeName;     //predefined electrode name (from ElectrodeTypes)

      {generate cglist for this probe}
      for c:= 0 to SURF_MAX_CHANNELS - 1 do
        //if Config.CGList[c].ProbeId = p then
        if c < Config.Setup.Probe[p].NChannels then
        begin
          //Chanlist[c]:= Config.CGList[c].Channel;
          Chanlist[c]:= Config.Setup.Probe[p].ChanStart + c;
          ExtGain[c]:= EXTERNAL_AMPLIFIER_GAIN; //MCS hardware has a single, fixed external gain...
        end else                                //... array type kept for flexibility back compatibility
        begin
          Chanlist[c]:= -1{unused, this probe};
          ExtGain[c] :=  0{unused, this probe};
        end;

      {ExtGain form not currently used}
      {ExtGainForm.Probe := i;
      ExtGainForm.NumChannels := Config.NumAnalogChannels;
      ExtGainForm.ShowModal;
      For c := 0 to SURF_MAX_CHANNELS-1 do
        if c < Config.NumAnalogChannels
          then extgain[c] := Word(StrToInt(ExtGainForm.ExtGainArray[i].Text))
          else extgain[c] := 0;//unused}

      ProbewinLayout.left  := Config.WindowLoc[p].left;
      ProbewinLayout.top   := Config.WindowLoc[p].top;
      ProbewinLayout.width := Config.WindowLoc[p].width;
      ProbewinLayout.height:= Config.WindowLoc[p].height;
    end{with slr};

    if not SurfFile.PutSurfRecord(slr) then
    begin
      ShowMessage('Error writing to Surf file. Data file closed.');
      CloseDataFile;
    end;
    inc(prbIndex);
  end{p};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.SetupProbes;
var i, j, min, left, top, CGLIndex : integer;
    ProbeRowTemp : TIndivProbeSetup;
    PtrodeGUITmp : array [0..SURF_MAX_PROBES - 1] of TWindowLoc;
    ProbeZoomTmp : array [0..SURF_MAX_PROBES - 1] of integer;
    MinorModify : boolean;
begin
  MinorModify:= False;
  for i:= 0 to SURF_MAX_PROBES - 1 do ProbeZoomTmp[i]:= DEFAULTYZOOM;
  for i:= 0 to Config.NumProbes - 1 do //save current window locations
  begin
    if ProbeWin[i].win.WindowState = wsNormal then
    begin
      Config.Setup.Probe[i].View:= True; //update config if user restored probewin
      PutWindowLocationsInConfig(i);
    end;
    ProbeZoomTmp[i]:= ProbeWin[i].win.CBZoom.ItemIndex; //default zoom, 100%
    Dec(Config.Setup.Probe[i].Threshold, 2048); //revert to signed thresholds for setup
    if ProbeWin[i].win.GUICreated then
    with ProbeWin[i].win.GUIForm do //save polytrode GUI window locations
    begin
      PtrodeGUITmp[i].Left:= Left;
      PtrodeGUITmp[i].Top:=  Top;
    end;
  end;
  try
    if ProbeSetupWinCreated then ProbeSetupWin.Release; //remove any existing ProbeSetupWin forms
    try
      ProbeSetupWin:= CProbeSetupWin.Create(Self); //...and create a new one
      ProbeSetupWin.BringToFront;
      ProbeSetupWinCreated:= True;
    except
      Showmessage('Unable to create probe configuration window');
      ProbeSetupWinCreated:= False;
      Exit;
    end;
    ProbeSetupWin.MaxADChannels:= MaxADChannels; //pass available A/D channels and...
    ProbeSetupWin.SampFreqPerChan.Value:= SampFreqPerChan; //...current sample rate to ProbeSetupWin
    with ProbeSetupWin do //display current hardware resources, constrain user options accordingly
    begin
      SampFreqPerChan.MaxValue:= MaxSampFreqPerChan;
      SampFreqPerChan.MinValue:= MinSampFreqPerChan;
      DINCheckBox.Checked:= Config.AcqDIN;
      lb_NumADBoards.Caption:= Inttostr(NumADBoards);
      lb_ADChans.Caption:= Inttostr(MaxADChannels);
      lb_ADCTotFreq.Caption:= Floattostr(ADCFrequency * NumADBoards / MASTER_CLOCK_FREQUENCY);
      if StimulusDINEnabled then lb_DINSS.Caption:= 'DT3010A/B'
        else DINCheckBox.Enabled:= False; //disable DIN acquire option if no DIN subsystem available
      if ExpTimerEnabled then lb_TimerSS.Caption:= 'windows';//DTExpTimer.Board;
      if MUXDOUTEnabled then lb_MUXSS.Caption:= DTMUX.Board;
    end;

    InfoWin.Hide;
    FreeChanWindows;
    if AvgWinCreated then AvgWin.Hide;

    if not Config.Empty then  //copy current setup for 'Modify' (if one exists)
      Move(Config.Setup, ProbeSetupWin.Setup, SizeOf(TProbeSetup));

    ProbeSetupWin.ShowModal; //switch user control to ProbeSetupWin

    {if valid update before finalising subsystem configuration}
    if ProbeSetupWin.OK then
    begin {copy the new/modified ProbeWin Setup to the Config.Setup}
      if (ProbeSetupWin.Setup.TotalChannels = Config.Setup.TotalChannels) and
         (ProbeSetupWin.Setup.NProbes = Config.Setup.NProbes) then MinorModify:= True;
      Move(ProbeSetupWin.Setup, Config.Setup, SizeOf(TProbeSetup));
      SampFreqPerChan:= ProbeSetupWin.SampFreqPerChan.Value;
      Config.NumProbes:= ProbeSetupWin.Setup.NProbes;
      Config.NumAnalogChannels:= ProbeSetupWin.Setup.TotalChannels;
      Config.BaseSampFreqPerChan:= ProbeSetupWin.SampFreqPerChan.Value;
      Config.AcqDIN := ProbeSetupWin.DINCheckBox.Checked;
      if (ProbeSetupWin.Setup.NProbes = 0) or (ProbeSetupWin.Setup.TotalChannels = 0)
        then Config.Empty:= True
        else Config.Empty:= False;
    end{ProbeSetupOK};

    {retrieve (and sort) the CGL from the ProbeWin setup...
     other procedures expect successive probes to have channels
     in ascending order, so probes are sorted accordingly here}
    for i:= 0 to Config.NumProbes - 1 do
    begin
      min:= i;
      for j:= (i+1) to Config.NumProbes - 1 do
        if Config.Setup.Probe[j].ChanStart < Config.Setup.Probe[min].ChanStart then
          min:= j;
      ProbeRowTemp:= Config.Setup.Probe[i];
      Config.Setup.Probe[i]:= Config.Setup.Probe[min];
      Config.Setup.Probe[min]:= ProbeRowTemp;
    end{i};

    {(re)program the analog CGL}
    for i:= 0 to SURF_MAX_CHANNELS - 1 do
      Config.CGList[i].ProbeId:= -1; //initialise CGList (since zero is a valid ProbeID)
    CGLIndex := 0;
    for i:= 0 to Config.NumProbes - 1 do
      for j:= 0 to Config.Setup.Probe[i].NChannels - 1 do
      begin
        Config.CGList[CGLIndex].ProbeId:= i;
        Config.CGList[CGLIndex].Channel:= Config.Setup.Probe[i].ChanStart + j;
        Config.CGList[CGLIndex].Decimation:= Config.Setup.Probe[i].SkipPts;
        inc(CGLIndex);
      end;

    {try to finalise configuration of A/D subsystem(s)}
    if not ConfigBoard then
    begin
      ShowMessage('Error: Unable to configure DT-board(s) properly');
      Exit;
    end;

    {restore main form, then restore probe/infowin windows}
    left := 3;
    top := tb_Main.Height + 3;
    for i := 0 to Config.NumProbes - 1 do
    begin
      if not CreateAProbeWindow(i, left, top, Config.Setup.Probe[i].NPtsPerChan) then
      begin
        ShowMessage('Error: plot windows not configured properly');
        Exit;
      end;

      if not (Config.Setup.Probe[i].ProbeType = CONTINUOUS) then
        ProbeWin[i].win.seThreshold.OnChange(Self); //restore unsigned threshold
      ProbeWin[i].win.CBZoom.ItemIndex:= ProbeZoomTmp[i]; //restore zoom setting
      ProbeWin[i].win.CBZoomChange(Self);

      {either save or restore probe window loc'ns depending on whether 'major' changes to config}
      if not MinorModify and ProbeSetupWin.OK then
      begin
        if (left + ProbeWin[i].win.Width) > ClientWidth then
        begin //simulate MDI-cascade of windows if they don't fit on form
          inc(top, 12);
          left:= top;
          ProbeWin[i].win.left:= left; //move new window
          ProbeWin[i].win.top:= top;
        end;
        PutWindowLocationsInConfig(i); //save window location
        inc(left, ProbeWin[i].win.Width + 2);
        if ClientWidth < left  //resize ContAcqForm height and width to accommodate...
          then ClientWidth:= left; //...probe windows (effect only visible if form is normalised)}
        if ClientHeight < WaveFormPanel.Top + tb_Main.Height + MsgPanel.Height + ProbeWin[i].win.Height
          then ClientHeight:= WaveFormPanel.Top + tb_Main.Height + MsgPanel.Height + ProbeWin[i].win.Height + 30;
      end{ok} else
      begin
        GetWindowLocationsFromConfig(i); //restore previous probewin window locns.
//        if PtrodeGUITmp[i].Left <> 0 then //also restore ptrode GUI window locns.
//        with ProbeWin[i].win do
//        begin                             // DISABLED BECAUSE
//          muPolytrodeGUI.Checked:= True;  // WITHOUT SHOWMESSAGE DELAY, GET RUNTIME BUG
//          CreatePolytrodeGUI(i);
//          if GUICreated then
//          begin
//            GUIForm.Left:= PtrodeGUITmp[i].Left;
//            GUIForm.Top:= PtrodeGUITmp[i].Top;
//          end;
//        end;
      end;
      if Config.Setup.Probe[i].View = False then
        ProbeWin[i].win.WindowState:= wsMinimized;
    end{i};

    {update status bar, menu items}
    with StatusBar do
    begin
      if Config.Empty then
      begin
        muCfgModify.Visible:= False;
        muData.Enabled:= False;
        tbNewDataFile.Enabled:= False;
        Panels[1].Text:= '';
        Panels[2].Text:= '';
      end else
      begin
        muCfgModify.Visible:= True;
        muData.Enabled:= True;
        muNewFile.Visible:= True;
        tbNewDataFile.Enabled:= True;
        if GotCfgFileName then Panels[1].Text:= ExtractFileName(CfgFileName)
          else Panels[1].Text:= 'Configuration not saved';
        Panels[2].Text:= Inttostr(Config.NumAnalogChannels) + ' channels, '
                       + Inttostr(SampFreqPerChan) + 'Hz/channel (base), '
                       + FloattostrF(DTBandWidth / MASTER_CLOCK_FREQUENCY, ffGeneral, 3, 1) + 'MSamp/s';
        Panels[1].Width:= Round(Length(Panels[1].Text) * Font.Size * 2/3) + 10;
        Panels[2].Width:= Round(Length(Panels[2].Text) * Font.Size * 2/3) + 10;
      end;
    end{StatusBar};
    if tb_expinfo.Down then InfoWin.Show;
    AddRemoveCSDWin; //update (or remove) CSDWin depending on changes to setup
  finally
    ProbeSetupWin.Release;
    ProbeSetupWinCreated:= False;
  end{try};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.SaveConfigClick(Sender: TObject);
begin
if GotCfgFileName = False then
  begin
    if SaveConfigDialog.Execute then
      SaveSurfConfig(SaveConfigDialog.FileName)
    else ShowMessage('Config not saved');
  end else SaveSurfConfig(CfgFileName);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.SaveConfigAsClick(Sender: TObject);
begin
  if SaveConfigDialog.Execute then
    SaveSurfConfig(SaveConfigDialog.FileName)
  else ShowMessage('Config not saved');
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.OpenConfigClick(Sender: TObject);
begin
  with OpenConfigDialog do
    if Execute then OpenSurfConfig(Filename);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.PutWindowLocationsInConfig(probe : integer);
begin
  Config.WindowLoc[probe].left  := ProbeWin[probe].win.left;
  Config.WindowLoc[probe].top   := ProbeWin[probe].win.top;
  Config.WindowLoc[probe].width := ProbeWin[probe].win.width;
  Config.WindowLoc[probe].height:= ProbeWin[probe].win.height;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.GetWindowLocationsFromConfig(probe : integer);
begin
  ProbeWin[probe].win.left  := config.WindowLoc[probe].left;
  ProbeWin[probe].win.top   := config.WindowLoc[probe].top;
  ProbeWin[probe].win.width := config.WindowLoc[probe].width;
  ProbeWin[probe].win.height:= config.WindowLoc[probe].height;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.SaveSurfConfig(Filename : string);
var fs : TFileStream;
    i : integer;
begin
   GotCfgFileName := False;
   CfgFileName:= Filename;
   Config.MainWinLeft  := Left;
   Config.MainWinTop   := Top;
   Config.MainWinHeight:= Height;
   Config.MainWinWidth := Width;
   Config.InfoWinLeft  := InfoWin.Left;
   Config.InfoWinTop   := InfoWin.Top;
   Config.InfoWinHeight:= InfoWin.Height;
   Config.InfoWinWidth := InfoWin.Width;
   Config.MsgMemoHeight:= MsgPanel.Height;
   fs:= nil;
   try
     fs:= TFileStream.Create(CfgFileName, fmCreate);
     fs.WriteBuffer('SCFG', 4);
     fs.WriteBuffer('v1.3.1', 6); //write version number
     for i:= 0 to Config.NumProbes - 1 do
     begin
       if Config.Setup.Probe[i].ProbeType <> CONTINUOUS then
         dec(Config.Setup.Probe[i].Threshold, 2048); //save signed thresholds

       if ProbeWin[i].win.WindowState = wsNormal then
       begin
         Config.Setup.Probe[i].View:= True; //update config if user restored probewin
         PutWindowLocationsInConfig(i); //save probewin location if not minimized
       end;
     end;
     fs.WriteBuffer(Config, SizeOf(Config)); //write config to file
     for i:= 0 to Config.NumProbes - 1 do
       if Config.Setup.Probe[i].ProbeType <> CONTINUOUS then
         inc(Config.Setup.Probe[i].Threshold, 2048); //restores unsigned thresholds
     GotCfgFileName := True;
   except
     ShowMessage('Exception raised when saving configuration.');
     fs.Free;
     Exit;
   end;
   fs.free;
   StatusBar.Panels[1].Text := ExtractFileName(CfgFileName);
   StatusBar.Panels[1].Width:= Round(Length(StatusBar.Panels[1].Text)
                                * StatusBar.Font.Size * 2/3) + 10;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.OpenSurfConfig(Filename : string);
var fs : TFileStream;
    ok : boolean;
    p  : integer;
    headerstr  : array[0..3] of char;
    versionstr : array[0..5] of char;
begin
  ok := True;
  GotCfgFileName:= False;
  fs:= nil;
  try
    fs:= TFileStream.Create(FileName, fmOpenRead);
    fs.ReadBuffer(headerstr, 4);
    if headerstr <> 'SCFG' then
    begin
      ShowMessage('This is not a SURF configuration file');
      ok := False;
      Exit;
    end;

    fs.ReadBuffer(versionstr, 6);
    if versionstr <> 'v1.3.1' then //must contain {DT3010DIN}, InfoWinVisible,
    begin                          //and BaseSampFreqPerChan fields
      ShowMessage('Unsupported version of configuration file.');
      ok := False;
      Exit;
    end;

    FreeChanWindows; //clear any existing probe windows
    fs.ReadBuffer(Config, Sizeof(Config)); //read entire config file
    SampFreqPerChan:= Config.BaseSampFreqPerChan;

    Left:= Config.MainWinLeft;
    Top:= Config.MainWinTop;
    Height:= Config.MainWinHeight;
    Width:= Config.MainWinWidth;
    InfoWin.Left:= Config.InfoWinLeft;
    InfoWin.Top:= Config.InfoWinTop;
    InfoWin.Height:= Config.InfoWinHeight;
    InfoWin.Width:= Config.InfoWinWidth;
    MsgPanel.Height:= Config.MsgMemoHeight;

    if not ConfigBoard then
    begin
      ShowMessage('Error: Hardware not configured properly for this config file.');
      ok := False;
      Exit;
    end;

    {create the probe and info windows and draw them to the screen}
    for p := 0 to Config.NumProbes - 1 do
    begin
      if not CreateAProbeWindow(p, Config.WindowLoc[p].left,
                                Config.WindowLoc[p].top,
                                Config.Setup.Probe[p].NPtsPerChan)
        then Showmessage('Error creating probe window from cfg file');
      if not (Config.Setup.Probe[p].ProbeType = CONTINUOUS) then
         ProbeWin[p].win.seThreshold.OnChange(Self); //restore unsigned threshold
      if Config.Setup.Probe[p].View = False then
        ProbeWin[p].win.WindowState:= wsMinimized
      else begin
        ProbeWin[p].win.Width := Config.WindowLoc[p].Width;
        ProbeWin[p].win.Height := Config.WindowLoc[p].Height;
        if ClientWidth < ProbeWin[p].win.Left + ProbeWin[p].win.Width then
          ClientWidth := ProbeWin[p].win.Left + ProbeWin[p].win.Width;
      end;
    end{p};
    GotCfgFileName:= True;
    CfgFileName:= Filename;
    InfoWin.Visible:= Config.InfoWinVisible;
    tb_expinfo.Down:= Config.InfoWinVisible;
  finally
    fs.Free;
    if not ok then
      ShowMessage('Configuration file not loaded');
  end;

  AddRemoveCSDWin; //update depending on changes in setup

  {update status bar/user controls}
  with StatusBar do
  begin
    Panels[1].Text:= ExtractFileName(CfgFileName);
    Panels[1].Width:= Round(Length(Panels[1].Text) * Font.Size * 2/3) + 10;
    Panels[2].Text:= Inttostr(Config.NumAnalogChannels) + ' channels, '
                   + Inttostr(SampFreqPerChan) + 'Hz/channel (base), '
                   + FloattostrF(DTBandWidth / MASTER_CLOCK_FREQUENCY, ffGeneral, 3, 1) + 'MSamp/s';
    Panels[2].Width:= Round(Length(Panels[2].Text) * Font.Size * 2/3) + 10;
    BringToFront;
  end;
  if Config.Empty then
  begin
    muCfgModify.Visible:= False;
    muData.Enabled:= False;
    tbNewDataFile.Enabled:= False;
  end else
  begin
    muCfgModify.Visible:= True;
    muData.Enabled:= True;
    muNewFile.Visible:= True;
    tbNewDataFile.Enabled:= True;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.FreeChanWindows;
var i : integer;
begin
  for i := 0 to SURF_MAX_PROBES{Config.NumProbes}- 1 do
  if ProbeWin[i].exists then
    begin
      if ProbeWin[i].win.GUICreated then
        ProbeWin[i].win.GUIForm.Release;
      ProbeWin[i].win.Release;
      ProbeWin[i].exists:= False;
    end;
end;

{-------------------------------------------------------------------------}
function TContAcqForm.CreateAProbeWindow(probenum, left, top, npts : integer) : boolean;
var Electrode : TElectrode; ProbeWinTitle : ShortString; DispDecFactor : integer;
begin
  Result := True;
  if not GetElectrode(Electrode, Config.Setup.Probe[probenum].ElectrodeName) then
  begin
    ShowMessage(Config.Setup.Probe[probenum].ElectrodeName + ' is an invalid electrode name');
    Result := False;
    Exit;
  end;
  ProbeWinTitle:= Config.Setup.Probe[probenum].Descrip + ' (ch '
                + inttostr(Config.Setup.Probe[probenum].ChanStart);
  if Config.Setup.Probe[probenum].NChannels = 1 then ProbeWinTitle:= ProbeWinTitle + ')'
    else ProbeWinTitle:= ProbeWinTitle +  '..' + inttostr(Config.Setup.Probe[probenum].ChanEnd) + ')';
  if not ProbeWin[probenum].exists then
    ProbeWin[probenum].win:= CProbeWin.CreateParented(WaveFormPanel.Handle);
  if Config.Setup.Probe[probenum].ProbeType = CONTINUOUS then
    DispDecFactor:= {npts div} EXP_TIMER_FREQ //?!
  else DispDecFactor:= 1;
  ProbeWin[probenum].win.InitPlotWin(Electrode,
                            {npts}npts,
                            {left}Left,
                            {top}Top,
                            {thresh}Config.Setup.Probe[probenum].Threshold,
                            {trigpt}Config.Setup.Probe[probenum].TrigPt,
                            {probeid}probenum,
                            {probetype}Config.Setup.Probe[probenum].ProbeType,
                            {title}ProbeWinTitle,
                            {acquisition mode}True,
                            {intgain}Config.Setup.Probe[probenum].InternalGain,
                            {extgain}EXTERNAL_AMPLIFIER_GAIN{assumes MCS w/fixed gains},
                            {sampfreq}Config.Setup.Probe[probenum].SampFreq,
                            {pts2plot}DispDecFactor);
  ProbeWin[probenum].exists := True;
  ProbeWin[probenum].win.Visible := True;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.DTAcqBufferDone(Sender: TObject);
var  ValidSamples : integer; //BuffersAligned : boolean;
begin
  if not Acquiring then Exit; {if hBuf = Null then Exit}
  with TDTAcq32(Sender), ADC[Tag] do
  begin
    BufferStartTime:= CLOCK_TICKS_PER_BUFFER * Int64(BuffersCollected);
    hbuf:= Queue; //retrieve buffer from done cue
    ErrorMsg(olDmGetValidSamples(hBuf, ULNG(ValidSamples))); //get number of valid samples from buffer
    if ValidSamples = BuffSize then //check for dropped samples
    begin
      CopyChannelsFromDTBuffer(hbufaddresses[hbufPtrIndex], Tag);
      if {not ADC[1].Configured}NumADBoards = 1 then NotBoardInQueue:= 0 else //assumes only two boards installed
        if Tag = 0{master board} then NotBoardInQueue:= 1
          else NotBoardInQueue:= 0;
      if (BuffersCollected + 1 = ADC[NotBoardInQueue].BuffersCollected)
      or (NumADBoards = 1){(ADC[1].Configured = False)}{assumes only two boards installed}then //ensure both board's buffers received
      begin
        if ContAcqForm.Config.AcqDIN then DecodeStimulusDIN(tag);
        if AvgFillBuffer then CopyChannelsToAverager(tag);
        if Recording then
        begin
          PutPTRecordsOnFileBuffer(Tag);
          if SSRingBufferIndex = SSRINGARRAYSIZE then //write to file...
          while SSRingBufferIndex > 0 do
            begin
              SurfFile.PutPolytrodeRecord(SSRingBuffer[SSRingSaveIndex]);
              SSRingSaveIndex:= (SSRingSaveIndex + 1) mod SSRINGARRAYSIZE;
              Dec(SSRingBufferIndex);
            end;
        end{recording};
        {ADD CRC32 and OTHER ERROR CHECKING HERE... }
      end{aligned buffers}else
      if (ADC[0].BuffersCollected <> ADC[1].BuffersCollected) and (NumADBoards > 1){ADC[1].Configured} then
      begin
        DTBoardsOutOfSync;
        Exit;
      end;
      Queue:= {ULNG}hbuf; //recycle current buffer
      inc(BuffersCollected);
      inc(SamplesCollected, ValidSamples);
      hbufPtrIndex:= (hbufPtrIndex + 1) mod DEFAULT_NUM_BUFFS;
      //if StopPauseRecordMode then StartStopRecording;
    end{validsamples} else
    begin
      PostSurfMessage('Warning! DT buffer samples dropped.');
      FlagCriticalError;
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CopyChannelsFromDTBuffer({const} hbufAddress : LPUSHRT;
                                                        const Board : integer);
{demultiplexes DT buffers comprising all channels across
 all boards, into one contiguous 'board-transparent' ring buffer}
{also checks for ADC saturation, and sends data to spectrogram window, if applicable}
{CPU load: eg. for 54chan + DINprobe with(out) channel subsampling, takes ~20ms/sec}
var c, s, deMUXIndex, NChans, NextSample, cgIndex, SkipPts : integer;
  hbufAddressCopy : LPUSHRT;
begin
  with ADC[board] do
  begin
    deMUXIndex:= BoardOffset + ((BuffersCollected mod BUFFERS_PER_SECOND)
                               * Length(DeMUXedDTBuffer) div BUFFERS_PER_SECOND);
    NChans:= ChansPerBuff;
    for c:= DINOffset to NChans - 1 do
    begin
      cgIndex:= c + board*(ADC[0].ChansPerBuff-DINOffset)-DINOffset;
      Skippts:= Config.CGList[cgIndex].Decimation; {get this out of 'c' loop!}
      (* ProbeID:= Config.CGList[cgIndex].ProbeID; {messy, and inefficent!}
      ProbeThreshold:= Config.Setup.Probe[ProbeID].Threshold; *)
      NextSample:= NChans * Skippts; //decimates channel, if probe.skippts > 1
      hbufAddressCopy:= hbufAddress;
      inc(hbufAddressCopy, c);
      for s:= 0 to SampPerChanPerBuff div Skippts - 1 do
      begin
        DeMUXedDTBuffer[deMUXIndex]:= hbufAddressCopy^; //copy sample to ring buffer
        {check every sample for ADC saturation}
        if (hbufAddressCopy^ = ADC_SATURATED_LOW) or (hbufAddressCopy^ = ADC_SATURATED_HIGH) then
          if ADCSaturated = False then
          begin
            ADCSaturated:= True;
            PostSurfMessage('A/D saturated; channel ' + inttostr(Config.CGList[cgIndex].Channel)
                          + '. Reduce signal amplitude or internal ADC gain.',
                            BufferStartTime + s * Skippts * SamplePeriod);
            FlagCriticalError;
          end{saturated};
        inc(hbufAddressCopy, NextSample);
        inc(deMUXIndex);
      end{s};
      {copy data to spectrogram window for this channel, if running}
      {if Config.CGList[cgIndex].Channel in EEGWinsOpen then     REPLACE WITH MOVE!
      EEGWin[0].UpdateEEG(Copy(DeMUXedDTBuffer, 100, 100), 100); //remove hard-coding!}
    end{c};
  end{ADC[board]};
  {slightly slower routine - 22ms/sec for 54chan + DIN, with NO decimation capability}
  {for s:= 0 to SampPerChanPerBuff - 1 do
  begin
    inc(hbufAddress, DINOffset);
    for c:= DINOffset to ADC[board].ChansPerBuff - 1 do
    begin
      DeMUXedDTBuffer[boardoffset+s+((c-DINoffset)*SampPerChanPerBuff)]:= hbufAddress^;
      inc(hbufAddress);
    end;
  end;}
  {nb: using oldmCopyChannelFromBuffer takes ~21ms, but doesn't allow for decimation}
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CopyChannelsToAverager(BoardTag : integer);
var p, s, NumSamples, NumCopySamples, PrbIndex, BufferIndex, TrigOffset : integer;
begin
  PrbIndex:= (((ADC[BoardTag].BuffersCollected) mod BUFFERS_PER_SECOND)  //point to first sample...
                      * Length(DeMUXedDTBuffer) div BUFFERS_PER_SECOND); //...channel...probe
  with AvgWin do
  begin
    BufferIndex:= AvgBufferIndex;
    for p:= 0 to Config.NumProbes - 1 do
    begin
      with Config.Setup.Probe[p] do
      begin
        NumSamples:= SampPerChanPerBuff div Skippts;
        if ChanStart in CSDChannelList then
        begin
          TrigOffset:= AvgTriggerOffset div SkipPts;
          NumCopySamples:= NumSamples - TrigOffset;
          if (BufferIndex + NumCopySamples) >= Length(AvgRingBuffer) then
          begin //with this raw buffer, Avg buffer is full, so display and reset
            BuffersEmpty:= False;
            if muContinuous.Checked then
            begin
              for s:= 0 to High(AvgRingBuffer) - BufferIndex do
                SumRingBuffer[BufferIndex + s]:= DeMUXedDTBuffer[PrbIndex + s];
              n:= 1;
            end else
            begin
              for s:= 0 to High(AvgRingBuffer) - BufferIndex do
                inc(SumRingBuffer[BufferIndex + s], DeMUXedDTBuffer[PrbIndex + s]);
              inc(n);
            end;
            for s:= 0 to High(AvgRingBuffer) do AvgRingBuffer[s]:= Round(SumRingBuffer[s] / n);
            {if muCSDWave.Checked or muCSDColMap.Checked then} Compute1DCSD;
            RefreshChartPlot;
            AvgBufferIndex:= 0;
            AvgFillBuffer:= False;
            if StopAvgWhenFull then //only stop if RingBuffer complete...
            begin
              StopAvgWhenFull:= False;
              tbStartStopClick(Self);
            end;
            Exit;
          end else
          begin
            if AvgBufferIndex + NumCopySamples >= NumWavPts then
              NumCopySamples:= NumWavPts - AvgBufferIndex;
            if muContinuous.Checked then
              for s:= 0 to NumCopySamples - 1 do
                SumRingBuffer[BufferIndex + s]:= DeMUXedDTBuffer[PrbIndex + TrigOffset + s]
            else
              for s:= 0 to NumCopySamples - 1 do
                inc(SumRingBuffer[BufferIndex + s], DeMUXedDTBuffer[PrbIndex + TrigOffset + s]);
            inc(BufferIndex, NumWavPts);
          end;
        end{in};
        inc(PrbIndex, NumSamples * NChannels);
      end{probe};
    end{p};
    inc(AvgBufferIndex, NumCopySamples);
    AvgTriggerOffset:= 0;
  end{AvgWin};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.PutPTRecordsOnFileBuffer(Board : integer);
var p, PrbSaveNum, PrbIndex, NumWaveSamples : integer;
begin
  PrbIndex:= (ADC[board].BuffersCollected mod BUFFERS_PER_SECOND) * Length(DeMUXedDTBuffer) div BUFFERS_PER_SECOND;
  PrbSaveNum:= 0;
  for p:= 0 to Config.NumProbes - 1 do
  with Config.Setup.Probe[p] do
  begin
    NumWaveSamples:= NChannels * SampPerChanPerBuff div Skippts;
    if Save then
    with SSRingBuffer[SSRingBufferIndex] do
    begin
      UffType:= SURF_PT_REC_UFFTYPE; //this, and some fields of other ring buffer types could be pre-assigned
      SubType:= ProbeType;
      TimeStamp:= BufferStartTime;
      Probe:= PrbSaveNum;
      CRC32:= 0; //not yet implemented
      NumSamples:= NumWaveSamples;
      ADCWaveform:= @DeMUXedDTBuffer[PrbIndex]; //ADCWaveform:= Copy{aka slice}(DeMUXedDTBuffer, PrbIndex, NumWaveSamples{[PrbIndex]});
      inc(SSRingBufferIndex);//:= (SSRingBufferIndex + 1) mod SSRINGARRAYSIZE;
      inc(PrbSaveNum);
    end{save};
    inc(PrbIndex, NumWaveSamples);
  end{p};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CheckForPTRecordsToSave;
begin {this routine checks the SS/SE_PT_RingBuffer and saves them to disk}
  while (SSRingSaveIndex <> SSRingBufferIndex) do
  begin
    SurfFile.PutPolytrodeRecord(SSRingBuffer[SSRingSaveIndex]);
    SSRingSaveIndex:= (SSRingSaveIndex + 1) mod SSRINGARRAYSIZE;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CheckForSVRecordsToSave;
begin {this routine checks the SVRingBuffer and saves values to disk}
  while (SVRingSaveIndex <> SVRingBufferIndex) do
  begin
    SurfFile.PutSingleValueRecord(SVRingBuffer[SVRingSaveIndex]);
    SVRingSaveIndex:= (SVRingSaveIndex + 1) mod SVRINGARRAYSIZE;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CheckForMsgRecordsToSave;
begin {this routine checks the MsgRingBuffer and saves messages to disk}
  while (MsgRingSaveIndex <> MsgRingBufferIndex) do
  begin
    SurfFile.PutMessageRecord(MsgRingBuffer[MsgRingSaveIndex]);
    MsgRingSaveIndex:= (MsgRingSaveIndex + 1) mod MSGRINGARRAYSIZE;
  end;
end;

{-------------------------------------------------------------------------}
function CProbeSetupWin.CalcActualFreqPerChan(DesiredSampFreq : integer) : integer;
var i, b, s, nc, MaxChans, MaxExtRetrigFreq : integer;
  ChansPerBoard : array of integer;
begin
  {round SampFreqPerChan to precise board-capable frequency;
   also, limit SampFreqPerChan depending on current listsize;}
  {first, determine the maximum number of channels allocated to any one DT board}
  MaxChans:= 0;
  SetLength(ChansPerBoard, ContAcqForm.NumADBoards);
  for i:= 0 to ContAcqForm.NumADBoards - 1 do //initialise local array
    ChansPerBoard[i]:= 0;
  with ProbeSetupWin do
    for i := 0 to Setup.NProbes - 1 do
    begin
      s:= ProbeRow[i].ChanStartSpin.Value;
      b:= s div DT3010_MAX_SE_CHANS;
      nc:= ProbeRow[i].NumChanSpin.Value;
      if ((s mod DT3010_MAX_SE_CHANS) + nc) > DT3010_MAX_SE_CHANS then //probe traverses two (or more) boards
      begin
        inc(ChansPerBoard[b], DT3010_MAX_SE_CHANS - (s mod DT3010_MAX_SE_CHANS));
        inc(ChansPerBoard[b+1], ((s mod DT3010_MAX_SE_CHANS) + nc) mod DT3010_MAX_SE_CHANS);
      end else
        inc(ChansPerBoard[b], nc);
    end;
  b:= 0;
  for i:= 0 to ContAcqForm.NumADBoards - 1 do //determine board, b, with most chans, maxchans
    if ChansPerBoard[i] > MaxChans then
    begin
      MaxChans:= ChansPerBoard[i];
      b:= i;
    end;
  if (DINCheckBox.Checked) and (b < 2) //include DINPROBE if MaxChans is on board 0 or 1
    then inc(MaxChans);
  if MaxChans = 0 then MaxChans:= 1; //avoids DTxEz error if no channels yet allocated to any board
  if ContAcqForm.ADC[0].DTAcq.ClockSource = OL_CLK_EXTERNAL then //limit according to DTx guidelines...
    begin
      MaxExtRetrigFreq:= Round(1/((MaxChans/ContAcqForm.ADCFrequency)
                       + (1 / MASTER_CLOCK_FREQUENCY * 1{2}))); //...with a little more latitude
      if DesiredSampFreq > MaxExtRetrigFreq then
        DesiredSampFreq:= MaxExtRetrigFreq;
    end;
  try
    ContAcqForm.ADC[0].DTAcq.ListSize:= MaxChans;
    ContAcqForm.ADC[0].DTAcq.RetriggerFreq:= DesiredSampFreq;
    ContAcqForm.ADC[0].DTAcq.Config; //if using internal clock, driver limits retrigger frequency
  finally
    Result:= Round(ContAcqForm.ADC[0].DTAcq.RetriggerFreq);
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.DecodeStimulusDIN(BoardTag : integer);
var i{, SamplesPerChan} : Integer;
    DINDataPtr, DINStatusPtr : LPUSHRT;
begin
  DINStatusPtr:= ADC[0].hbufaddresses[ADC[BoardTag].hbufPtrIndex]; //point to first status DIN sample
  DINDataPtr:=   ADC[1].hbufaddresses[ADC[BoardTag].hbufPtrIndex]; //point to first data DIN sample
  if ADC[0].BuffersCollected = 0 then LastDINStatus:= DINStatusPtr^; //initialise LastDINStatus if undefined
  //SamplesPerChan:= ADC[0].BuffSize div ADC[0].ChansPerBuff; //use pre-computed SampPerChanPerBuff
  for i:= 0 to SampPerChanPerBuff - 1 do
  begin
    if StimulusDisplayRunning then
    begin
      if (LastDINStatus and DIN_DISP_RUNNING) - (DINStatusPtr^ and DIN_DISP_RUNNING) = DIN_DISP_RUNNING then
      begin //detected falling edge of run bit...
        StimulusDisplayRunning:= False;
        if RecordingPaused then
        begin
          sb_record.Enabled:= True;
          RecordingPaused:= False;
          Recording:= True;
        end;
        if PauseRecordMode and Recording then
        begin
          PostSurfMessage('Recording paused. Waiting for the next stimulus to begin...');
          sb_record.Enabled:= False;
          sb_record.Down:= False;
          Recording:= False;
          //StopPauseRecordMode:= True;
          //Break; //return to BufferDone proc. to handle queue before calling StartStopRecording
        end;
      end else
      if (DINStatusPtr^ and DIN_DISP_SWEEP) - (LastDINStatus and DIN_DISP_SWEEP) = DIN_DISP_SWEEP{high edge}then
      begin //update stimulus sweep variables...
        Dec(StimulusSweepsRemaining);
        //StimulusTimeRemaining:= Round(StimulusSweepsRemaining / InfoWin.GaugeDS.MaxValue
        //                      * StimulusHeader.est_runtime); //corrects for DS processing delays
        if AvgWinCreated and AvgWin.Running and (AvgFillBuffer = False) then
          with AvgWin do
          begin
            AvgTriggerOffset:= i;
            AvgFillBuffer:= True;
          end;
        //SweepChecksum:= Word(SweepChecksum + DINDataPtr^);   //update checksum
        //InfoWin.GaugeDS.Progress:= InfoWin.GaugeDS.MaxValue - StimulusSweepsRemaining;
      end;{else?}
      //OLD WAY IS TOGGLE, NOW VSync POSITIVE EDGE if ((DINStatusPtr^ and DIN_DISP_FRAME) <> (LastDINStatus and DIN_DISP_FRAME)) and GotValidDSHeader then
      if ((DINStatusPtr^ and DIN_DISP_FRAME) - (LastDINStatus and DIN_DISP_FRAME) = DIN_DISP_FRAME) and GotValidDSHeader then
      begin //get display frame-related data...
        inc(TotalNumFrames);
        if ((DINStatusPtr^ and DIN_DISP_SWEEP)= DIN_DISP_SWEEP) then //sweep bit high...
        begin //...only record frame toggles *within* sweeps (i.e. not during ISIs)
          inc(NumDINFrames);
          SweepChecksum:= Word(SweepChecksum + DINDataPtr^); //update checksum
          if StimulusDisplayPaused then
          begin
            StimulusDisplayPaused:= False;
            if RecordingPaused then
            begin //resume recording from paused mode...
              Recording:= True;
              RecordingPaused:= False;
              sb_record.Enabled:= True;
              PostSurfMessage('Stimulus resumed. Recording continued. Data file: '''
                      + ExtractFileName(SurfFile.SurfFileName) + '''');
            end{recpause}else
              PostSurfMessage('Stimulus resumed.');
          end;
          if Recording then
          begin
            with SVRingBuffer[SVRingBufferIndex] do
            begin
              UffType:= SURF_SV_REC_UFFTYPE;
              SubType:= SURF_DIGITAL;
              TimeStamp:= BufferStartTime + i * SamplePeriod + VSyncLagUsec;
              SVal:= DINDataPtr^;
            end;
            SVRingBufferIndex:= (SVRingBufferIndex + 1) mod SVRINGARRAYSIZE;
          end{recording};
        end{sweep bit high};
      end{frame vsync};
      if (DINStatusPtr^ and DIN_DATA_STROBE) <> (LastDINStatus and DIN_DATA_STROBE) then
      begin  //get data strobe-related data...
        if GotValidDSHeader then //data word must be final sweep-checksum
        begin
          InfoWin.lbStimTime.OnClick:= nil; //disable time mode toggle
          if StimulusSweepsRemaining > 0 then
          begin //stimulus run aborted
            if RecordingPaused then Recording:= True; //need so following msg written to file...
            InfoWin.lbStimTime.Caption:= 'Stimulus interrupted      ';// before complete      ';
            if Recording then CheckForSVRecordsToSave; //ensures residual stimulus DINs precede SurfMessage in data file
            if DINDataPtr^ = SweepChecksum then PostSurfMessage(InfoWin.lbStimTime.Caption
                           + '(checksum good).', BufferStartTime + i * SamplePeriod)
              else begin
                PostSurfMessage(InfoWin.lbStimTime.Caption + '(checksum bad).', BufferStartTime + i * SamplePeriod);
                FlagCriticalError;
              end;
          end else
          begin //stimulus run finished
            if Recording then CheckForSVRecordsToSave; //ensure residual stimulus DINs precede SurfMessage in data file
            if DINDataPtr^ = SweepChecksum then
            begin
              InfoWin.lbStimTime.Caption:= 'Done! (checksum good) ';
              PostSurfMessage(InfoWin.Label8.Caption + ' stimulus finished (checksum good).',
                              BufferStartTime + i * SamplePeriod)
            end else
            begin
              InfoWin.lbStimTime.Caption:= 'Done! (checksum bad) ';
              PostSurfMessage(InfoWin.Label8.Caption + ' stimulus finished (checksum bad).',
                              BufferStartTime + i * SamplePeriod);
              FlagCriticalError;
            end;
          end;
          GotValidDSHeader:= False;
          RefreshStimulusInfo;
          LastTotalFrames:= -1; //stops spurious 'paused' msgs
        end else
        begin //data word must be stimulus header-related (or spurious)...
          DINHeaderBuffer[NumDINHeader]:= DINDataPtr^;
          inc(NumDINHeader);
          if (NumDINHeader * 2 = SizeOf(StimulusHeader)) then //complete header sent
          begin
            ResetStimulusInfo;
            if ValidateStimulusHeader then
            begin
              GotValidDSHeader:= True;
              if Recording then //write header to file
              begin
                with DSPRecord do
                begin
                  UffType:= SURF_DSP_REC_UFFTYPE;
                  TimeStamp:= BufferStartTime + i * SamplePeriod;
                  DateTime:= (Now);
                  Header:= StimulusHeader;
                end;
                if not SurfFile.PutDSPHeaderRecord(DSPRecord) then
                begin
                  PostSurfMessage('Error writing stimulus display header to file.', BufferStartTime + i * SamplePeriod);
                  FlagCriticalError;
                  Continue;
                end;
              end{if recording};
              PostSurfMessage('''' + StimulusHeader.filename + ''' ('
                            + StimulusType[round(StimulusHeader.parameter_tbl[25])] + ')'
                            + ' stimulus started.', BufferStartTime + i * SamplePeriod);
            end{validDS} else
            begin //invalid stimulus header...
              DINStatusPtr:= ADC[0].hbufaddresses[ADC[0].hbufPtrIndex]; //get the DINStatus of the...
              inc(DINStatusPtr, (SampPerChanPerBuff - 1) * ADC[0].ChansPerBuff); //...last sample in the buffer
              LastDINStatus:= DINStatusPtr^;
              ResetStimulusInfo;
              StimulusDisplayRunning:= False;
              Break; //...to prevent residual DINHeader values from corrupting the next DS header
            end{invalidDS};
          end{header complete};
        end{not GotValidStimulusHeader};
      end{data strobe};
    end{disp_running}else
    if (DINStatusPtr^ and DIN_DISP_RUNNING) - (LastDINStatus and DIN_DISP_RUNNING) = DIN_DISP_RUNNING then
    begin //detected rising edge of run bit...
      StimulusDisplayRunning:= True;
      NumDINHeader:= 0; //discards any spurious data words (eg. frame checksums) from headerbuffer
      LastTotalFrames:= -1; //stops spurious 'paused' msgs at onset of this stimulus
      if PauseRecordMode then
      begin
        Recording:= True;
        RecordingPaused:= False;
        PostSurfMessage('Recording started. Data file: '''
                        + ExtractFileName(SurfFile.SurfFileName) + '''', BufferStartTime);
        sb_record.Enabled:= True;
        sb_record.Down:= True;
      end{pauserec};
    end;
    LastDINStatus:= DINStatusPtr^;
    inc(DINStatusPtr, ADC[0].ChansPerBuff);
    inc(DINDataPtr,   ADC[1].ChansPerBuff);
  end{i};
end;

{-------------------------------------------------------------------------}
function TContAcqForm.ValidateStimulusHeader : boolean; //success/failure
var StimHeaderPtr : PStimulusHeader;
  DINCheckSum : word;
  i, j, k, StimulusCode, StimRuns, {NumMSeqCells,} voff: integer;
begin
  Result:= True;
  try
    StimHeaderPtr:= @DINHeaderBuffer[0];
    StimulusHeader:= StimHeaderPtr^; //copy contents of DIN array to StimHeader record

    DINCheckSum:= 0; //calculate checksum
    for i:= 0 to Length(DINHeaderBuffer) - 2{exclude checksum itself} do
      DINCheckSum:= Word(DINCheckSum + DINHeaderBuffer[i]);

    with InfoWin do //update experiment infowin
    begin
      if DINCheckSum <> StimulusHeader.checksum then
      begin
        Label8.Font.Color:= clRed;
        Label8.Caption:= 'Bad checksum!';
        PostSurfMessage(Label8.Caption + ' Display stimulus header not written to file.');
        FlagCriticalError;
        Result:= False;
        Exit;
      end else
        Label8.Caption:= 'Checksum good';
      if (StimulusHeader.header <> 'DS') or (StimulusHeader.version <> 110) then
      begin
        Label8.Font.Color:= clRed;
        Label8.Caption:= 'DS version not supported.';
        PostSurfMessage(Label8.Caption + ' Display stimulus header not saved.');
        FlagCriticalError;
        Result:= False;
        Exit;
      end;
      with StimulusHeader do
      begin //calculate #stimulus sweeps for est. time remaining...
        StimulusCode:= Round(parameter_tbl[25]); //if NAN will fail to parse header
        {try StimRuns:= Round(parameter_tbl[27])
        except
          StimRuns:= 0; //if NAN
        end;}
        case StimulusCode of
      1..5, 10, 12, 14 : begin {multi-dimensional indexed stimuli}
                           //voff:= Round(parameter_tbl[2] - 1){dim table offset};
                           //i:= Round(parameter_tbl[voff]);
                           //j:= Round(parameter_tbl[voff + 1]);
                           //k:= Round(parameter_tbl[voff + 2]);
                           //StimulusSweepsRemaining:= i * j * k;
                           StimulusSweepsRemaining:= Round(parameter_tbl[59]); //stt_total_sweeps
                           Label14.Caption:= 'Sweeps:';
                         end;
      (*11 {sparse noise}: begin
                           StimulusSweepsRemaining:= Round(parameter_tbl[24]{duration} * frame_rate
                                                         / parameter_tbl[44]{frames per stim});
                           Label14.Caption:= 'Frames:';
                         end;*)
        11, 13, 15, 16 : begin {sparse noise, m sequence, image file sequence, movie}
                           {NumMSeqCells:= Round(parameter_tbl[45]);
                           if NumMSeqCells = 16 then StimulusSweepsRemaining:= 16383
                             else StimulusSweepsRemaining:= 65535;}
                           StimulusSweepsRemaining:= Round(parameter_tbl[59]); //stt_total_sweeps
                           Label14.Caption:= 'Frames:';
                         end;
             (*16 {movie}: begin
                           StimulusSweepsRemaining:= Round(est_runtime * frame_rate);
                           Label14.Caption:= 'Frames:';
                         end;*)
                    else begin {stimulus duration undefined}
                           StimulusSweepsRemaining:= 0;
                           StimulusTimeRemaining:= 0;
                         end;
        end{case};
        //StimulusSweepsRemaining:= StimulusSweepsRemaining * (StimRuns {+ 1});
        if StimulusSweepsRemaining <> 0 then GaugeDS.MaxValue:= StimulusSweepsRemaining;
        StimulusTimeRemaining:= Round(StimulusSweepsRemaining / InfoWin.GaugeDS.MaxValue
                              * StimulusHeader.est_runtime);
        Label8.Font.Color:= clNavy;
        Label8.Caption:= '''' + filename + ''' (' + StimulusType[StimulusCode] + ')';
        Label5.Caption:= FloattoStrF(frame_rate, ffGeneral, 3,1) + 'Hz';
        VSyncLagUsec:= round(1 / frame_rate * 1000000); //corrects for one frame OpenGL latency with MAS
      end{StimulusHeader};
    end{infowin};
  except
    PostSurfMessage('Error passing stimulus header information.');
    Result:= False;
  end;
end;


{-------------------------------------------------------------------------}
procedure CProbeWin.ThreshChange(pid, seThreshold : integer);
begin
  with ContAcqForm, ProbeWin[pid] do
  begin
    DispTrigBipolar:= muBipolarTrig.Checked;
    if DispTrigBipolar then
    begin
      Config.Setup.Probe[pid].Threshold:= Abs(seThreshold) + 2048;
      PolarityLbl:= '±';
    end else
    begin
      Config.Setup.Probe[pid].Threshold:= seThreshold + 2048;
      if seThreshold > 0 then
      begin
        DispTrigPositive:= True;
        PolarityLbl:= '+';
      end else
      begin
        DispTrigPositive:= False;
        PolarityLbl:= '-';
      end;
    end;
  end{with};
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.tb_expinfoClick(Sender: TObject);
begin
  InfoWin.Visible:= not(InfoWin.Visible);
  tb_expinfo.down:= InfoWin.Visible;
  Config.InfoWinVisible:= InfoWin.Visible;
  if InfoWin.Visible then InfoWin.BringToFront;
end;

{-------------------------------------------------------------------------}
procedure CProbeWin.ClickProbeChan(pid, ChanNum : byte);
begin
  with ContAcqForm do
  begin
    inc(ChanNum, Config.Setup.Probe[pid].ChanStart);
    if SelectEEGChannel then //create new/select spectrogram window...
    begin
      if ChanNum in EEGWinsOpen then Exit else //WHAT IF USER CLOSES FORM WINDOW?!
      EEGWin[EEGWinIndex]:= TEEGWin.Create(Self);
      with EEGWin[EEGWinIndex] do
      begin
        Top:= 100;
        Left:= ClientWidth-InfoWin.Width - 10;
        Channel:= ChanNum;
        SampleFreq:= Config.Setup.Probe[pid].SampFreq;
        TotalGain:= Config.Setup.Probe[pid].InternalGain * EXTERNAL_AMPLIFIER_GAIN;
        Show;
      end;
      inc(EEGWinIndex);
      include(EEGWinsOpen, ChanNum);
      Screen.Cursor:= crDefault;
      SelectEEGChannel:= False;
    end else
    if MUXChan.Enabled then //select MUX channel...
    begin
      if not (Chan2MUX(ChanNum)) then tb_MUX.Caption:= '-- error -- '
      else begin
        tb_MUX.Caption:= 'CH ' + inttostr(ChanNum);
        MUXChan.Position:= ChanNum;
      end;
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure CProbeWin.CreatePolytrodeGUI(pid : integer);
var ProbeName : shortstring;
begin
  if muPolytrodeGUI.Checked = False then
  begin
    if GUICreated then GUIForm.Close;
    GUICreated:= False;
    Exit;
  end else
  try
    if GUICreated then GUIForm.Close; //remove any existing GUIForms...
    GUIForm:= TPolytrodeGUIForm.CreateParented(ContAcqForm.WaveFormPanel.Handle);//..and create a new one
    GUIForm.Left:= Left + 10;
    GUIForm.Top:= ContAcqForm.ProbeWin[pid].Win.Top + 20;
    GUIForm.Show;
    GUIForm.BringToFront;
    GUICreated:= True;
  except
    muPolytrodeGUI.Checked:= False;
    GUICreated:= False;
    Exit;
  end;
  ProbeName:= ContAcqForm.Config.Setup.Probe[pid].ElectrodeName;
  if not GUIForm.CreateElectrode(ProbeName, True) then
  begin
    GUIForm.Free;
    muPolytrodeGUI.Checked:= False;
    GUICreated:= False;
    Exit;
  end;
  GUIForm.Caption:= ProbeName;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.tb_EEGClick(Sender: TObject);
var i : integer;
begin
  {determine how many EEG windows already open}
  {nb. using 'Assigned' or '=nil' doesn't work}
  if Config.NumAnalogChannels = 0 then Exit;
  EEGWinIndex:= 0;
  for i:= 0 to Screen.FormCount - 1 do
    if Screen.Forms[i] is TEEGWin then
      inc(EEGWinIndex);
  if EEGWinIndex = MAX_EEG_WINDOWS then Exit;
  Screen.Cursor:= crHelp;
  SelectEEGChannel:= True;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.tb_CSDClick(Sender: TObject);
begin
  AddRemoveCSDWin;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.AddRemoveCSDWin;
var i, s, c : integer;
begin
  if Config.Setup.NCRProbes = 0 then tb_CSD.Down:= False;
  if tb_CSD.Down = False then
  begin
    if AvgWinCreated then AvgWin.Release;
    AvgWinCreated:= False;
    AvgFillBuffer:= False;
    Exit;
  end else
  try
    if AvgWinCreated then AvgWin.Release; //remove any existing CSDForms...
    AvgWin:= CAveragerWin.CreateParented(ContAcqForm.WaveFormPanel.Handle);//...and create a new one
    with AvgWin do
    begin
      NumChans:= 0;//Config.Setup.NCRProbes;
      for i:= 0 to Config.NumProbes -1 do //include all EEG (CR) probe channels in CSD by default
      if Config.Setup.Probe[i].ProbeType = CONTINUOUS then
        begin                      //but currently only if 'EEG ch' string found in probe description
          c:= Pos(CSD_SEARCH_STRING, Config.Setup.Probe[i].Descrip);
          if c = 0 then
          begin //not LFP CR probe, so ignore...
            exclude(CSDChannelList, Config.Setup.Probe[i].ChanStart);
            Continue;
          end;
          s:= strtoint(Copy(Config.Setup.Probe[i].Descrip, c + 6, 2));
          SitesSelected[s]:= True;
          include(CSDChannelList, Config.Setup.Probe[i].ChanStart);
          inc(NumChans);
        end;
      for i:= 0 to KNOWNELECTRODES - 1 {from ElectrodeTypes} do //select current electrode
        if Config.Setup.Probe[0{assumes first probe is CSD probe}].ElectrodeName = KnownElectrode[i].name then
          CElectrode.ItemIndex:= i; //REMOVE FIRST PROBE HARDCODING ASSUMPTION!
      NumWavPts:= DEFAULT_NUM_CSD_SAMPLES;
      for i:= 0 to Config.NumProbes -1 do //also ASSUMES INT. GAIN AND SAMPLE RATE OF FIRST CR PROBE APPLIES TO ALL...
        if Config.Setup.Probe[i].ProbeType = CONTINUOUS then
        begin
          AD2uV:= (20 / Config.Setup.Probe[i].InternalGain / RESOLUTION_12_BIT) / EXTERNAL_AMPLIFIER_GAIN * V2uV;
          SampleRate:= Config.Setup.Probe[i].SampFreq;
          Break;
        end;
      Setlength(SumRingBuffer, NumChans * NumWavPts);
      Setlength(AvgRingBuffer, NumChans * NumWavPts);
      Setlength(CSDRingBuffer, (NumChans - CSD_nDELTAY - CSD_nDELTAY) * NumWavPts);//nb: if numchannels < CSD_nDELTAY...
      InitialiseSplineArrays;
      ResetAverager; //...code will pass to exception clause and close CSDWin (messy, but works)
      SetAutoYScale;
      OnResize(Self);//compute y indicies depending on numchans
      CElectrode.OnChange(Self); //program....
      CElectrode.Free; //for offline chartwins only
      Spacer1.Free;
      Caption:= 'CSD Profile';
      Height:= NumChans * 50;
      Left:= 30;
      Top:= 75;
      Show;
    end{with};
    AvgWinCreated:= True;
  except
    AvgWin.Release;
    AvgWinCreated:= False;
    AvgFillBuffer:= False;
    tb_CSD.Down:= False;
  end;
end;

{-----------------------------------------------------------------------------}
procedure CAveragerWin.MoveTimeMarker(MouseXFraction : single);
begin
  if (MouseXFraction <= 0.0) or (MouseXFraction >= 1.0) then Exit;
  PlotVertLine(Round(MouseXFraction * NumWavPts));
  ChartHintWindow.ActivateHint(Rect(Mouse.CursorPos.x - 50, Mouse.CursorPos.y - 18,
                               Mouse.CursorPos.x - 10, Mouse.CursorPos.y - 5),
                               inttostr(Round(NumWavPts*MouseXFraction/SampleRate*1000))+'ms');
end;

{-----------------------------------------------------------------------------}
procedure CAveragerWin.OneShotFillBuffers;
begin
  ContAcqForm.AvgFillBuffer:= True;
end;

{-----------------------------------------------------------------------------}
procedure CAveragerWin.RefreshChartPlot;
var i, j  : integer;
    TestImage : TWaveformArray;
begin
  //if BuffersEmpty then Exit;
  if LeftButtonDown then Exit;
  if muLFPColMap.Checked then
  begin //  !!!!ComputeXChanSpline to work with AvgRingBuffer (1D array)!!!!
    Setlength(TestImage, NumChans, NumWavPts); //TEMPORARY CODE!
    for i:= 0 to NumChans - 1 do               //TEMPORARY CODE!
      for j:= 0 to NumWavPts - 1 do            //TEMPORARY CODE!
        Testimage[i, j]:= AvgRingBuffer[NumWavPts * i + j];
    if muSpline.Checked then
      SplineInterpXChan(TestImage, NumChans)
    else if muLinear.Checked then
      LinInterpXChan(TestImage, NumChans)
    else
      NoInterpXChan(TestImage, NumChans);
    PlotColMap(ColourMap2D, 12, 0.5);
  end else //!!!!ComputeXChanSpline to work with AvgRingBuffer (1D array)
  if muCSDColMap.Checked then
  begin
    Setlength(TestImage, NumChans - CSD_nDELTAY - CSD_nDELTAY, NumWavPts); //TEMPORARY CODE!
    for i:= 0 to NumChans - CSD_nDELTAY - CSD_nDELTAY - 1 do //TEMPORARY CODE!
      for j:= 0 to NumWavPts - 1 do                          //TEMPORARY CODE!
        Testimage[i, j]:= CSDRingBuffer[NumWavPts * i + j];
    if muSpline.Checked then
      SplineInterpXChan(TestImage, NumChans - CSD_nDELTAY - CSD_nDELTAY)
    else if muLinear.Checked then
      LinInterpXChan(TestImage, NumChans - CSD_nDELTAY - CSD_nDELTAY)
    else
      NoInterpXChan(TestImage, NumChans - CSD_nDELTAY - CSD_nDELTAY);
    PlotColMap(ColourMap2D, 12, 1.5{CSD_nDELTAY / 2 + 0.5}, 1);
  end else
    BlankPlotArea;
  if muLFPWave.Checked then PlotLFP(@AvgRingBuffer[0], NumWavPts);
  if muCSDWave.Checked then PlotCSD(@CSDRingBuffer[0], NumWavPts); //plotCSD same as plotLFP expect latter expects LONGS in the array
  PlotStatusLabels;
  //PlotXAxis;
  Paint; //blit bitmap to Chartwin's canvas
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.WaveFormPanelClick(Sender: TObject);
begin
  if SelectEEGChannel = False then Exit;
  SelectEEGChannel:= False; //if user clicks outside probe window...
  Screen.Cursor:= crDefault;//...cancel EEG channel select
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.ReduceFlicker(var message : TWMEraseBkgnd);
begin
  message.result:= 1;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.PrecisionClockOverflow(Sender: TObject; var lStatus: Integer);
begin {this applied to DT340 interrupt only}
  PostSurfMessage('Warning: 32bit counter overflow!');
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.tb_msgClick(Sender: TObject);
var UserMessage : string;
begin
  if (InputQuery('Memo', 'Enter user message, <esc> to cancel...',
    UserMessage)) and (UserMessage <> '') then
      PostSurfMessage(UserMessage, 0, USER_MESSAGE);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.PostSurfMessage(const MessageString : string;
                                       const SpecifiedTimeStamp : Int64{default is zero};
                                       const MsgType   : char{default is SurfMessage});
var
  DateAndTime : TDateTime;
begin
  DateAndTime:= (Now);
  with MsgRingBuffer[MsgRingBufferIndex] do
  begin
    UffType  := SURF_MSG_REC_UFFTYPE;
    SubType  := MsgType;
    if Recording then
      if SpecifiedTimeStamp = 0 then TimeStamp:= GetPrecisionTime else
        TimeStamp:= SpecifiedTimeStamp
    else TimeStamp:= 0; //if not recording
    DateTime := DateAndTime;
    MsgLength:= Length(MessageString);
    Msg      := MessageString;
  end;
  StatusBar.Panels[3].Text:= TimeToStr(Time)   + ': '  + MessageString; //add entry to StatusBar
  MsgMemo.Lines.Append(DateTimeToStr(DateAndTime) + ':  ' + MessageString); //add entry to Surflog
  if Recording or (MsgType = USER_MESSAGE) then //if recording, spool message for saving to file
    MsgRingBufferIndex:= (MsgRingBufferIndex + 1) mod MSGRINGARRAYSIZE;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.MsgMemoChange(Sender: TObject);
begin //limit surf log history to MSGRINGARRAYSIZE lines, and control cursor/scrolling
  MsgMemo.Lines.BeginUpdate;
  if MsgMemo.Lines.Count > MSGRINGARRAYSIZE then
  begin
    MsgMemo.Lines.Delete(0);
    SendMessage(MsgMemo.Handle, EM_LINEINDEX, MSGRINGARRAYSIZE, 0);
  end else
    SendMessage(MsgMemo.Handle, EM_LINESCROLL, 0, -1);
  MsgMemo.Lines.EndUpdate;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.DTAcqQueueUnderrun(Sender: TObject);
begin
  if DTUnderrrunError then Exit; //ignore all but first underrun
  PostSurfMessage('Critical error! Software buffer underrun. Halting subsystems.');
  DTUnderrrunError:= True;
  sb_play.down:= False;
  sb_record.down:= False;
  sb_StopClick(Self);
  LEDs.Tag:= YELLOW_LED;
  StatusBar.Invalidate;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.DTAcqOverrun(Sender: TObject);
begin
  if DTUnderrrunError then Exit; //ignore all but first underrun
  PostSurfMessage('Critical error! Hardware buffer overrun. Halting subsystems.');
  DTUnderrrunError:= True;
  sb_play.down:= False;
  sb_record.down:= False;
  sb_StopClick(Self);
  LEDs.Tag:= YELLOW_LED;
  StatusBar.Invalidate;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.DTBoardsOutOfSync;
begin
  PostSurfMessage('Critical error! ADC boards out of sync. Halting subsystems.');
  sb_play.down:= False;
  sb_record.down:= False;
  sb_StopClick(Self); //halt acquisition/recording
  LEDs.Tag:= YELLOW_LED;
  StatusBar.Invalidate;
  {re}ConfigBoard; //essential to realign buffers!
  //BoardsOutOfSync:= False;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.NewDatafileClick(Sender: TObject);
var Confirm : boolean;
begin
  if Config.Empty then
    MessageDlg('No probe configuration', mtWarning, [mbOk], 0)
  else
    if NewDataFileDialog.Execute then
    begin
      Confirm:= True;
      DataFileOpen:= False;
      if FileExists(NewDataFileDialog.Filename) then
        if MessageDlg(NewDataFileDialog.Filename+' already exists.  Overwrite?', mtWarning,
          [mbYes,mbNo],0) = mrNo then Confirm:= False;
      if Confirm then
      begin
        if not SurfFile.OpenSurfFileForWrite(NewDataFileDialog.Filename)
          then MessageDlg('Error creating/overwriting ' + NewDataFileDialog.Filename, mtWarning, [mbOk], 0)
        else begin
          DataFileOpen:= True;
          WriteSurfLayoutRecords;
          InfoWin.Label6.Hint:= SurfFile.SurfFileName;
          if (Length(SurfFile.SurfFileName) * InfoWin.Label6.Font.Size * 2/3 - 20)
            > InfoWin.Label6.ClientWidth{available space for filename} then
            InfoWin.Label6.Caption:= Copy(SurfFile.SurfFileName, 0, 30) + '....'
          else InfoWin.Label6.Caption:= SurfFile.SurfFileName{path + name fits};
          if NumSaveProbes{analog} > 0 then InfoWin.Label12.Caption:=
              'Saving ' + inttostr(NumSaveProbes) + '/'
            + Inttostr(Config.NumProbes) + ' probes ('
            + FloattostrF(FileBandWidth / 1048576, ffNumber, 4, 1) + 'MB/s; '
            + FloattostrF(FileBandWidth / 291.271, ffNumber, 4, 1) + 'MB/hr)'
          else if Config.AcqDIN then InfoWin.Label12.Caption:= '(only stimulus data will be saved)'
          else InfoWin.Label12.Caption:= '(no data will be saved)';
          RecTime:= 0;
          TotalRecTime:= 0;
          RefreshFileInfo;
          StatusBar.Panels[1].Text:= ExtractFileName(SurfFile.SurfFileName);
          StatusBar.Panels[1].Width:= Round(Length(StatusBar.Panels[1].Text)
                                       * StatusBar.Font.Size * 2/3) + 10;
          {disable/enable user controls}
          tbCloseDataFile.Enabled:= True;
          tbNewDataFile.Enabled:= False;
          muCloseFile.Visible:= True;
          muNewFile.Visible:= False;
          muCfgNew.Visible:= False;
          muCfgOpen.Visible:= False;
          muCfgModify.Visible:= False;
          tbNewConfig.Enabled:= False;
        end;
      end{confirm} else DataFileOpen:= False;
    end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CloseDataFileClick(Sender: TObject);
begin
  CloseDataFile;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.CloseDataFile;
begin
  if DataFileOpen then
  begin
    PostSurfMessage('''' + ExtractFileName(SurfFile.SurfFileName) + ''' closed.');
    CheckForMsgRecordsToSave; //writes to file any user messages entered post-record
    RefreshFileInfo;
    SurfFile.CloseFile;
    DataFileOpen:= False;
    {disable/enable user controls}
    tbCloseDataFile.Enabled:= False;
    muCloseFile.Visible:= False;
    muCfgNew.Visible:= True;
    muCfgOpen.Visible:= True;
    if not Config.Empty then
    begin
      muCfgModify.Visible:= True;
      muNewFile.Visible:= True;
    end;
    tbNewConfig.Enabled:= True;
    tbNewDataFile.Enabled:= True;
  {end else
      MessageDlg('No data file open', mtWarning, [mbOk], 0);
    if not DataFileOpen then
  begin}
    InfoWin.Label6.Caption:= 'Data file closed';
    InfoWin.Label12.Caption:= '';
    StatusBar.Panels[1].Text:= ExtractFileName(CfgFileName);
    StatusBar.Panels[1].Width:= Round(Length(StatusBar.Panels[1].Text)
                                 * StatusBar.Font.Size * 2/3) + 10;
  end;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.FlagCriticalError;
begin
  LEDs.Tag:= 21; //yellow LED TImageList index
  StatusBar.Invalidate;
  ErrorTimeCount:= ADC_SATURATED_ERROR_INTERVAL;
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.StatusBarDrawPanel(StatusBar: TStatusBar;
                       Panel: TStatusPanel; const Rect: TRect);
begin
  with LEDs do Draw(StatusBar.Canvas, 3, 4, Tag);
end;

{-------------------------------------------------------------------------}
function TContAcqForm.GetPrecisionTime : Int64;
begin                                   //with a 1MHz master clock...
  Result:= DT32BitCounter.CTReadEvents; //...overflows at 2^32µsec, ~112 minutes
  if Old32BitTime > Result then //timer has wrapped...
    inc(Time32BitOverflow, RESOLUTION_32_BIT{2^32});
  Old32BitTime:= Result;
  inc(Result, Time32BitOverflow);
end;

{-------------------------------------------------------------------------}
procedure TContAcqForm.FormClose(Sender: TObject; var Action: TCloseAction);
var b : integer;
begin
  if Recording then StartStopRecording else
    if Acquiring then StartStopAcquisition;
  CloseDataFile;
  FreeAndNil(SurfFile);

  {free h/w and s/w buffers}
  FreeDTBuffers;
  for b:= 0 to High(SSRingBuffer) do //free dynamic arrays in SSRingBuffer
    SSRingBuffer[b].ADCWaveform:= nil;
  DeMUXedDTBuffer:= nil;
  SSRingBuffer:= nil;
  {others?}

  {free runtime-created windows/forms/dialogs}
  FreeChanWindows;
  InfoWin.Release;
  if AvgWinCreated then AvgWin.Release;

  //FreeEEGWindows; <-- not needed, as owner is Contacqform
  {others?}

  {free DT objects/resources}
  for b:= 0 to NumADBoards - 1 do
    FreeAndNil(ADC[b].DTAcq);
  ADC:= nil;
  if DT32BitCounterEnabled and MasterClockEnabled then
  begin
    DT32BitCounter.Free;
    DTMasterClock.Free;
  end;
  if ExpTimerEnabled then WinExpTimer.Free;//DTExpTimer.Free;
  if MUXDOUTEnabled then DTMUX.Free;
  {others?}
  Action:= caFree; //free main form, exit SURF
end;

end.
