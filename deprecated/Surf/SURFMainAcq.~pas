unit SURFMainAcq;

interface
//saving to files
//check problems with increase in surf_max_channels

uses About,Windows, Messages, ShellApi, ExtCtrls, Buttons, StdCtrls, ToolWin, ComCtrls,
  Controls, OleCtrls, Classes, Gauges, DTxPascal, DTAcq32Lib_TLB,
  Forms, Dialogs, Math, Menus, SysUtils, Graphics, SurfComm, SuperTimer, SurfFile, SurfTypes,
  SurfPublicTypes, {WaveFormUnit,-old-} ProbeSet, InfoWinUnit, PahUnit, SurfMessage,
  EnterExtGain, {SuperTimer, Surf2SurfBridge,} WaveFormPlotUnit, ElectrodeTypes;

{$DEFINE ADBOARD} //select $undef if no adboard present

const
  SPIKERINGARRAYSIZE = 4000;
  CRRINGARRAYSIZE    = 500;
  SVRINGARRAYSIZE    = 3000;

  DINPROBE = 32;

type
  PolytrodeRecord = record
    Time     : LNG;
    ProbeNum : SHRT;
    Cluster  : SHRT;
    Waveform : TWaveForm;
  end;

  CRTempRecordType = record
    Time     : LNG;
    ProbeNum : SHRT;
    Waveform : array[0..SURF_MAX_WAVEFORM_PTS-1] of SHRT;
  end;

  SVRecordType = record
    Time : LNG;
    SubType : CHAR;//'D'(digital) or 'A'(analog)
    SVal : WORD;
  end;

  TheTimeRec = object
    public
      TenthMS,MS,Sec,Min : Longint;
    private
      LastCount,CurCount,LastTenthMS,LastMS,LastSec : Longint;
  end;

  TSurfComObj = class(TSurfComm)
    public
      Procedure PutDACOut(DAC : TDAC); override;
      Procedure PutDIOOut(DIO : TDIO); override;
  end;

  TSurfMesg = class(TMesgQueryForm)
    public
      procedure MessageSent(mesg : ShortString); override;
    private
  end;

  CProbeWin = class(TWaveFormPlotForm)
    public
      Procedure ThreshChange(pid,threshold : integer); override;
    end;

  TProbeWin = record
    exists : boolean;
    win : CProbeWin;
  end;

  TWindowLoc = record
    Left,Top,Width,Height : integer;
  end;

  TCglList = record
    ProbeId       : integer;
    ChanOffset    : integer;
  end;

  TConfig = record
    NumProbes,NumAnalogChannels : integer;
    DinChecked : boolean;
    CglList : array[0..SURF_MAX_CHANNELS-1] of TCGLList;
    WindowLoc : array[0..SURF_MAX_PROBES-1] of TWindowLoc;
    Setup : TProbeSetup; //same as probewin.setup
    MainWinLeft,MainWinTop,MainWinHeight,MainWinWidth : Integer;
    InfoWinLeft,InfoWinTop,InfoWinHeight,InfoWinWidth : Integer;
    empty : boolean;
  end;

  TSurfAcqForm = class(TForm)
    FileInfoPanel: TStatusBar;
    NewDialog: TSaveDialog;
    SaveConfigDialog: TSaveDialog;
    OpenConfigDialog: TOpenDialog;
    WaveFormPanel: TPanel;
    ToolBar2: TToolBar;
    Splitter9: TSplitter;
    MainMenu: TMainMenu;
    About1: TMenuItem;
    WriteMessage: TButton;
    Splitter1: TSplitter;
    stimer: TSuperTimer;
    DataFile1: TMenuItem;
    NewDataFile: TMenuItem;
    CloseDataFile: TMenuItem;
    Config1: TMenuItem;
    ConfigProbesBut: TMenuItem;
    OpenConfig: TMenuItem;
    SaveConfig: TMenuItem;
    SaveConfigAs: TMenuItem;

    DTClock: TDTAcq32;
    DTDAC: TDTAcq32;
    DTCounter: TDTAcq32;

    Splitter2: TSplitter;
    PAcquire: TPanel;
    PRecord: TPanel;
    PInfoWin: TPanel;
    Splitter3: TSplitter;
    DTDIO: TDTAcq32;
    DTAcq: TDTAcq32;
    procedure ExitItemClick(Sender: TObject);
    procedure About1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);

    procedure DTAcqOverrunError(Sender: TObject);
    procedure DTAcqTriggerError(Sender: TObject);
    procedure DTAcqQueueStopped(Sender: TObject);
    procedure DTAcqBufferDone(Sender: TObject);
    procedure DTAcqUnderrunError(Sender: TObject);

    procedure ConfigProbesButClick(Sender: TObject);
    procedure stimerTimer(Sender: TObject);
    procedure RecButClick(Sender: TObject);
    procedure FileExitClick(Sender: TObject);
    procedure CloseDataFileClick(Sender: TObject);
    procedure SaveConfigClick(Sender: TObject);
    procedure SaveConfigAsClick(Sender: TObject);
    procedure OpenConfigClick(Sender: TObject);
    procedure NewDataFileClick(Sender: TObject);
    procedure WriteMessageClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure FormHide(Sender: TObject);
    procedure PAcquireClick(Sender: TObject);
    procedure PRecordClick(Sender: TObject);
    procedure PInfoWinClick(Sender: TObject);
  private
    { Private declarations }
    ClockOnBoard : boolean;
    RawRingArray : array of WORD;//SHRT;
    RawRingBuffNum : Integer;
    SpikeRingSaveReadIndex,SpikeRingDisplayReadIndex : LNG;
    CRRingSaveReadIndex,CRRingDisplayReadIndex :LNG;
    {DinRingSaveReadIndex,}SvRingSaveReadIndex : LNG;
    CRPointCount,CRSkipCount :array[0..SURF_MAX_PROBES-1] of Integer;
    CRTempRecordArray : array[0..SURF_MAX_PROBES-1] of CRTempRecordType;
    GotCfgFileName : Boolean;
    CfgFileName : String;
    FirstBufferDone,SecondBufferDone : boolean;
    nrawspikes,nsavedspikes,ndisplayedspikes,ndroppedspikes : LNG;
    nsavedcrbuffers,ndisplayedcrbuffers,nrawcrbuffers : LNG;
    nsaveddinbuffers,nrawdinbuffers : LNG;
    TotalBuffersAcquired : LNG;
    TimeScale : Double;
    BuffSize,NumBuffs : LNG;
    ShiftDown,BuffersAllocated : boolean;
    ProbeWin : array[0..SURF_MAX_PROBES-1] of TProbeWin;
    RawRingArraySize : {U}LNG;
    SampFreq : Integer;
    SingleWave : TWaveForm;
    InfoWin : TInfoWin;
    LastSpikeTrigPtofProbe  :array[0..SURF_MAX_PROBES-1] of Integer;
    LastSpikeTimeofProbe  :array[0..SURF_MAX_PROBES-1] of Integer;
    LastDinVal : Word;
    MaxFreq,MinFreq : LNG;
    SurfComm : TSurfComObj;
    //UserWinHandle{,Lynx8AmpHandle} : HWnd;

    SaveSURF : TSurfFile;
    Acquisition,Recording,FileIsOpen : boolean;
    TheTime : TheTimeRec;
    SpikeRingWriteIndex,CRRingWriteIndex,{DinRingWriteIndex,}SvRingWriteIndex : LNG;
    SpikeRingArray : array[0..SPIKERINGARRAYSIZE-1] of PolytrodeRecord;
    CRRingArray    : array[0..CRRINGARRAYSIZE-1] of PolytrodeRecord;
    SvRingArray    : array[0..SVRINGARRAYSIZE-1] of SvRecordType;

    UserWinExists{,Lynx8AmpControlExists,SettingLynx8} : boolean;
    config : TConfig;
    msgrec : SURF_MSG_REC;
    MesgQueryForm: TSurfMesg;

    Procedure StartStopAcquisition;
    Procedure StartStopRecording;
    Procedure PutWindowLocationsInConfig(probe : integer);
    //Procedure GetWindowLocationsFromConfig(probe : integer);
    procedure SetUpProbes;
    //Procedure CreateAChanWindow(id,probenum,channum,left,top,npts : integer);
    Procedure FreeChanWindows;
    Function  CreateAProbeWindow(probenum,left,top,npts : integer) : boolean;
{$IFDEF ADBOARD}
    Function InitBoard : boolean; //false if error
{$ENDIF}
    function ConfigBoard : boolean; //false if error
    function UnloadBuffers : boolean; //false if error
    function SetupProbeWins : boolean; //false if error
    Procedure CheckForSpikesToSave;
    Procedure CheckForSpikesToDisplay;
    Procedure CheckForCRRecordsToSave;
    Procedure CheckForCRRecordsToDisplay;
    //Procedure CheckForDinRecordsToSave;
    Procedure CheckForSvRecordsToSave;
    Procedure WriteSURFRecords;
    Procedure SaveSurfConfig(Filename : String);
    Procedure OpenSurfConfig(Filename : String);
    procedure UpdateTimer;
  public
    NumUserApps : integer;
    UserFileNames : array of string;

    procedure AcceptFiles( var msg : TMessage ); message WM_DROPFILES;
  end;

var
  SurfAcqForm: TSurfAcqForm;

implementation

{$R *.DFM}

procedure TSurfAcqForm.ExitItemClick(Sender: TObject);
begin              
  Close;
end;

procedure TSurfAcqForm.About1Click(Sender: TObject);
begin
  AboutBox.ShowModal;
end;

{==============================================================================}
procedure TSurfAcqForm.FormCreate(Sender: TObject);
begin
  DragAcceptFiles( Handle, True ); //let Windows know form accepts dropped files}
end;

{==============================================================================}
procedure TSurfAcqForm.AcceptFiles( var msg : TMessage );
const
  cnMaxFileNameLen = 255;
var
  acFileName : array [0..cnMaxFileNameLen] of char;
begin
  //this gives you how many files were dragged on
  DragQueryFile( msg.WParam,$FFFFFFFF,acFileName,cnMaxFileNameLen );
  Application.BringToFront;
  //take first one
  DragQueryFile( msg.WParam, 0, acFileName, cnMaxFileNameLen );

  OpenSurfConfig (acfilename);
  DragFinish( msg.WParam );
end;

{==============================================================================}
{$IFDEF ADBOARD}
Function TSurfAcqForm.InitBoard : boolean; //false if error
begin
  RESULT := TRUE;
  ClockOnBoard := FALSE;

  //Select Board
  if DTAcq.numboards > 0 then
    DTAcq.Board  := DTAcq.BoardList[0]
  else begin
    ShowMessage('No boards found');
    RESULT := FALSE;
    Exit;
  end;

// Setup subsystem for A to D
  DTAcq.Subsystem := OLSS_AD;
// Setup DataFlow
  //Can this board can handle continuous mode? If so then set it.
  if DTAcq.GetSSCaps(OLSSC_SUP_CONTINUOUS) <> 0
  then DTAcq.DataFlow := OL_DF_CONTINUOUS //set up for continuous about trig operation
  else begin
    ShowMessage(DTAcq.Board+ ' can not support continuous about trig mode');
    RESULT := FALSE;
    Exit;
  end;
//Set subsystem params
  //chan type
  if DTAcq.GetSSCaps(OLSSC_SUP_SINGLEENDED) <> 0 then
    DTAcq.ChannelType := OL_CHNT_SINGLEENDED //set up for single ended acquisition
  else begin
    ShowMessage(DTAcq.Board+ ' can not support single ended acquisition');
    RESULT := FALSE;
    Exit;
  end;
  if DTAcq.GetSSCaps(OLSSC_SUP_BINARY) <> 0 then
    DTAcq.Encoding := OL_ENC_BINARY //set up for binary encoding
  else begin
    ShowMessage(DTAcq.Board+ ' can not support binary encoding');
    RESULT := FALSE;
    Exit;
  end;
  if DTAcq.GetSSCaps(OLSSC_SUP_INTCLOCK) <> 0 then
    DTAcq.ClockSource := OL_CLK_INTERNAL //Set clock to internal
  else begin
    ShowMessage(DTAcq.Board+ ' does not support internal clock, required for precise timing');
    RESULT := FALSE;
    Exit;
  end;

  //Setup buffering
  DTAcq.WrapMode := OL_WRP_NONE; //set up for no buffering

  //get Maximum Frequency
  DTAcq.Frequency := 10000000;
  MaxFreq := round(DTAcq.Frequency);
  //get Min Frequency
  DTAcq.Frequency := 0;
  MinFreq := round(DTAcq.Frequency);

  //---------------------------  DAC OUT -------------------------------
//Select Board--set same as acq board
  DTDAC.Board  := DTAcq.BoardList[0];

  //Setup subsystem for DAC Operation
  DTDAC.SubSystem := OLSS_DA;//set type = DAC

  if DTDAC.GetSSCaps(OLSSC_SUP_SINGLEVALUE) <> 0
  then DTDAC.DataFlow := OL_DF_SINGLEVALUE //set up for singlevalue operation
  else begin
    ShowMessage(DTDAC.Board+ ' can not support singlevalue output');
    RESULT := FALSE;
    Exit;
  end;

  DTDAC.Config;	// configure subsystem

  //---------------------------  DIO OUT -------------------------------
//Select Board--set same as acq board
  DTDIO.Board  := DTAcq.BoardList[0];

  // Setup subsystem for D2A Operation
  DTDIO.SubSysType := OLSS_DOUT;//set type = Digital IO
  DTDIO.SubSysElement := 1;//second element (Lynx-8 is 0)

  if DTDIO.GetSSCaps(OLSSC_SUP_SINGLEVALUE) <> 0
  then DTDIO.DataFlow := OL_DF_SINGLEVALUE //set up for singlevalue operation
  else begin
    ShowMessage(DTDIO.Board+ ' can not support singlevalue output');
    RESULT := FALSE;
    Exit;
  end;
  DTDIO.Resolution := 16;
  DTDIO.Config;	// configure subsystem

//---------------------------  CLOCK -------------------------------
//Now setup clock board
  DTClock.Board  := DTAcq.BoardList[0];

  // Setup subsystem for Clock/Timer Operation
  try
    DTClock.SubSysType := OLSS_CT;//set type = Counter timer
  except
    ShowMessage(DTClock.Board+ ' does not support counter timer operations, required for precise timing operations.');
    ShowMessage('Using system ms timer instead.');
    //RESULT := FALSE;//system timer will be used
    Exit;
  end;

  DTClock.SubSysElement := 0;

  if DTClock.GetSSCaps(OLSSC_SUP_CTMODE_RATE) <> 0
  then DTClock.CTMode := OL_CTMODE_RATE //set up for counting each event
  else begin
    ShowMessage(DTClock.Board+ ' does not support rate generation, required for timing operations.');
    RESULT := FALSE;
    Exit;
  end;

  //It is already known that the internal clock is supported (above)
// Setup Clocks
  if DTClock.GetSSCaps(OLSSC_SUP_INTCLOCK) <> 0 then
  begin
    //Set clock to internal
    DTClock.ClockSource := OL_CLK_INTERNAL; //set up for internal clocksource
    //Set acquisition frequency
    DTClock.Frequency := 10000; //set the clock frequency in Hz
  end else
  begin
    RESULT := FALSE;
    Exit;
  end;

  //Continuous will work because the acquisition board has already passed this test
  DTClock.DataFlow := OL_DF_CONTINUOUS;	// set run continuously
  DTClock.PulseWidth:= 50;	        // 50% pulse width for square wave

  DTClock.Config;	// configure subsystem
  DTClock.Start();

  //---------------------------  CLOCK TRIGGER -------------------------------
//Now setup trigger system
//Select Board--set same as clock board
  DTCounter.Board  := DTClock.BoardList[0];

  // Setup subsystem for Clock/Timer Operation
  DTCounter.SubSysType := OLSS_CT;//set type = Counter timer
  DTCounter.SubSysElement := 1;//Second element

  if DTCounter.GetSSCaps(OLSSC_SUP_CTMODE_COUNT) <> 0
  then DTCounter.CTMode := OL_CTMODE_COUNT //set up for event counting
  else begin
    ShowMessage(DTCounter.Board+ ' does not support event counting');
    RESULT := FALSE;
    Exit;
  end;

  if DTCounter.GetSSCaps(OLSSC_SUP_EXTCLOCK) <> 0
  then begin
    DTCounter.ClockSource := OL_CLK_EXTERNAL; //set up for external clock
    DTCounter.ClockDivider := 2;
  end else begin
    ShowMessage(DTCounter.Board+ ' does not support counting of external clock');
    RESULT := FALSE;
    Exit;
  end;

  //Continuous will work because the acquisition board has already passed this test
  DTCounter.DataFlow := OL_DF_CONTINUOUS;	// set run continuously

  if DTCounter.GetSSCaps(OLSSC_SUP_GATE_NONE) <> 0
  then DTCounter.GateType := OL_GATE_NONE //trigger when clock output is high
  else begin
    ShowMessage(DTCounter.Board+ ' does not support software gating');
    RESULT := FALSE;
    Exit;
  end;

  DTCounter.Config;	// configure subsystem
  ClockOnBoard := TRUE;
end;
{$ENDIF}

{==============================================================================}
Function TSurfAcqForm.UnloadBuffers : boolean; //false if error
var i : integer;
    hbuf : HBUFTYPE;
begin  // !!!!!!!!!!!!!!!!!!! MODIFY FOR MULTIPLE BOARDS !!!!!!!!!!!!!!!!!!!!!!
  UnloadBuffers := TRUE;
  //deallocate buffers
  if not BuffersAllocated then exit;
  if(DTAcq.HDass <> NULL) then
  begin
    DTAcq.Flush;
    For i := 1 to NUMBUFFS do
    begin
      hBuf := HBUFTYPE(DTAcq.Queue);
      if hBuf <> NULL then
        if ErrorMsg(olDmFreeBuffer(hBuf)) then UnloadBuffers := FALSE;
    end;
  end;
  BuffersAllocated := FALSE;
end;

{==============================================================================}
(*procedure TSurfAcqForm.CreateAChanWindow(id,probenum,Channum,left,top,npts : integer);
var  s : string;
begin
  ChanForm[id].exists := TRUE;
  ChanForm[id].win := TChannelObj.CreateParented(WaveFormPanel.Handle);

  s := 'P'+inttostr(probenum)+ ' C' + IntToStr(channum) {+ ' ('+inttostr(Config.ProbeChannelList[CHANINDEX,id])+') '};
  if Config.Setup.Probe[probenum].ProbeType = CONTINUOUSTYPE then
    s := s + Config.Setup.Probe[probenum].Descrip;

  ChanForm[id].win.InitPlotWin({npts}npts,
                               {left}Left,
                                {top}Top,
                           {bmheight}50,
                            {intgain}Config.Setup.Probe[probenum].InternalGain,
                             {thresh}Config.Setup.Probe[probenum].Threshold,
                             {trigpt}Config.Setup.Probe[probenum].TrigPt,
                            {probeid}probenum,
                              {winid}id,
                          {probetype}Config.Setup.Probe[probenum].ProbeType,
                              {title}s,
                              {view} Config.Setup.Probe[probenum].view,
                    {acquisitionmode}TRUE);

  if SurfAcqForm.ClientWidth < Left + ChanForm[id].win.width
    then SurfAcqForm.ClientWidth := Left + ChanForm[id].win.width;
  if SurfAcqForm.ClientHeight      < FileInfoPanel.height + WaveFormPanel.Top + ChanForm[id].win.top + ChanForm[id].win.height
    then SurfAcqForm.ClientHeight := FileInfoPanel.height + WaveFormPanel.Top + ChanForm[id].win.top + ChanForm[id].win.height;
end;
*)
{==============================================================================}
procedure TSurfAcqForm.FreeChanWindows;
var i : integer;
begin
  For i := 0 to Config.NumProbes-1 do
    if ProbeWin[i].exists then
    begin
      ProbeWin[i].win.free;
      ProbeWin[i].exists := false;
    end;
end;

{==============================================================================}
function TSurfAcqForm.CreateAProbeWindow(probenum,left,top,npts : integer) : boolean;
var Electrode : TElectrode;
begin
  Result := TRUE;
  if not GetElectrode(Electrode,Config.Setup.Probe[probenum].ElectrodeName) then
  begin
    ShowMessage(Config.Setup.Probe[probenum].ElectrodeName+' is an invalid electrode name');
    Result := False;
    exit;
  end;

  if not ProbeWin[probenum].exists then
    ProbeWin[probenum].win := CProbeWin.CreateParented(WaveFormPanel.Handle);
  ProbeWin[probenum].win.InitPlotWin(Electrode,
                            {npts}npts,
                            {left}Left,
                             {top}Top,
                            {thresh}Config.Setup.Probe[probenum].Threshold,
                            {trigpt}Config.Setup.Probe[probenum].TrigPt,
                            {probeid}probenum,
                            {probetype}Config.Setup.Probe[probenum].ProbeType,
                            {title}Config.Setup.Probe[probenum].ElectrodeName,
                            {acquisitionmode}TRUE);

  ProbeWin[probenum].exists := TRUE;
  if SurfAcqForm.ClientWidth < ProbeWin[probenum].win.Width then SurfAcqForm.ClientWidth := ProbeWin[probenum].win.Width;
  if SurfAcqForm.ClientHeight  < FileInfoPanel.height + WaveFormPanel.Top + ProbeWin[probenum].win.Height + 10
    then SurfAcqForm.ClientHeight  := FileInfoPanel.height + WaveFormPanel.Top + ProbeWin[probenum].win.Height + 10;
  ProbeWin[probenum].win.Visible := TRUE;
end;

{==============================================================================}
function TSurfAcqForm.SetupProbeWins : boolean; //false if error
var probenum : integer;
    Electrode : TElectrode;
begin
  For probenum := 0 to Config.NumProbes-1 do
  begin
    GetElectrode(Electrode,Config.Setup.Probe[probenum].ElectrodeName);
    ProbeWin[probenum].win.InitPlotWin(Electrode,
                              {npts}Config.Setup.Probe[probenum].NPtsPerChan,
                              {left}ProbeWin[probenum].win.Left,
                               {top}ProbeWin[probenum].win.Top,
                            {thresh}Config.Setup.Probe[probenum].Threshold,
                            {trigpt}Config.Setup.Probe[probenum].TrigPt,
                           {probeid}probenum,
                         {probetype}Config.Setup.Probe[probenum].ProbeType,
                             {title}Config.Setup.Probe[probenum].ElectrodeName,
                   {acquisitionmode}TRUE);
    if SurfAcqForm.ClientWidth < ProbeWin[probenum].win.Width then SurfAcqForm.ClientWidth := ProbeWin[probenum].win.Width;
    if SurfAcqForm.ClientHeight  < FileInfoPanel.height + WaveFormPanel.Top + ProbeWin[probenum].win.Height + 10
      then SurfAcqForm.ClientHeight  := FileInfoPanel.height + WaveFormPanel.Top + ProbeWin[probenum].win.Height + 10;
    ProbeWin[probenum].win.Visible := TRUE;
  end;
  SetupProbeWins := TRUE;
end;

{==============================================================================}
procedure TSurfAcqForm.PAcquireClick(Sender: TObject);
begin
  StartStopAcquisition;
end;

{==============================================================================}
Procedure TSurfAcqForm.StartStopAcquisition;
var i,maxspkwvln,maxcrwvln : integer;
begin
  If Acquisition then //Stop acquisition
  begin
    Acquisition := FALSE;
    //AcqAnim.Animate := FALSE;
    STimer.Enabled := FALSE;

    if Recording then
    begin
      While (SpikeRingSaveReadIndex <> SpikeRingWriteIndex) do
        CheckForSpikesToSave;
      While (CRRingSaveReadIndex <> CRRingWriteIndex) do
        CheckForCRRecordsToSave;
      While (SVRingSaveReadIndex <> SVRingWriteIndex) do
        CheckForSVRecordsToSave;
    end;

    //AcquisitionBut.Caption := 'Start &Acquisition';
    PAcquire.BevelOuter := bvRaised;
    ConfigProbesBut.Enabled := TRUE;
    //if UserWinExists then PostMessage(UserWinHandle,WM_ACQSTOP,0,0);
    if ClockOnBoard then DTCounter.Stop;
    try
      DTAcq.Stop;
      //DTAcq.Reset;
    except
    end;
    Application.ProcessMessages;
    for i := 0 to MainMenu.Items.Count-1 do MainMenu.Items[i].Enabled := TRUE;

    maxspkwvln := 0;
    maxcrwvln := 0;
    for i := 0 to Config.Setup.NProbes-1 do
      case Config.Setup.Probe[i].ProbeType of
        CONTINUOUSTYPE :
          if maxcrwvln < Config.Setup.Probe[i].NPtsPerChan
            then maxcrwvln := Config.Setup.Probe[i].NPtsPerChan;
        SPIKEEPOCH :
          if maxspkwvln < Config.Setup.Probe[i].NChannels * Config.Setup.Probe[i].NPtsPerChan
            then maxspkwvln := Config.Setup.Probe[i].NChannels * Config.Setup.Probe[i].NPtsPerChan;
      end;

    For i := 0 to  CRRINGARRAYSIZE-1 do
      SetLength(CRRingArray[i].Waveform,maxcrwvln);
    For i := 0 to  SPIKERINGARRAYSIZE-1 do
      SetLength(SpikeRingArray[i].Waveform,maxspkwvln);

    //For i := 0 to  CRRINGARRAYSIZE-1 do
      //CRRingArray[i].Waveform := NIL;
    //For i := 0 to  SPIKERINGARRAYSIZE-1 do
      //SpikeRingArray[i].Waveform := NIL;

    SetLength(SingleWave,maxspkwvln);// := NIL;

  end else //Start it
  begin
    if Config.Setup.TotalChannels = 0 then
    begin
      ShowMessage('No channels setup');
      exit;
    end;
    Acquisition := TRUE;
    //AcquisitionBut.Caption := 'Stop &Acquisition';
    PAcquire.BevelOuter := bvLowered;
    for i := 0 to MainMenu.Items.Count-1 do MainMenu.Items[i].Enabled := FALSE;
    SpikeRingWriteIndex     := 0;
    SpikeRingSaveReadIndex  := 0;
    SpikeRingDisplayReadIndex := 0;
    CRRingWriteIndex        := 0;
    CRRingSaveReadIndex     := 0;
    CRRingDisplayReadIndex  := 0;
    SvRingWriteIndex       := 0;
    SvRingSaveReadIndex    := 0;

    RawRingBuffNum := 0;
    FirstBufferDone := FALSE;
    SecondBufferDone := FALSE;
    ShiftDown := FALSE;

    nrawspikes := 0;
    nsavedspikes := 0;
    ndisplayedspikes := 0;
    ndroppedspikes := 0;

    nrawcrbuffers := 0;
    ndisplayedcrbuffers := 0;
    nsavedcrbuffers := 0;

    nsaveddinbuffers := 0;
    nrawdinbuffers := 0;

    if Infowin.Visible then
    begin
      InfoWin.SpikesAcquired.Caption := IntToStr(nrawspikes);
      InfoWin.SpikesDisplayed.Caption := IntToStr(nDisplayedspikes);
      InfoWin.CRBuffersAcquired.Caption := IntToStr(nrawcrbuffers);
      InfoWin.CRBuffersDisplayed.Caption := IntToStr(ndisplayedcrbuffers);
      InfoWin.DinBuffersAcquired.Caption := IntToStr(nrawdinbuffers);
    end;

    TotalBuffersAcquired := 0;
    TimeScale := 10000/DTAcq.Frequency;
    {
    UserWinHandle := FindWindow('TUserWin','Surf Users Window');
    if UserWinHandle = 0
      then UserWinExists := FALSE
      else UserWinExists := TRUE;
    }
    //Lynx8AmpControlExists := TRUE;
    //Lynx8AmpHandle := FindWindow('TLynx8Form','Lynx-8 Control Window');
    //if Lynx8AmpHandle = 0 then Lynx8AmpControlExists := FALSE;
    //SettingLynx8 := FALSE;

    {InfoWin.WriteGuage.MinValue := 0;
    InfoWin.WriteGuage.MaxValue := SPIKERINGARRAYSIZE;
    InfoWin.WriteGuage.Progress := 0;
    InfoWin.SaveReadGuage.MinValue := 0;
    InfoWin.SaveReadGuage.MaxValue := SPIKERINGARRAYSIZE;
    InfoWin.SaveReadGuage.Progress := 0;
    InfoWin.DisplayReadGuage.MinValue := 0;
    InfoWin.DisplayReadGuage.MaxValue := SPIKERINGARRAYSIZE;
    InfoWin.DisplayReadGuage.Progress := 0;

    InfoWin.CRDisplayReadGuage.MinValue := 0;
    InfoWin.CRDisplayReadGuage.MaxValue := CRRINGARRAYSIZE;
    InfoWin.CRWriteGuage.Progress := 0;
    InfoWin.CRWriteGuage.MaxValue := CRRINGARRAYSIZE;

    InfoWin.CRSaveReadGuage.MinValue := 0;
    InfoWin.CRSaveReadGuage.MaxValue := CRRINGARRAYSIZE;
    InfoWin.CRSaveReadGuage.Progress := 0;
    }
    For i := 0 to SURF_MAX_PROBES-1 do
    begin
      LastSpikeTimeofProbe[i] := 0;
      LastSpikeTrigPtofProbe[i] := 0;
      CRPointCount[i] := 0;
      CRSkipCount[i] := 0;
    end;
    ConfigProbesBut.Enabled := FALSE;
    With TheTime do
    begin
      TenthMS := 0;//10th ms
      LastTenthMS := 0;//10th ms
      MS := 0;
      LastMS := 0;
      Sec := 0;
      LastSec := 0;
      Min := 0;
    end;
    TheTime.CurCount := 0;
    DTAcq.ClearError;
    DTAcq.Config;
    //if userwinexists then PostMessage(UserWinHandle,WM_ACQSTART,0,0);
    //AcqAnim.Animate := TRUE;
    DTAcq.Start;
    if ClockOnBoard then
    begin
      DTCounter.Start;
      TheTime.LastCount := DTCounter.CTReadEvents;
    end else TheTime.LastCount := 0;
    STimer.NowMs := 0;
    STimer.Enabled := TRUE;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.PRecordClick(Sender: TObject);
begin
  StartStopRecording;
end;

{==============================================================================}
Procedure TSurfAcqForm.StartStopRecording;
begin
  if Recording then  //Stop recording
  begin
    Recording := FALSE;
    PRecord.BevelOuter := bvRaised;
    //RecordingBut.Caption := 'Start &Recording';
    //PostMessage(UserWinHandle,WM_RECSTOP,0,0);
    (*
    For p := 0 to Config.NumProbes-1 do// .NumAnalogChannels-1 do
    begin
      if not Config.CglList[i].IsAMultiChan then
      begin
        winid := Config.CglList[i].WindowId;
        ChanForm[winid].win.MarkerV.Visible := TRUE;
        ChanForm[winid].win.MarkerH.Visible := TRUE;
      end;
    end;
    *)
  end else
  begin             //Start Recording
    If FileIsOpen then
    begin
      Recording := TRUE;
      PRecord.BevelOuter := bvLowered;
     // RecordingBut.Caption := 'Stop &Recording';
      //PostMessage(UserWinHandle,WM_RECSTART,0,0);
      WriteSURFRecords;
      (*
      For i := 0 to Config.NumAnalogChannels-1 do
      begin
        if not Config.CglList[i].IsAMultiChan then
        begin
          winid := Config.CglList[i].WindowId;
          ChanForm[winid].win.MarkerV.Visible := FALSE;
          ChanForm[winid].win.MarkerH.Visible := FALSE;
        end;
      end;
      *)
    end else
    begin
      Beep;
      ShowMessage('Error: no data file open');
      Recording := FALSE;
      PRecord.BevelOuter := bvRaised;
      //RecordingBut.Caption := 'Start Recording';
    end;
  end;
end;

{==============================================================================}
Procedure TSurfAcqForm.PutWindowLocationsInConfig(probe : integer);
begin
  config.WindowLoc[probe].left   := ProbeWin[probe].win.left;
  config.WindowLoc[probe].top    := ProbeWin[probe].win.top;
  config.WindowLoc[probe].width  := ProbeWin[probe].win.width;
  config.WindowLoc[probe].height := ProbeWin[probe].win.height;
end;

{==============================================================================}
{Procedure TSurfAcqForm.GetWindowLocationsFromConfig(probe : integer);
begin
  ProbeWin[probe].win.left   := config.WindowLoc[probe].left;
  ProbeWin[probe].win.top    := config.WindowLoc[probe].top;
  ProbeWin[probe].win.width  := config.WindowLoc[probe].width;
  ProbeWin[probe].win.height := config.WindowLoc[probe].height;
end; }

{==============================================================================}
Procedure TSurfAcqForm.WriteSurfRecords;
var i,c : integer;
    slr : SURF_LAYOUT_REC;
begin
  For i := 0 to Config.NumProbes-1 do
    PutWindowLocationsInConfig(i);

  With slr do
  begin
    ufftype    := SURF_PL_REC_UFFTYPE; // Record type  chr(234)
    time_stamp := TheTime.TenthMS;  // Time stamp
    surf_major := 1; // SURF major version number
    surf_minor := 0; // SURF minor version number
    FillChar(pad,sizeof(pad),0);
  end;

  For i := 0 to Config.NumProbes-1 do
  begin
    With slr do //probe specific settings
    begin
      probe          := i;                   //Probe number
      ProbeSubType   := Config.Setup.probe[i].ProbeType; {=S,C for spike or continuous }
      nchans         := Config.Setup.probe[i].NChannels; //number of channels in this spike waveform
      pts_per_chan   := Config.Setup.probe[i].NPtsPerChan; //number of pts per waveform
      trigpt         := Config.Setup.probe[i].TrigPt; // pts before trigger
      lockout        := Config.Setup.probe[i].Lockout; // Lockout in pts
      intgain        := Config.Setup.probe[i].InternalGain; // A/D board internal gain
      threshold      := Config.Setup.probe[i].Threshold; // A/D board threshold for trigger
      skippts        := Config.Setup.probe[i].SkipPts;
      sampfreqperchan:= Config.Setup.probe[i].SampFreq;  // A/D sampling frequency
      probe_descrip  := Config.Setup.probe[i].Descrip;      //description of the electrode type
      electrode_name := Config.Setup.probe[i].ElectrodeName;//predefined name of the electrode

      For c := 0 to SURF_MAX_CHANNELS-1 do
        if c < Config.NumAnalogChannels
          then chanlist[c] := Config.Setup.probe[i].ChanStart+c
          else chanlist[c] := -1;
      ExtGainForm.Probe := i;
      ExtGainForm.NumChannels := Config.NumAnalogChannels;
      ExtGainForm.ShowModal;
      For c := 0 to SURF_MAX_CHANNELS-1 do
        if c < Config.NumAnalogChannels
          then extgain[c] := Word(StrToInt(ExtGainForm.ExtGainArray[i].Text))
          else extgain[c] := 0;//unused

      ProbewinLayout.left := config.WindowLoc[i].left;//CglList[c+ci].winloc.left;
      ProbewinLayout.top := config.WindowLoc[i].top;//config.CglList[c+ci].winloc.top;
      ProbewinLayout.width := config.WindowLoc[i].width;//CglList[c+ci].winloc.left;
      ProbewinLayout.height := config.WindowLoc[i].height;//config.CglList[c+ci].winloc.top;
    end;

    if not SaveSURF.PutSurfRecord(slr) then
    begin
      Beep;
      ShowMessage('Error writing to Surf file');
      Recording := FALSE;
      PRecord.BevelOuter := bvRaised;
      //RecordingBut.Caption := 'Start Recording';
    end;
  end;
end;

{==============================================================================}
{==============================================================================}
procedure TSurfAcqForm.DTAcqOverrunError(Sender: TObject);
begin
 {This message is sent when the hardware of an input subsystem runs out of
  buffer space. An overrun error indicates that the input data was not
  transferred before the next sample was received. This error occurs when data
  transfer from the hardware to the driver cannot keep up with the input clock
  rate. To avoid this error, reduce the sampling rate or increase the size of
  the buffers.}
  {ztimer.enabled := FALSE;
  FileInfoPanel.SimpleText := 'DT Acqisition: Overrun error.';}
  Showmessage('DT Acqisition: Overrun error.  Reduce sampling rate or increase buffer size.');
  StartStopAcquisition;
end;

{==============================================================================}
procedure TSurfAcqForm.DTAcqUnderrunError(Sender: TObject);
begin
  {ztimer.enabled := FALSE;
  FileInfoPanel.SimpleText := 'Underrun error';}
  ShowMessage('Underrun error');
  StartStopAcquisition;
end;

{==============================================================================}
procedure TSurfAcqForm.DTAcqTriggerError(Sender: TObject);
begin
 {This message is sent when a trigger error occurs. A trigger error occurs when
  unexpected software or external triggers are received during data transfer. }
  {ztimer.enabled := FALSE;
  FileInfoPanel.SimpleText := 'Unexpected trigger event has occurred';
  }Showmessage('DT Acqisition: Unexpected trigger event has occurred.');
  StartStopAcquisition;
end;

{==============================================================================}
procedure TSurfAcqForm.DTAcqQueueStopped(Sender: TObject);
begin
 {This event occurs when the operation is stopped as a result of the Stop method.
  This event is always preceded by a BufferDone event.}
  STimer.enabled := FALSE;
  FileInfoPanel.SimpleText := 'Queue stopped';
  //ShowMessage('Queue stopped');
  //StartStopAcquisition;
end;

{==============================================================================}
procedure TSurfAcqForm.DTAcqBufferDone(Sender: TObject);
var s,i,MaxSamples : LNG;
  RawRingPtr : Pointer;
  CheckRawRingArrayIndex : LNG;
  hbuf : HBUFTYPE;
  WaveformStart,ProbeId : integer;
  cglindex : integer;
  RawRingAddressOffset : LNG;
  ptdiff,npts,totchans : Integer;
  BeginTime : LNG;
  wvalue : WORD;
  Threshold : array[0..SURF_MAX_PROBES-1] of integer;
  TrigTime : integer;

Procedure GrabWave;  //Grab the spike from the buffer
var w,w2,p,xoffset,lockoutpts : LNG;//ULNG;
    trigpt,pchans : integer;
begin
  //grab the spike and place in the spike ring buffer
  xoffset    := totchans * LNG(Config.Setup.Probe[probeid].InternalGain);
  lockoutpts := totchans * LNG(Config.Setup.Probe[probeid].Lockout);
  trigpt     := s - (Config.CglList[cglindex].ChanOffset - Config.Setup.Probe[probeid].ChanStart);
  WaveformStart := (RawRingArraySize+CheckRawRingArrayIndex+trigpt-xoffset) mod RawRingArraySize;


  With SpikeRingArray[SpikeRingWriteIndex] do
  begin
    //get the time of the spike to the nearest 1/10th ms
    Time := BeginTime{begin of last buffer}+round(trigpt*TimeScale);//This should be accurate to 1/10th ms
    ProbeNum := probeid;
    Cluster := 0;
    Npts := Config.Setup.Probe[probeid].NPtsPerChan;
    pchans := Config.Setup.probe[probeid].NChannels;
    ptdiff := trigpt - LastSpikeTrigPtofProbe[probeid];
    if ptdiff < 0 then ptdiff := ptdiff + RawRingArraySize;
    if ptdiff > lockoutpts then
    begin
      //Transfer the waveform to the spike record
      if Length(Waveform) <> Npts*pchans
        then SetLength(Waveform,Npts*pchans);
      For p := 0 to Npts-1 do
      begin
        w2 := p * Config.Setup.TotalChannels{Config.NumChans};
        For w := 0 to pchans-1 do
          Waveform[p*pchans+w] := RawRingArray[(WaveformStart+w2+w) mod RawRingArraySize];
      end;

      LastSpikeTrigPtofProbe[ProbeId] := trigpt;//Time;
      inc(nrawspikes);

      if Infowin.Visible then
        InfoWin.SpikesAcquired.Caption := IntToStr(nrawspikes);

      //Increment the spike ring write buffer pointer
      SpikeRingWriteIndex := (SpikeRingWriteIndex + 1) mod SPIKERINGARRAYSIZE;
      //InfoWin.WriteGuage.Progress := SpikeRingWriteIndex;
      LastSpikeTimeofProbe[ProbeId] := Time;
      (*if userwinexists then
      begin
        //copy polytrode record to a TSpike record
        //SendSpikeToSurfBridge(SpikeRingArray[SpikeRingWriteIndex]);
      end; *)
      //if userwinexists then PostMessage(UserWinHandle,WM_SPKAVAIL,0,0);
    end;
    s := trigpt + pchans - 1;
  end;
end;

begin
 {This event occurs whenever a buffer transfer operation completes. An input
  subsystem generates this message when a buffer is filled. An output subsystem
  generates this message when a buffer is emptied.}
                                                 //10000/(pts/s) * pts/numchans
  BeginTime := round((TotalBuffersAcquired-1) * (TimeScale * BuffSize * Config.Setup.TotalChannels{Config.NumChans}{ + 0.02}){constant loss of 2us});
  // graph data & then recycle buffer
  hbuf := HBUFTYPE(DTAcq.Queue);  //retreiving buffer
  if hBuf = NULL then exit;

  (*If Acquisition and Recording then
  begin
    With SVRecord do
    begin
      ufftype := SURF_SV_REC_UFFTYPE;
      subtype := 'T';//for Time
      time_stamp := BeginTime;
      sval := TotalBuffersAcquired;
    end;
    SaveSURF.PutSingleValueRecord(SVRecord);
  end;  *)

  //Get the number of valid samples from the buffer
  ErrorMsg(olDmGetValidSamples(hBuf,ULNG(MaxSamples)));

  totchans := Config.Setup.TotalChannels;

  if MaxSamples = BuffSize*totchans then
  begin
    //compute start location in the ring buffer to which we are going to write
    RawRingAddressOffset := RawRingBuffNum*MaxSamples;
    RawRingPtr := @(RawRingArray[RawRingAddressOffset]);
    //Copy the acquired buffer into the raw ring buffer
    olDmCopyFromBuffer(hbuf,RawRingPtr,MaxSamples);
    //Check for a threshold crossing in previous buffer (if exists),
    //and overflow into current buffer if needed

    if SecondBufferDone then
    begin
      CheckRawRingArrayIndex := ((NUMBUFFS+RawRingBuffNum-1) mod NUMBUFFS) * MaxSamples;
      For probeid := 0 to Config.NumProbes-1 do
        Threshold[probeid] := Config.Setup.Probe[probeid].Threshold+2048;

      //Search entire buffer and save any waveform that passes a positive or negative threshold
      s := 0;
      While s < MaxSamples do
      begin
        cglindex := s mod totchans;//Config.NumChans;
        probeid := Config.CglList[cglindex].ProbeId;
        wvalue := RawRingArray[(CheckRawRingArrayIndex+s) mod rawringarraysize];
        TrigTime := BeginTime+round(s*TimeScale);//This should be accurate to 1/10th ms

        if (probeid = DINPROBE) then
        begin
          //move(RawRingArray[(CheckRawRingArrayIndex+s) mod rawringarraysize],wvalue,2);//an equivalence move (short to word)
          if wvalue <> LastDinVal then
            if (TrigTime > SvRingArray[SvRingWriteIndex-1].Time+10) then  //tjb addition to fix DIN glitch
            begin
              With SvRingArray[SvRingWriteIndex] do
                begin
                  Time := TrigTime;
                  SubType := SURF_DIGITAL;
                  SVal := wvalue;
                end;
            //if userwinexists then PostMessage(UserWinHandle,WM
            //_DINAVAIL,0,0);
            {if userwinexists then
            begin
              //copy din to a TSVal record and send it
              tmpSVal.time_stamp := SvRingArray[SvRingWriteIndex].Time;
              tmpSVal.subtype := SURF_DIGITAL;
              tmpSVal.EventNum := -1;
              tmpSVal.sval := wvalue;
              SendSVToSurfBridge(tmpSVal);
            end;}
            SvRingWriteIndex := (SvRingWriteIndex + 1) mod SVRINGARRAYSIZE; // advance the write index
            LastDinVal := wvalue;
            inc(nrawdinbuffers);
            if Infowin.Visible then
              InfoWin.DinBuffersAcquired.Caption := IntToStr(nrawdinbuffers);
            end else
            begin
              LastDinVal :=wvalue;
              SvRingArray[SvRingWriteIndex-1].SVal :=wvalue;  //tjb addition to fix DIN glitch
            end;
        //Check if this point is a cont rec channel
          end else
        if Config.Setup.Probe[probeid].probetype = CONTINUOUSTYPE then
        begin
//FileInfoPanel.Panels[1].Text := inttostr(TheTime.Sec)+','+inttostr(CRPointCount[probeid]);
          if (CRSkipCount[probeid]+1) mod Config.Setup.Probe[probeid].SkipPts = 0 then
          begin
            //is this a single pt cr channel?  If so then just put the single value on the ring buffer
            if Config.Setup.Probe[probeid].NPtsPerChan = 1 then
            begin
              With SvRingArray[SvRingWriteIndex] do
              begin
                Time := TrigTime;
                SubType := SURF_ANALOG;
                SVal := wvalue;
              end;
              {if userwinexists then
              begin
                //copy value to a TSVal record and send it
                tmpSVal.time_stamp := SvRingArray[SvRingWriteIndex].Time;
                tmpSVal.subtype := SURF_ANALOG;
                tmpSVal.EventNum := -1;
                tmpSVal.sval := wvalue;//SvRingArray[SvRingWriteIndex].SVal;
                SendSVToSurfBridge(tmpSVal);
              end;}
              SvRingWriteIndex := (SvRingWriteIndex + 1) mod SVRINGARRAYSIZE; // advance the write index
            end else
            begin
              //see if this is the first point in the waveform buffer of this probe
              if CRPointCount[probeid] = 0 then
              begin
                CRTempRecordArray[probeid].Time := TrigTime;
                CRTempRecordArray[probeid].ProbeNum := probeid;
              end;
              //assign the param to the waveform
              CRTempRecordArray[probeid].Waveform[CRPointCount[probeid]] := wvalue;
              inc(CRPointCount[probeid]);
              //see if the waveform is finished
              npts := Config.Setup.Probe[probeid].NPtsPerChan;
              if CRPointCount[probeid] > npts-1 then
              begin //write the buffer CR to the ring
                With CRRingArray[CRRingWriteIndex] do
                begin
                  Time := CRTempRecordArray[probeid].Time;
                  ProbeNum := CRTempRecordArray[probeid].ProbeNum;
                  if Length(Waveform) <> npts then SetLength(Waveform,npts);
                  For i := 0 to npts-1 do
                    Waveform[i] := CRTempRecordArray[probeid].Waveform[i];
                end;
                if userwinexists then
                begin
                  //create a cr rec and send it to surfacq
                  //  Procedure SendCrToSurfBridge(Cr : TCr);
                end;
                CRRingWriteIndex := (CRRingWriteIndex + 1) mod CRRINGARRAYSIZE; // advance the write index
                //if userwinexists then PostMessage(UserWinHandle,WM_CRAVAIL,0,0);
                //InfoWin.CRWriteGuage.Progress := CRRingWriteIndex;
                CRPointCount[probeid] := 0; //reset the point counter for this CR probe
                inc(nrawcrbuffers);
                if Infowin.Visible then
                  InfoWin.CRBuffersAcquired.Caption := IntToStr(nrawcrbuffers);
              end;
            end;
          end;
          inc(CRSkipCount[probeid]);
        end else
        begin
          if (Threshold[probeid] > 2058) then   //positive trigger
          begin
            if wvalue > Threshold[probeid] then GrabWave;
          end else
          if (Threshold[probeid] < 2038) then  //negative trigger
          begin
            if wvalue < Threshold[probeid] then GrabWave;
          end;
        end;
(*
      //old method
      //threshold was set like:
      //For i := 0 to Config.NumAnalogChannels-1 do
        //Threshold[i] := ChanForm[i].win.SlideBar.Max-ChanForm[i].win.SlideBar.Position;

      //if abs(Threshold[cglindex]-2048) > 10 then
        begin
          param := wvalue;
          if Threshold[cglindex]-2048 > 0
            then begin if Param > Threshold[cglindex] then GrabWave; end
            else if Param < Threshold[cglindex] then GrabWave;
        end;
*)
        inc(s);
      end;
    end;
  end;
//FileInfoPanel.Panels[1].Text := inttostr(dbgval);
  //increment the raw ring buffer index
  RawRingBuffNum := (RawRingBuffNum + 1) mod NUMBUFFS;

  if FirstBufferDone
    then SecondBufferDone := TRUE
    else FirstBufferDone := TRUE;
  DTAcq.Queue := {U}LNG(hBuf);  //recycle buffer
  inc(TotalBuffersAcquired);
  //Application.ProcessMessages;   DO NOT CALL THIS IN HERE
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForSpikesToSave;
var PTRecord : SURF_PT_REC;
  npts : integer;
begin
  //This routine checks the spike buffer and saves them to disk.
  //Rough tests demonstrated a sustained rate of 32 channels at 600Hz
  If Acquisition and Recording then
  begin
    If (SpikeRingSaveReadIndex <> SpikeRingWriteIndex) then
    begin
      With PTRecord do
      begin
        ufftype := SURF_PT_REC_UFFTYPE;
        subtype := SPIKEEPOCH;
        time_stamp := SpikeRingArray[SpikeRingSaveReadIndex].Time;
        probe := SpikeRingArray[SpikeRingSaveReadIndex].ProbeNum;
        cluster := 0;
        npts := Config.Setup.Probe[probe].NChannels * Config.Setup.Probe[probe].NPtsPerChan;
        SetLength(adc_waveform,npts);
        Move(SpikeRingArray[SpikeRingSaveReadIndex].Waveform[0],adc_waveform[0],npts*2{sizeof shrt});
      end;
      SaveSURF.PutPolytrodeRecord(PTRecord);
      inc(nsavedspikes);
      SpikeRingSaveReadIndex := (SpikeRingSaveReadIndex + 1) mod SPIKERINGARRAYSIZE;
      if Infowin.Visible then
        InfoWin.SpikesSaved.Caption := IntToStr(nsavedspikes);
      //InfoWin.SaveReadGuage.Progress := SpikeRingSaveReadIndex;
    end;
  PTRecord.adc_waveform:= nil;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForCRRecordsToSave;
var PTRecord : SURF_PT_REC;
  w : integer;
begin
  //This routine checks the spike buffer and saves them to disk.
  //Rough tests demonstrated a sustained rate of 32 channels at 600Hz
  If Acquisition and Recording then
  begin
    If (CRRingSaveReadIndex <> CRRingWriteIndex) then
    begin
      With PTRecord do
      begin
        ufftype := SURF_PT_REC_UFFTYPE;
        subtype := 'C';
        time_stamp := CRRingArray[CRRingSaveReadIndex].Time;
        probe := CRRingArray[CRRingSaveReadIndex].ProbeNum;
        cluster := 0;
        adc_waveform := nil;
        SetLength(adc_waveform,Config.Setup.Probe[probe].NPtsPerChan);
        For w := 0 to Config.Setup.Probe[probe].NPtsPerChan-1 do
          adc_waveform[w] := CRRingArray[CRRingSaveReadIndex].Waveform[w];
      end;
      SaveSURF.PutPolytrodeRecord(PTRecord);
      inc(nsavedcrbuffers);
      CRRingSaveReadIndex := (CRRingSaveReadIndex + 1) mod CRRINGARRAYSIZE;
      if Infowin.Visible then
        InfoWin.CRBuffersSaved.Caption := IntToStr(nsavedcrbuffers);
      //InfoWin.CRSaveReadGuage.Progress := CRRingSaveReadIndex;
    end;
  PTRecord.adc_waveform:= nil;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForSvRecordsToSave;
var SVRecord : SURF_SV_REC;
begin
  //This routine checks the din buffer and saves dinvalues to disk.
  If Acquisition and Recording then
  begin
    If (SvRingSaveReadIndex <> SvRingWriteIndex) then
    begin
      With SVRecord do
      begin
        ufftype := SURF_SV_REC_UFFTYPE;
        subtype := SvRingArray[SvRingSaveReadIndex].subtype;
        time_stamp := SvRingArray[SvRingSaveReadIndex].Time;
        sval := SvRingArray[SvRingSaveReadIndex].SVal;
      end;
      SaveSURF.PutSingleValueRecord(SVRecord);
      if SVRecord.subtype = SURF_DIGITAL then inc(nsaveddinbuffers);
      SvRingSaveReadIndex := (SvRingSaveReadIndex + 1) mod SVRINGARRAYSIZE;
      if SVRecord.subtype = SURF_DIGITAL then
        if Infowin.Visible then
          InfoWin.DinBuffersSaved.Caption := IntToStr(nsaveddinbuffers);
    end;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForSpikesToDisplay;
var diff,drop : LNG;
begin
  //This routine checks the spike buffer and can handle spikes up to 2000Hz
  if Config.Setup.Probe[SpikeRingArray[SpikeRingDisplayReadIndex].probenum].View = FALSE then
    SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + 1) mod SPIKERINGARRAYSIZE
  else
  begin
    diff := SpikeRingWriteIndex - SpikeRingDisplayReadIndex;
    If diff <> 0 then
    begin
      If diff > 50 then {skip some spikes}
      begin
        drop := diff div 5;//drop 20 percent of the spikes
        SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + drop) mod SPIKERINGARRAYSIZE;
        inc(ndroppedspikes,drop);
        exit;
      end;
      If diff < 0 then {writer ptr has wrapped around}
        if diff + SPIKERINGARRAYSIZE > 50 then
        begin
          drop := (SPIKERINGARRAYSIZE + diff) div 5;//drop 20 percent of the spikes
          SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + drop) mod SPIKERINGARRAYSIZE;
          inc(ndroppedspikes,drop);
          exit;
        end;

      With SpikeRingArray[SpikeRingDisplayReadIndex] do
      begin
//        ProbeWin[ProbeNum].win.PlotWaveForm(Waveform,Cluster);
        ProbeWin[ProbeNum].win.update;
      end;
      //Application.ProcessMessages;//don't tie up the timer
      inc(ndisplayedspikes);

      SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + 1) mod SPIKERINGARRAYSIZE;
      if Infowin.Visible then
        InfoWin.SpikesDisplayed.Caption := IntToStr(ndisplayedspikes);
      //InfoWin.DisplayReadGuage.Progress := SpikeRingDisplayReadIndex;
    end;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForCRRecordsToDisplay;
var diff,drop : LNG;
begin
  if Config.Setup.Probe[CRRingArray[CRRingDisplayReadIndex].probenum].View = FALSE then
    CRRingDisplayReadIndex := (CRRingDisplayReadIndex + 1) mod CRRINGARRAYSIZE
  else
  begin
    diff := CRRingWriteIndex - CRRingDisplayReadIndex;
    If diff <> 0 then
    begin
      If diff > 10 then {skip some cr records}
      begin
        drop := diff div 50;//drop 50 percent of the records
        CRRingDisplayReadIndex := (CRRingDisplayReadIndex + drop) mod CRRINGARRAYSIZE;
        exit;
      end;
      If diff < 0 then {writer ptr has wrapped around}
        if CRRINGARRAYSIZE + diff > 100{potentiually 100ms behind} then
        begin
          drop := (CRRINGARRAYSIZE + diff) div 50;//drop 50 percent of the spikes
          CRRingDisplayReadIndex := (CRRingDisplayReadIndex + drop) mod CRRINGARRAYSIZE;
          exit;
        end;
      With CRRingArray[CRRingDisplayReadIndex] do
      begin
//        ProbeWin[ProbeNum].win.PlotWaveForm(Waveform,4{yellow});
        ProbeWin[ProbeNum].win.update;
      end;
      //Application.ProcessMessages;//don't tie up the timer
      inc(ndisplayedcrbuffers);
      CRRingDisplayReadIndex := (CRRingDisplayReadIndex + 1) mod CRRINGARRAYSIZE;
      if Infowin.Visible then
        InfoWin.CRBuffersDisplayed.Caption := IntToStr(ndisplayedcrbuffers);
      //InfoWin.CRDisplayReadGuage.Progress := CRRingDisplayReadIndex;
    end;
  end;
end;

{==============================================================================}
function TSurfAcqForm.ConfigBoard : boolean; //false if error
var i : integer;
 hbuf : HBUFTYPE;
begin
  ConfigBoard := TRUE;

  //Set channel list
  DTAcq.ListSize := Config.Setup.TotalChannels;
  For i := 0 to Config.Setup.TotalChannels-1 do
  begin
    DTAcq.ChannelList[i] := Config.CglList[i].ChanOffset;// ProbeChannelList[CHANINDEX,i];
    if (i=Config.Setup.TotalChannels-1) and Config.DinChecked
      then DTAcq.GainList[i] := 1
      else DTAcq.GainList[i] := Config.Setup.Probe[Config.CglList[i].ProbeId].InternalGain;// .GainTrigLockList[GAININDEX,Config.ProbeChannelList[PROBEINDEX,i]];
  end;

  //Set acquisition frequency
  DTAcq.Frequency := SampFreq*Config.Setup.TotalChannels; //set the clock frequency in Hz

  //allocate buffers and put on DTAcq's Ready Queue
  For i := 1 to NUMBUFFS do
  begin
    if ErrorMsg(olDmAllocBuffer(0, BuffSize * DTAcq.ListSize, @hbuf)) then
    begin
      ConfigBoard := FALSE;
      BuffersAllocated := FALSE;
      Exit;
    end;
    DTAcq.Queue := {U}LNG(hbuf);
  end;

  //if dio required then set dio for read mode else set it for write mode
  if config.DinChecked then
  begin//read mode
    //Select Board--set same as acq board
    DTDIO.Board  := DTAcq.BoardList[0];
    // Setup subsystem for D2A Operation
    DTDIO.SubSysType := OLSS_DIN;//set type = Digital IO
    DTDIO.SubSysElement := 1;//second element (Lynx-8 is 0)

    if DTDIO.GetSSCaps(OLSSC_SUP_SINGLEVALUE) <> 0
    then DTDIO.DataFlow := OL_DF_SINGLEVALUE //set up for singlevalue operation
    else begin
      ShowMessage(DTDIO.Board+ ' can not support single value output');
      RESULT := FALSE;
      Exit;
    end;
    DTDIO.Resolution := 16;
    DTDIO.Config;	// configure subsystem
  end else
  begin
    //Select Board--set same as acq board
    DTDIO.Board  := DTAcq.BoardList[0];
    // Setup subsystem for D2A Operation
    DTDIO.SubSysType := OLSS_DOUT;//set type = Digital IO
    DTDIO.SubSysElement := 1;//second element (Lynx-8 is 0)

    if DTDIO.GetSSCaps(OLSSC_SUP_SINGLEVALUE) <> 0
    then DTDIO.DataFlow := OL_DF_SINGLEVALUE //set up for singlevalue operation
    else begin
      ShowMessage(DTDIO.Board+ ' can not support singlevalue output');
      RESULT := FALSE;
      Exit;
    end;
    DTDIO.Resolution := 16;
    DTDIO.Config;	// configure subsystem
  end;
(*  DIO.SubSysType := OLSS_DIN;//set type = Digital Input, for now
  DIO.SubSysElement := 0;
  DIO.DataFlow := OL_DF_SINGLEVALUE; //set up for single value operation
  DIO.Resolution := 16;
  DIO.Config;
*)

  BuffersAllocated := TRUE;
end;

{==============================================================================}
procedure TSurfAcqForm.ConfigProbesButClick(Sender: TObject);
begin
  if Recording then PRecord.BevelOuter := bvRaised;// RecordingBut.Caption := 'Start Recording';
  SetupProbes;
  {UserWinExists := TRUE;
  UserWinHandle := FindWindow('TUserWin','Surf Users Window');
  if UserWinHandle = 0 then UserWinExists := FALSE;
  }//Lynx8AmpControlExists := TRUE;
  //Lynx8AmpHandle := FindWindow('TLynx8Form','Lynx-8 Control Window');
  //if Lynx8AmpHandle = 0 then Lynx8AmpControlExists := FALSE;
  //if UserWinExists then PostMessage(UserWinHandle,WM_USERWINSHOW,0,0);
  //if Lynx8AmpControlExists then PostMessage(Lynx8AmpHandle,WM_LYNX8SHOW,0,0);
end;

{==============================================================================}
procedure TSurfAcqForm.SetUpProbes;
var i,j,l,t,cglindex,nc : integer;
    LastSetup : TProbeSetup;
begin
  For i := 0 to Config.NumProbes-1 do
    PutWindowLocationsInConfig(i);
  Move(Config.Setup,LastSetup,sizeof(TProbeSetup));//backup old setup

  ProbeSetupWin.DinCheckBox.Checked := Config.DinChecked;

  UnloadBuffers;

  {if Config.NumChans > 0 then }
  FreeChanWindows;

  //copy config.setup to probewin.setup???
  Move(Config.Setup,ProbeSetupWin.Setup,sizeof(TProbeSetup));

  //show the probewin
  ProbeSetupWin.ShowModal;

  //see if valid
  SampFreq := ProbeSetupWin.SampFreqPerChan.Value;
  If (ProbeSetupWin.Setup.NProbes=0) or (ProbeSetupWin.Setup.TotalChannels=0) then ProbeSetupWin.ok := FALSE;

  //if not ProbeSetupWin.ok then Move(LastSetup,ProbeSetupWin.Setup,sizeof(ProbeSetupRec));

  if ProbeSetupWin.ok then
  begin
    //copy the probewin setup to the config.setup
    Move(ProbeSetupWin.Setup,Config.Setup,sizeof(TProbeSetup));
    Config.DinChecked := ProbeSetupWin.DinCheckBox.Checked;
    Config.NumProbes := ProbeSetupWin.Setup.NProbes;
    Config.NumAnalogChannels := ProbeSetupWin.Setup.TotalChannels;
    if Config.DinChecked
      then dec(Config.NumAnalogChannels);

    Config.empty := false;
    ConfigProbesBut.Caption := '&Modify';

    rawringarray := NIL;
    nc := Config.NumAnalogChannels;
    if nc < 8 then nc := 8;
    BuffSize := 32767 div nc;
    NumBuffs := 3 + round(sqrt((SampFreq*Config.NumAnalogChannels) / buffsize));
    RawRingArraySize := {U}LNG(BuffSize*NumBuffs*Config.NumAnalogChannels);

    SetLength(rawringarray,RawRingArraySize);
    cglindex := 0;
    For i := 0 to Config.NumProbes-1 do
      For j := 0 to Config.Setup.Probe[i].NChannels-1 do
      begin
        Config.CglList[cglindex].ProbeId := i;
        Config.CglList[cglindex].ChanOffset := Config.Setup.Probe[i].ChanStart + j;
        inc(cglindex);
      end;

    if Config.DinChecked then
    begin
      Config.CglList[cglindex{the last entry}].ProbeId := DINPROBE{which is 32};
      Config.CglList[cglindex].ChanOffset := DINPROBE{which is 32};
    end;

    //Setup the acquisition and plot objects
{$IFDEF ADBOARD}
    if not ConfigBoard then
    begin
      ShowMessage('Error: Board not configured properly');
      exit;
    end;
{$ENDIF}

    //Draw the windows to the screen
    l := 0;
    t := 0;
    For i := 0 to Config.NumProbes-1 do
    begin
      CreateAProbeWindow(i,l,t,Config.Setup.Probe[i].NPtsPerChan);
      ProbeWin[i].win.top := 0;
      inc(l,ProbeWin[i].win.Width+1);
      if SurfAcqForm.ClientWidth < l
        then SurfAcqForm.ClientWidth := l;
      if SurfAcqForm.ClientHeight  < WaveFormPanel.Top + ProbeWin[i].win.Height + 10
        then SurfAcqForm.ClientHeight  := WaveFormPanel.Top + ProbeWin[i].win.Height + 10;
    end;
    FileInfoPanel.BringToFront;

    if not SetupProbeWins
    then begin
      ShowMessage('Error: Plot windows not configured properly');
      exit;
    end;

    //set the locations of the windows to what they were before if no major changes to config

    For i := 0 to Config.NumProbes-1 do
        PutWindowLocationsInConfig(i);
    {  if (Config.Setup.Probe[i].ChanStart = LastSetup.Probe[i].ChanStart)
      and (Config.Setup.Probe[i].NPtsPerChan = LastSetup.Probe[i].NPtsPerChan)
        then GetWindowLocationsFromConfig(i)
        else PutWindowLocationsInConfig(i);
    }
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.UpdateTimer;
var diff : LNG;
begin
  With TheTime do
  begin
    if ClockOnBoard
      then CurCount := DTCounter.CTReadEvents
      else CurCount := STimer.NowMs * 10;
    diff := CurCount-LastCount;
    if diff <> 0 then
    begin
      if diff < 0 then diff := 65536+diff;
      inc(TenthMS,diff);
    end;
    LastCount := CurCount;
  end;
  TheTime.Min := TheTime.TenthMS div 600000;
  TheTime.Sec := (TheTime.TenthMS - TheTime.Min * 600000) div 10000;
  TheTime.MS  := TheTime.TenthMS div 10;
end;

{==============================================================================}
procedure TSurfAcqForm.stimerTimer(Sender: TObject);
begin
  //this is a 1 ms timer
  UpdateTimer;
  CheckForSpikesToSave;
  CheckForSvRecordsToSave;
  //if TheTime.MS mod 10 = 0 {every 10 ms} then
    CheckForSpikesToDisplay;
  if TheTime.MS mod 100 = 0 {every 100 ms} then
  begin
    CheckForCRRecordsToSave;
    CheckForCRRecordsToDisplay;
  end;
  if TheTime.Sec <> TheTime.LastSec then
  begin
    if Infowin.Visible then
    begin
      InfoWin.TimeMin.Caption := IntToStr(TheTime.Min)+':';
      InfoWin.TimeSec.Caption := IntToStr(TheTime.Sec);
    end;
    TheTime.LastSec := TheTime.Sec;
  end;
  TheTime.LastTenthMS := TheTime.TenthMS;
  TheTime.LastMS := TheTime.MS;
  Application.ProcessMessages;
end;

{==============================================================================}
(*Procedure TChannelObj.ThreshChange(pid,winid : integer; ShiftDown,CtrlDown : boolean);
var id,StartWinId,StopWinId,SelectWinId,y : integer;
begin
  //cid is the cglindex
  //because this is called by TChannelObj, all channels of the probe will be chanform objects, not mucltichan
  SelectWinId := winid;//SurfAcqForm.Config.CglList[SurfAcqForm.Config.Setup.Probe[pid].CglOffset+cid].WindowId;

  if ShiftDown then  //Set threshold across all chans of this probe
  begin
    StartWinId := SurfAcqForm.Config.CglList[SurfAcqForm.Config.Setup.Probe[pid].CglOffset].WindowId;// .ChannelWinId[pid,0];
    StopWinId :=  StartWinId+SurfAcqForm.Config.Setup.Probe[pid].NChannels-1;// CglList[SurfAcqForm.Config.Setup.Probe[pid].CglOffset]. .NChans-1;
  end else
  begin
    StartWinId := SelectWinId;
    StopWinId := StartWinId;
  end;

  For id := StartWinId to StopWinId do
  With SurfAcqForm.Config.Setup.Probe[pid] do
  begin
    if (ProbeType <> CONTINUOUSTYPE) then
    begin
      Threshold := 2047-SurfAcqForm.ChanForm[SelectWinId].win.SlideBar.Position;
      y := round(2047+threshold);
      if y < 0 then y := 0;
      if y > Length(Screeny)-1 then y := Length(Screeny)-1;
      SurfAcqForm.ChanForm[id].win.MarkerH.Top := Screeny[y];
      SurfAcqForm.ChanForm[id].win.Threshold.Caption := InTToStr(threshold);
      SurfAcqForm.ChanForm[id].win.SlideBar.Position := SlideBar.Position;
      SurfAcqForm.ChanForm[id].win.Refresh;
    end;
  end;
end;
*)
{==============================================================================}
Procedure CProbeWin.ThreshChange(pid,threshold : integer);
begin
  SurfAcqForm.Config.Setup.Probe[pid].Threshold := Threshold
end;

{==============================================================================}
procedure TSurfAcqForm.RecButClick(Sender: TObject);
begin
  StartStopRecording;
end;

{==============================================================================}
procedure TSurfAcqForm.FileExitClick(Sender: TObject);
begin
  Close;
end;

{==============================================================================}
procedure TSurfAcqForm.NewDataFileClick(Sender: TObject);
var doit : boolean;
begin
  If FileIsOpen then
    MessageDlg('Another file, '+SaveSURF.SurfFileName+', is already open',mtWarning,[mbOk],0)
  else
    If NewDialog.Execute then
    begin
      doit := TRUE;
      FileIsOpen := FALSE;
      if FileExists(NewDialog.Filename) then
        If MessageDlg(NewDialog.Filename+' already exists.  Overwrite?',mtWarning,
          [mbYes,mbNo],0) = mrNo then doit := FALSE;
      if doit then
      begin
        if not SaveSURF.OpenSurfFileForWrite(NewDialog.Filename)
          then MessageDlg('Error creating '+NewDialog.Filename,mtWarning,[mbOk],0)
          else FileIsOpen := TRUE;
      end else FileIsOpen := FALSE;
    end;
  if FileIsOpen then
  begin
    FileInfoPanel.Panels[0].Text := 'Data File: '+ SaveSURF.SurfFileName;
    FileInfoPanel.Panels[0].Width := round(length(FileInfoPanel.Panels[0].Text)*FileInfoPanel.Font.Size* 2/3);
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CloseDataFileClick(Sender: TObject);
begin
  If FileIsOpen
    then begin SaveSURF.CloseFile; FileIsOpen := FALSE; end
    else MessageDlg('No file open',mtWarning,[mbOk],0);
  if not FileIsOpen then
  begin
    FileInfoPanel.Panels[0].Text := 'No data file open';
    FileInfoPanel.Panels[0].Width := round(length(FileInfoPanel.Panels[0].Text)*FileInfoPanel.Font.Size* 2/3);
  end;
  PRecord.BevelOuter := bvRaised;// RecordingBut.Caption := 'Start Recording';
end;

{==============================================================================}
procedure TSurfAcqForm.SaveSurfConfig(Filename : String);
var fs : TFileStream;
    i : integer;
begin
   GotCfgFileName := FALSE;
   CfgFileName := Filename;
   FileInfoPanel.Panels[1].Text := 'Config File: '+ CfgFileName;

   Config.MainWinLeft  :=Left;
   Config.MainWinTop   :=Top;
   Config.MainWinHeight:=Height;
   Config.MainWinWidth :=Width;
   Config.InfoWinLeft  :=InfoWin.Left;
   Config.InfoWinTop   :=InfoWin.Top;
   Config.InfoWinHeight:=InfoWin.Height;
   Config.InfoWinWidth :=InfoWin.Width;
   Config.DinChecked   := Config.DinChecked;

   try
     fs := TFileStream.Create(CfgFileName,fmCreate);
     fs.WriteBuffer('SCFG',4);
     fs.WriteBuffer('v1.0.0',6);//write version number

     For i := 0 to Config.NumProbes-1 do
       PutWindowLocationsInConfig(i);
     fs.WriteBuffer(config,sizeof(config));
     GotCfgFileName := TRUE;
   except
     ShowMessage('Exception raised when saving configuration.  File not saved.');
     Exit;
   end;
   fs.free;
end;

{==============================================================================}
procedure TSurfAcqForm.SaveConfigClick(Sender: TObject);
begin
  if GotCfgFileName = FALSE then
  begin
    if SaveConfigDialog.Execute then
      SaveSurfConfig(SaveConfigDialog.FileName)
    else ShowMessage('Config not saved');
  end else SaveSurfConfig(CfgFileName);
end;

{==============================================================================}
procedure TSurfAcqForm.SaveConfigAsClick(Sender: TObject);
begin
  if SaveConfigDialog.Execute then
    SaveSurfConfig(SaveConfigDialog.FileName)
  else ShowMessage('Config not saved');
end;

{==============================================================================}
procedure TSurfAcqForm.OpenSurfConfig(Filename : String);
var fs : TFileStream;
    ok : boolean;
    i,p,nc :integer;
    headerstr : array[0..3] of char;
    versionstr : array[0..5] of char;
begin
  ok := TRUE;
  GotCfgFileName := FALSE;
  try
    fs := TFileStream.Create(FileName,fmOpenRead);
    CfgFileName := FileName;
    FileInfoPanel.Panels[1].Text := 'Config File: '+ CfgFileName;

    fs.ReadBuffer(headerstr,4);
    if headerstr <> 'SCFG' then
    begin
      ShowMessage('This is not a SURF configuration file');
      ok := FALSE;
      exit;
    end;

    fs.ReadBuffer(versionstr,6);
    if versionstr <> 'v1.0.0' then
    begin
      ShowMessage('Unsupported version of configuration file.');
      ok := FALSE;
      exit;
    end;

    rawringarray := NIL;
{$IFDEF ADBOARD}
    UnloadBuffers;
{$ENDIF}

    FreeChanWindows;

    fs.ReadBuffer(config,sizeof(config));
    left := Config.MainWinLeft;
    Top := Config.MainWinTop;
    Height := Config.MainWinHeight;
    Width := Config.MainWinWidth;
    InfoWin.Left := Config.InfoWinLeft;
    InfoWin.Top := Config.InfoWinTop;
    InfoWin.Height := Config.InfoWinHeight;
    InfoWin.Width := Config.InfoWinWidth;

    //Move(Config.Setup,ProbeSetupWin.Setup,sizeof(ProbeSetupRec));
    //ProbeSetupWin.NSpikeProbeSpin.Value := Config.Setup.NSpikeProbes;
    //ProbeSetupWin.NCRProbesSpin.Value := Config.Setup.NCRProbes;
    //ProbeSetupWin.CreateProbeRows;
    //ProbeSetupWin.DinCheckBox.Checked := Config.DinChecked;

    ConfigProbesBut.Caption := '&Modify';

    //only reference to probewin here
    SampFreq := ProbeSetupWin.SampFreqPerChan.Value;
    nc := Config.Setup.TotalChannels;
    if nc < 8 then nc := 8;
    BuffSize := 32767 div nc;

    //nc := Config.NumChans;
    //if Config.DinChecked then inc(nc);

    NumBuffs := 3 + round(sqrt((SampFreq*Config.Setup.TotalChannels) / buffsize));
    RawRingArraySize := {U}LNG(BuffSize*NumBuffs*Config.Setup.TotalChannels);
    rawringarray := nil;
    SetLength(rawringarray,RawRingArraySize);

{$IFDEF ADBOARD}
    if not ConfigBoard then
    begin
      ShowMessage('Error: Board not configured properly');
      ok := FALSE;
      exit;
    end;
{$ENDIF}

    //create the windows and draw them to the screen
    For p := 0 to Config.NumProbes-1 do
    begin
        CreateAProbeWindow(p,
                           Config.WindowLoc[p].left,
                           Config.WindowLoc[p].top,
                           Config.Setup.Probe[p].NPtsPerChan);
        ProbeWin[p].win.Width := Config.WindowLoc[p].Width;
        ProbeWin[p].win.Height := Config.WindowLoc[p].Height;
        if ClientWidth < ProbeWin[p].win.Left + ProbeWin[p].win.Width then
          ClientWidth := ProbeWin[p].win.Left + ProbeWin[p].win.Width;
    end;
    {For i := 0 to Config.NumAnalogChannels-1 do
    begin
      winid := Config.CglList[i].WindowId;
      if not Config.CglList[i].IsAMultiChan then
      begin
        CreateAChanWindow(winid,Config.CglList[i].ProbeId,
                                Config.CglList[i].ChanOffset-Config.Setup.Probe[Config.CglList[i].ProbeId].ChanStart,
                                Config.CglList[i].WinLoc.left,
                                Config.CglList[i].WinLoc.top,
                                Config.Setup.Probe[Config.CglList[i].ProbeId].NPtsPerChan);
        ChanForm[winid].win.Width := Config.CglList[i].WinLoc.Width;
        ChanForm[winid].win.Height := Config.CglList[i].WinLoc.Height;
        if ClientWidth < ChanForm[winid].win.Left + ChanForm[winid].win.Width then
          ClientWidth := ChanForm[winid].win.Left + ChanForm[winid].win.Width;
      end else
      //if MultiChan[winid] = nil then
      begin
        CreateAMultiChanWindow(winid,Config.CglList[i].ProbeId,
                                     Config.CglList[i].WinLoc.left,
                                     Config.CglList[i].WinLoc.top,
                                     Config.Setup.Probe[Config.CglList[i].ProbeId].NPtsPerChan);
        MultiChan[winid].win.Width := Config.CglList[i].WinLoc.Width;
        MultiChan[winid].win.Height := Config.CglList[i].WinLoc.Height;
        if ClientWidth < MultiChan[winid].win.Left + MultiChan[winid].win.Width then
          ClientWidth := MultiChan[winid].win.Left + MultiChan[winid].win.Width;
      end;
    end;
    }
    fs.Free;
    GotCfgFileName := TRUE;

    if not SetupProbeWins
    then begin
      ShowMessage('Error: Plot windows not configured properly');
      ok := FALSE;
      exit;
    end;

    For i := 0 to Config.NumAnalogChannels-1 do
    begin
      {if Config.Setup.Probe[Config.CglList[i].ProbeId].InternalGain > 0
        then vf := 10 / Config.Setup.Probe[Config.CglList[i].ProbeId].InternalGain
        else vf := 0;  }
{$IFDEF ADBOARD}
      DTAcq.GainList[i] := Config.Setup.Probe[Config.CglList[i].ProbeId].InternalGain;//GainTrigLockList[GAININDEX,Config.ProbeChannelList[PROBEINDEX,i]];
      {if DTAcq.GainList[i] > 0
        then vf := 10 / DTAcq.GainList[i]
        else vf := 0; }
{$ENDIF}
      //vs := FloatToStr(vf);
      {if not Config.CglList[i].IsAMultiChan then
      begin
        j := Config.CglList[i].WindowId;
        ChanForm[j].win.HiVolt.Caption := '+'+ vs + 'V';
        ChanForm[j].win.LoVolt.Caption := '-'+ vs + 'V';
        if ChanForm[j].win.MarkerV.Visible then
          ChanForm[j].win.MarkerV.left := ChanForm[j].win.plot.left + Config.Setup.Probe[Config.CglList[i].ProbeId].TrigPt;// Config.GainTrigLockList[TRIGINDEX,Config.ProbeChannelList[PROBEINDEX,i]];
      end;
      }
    end;
  finally
  end;

  if not ok then
  begin
    ShowMessage('Configuration file not loaded');
    exit;
  end;

  {UserWinExists := TRUE;
  UserWinHandle := FindWindow('TUserWin','Surf Users Window');
  if UserWinHandle = 0 then UserWinExists := FALSE;
  }//Lynx8AmpControlExists := TRUE;
  //Lynx8AmpHandle := FindWindow('TLynx8Form','Lynx-8 Control Window');
  //PostMessage(UserWinHandle,WM_USERWINSHOW,0,0);
  //PostMessage(Lynx8AmpHandle,WM_LYNX8SHOW,0,0);
end;

{==============================================================================}
procedure TSurfAcqForm.OpenConfigClick(Sender: TObject);
begin
  With OpenConfigDialog do
     if Execute then OpenSurfConfig(Filename);
end;

procedure TSurfAcqForm.WriteMessageClick(Sender: TObject);
begin
  if not Recording then
  begin
    ShowMessage('Not Recording to disk');
    exit;
  end;
  msgrec.time_stamp  := TheTime.TenthMS; //4 bytes
  msgrec.ufftype := SURF_MSG_REC_UFFTYPE;
  msgrec.subtype := '0';
  MesgQueryForm := TSurfMesg.Create(Self);
  MesgQueryForm.Show;
end;

Procedure TSurfMesg.MessageSent(mesg : ShortString);
begin
  SurfAcqForm.msgrec.msg := Mesg;
  if (SurfAcqForm.msgrec.msg <> '') and SurfAcqForm.Recording then
    SurfAcqForm.SaveSURF.PutMessageRecord(SurfAcqForm.msgrec);
  Free;
end;

procedure TSurfAcqForm.FormShow(Sender: TObject);
var i : integer;
begin
  Acquisition := FALSE;
  Recording := FALSE;
  FileIsOpen := FALSE;

  SurfComm.Free;

  if NumUserApps > 0 then
  begin
    SurfComm := TSurfComObj.CreateParented(Handle);
    for i := 0 to NumUserApps-1 do
      SurfComm.CallUserApp(UserFileNames[i]);
    UserWinExists := TRUE;
  end else UserWinExists := FALSE;

  InfoWin := TInfoWin.CreateParented(WaveFormPanel.Handle);
  InfoWin.Top := 0;
  InfoWin.Left := ClientWidth-InfoWin.Width-10;
  //InfoWin.Show;

  SaveSURF := TSurfFile.Create;
  FileInfoPanel.Panels[0].Width := round(length(FileInfoPanel.Panels[0].Text)*FileInfoPanel.Font.Size* 2/3);

  TheTime.TenthMS := 0;//10th ms

  //Initialize numchans and RawRingArraySize for memory allocation
  Config.NumAnalogChannels := 0;
  Config.NumProbes := 0;
  Config.Setup.TotalChannels := 0;
  
  RawRingArraySize := 0;
  SetLength(rawringarray,RawRingArraySize);
  SampFreq := 32000;

  GotCfgFileName := FALSE;
  CfgFileName := '';

{$IFDEF ADBOARD}
    if not InitBoard then
    begin
      ShowMessage('Error: A/D Board would not initialize.');
      Exit;
    end;
{$ENDIF}

  //FillChar(config.winloc,sizeof(config.winloc),0);
  Config.empty := true;
  ConfigProbesBut.Caption := '&New';
end;

procedure TSurfAcqForm.FormHide(Sender: TObject);
var i : integer;
begin
  if Acquisition then StartStopAcquisition;
  UnloadBuffers;
  If Recording then SaveSURF.CloseFile;
  FreeChanWindows;

  For i := 0 to  CRRINGARRAYSIZE-1 do
    CRRingArray[i].Waveform := nil;
  For i := 0 to  SPIKERINGARRAYSIZE-1 do
    SpikeRingArray[i].Waveform := nil;

  rawringarray := nil;
  SaveSURF.Free;
  SurfComm.Free;
  UserFileNames := nil;

  InfoWin.Free;
  STimer.Enabled := FALSE;
  //if UserWinExists then PostMessage(UserWinHandle,WM_USERWINCLOSE,0,0);
  //if Lynx8AmpControlExists then PostMessage(Lynx8AmpHandle,WM_LYNX8CLOSE,0,0);
end;

Procedure TSurfComObj.PutDACOut(DAC : TDAC);
const VOLTTOVAL = 2047/10;
      NOUTBUFS = 300;
      OUTBUFSIZE = 100;
var ival,i : integer;
    hBuf : HBUFTYPE;
    lpAppBuffer : LPSHRT;
    outwave : array of SHRT;
    stmp,stmp2 : single;
begin
  if not (dac.channel in [0,1]) then exit;
  ival := round((dac.voltage+10) * VOLTTOVAL);

  if dac.frequency > 0 {ac} then
  begin
    //if not already continuous then make it
    SetLength(outwave,OUTBUFSIZE*NOUTBUFS);
    stmp := PI*2/OUTBUFSIZE;
    stmp2 := dac.voltage * VOLTTOVAL;
    For i := 0 to OUTBUFSIZE*NOUTBUFS-1 do
      outwave[i] := 2047 + round(sin(i*stmp)* stmp2);
    try
      SurfAcqForm.DTDAC.Abort;
      SurfAcqForm.DTDAC.Reset;
    except
    end;
      SurfAcqForm.DTDAC.Flush;
      hBuf := HBUFTYPE(SurfAcqForm.DTDAC.Queue);
      if hBuf <> NULL then olDmFreeBuffer(hBuf);
    //Delay(0,100); //give it time to flush out
    //if SurfAcqForm.DTDAC.DataFlow <> OL_DF_CONTINUOUS then
    begin
      SurfAcqForm.DTDAC.DataFlow := OL_DF_CONTINUOUS;
      SurfAcqForm.DTDAC.WrapMode := OL_WRP_MULTIPLE; //set up for multiple buffer
      SurfAcqForm.DTDAC.Encoding := OL_ENC_BINARY;
      SurfAcqForm.DTDAC.ListSize := 1;
      SurfAcqForm.DTDAC.ChannelList[0] := 0;
      SurfAcqForm.DTDAC.ClockSource := OL_CLK_INTERNAL;
      //create a buffer
      olDmAllocBuffer(0, 2*OUTBUFSIZE*NOUTBUFS, @hbuf);
    end;

    SurfAcqForm.DTDAC.Frequency := dac.frequency * OUTBUFSIZE;

    lpAppBuffer := @outwave[0];
    olDmCopyToBuffer (hbuf,lpAppBuffer,OUTBUFSIZE*NOUTBUFS);
    SurfAcqForm.DTDAC.Queue := {U}LNG(hBuf);

    SurfAcqForm.DTDAC.config;
    SurfAcqForm.DTDAC.Start;
    outwave := nil;
  end else
  begin
    if SurfAcqForm.DTDAC.DataFlow <> OL_DF_SINGLEVALUE then
    begin
      try
        SurfAcqForm.DTDAC.Abort;
        SurfAcqForm.DTDAC.Reset;
      except;
      end;
      SurfAcqForm.DTDAC.Flush;
      hBuf := HBUFTYPE(SurfAcqForm.DTDAC.Queue);
      if hBuf <> NULL then olDmFreeBuffer(hBuf);

      SurfAcqForm.DTDAC.DataFlow := OL_DF_SINGLEVALUE;
      SurfAcqForm.DTDAC.config;
    end;
    SurfAcqForm.DTDAC.PutSingleValue(dac.channel,1.0,ival);
  end;
end;

Procedure TSurfComObj.PutDIOOut(DIO : TDIO);
begin
  SurfAcqForm.DTDIO.PutSingleValue(0,1.0,{U}LNG(dio.val));
  //showmessage('about to put out dio' + inttostr(dio.val));
  //SurfAcqForm.DTDIO.PutSingleValue(ichan,dgain,ival);
end;

procedure TSurfAcqForm.PInfoWinClick(Sender: TObject);
begin
  InfoWin.Visible := not InfoWin.Visible;
  if InfoWin.Visible
    then PInfoWin.BevelOuter := bvLowered
    else PInfoWin.BevelOuter := bvRaised;
  InfoWin.BringToFront;  
end;

end.

