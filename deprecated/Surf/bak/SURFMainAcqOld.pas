unit SURFMainAcq;
interface

uses About,Windows, Messages, ShellApi, ExtCtrls, Buttons, StdCtrls, ToolWin, ComCtrls,
  Controls, OleCtrls, Classes, Gauges, DTxPascal, DTAcq32Lib_TLB,
  Forms, Dialogs, Math, Menus, SysUtils,Graphics, SuperTimer,{, Animate, GIFCtrl,}
  SurfComm, SurfFile,SurfTypes, SurfPublicTypes, WaveFormUnit, ProbeSet,
  InfoWinUnit, PahUnit, SurfMessage,ExtGain, Surf2SurfBridge;

const
  SPIKERINGARRAYSIZE = 2000;
  CRRINGARRAYSIZE    = 100;
  SVRINGARRAYSIZE    = 3000;

  PROBEINDEX = 1;
  CHANINDEX = 2;
  GAININDEX = 1;
  TRIGINDEX = 2;
  LOCKINDEX = 3;
  DINPROBE = 32;

type
  PolytrodeRecord = record
    Time     : LNG;
    ProbeNum : SHRT;
    Waveform : TWaveForm;
  end;

  CRTempRecordType = record
    Time     : LNG;
    ProbeNum : SHRT;
    Waveform : array[0..SURF_MAX_WAVEFORM_PTS-1] of SHRT;
  end;

  {DinRecordType = record
    Time     : LNG;
    DinVal   : WORD;
  end;}

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

  TChannelObj = class(TWaveFormWin)
    public
      Procedure ThreshChange(pid,cid : integer; ShiftDown,CtrlDown : boolean); override;
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

  TChanRec = record
    exists : boolean;
    ProbeId,ChanId : integer;
    win : TChannelObj;
  end;

  ProbeChannelListType = array[PROBEINDEX..CHANINDEX,0..MAXENTRIES-1] of Integer;
  GainTrigLockListType = array[GAININDEX..LOCKINDEX,0..SURF_MAX_PROBES-1] of Integer;
  ChannelWinIdType     = array[0..SURF_MAX_PROBES-1,0..SURF_MAX_CHANNELS-1] of Integer;

  ChanLocRec = record
    Left,top,width,height : integer;
    chantype : char;
  end;

  ConfigRec = record
    NumProbes,NumChannels : integer;
    ProbeChannelList : ProbeChannelListType;
    GainTrigLockList : GainTrigLockListType;
    ChannelWinId     : ChannelWinIdType;
    Setup : ProbeSetupRec; //same as probewin.setup
    MainWinLeft,MainWinTop,MainWinHeight,MainWinWidth : Integer;
    InfoWinLeft,InfoWinTop,InfoWinHeight,InfoWinWidth : Integer;
    DinChecked : boolean;
    chanloc : array[0..255] of ChanLocRec;
  end;

  TSurfAcqForm = class(TForm)
    FileInfoPanel: TStatusBar;
    NewDialog: TSaveDialog;
    SaveConfigDialog: TSaveDialog;
    OpenConfigDialog: TOpenDialog;
    WaveFormPanel: TPanel;
    ToolBar2: TToolBar;
    AcquisitionBut: TBitBtn;
    Splitter9: TSplitter;
    RecordingBut: TBitBtn;
    MainMenu: TMainMenu;
    About1: TMenuItem;
    WriteMessage: TButton;
    Splitter1: TSplitter;
    //ztimer: TSuperTimer;
    DTClock: TDTAcq32;
    DTCounter: TDTAcq32;
    DataFile1: TMenuItem;
    NewDataFile: TMenuItem;
    CloseDataFile: TMenuItem;
    Config1: TMenuItem;
    ConfigProbesBut: TMenuItem;
    OpenConfig: TMenuItem;
    SaveConfig: TMenuItem;
    SaveConfigAs: TMenuItem;
    DTDAC: TDTAcq32;
    DTDIO: TDTAcq32;
    DTAcq: TDTAcq32;
    stimer: TSuperTimer;
    procedure ExitItemClick(Sender: TObject);
    procedure About1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);

    procedure DTAcqOverrunError(Sender: TObject);
    procedure DTAcqTriggerError(Sender: TObject);
    procedure DTAcqQueueStopped(Sender: TObject);
    procedure DTAcqBufferDone(Sender: TObject);
    procedure DTAcqUnderrunError(Sender: TObject);

    procedure ConfigProbesButClick(Sender: TObject);
    //procedure ZTimerTimer(Sender: TObject);
    procedure RecButClick(Sender: TObject);
    procedure FileExitClick(Sender: TObject);
    procedure CloseDataFileClick(Sender: TObject);
    procedure SaveConfigClick(Sender: TObject);
    procedure SaveConfigAsClick(Sender: TObject);
    procedure OpenConfigClick(Sender: TObject);
    procedure AcquisitionButClick(Sender: TObject);
    procedure RecordingButClick(Sender: TObject);
    procedure NewDataFileClick(Sender: TObject);
    procedure WriteMessageClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure FormHide(Sender: TObject);
    procedure stimerTimer(Sender: TObject);
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
    GotCfgFileName,AllChannelsExist : Boolean;
    CfgFileName : String;
    FirstBufferDone,SecondBufferDone : boolean;
    nrawspikes,nsavedspikes,ndisplayedspikes,ndroppedspikes : LNG;
    nsavedcrbuffers,ndisplayedcrbuffers,nrawcrbuffers : LNG;
    nsaveddinbuffers,nrawdinbuffers : LNG;
    TotalBuffersAcquired : LNG;
    TimeScale : Double;
    BuffSize,NumBuffs : LNG;
    ShiftDown,BuffersAllocated : boolean;
    ChanForm : array[0..SURF_MAX_CHANNELS-1] of TChanRec;
    RawRingArraySize : {U}LNG;
    SampFreq : Integer;
    SingleWave : TWaveForm;
    InfoWin : TInfoWin;
    ProbeChannelList : ProbeChannelListType;
    GainTrigLockList : GainTrigLockListType;
    ChannelWinId     : ChannelWinIdType;
    LastSpikeTrigPtofProbe  :array[0..SURF_MAX_PROBES-1] of Integer;
    LastSpikeTimeofProbe  :array[0..SURF_MAX_PROBES-1] of Integer;
    LastDinVal : WORD;
    MaxFreq,MinFreq : LNG;
    SurfComm : TSurfComObj;
    //UserWinHandle{,Lynx8AmpHandle} : HWnd;

    NumChans,NumProbes : LNG;//ULNG{Integer};
    SaveSURF : TSurfFile;
    Acquisition,Recording,FileIsOpen : boolean;
    TheTime : TheTimeRec;
    SpikeRingWriteIndex,CRRingWriteIndex,{DinRingWriteIndex,}SvRingWriteIndex : LNG;
    SpikeRingArray : array[0..SPIKERINGARRAYSIZE-1] of PolytrodeRecord;
    CRRingArray    : array[0..CRRINGARRAYSIZE-1] of PolytrodeRecord;
    SvRingArray    : array[0..SVRINGARRAYSIZE-1] of SvRecordType;

    UserWinExists{,Lynx8AmpControlExists,SettingLynx8} : boolean;
    config : configrec;
    msgrec : SURF_MSG_REC;
    MesgQueryForm: TSurfMesg;

    Procedure StartStopAcquisition;
    Procedure StartStopRecording;
    procedure SetUpProbes;
    Procedure CreateAChanWindow(id,probenum,channum,left,top,npts : integer);
    Procedure FreeChanWindows;
    Function InitBoard : boolean; //false if error
    Function ConfigBoard : boolean; //false if error
    function UnloadBuffers : boolean; //false if error
    function SetupPlotWins : boolean; //false if error
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
    //User access functions
    (*{Global}
      Function GetNumChans : Integer;
      Function GetNumProbes : Integer;
      Function GetTime : TheTimeRec;
      Function GetAcquisitionState : boolean;
      Function GetRecordingState : boolean;
    {Polytrode Records}
      Function GetSpikeRingWriteIndex : Longint;
      Function GetCRRingWriteIndex : Longint;
      Function GetDINRingWriteIndex : Longint;
      Procedure GetSpikeFromRing(index : longint; var time,probenum : longint);
      Function GetCRFromRing(index : longint) : PolytrodeRecord;
      Function GetChanStartNumberForProbe(probenum : integer) : integer;
      Function GetNumChansForProbe(probenum : integer) : integer;
    {Digital}
      Function GetDINFromRing(index : longint) : DinRecordType;
    {I/O}
      Function GetFileOpenState : boolean;
     *)
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
  DragAcceptFiles( Handle, True );
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

{==============================================================================}
Function TSurfAcqForm.UnloadBuffers : boolean; //false if error
var i : integer;
    hbuf : HBUFTYPE;
begin
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
procedure TSurfAcqForm.CreateAChanWindow(id,probenum,channum,left,top,npts : integer);
var  s : string;
begin
  ChanForm[id].exists := TRUE;
  ChanForm[id].win := TChannelObj.CreateParented(WaveFormPanel.Handle);
  s := 'P'+inttostr(probenum)+ ' C' + IntToStr(channum) + ' ('+inttostr(ProbeChannelList[CHANINDEX,id])+') ';
  if ProbeWin.Setup.Probe[probenum].ProbeType = CONTINUOUSTYPE then
    s := s + ProbeWin.Setup.Probe[probenum].Descrip;

  ChanForm[id].win.InitPlotWin({npts}npts,
                               {left}Left,
                                {top}Top,
                           {bmheight}50,
                            {intgain}ProbeWin.Setup.Probe[probenum].InternalGain,
                             {thresh}ProbeWin.Setup.Probe[probenum].Threshold,
                             {trigpt}ProbeWin.Setup.Probe[probenum].TrigPt,
                            {probeid}probenum,
                             {chanid}id,
                          {probetype}ProbeWin.Setup.Probe[probenum].ProbeType,
                              {title}s,
                              {view} ProbeWin.Setup.Probe[probenum].view,
                    {acquisitionmode}TRUE);

  if SurfAcqForm.ClientWidth < Left + ChanForm[id].win.width
    then SurfAcqForm.ClientWidth := Left + ChanForm[id].win.width;
  if SurfAcqForm.ClientHeight      < FileInfoPanel.height + WaveFormPanel.Top + ChanForm[id].win.top + ChanForm[id].win.height
    then SurfAcqForm.ClientHeight := FileInfoPanel.height + WaveFormPanel.Top + ChanForm[id].win.top + ChanForm[id].win.height;
end;

{==============================================================================}
procedure TSurfAcqForm.FreeChanWindows;
var i,nc : integer;
begin
  if ProbeWin.DinCheckBox.Checked
    then nc :=  NumChans-1
    else nc :=  NumChans;
  if nc > 0 then
    For i := 0 to nc-1 do if ChanForm[i].exists then
    begin
      ChanForm[i].win.Free;
      ChanForm[i].exists := false;
    end;
  AllChannelsExist := FALSE;
end;

{==============================================================================}
function TSurfAcqForm.SetupPlotWins : boolean; //false if error
var i,nc,probenum,channum : integer;
    s : string;
begin
  SetupPlotWins := TRUE;
  if ProbeWin.DinCheckBox.Checked
    then nc :=  NumChans-1
    else nc :=  NumChans;

  For i := 0 to nc-1 do if ChanForm[i].exists then
  begin
    probenum := SurfAcqForm.ProbeChannelList[PROBEINDEX,i];
    channum  := SurfAcqForm.ProbeChannelList[CHANINDEX,i];
    s := 'P'+inttostr(probenum)+ ' C' + IntToStr(channum) + ' ('+inttostr(ProbeChannelList[CHANINDEX,i])+') ';
    if ProbeWin.Setup.Probe[probenum].ProbeType = CONTINUOUSTYPE then
      s := s + ProbeWin.Setup.Probe[probenum].Descrip;

    ChanForm[i].win.InitPlotWin({npts}ProbeWin.Setup.Probe[probenum].NPtsPerChan,
                                {left}ChanForm[i].win.Left,
                                 {top}ChanForm[i].win.Top,
                            {bmheight}ChanForm[i].win.ClientHeight,
                             {intgain}ProbeWin.Setup.Probe[probenum].InternalGain,
                              {thresh}ProbeWin.Setup.Probe[probenum].Threshold,
                              {trigpt}ProbeWin.Setup.Probe[probenum].TrigPt,
                             {probeid}probenum,
                              {chanid}i,
                           {probetype}ProbeWin.Setup.Probe[probenum].ProbeType,
                               {title}s,
                                {view}ProbeWin.Setup.Probe[probenum].view,
                     {acquisitionmode}TRUE);
    if SurfAcqForm.ClientWidth < Left + ChanForm[i].win.width
      then SurfAcqForm.ClientWidth := Left + ChanForm[i].win.width;
    if SurfAcqForm.ClientHeight      < FileInfoPanel.height + WaveFormPanel.Top + ChanForm[i].win.top + ChanForm[i].win.height
      then SurfAcqForm.ClientHeight := FileInfoPanel.height + WaveFormPanel.Top + ChanForm[i].win.top + ChanForm[i].win.height;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.AcquisitionButClick(Sender: TObject);
begin
  StartStopAcquisition;
end;


{==============================================================================}
Procedure TSurfAcqForm.StartStopAcquisition;
var i : integer;
begin
  If Acquisition then //Stop acquisition
  begin
    Acquisition := FALSE;
    //AcqAnim.Animate := FALSE;
    STimer.Enabled := FALSE;
    AcquisitionBut.Caption := 'Start &Acquisition';
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
    For i := 0 to  CRRINGARRAYSIZE-1 do
      CRRingArray[i].Waveform := NIL;
    For i := 0 to  SPIKERINGARRAYSIZE-1 do
      SpikeRingArray[i].Waveform := NIL;
    SingleWave := NIL;
  end else //Start it
  begin
    if NumChans = 0 then
    begin
      ShowMessage('No channels setup');
      exit;
    end;
    Acquisition := TRUE;
    AcquisitionBut.Caption := 'Stop &Acquisition';
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

    if Infowin.UpdateCheck.Checked then
    begin
      InfoWin.SpikesAcquired.Caption := IntToStr(nrawspikes);
      InfoWin.SpikesDisplayed.Caption := IntToStr(nDisplayedspikes);
      InfoWin.CRBuffersAcquired.Caption := IntToStr(nrawcrbuffers);
      InfoWin.CRBuffersDisplayed.Caption := IntToStr(ndisplayedcrbuffers);
      InfoWin.DinBuffersAcquired.Caption := IntToStr(nrawdinbuffers);
    end;

    TotalBuffersAcquired := 0;
    LastDinVal := 0;

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
procedure TSurfAcqForm.RecordingButClick(Sender: TObject);
begin
  StartStopRecording;
end;

{==============================================================================}
Procedure TSurfAcqForm.StartStopRecording;
var nc,i : integer;
begin
  if Recording then  //Stop recording
  begin
    Recording := FALSE;
    RecordingBut.Caption := 'Start &Recording';
    //PostMessage(UserWinHandle,WM_RECSTOP,0,0);
    if ProbeWin.DinCheckBox.Checked
      then nc :=  NumChans-1
      else nc :=  NumChans;
    For i := 0 to nc-1 do
    begin
      ChanForm[i].win.MarkerV.Visible := TRUE;
      ChanForm[i].win.MarkerH.Visible := TRUE;
    end;
  end else
  begin             //Start Recording
    If FileIsOpen then
    begin
      Recording := TRUE;
      RecordingBut.Caption := 'Stop &Recording';
      //PostMessage(UserWinHandle,WM_RECSTART,0,0);
      WriteSURFRecords;

      if ProbeWin.DinCheckBox.Checked
        then nc :=  NumChans-1
        else nc :=  NumChans;
      For i := 0 to nc-1 do
      begin
        ChanForm[i].win.MarkerV.Visible := FALSE;
        ChanForm[i].win.MarkerH.Visible := FALSE;
      end;
    end else
    begin
      Beep;
      ShowMessage('Error: no data file open');
      Recording := FALSE;
      RecordingBut.Caption := 'Start Recording';
    end;
  end;
end;

{==============================================================================}
Procedure TSurfAcqForm.WriteSurfRecords;
var i,c,nc,ci : integer;
    slr : SURF_LAYOUT_REC_V1;
begin
  if ProbeWin.DinCheckBox.Checked
    then nc :=  NumChans-1
    else nc :=  NumChans;

  For i := 0 to nc-1 do
  begin
    config.chanloc[i].left := ChanForm[i].win.Left;
    config.chanloc[i].top := ChanForm[i].win.top;
    config.chanloc[i].width := ChanForm[i].win.width;
    config.chanloc[i].height := ChanForm[i].win.height;
    config.chanloc[i].chantype := ProbeWin.Setup.Probe[ProbeChannelList[PROBEINDEX,i]].Probetype;
  end;

  With slr do
  begin
    ufftype    := SURF_PL_REC_UFFTYPE; // Record type  chr(234)
    time_stamp := TheTime.TenthMS;  // Time stamp
    surf_major := 1; // SURF major version number
    surf_minor := 0; // SURF minor version number
    FillChar(pad,sizeof(pad),0);
  end;

  ci := 0;
  For i := 0 to NumProbes-1 do
  begin
    With slr do //probe specific settings
    begin
      probe          := i;                   //Probe number
      ProbeSubType   := ProbeWin.Setup.probe[i].ProbeType; {=S,C for spike or continuous }
      nchans         := ProbeWin.Setup.probe[i].NChannels; //number of channels in this spike waveform
      pts_per_chan   := ProbeWin.Setup.probe[i].NPtsPerChan; //number of pts per waveform
      trigpt         := ProbeWin.Setup.probe[i].TrigPt; // pts before trigger
      lockout        := ProbeWin.Setup.probe[i].Lockout; // Lockout in pts
      intgain        := ProbeWin.Setup.probe[i].InternalGain; // A/D board internal gain
      threshold      := ProbeWin.Setup.probe[i].Threshold; // A/D board threshold for trigger
      skippts        := ProbeWin.Setup.probe[i].SkipPts;
      sampfreqperchan:= ProbeWin.Setup.probe[i].SampFreq;  // A/D sampling frequency
      probe_descrip  := ProbeWin.Setup.probe[i].Descrip;

      For c := 0 to SURF_MAX_CHANNELS-1 do
        if c < nc
          then chanlist[c] := ProbeWin.Setup.probe[i].ChanStart+c
          else chanlist[c] := -1;
      ExtGainForm.Probe := i;
      ExtGainForm.NumChannels := nc;
      ExtGainForm.ShowModal;
      For c := 0 to SURF_MAX_CHANNELS-1 do
        if c < nc
          then extgain[c] := Word(StrToInt(ExtGainForm.ExtGainArray[i].Text))
          else extgain[c] := 0;//unused
      For c := 0 to nchans-1 do
      begin
        screenlayout[c].x := config.chanloc[c+ci].left;
        screenlayout[c].y := config.chanloc[c+ci].top;
      end;
      inc(ci,nchans);
    end;

    if not SaveSURF.PutSurfRecord(slr) then
    begin
      Beep;
      ShowMessage('Error writing to Surf file');
      RecordingBut.Caption := 'Start Recording';
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
  ShowMessage('Queue stopped');
  StartStopAcquisition;
end;

{==============================================================================}
procedure TSurfAcqForm.DTAcqBufferDone(Sender: TObject);
var s,i,MaxSamples : LNG;//ULNG;
  RawRingPtr : Pointer;
  CheckRawRingArrayIndex : LNG;//ULNG;
  Param : Integer;
  hbuf : HBUFTYPE;
  WaveformStart,ProbeId : integer;
  cglindex : integer;
  RawRingAddressOffset : LNG;//ULNG;
  ptdiff,npts,nc : Integer;
  BeginTime : LNG;
  wvalue : WORD;
  Threshold : array[0..SURF_MAX_CHANNELS-1] of Integer;
  //SVRecord : SURF_SV_REC;
  //tmpSpike : TSpike;
  tmpSVal : TSVal;

Procedure GrabWave;  //Grab the spike from the buffer
var w,w2,p,xoffset,lockoutpts : LNG;//ULNG;
    trigpt,pchans : integer;
begin
  //grab the spike and place in the spike ring buffer
  xoffset    := NumChans * {U}LNG(GainTrigLockList[TRIGINDEX,probeid]);
  lockoutpts := NumChans * {U}LNG(GainTrigLockList[LOCKINDEX,probeid]);
  //if ProbeChannelList[CHANINDEX,cglindex] = 32 then showmessage('spk chan 32!');
  trigpt     := s - (ProbeChannelList[CHANINDEX,cglindex]-ProbeWin.Setup.Probe[probeid].ChanStart);
  WaveformStart := (RawRingArraySize+CheckRawRingArrayIndex+trigpt-xoffset) mod RawRingArraySize;

  With SpikeRingArray[SpikeRingWriteIndex] do
  begin
    //get the time of the spike to the nearest 1/10th ms
    Time := BeginTime{begin of last buffer}+round(trigpt*TimeScale);//This should be accurate to 1/10th ms
    ProbeNum := probeid;
    Npts := ProbeWin.Setup.Probe[probeid].NPtsPerChan;
    pchans := ProbeWin.Setup.probe[probeid].NChannels;
    ptdiff := trigpt - LastSpikeTrigPtofProbe[probeid];
    if ptdiff < 0 then ptdiff := ptdiff + RawRingArraySize;
    if ptdiff > lockoutpts then
    begin
      //Transfer the waveform to the spike record
      Waveform := nil;
      SetLength(Waveform,Npts*pchans);
      For p := 0 to Npts-1 do
      begin
        w2 := p * NumChans;
        For w := 0 to pchans-1 do
          Waveform[p*pchans+w] := RawRingArray[(WaveformStart+w2+w) mod RawRingArraySize];
      end;

      LastSpikeTrigPtofProbe[ProbeId] := trigpt;//Time;
      inc(nrawspikes);

      if Infowin.UpdateCheck.Checked then
        InfoWin.SpikesAcquired.Caption := IntToStr(nrawspikes);

      //Increment the spike ring write buffer pointer
      SpikeRingWriteIndex := (SpikeRingWriteIndex + 1) mod SPIKERINGARRAYSIZE;
      //InfoWin.WriteGuage.Progress := SpikeRingWriteIndex;
      LastSpikeTimeofProbe[ProbeId] := Time;
      if userwinexists then
      begin
        //copy polytrode record to a TSpike record
        //SendSpikeToSurfBridge(SpikeRingArray[SpikeRingWriteIndex]);
      end;
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
  BeginTime := round((TotalBuffersAcquired-1) * (TimeScale * BuffSize * NumChans{ + 0.02}){constant loss of 2us});

  // graph data & then recycle buffer
  hbuf := HBUFTYPE(DTAcq.Queue);  //retreiving buffer
//FileInfoPanel.Panels[1].Text := inttostr(BeginTime)+','+inttostr(hbuf);
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


  if MaxSamples = BuffSize*NumChans then
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
      if ProbeWin.DinCheckBox.Checked
        then nc :=  NumChans-1
        else nc :=  NumChans;
      For i := 0 to nc-1 do
        Threshold[i] := ChanForm[i].win.SlideBar.Max-ChanForm[i].win.SlideBar.Position;

      //Search entire buffer and save any waveform that passes a positive or negative threshold
      s := 0;
      While s < MaxSamples do
      begin
        cglindex := s mod NumChans;
        probeid := ProbeChannelList[PROBEINDEX,cglindex];

        wvalue := RawRingArray[(CheckRawRingArrayIndex+s) mod rawringarraysize];

        if (probeid = DINPROBE) then
        begin
          //move(RawRingArray[(CheckRawRingArrayIndex+s) mod rawringarraysize],wvalue,2);//an equivalence move (short to word)
          //wvalue := RawRingArray[(CheckRawRingArrayIndex+s) mod rawringarraysize];
          if wvalue <> LastDinVal then
          begin
            With SvRingArray[SvRingWriteIndex] do
            begin
              Time := BeginTime+round(s*TimeScale);//This should be accurate to 1/10th ms
              SubType := SURF_DIGITAL;
              SVal := wvalue;
            end;
            //if userwinexists then PostMessage(UserWinHandle,WM_DINAVAIL,0,0);
            if userwinexists then
            begin
              //copy din to a TSVal record and send it
              tmpSVal.time_stamp := SvRingArray[SvRingWriteIndex].Time;
              tmpSVal.subtype := SURF_DIGITAL;
              tmpSVal.EventNum := -1;
              tmpSVal.sval := wvalue;
              SendSVToSurfBridge(tmpSVal);
            end;
            SvRingWriteIndex := (SvRingWriteIndex + 1) mod SVRINGARRAYSIZE; // advance the write index
            LastDinVal := wvalue;
            inc(nrawdinbuffers);
            if Infowin.UpdateCheck.Checked then
              InfoWin.DinBuffersAcquired.Caption := IntToStr(nrawdinbuffers);
          end;
        //Check if this point is a cont rec channel
        end else
        if ProbeWin.Setup.Probe[probeid].probetype = CONTINUOUSTYPE then
        begin
//FileInfoPanel.Panels[1].Text := inttostr(TheTime.Sec)+','+inttostr(CRPointCount[probeid]);
          if (CRSkipCount[probeid]+1) mod ProbeWin.Setup.Probe[probeid].SkipPts = 0 then
          begin
            //is this a single pt cr channel?  If so then just put the single value on the ring buffer
            if ProbeWin.Setup.Probe[probeid].NPtsPerChan = 1 then
            begin
              With SvRingArray[SvRingWriteIndex] do
              begin
                Time := BeginTime+round(s*TimeScale);//This should be accurate to 1/10th ms
                SubType := SURF_ANALOG;
                SVal := wvalue;
              end;
              if userwinexists then
              begin
                //copy value to a TSVal record and send it
                tmpSVal.time_stamp := SvRingArray[SvRingWriteIndex].Time;
                tmpSVal.subtype := SURF_ANALOG;
                tmpSVal.EventNum := -1;
                tmpSVal.sval := wvalue;
                SendSVToSurfBridge(tmpSVal);
              end;
              SvRingWriteIndex := (SvRingWriteIndex + 1) mod SVRINGARRAYSIZE; // advance the write index
            end else
            begin
              //see if this is the first point in the waveform buffer of this probe
              if CRPointCount[probeid] = 0 then
              begin
                CRTempRecordArray[probeid].Time := BeginTime+round(s*TimeScale);//This should be accurate to 1/10th ms
                CRTempRecordArray[probeid].ProbeNum := probeid;
              end;
              param := wvalue;//RawRingArray[CheckRawRingArrayIndex+s];
              //assign the param to the waveform
              CRTempRecordArray[probeid].Waveform[CRPointCount[probeid]] := param;
              inc(CRPointCount[probeid]);
              //see if the waveform is finished
              npts := ProbeWin.Setup.Probe[probeid].NPtsPerChan;
              if CRPointCount[probeid] > npts-1 then
              begin //write the buffer CR to the ring
                With CRRingArray[CRRingWriteIndex] do
                begin
                  Time := CRTempRecordArray[probeid].Time;
                  ProbeNum := CRTempRecordArray[probeid].ProbeNum;
                  SetLength(Waveform,npts);
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
                if Infowin.UpdateCheck.Checked then
                  InfoWin.CRBuffersAcquired.Caption := IntToStr(nrawcrbuffers);
              end;
            end;
          end;
          inc(CRSkipCount[probeid]);
        end else
        if abs(Threshold[cglindex]-2048) > 10 then
        begin  //its a spike so see if it passes threshold
          param := wvalue;//RawRingArray[CheckRawRingArrayIndex+s];
          if Threshold[cglindex]-2048 > 0
            then begin if Param > Threshold[cglindex] then GrabWave; end
            else if Param < Threshold[cglindex] then GrabWave;
        end;
        inc(s);
      end;
    end;
  end;

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
  w,npts : integer;
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
        subtype := 'S';
        time_stamp := SpikeRingArray[SpikeRingSaveReadIndex].Time;
        probe := SpikeRingArray[SpikeRingSaveReadIndex].ProbeNum;
        cluster := 0;
        npts := ProbeWin.Setup.Probe[probe].NChannels * ProbeWin.Setup.Probe[probe].NPtsPerChan;
        adc_waveform := nil;
        SetLength(adc_waveform,npts);
        For w := 0 to npts-1 do
          adc_waveform[w] := SpikeRingArray[SpikeRingSaveReadIndex].Waveform[w];
      end;
      SaveSURF.PutPolytrodeRecord(PTRecord);
      //PTRecord.adc_waveform := NIL;
      inc(nsavedspikes);
      SpikeRingSaveReadIndex := (SpikeRingSaveReadIndex + 1) mod SPIKERINGARRAYSIZE;
      if Infowin.UpdateCheck.Checked then
        InfoWin.SpikesSaved.Caption := IntToStr(nsavedspikes);
      //InfoWin.SaveReadGuage.Progress := SpikeRingSaveReadIndex;
    end;
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
        SetLength(adc_waveform,ProbeWin.Setup.Probe[probe].NPtsPerChan);
        For w := 0 to ProbeWin.Setup.Probe[probe].NPtsPerChan-1 do
          adc_waveform[w] := CRRingArray[CRRingSaveReadIndex].Waveform[w];
      end;
      SaveSURF.PutPolytrodeRecord(PTRecord);
      //This may not be a good idea because ptrecord is passed as var
      //PTRecord.adc_waveform := NIL;
      inc(nsavedcrbuffers);
      CRRingSaveReadIndex := (CRRingSaveReadIndex + 1) mod CRRINGARRAYSIZE;
      if Infowin.UpdateCheck.Checked then
        InfoWin.CRBuffersSaved.Caption := IntToStr(nsavedcrbuffers);
      //InfoWin.CRSaveReadGuage.Progress := CRRingSaveReadIndex;
    end;
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
        if Infowin.UpdateCheck.Checked then
          InfoWin.DinBuffersSaved.Caption := IntToStr(nsaveddinbuffers);
    end;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForSpikesToDisplay;
var c,ch,w,diff,drop,startchan,stopchan,numchan,npts : LNG;
begin
  //This routine checks the spike buffer and can handle spikes up to 2000Hz
  if ProbeWin.Setup.Probe[SpikeRingArray[SpikeRingDisplayReadIndex].probenum].View = FALSE then
    SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + 1) mod SPIKERINGARRAYSIZE
  else
  begin
    diff := SpikeRingWriteIndex - SpikeRingDisplayReadIndex;
    If diff <> 0 then
    begin
      If diff > 20 then {skip some spikes}
      begin
        drop := diff div 5;//drop 20 percent of the spikes
        SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + drop) mod SPIKERINGARRAYSIZE;
        inc(ndroppedspikes,drop);
        exit;
      end;
      If diff < 0 then {writer ptr has wrapped around}
        if diff + SPIKERINGARRAYSIZE > 20 then
        begin
          drop := (SPIKERINGARRAYSIZE + diff) div 5;//drop 20 percent of the spikes
          SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + drop) mod SPIKERINGARRAYSIZE;
          inc(ndroppedspikes,drop);
          exit;
        end;
      With SpikeRingArray[SpikeRingDisplayReadIndex] do
      begin
        if ProbeWin.Setup.Probe[ProbeNum].NChannels = 1 then
          ChanForm[ChannelWinId[probenum,0]].win.PlotWaveform(Waveform,2{green},FALSE{overlay}{,FALSE}{fast draw})
        else
        begin
          startchan := ProbeWin.Setup.Probe[ProbeNum].ChanStart;
          numchan   := ProbeWin.Setup.Probe[ProbeNum].NChannels;
          stopchan  := ProbeWin.Setup.Probe[ProbeNum].ChanEnd;
          npts      := ProbeWin.Setup.Probe[ProbeNum].NPtsPerChan;
          SetLength(SingleWave,npts);
          For c := startchan to stopchan do
          begin
            ch := c-startchan;
            for w := 0 to npts-1 do
              SingleWave[w] := Waveform[ch+w*numchan];
            ChanForm[ChannelWinId[probenum,ch]].win.PlotWaveform(SingleWave,2{green},FALSE{overlay}{,FALSE}{fastdraw});
          end;
        end;
        Application.ProcessMessages;//don't tie up the timer
        inc(ndisplayedspikes);

        SpikeRingDisplayReadIndex := (SpikeRingDisplayReadIndex + 1) mod SPIKERINGARRAYSIZE;
        if Infowin.UpdateCheck.Checked then
          InfoWin.SpikesDisplayed.Caption := IntToStr(ndisplayedspikes);
        //InfoWin.DisplayReadGuage.Progress := SpikeRingDisplayReadIndex;
      end;
    end;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.CheckForCRRecordsToDisplay;
var diff,drop : LNG;
begin
  if ProbeWin.Setup.Probe[CRRingArray[CRRingDisplayReadIndex].probenum].View = FALSE then
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
        ChanForm[ChannelWinId[probenum,0]].win.PlotWaveform(Waveform,4{yellow},FALSE{overlay}{,FALSE}{fastdraw});
      Application.ProcessMessages;//don't tie up the timer
      inc(ndisplayedcrbuffers);
      CRRingDisplayReadIndex := (CRRingDisplayReadIndex + 1) mod CRRINGARRAYSIZE;
      if Infowin.UpdateCheck.Checked then
        InfoWin.CRBuffersDisplayed.Caption := IntToStr(ndisplayedcrbuffers);
      //InfoWin.CRDisplayReadGuage.Progress := CRRingDisplayReadIndex;
    end;
  end;
end;

{==============================================================================}
Function TSurfAcqForm.ConfigBoard : boolean; //false if error
var i : integer;
 hbuf : HBUFTYPE;
begin
  ConfigBoard := TRUE;

  //Set channel list
  DTAcq.ListSize := NumChans;
  For i := 0 to NumChans-1 do
  begin
    DTAcq.ChannelList[i] := ProbeChannelList[CHANINDEX,i];
    if (i=NumChans-1) and ProbeWin.DinCheckBox.Checked
      then DTAcq.GainList[i] := 1
      else DTAcq.GainList[i] := GainTrigLockList[GAININDEX,ProbeChannelList[PROBEINDEX,i]];
  end;

  //Set acquisition frequency
  DTAcq.Frequency := SampFreq*NumChans; //set the clock frequency in Hz

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
      ShowMessage(DTDIO.Board+ ' can not support singlevalue output');
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
  if Recording then RecordingBut.Caption := 'Start Recording';
  SetupProbes;
  if not ProbeWin.ok then exit;
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
var vf : single;
    vs : string;
    i,j,l,t,cglindex,chanwinheight,nc : integer;
    LastProbeChannelList : ProbeChannelListType;
    multichannelprobes : boolean;
    LastSetup : ProbeSetupRec;
begin
  Move(ProbeWin.Setup,LastSetup,sizeof(ProbeSetupRec));//backup old setup

  ProbeWin.TotFreq.MinValue := MinFreq;
  ProbeWin.TotFreq.MaxValue := MaxFreq;
  ProbeWin.SampFreqPerChan.MaxValue := ProbeWin.TotFreq.MaxValue;
  ProbeWin.SampFreqPerChan.MinValue := ProbeWin.TotFreq.MinValue;

  UnloadBuffers;

  FillChar(config.chanloc,sizeof(config.chanloc),0);
  if ProbeWin.DinCheckBox.Checked
    then nc :=  NumChans-1
    else nc :=  NumChans;

  For i := 0 to nc-1 do if ChanForm[i].exists then
  begin
    config.chanloc[i].left := ChanForm[i].win.Left;
    config.chanloc[i].top := ChanForm[i].win.top;
    config.chanloc[i].width := ChanForm[i].win.width;
    config.chanloc[i].height := ChanForm[i].win.height;
    config.chanloc[i].chantype := ProbeWin.Setup.Probe[ProbeChannelList[PROBEINDEX,i]].Probetype;
  end;
  Move(ProbeChannelList,LastProbeChannelList,sizeof(ProbeChannelListType));

  if NumChans > 0 then FreeChanWindows;

  //show the probewin
  ProbeWin.ShowModal;

  NumChans := ProbeWin.Setup.TotalChannels;
  NumProbes := ProbeWin.Setup.NProbes;
  SampFreq := ProbeWin.SampFreqPerChan.Value;
  If (NumProbes=0) or (NumChans=0) then ProbeWin.ok := FALSE;

  if not ProbeWin.ok then Move(LastSetup,ProbeWin.Setup,sizeof(ProbeSetupRec));

  if Probewin.ok then
  begin
    rawringarray := NIL;
    nc := NumChans;
    if nc < 8 then nc := 8;
    BuffSize := 32767 div nc;
    NumBuffs := 3 + round(sqrt((SampFreq*NumChans) / buffsize));
    RawRingArraySize := {U}LNG(BuffSize*NumBuffs*NumChans);

    SetLength(rawringarray,RawRingArraySize);

    //Build the probechannelgainlist array
    cglindex := 0;
    For i := 0 to NumProbes-1 do
    begin
      GainTrigLockList[GAININDEX,i] := ProbeWin.Setup.Probe[i].InternalGain;
      GainTrigLockList[TRIGINDEX,i] := ProbeWin.Setup.Probe[i].TrigPt;
      GainTrigLockList[LOCKINDEX,i] := ProbeWin.Setup.Probe[i].Lockout;
      For j := 0 to ProbeWin.Setup.Probe[i].NChannels-1 do
      begin
        ProbeChannelList[PROBEINDEX,cglindex] := i;
        ProbeChannelList[CHANINDEX,cglindex] := ProbeWin.Setup.Probe[i].ChanStart + j;
        inc(cglindex);
      end;
    end;

    if ProbeWin.DinCheckBox.Checked then
    begin
      ProbeChannelList[PROBEINDEX,cglindex{the last entry}] := DINPROBE{which is 32};
      ProbeChannelList[CHANINDEX,cglindex] := 32;//the digital-in port
    end;

    //Setup the acquisition and plot objects
    if not ConfigBoard then
    begin
      ShowMessage('Error: Board not configured properly');
      exit;
    end;

    //Draw the windows to the screen
    multichannelprobes := FALSE;
    For i := 0 to NumProbes-1 do
      if ProbeWin.Setup.Probe[i].NChannels > 1 then
        multichannelprobes := TRUE;
    t := 0;//'ToolPanel.Height;
    l := 0;
    cglindex := 0;
    chanwinheight := 0;
    For i := 0 to NumProbes-1 do
    begin
      For j := 0 to ProbeWin.Setup.Probe[i].NChannels-1 do
      begin
        CreateAChanWindow(cglindex,i,j,l,t,ProbeWin.Setup.Probe[i].NPtsPerChan);
        ChannelWinId[i,j] := cglindex;
        chanwinheight := ChanForm[cglindex].win.Height+1;
        inc(l,ChanForm[cglindex].win.Width+1);
        if l > ClientWidth then
        begin
          l := 0;
          inc(t,chanwinheight);
        end;
        inc(cglindex);
      end;
      if multichannelprobes then
      begin
        l := 0;
        inc(t,chanwinheight);
      end;
    end;
    if not SetupPlotWins
    then begin
      ShowMessage('Error: Plot windows not configured properly');
      exit;
    end;

    if ProbeWin.DinCheckBox.Checked
      then nc :=  NumChans-1
      else nc :=  NumChans;
    For i := 0 to nc-1 do
    begin
      DTAcq.GainList[i] := GainTrigLockList[GAININDEX,ProbeChannelList[PROBEINDEX,i]];
      vf := 10 / DTAcq.GainList[i];
      vs := FloatToStr(vf);

      ChanForm[i].win.HiVolt.Caption := '+'+ vs + 'V';
      ChanForm[i].win.LoVolt.Caption := '-'+ vs + 'V';

      if ChanForm[i].win.MarkerV.Visible then
        ChanForm[i].win.MarkerV.Left := ChanForm[i].win.plot.left + GainTrigLockList[TRIGINDEX,ProbeChannelList[PROBEINDEX,i]];
      //here, if no change in chan windows, then set them back to where they were
      //If this channel belongs to the same probe as before, has the same channel id,
      //and the number of pts per channel has not changed then put it in the same place
      if (LastProbeChannelList[PROBEINDEX,i] = ProbeChannelList[PROBEINDEX,i])
      and(LastProbeChannelList[CHANINDEX,i] = ProbeChannelList[CHANINDEX,i])
      and(LastSetup.probe[ProbeChannelList[PROBEINDEX,i]].NPtsPerChan =
          ProbeWin.Setup.probe[ProbeChannelList[PROBEINDEX,i]].NPtsPerChan) then
      begin
        ChanForm[i].win.Left := config.chanloc[i].left;
        ChanForm[i].win.top := config.chanloc[i].top;
        ChanForm[i].win.width := config.chanloc[i].width;
        ChanForm[i].win.height := config.chanloc[i].height;
        ProbeWin.Setup.Probe[ProbeChannelList[PROBEINDEX,i]].Probetype := config.chanloc[i].chantype;
      end else
      begin
        config.chanloc[i].left := ChanForm[i].win.Left;
        config.chanloc[i].top := ChanForm[i].win.top;
        config.chanloc[i].width := ChanForm[i].win.width;
        config.chanloc[i].height := ChanForm[i].win.height;
        config.chanloc[i].chantype := ProbeWin.Setup.Probe[ProbeChannelList[PROBEINDEX,i]].Probetype;
      end;
    end;
    AllChannelsExist := TRUE;
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
Procedure TChannelObj.ThreshChange(pid,cid : integer; ShiftDown,CtrlDown : boolean);
var id2,startid,stopid,nc,y : integer;
begin
  if ProbeWin.DinCheckBox.Checked
    then nc :=  SurfAcqForm.NumChans-1
    else nc :=  SurfAcqForm.NumChans;

  if ShiftDown then  //Set threshold across all chans of this probe
  begin
    Startid := SurfAcqForm.ChannelWinId[pid,0];
    Stopid :=  SurfAcqForm.ChannelWinId[pid,ProbeWin.Setup.Probe[pid].NChannels-1];
  end else
  begin
    if CtrlDown then //Set threshold across all chans of all probes
    begin
      Startid := 0;
      Stopid := nc-1;
    end else
    begin
      Startid := cid;
      Stopid := cid;
    end;
  end;

  For id2 := Startid to Stopid do
  With ProbeWin.Setup.Probe[SurfAcqForm.ProbeChannelList[PROBEINDEX,id2]] do
  begin
    if ProbeType <> CONTINUOUSTYPE then
    begin
      Threshold := 2047-SurfAcqForm.ChanForm[cid].win.SlideBar.Position;
      y := round(2047+threshold);
      if y < 0 then y := 0;
      if y > Length(Screeny)-1 then y := Length(Screeny)-1;
      SurfAcqForm.ChanForm[id2].win.MarkerH.Top := Screeny[y];
      SurfAcqForm.ChanForm[id2].win.Threshold.Caption := InTToStr(threshold);
      SurfAcqForm.ChanForm[id2].win.SlideBar.Position := SlideBar.Position;
      SurfAcqForm.ChanForm[id2].win.Refresh;
    end;
  end;
end;

{==============================================================================}
procedure TSurfAcqForm.RecButClick(Sender: TObject);
begin
  StartStopRecording;
end;

{==============================================================================}
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
  RecordingBut.Caption := 'Start Recording';
end;

{==============================================================================}
procedure TSurfAcqForm.SaveSurfConfig(Filename : String);
var fs : TFileStream;
    i,nc :integer;
begin
   GotCfgFileName := FALSE;
   CfgFileName := Filename;
   FileInfoPanel.Panels[1].Text := 'Config File: '+ CfgFileName;
   Config.NumProbes := NumProbes;
   Config.NumChannels := NumChans;

   Move(ProbeChannelList,Config.ProbeChannelList,sizeof(ProbeChannelListType));
   Move(GainTrigLockList,Config.GainTrigLockList,sizeof(GainTrigLockListType));
   Move(ChannelWinId,Config.ChannelWinId,sizeof(ChannelWinIdType));

   Move(ProbeWin.Setup,Config.Setup,sizeof(ProbeSetupRec));
   //Config.Setup.probe[0].Descrip := ProbeWin.Setup.Probe[0].Descrip;

   Config.MainWinLeft  :=Left;
   Config.MainWinTop   :=Top;
   Config.MainWinHeight:=Height;
   Config.MainWinWidth :=Width;
   Config.InfoWinLeft  :=InfoWin.Left;
   Config.InfoWinTop   :=InfoWin.Top;
   Config.InfoWinHeight:=InfoWin.Height;
   Config.InfoWinWidth :=InfoWin.Width;
   Config.DinChecked   := ProbeWin.DinCheckBox.Checked;

   try
     fs := TFileStream.Create(CfgFileName,fmCreate);
     fs.WriteBuffer('SCFG',4);

     FillChar(config.chanloc,sizeof(config.chanloc),0);
     if ProbeWin.DinCheckBox.Checked
       then nc :=  NumChans-1
       else nc :=  NumChans;
     if nc > 0 then
     For i := 0 to nc-1 do
     begin
       config.chanloc[i].left := ChanForm[i].win.Left;
       config.chanloc[i].top := ChanForm[i].win.top;
       config.chanloc[i].width := ChanForm[i].win.width;
       config.chanloc[i].height := ChanForm[i].win.height;
       config.chanloc[i].chantype := ProbeWin.Setup.Probe[ProbeChannelList[PROBEINDEX,i]].Probetype;
     end;
     fs.WriteBuffer(config,sizeof(config));
     GotCfgFileName := TRUE;
   Except
     ShowMessage('Failure to write '+CfgFileName);
     exit;
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
    i,j,nc,cglindex :integer;
    vf : single;
    vs : string;
    headerstring : array[0..3] of char;
begin
  ok := TRUE;
  GotCfgFileName := FALSE;
  try
    fs := TFileStream.Create(FileName,fmOpenRead);
    CfgFileName := FileName;
    FileInfoPanel.Panels[1].Text := 'Config File: '+ CfgFileName;

    fs.ReadBuffer(headerstring,4);
    if headerstring <> 'SCFG' then
    begin
      ShowMessage('This is not a Surf Configuration Record');
      ok := FALSE;
      exit;
    end;
    rawringarray := NIL;
    UnloadBuffers;

    if NumChans > 0 then FreeChanWindows;

    fs.ReadBuffer(config,sizeof(config));
    left := Config.MainWinLeft;
    Top := Config.MainWinTop;
    Height := Config.MainWinHeight;
    Width := Config.MainWinWidth;
    InfoWin.Left := Config.InfoWinLeft;
    InfoWin.Top := Config.InfoWinTop;
    InfoWin.Height := Config.InfoWinHeight;
    InfoWin.Width := Config.InfoWinWidth;

    NumProbes := Config.NumProbes;
    NumChans := Config.NumChannels;
    Move(Config.ProbeChannelList,ProbeChannelList,sizeof(ProbeChannelListType));
    Move(Config.GainTrigLockList,GainTrigLockList,sizeof(GainTrigLockListType));
    Move(Config.ChannelWinId,ChannelWinId,sizeof(ChannelWinIdType));
    Move(Config.Setup,ProbeWin.Setup,sizeof(ProbeSetupRec));

    ProbeWin.NSpikeProbeSpin.Value := ProbeWin.Setup.NSpikeProbes;
    ProbeWin.NCRProbesSpin.Value := ProbeWin.Setup.NCRProbes;

    ProbeWin.CreateProbeRows;

    ProbeWin.DinCheckBox.Checked := Config.DinChecked;

    SampFreq := ProbeWin.SampFreqPerChan.Value;
    nc := NumChans;
    if nc < 8 then nc := 8;
    BuffSize := 32767 div nc;
    NumBuffs := 3 + round(sqrt((SampFreq*NumChans) / buffsize));
    RawRingArraySize := {U}LNG(BuffSize*NumBuffs*NumChans);
    rawringarray := nil;
    SetLength(rawringarray,RawRingArraySize);

    if not ConfigBoard then
    begin
      ShowMessage('Error: Board not configured properly');
      ok := FALSE;
      exit;
    end;

    //Draw the windows to the screen
    cglindex := 0;
    For i := 0 to NumProbes-1 do
      For j := 0 to ProbeWin.Setup.Probe[i].NChannels-1 do
      begin
        With config.chanloc[cglindex] do
        begin
          CreateAChanWindow(cglindex,i,j,left,top,ProbeWin.Setup.Probe[i].NPtsPerChan);
          ChanForm[cglindex].win.Width := width;
          ChanForm[cglindex].win.Height := height;
        end;
        inc(cglindex);
      end;
    fs.Free;
    GotCfgFileName := TRUE;

    if not SetupPlotWins
    then begin
      ShowMessage('Error: Plot windows not configured properly');
      ok := FALSE;
      exit;
    end;
    if ProbeWin.DinCheckBox.Checked
      then nc :=  NumChans-1
      else nc :=  NumChans;
    For i := 0 to nc-1 do
    begin
      DTAcq.GainList[i] := GainTrigLockList[GAININDEX,ProbeChannelList[PROBEINDEX,i]];
      vf := 10 / DTAcq.GainList[i];
      vs := FloatToStr(vf);
      ChanForm[i].win.HiVolt.Caption := '+'+ vs + 'V';
      ChanForm[i].win.LoVolt.Caption := '-'+ vs + 'V';
      if ChanForm[i].win.MarkerV.Visible then
        ChanForm[i].win.MarkerV.left := ChanForm[i].win.plot.left + GainTrigLockList[TRIGINDEX,ProbeChannelList[PROBEINDEX,i]];
    end;
    AllChannelsExist := TRUE;
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

{=============  USER ACCESS FUNCTIONS   =======================================}
(*Function TSurfAcqForm.GetNumChans : Integer;
begin
  GetNumChans := NumChans;
end;

Function TSurfAcqForm.GetNumProbes : Integer;
begin
  GetNumProbes := NumProbes;
end;

Function TSurfAcqForm.GetTime : TheTimeRec;
begin
  GetTime := TheTime;
end;

Function TSurfAcqForm.GetSpikeRingWriteIndex : Longint;
begin
  GetSpikeRingWriteIndex := SpikeRingWriteIndex-1;
end;

Function TSurfAcqForm.GetCRRingWriteIndex : Longint;
begin
  GetCRRingWriteIndex := CRRingWriteIndex-1;
end;

Function TSurfAcqForm.GetDINRingWriteIndex : Longint;
begin
  GetDINRingWriteIndex := DINRingWriteIndex-1;
end;

Procedure TSurfAcqForm.GetSpikeFromRing(index : longint; var time,probenum : longint);
var ind : integer;
begin
  ind := (index+SpikeRingArraySize) mod SpikeRingArraySize;
  time := SpikeRingArray[ind].time;
  probenum := SpikeRingArray[ind].probenum;
end;

Function TSurfAcqForm.GetCRFromRing(index : longint) : PolytrodeRecord;
begin
  GetCRFromRing := CRRingArray[(index+CRRingArraySize) mod CRRingArraySize];
end;

Function TSurfAcqForm.GetDINFromRing(index : longint) : DinRecordType;
begin
  GetDINFromRing := DINRingArray[(index+DINRingArraySize) mod DINRingArraySize];
end;

Function TSurfAcqForm.GetChanStartNumberForProbe(probenum : integer) : integer;
begin
  if (probeNum >= 0) and (probenum < NumProbes)
    then GetChanStartNumberForProbe := ProbeWin.Setup.Probe[ProbeNum].ChanStart
    else GetChanStartNumberForProbe := -1;
end;

Function TSurfAcqForm.GetNumChansForProbe(probenum : integer) : integer;
begin
  if (probeNum >= 0) and (probenum < NumProbes)
    then GetNumChansForProbe := ProbeWin.Setup.Probe[ProbeNum].NChannels
    else GetNumChansForProbe := -1;
end;

Function TSurfAcqForm.GetAcquisitionState : boolean;
begin
  GetAcquisitionState := Acquisition;
end;
Function TSurfAcqForm.GetRecordingState : boolean;
begin
  GetRecordingState := Recording;
end;
Function TSurfAcqForm.GetFileOpenState : boolean;
begin
  GetFileOpenState := FileIsOpen;
end;
*)
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
  InfoWin.Show;

  SaveSURF := TSurfFile.Create;
  FileInfoPanel.Panels[0].Width := round(length(FileInfoPanel.Panels[0].Text)*FileInfoPanel.Font.Size* 2/3);

  TheTime.TenthMS := 0;//10th ms

  //Initialize numchans and RawRingArraySize for memory allocation
  NumChans := 0;
  RawRingArraySize := 0;
  SetLength(rawringarray,RawRingArraySize);
  SampFreq := 32000;

  GotCfgFileName := FALSE;
  AllChannelsExist := FALSE;
  CfgFileName := '';
  if not InitBoard then Close;
  FillChar(config.chanloc,sizeof(config.chanloc),0);
end;

procedure TSurfAcqForm.FormHide(Sender: TObject);
begin
  if Acquisition then StartStopAcquisition;
  UnloadBuffers;
  If Recording then SaveSURF.CloseFile;
  FreeChanWindows;
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


procedure TSurfAcqForm.stimerTimer(Sender: TObject);
begin
  //this is a 1 ms timer
  UpdateTimer;
  CheckForSpikesToSave;
  CheckForSvRecordsToSave;

  if TheTime.MS mod 10 = 0 {every 10 ms} then
    CheckForSpikesToDisplay;
  if TheTime.MS mod 100 = 0 {every 100 ms} then
  begin
    CheckForCRRecordsToSave;
    CheckForCRRecordsToDisplay;
  end;
  if TheTime.Sec <> TheTime.LastSec then
  begin
    if Infowin.UpdateCheck.Checked then
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

end.

