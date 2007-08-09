{ (c) 2002-2004 Tim Blanche, University of British Columbia }
unit SurfBawdMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  SurfFileAccess, ElectrodeTypes, SurfPublicTypes, WaveFormPlotUnit, SurfMathLibrary,
  RasterPlotUnit, InfoWinUnit, ExtCtrls, StdCtrls, ComCtrls, ToolWin, ImgList, Math,
  Spin, PolytrodeGUI, ChartFormUnit, TemplateFormUnit, HistogramUnit, FileProgressUnit,
  ClipBrd {temporary, move to chartwin};

const
  ParamNames : array[0..7] of string[20] = ('Channel',  'Amplitude', 'Peak',
                         'Valley', 'TimeAtPeak', 'Width', 'Area', 'Polarity');
  V2uV = 1000000;
  //LockoutRadius = 105{µm};
  DefaultSampFreq = 25000; //normally read from file
  DefaultMasterClockFreq = 1000000; // "     "    "
  DefaultUpsampleFactor = 4;
  SincKernelLength = 15;
  TransBufferSamples = 25; {raw data points}

type

  CProbeWin = class(TWaveFormPlotForm)
    public
      procedure NotAcqOnMouseMove(ChanNum : byte); override;
      procedure ThreshChange(pid, threshold : integer); override;
    end;

  CChartWin = class(TChartWin)
    public
      procedure MoveTimeMarker(MouseXPercent : Single); override;
      procedure RefreshChartPlot; override;
    end;

  CTemplateWin = class(TTemplateWin)
    public
      procedure BuildTemplates; override;
      procedure ChangeTab(TabIndex : integer); override;
      procedure ReloadSpikeSet; override;
      procedure DeleteTemplate(TemplateIndex : integer); override;
      procedure CombineTemplates(TemplateIdxA, TemplateIdxB : integer); override;
      procedure SplitTemplate(TemplateIndex : integer); override;
      procedure ToggleClusterLock(TemplateIndex : integer); override;
  end;

  CHistogramWin = class(THistogramWin)
    public
      procedure MoveGUIMarker(MouseX : Single); override;
    end;

  CISIHistoWin = class(THistogramWin)
    public
      procedure UpdateISIHistogram; override;
    end;

  TProbeWin = record
    Win : CProbeWin;
    Electrode : TElectrode;
    Exists : boolean;
    DispTrigOffset, TrigChan, LastTrigChan : integer;
    DispTrigBipolar, DispTrigPositive : boolean;
  end;

(*  TTemplate = record
    Sites : TSites //set of channels in this template
    ??Win : CProbeWin?
    ??Electrode : TElectrode;
    Exists : boolean;
  end;

  TTemplateArray : array of TTemplate; *)


  TSurfBawdForm = class(TForm)
    FileStatsPanel: TPanel;
    Label1: TLabel;
    Label2: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    DataFileSize: TLabel;
    FileNameLabel: TLabel;
    NProbes: TLabel;
    NStim: TLabel;
    NSpikes: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    NMsgs: TLabel;
    NSVals: TLabel;
    SurfFile: TSurfFileAccess;
    CElectrode: TComboBox;
    Waveforms: TPanel;
    StatusBar: TStatusBar;
    TrackBar: TTrackBar;

    ToolBar1: TToolBar;
    SmlButtonImages: TImageList;

    ToolBar2: TToolBar;
    LrgButtonImages: TImageList;
    tbStepBack: TToolButton;
    tbReversePlay: TToolButton;
    tbStop: TToolButton;
    tbPlay: TToolButton;
    tbStepFoward: TToolButton;
    tbPause: TToolButton;
    tbExportData: TToolButton;
    tbExport2File: TToolButton;
    ToolButton3: TToolButton;
    ToolBar3: TToolBar;
    tbToglWform: TToolButton;
    tbToglStatsWin: TToolButton;
    tbRasterPlot: TToolButton;
    DisplayToggleImages: TImageList;
    GroupBox1: TGroupBox;
    SpinEdit1: TSpinEdit;
    cbuV: TCheckBox;
    cbHeader: TCheckBox;
    MsgPanel: TPanel;
    MsgMemo: TMemo;
    SampleRate: TLabel;
    RecTimeLabel: TLabel;
    Label11: TLabel;
    ExportDataDialog: TSaveDialog;
    tbToglProbeGUI: TToolButton;
    cbThreshFilter: TCheckBox;
    cbExportTimes: TCheckBox;
    GroupBox2: TGroupBox;
    Button1: TButton;
    Label9: TLabel;
    seLockRadius: TSpinEdit;
    PlayTimer: TTimer;
    cbDetectRaw: TCheckBox;
    GroupBox3: TGroupBox;
    seFactor: TSpinEdit;
    Label10: TLabel;
    cbSHcorrect: TCheckBox;
    rgThreshold: TRadioGroup;
    cbPCAClean: TCheckBox;
    cbAlignData: TCheckBox;
    lblUpsample: TLabel;
    Label4: TLabel;
    tbChartWin: TToolButton;
    tbTemplateWin: TToolButton;
    tbFindTemplates: TToolButton;
    tbToglHistWin: TToolButton;
    seLockTime: TSpinEdit;
    GroupBox4: TGroupBox;
    btReset: TButton;
    lbSpikeCount: TLabel;
    cbExportSVALs: TCheckBox;
    cbExportEEG: TCheckBox;
    CheckBox1: TCheckBox;
    Button2: TButton;
    ClusterLog: TMemo;
    Button3: TButton;
    tbToglISIHist: TToolButton;
    Button4: TButton;
    Button5: TButton;
    Button6: TButton;
    Button7: TButton;
    Button8: TButton;
    Button9: TButton;
    cbDecimate: TCheckBox;
    SpinEdit2: TSpinEdit;
    Label3: TLabel;

    procedure SurfFileNewFile(acFileName: WideString);
    procedure CElectrodeChange(Sender: TObject);
    procedure TrackBarChange(Sender: TObject);
    procedure TrackBarKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure TrackBarKeyUp(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure UpdateStatusBar;
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure tbPlayClick(Sender: TObject);
    procedure tbStopClick(Sender: TObject);
    procedure tbStepBackClick(Sender: TObject);
    procedure tbStepFowardClick(Sender: TObject);
    procedure StatusBarDblClick(Sender: TObject);
    procedure tbRethresholdClick(Sender: TObject);
    procedure tbToglStatsWinClick(Sender: TObject);
    procedure tbToglWformClick(Sender: TObject);
    procedure tbLocNSortClick(Sender: TObject);
    procedure tbRasterPlotClick(Sender: TObject);
    procedure tbExportDataClick(Sender: TObject);
    procedure SpinEdit1Change(Sender: TObject);
    procedure SpinEdit2Change(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure tbToglProbeGUIClick(Sender: TObject);
    procedure cbThreshFilterClick(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure seLockRadiusChange(Sender: TObject);
    procedure tbPauseClick(Sender: TObject);
    procedure PlayTimerTimer(Sender: TObject);
    procedure seFactorChange(Sender: TObject);
    procedure cbSHcorrectClick(Sender: TObject);
    procedure cbDetectRawClick(Sender: TObject);
    procedure tbChartWinClick(Sender: TObject);
    procedure tbTemplateWinClick(Sender: TObject);
    procedure tbFindTemplatesClick(Sender: TObject);
    procedure tbToglHistWinClick(Sender: TObject);
    procedure seLockTimeChange(Sender: TObject);
    procedure btResetClick(Sender: TObject);
    procedure MsgMemoKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure MsgMemoMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure MsgMemoMouseUp(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure Button2Click(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure FormKeyUp(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure tbExport2FileClick(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure tbToglISIHistClick(Sender: TObject);
    procedure Button4Click(Sender: TObject);
    procedure Button5Click(Sender: TObject);
    procedure Button6Click(Sender: TObject);
    procedure Button7Click(Sender: TObject);
    procedure Button8Click(Sender: TObject);
    procedure Button9Click(Sender: TObject);
  private
    { Private declarations }

    BinFileLoaded : boolean; //temporary HACK for loading simulated bin files into surfbawd
    WholeBinFile  : TWaveform ;

    NumSpikes2Save : integer;

    m_UpSampleFactor  : integer;
    m_ConcatRawBuffer : TWaveform;
    m_InterpWaveform  : TWaveform;
    m_SincKernelArray : array{shdelay} of array{interp} of TReal32Array{kernel};
    Cat9SpecialSincs : boolean; //temporary flag

    FileNameOnly, Panel1Str : string;
    ChanWithMaxAmp: Byte;
    SplineX, SplineY, Spline2ndDeriv : TSplineArray;

    WaveformStatsWin : TInfoWin;
    RasterWin : TRasterForm;
    GUIForm  : TPolytrodeGUIForm;
    FitHistoWin : CHistogramWin;
    ISIHistoWin : CISIHistoWin;
    ChartWin : CChartWin;
    TemplWin : CTemplateWin;

    m_SpikeCount : integer;

    m_recStartTime : int64;
    m_SelStartCRIdx, m_SelEndCRIdx, m_SelStimHdrIdx, m_SelStartSValIdx, m_SelEndSValIdx : integer;
    m_ExportStream : TFileStream{64};
    FileProgressWin : TFileProgressWin;

    ShiftKeyDown, DisplayStatusBar, GUICreated, ChartWinCreated, TemplateWinCreated,
      FitHistoWinCreated, ISIHistoWinCreated, BuildTemplateMode, ShowTemplateMode, Exporting,
      ESCPressed : boolean;

    Clusters   : TClusterArray;
    binCluster : TCluster;
    SpikeSet   : TWaveformArray;
    //AmpSet     : array of array of integer;
    SpikeTimes : array of int64;
    NClust, NSamples, NClustDims, NDesiredClust, NSamplesMoved, NkIterations,
    OutliersDeleted, ClustError : integer;

    m_residual : array of cardinal{int64}; //stores residuals from fit of active template to current buffer

    procedure AddElectrodeTypes;
    procedure CreateProbeWin(const Electrode : TElectrode);
    function  CreateTemplateWin : boolean; //success/failure
    procedure ComputeWaveformParams(var Spike : TSpike; Precision : boolean = true;
                                                OneChanOnly : shortint = -1);
    procedure AddRemoveProbeGUI;
    procedure AddRemoveFitHistogram;
    procedure AddRemoveISIHistogram;
    procedure WaveFormStatsUpdate;
    procedure PredNSort;

    procedure UpdateTemplates;
    procedure ShowHideTemplates;
    procedure RipFileWithTemplates;
    procedure FitTemplate(TemplateIndex : integer);
    procedure OverplotTemplateFits;
    procedure ExportSurfData(Filename : string);
    procedure ExportEventData(Filename : string);
    procedure DisplaySpikeFromStream(increment : integer = 1);
    procedure BuildSiteProximityArray(var SiteArray : TSiteArray; radius : integer;
                                      inclusive : boolean = False);
    procedure ResetChanLocks;
    procedure GenerateSincKernels;
    procedure GetFileBuffer;
    procedure UpSampleWaveform(const RawWaveform : TWaveform;
                               const UpSampledWaveform : TWaveform;
                               NPtsInterpolation : integer);
    function TimeStamp2Str(TimeStamp : int64) : string;
    function FindNextThresholdX(const Waveform : TWaveform{CRRecord : TCR}) : boolean;
    function ADValue2uV(ADValue : SmallInt): SmallInt;
    function FindEvent(Time : int64; BeforeExactAfter : TEventTime;
                       EventType : Char; EventSubtype : Char = ' ') : integer;


    procedure InitialiseClusters;
    function  BuildSpikeSampleSet : boolean;
    procedure BinarySplitClusters;
 //   procedure ComputeCentroid(var Cluster : TCluster);
 //   procedure ComputeDistortion(var Cluster : TCluster);
    function MinCentroidDist(const ClusterArray : array of TCluster) : single;

    //procedure SplitCluster(var Cluster : TCluster);
    procedure kMeans(var ClusterArray : array of TCluster; k : integer);
    procedure IsoClus(SplitClusterIndex : integer = -1);
    procedure DeleteCluster(ClusterIndex : integer);
    procedure Clusters2Templates;
    procedure StupidHack2EnableDel; // !temporary, remove when deltem/clust procedure bugs fixed
    procedure SendClusterResultsToFile(SaveFilename : string);

  public
    { Public declarations }
    m_NEvents, m_NProbes, m_SumSpikes, m_SumCRs, m_NSVals, m_NMsgs, m_NStim : integer; //...in file
    m_NEpochs : array of integer; //...in probe[n], either spike- or buffer-epochs
    m_TStep : array of single;
    m_ProbeThreshold : array of integer;

    m_NearSites    : TSiteArray; //for each probe site, the set of neighbouring that comprise 'lockout domain'
    //m_LockedSites  : TSites; //contains the set of sites currently locked out
    m_PosLockedSamp : array [0..SURF_MAX_CHANNELS-1] of integer; //for each site, the number of +ve samples before lock disabled
    m_NegLockedSamp : array [0..SURF_MAX_CHANNELS-1] of integer; //for each site, the number of -ve samples before lock disabled
    m_SimpleLockSamples : integer;

    m_TrigIndex, m_SpikeMaxMinChan : integer;
    m_ProbeIndex : byte;
    m_TemplateIndex : integer;
    m_CurrentChan : byte;
    m_CurrentBuffer : integer;
    m_SpikeTimeStamp : array of lng;
    m_SpikeClusterID : array of smallint;

    m_HardwareInfo : THardwareInfo;

    m_EventArray : TSurfEventArray;

    Spike : TSpike; //current spike record (see SurfPublicTypes)
    ProbeWin : array of TProbeWin;
    ProbeArray : TProbeArray; //probe layouts for all probes in file
    CR, CR2, CR3 : TCR;
    SVal : TSVal;
    SurfMsg : TSurfMsg;
    StimHdr : TStimulusHeader;
    //put some of these in Private or even local?

    procedure ComputeCentroid(var Cluster : TCluster); // only here because combinecluster...
    procedure ComputeDistortion(var Cluster : TCluster); //procedure needs access to them...
  end;

var
  SurfBawdForm: TSurfBawdForm;

implementation

uses SurfLocateAndSort;
{$R *.DFM}

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.SurfFileNewFile(acFileName: WideString);
var
  p, e      : integer;
  StartRecTime, EndRecTime : TTimeStamp;
  fs : TFileStream;
const BinFileNBuffs = 45{*100ms}; //temporary hack
begin
  SurfBawdForm.BringToFront;
  StatusBar.Visible:= True;
  ShiftKeyDown:= False;
  WaveForms.Visible:= True;

  //initialise and display global file information
  FileNameOnly:= ExtractFileName(acFileName);
  Application.Title:= FileNameOnly;
  FileNameLabel.Caption:= FileNameOnly;
  FileNameLabel.Hint:= acFileName;
  DataFileSize.Caption:= FloattostrF(SurfFile.Get64FileSize / 1048576, ffNumber, 4, 1) + 'Mb';
  m_HardWareInfo:= SurfFile.GetHardwareInfo;
  if m_HardWareInfo.iADCRetriggerFreq = 0 then //use defaults
  begin
    m_HardWareInfo.iMasterClockFreq := DefaultMasterClockFreq;
    m_HardWareInfo.iADCRetriggerFreq:= DefaultSampFreq;
  end;

  // START SIMULATED FILE HACK //
  if ExtractFileExt(acFileName) = '.bin' then //hack code to read in simulated files
  begin
    Showmessage('Simulation .bin file!');
    BinFileLoaded:= True;
    m_HardWareInfo.iMasterClockFreq := DefaultMasterClockFreq;
    m_HardWareInfo.iADCRetriggerFreq:= DefaultSampFreq;
    DataFileSize.Caption:= FloattostrF(BinFileNBuffs * 1080000 / 1048576, ffNumber, 4, 1) + 'Mb';
    m_NProbes:= 1;
    m_NSVals:= 0;
    NSVals.Caption:= inttostr(m_NSVals);
    m_NMsgs:= 0;
    NMsgs.Caption:= inttostr(m_NMsgs);
    m_NStim:= 0;
    NStim.Caption:= inttostr(m_NStim);

    m_SumSpikes:= 0;
    m_SumCRs:= 0;
    SetLength(ProbeArray, m_NProbes);
    SetLength(ProbeWin, m_NProbes);
    SetLength(m_NEpochs, m_NProbes);
    SetLength(m_TStep, m_NProbes);
    SetLength(m_ProbeThreshold, m_NProbes);

    with ProbeArray[0] do
    begin
      ProbeSubType   := 'S';
      numchans       := 54;
      pts_per_chan   := 100;
      pts_per_buffer := 540000;
      trigpt         := 7;
      lockout        := 3;
      intgain        := 8;
      threshold      := 3000;
      skippts        := 1;
      sampfreqperchan:= 100000;
      sh_delay_offset:= 0;
      for p:= 0 to 53 do chanlist[p]:= p;
      //ProbeWinLayout : TProbeWinLayout;//= array[0..SURF_MAX_CHANNELS-1] of TPoint;
      probe_descrip  := 'Simulated data';
      electrode_name := 'µMap54_2b';
      CElectrode.Items.Add(electrode_name);
      CElectrode.ItemIndex:= 0;

      for p:= 0 to 53 do ExtGain[p]:= 5000;

      m_NEpochs[0]:= BinFileNBuffs;

      m_TStep[0]:= 1 / ProbeArray[0].sampfreqperchan * 1000000;
      seLockTime.OnChange(Self);
      ProbeWin[0].Exists:= False;
    end;

    m_ProbeIndex:= 0;
    GetElectrode(ProbeWin[0].electrode, ProbeArray[0].electrode_name);
    CreateProbeWin(ProbeWin[0].electrode); //create the probe window for this electrode

    TrackBar.Min:= 0;
    m_CurrentChan:= 0;
    m_ProbeIndex:= 0;
    m_CurrentBuffer:= -1;
    m_TrigIndex:= TransBufferSamples * m_UpsampleFactor; //skip over transbuffer samples

    //load entire bin file into memory;
    Setlength(WholeBinFile, BinFileNBuffs * ProbeArray[0].pts_per_buffer);
    for p:= 0 to High(WholeBinFile) do WholeBinFile[p]:= 2048; //initialize zero uV
    fs:= TFileStream{64}.Create(acFileName, fmOpenRead);
    fs.Seek{64}(0, soFromBeginning); //overwrite any existing file
    fs.ReadBuffer(WholeBinFile[0], Length(WholeBinFile) * 2{shrts});
    fs.Free;

    Setlength(CR.Waveform, ProbeArray[0].pts_per_buffer);

    seFactorChange(Self); //update effective (base) sample freqency --> move to Celectrode change?
    CElectrode.ItemIndex:= 0; //selects first probe in list
    DisplayStatusBar:= True;
    Application.ProcessMessages;
    CElectrode.OnChange(Self);//update info for this probe, display first spike
    WindowState:= wsMaximized;

    GetFileBuffer;

    Exit;
  end;
  // END SIMULATED FILE LOAD HACK //
  m_NProbes:= SurfFile.GetNumProbes;
  m_NSVals:= SurfFile.GetNumSVals;
  NSVals.Caption:= inttostr(m_NSVals);
  m_NMsgs:= SurfFile.GetNumMessages;
  NMsgs.Caption:= inttostr(m_NMsgs);
  m_NStim:= SurfFile.GetNumStimuli;
  NStim.Caption:= inttostr(m_NStim);

  m_EventArray:= SurfFile.GetEventArray;
  m_NEvents:= Length(m_EventArray);

  m_SelStartCRIdx:= -1;
  m_SelEndCRIdx:= -1;
  m_SelStartSValIdx:= -1;
  m_SelEndSValIdx:= -1;

  {calculate recording period}
  SurfFile.GetSurfMsg(0, SurfMsg); {does not take into account multiple start/stop records!}
  m_recStartTime:= SurfMsg.TimeStamp;{nor does this take into account multiple start/stop acquistions, which reset precision clock}

  StartRecTime:= DateTimeToTimeStamp(SurfMsg.DateTime);
  SurfFile.GetSurfMsg(m_NMsgs-1, SurfMsg);
  if pos('Recording stopped', SurfMsg.msg) > 0 then
    EndRecTime:= DateTimeToTimeStamp(SurfMsg.DateTime)
  else EndRecTime:= StartRecTime;
  if EndRecTime.Time - StartRecTime.Time > 0 then
    RecTimeLabel.Caption:= TimeStamp2Str(int64(EndRecTime.Time - StartRecTime.Time) * 1000{HARDCODED!})
  else RecTimeLabel.Caption:= 'file incompletely loaded';

  //Set the arrays for probe-layout records, individual probe and file spikes/CRs
  m_SumSpikes:= 0;
  m_SumCRs:= 0;
  SetLength(ProbeArray, m_NProbes);
  SetLength(ProbeWin, m_NProbes);
  SetLength(m_NEpochs, m_NProbes);
  SetLength(m_TStep, m_NProbes);
  SetLength(m_ProbeThreshold, m_NProbes);

  for p := 0 to m_NProbes - 1 do //THIS NEEDS TO BE CLEANED UP!!!
  begin
    SurfFile.GetProbeRecord(p, ProbeArray[p]);
    //m_NSpikes[p]:= SurfFile.GetNumSpikes(p);
    m_NEpochs[p]:= SurfFile.GetNumEpochs(p);
   // inc(m_SumSpikes, m_NSpikes[p]);   //total ALL spikes...
   // inc(m_SumCRs, m_NCRs[p]);         //and CRs from ALL probes
    if probearray[p].sampfreqperchan = 0 then          //patch for UFF files without
      probearray[p].sampfreqperchan:= DefaultSampFreq; //sampfreqperchan saved
    m_TStep[p]:= 1 / ProbeArray[p].sampfreqperchan * 1000000;
    seLockTime.OnChange(Self);
    ProbeWin[p].Exists:= False;
  end;

  //Generate electrode list from those in file
  for p:= 0 to m_NProbes - 1 do
  begin
    for e := 0 to KNOWNELECTRODES-1 {from ElectrodeTypes} do
      if ProbeArray[p].Electrode_name = KnownElectrode[e].Name then
      begin
        if ProbeArray[p].numchans > 1 then
          CElectrode.Items.Add(KnownElectrode[e].Name)
        else CElectrode.Items.Add(ProbeArray[p].probe_descrip); //more descriptive
        CElectrode.ItemIndex:= p{m_NProbes};//select global file stats from list
        Break; // found electrode type, continue with next probe
      end;
    {end e}
    if ProbeArray[p].electrode_name = 'UnDefined' then
    begin //pick from the list...
      if (CElectrode.ItemIndex >= 0) and (CElectrode.ItemIndex < KNOWNELECTRODES) then
        if ProbeArray[p].numchans = KnownElectrode[CElectrode.ItemIndex].NumSites then
          ProbeArray[p].electrode_name := KnownElectrode[CElectrode.ItemIndex].Name;
    end;
    //if still undefined...
    if ProbeArray[p].electrode_name = 'UnDefined' then
    begin
      ShowMessage('Please select an electrode from the list.');
      //SurfFile.Close;
      Exit;
    end;
    if not GetElectrode(ProbeWin[p].electrode, ProbeArray[p].electrode_name) then
    begin
      ShowMessage(ProbeArray[p].electrode_name + ' is an invalid electrode name');
      Exit;
    end;
    m_ProbeIndex:= p;
    CreateProbeWin(ProbeWin[p].electrode); //create the probe window for this electrode
  end{p};

  if m_NSVals > 0 then //add drop-down entry for SVAL records
    CElectrode.Items.Add('SVal Stream');

  {add Surf/user messages to msg memo}
  for p:= 0 to m_NMsgs - 1 do
  begin
    SurfFile.GetSurfMsg(p, SurfMsg);
    MsgMemo.Lines.Append(TimeToStr(SurfMsg.DateTime)+ ':  ' + SurfMsg.msg);
  end;
  MsgMemo.SelStart:= 0;
  MsgMemo.SelLength:= 0; //move TMemo cursor to start of log

  TrackBar.Min:= 0;
  m_CurrentChan:= 0;
  m_ProbeIndex:= 0;
  m_CurrentBuffer:= -1;
  m_TrigIndex:= TransBufferSamples * m_UpsampleFactor; //skip over transbuffer samples

  seFactorChange(Self); //update effective (base) sample freqency --> move to Celectrode change?
  CElectrode.ItemIndex:= 0; //selects first probe in list
  DisplayStatusBar:= True;
  if m_NProbes > 0 then SetLength(Spike.param, ProbeArray[m_ProbeIndex].numchans, Length(ParamNames));
  Application.ProcessMessages;
  CElectrode.OnChange(Self);//update info for this probe, display first spike
  WindowState:= wsMaximized;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.ComputeWaveformParams(var Spike : TSpike; Precision : boolean = true;
                                                OneChanOnly : shortint = -1);
var c, pt1, pt2, tmppt, peak, negpeak, maxamp, npts : smallint;
  area : integer;
  precise_t1, precise_t2, rx, ry : single;
  SplintArray: array of array of smallint;
  InvertedSpk : boolean;

  procedure RoughParams; //compute approximate parameters...
  var w : smallint;
  begin
    area:= 0;
    pt1 := ProbeArray[m_ProbeIndex].trigpt;
    pt2 := ProbeArray[m_ProbeIndex].trigpt;
    with CR do
    begin
      peak := waveform[c * 2500 + ProbeArray[m_ProbeIndex].trigpt{0}]; //assume peak at or beyond trigpt
      negpeak := waveform[c * 2500 + ProbeArray[m_ProbeIndex].trigpt]; //assume valley at or beyond trigpt
      for w:={1}ProbeArray[m_ProbeIndex].trigpt+1 to ProbeArray[m_ProbeIndex].trigpt+16 do //assume waveform width <16pts from trigpt
      begin
        area:= area + abs(waveform[c] - 2048);
        if peak < waveform[c * 2500 + w] then
        begin
          peak:= waveform[c * 2500 + w];
          pt1:= w;
        end else
        if negpeak > waveform[c * 2500 + w] then
        begin
          negpeak:= waveform[c * 2500 + w];
          pt2:= w;
        end;
      end;
      if pt1 > pt2 then //inverted spike...
      begin
        tmppt:= pt1;
        pt1:= pt2;   //swap contents of pt1 and pt2
        pt2:= tmppt;
        InvertedSpk := true;
      end else InvertedSpk:= false;
    end;
  end;

  procedure PreciseParams; //compute precise parameters (spline about peak/valley)...
  var w : smallint;
  begin                    //ideally, convolve peak and valley with sinc function
    precise_t1:= pt1;
    precise_t2:= pt2;
    npts := pt2-pt1 + 3;   //1 pt either side of approx max/min
    for w := 1 to npts do
    begin
      spliney[w] := Spike.waveform[c,pt1-2+w];
      splinex[w] := w;
      spline2ndderiv[w] := 0;
    end;
    //Spline(SplineX, SplineY, Spline2ndDeriv);
    for w := 1 to npts * m_UpSampleFactor do
    begin
      rx := w/m_UpSampleFactor;
      Splint(SplineX, SplineY, Spline2ndDeriv, rx, ry);
      SplintArray[c, w-1]:= round(ry); // appropriate to round here???!!!!
      //Showmessage(inttostr(w)+'='+inttostr(SplintArray[c, w-1]));
      if peak < ry then
      begin
        peak:= round(ry);
        precise_t1:= pt1 + rx;
      end else
      if negpeak > ry then
      begin
        negpeak:= round(ry);
        precise_t2:= pt1 + rx;
      end;
    end;
  end;

  procedure AllocateParams; // nb: AREA calculation is rough!
  begin
    with Spike do
    begin
      if InvertedSpk then Param[c, 0]:= -1 else Param[c, 0]:= 1; //polarity
      Param[c,1]:= ADValue2uV(Peak);
      Param[c,2]:= ADValue2uV(NegPeak);
      Param[c,3]:= Param[c,1]-Param[c,2]; //amplitude
      Param[c,4]:= pt1; //time at first max/min
      Param[c,5]:= pt2; //time at second max/min
      Param[c,6]:= Area * 600 div 1000; // assumes 600us spike width
    end;
  end;

begin // Main ComputeWaveformParams procedure...
{NOT HERE?!}SetLength(SplintArray, ProbeArray[m_ProbeIndex].numchans, 20 * m_UpSampleFactor);
  if OneChanOnly > -1 then
  begin
    c:= OneChanOnly;
    RoughParams;
    if Precision then
    begin
      PreciseParams;
      pt1:= round(precise_t1 * m_TStep[m_ProbeIndex]);
      pt2:= round(precise_t2 * m_TStep[m_ProbeIndex]);
    end else
    begin
      pt1:= round(pt1 * m_TStep[m_ProbeIndex]);
      pt2:= round(pt2 * m_TStep[m_ProbeIndex]);
    end;
    AllocateParams;
  end else
  begin
    ChanWithMaxAmp:= 0;
    MaxAmp:= 0;
    for c:= 0 to ProbeArray[m_ProbeIndex].numchans-1 do //find channel with largest waveform
    begin
      RoughParams;
      if precision then PreciseParams;
      if MaxAmp < (Peak-NegPeak) then
      begin
        MaxAmp:= (Peak-NegPeak);
        ChanWithMaxAmp:= c;
      end;
    end;
    c:= ChanWithMaxAmp;
    RoughParams;
    if precision then PreciseParams;  //get max/min times for largest waveform
    for c:= 0 to ProbeArray[m_ProbeIndex].numchans-1 do //determine voltages at peak times on other chans
    begin
      //if c:= ChanWithMaxAmp then continue;
      if precision then
      begin
        pt1:= round(ProbeArray[m_ProbeIndex].trigpt + precise_t1);
        pt2:= round(ProbeArray[m_ProbeIndex].trigpt + precise_t2);
        Showmessage('pt1='+inttostr(pt1)+'; precise_t1='+floattostr(precise_t1));
        Peak:= SplintArray[c, pt1];
        NegPeak:= SplintArray[c, pt2];
        //pt1:= round(precise_t1 * m_TStep[m_ProbeIndex]);
        //pt2:= round(precise_t2 * m_TStep[m_ProbeIndex]);
      end else
      begin
        Peak:= Spike.waveform[c, pt1];
        NegPeak:= Spike.waveform[c, pt2];
        //pt1:= round(pt1 * m_TStep[m_ProbeIndex]);
        //pt2:= round(pt2 * m_TStep[m_ProbeIndex]);
      end;
      AllocateParams;
      Showmessage('Chan:'+inttostr(c)+'; Peak: '+inttostr(Spike.Param[c,1]));
    end;
  end;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.CElectrodeChange(Sender: TObject);
begin
  if m_NProbes <> 0 then
  begin
    tbStop.Click;
    m_ProbeIndex:= CElectrode.ItemIndex;
    if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream' then
    begin
      Panel1Str:= 'SVal# ';
      Label4.Caption:= 'SVal word:';
      NProbes.Caption:= '';
      SampleRate.Caption:= '';
      TrackBar.Max:= m_NSVals - 1;
      ProbeArray[m_ProbeIndex].numchans:= 1;
      ProbeArray[m_ProbeIndex].pts_per_chan:= 1;
      //ProbeArray[m_ProbeIndex].
    end else
    begin
      SampleRate.Caption:= '(' + FormatFloat('#,;;0', ProbeArray[m_ProbeIndex].sampfreqperchan) + 'Hz)';
      NSpikes.Caption:= FormatFloat('#,;;not saved', m_NEpochs[CElectrode.ItemIndex]);
      NProbes.Caption:= inttostr(m_ProbeIndex + 1) + ' of ' + inttostr(m_NProbes);
      if ProbeArray[m_ProbeIndex].ProbeSubType = SPIKEEPOCH then
      begin
        Panel1Str:= 'Spike# ';
        Label4.Caption:= '# Spike epochs:';
      end else
      begin
        Panel1Str:= 'Buff# ';
        Label4.Caption:= '# Buffer epochs:';
      end;
      if m_NEpochs[m_ProbeIndex] = 0 then TrackBar.Max:= 0 else
      begin
        if ProbeArray[m_ProbeIndex].ProbeSubType = SPIKESTREAM then
        begin
          {re}BuildSiteProximityArray(m_NearSites, seLockRadius.Value, True);
          TrackBar.Max:= m_NEpochs[m_ProbeIndex] * 100 - 1;
          m_CurrentBuffer:= -1; //ensures reload of buffer
        end else
        if ProbeArray[m_ProbeIndex].ProbeSubType = CONTINUOUS then
          TrackBar.Max:= m_NEpochs[m_ProbeIndex] - 1;
      end;
      tbToglProbeGUI.Click; //update (or remove) probe layout window
    end;
    TrackBar.OnChange(Self); //implicitly displays current epoch
    {allocate dynamic array for upsampled waveform} //--  ASSUMES A CR WAVEFORM LENGTH!!!
    if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream' then m_InterpWaveform:= nil;//
//      else SetLength(m_InterpWaveform, m_UpSampleFactor * (ProbeArray[m_ProbeIndex].pts_per_buffer{(CR.WaveForm)} - length(m_SincKernelArray[0])));
  end;
  TrackBar.SetFocus; //return keybd control to trackbar
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.TrackBarChange(Sender: TObject);
var SelLength : Int64;
begin
  if tbPause.Down then Exit;
  if ShiftKeyDown then
    with TrackBar do
    begin
      SelEnd:= Position;
      SelLength:= SelEnd - SelStart + 1;
      if SelLength < 0 then SelLength:= 0;
      StatusBar.Panels[2].Text:= 'Selection: Epoch# ' + inttostr(SelStart) +
                                 '-' + inttostr(SelEnd) + '(' + inttostr(SelLength) +
                                 '); Duration: ' + TimeStamp2Str(SelLength * 1000){remove HARDCODING};
    end;
  if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream' then
  begin
    SurfFile.GetSVal(TrackBar.Position, SVal);
    Nspikes.Caption:= inttostr(SVal.sval){ + ', '+ inttohex(SVal.sval, 4) + 'H'};
  end else
  if ProbeArray[m_ProbeIndex].ProbeSubType = CONTINUOUS then
  begin
    if m_NEpochs[m_ProbeIndex] = 0 then Exit;
    {if not} SurfFile.GetCR(m_ProbeIndex, TrackBar.Position, CR);
    ProbeWin[m_ProbeIndex].win.PlotWaveForm(CR.Waveform, 0);
    //if tbRasterPlot.Down then RasterForm.UpdateRasterPlot;
    //WaveformStatsUpdate;
  end else
  if ProbeArray[m_ProbeIndex].ProbeSubType = SPIKESTREAM then
  begin

    //hack to step through simulated bin file...
    if BinFileLoaded then
    begin
      if m_NEpochs[m_ProbeIndex] = 0 then Exit;
      if TrackBar.Position div 100 {WaveEpochs per Buffer} <> m_CurrentBuffer then
      begin
        m_CurrentBuffer:= TrackBar.Position div 100;
        Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer], CR.Waveform[0], ProbeArray[0].pts_per_buffer * 2);
        CR.time_stamp:= m_CurrentBuffer * 100000;
        if ChartWinCreated then
        begin
          ChartWin.PlotChart(@CR.Waveform[0], 10000); //update chartwin
          //if ShowTemplateMode then ShowHideTemplates;
        end;
      end;
      m_TrigIndex:= ((TrackBar.Position mod 100) * 25) * m_UpSampleFactor;
      ProbeWin[m_ProbeIndex].win.PlotWaveForm(@CR.Waveform[m_TrigIndex {div m_UpSampleFactor}],
                                              10000{SampPerChanPerBuff}, 4{interp}, 5{white});

      if ChartWinCreated then ChartWin.PlotVertLine(m_TrigIndex {div m_UpSampleFactor});
      if DisplayStatusBar then UpdateStatusBar;
      Exit;
    end{binfileloaded};
    // end HACK //

    if m_NEpochs[m_ProbeIndex] = 0 then Exit;
    if TrackBar.Position div 100 {WaveEpochs per Buffer} <> m_CurrentBuffer then
    begin
      m_CurrentBuffer:= TrackBar.Position div 100;
      GetFileBuffer;
      if ChartWinCreated then
      begin
        ChartWin.PlotChart(@CR.Waveform[0], 2500); //update chartwin
        if ShowTemplateMode then ShowHideTemplates;
      end;
    end;
    {REMOVE HARD CODING!!!!!!!!!!!!!!!!}
    m_TrigIndex:= ((TrackBar.Position mod 100) * 25) * m_UpSampleFactor;
    ProbeWin[m_ProbeIndex].win.PlotWaveForm(@CR.Waveform[m_TrigIndex div m_UpSampleFactor],
                                        2500{SampPerChanPerBuff}, 1,{no decimn,} 0{white});

    if ChartWinCreated then ChartWin.PlotVertLine(m_TrigIndex div m_UpSampleFactor);
    //if tbRasterPlot.Down then RasterForm.UpdateRasterPlot;
    //WaveformStatsUpdate;
  end;
  if DisplayStatusBar then UpdateStatusBar;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.UpdateStatusBar;
var TimeStamp : int64;
begin
  with StatusBar do
  begin
    TimeStamp:= 0;
    if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream' then
    begin
      TimeStamp:= SVal.time_stamp;
      Panels[1].Text:= Panel1Str + inttostr(TrackBar.Position + 1) + '/'
                                         + inttostr(m_NSVals);
    end else
    begin
      case ProbeArray[m_ProbeIndex].ProbeSubType of
        SPIKESTREAM : begin
                        Panels[1].Text:= Panel1Str + inttostr(m_CurrentBuffer + 1) + '/'
                                                   + inttostr(m_NEpochs[m_ProbeIndex]{remove HARDCODING});
                        TimeStamp:= CR.time_stamp  + int64(TrackBar.Position) * 1000 mod 100000;
                      end;
        CONTINUOUS,
        SPIKEEPOCH  : begin
                        Panels[1].Text:= Panel1Str + inttostr(m_CurrentBuffer) + '/'
                                                   + inttostr(m_NEpochs[m_ProbeIndex]);
                        TimeStamp:= CR.time_stamp;
                      end;
      end;
    end;

    with m_HardWareInfo do
      Panels[0].Text:= 'Time: ' + TimeStamp2Str(TimeStamp);
  end{StatusBar};
  {ComputeWaveformParams(Spike);}
end;

{---------------------------------------------------------------------------------------------}
function TSurfBawdForm.TimeStamp2Str(TimeStamp : int64) : string;
begin
  with m_HardwareInfo do
  begin
    Result:= FormatFloat('00:', TimeStamp div (int64(iMasterClockFreq) * 3600)) //hours
           + FormatFloat('00:', TimeStamp div (iMasterClockFreq * 60) mod 60)   //minutes
           + FormatFloat('00.000', TimeStamp mod (iMasterClockFreq * 60) / iMasterClockFreq); //seconds.milliseconds
{    Result:= FormatFloat('00:', TimeStamp / iMasterClockFreq / 3600)     //hours
           + FormatFloat('00:', TimeStamp / (iMasterClockFreq * 60))   //minutes
           + FormatFloat('00.000', TimeStamp mod (iMasterClockFreq * 60) / iMasterClockFreq); //seconds.milliseconds}
  end;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.TrackBarKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if tbPause.Down and ((key = VK_ESCAPE) or (key = VK_SHIFT)) then
    tbPause.Down:= False else
   //if key in [37, 38]{left, up arrow} then DisplaySpikeFromStream(-1) else
     //if key in [39, 40]{right, down arrow} then DisplaySpikeFromStream;
  if ShiftKeyDown then Exit;
  if key = VK_SHIFT then //reset, begin new selection start
  begin
    ShiftKeyDown:= True;
    TrackBar.SelStart:= TrackBar.Position;
    TrackBar.SelEnd:= TrackBar.Position;
    StatusBar.Panels[2].Text:= 'No range selected';
    UpdateStatusBar;
    m_SelStartCRIdx:= -1;
    m_SelEndCRIdx:= -1;
    m_SelStartSValIdx:= -1;
    m_SelEndSValIdx:= -1;
  end;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.TrackBarKeyUp(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if key = VK_SHIFT then ShiftKeyDown:= False;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.FormClose(Sender: TObject; var Action: TCloseAction); // free memory
var p : byte;
begin
  if SurfFile.GetNumProbes = 0 then
  begin //no file open
    Action := caFree;
    Exit;
  end;

  PlayTimer.Enabled:= False;

  for p := 0 to SurfFile.GetNumProbes-1 do
    if ProbeWin[p].exists then
    begin
      ProbeWin[p].win.free;
      ProbeWin[p].exists := false;
    end;

  ProbeArray := nil;
  Spike.waveform := nil;
  CR.waveform := nil;
  CR2.waveform:= nil;

  if tbToglStatsWin.Down then WaveFormStatsWin.Release;

  if GUICreated then GUIForm.Close;

  if ChartWinCreated then Chartwin.Release;
  if TemplateWinCreated then TemplWin.Release;
  if FitHistoWinCreated then FitHistoWin.Release;
  if ISIHistoWinCreated then ISIHistoWin.Release;

  m_ExportStream.Free;
  SurfFile.Close;

  Action := caFree;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbPlayClick(Sender: TObject);
begin
  if tbReversePlay.Down then tbPause.Down:= False; //no play-search in reverse
  GroupBox2.Enabled:= False; //avoid error(s) if user checks/unchecks mid-search play
  PlayTimer.Enabled:= True;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbStopClick(Sender: TObject);
begin
  tbPlay.Down := false;
  tbReversePlay.Down := false;
  tbPause.Down:= false;
  TrackBar.Position:= 0;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbStepFowardClick(Sender: TObject);
begin
  if tbPause.Down then DisplaySpikeFromStream{(+1))}
    else TrackBar.Position:= TrackBar.Position + 1;
  //if tbLocNSort.Down then PredNSort;}
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbStepBackClick(Sender: TObject);
begin
  tbPause.Down:= False; //only forward spike search currently implemented
  Trackbar.Position:= TrackBar.Position - 1;
  //if tbLocNSort.Down then PredNSort;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.DisplaySpikeFromStream(increment : integer);
var c, PlotIndex : integer;
begin
  with ProbeWin[m_ProbeIndex].win do
  begin
    Canvas.Font.Color:= clYellow;
    Canvas.TextOut(8, tbControl.Height + 5, 'Searching for spike...');
    if cbDetectRaw.Checked = False then //search interpolated data...
    //STILL TO DO: need to ensure interpolated buffer is loaded...
    while not FindNextThresholdX(m_InterpWaveform) and tbPause.Down do
    begin
      m_TrigIndex:= TransBufferSamples * m_UpsampleFactor; //skip over leading transbuffer samples
      inc(m_CurrentBuffer, increment); //get next/previous buffer
      TrackBar.Position:= m_CurrentBuffer * 100; //update trackbar position/statusbar time
      if TrackBar.Position = TrackBar.Max then //stop searching if @EOF
      begin
        Dec(m_CurrentBuffer);
        Canvas.FillRect(Rect(8, tbControl.Height, 120, tbControl.Height + 20)); //erase textout
        if DisplayStatusBar then UpdateStatusBar;
        DrawWaveForms:= tbToglWform.Down;
        Exit;
      end;
      if Exporting and (TrackBar.Position > TrackBar.SelEnd + 1) then Exit;
      GetFileBuffer;//SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR);
      if ChartWinCreated then ChartWin.PlotChart(@CR.Waveform[0], 2500); //update chartwin
      if DisplayStatusBar then UpdateStatusBar;
      Application.ProcessMessages;
    end{while}else //search raw data...  REMOVE this code when CR.Waveform = m_InterpWaveform
    while not FindNextThresholdX(CR.Waveform) and tbPause.Down do
    begin
      m_TrigIndex:= 0;
      inc(m_CurrentBuffer, increment); //get next/previous buffer
      TrackBar.Position:= m_CurrentBuffer * 100; //update trackbar position/statusbar time
      if TrackBar.Position = TrackBar.Max then //stop searching if @EOF
      begin
        Dec(m_CurrentBuffer);
        Canvas.FillRect(Rect(8, tbControl.Height, 120, tbControl.Height + 20)); //erase textout
        if DisplayStatusBar then UpdateStatusBar;
        DrawWaveForms:= tbToglWform.Down;
        Exit;
      end;
      if Exporting and (TrackBar.Position > TrackBar.SelEnd + 1) then Exit;
      SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR);
      if DisplayStatusBar then UpdateStatusBar;
      if ChartWinCreated then ChartWin.PlotChart(@CR.Waveform[0], 2500); //update chartwin
      Application.ProcessMessages;
    end{while};
    inc(m_SpikeCount);
    lbSpikeCount.Caption:= inttostr(m_SpikeCount);
    if cbDetectRaw.Checked then
    begin
      PlotIndex:= m_TrigIndex;
      TrackBar.Position:= m_CurrentBuffer * 100 + Trunc((m_TrigIndex mod 2500)/2500*100); //update trackbar position/statusbar time
      if not Exporting then m_TrigIndex:= (m_TrigIndex + 2500) mod 134999; //mod minus 1 advances to next sample! REMOVE HARDCODING
    end else
    begin //display raw data from upsampled data search...
      //PlotWaveForm(@m_InterpWaveform[m_TrigIndex mod 10200 + m_PosLockedSamp[m_TrigIndex div 10200]
      //           - ProbeArray[m_ProbeIndex].Trigpt * m_UpsampleFactor], 10200{SampPerChanPerBuff}, {m_UpsampleFactor{dispdecimn{}{,} 0{white});
      PlotIndex:= ((m_TrigIndex - 100) mod 10200) div 4;
      TrackBar.Position:= m_CurrentBuffer * 100 + Trunc((m_TrigIndex mod 10200)/10200*100); //update trackbar position/statusbar time
      if not Exporting then m_TrigIndex:= (m_TrigIndex + 10200) mod (550800 -1); //mod minus 1 advances to next sample! REMOVE HARDCODING!
    end;
    if ((PlotIndex mod 2500) + 25 - ProbeArray[m_ProbeIndex].Trigpt) > 2500 then
    begin //load next buffer as spike traverses two file buffers...
      SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer + 1, CR2);
      Setlength(CR.Waveform, Length(CR.Waveform) + 25);
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
      Move(CR2.Waveform[c*2500], CR.Waveform[2500{end of CR...}+(c*2500{...of chan 'c'})], 50{bytes}); {copy less?}
      PlotWaveForm(@CR.Waveform[PlotIndex mod 2500 {+ m_PosLockedSamp[PlotIndex div 2500]}
                   - ProbeArray[m_ProbeIndex].Trigpt], 2500{SampPerChanPerBuff},  1, 0{white});
      if not Exporting then SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR); //restore CR
    end else
    if ((PlotIndex mod 2500) - ProbeArray[m_ProbeIndex].Trigpt) < 0 then
    begin //load previous buffer as spike traverses two file buffers...
      SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer - 1, CR2);
      Setlength(CR2.Waveform, Length(CR.Waveform) + 25);
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
        Move(CR.Waveform[c*2500], CR2.Waveform[2500{end of CR2...}+(c*2500{...of chan 'c'})], 50{bytes}); {copy less?}
      PlotWaveForm(@CR2.Waveform[2500 + PlotIndex mod 2500 {+ m_NumLockedSamp[m_TrigIndex div 2500]}
                   - ProbeArray[m_ProbeIndex].Trigpt], 2500{SampPerChanPerBuff}, 1, 0{white});
    end else //spike within current buffer, so just plot it...
      PlotWaveForm(@CR.Waveform[PlotIndex mod 2500 {+ m_PosLockedSamp[m_TrigIndex div 2500]}
                   - ProbeArray[m_ProbeIndex].Trigpt], 2500{SampPerChanPerBuff},  1, 0{white});
//    if BuildTemplateMode then UpdateTemplates;
    if ChartWinCreated then ChartWin.PlotSpikeMarker(PlotIndex mod 2500, m_SpikeMaxMinChan);
  end{with};
  if DisplayStatusBar then UpdateStatusBar;
end;

{-----------------------------------------------------------------------------}
function TSurfBawdForm.FindNextThresholdX(const Waveform : TWaveform) : boolean;
var c, l, TrigIdx, TrigMaxMin, SpikeMaxMin, TotalBufferLen, LastSample, SamplesPerChan {<- make this global when removing hardcoding?}: integer;
    Positive, PosNegTrig, GlobalSearch : boolean;
    nSkip : integer;
begin
  Result:= False;
  TotalBufferLen:= Length(Waveform);
  SamplesPerChan:= 2500; //remove hardcoding!!!

  if cbDecimate.Checked then nSkip:= 8 else nSkip:= 1;

  if cbDetectRaw.Checked = False then
  begin
    LastSample:= (SamplesPerChan + TransBufferSamples) * m_UpsampleFactor;
    SamplesPerChan:= LastSample + TransBufferSamples * m_UpsampleFactor;
  end else
    LastSample:= 0; //not sure why zero gives 134999 iterations
  if GUICreated and (GUIForm.m_iNSitesSelected <> 0) then GlobalSearch:= False
    else GlobalSearch:= True;
  if m_ProbeThreshold[m_ProbeIndex] > 2048 then Positive:= True
    else Positive:= False;
  if ProbeWin[m_ProbeIndex].DispTrigBipolar then PosNegTrig:= True
    else PosNegTrig:= False;
  repeat
    c:= m_TrigIndex div SamplesPerChan; //channel of current sample
    if GlobalSearch or (GUIForm.SiteSelected[c]) {or (m_NumLockedSamp[c] > 0)} then
    begin
      if rgThreshold.ItemIndex = 0 {simple threshold with global, user-specified lockout} then
      begin
        if PosNegTrig then
        begin //bipolar trigger...
          if Waveform[m_TrigIndex] > m_ProbeThreshold[m_ProbeIndex] then
          begin //...over threshold
            if (m_PosLockedSamp[c] = 0) and (m_NegLockedSamp[c] = 0) then
            begin //site not locked, so new spike...
              Result:= True;
              while (Waveform[m_TrigIndex] < Waveform[m_TrigIndex + nSkip{1}])
                do inc(m_TrigIndex, nSkip); //find waveform peak (local maxima)
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans -1 do
                inc(m_PosLockedSamp[l], m_SimpleLockSamples);//lockout all channels for specified time
              SpikeMaxMin:= Waveform[m_TrigIndex];
              m_SpikeMaxMinChan:= c;
              TrigIdx:= (m_TrigIndex mod SamplesPerChan); //chan 0 sample at t=max of initial trig chan
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //find local channel with maximum amplitude...
                if (l in m_NearSites[c]) then
                begin
                  if Waveform[TrigIdx] > SpikeMaxMin then
                  begin
                    SpikeMaxMin:= Waveform[TrigIdx];
                    m_SpikeMaxMinChan:= l;
                  end;
                end;
                inc(TrigIdx, SamplesPerChan)
              end{l};

(*              for l:= 0 to ProbeArray[m_ProbeIndex].numchans -1 do
                  if (l in m_NearSites[m_SpikeMaxMinChan]) then
                    inc(m_PosLockedSamp[l], m_SimpleLockSamples);//lockout LOCAL for specified time
*)
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(m_SpikeMaxMinChan, clOrange); //highlight spike channel
                GUIForm.Refresh;
              end;
              Exit;
            end;
          end{over threshold};
          if m_PosLockedSamp[c] > 0 then dec(m_PosLockedSamp[c]); //decrement lock counter for this site
          if Waveform[m_TrigIndex] < (RESOLUTION_12_BIT - m_ProbeThreshold[m_ProbeIndex]) then
          begin //also check negative threshold trigger...
            if (m_NegLockedSamp[c] = 0) and (m_PosLockedSamp[c] = 0) then
            begin //site not locked, so new spike...
              Result:= True;
              while (Waveform[m_TrigIndex] > Waveform[m_TrigIndex + nSkip{1}])
                do inc(m_TrigIndex, nSkip); //find waveform valley (local minima)
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans -1 do
                inc(m_NegLockedSamp[l], m_SimpleLockSamples);//lockout all channels for specified time
              SpikeMaxMin:= Waveform[m_TrigIndex];
              m_SpikeMaxMinChan:= c;
              TrigIdx:= (m_TrigIndex mod SamplesPerChan); //chan 0 sample at t=max of initial trig chan
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //find local channel with minimum amplitude...
                if (l in m_NearSites[c]) then
                begin
                  if Waveform[TrigIdx] < SpikeMaxMin then
                  begin
                    SpikeMaxMin:= Waveform[TrigIdx];
                    m_SpikeMaxMinChan:= l;
                  end;
                end;
                inc(TrigIdx, SamplesPerChan)
              end{l};

(*              for l:= 0 to ProbeArray[m_ProbeIndex].numchans -1 do
                if (l in m_NearSites[m_SpikeMaxMinChan]) then
                  inc(m_PosLockedSamp[l], m_SimpleLockSamples);//lockout LOCAL for specified time
*)
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(m_SpikeMaxMinChan, clOrange); //highlight spike channel
                GUIForm.Refresh;
              end;
              Exit;
            end;
          end{under threshold};
          if m_NegLockedSamp[c] > 0 then dec(m_NegLockedSamp[c]); //decrement lock counter for this site...
        end{bipolar trigger} else

        if Positive then
        begin //positive threshold trigger...
          if Waveform[m_TrigIndex] > m_ProbeThreshold[m_ProbeIndex] then
          begin //...over threshold
            if m_PosLockedSamp[c] = 0 then
            begin //site not locked, so new spike...
              Result:= True;
              while (Waveform[m_TrigIndex] < Waveform[m_TrigIndex + nSkip{1}])
                do inc(m_TrigIndex, nSkip); //find waveform peak (local maxima)
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans -1 do
                inc(m_PosLockedSamp[l], m_SimpleLockSamples);//lockout all channels for specified time
              SpikeMaxMin:= Waveform[m_TrigIndex];
              m_SpikeMaxMinChan:= c;
              TrigIdx:= (m_TrigIndex mod SamplesPerChan); //chan 0 sample at t=max of initial trig chan
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //find local channel with maximum amplitude...
                if (l in m_NearSites[c]) then
                begin
                  if Waveform[TrigIdx] > SpikeMaxMin then
                  begin
                    SpikeMaxMin:= Waveform[TrigIdx];
                    m_SpikeMaxMinChan:= l;
                  end;
                end;
                inc(TrigIdx, SamplesPerChan)
              end{l};
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(m_SpikeMaxMinChan, clOrange); //highlight spike channel
                GUIForm.Refresh;
              end;
              Exit;
            end;
          end{over threshold};
          if m_PosLockedSamp[c] > 0 then dec(m_PosLockedSamp[c]); //decrement lock counter for this site
        end{+ve threshold}else
        begin //negative threshold trigger...
          if Waveform[m_TrigIndex] < m_ProbeThreshold[m_ProbeIndex] then
          begin //...under threshold
            if m_NegLockedSamp[c] = 0 then
            begin //site not locked, so new spike...
              Result:= True;
              while (Waveform[m_TrigIndex] > Waveform[m_TrigIndex + nSkip{1}])
                do inc(m_TrigIndex, nSkip); //find waveform valley (local minima)
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans -1 do
                inc(m_NegLockedSamp[l], m_SimpleLockSamples);//lockout all channels for specified time
              SpikeMaxMin:= Waveform[m_TrigIndex];
              m_SpikeMaxMinChan:= c;
              TrigIdx:= (m_TrigIndex mod SamplesPerChan); //chan 0 sample at t=max of initial trig chan
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //find local channel with minimum amplitude...
                if (l in m_NearSites[c]) then
                begin
                  if Waveform[TrigIdx] < SpikeMaxMin then
                  begin
                    SpikeMaxMin:= Waveform[TrigIdx];
                    m_SpikeMaxMinChan:= l;
                  end;
                end;
                inc(TrigIdx, SamplesPerChan)
              end{l};
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(m_SpikeMaxMinChan, clOrange); //highlight spike channel
                GUIForm.Refresh;
              end;
              Exit;
            end;
          end{under threshold};
          if m_NegLockedSamp[c] > 0 then dec(m_NegLockedSamp[c]); //decrement lock counter for this site...
        end{-ve threshold};
      end{simple threshold};// else

  (*    if PosNegTrig then
      begin //bipolar threshold trigger...
        if Waveform[m_TrigIndex] > m_ProbeThreshold[m_ProbeIndex] then
        begin //...over threshold
          if m_PosLockedSamp[c] = 0 then
          begin //site not locked
            while Waveform[m_TrigIndex + m_PosLockedSamp[c]] >
                  Waveform[m_TrigIndex + m_PosLockedSamp[c]-1] do
              inc(m_PosLockedSamp[c]); //lockout 'till waveform peak
            if m_PosLockedSamp[c] > 0 then
            begin //above threshold, and dV.dt +ve so got a (new) spike...
              Result:= True;
              dec(m_PosLockedSamp[c]); //don't need to lock current sample

              if PosNegTrig then //lock out negative trigger...BUT WHAT ABOUT SURROUNDING CHANNELS?!
              for l:= 0 to 10 do //10 rawsamples = 400usec <-- remove hardcoding!
                if Waveform[m_TrigIndex + l] < (RESOLUTION_12_BIT - m_ProbeThreshold[m_ProbeIndex]) then
                begin //if valley of spike below -ve threshold then...
                  while Waveform[m_TrigIndex + l + m_NegLockedSamp[c]] <
                        Waveform[m_TrigIndex + l + m_NegLockedSamp[c]-1] do
                          inc(m_NegLockedSamp[c]); //...lockout -ve trig 'till valley bottom
                  Break;
                end;

              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //lock neighbouring sites to peak (samples over threshold)
                if (l in m_NearSites[c]) and (m_PosLockedSamp[l] = 0) then
                begin
                  TrigIdx:= m_TrigIndex mod SamplesPerChan + l * SamplesPerChan; //trig' sample of 'l' chan
                  while Waveform[TrigIdx] > Waveform[TrigIdx-1] do
                  begin
                    if Waveform[TrigIdx] > m_ProbeThreshold[m_ProbeIndex]
                      then inc(m_PosLockedSamp[l]); //lockout 'till waveform peak
                    inc(TrigIdx);
                  end;
                end;
                if GUICreated and (GlobalSearch or GUIForm.SiteSelected[l]) then //show lockout...
                  if m_PosLockedSamp[l] > 0 then GUIForm.ChangeSiteColor(l, (m_PosLockedSamp[l]* 35) and $000000FF)
                  else GUIForm.ChangeSiteColor(l, clLime);
              end{l};
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(c, clOrange); //highlight trigger channel
                GUIForm.Refresh;
              end;
              if ChartWinCreated then ChartWin.PlotSpikeMarker(m_TrigIndex mod SamplesPerChan, c);
              Exit; //got (new) spike, so exit
            end{+ve dV/dt};
          end else
          dec(m_PosLockedSamp[c]); //decrement lock counter for this site...
        end{over threshold};
        if Waveform[m_TrigIndex] < (RESOLUTION_12_BIT - m_ProbeThreshold[m_ProbeIndex]) then
        begin //also check negative threshold trigger...
          if m_NegLockedSamp[c] = 0 then
          begin //site not locked
            while Waveform[m_TrigIndex + m_NegLockedSamp[c]] <
                  Waveform[m_TrigIndex + m_NegLockedSamp[c]-1] do
              inc(m_NegLockedSamp[c]); //lockout 'till waveform valley
            if m_NegLockedSamp[c] > 0 then
            begin //below threshold, and dV.dt -ve so got a (new) spike...
              Result:= True;
              dec(m_NegLockedSamp[c]); //don't meed to lock trigger sample

              if PosNegTrig then //lock out positive trigger...
              for l:= 0 to 10 do //10 rawsamples = 400usec <-- remove hardcoding!
                if Waveform[m_TrigIndex + l] > m_ProbeThreshold[m_ProbeIndex] then
                begin //if peak of spike above +ve threshold then...
                  while Waveform[m_TrigIndex + l + m_PosLockedSamp[c]] >
                        Waveform[m_TrigIndex + l + m_PosLockedSamp[c]-1] do
                          inc(m_PosLockedSamp[c]); //...lockout -ve trig 'till valley bottom
                  Break;
                end;

              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //lock neighbouring sites to valley (samples under threshold)
                if (l in m_NearSites[c]) and (m_NegLockedSamp[l] = 0) then
                begin
                  TrigIdx:= m_TrigIndex mod SamplesPerChan + l * SamplesPerChan; //trig' sample of 'l' chan
                  while Waveform[TrigIdx] < Waveform[TrigIdx-1] do
                  begin //only lock channel if current spike under threshold
                    if Waveform[TrigIdx] < m_ProbeThreshold[m_ProbeIndex] then
                      inc(m_NegLockedSamp[l]); //lockout 'till waveform valley
                    inc(TrigIdx);
                  end;
                end;
                if GUICreated and (GlobalSearch or GUIForm.SiteSelected[l]) then //show lockout...
                  if m_NegLockedSamp[l] > 0 then GUIForm.ChangeSiteColor(l, (m_NegLockedSamp[l]* 35) and $000000FF)
                  else GUIForm.ChangeSiteColor(l, clLime);
              end{l};
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(c, clOrange); //highlight trigger channel
                GUIForm.Refresh;
              end;
              if ChartWinCreated then ChartWin.PlotSpikeMarker(m_TrigIndex mod SamplesPerChan, c);
              Exit; //got (new) spike, so exit
            end{-ve dV/dt};
          end else
          dec(m_NegLockedSamp[c]); //decrement lock counter for this site...
        end{-ve threshold}
      end{bipolarTrig}else
      if Positive then
      begin //positive threshold trigger...
        if Waveform[m_TrigIndex] > m_ProbeThreshold[m_ProbeIndex] then
        begin //...over threshold
          if m_PosLockedSamp[c] = 0 then
          begin //site not locked
            //while Waveform[m_TrigIndex + m_PosLockedSamp[c]] >
            //      Waveform[m_TrigIndex + m_PosLockedSamp[c]-1] do
            TrigMaxMin:= m_TrigIndex;
            while Waveform[TrigMaxMin] > Waveform[TrigMaxMin - 1] do
            begin
              inc(m_PosLockedSamp[c]); //lockout 'till waveform peak
              inc(TrigMaxMin);
            end;
            if m_PosLockedSamp[c] > 0 then
            begin //above threshold, and dV.dt +ve so got a (new) spike...
              Result:= True;
              dec(m_PosLockedSamp[c]); //don't need to lock current sample

              //Showmessage('Trigchan = ' + inttostr(c));
              SpikeMaxMin:= Waveform[TrigMaxMin + 1{???}]; //find 'centre of mass' of current spike
              m_SpikeMaxMinChan:= c;
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
                if (l in m_NearSites[c]) and (m_PosLockedSamp[l] = 0) then
                begin
                  TrigIdx:= m_TrigIndex mod SamplesPerChan + l * SamplesPerChan; //sample of 'l' chan at max of initial trig chan
                  if Waveform[TrigIdx] > SpikeMaxMin then
                  begin
                    SpikeMaxMin:= Waveform[TrigIdx];
                    m_SpikeMaxMinChan:= l; //trigchan is now maxtrigchan
                  end;
                end;


              //Showmessage('MaxMinTrigchan = '+inttostr(m_SpikeMaxMinChan));
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //now lock neighbouring sites to peak (samples over threshold)
                if (l in m_NearSites[m_SpikeMaxMinChan]) and (m_PosLockedSamp[l] = 0) then
                begin
                  TrigIdx:= m_TrigIndex mod SamplesPerChan + l * SamplesPerChan; //trig' sample of 'l' chan
                  while Waveform[TrigIdx] > Waveform[TrigIdx-1] do
                  begin
                    if Waveform[TrigIdx] > m_ProbeThreshold[m_ProbeIndex]
                      then inc(m_PosLockedSamp[l]); //lockout 'till waveform peak
                    inc(TrigIdx);
                  end;
                end;
                if GUICreated and (GlobalSearch or GUIForm.SiteSelected[l]) then //show lockout...
                  if m_PosLockedSamp[l] > 0 then GUIForm.ChangeSiteColor(l, (m_PosLockedSamp[l]* 35) and $000000FF)
                  else GUIForm.ChangeSiteColor(l, clLime);
              end{l};
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(m_SpikeMaxMinChan, clOrange); //highlight trigger channel
                GUIForm.Refresh;
              end;
              if ChartWinCreated then ChartWin.PlotSpikeMarker(m_TrigIndex mod SamplesPerChan, m_SpikeMaxMinChan);
              Exit; //got (new) spike, so exit
            end{+ve dV/dt};
          end else
          dec(m_PosLockedSamp[c]); //decrement lock counter for this site...
        end{over threshold};
      end{+ve threshold}else
      begin //negative threshold trigger...
        if Waveform[m_TrigIndex] < m_ProbeThreshold[m_ProbeIndex] then
        begin //...under threshold
          if m_NegLockedSamp[c] = 0 then
          begin //site not locked
            while Waveform[m_TrigIndex + m_NegLockedSamp[c]] <
                  Waveform[m_TrigIndex + m_NegLockedSamp[c]-1] do
              inc(m_NegLockedSamp[c]); //lockout 'till waveform valley
            if m_NegLockedSamp[c] > 0 then
            begin //below threshold, and dV.dt -ve so got a (new) spike...
              Result:= True;
              dec(m_NegLockedSamp[c]); //don't need to lock trigger sample
              for l:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
              begin //now lock neighbouring sites to valley (samples under threshold)
                if (l in m_NearSites[c]) and (m_NegLockedSamp[l] = 0) then
                begin
                  TrigIdx:= m_TrigIndex mod SamplesPerChan + l * SamplesPerChan; //trig' sample of 'l' chan
                  while Waveform[TrigIdx] < Waveform[TrigIdx-1] do
                  begin //only lock channel if current spike under threshold
                    if Waveform[TrigIdx] < m_ProbeThreshold[m_ProbeIndex] then
                      inc(m_NegLockedSamp[l]); //lockout 'till waveform valley
                    inc(TrigIdx);
                  end;
                end;
                if GUICreated and (GlobalSearch or GUIForm.SiteSelected[l]) then //show lockout...
                  if m_NegLockedSamp[l] > 0 then GUIForm.ChangeSiteColor(l, (m_NegLockedSamp[l]* 35) and $000000FF)
                  else GUIForm.ChangeSiteColor(l, clLime);
              end{l};
              if GUICreated then
              begin
                GUIForm.ChangeSiteColor(c, clOrange); //highlight trigger channel
                GUIForm.Refresh;
              end;
              if ChartWinCreated then ChartWin.PlotSpikeMarker(m_TrigIndex mod SamplesPerChan, c);
              Exit; //got (new) spike, so exit
            end{-ve dV/dt};
          end else
          dec(m_NegLockedSamp[c]); //decrement lock counter for this site...
        end{under threshold};
      end{-ve threshold} *)
    end{site selected};
    m_TrigIndex:= (m_TrigIndex + SamplesPerChan{ * nSkip}) mod (TotalBufferLen - nSkip{1}); //mod minus 1 advances to next sample
  until m_TrigIndex = LastSample; //last 'central buffer' sample of last channel, so exit...
  { OLD METHOD FOLLOWS, SIMPLER AND FASTER, BUT RESULTS IN LARGER SPATIOTEMPORAL LOCKS }
         (* if m_NearSites[c] *{intersection} m_LockedSites <> [] then
          begin //channel in current lockout zone
            if m_NumLockedSamp[c] > 0 then dec(m_NumLockedSamp[c])
              else Exclude(m_LockedSites, c);
            UpdateSiteStatus;
          end else //channel (and its neighbours) not locked
          begin
            while Waveform[m_TrigIndex + m_NumLockedSamp[c]] >
                  Waveform[m_TrigIndex + m_NumLockedSamp[c] -1] do
                    inc(m_NumLockedSamp[c]); //lockout 'till waveform peak
            if m_NumLockedSamp[c] > 0 then
            begin //above threshold, and dV.dt +ve so got a (new) spike...
              Result:= True;
              dec(m_NumLockedSamp[c]);
              if m_NumLockedSamp[c] > 0 then Include(m_LockedSites, c)
                else Exclude(m_LockedSites, c); //ONLY IF NUMLOCKEDSAMP STILL > 0 AFTER DEC !!!
              UpdateSiteStatus;
              inc(m_tempsum, m_numLockedSamp[c]);
              inc(m_tempn);
              NSpikes.Caption:= Floattostr(m_tempsum/m_tempn);
              NStim.Caption:= inttostr(m_tempn);
              Exit; //...new spike found, so exit search
            end;
          end;
        end;// else
          //if m_NumLockedSamp[c] = 0 then Exclude(m_LockedSites, c);
      end else
      begin //negative trigger...
        if Waveform[m_TrigIndex] < m_ProbeThreshold[m_ProbeIndex] then
        begin //...over threshold
          if m_NearSites[c] *{intersection} m_LockedSites <> [] then
          begin //channel in current lockout zone
            if m_NumLockedSamp[c] > 0 then dec(m_NumLockedSamp[c])
              else Exclude(m_LockedSites, c);
            UpdateSiteStatus;
          end else //channel (and its neighbours) not locked
          begin
            while Waveform[m_TrigIndex + m_NumLockedSamp[c]{correct lock?}] <
                  Waveform[m_TrigIndex + m_NumLockedSamp[c] -1] do
                    inc(m_NumLockedSamp[c]); //lockout 'till waveform peak
            if m_NumLockedSamp[c] > 0 then
            begin //above threshold, and dV.dt +ve so got a (new) spike...
              Result:= True;
              Include(m_LockedSites, c);
              dec(m_NumLockedSamp[c]); //remove dec... too shorter lockout?
              UpdateSiteStatus;
              Exit; //...new spike found, so exit procedure
            end;
          end;
        end else
          if m_NumLockedSamp[c] = 0 then Exclude(m_LockedSites, c);
      end;
    end;
    m_TrigIndex:= (m_TrigIndex + SamplesPerChan) mod (TotalBufferLen - 1); //mod minus 1 advances to next sample! REMOVE HARDCODING!
  until m_TrigIndex = 10100; //last 'central buffer' sample of last channel, so exit...*)
  //(m_TrigIndex mod SamplesPerChan) = 2499; // unsure whether the all samples in buffer tested for threshold crossings!
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.UpdateTemplates;
var i, tchans, index, wavoffset, BestFitSoFar, iTemplateFit, bf : integer;
  NewTemplate : TSpikeTemplate;
begin
  { copy ALL channels of new epoch into local template }
  with ProbeArray[m_ProbeIndex], TemplWin, NewTemplate do
  begin
    index:= 0;
    Sites:= NearSites[m_SpikeMaxMinChan];
    MaxChan:= m_SpikeMaxMinChan;
    if cbDetectRaw.Checked then
    begin //copy epoch from raw buffer...
      wavoffset:= (m_TrigIndex - 10{400µsec}{trigpt}) mod 2500; // MESSY! REMOVE HARDCODING!!!
      Setlength(AvgWaveform, NumChans * pts_per_chan); //all channels, temporarily...
      for i:= 0 to NumChans - 1 do
        //if i in Sites then
        begin
          Move(SurfBawdForm.CR.Waveform[wavoffset + i * 2500], AvgWaveform[index], pts_per_chan * 2);
          inc(index, pts_per_chan);
          //inc(tchans);
        end;
    end else
    begin //copy epoch from upsampled buffer...
      wavoffset:= (m_TrigIndex - 10{400µsec} * m_UpSampleFactor) mod 10200; // MESSY! REMOVE HARDCODING!!!
      Setlength(AvgWaveform, NumChans * pts_per_chan * m_UpsampleFactor); //all channels, temporarily...
      for i:= 0 to NumChans - 1 do
        //if i in Sites then
        begin
          Move(m_InterpWaveform[wavoffset + i * 10200], AvgWaveform[index], pts_per_chan * m_UpsampleFactor * 2);
          inc(index, pts_per_chan * m_UpsampleFactor);
          //inc(tchans);
        end;
    end;

    { check for *best* match with an existing template }
    bf:= -1;
    BestFitSoFar:= High(Integer);
    for i:= 0 to NumTemplates - 1 do
    begin //find best match to existing templates...
      iTemplateFit:= MatchTemplate(NewTemplate, SpikeTemplates[i]);
      if (iTemplateFit > -1) and (iTemplateFit < BestFitSoFar) then
      begin
        BestFitSoFar:= iTemplateFit;
        bf:= i;
      end;
    end{i};

    { re-copy RELEVANT channels of new epoch into local template }
    { sites depend on whether creating new or adding to existing template }
    { this is inefficient - but is simple and necessary to deal with 'shifting maxchan' }
    if bf > - 1 then Sites:= SpikeTemplates[bf].Sites;
    index:= 0;
    tchans:= 0;
    if cbDetectRaw.Checked then
    begin //copy epoch from raw buffer...
      wavoffset:= (m_TrigIndex - 10{400µsec}{trigpt}) mod 2500; // MESSY! REMOVE HARDCODING!!!
      for i:= 0 to NumChans - 1 do
        if i in Sites then
        begin
          Move(SurfBawdForm.CR.Waveform[wavoffset + i * 2500], AvgWaveform[index], pts_per_chan * 2);
          inc(index, pts_per_chan);
          inc(tchans);
        end;
    end else
    begin //copy epoch from upsampled buffer...
      wavoffset:= (m_TrigIndex - 10{400µsec} * m_UpSampleFactor) mod 10200; // MESSY! REMOVE HARDCODING!!!
      Setlength(AvgWaveform, NumChans * pts_per_chan * m_UpsampleFactor); //all channels, temporarily...
      for i:= 0 to NumChans - 1 do
        if i in Sites then
        begin
          Move(m_InterpWaveform[wavoffset + i * 10200], AvgWaveform[index], pts_per_chan * m_UpsampleFactor * 2);
          inc(index, pts_per_chan * m_UpsampleFactor);
          inc(tchans);
        end;
    end;
    Setlength(AvgWaveform, index); //...finally, shrink to actual size of newtemplate

    if bf = -1 then
    begin //...no match, so create new template
      CreateNewTemplate(NearSites[m_SpikeMaxMinChan], tchans, index{total template length});
      //next line temporary... remove once add2template has own routine for updating maxchan
      {!}SpikeTemplates[NumTemplates - 1].MaxChan:= MaxChan;{!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!}
      //Add2Template(SpikeTemplates[NumTemplates - 1], AvgWaveform);
      Exit;
    end; //found match, so add to existing template, if not locked...
    if SpikeTemplates[bf].Locked then Exit;
    //Add2Template(SpikeTemplates[bf], AvgWaveform);
    if bf = m_TemplateIndex then
    begin
      BlankTabCanvas;
      PlotTemplate(bf);
    end;
  end;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.StatusBarDblClick(Sender: TObject);
var
  InputString: string;
begin
  //add jump to time, epoch, or ..... functionality, depending on which panel clicked
  InputString:= InputBox('Epoch Select', 'Input buffer/spike # to jump to', '');
  if InputString <> '' then
  try
    TrackBar.Position:= StrToInt(InputString) - 1;
  except
    Showmessage('Invalid entry');
  end;
  //dbl click used to switch off status bar
  {DisplayStatusBar:= not (DisplayStatusBar);
  if DisplayStatusBar then UpdateStatusBar else
  begin
    StatusBar.Panels[0].Text:= '';
    StatusBar.Panels[1].Text:= '';
  end}
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.BuildSiteProximityArray(var SiteArray : TSiteArray ; radius : integer;
                                                inclusive : boolean {default = false});
var i, j : integer;
begin
  {build site-lockout sets for each site}
  with ProbeWin[m_ProbeIndex].electrode do
  begin
    for i:= 0 to SURF_MAX_CHANNELS -1 do
      for j:= 0 to SURF_MAX_CHANNELS -1 do
        if (Sqrt(Sqr(SiteLoc[i].x - SiteLoc[j].x) +
             Sqr(SiteLoc[i].y - SiteLoc[j].y)) < radius)
          and ((i <> j) or inclusive){self} then Include(SiteArray[i], j)
        else
          Exclude(SiteArray[i], j);
  end{with};
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbRethresholdClick(Sender: TObject);
var spk, w, threshold : integer;
  c : byte;
  keepit : boolean;
begin
  ToolBar1.Enabled:= false;
  ToolBar2.Enabled:= false;
  TrackBar.Enabled:= false;

  Threshold:= 2048 + ProbeWin[m_ProbeIndex].win.seThreshold.value;

  for spk:= 0 to m_NEpochs[m_ProbeIndex] do
  begin
    SurfFile.GetSpike(m_ProbeIndex,Spk,Spike);
    Keepit:= false;
    //if seSingleThreshold.Value = -1 then  //check all channels for waveform over threshold
    begin
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans-1 do
      begin
        for w:= ProbeArray[m_ProbeIndex].trigpt to ProbeArray[m_ProbeIndex].pts_per_chan-1 do
        begin
          if Spike.waveform[c,w] > Threshold then
          begin
            Keepit:= True;
            Break;
          end;
        end;
        if Keepit then break;
      end;
    end;// else
    begin //just check single channel for waveform over threshold
      for w:= ProbeArray[m_ProbeIndex].trigpt to ProbeArray[m_ProbeIndex].pts_per_chan-1 do
      begin
        //if Spike.waveform[seSingleThreshold.Value, w] > Threshold then
        begin
          Keepit:= true;
          break;
        end;
      end;
    end;
    if not(keepit) then SurfFile.SetSpikeClusterID(m_ProbeIndex,Spk,-1) //exclude
      else SurfFile.SetSpikeClusterID(m_ProbeIndex,Spk,0); //include (but will overwrite ID)
    TrackBar.Position:= spk;  //implicit way to display waveforms and progress thru file
  end;
  ToolBar1.Enabled:= true;
  ToolBar2.Enabled:= true;
  TrackBar.Enabled:= true;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbToglStatsWinClick(Sender: TObject);
var i : byte;
begin
  if tbToglStatsWin.down = True then
  begin
    WaveformStatsWin := TInfoWin.CreateParented(Waveforms.Handle);
    with waveformstatswin do
    begin
      for i := 0 to High(ParamNames) do
        AddInfoLabel(ParamNames[i]);
      Constraints.MinWidth:= 200;
      Top := Toolbar2.Height + 10;
      Left := ProbeWin[m_ProbeIndex].Win.Width + 10;
      Caption := 'Waveform Parameters';
      BringToFront;
    end;
    WaveFormStatsUpdate;
  end else
  WaveFormStatsWin.Release;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.WaveFormStatsUpdate;
var rx, ry : single;
  i : smallint;
begin
  if not tbToglStatsWin.Down then Exit;
  ComputeWaveformParams(Spike, False, m_CurrentChan);
  with WaveformStatsWin do
  begin
    ChangeInfoData('Channel', inttostr(m_CurrentChan));
    ChangeInfoData('Peak', inttostr(Spike.param[m_CurrentChan,1])+' µV');
    ChangeInfoData('Valley', inttostr(Spike.param[m_CurrentChan,2])+' µV');
    ChangeInfoData('Amplitude', inttostr(Spike.param[m_CurrentChan,3])+' µV');
    ChangeInfoData('TimeAtPeak', inttostr(Spike.param[m_CurrentChan,4]-ProbeArray[m_ProbeIndex].trigpt)+' µs');
    ChangeInfoData('Width', inttostr(Spike.param[m_CurrentChan,5]-Spike.param[m_CurrentChan,4])+' µs');
    ChangeInfoData('Area', inttostr(Spike.param[m_CurrentChan,6])+' µV.ms');
    if Spike.param[m_CurrentChan,0]=-1 then ChangeInfoData('Polarity', 'Inverted')
      else ChangeInfoData('Polarity', 'Positive');

{    for i := 1 to 32 do  // display zoomed waveform...
    begin
      spliney[i] := Spike.Waveform[m_CurrentChan,i-1];
      splinex[i] := i;
      spline2ndderiv[i] := 0;
    end;
    Spline(splinex, spliney,32,spline2ndderiv);    // cubic spline the wave

    Canvas.Brush.Color:= clBtnFace;
    Canvas.FillRect(Rect(97,0,ClientWidth,ClientHeight)); // blank plot space
    Canvas.Brush.Color:= clGray;
    Canvas.MoveTo(100, 65);
    Canvas.LineTo(200, 65);
    for i := 1 to 32 * SplineRes do
    begin
      rx := i/SplineRes;
      Splint(splinex,spliney,spline2ndderiv,32,rx,ry);
      Canvas.Pixels[round((rx-1)*3)+100, 145-round(ry/25)]:= clGray; // plot spline points
    end;

    for i := 0 to 31 do    // now plot the raw waveform data
      Canvas.Pixels[i*3+100, 145-Spike.Waveform[m_CurrentChan, i]div 25]:=clRed;}
  end;
  {  for i:= 0 to high (Spike.param) do
  begin
    WaveformStatsWin.ChangeInfoData(paramnames[i+1],inttostr(Spike.param[i]));
  end;}
end;

{-----------------------------------------------------------------------------}
procedure CProbeWin.NotAcqOnMouseMove (ChanNum : byte);
begin
  SurfBawdForm.m_CurrentChan:= ChanNum;
  SurfBawdForm.WaveFormStatsUpdate; //update waveform stats
end;

{-----------------------------------------------------------------------------}
procedure CProbeWin.ThreshChange(pid, Threshold : integer);
begin
  with SurfBawdForm, ProbeWin[pid] do
  begin
    DispTrigBipolar:= muBipolarTrig.Checked;
    if DispTrigBipolar then
    begin
      m_ProbeThreshold[pid]:= Abs(seThreshold.Value) + 2048;
      PolarityLbl:= '±';
    end else
    begin
      m_ProbeThreshold[pid]:= seThreshold.Value + 2048;
      if seThreshold.Value > 0 then
      begin
        DispTrigPositive:= True;
        PolarityLbl:= '+';
      end else
      begin
        DispTrigPositive:= False;
        PolarityLbl:= '-';
      end;
    end;
    ResetChanLocks;
  end;
end;

{-----------------------------------------------------------------------------}
function TSurfBawdForm.ADValue2uV(ADValue : SmallInt): SmallInt;
begin
  Result:= Round((ADValue - 2048)*(10 / (2048 * ProbeArray[m_ProbeIndex].IntGain
                 * ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan]))
                 * V2uV);
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbToglWformClick(Sender: TObject);
begin
  ProbeWin[m_ProbeIndex].win.FormDblClick(tbToglWform); //blanks waveform window
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbLocNSortClick(Sender: TObject);
begin
  {should make LocSortForm runttime created}
  {should check if not already there...} LocSortForm.Show;
  LocSortForm.BringToFront;
  LocSortForm.CreateElectrode(ProbeArray[m_ProbeIndex].electrode_name,
    ProbeArray[0].IntGain, ProbeArray[0].ExtGain[0]); // what if ext. gain varies from channel to channel?!

{  tbPlay.Down:= true;
  tbPlayClick(nil);}
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.PredNSort;
var z,vo,m,ro,  chi : single;
begin
  if ProbeArray[m_ProbeIndex].ProbeSubType = SPIKEEPOCH {'S'} then //spike record found
    if Spike.Cluster >= 0 then
    begin
      z := 100;//strtofloat(ez.text);
      vo := 10;//strtofloat(evo.text);
      m := 50;//strtofloat(em.text);
      ro := 10;//strtofloat(eo.text);
      if WaveFormStatsWin.Enabled = False then ComputeWaveformParams(Spike);
      LocSortForm.ComputeLoc(Spike, z, vo, m, ro, true, false, false, false, 0, chi
      {LockZ.Checked,LockVo.Checked,LockM.Checked,LockO.Checked,FunctionNum.ItemIndex,chi});
    end;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbRasterPlotClick(Sender: TObject);
var e, index : integer;
  id, current_ori : smallint;
  //CR : TCr;
  SVal : TSVal;
  ori, phase, oribin : array of smallint;
  msb, lsb, c : byte;
  phase_time : array of integer;

begin
  //display raster plot window - should be dynamically created!
  //RasterWin:= TRasterForm.Create(Self);
  RasterForm.Show;
  RasterForm.BringToFront;

  //retrieve all spike times and IDs into public array for RasterWin to plot
  Setlength(m_SpikeTimeStamp, length(SurfFile.GetEventArray)); //allocate space for all units
  Setlength(m_SpikeClusterID, length(SurfFile.GetEventArray));
  index:= 0;
  for e:= 0 to High(SurfFile.GetEventArray) do
  begin
    if SurfFile.GetEventArray[e].SubType = 'S' then
    begin
      id:= SurfFile.GetClusterID(m_ProbeIndex, e);
      if id >= 0 then //currently, shows spike rasters of clustered and unclustered units
      begin
        m_SpikeTimeStamp[index] := SurfFile.GetEventArray[e].Time_Stamp; //need to call every time?
        m_SpikeClusterID[index] := id;
        inc(index);// must be a better way!!!
      end;
    end;
  end;

  //retrive all orientation and phase information
  Setlength (ori, SurfFile.GetNumSVals);
  Setlength (phase, SurfFile.GetNumSVals);
  SetLength (phase_time, SurfFile.GetNumSVals);

  for e:= 0 to SurfFile.GetNumSVals-1 do
  begin
    SurfFile.GetSVal(e, SVal);
    msb := Sval.sval and $00FF; {get the last byte of this word}
    lsb := Sval.sval shr 8;     {get the first byte of this word}
    ori[e] := ((msb and $01) shl 8 + lsb); //get the last bit of the msb
    phase[e] := (msb shr 1); //get the first 7 bits of the msb
    phase_time[e] := sval.time_stamp;
    {Showmessage('Orientation = '+inttostr(Ori[e])
                +', Phase = '+inttostr(phase[e])
                +', at time'
                +inttostr(Sval.time_stamp div 10)
                +'ms');}
  end;

  {for e:= 0 to SurfFile.GetNumCRs(1)-1 do
  begin
    SurfFile.GetCR(1, e, CR);
    Showmessage('CR timestamp = ' + inttostr(Cr.time_stamp)+'; CR = '+ inttostr(CR.waveform[0]));
  end;}

  //build orientation spike histograms

  Setlength (oribin, 23);
  index:= 0;
  e:= 0;
  for c:= 0 to 23 do// for (second) 12 orientation conditions (incl. 0 ori)
  begin
    current_ori := ori[index];
    while ori[index] = current_ori do inc (index); //find index at change of oriention
    inc (index);
    while m_SpikeTimeStamp[e] < phase_time[index] do
    begin
      inc(oribin[c]);
      inc(e);
    end;
    Showmessage('Orientation: '+inttostr(current_ori)
              + ' had '+inttostr(oribin[c])+' spikes.');
  end;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbExportDataClick(Sender: TObject);
begin
  if (ProbeArray[m_ProbeIndex].numchans = 1) or
     (GUICreated and (GUIForm.m_iNSitesSelected > 0)) then
  begin
    ExportDataDialog.FileName:= Copy(FileNameOnly, 0, Length(FileNameOnly) - 4); //remove .srf extension
    if ExportDataDialog.Execute then
      ExportSurfData(ExportDataDialog.FileName);
  end else
  begin
    Showmessage('Select channels for export');
    if not tbToglProbeGUI.Down then
    begin
      tbToglProbeGUI.Down:= True;
      tbToglProbeGUI.Click;
    end;
  end
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.ExportSurfData(Filename : string);
{const ExportSplineRes = 31;}
var
  v : array [1..({ExportSplineRes}100 * 32)]of SmallInt;
  c, w, EpochsToExport, EpochsExported : integer;
  Output : {Text}file of SHRT;
  InfFile : Textfile;
  OutFileName : string;
  AD2uV, AD2usec : Single; {32bit}
begin
  Exporting:= True;
  Screen.Cursor:= crHourGlass;

  if TrackBar.SelEnd = Trackbar.SelStart then //if no range selected...
  begin
    TrackBar.SelEnd:= TrackBar.Position;
    TrackBar.SelStart:= TrackBar.Position; //...export (from) currently displayed spike
  end;
  TrackBar.Position:= TrackBar.SelStart;

  {export textfile containing sval records}
  OutFileName:= Filename;
  if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream' then
  begin
    OutFileName := OutFileName + '.csv';
    AssignFile(InfFile, OutFileName);
    {if FileExists(OutFileName)
      then Append(Output)
      else }Rewrite(InfFile); //overwrites any existing file of the same name
    while TrackBar.Position < (TrackBar.SelEnd) do
    begin
      Writeln(InfFile, inttostr(SVal.time_stamp) + ',' + Inttostr(SVal.sval));
      TrackBar.Position:= TrackBar.Position + 1;
    end;
    Writeln(InfFile, inttostr(SVal.time_stamp) + ',' + Inttostr(SVal.sval)); //write value at selend!
    CloseFile(InfFile);
    Screen.Cursor:= crDefault;
    Exporting:= False;
    Exit;
  end;

  {export binary files with spike data}
  if cbuV.Checked then OutFileName := OutFileName + '[uV].bin'
    else OutFileName := OutFileName + '.bin';
  AssignFile(Output, OutFileName);
  {if FileExists(OutFileName)
    then Append(Output)
    else }Rewrite(Output); //overwrites any existing file of the same name

{!}if TrackBar.Position div 100 {WaveEpochs per Buffer} <> m_CurrentBuffer then //REMOVE HARDCODING!!
  begin //load first buffer at selstart, if required
    m_CurrentBuffer:= TrackBar.Position div 100;
    SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR);
  end;

  GetFileBuffer; //retrieves buffer (with transbuffer samples), centered on m_CurrentBuffer

  ProbeWin[m_ProbeIndex].win.DrawWaveForms:= False; //don't display spikes
  EpochsExported:= 0;
  if cbThreshFilter.Checked then
    tbPause.Down:= True else
  begin
    tbPause.Down:= False; //must be up for continuous exports...
    NumSpikes2Save:= TrackBar.SelEnd - TrackBar.SelStart + 1;
  end;

  repeat
    if cbThreshFilter.Checked then
    begin
      DisplaySpikeFromStream;
      if (EpochsExported = NumSpikes2Save) or
         (TrackBar.Position > TrackBar.SelEnd + 1) then Break;
    end;

    if cbDecimate.Checked then
    begin //export 12.5kHz data (contiguous)...
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
      begin
        if (ProbeArray[m_ProbeIndex].numchans > 1) and not GUIForm.SiteSelected[c] then Continue; //skip site
          for w:= 0 to 11 do
            BlockWrite(Output, CR.Waveform[(TrackBar.Position mod 100)*25 + c*2500 + w*2], 1); //export continuous selection to file
      end{c};
    end else
    if m_UpSampleFactor > 1 then
    begin //write interpolated data...
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
      begin
        if (ProbeArray[m_ProbeIndex].numchans > 1) and not GUIForm.SiteSelected[c] then Continue; //skip site
        if cbThreshFilter.Checked then
        begin
          //BlockWrite(@m_InterpWaveform[m_TrigIndex mod 10200 + m_PosLockedSamp[m_TrigIndex div 10200]
          //    - ProbeArray[m_ProbeIndex].Trigpt * m_UpsampleFactor], 10200{SampPerChanPerBuff}, {m_UpsampleFactor{dispdecimn{}{,} 0{white});

          BlockWrite(Output, m_InterpWaveform[c*10200 + TransBufferSamples
                             + m_TrigIndex mod 10200  - m_PosLockedSamp[m_TrigIndex div 10200]
                             + ProbeArray[m_ProbeIndex].Trigpt * m_UpsampleFactor], 25 * m_UpSampleFactor); //export interpolated waveform epoch to file

          //BlockWrite(Output, m_InterpWaveForm[((TrackBar.Position mod 100) * 25
          //          + c*(2500 + TransBufferSamples*2) + TransBufferSamples - ProbeArray[m_ProbeIndex].Trigpt) * m_UpSampleFactor
          //           + m_PosLockedSamp[m_TrigIndex div 10200]], 25 * m_UpSampleFactor); //export interpolated waveform epoch to file

          {BlockWrite(Output, CR.Waveform[m_TrigIndex mod 2500 + m_PosLockedSamp[m_TrigIndex div 2500]
                     + c*2500 - ProbeArray[m_ProbeIndex].Trigpt], 25);}
        end else
          BlockWrite(Output, m_InterpWaveForm[((TrackBar.Position mod 100) * 25
                     + c*(2500 + TransBufferSamples*2) + TransBufferSamples) * m_UpSampleFactor],
                     25 * m_UpSampleFactor); //export continuous, interpolated waveform to file
      end{c};
    end else
    begin //write raw data...
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
      begin
        if (ProbeArray[m_ProbeIndex].numchans > 1) and not GUIForm.SiteSelected[c] then Continue; //skip site
        if cbThreshFilter.Checked then
        begin
          if ((m_TrigIndex mod 2500) - ProbeArray[m_ProbeIndex].Trigpt -10) < 0 then continue//spikes that traverse...
            (*BlockWrite(Output, CR2.Waveform[2500 + m_TrigIndex mod 2500 //...previous buffer handled differently
{!modified to export 50pts of raw data!}       + c*2500 - ProbeArray[m_ProbeIndex].Trigpt - 10], 50{25}) *)else
            BlockWrite(Output, CR.Waveform[m_TrigIndex mod 2500 +  m_PosLockedSamp[m_TrigIndex div 2500]
                       + c*2500 - ProbeArray[m_ProbeIndex].Trigpt - 10], 50{25});
        end else
          BlockWrite(Output, CR.Waveform[(TrackBar.Position mod 100)*25 + c*2500 - 10], 25);
      end{c};
    end;
    inc(EpochsExported);
    if cbThreshFilter.Checked then
    begin
      if m_UpSampleFactor = 1 then
      begin
        if ((m_TrigIndex mod 2500) + 25 - ProbeArray[m_ProbeIndex].Trigpt) > 2500
          then SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR); //restore CR, if spike traversed forward buffer
        m_TrigIndex:= (m_TrigIndex + 2500) mod 134999 //for spike export, queue to next sample...
      end else
        m_TrigIndex:= (m_TrigIndex + 10200) mod (550800 -1); //for spike export, queue to next sample...
    end else
      TrackBar.Position:= TrackBar.Position + 1; //...for contiguous export, get next epoch
    Application.ProcessMessages;
  until EpochsExported = NumSpikes2Save;

  CloseFile(Output);
  if TrackBar.SelEnd = Trackbar.SelStart then TrackBar.Position:= TrackBar.Position - 1; //keep on currently displayed spike
  Exporting:= False;

  if EpochsExported = 0 then
  begin
    ProbeWin[m_ProbeIndex].win.Canvas.TextOut(8, ProbeWin[m_ProbeIndex].win.tbControl.Height + 5, 'No data exported.      ');
    Screen.Cursor:= crDefault;
    DeleteFile(OutFileName);
    ProbeWin[m_ProbeIndex].win.DrawWaveForms:= tbToglWform.Down;
    Exit;
  end;

  ProbeWin[m_ProbeIndex].win.Canvas.TextOut(8, ProbeWin[m_ProbeIndex].win.tbControl.Height + 5, 'Finished exporting.     ');
  {now write associated .inf file, if cbHeader checked}
  AD2uV:= ((20/ProbeArray[m_ProbeIndex].IntGain) / 4096)
          / ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan] * V2uV;
  AD2usec:= 1/(ProbeArray[m_ProbeIndex].SampFreqPerChan
            * (m_UpSampleFactor) / V2uV);
  if cbHeader.Checked then
  begin
    OutFileName:= FileName;
    OutFileName:= OutFileName + '.inf';
    AssignFile(InfFile, OutFileName);
    {if FileExists(OutFileName)
      then Append(Output)
      else }Rewrite(InfFile); //overwrites any existing file of the same name

    Writeln(InfFile, '%Information for: ''' + ExtractFileName(FileName) + '.bin''');
    Writeln(InfFile, '%SURF source file: ''' + FileNameOnly + '''');
    Writeln(InfFile, '%Export epoch range : ' + inttostr(TrackBar.SelStart) + '-' + inttostr(TrackBar.SelEnd));
    Writeln(InfFile, inttostr(EpochsExported) + ' %number of spikes/epochs'); //#spikes/bufferwins/events
    Writeln(InfFile, floattostr(AD2uV) + ' %uV per ADC value'); //conversion factor(V)
    Writeln(InfFile, floattostr(AD2usec) + ' %sample period usec (interpolated)');
    Writeln(InfFile, inttostr(m_UpSampleFactor -1) + ' %number of interpolated points'); //#interpolated pts

    with ProbeWin[m_ProbeIndex].electrode do
    if ProbeArray[m_ProbeIndex].numchans > 1 then
    begin
      Writeln(InfFile, inttostr(GUIForm.m_iNSitesSelected) + ' %number of channels'); //#channels
      Writeln(InfFile, '%channel list follows (ch, x coords, y coords)...');
      for c:= 0 to NumSites - 1 do
      begin
        if not GUIForm.SiteSelected[c] and (NumSites > 1) then Continue; //skip site
        Writeln(InfFile, inttostr(c) + ', ' + inttostr(SiteLoc[c].x) + ', '
                                            + inttostr(SiteLoc[c].y));
      end;
    end else
      Writeln(InfFile, '1 %single channel'); //single channel
    CloseFile(InfFile);
  end;
  Screen.Cursor:= crDefault;
  if ProbeArray[m_ProbeIndex].numchans > 1 then
    Showmessage(inttostr(GUIForm.m_iNSitesSelected) + ' site(s), ' + inttostr(EpochsExported) + ' epoch(s)/spike(s) exported.'+ #13#10 +
                floattostr(AD2uV)  + ' uV/ADC' + #13#10 +
                floattostr(AD2uSec)+ ' usec/samplept.')
  else Showmessage(inttostr(EpochsExported) + ' epoch(s)/spike(s) extracted from ' + ProbeArray[m_ProbeIndex].probe_descrip + #13#10 +
                floattostr(AD2uV)  + ' uV/ADC' + #13#10 +
                floattostr(AD2uSec)+ ' usec/samplept.');
  ProbeWin[m_ProbeIndex].win.DrawWaveForms:= tbToglWform.Down;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.SpinEdit1Change(Sender: TObject);
begin
  NumSpikes2Save:= SpinEdit1.Value;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.SpinEdit2Change(Sender: TObject);
begin
 // if SizeOf(StimHdr) = 0 then
  if not(SurfFile.GetStimulusRecord(0, StimHdr)) then Showmessage('Could not read');;
  Label3.Caption:= floattostr(StimHdr.parameter_tbl[spinedit2.value]);
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.seFactorChange(Sender: TObject);
begin
  m_UpSampleFactor:= seFactor.Value;
  {re}GenerateSincKernels;
  if m_UpSampleFactor = 1 then
  begin
    lblUpsample.Caption:= '(raw)';
    cbSHCorrect.Enabled:= False;
    cbAlignData.Enabled:= False;
  end else
  begin
    lblUpsample.Caption:= '('+inttostr((m_UpSampleFactor * m_HardWareInfo.iADCRetriggerFreq) div 1000) + 'kHz)';
    cbSHCorrect.Enabled:= True;
    cbAlignData.Enabled:= True;
  end;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.cbThreshFilterClick(Sender: TObject);
begin
  SpinEdit1.Enabled:= cbThreshFilter.Checked;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.FormCreate(Sender: TObject);
begin
  AddElectrodeTypes;
  NumSpikes2Save:= SpinEdit1.Value;
  seFactor.Value:= DefaultUpsampleFactor;
  m_UpSampleFactor:= seFactor.Value;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.AddElectrodeTypes;
var e : integer;
begin
  CElectrode.Items.Clear;
  Exit;
  for e := 0 to KNOWNELECTRODES-1 {from ElectrodeTypes} do
    CElectrode.Items.Add(KnownElectrode[e].Name);
  CElectrode.Items.Add('UnDefined');
  CElectrode.ItemIndex := KNOWNELECTRODES;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.CreateProbeWin(const Electrode : TElectrode);
begin
  ProbeWin[m_ProbeIndex].win:= CProbeWin.CreateParented(SurfBawdForm.Handle);
  ProbeWin[m_ProbeIndex].win.show;
  ProbeWin[m_ProbeIndex].exists:= True;
  ProbeWin[m_ProbeIndex].win.InitPlotWin(Electrode,
                            {npts}ProbeArray[m_ProbeIndex].pts_per_chan,
                            {left}ProbeArray[m_ProbeIndex].ProbeWinLayout.Left,
                            {top}ProbeArray[m_ProbeIndex].ProbeWinLayout.Top+ToolBar1.Height + 20,
                            {thresh}ProbeArray[m_ProbeIndex].Threshold - 2048, //revert to signed thresholds for setup
                            {trigpt}ProbeArray[m_ProbeIndex].TrigPt,
                            {probeid}m_ProbeIndex,
                            {probetype}ProbeArray[m_ProbeIndex].ProbeSubType,
                            {title}ProbeArray[m_ProbeIndex].probe_descrip,
                            {acquisitionmode}{TRUE}False,
                            {intgain}ProbeArray[m_ProbeIndex].intgain,
    {!!!}                   {extgain}5000{ProbeArray[m_ProbeIndex].ExtGain[]},
    {100000, 4});
  if SurfBawdForm.ClientWidth < ProbeWin[m_ProbeIndex].win.Width then SurfBawdForm.ClientWidth := ProbeWin[m_ProbeIndex].win.Width;
  if SurfBawdForm.ClientHeight  < StatusBar.height + ToolBar1.Height + ProbeWin[m_ProbeIndex].win.Height + 25
    then SurfBawdForm.ClientHeight  := StatusBar.height + ToolBar1.Height + WaveForms.Top + ProbeWin[m_ProbeIndex].win.Height + 30;
  ProbeWin[m_ProbeIndex].win.Visible := TRUE;
  if ProbeArray[m_ProbeIndex].ProbeSubType <> CONTINUOUS then
    ProbeWin[m_ProbeIndex].win.seThreshold.OnChange(Self); //initialise threshold
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbToglProbeGUIClick(Sender: TObject);
begin
  AddRemoveProbeGUI;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.AddRemoveProbeGUI;
begin
  if ProbeArray[m_ProbeIndex].numchans < 2 then
    tbToglProbeGUI.Down:= False;
  if tbToglProbeGUI.Down = False then
  begin
    if GUICreated then GUIForm.Close;
    GUICreated:= False;
    Exit;
  end else
  try
    if GUICreated then GUIForm.Close; //remove any existing GUIForms...
    GUIForm:= TPolytrodeGUIForm.CreateParented(SurfBawdForm.Handle);//..and create a new one
    with GUIForm do
    begin
      Left:= 292;
      Top:= 130;
      Show;
      BringToFront;
      GUICreated:= True;
    end;
  except
    GUICreated:= False;
    Exit;
  end;

  if not GUIForm.CreateElectrode(ProbeArray[m_ProbeIndex].electrode_name) then
  begin
    GUIForm.Free;
    GUICreated:= False;
    Exit;
  end;

  if GUIForm.Height > SurfBawdForm.Height - 100 then GUIForm.Height:= SurfBawdForm.Height - 100;
  GUIForm.Caption:= ProbeArray[m_ProbeIndex].electrode_name;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.Button1Click(Sender: TObject);
var i, j : integer;
begin
  if not GUICreated then
  begin
    tbToglProbeGUI.Down:= True;
    AddRemoveProbeGUI;
    if not GUICreated then Exit;
  end;
  {show user site lockout range}
  for j:= 0 to 53 do
  begin
    for i:= 0 to 53 do
      if i in m_NearSites[j] then GUIForm.ChangeSiteColor(i, clRed)
        else GUIForm.ChangeSiteColor(i, clLime);
    GUIForm.ChangeSiteColor(j, clOrange); //highlight trigger channel
    GUIForm.Refresh;
    Application.ProcessMessages;
    Sleep(200);
  end{j};
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.seLockRadiusChange(Sender: TObject);
begin
  BuildSiteProximityArray(m_NearSites, seLockRadius.Value, True);
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.seLockTimeChange(Sender: TObject);
begin
  if cbDetectRaw.Checked then
    m_SimpleLockSamples:= Round(seLockTime.Value * ProbeArray[m_ProbeIndex].sampfreqperchan / 1000000)
  else m_SimpleLockSamples:= Round(seLockTime.Value * ProbeArray[m_ProbeIndex].sampfreqperchan / 1000000 * seFactor.Value);
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbPauseClick(Sender: TObject);
begin
  if tbReversePlay.Down then tbPause.Down:= False //no reverse play-search mode
    else if not tbPause.Down and tbPlay.Down then PlayTimer.Enabled:= True;
  ResetChanLocks;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.ResetChanLocks;
var i : integer;
begin
  for i:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
  begin
    m_PosLockedSamp[i]:= 0;
    m_NegLockedSamp[i]:= 0;
    //m_LockedSites:= []{empty set};
  end;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.PlayTimerTimer(Sender: TObject);
begin
  with TrackBar do
  begin
    if tbPause.Down then
    begin //search for spike
      PlayTimer.Enabled:= False; //search can take longer than timer interval
      DisplaySpikeFromStream;
      PlayTimer.Enabled:= True;
    end else
    begin //step through epoch by epoch
    if tbPlay.Down then Position:= Position + 1
      else Position:= Position - 1;
    end;
    if (Position = 0) or (Position = Max) or not(tbPlay.Down or tbReversePlay.Down) then
    begin
      tbPlay.Down:= False;
      tbReversePlay.Down:= False;
      tbPause.Down:= False;
      PlayTimer.Enabled:= False;
      GroupBox2.Enabled:= True;
    end;
  end{with};
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.GenerateSincKernels;
{generates multiple Hamming-windowed sinc kernels for upsampling raw waveform data.
 An 'upsample' # of kernels are needed per S:H delay for each channel in the CGlist.
 Presence of DIN channels (cgl_entry 0) are also taken into account, such that
 the analog channel S:H delays for all channels of the first two boards are shifted 1usec}
var i, j : integer;
  interpOffset, SHdelayOffset : TReal32;
begin
  Setlength(m_SincKernelArray, DT3010_MAX_SE_CHANS + 1{for DIN}, m_UpSampleFactor);
  for i:= 0 to High(m_SincKernelArray) do
  begin //one kernel for each S:H delay...
    if Cat9SpecialSincs then //!!!!!temporary sinc adj. to align boards from cat9 data
      SHdelayOffset:= 1/m_HardwareInfo.iMasterClockFreq*m_HardwareInfo.iADCRetriggerFreq*(i-9)
    else SHdelayOffset:= 1/m_HardwareInfo.iMasterClockFreq*m_HardwareInfo.iADCRetriggerFreq*i;
    for j:= 0 to m_UpSampleFactor - 1 do
    begin //one kernel for each upsample point
      interpOffset:= j/m_UpSampleFactor;
      MakeSincKernel(m_SincKernelArray[i,j], SincKernelLength, interpOffset - SHdelayOffset);
    end{j};
    if not cbSHcorrect.Checked or cbSHCorrect.Enabled = False then
    begin //only zero-phase kernels needed if S:H correction not selected
      for j:= 1 to DT3010_MAX_SE_CHANS - 1 do
        m_sincKernelArray[j]:= nil; //free memory for unused kernels
      Exit;
    end;
  end{i};
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.UpSampleWaveform(const RawWaveform : TWaveform;
                                         const UpSampledWaveform : TWaveform;
                                               NPtsInterpolation : integer);
var w, NRawPts : integer;
    rx, ry : TReal32{single};
begin
  NRawPts:= Length(RawWaveform);
  if (NRawPts * NPtsInterpolation) <> Length(UpSampledWaveform) then Exit;
  {allocate/declare at create or user change to npts?}
  Setlength(SplineX, NRawPts + 1); //?! one based!
  Setlength(SplineY, NRawPts + 1);
  Setlength(Spline2ndDeriv, NRawPts + 1);

  for w:= 1 to NRawPts{-1} do
  begin
    SplineY[w]:= RawWaveform[w-1];
    SplineX[w]:= w;
    Spline2ndDeriv[w]:= 0;
  end;

  //Spline(SplineX, SplineY, Spline2ndDeriv); // cubic spline the current buffer

  for w:= 1 to Length(UpSampledWaveform){-1} do
  begin
    rx:= w/(NPtsInterpolation);
    Splint(SplineX, SplineY, Spline2ndDeriv, rx, ry);
    UpSampledWaveform[w-1]:= Round(ry);
    {if cbuV.Checked then v[w]:= ADValue2uV(SmallInt(Round(ry)))
      else v[w]:= SmallInt(Round(ry)-2048);}
  end;

end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.cbSHcorrectClick(Sender: TObject);
begin
  {re}GenerateSincKernels;
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.cbDetectRawClick(Sender: TObject);
begin
  ResetChanLocks;
  if cbDetectRaw.Checked then
    m_TrigIndex:= (TrackBar.Position mod 100) * 25 //REMOVE HARDCODING!!!!
  else m_TrigIndex:= ((TrackBar.Position mod 100) * 25) * m_UpSampleFactor;
  seLockTime.OnChange(Self); //update global m_simplelocksamples var
 { m_TrigIndex:= (m_TrigIndex - TransBufferSamples * m_UpSampleFactor) div m_UpSampleFactor;}
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.GetFileBuffer;
var c, j, i, KernelLength, HalfKernelLength, OneChanBufferLength, BufferPadLength, numChans : integer;
    pInterp : LPUSHRT;
//    freq, before, after : Int64;
begin
  ///////////////////////////////////////////////////////////////
  //S:H CORRECTION ASSUMES EVERY PROBE'S CGLIST STARTS AT ZERO!//
  //NEED TO RETRIEVE ACTUAL CGLIST AND MAP PROBES POSN TO IT!!!//
  ///////////////////////////////////////////////////////////////

  {get three adjacent raw waveform buffers}
  //CR1:= Copy(CR2, 0, Length(CR1)); <--- OPTIMIZE! add check for whether file buffer already...
  //...in CR1, CR2, CR3, then exchange pointers rather than reloading all three buffers!

  if not BinFileLoaded then //if not bin file in memory...
  begin

  SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR); //get central CR buffer
  if (m_UpSampleFactor = 1) or (not Exporting and cbDetectRaw.Checked) then Exit; //no need to load 'side' buffers or upsample
  if m_CurrentBuffer < 1 then //at first raw data buffer in file...
  begin //... so pad left side with DC
    SetLength(CR2.Waveform, Length(CR.Waveform));
    for i:= 0 to Length(CR2.Waveform)-1 do CR2.Waveform[i]:= 2047{CR.Waveform[0]}; //zero volts
    //FillChar(CR2.Waveform[0], Length(CR2.Waveform)*2, 0);
    SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer + 1, CR3);
  end else
  if m_CurrentBuffer = m_NEpochs[m_ProbeIndex] - 1 then //at last raw data buffer in file...
  begin //... so pad right side with DC
    SetLength(CR3.Waveform, Length(CR.Waveform));
    //FillChar(CR3.Waveform[0], Length(CR3.Waveform)*2, 0);
    for i:= 0 to Length(CR3.Waveform)-1 do CR3.Waveform[i]:= 2047{CR.Waveform[High(CR.Waveform)]}; //zero volts
    SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer - 1, CR2);
  end else
  begin //get raw file buffers either side of m_CurrentBuffer...
    SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer - 1, CR2);
    SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer + 1, CR3);
  end;

  end{HACK};

  KernelLength:= Length(m_SincKernelArray[0,0]);
  HalfKernelLength:= KernelLength div 2;
  NumChans:= ProbeArray[m_ProbeIndex].numchans;
  OneChanBufferLength:= {2500;// UNSPECIFIED FOR CAT 9 ONLY!!!} ProbeArray[m_ProbeIndex].pts_per_buffer div NumChans;

  {allocate memory for concatenated raw data and interpolated waveform (eventually to be in CElectrode change, so not reallocated repeatedly)}
  SetLength(m_ConcatRawBuffer, OneChanBufferLength + KernelLength + (TransBufferSamples * 2)); //<-- make buffer local?
  if BinFileLoaded then
    SetLength(m_InterpWaveform, m_UpSampleFactor * (135000 + numChans * TransBufferSamples * 2))
  else SetLength(m_InterpWaveform, m_UpSampleFactor * (Length(CR.Waveform) + numChans * TransBufferSamples * 2));
  {combine 3 raw buffers into 1 contiguous single channel buffer for upsampling}
  BufferPadLength:= HalfKernelLength + TransBufferSamples;
  pInterp:= @m_InterpWaveform[0];

  if BinFileLoaded then //if not bin file in memory...
  begin //HACK...
    if m_CurrentBuffer = 0 then //at first raw data buffer in file...
    begin //... so pad left side with DC
    for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
    begin
      for i:= 0 to 99 do m_InterpWaveform[c*10200 + i]:= 2047{CR.Waveform[0]}; //zero volts
      Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer + c*10000], m_InterpWaveform[c*10200 + 100], 20000);
      Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer + c*10000 + 10000], m_InterpWaveform[c*10200 + 10100], 200);
    end{c}
    end else
    if m_CurrentBuffer = m_NEpochs[0] - 1 then //at last raw data buffer in file...
    begin
    for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
    begin //... so pad right side with DC
      Move(WholeBinFile[(m_CurrentBuffer - 1) * ProbeArray[0].pts_per_buffer + 10000 + c*10000], m_InterpWaveform[c*10200], 200);
      Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer + c*10000], m_InterpWaveform[c*10200 + 100], 20000);
      for i:= 0 to 99 do m_InterpWaveform[c*10200 + i + 10100]:= 2047{CR.Waveform[0]}; //zero volts
    end{c};
    end else
    begin
    for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
    begin
      Move(WholeBinFile[(m_CurrentBuffer - 1) * ProbeArray[0].pts_per_buffer + 10000 + c*10000], m_InterpWaveform[c*10200], 200);
      Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer + c*10000], m_InterpWaveform[c*10200 + 100], 20000);
      Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer + 10000 + c*10000], m_InterpWaveform[c*10200 + 10100], 200);
    end{c};
    end;

    (*if m_CurrentBuffer = 0 then
      Move(WholeBinFile[0], m_InterpWaveform[0], Length(m_InterpWaveform) * 2)
    else
      Move(WholeBinFile[m_CurrentBuffer * ProbeArray[0].pts_per_buffer - 100], m_InterpWaveform[0], Length(m_InterpWaveform) * 2);*)
    CR.time_stamp:= m_CurrentBuffer * 100000;
    Exit;
  end{HACK};

  if cbAlignData.Checked and cbAlignData.Enabled then
  begin //temporary procedure to correct for board asynchrony with cat 9 dataset
    cbSHcorrect.Checked:= True;
    {re}GenerateSincKernels;
    for c:= 0 to 31 do //first 32 channels, nothing different
    begin        //ProbeArray[m_ProbeIndex].sh_delay_offset
      Move(CR2.Waveform[(c+1)*OneChanBufferLength - BufferPadLength], m_ConcatRawBuffer[0], BufferPadLength * 2); //copy CR-1 samples into start of concatenated raw buffer
      Move(CR.Waveform[c*OneChanBufferLength], m_ConcatRawBuffer[BufferPadLength], OneChanBufferLength * 2); //copy CR samples it into middle of concatenated buffer
      Move(CR3.Waveform[c*OneChanBufferLength], m_ConcatRawBuffer[BufferPadLength + OneChanBufferLength], BufferPadLength * 2);//copy CR+1 samples onto end of concatenated raw buffer
      Upsample(m_UpsampleFactor, m_SincKernelArray[c+1{DIN was recorded with ALL cat 9 data}],
               m_ConcatRawBuffer, pInterp);
    end;
    Cat9SpecialSincs:= True;
    {re}GenerateSincKernels;
    for c:= 32 to 53 do //channels from second board need special handling...
    begin
      Move(CR2.Waveform[(c+1)*OneChanBufferLength - BufferPadLength - 1], m_ConcatRawBuffer[0], BufferPadLength * 2 + 1); //copy CR-1 samples into start of concatenated raw buffer
      Move(CR.Waveform[c*OneChanBufferLength], m_ConcatRawBuffer[BufferPadLength + 1], OneChanBufferLength * 2); //copy CR samples it into middle of concatenated buffer
      Move(CR3.Waveform[c*OneChanBufferLength], m_ConcatRawBuffer[BufferPadLength + 1 + OneChanBufferLength], BufferPadLength * 2 - 1);//copy CR+1 samples onto end of concatenated raw buffer
      Upsample(m_UpsampleFactor, m_SincKernelArray[(c mod DT3010_MAX_SE_CHANS) + 1{+DIN}],
               m_ConcatRawBuffer, pInterp);
    end{c};
    Cat9SpecialSincs:= False;
    {re}GenerateSincKernels; //restore kernels for first board chans
  end{align cat9 data}else
  begin //permanent upsample/s:h correct procedure follows...
    //if m_NSVals > 0 then Showmessage('With DIN') else Showmessage('Without DIN'); //<-- ACQDIN boolean MISSING FROM CAT 10!
    //QueryPerformanceFrequency(Freq);
    //QueryPerformanceCounter(Before);

    for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
    begin
      Move(CR2.Waveform[(c+1)*OneChanBufferLength - BufferPadLength], m_ConcatRawBuffer[0], BufferPadLength * 2); //copy CR-1 samples into start of concatenated raw buffer
      Move(CR.Waveform[c*OneChanBufferLength], m_ConcatRawBuffer[BufferPadLength], OneChanBufferLength * 2); //copy CR samples it into middle of concatenated buffer
      Move(CR3.Waveform[c*OneChanBufferLength], m_ConcatRawBuffer[BufferPadLength + OneChanBufferLength], BufferPadLength * 2);//copy CR+1 samples onto end of concatenated raw buffer
      //NOT YET FINISHED... NEED TO RECONCILE APPROPRIATE KERNEL WITH CGLIST-BASED SHoFFSET WITH/WITHOUT DIN!!
      if cbSHcorrect.Checked then Upsample(m_UpsampleFactor,
        m_SincKernelArray[((ProbeArray[m_ProbeIndex].sh_delay_offset + c) mod (DT3010_MAX_SE_CHANS + 1))], m_ConcatRawBuffer, pInterp)
      else Upsample(m_UpsampleFactor, m_SincKernelArray[0], m_ConcatRawBuffer, pInterp);
    end{c};
  end;
//  QueryPerformanceCounter(After);
//  Showmessage(inttostr((After - Before)* 1000 div freq)+ 'msec');
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbChartWinClick(Sender: TObject);
var i : integer; AllChans : boolean;
begin
  if ChartWinCreated then
  begin
    ChartWin.Release;
    ChartWinCreated:= False;
  end;
  try
    ChartWin:= CChartWin.Create(Self);
    with ChartWin do
    begin
      { remove online-only toolbuttons }
      tbStartStop.Free;
      tbLUT.Free;
      tbOpen.Free;
      tbSave.Free;
      tbFS.Free;
      tbOneShot.Free;
      NumWavPts:= ProbeArray[m_ProbeIndex].pts_per_buffer div ProbeArray[m_ProbeIndex].numchans;
      if not GUICreated or (GUIForm.m_iNSitesSelected = 0) then AllChans:= True
        else AllChans:= False;
      if AllChans then NumChans:= ProbeArray[m_ProbeIndex].numchans
        else NumChans:= GUIForm.m_iNSitesSelected;
      for i:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
        if AllChans or GUIForm.SiteSelected[i] then SitesSelected[i]:= True
          else SitesSelected[i]:= False;
      for i:= 0 to KNOWNELECTRODES - 1 {from ElectrodeTypes} do //select current electrode
        if ProbeArray[m_ProbeIndex].electrode_name = KnownElectrode[i].name then
          CElectrode.ItemIndex:= i;
      OnResize(Self);//recompute y indicies depending on nchans
      CElectrode.OnChange(Self);
      Show;
    end;
    ChartWinCreated:= True;
  except
    ChartWinCreated:= False;
  end;
  TrackBar.SetFocus; //return keybd control to trackbar
end;

{-----------------------------------------------------------------------------}
procedure CChartWin.MoveTimeMarker(MouseXPercent : Single);
begin
  SurfBawdForm.tbPause.Down:= False;
  with SurfBawdForm.TrackBar do
    Position:= Position div 100 * 100{start of current buffer} + Round(MouseXPercent * 100);//remove hardcoding!
end;

{-----------------------------------------------------------------------------}
procedure CChartWin.RefreshChartPlot;
begin
  PlotChart(@SurfBawdForm.CR.Waveform[0], 2500); //replot current buffer (eg. upon chartwin resize)
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbTemplateWinClick(Sender: TObject);
begin
  if TemplateWinCreated then
  begin
    BuildTemplateMode:= tbTemplateWin.Down;
    Exit;
    //TemplateWin.Release;
    //TemplateWinCreated:= False;
  end;
  if CreateTemplateWin then BuildTemplateMode:= True;
end;

{-----------------------------------------------------------------------------}
function TSurfBawdForm.CreateTemplateWin : boolean;
//var AllChans : boolean;
begin
  try
    TemplWin:= CTemplateWin.CreateParented(SurfBawdForm.Handle);
    with TemplWin do
    begin
      SourceFile:= FileNameOnly;
      Electrode:= ProbeWin[m_ProbeIndex].Electrode;
      ViewRawPlotsEnabled:= True;
      FitToFileEnabled:= True;
      NumWavPts:= ProbeArray[m_ProbeIndex].pts_per_buffer div ProbeArray[m_ProbeIndex].numchans;
      AD2uV:= ((20/ProbeArray[m_ProbeIndex].IntGain) / RESOLUTION_12_BIT)
               / ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan] * V2uV;
      AD2usec:= 1/(ProbeArray[m_ProbeIndex].SampFreqPerChan
                * (m_UpSampleFactor) / V2uV);

      NumChans:= ProbeArray[m_ProbeIndex].numchans;
      (*
      AllChans:= True;
      if not GUICreated or (GUIForm.m_iNSitesSelected = 0) then AllChans:= True
        else AllChans:= False;
      if AllChans then NumChans:= ProbeArray[m_ProbeIndex].numchans
        else NumChans:= GUIForm.m_iNSitesSelected;
      *)
      BuildSiteProximityArray(NearSites, seTempRadius.Value, True);
      BuildSiteProximityArray(AdjacentSites, 80{um}, True); //REMOVE HARDCODING
//      for i:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
//      for i:= 0 to KNOWNELECTRODES - 1 {from ElectrodeTypes} do //select current electrode
//        if ProbeArray[m_ProbeIndex].electrode_name = KnownElectrode[i].name then
//          CElectrode.ItemIndex:= i;
      Caption:= 'Spike Templates';
//      OnResize(Self);//recompute y indicies depending on nchans
      Left:= 50;
      Top:= 250;
      Show;
    end;
    TemplateWinCreated:= True;
  except
    TemplateWinCreated:= False;
  end;
  TrackBar.SetFocus; //return keybd control to trackbar
end;

{-----------------------------------------------------------------------------}
procedure TSurfBawdForm.tbFindTemplatesClick(Sender: TObject);
begin
(*  if not(TemplateWinCreated and (TemplWin.NumTemplates > 0)) then
    tbFindTemplates.Down:= False;
  FindTemplateMode:= tbFindTemplates.Down;*)
(*  if (m_SelStartCRIdx >= 0) or (m_SelStartSValIdx >= 0) then
  begin
    ExportDataDialog.FileName:= Copy(FileNameOnly, 0, Length(FileNameOnly) - 4); //remove .srf extension
    if ExportDataDialog.Execute then
      ExportEventData(ExportDataDialog.FileName);
  end else
  begin
    Showmessage('Please select data for export');
    Exit;
  end;*)
  if (TemplateWinCreated and
     (TemplWin.NumTemplates > 0) and
      TemplWin.FitToFileEnabled) then
    RipFileWithTemplates;//FindTemplates;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.ExportEventData(Filename : string);
var i : integer; SpikeTime : int64; OutFileName : string;
begin
  Showmessage('Pow!');
  Exporting:= True;
  Screen.Cursor:= crHourGlass;
  FileProgressWin:= TFileProgressWin.CreateParented(SurfBawdForm.Handle);
  //if cbExportSVALs.Checked then //export digital data svals...
  try
    OutFileName:= FileName + '_digital.bin';
    m_ExportStream:= TFileStream{64}.Create(OutFileName, fmCreate);
    m_ExportStream.Seek{64}(0, soFromBeginning); //overwrite any existing file

    with FileProgressWin do
    begin
      FileProgress.MaxValue:= TrackBar.SelEnd;//m_SelEndSValIdx;
      FileProgress.MinValue:= TrackBar.SelStart;//m_SelStartSValIdx;
      FileProgressWin.Caption:= 'Exporting digital data to ' + OutFileName + '...';
      Show;
      BringToFront;
    end;

    if cbHeader.Checked then
    begin //export stimulus display header...
      SurfFile.GetStimulusRecord(m_SelStimHdrIdx, StimHdr);
      m_ExportStream.WriteBuffer(StimHdr, SizeOf(StimHdr));
    end;

    i:= TrackBar.SelStart;//m_SelStartSValIdx; //export all digital svals associated with header...
    while i <= TrackBar.SelEnd{m_SelEndSValIdx} do
    begin
      SurfFile.GetSVal(i, SVal);
      //inc(SVal.Time_stamp, 494700000);
      m_ExportStream.WriteBuffer(SVal.Time_stamp, SizeOf(SVal.Time_stamp));
      m_ExportStream.WriteBuffer(SVal.SVal, SizeOf(SVal.SVal));
      inc(i);
      FileProgressWin.FileProgress.Progress:= i;
      if FileProgressWin.ESCPressed then Break;
    end;
    Screen.Cursor:= crDefault;
    Showmessage('Finished exporting stimulus header with '+ inttostr(i - m_SelStartSValIdx - 1) +' digital records to file');
    m_ExportStream.Free;
  except
    Screen.Cursor:= crDefault;
    Showmessage('Error exporting digital data to file');
  end;

  FileProgressWin.Release;
  Exporting:= False;
  Exit;

  if cbExportTimes.Checked then //export spike times according to threshold-based event detection...
  try
    with FileProgressWin do
    begin
      OutFileName:= FileName + '_spiketimes.bin';
      m_ExportStream:= TFileStream{64}.Create(OutFileName, fmCreate);
      m_ExportStream.Seek{64}(0, soFromBeginning); //overwrite any existing file
      FileProgress.MinValue:= {Trackbar.SelStart}m_SelStartCRIdx;
      FileProgress.MaxValue:= {Trackbar.SelEnd}m_SelEndCRIdx;
      FileProgressWin.Caption:= 'Exporting spike event times ' + OutFileName + '...';
      Show;
      BringToFront;
    end;
    with TrackBar do
    begin
      Position:= SelStart; //jump to first CR for in range, if not already there
      tbPause.Down:= True;
      FileProgressWin.SetFocus;
      i:= 0; //spike event counter  KNOWN BUGS:
      while Position <= SelEnd do //1. ONLY CATCHES FIRST SPIKE IN LAST BUFFER IF SELEND IS LAST BUFFER IN THE FILE...
      begin                       //2. ADDS SPURIOUS SPIKE EXPORT OUTSIDE SELEND RANGE
        DisplaySpikeFromStream;
        SpikeTime:= CR.Time_Stamp + int64(m_TrigIndex) mod 2500 * 40{sample period};
        m_ExportStream.WriteBuffer(SpikeTime, SizeOf(SpikeTime));
        inc(i);
        if ((m_TrigIndex mod 2500) + 25 - ProbeArray[m_ProbeIndex].Trigpt) > 2500
          then SurfFile.GetCR(m_ProbeIndex, m_CurrentBuffer, CR); //restore CR, if spike traversed forward buffer
        m_TrigIndex:= (m_TrigIndex + 2500) mod 134999; //queue to next sample...
        FileProgressWin.FileProgress.Progress:= m_CurrentBuffer;//TrackBar.Position;
        if FileProgressWin.ESCPressed then Break;
      end;
    end;
    m_ExportStream.Free;
    Screen.Cursor:= crDefault;
    Showmessage('Finished exporting '+ inttostr(i) + ' spike times to file');
  except
    Screen.Cursor:= crDefault;
    Showmessage('Error exporting spike times to file');
  end;
  FileProgressWin.Release;
  Exporting:= False;
end;

{---------------------------------------------------------------------------}
procedure CTemplateWin.ChangeTab(TabIndex : integer);
const VTetSites : array[0..33] of TSites = ([0, 1, 35, 18], [18, 52, 51, 35], [1, 2, 19, 35],
       [35, 19, 53, 51], [2, 3, 34, 19], [19, 34, 50, 53], [3, 4, 20, 34], [34, 20, 49, 50],
       [4, 5, 33, 20], [20, 33, {48,} 49], [5, 6, 21, 33], [33, 21, 47{, 48}],
       [6, 7, 32, 21], [21, 32, 46, 47], [7, 8, 22, 32], [32, 22, 45, 46], [8, 9, 31, 22],
       [22, 31, {44,} 45], [9, 10, 23, 31], [31, 23, 43{, 44}], [10, 11, 30, 23], [23, 30, 42, 43],
       [11, {12,} 26, 30], [30, 26, 41, 42], [{12,} 13, 29, 26], [26, 29, 40, 41], [13, 14, 24, 29],
       [29, 24, 36, 40], [14, 15, 28, 24], [24, 28, 39, 36], [15, 17, 25, 28],[28, 25, {37,} 39],
       [17, 16, 27, 25], [25, 27, 38{, 37}]);//faulty channel, exclude
var i, j : integer;
  nullSites: TSites;
begin
  with SurfBawdForm do
  begin
    m_TemplateIndex:= TabIndex;
    if ControlKeyDown and (TabSelectA > 1) and (m_TemplateIndex > -1) then //combine two templates...
    begin
      BlankTabCanvas;
      PlotTemplate(TabTemplateOrder[TabSelectA - 2], ColorTable[TabTemplateOrder[TabSelectA - 2] mod (High(ColorTable)+1)]);
      PlotTemplate(TabTemplateOrder[m_TemplateIndex], ColorTable[TabTemplateOrder[m_TemplateIndex] mod (High(ColorTable)+1)]);
      if MessageDlg('Combine these templates?', mtConfirmation, [mbYes, mbNo], 0) = mrYes then
      begin
        CombineTemplates(TabTemplateOrder[TabSelectA - 2], TabTemplateOrder[m_TemplateIndex]);
        cbShowFits.Enabled:= False;
        ShowTemplateMode:= False;
        ControlKeyDown:= False; //shouldn't be necessary
        TabControlChange(Self); //refresh
        Exit;
      end;
      ControlKeyDown:= False; //shouldn't be necessary
    end;
    if ShowTemplateMode and FitHistoWinCreated and (m_TemplateIndex > -1) then
      with FitHistoWin do
      begin
        if GlobalFitResults[TabTemplateOrder[m_TemplateIndex]].Residuals <> nil then
        begin { update histogram with current template residuals }
          Reset{Histogram};
          FillHistogram(GlobalFitResults[TabTemplateOrder[m_TemplateIndex]].Residuals, seNHistBins.Value);
          PlotHistogram;
          PlotGUIMarker(Round(SpikeTemplates[TabTemplateOrder[m_TemplateIndex]].FitThreshold
                              / Max * ClientWidth));
        end else
        begin
          Reset;
          Resize; { clear histowin canvas }
        end;
      end{fithistowin};
    (*if ShowTemplateMode and FitHistoWinCreated and (m_TemplateIndex > -1) then
      with ISIHistoWin do
      begin
        UpdateISIHistogram;
      end{isihistowin};*)
    if m_TemplateIndex = -1 then //overplot all templates...
    begin
      UpDown1.Max:= high(VTetSites);
      BlankTabCanvas;
      j:= 0;
      nullSites:= [];
      for i:= 0 to NumTemplates -1 do
        if (SpikeTemplates[i].Enabled) and (SpikeTemplates[i].Sites * VTetSites[UpDown1.Position] <> nullSites) then
        begin
          PlotTemplate(i, ColorTable[i mod (High(ColorTable)+1)]); //plot avg templates
          inc(j);
        end;
      TemplWin.Label3.Caption:= inttostr(updown1.Position);
      //TemplWin.Label3.Caption:= inttostr(j) + ' per tet';
      //TemplWin.Label3.Caption:= inttostr(j) + ' of ' + inttostr(NumTemplates);
      for j:= 0 to 53 do
        if j in VTetSites[UpDown1.Position] then GUIForm.ChangeSiteColor(j, clLime) else
          GUIForm.ChangeSiteColor(j);
      GUIForm.Refresh;
    end else
    if m_TemplateIndex > -1 then //plot individual templates...
    begin
      BlankTabCanvas;
      with SpikeTemplates[TabTemplateOrder[m_TemplateIndex]] do
      begin
        if m_PlotRaw and ViewRawPlotsEnabled then //plot raw samples that comprise this template
        begin //plot raw waveforms in blocks of 20
          UpDown1.Max:= n;
          if UpDown1.Position < n then m_RawIdxStart:= UpDown1.Position;
          if (UpDown1.Position + 9) < n then m_RawIdxEnd:= UpDown1.Position + 9
            else m_RawIdxEnd:= n - 1;
          for i:= m_RawIdxStart to m_RawIdxEnd do
            PlotTemplateEpoch(SpikeSet[Members[i]], 0);
          TemplWin.tbRawAvg.Caption:= inttostr(m_RawIdxStart + 1) + '..' + inttostr(m_RawIdxEnd + 1);
        end{rawplot};
      end{spiketemplates};
      PlotTemplate(TabTemplateOrder[m_TemplateIndex], clWhite, True); //now plot mean template
      TemplWin.TabImage.Canvas.TextOut(5, 5, 'id '+ inttostr(TabTemplateOrder[m_TemplateIndex]));
    end;
  end;
end;

{---------------------------------------------------------------------------}
procedure CTemplateWin.ReloadSpikeSet;
var i, j, k, buffidx, buffoffset : integer;
  OrdinalSpikeTimes : array of int64;
  OriginalMembers, OrdinalMembers : TIntArray;
  GroupedSpikeTimes : array of int64;
begin
  tbExtractRaw.Enabled:= False;
  tbRawAvg.Enabled:= True;
  UpDown1.Enabled:= True;
  Setlength(SurfBawdForm.SpikeSet, NTotalSpikes, //allocate array for storing spike sample set
            SpikeTemplates[0].PtsPerChan * SpikeTemplates[0].NumSites);
  Setlength(SurfBawdForm.SpikeTimes, NTotalSpikes);

  { first, order spike times & member indexes sequentially for retrieval }
  Setlength(OrdinalMembers, NTotalSpikes);
  Setlength(OrdinalSpikeTimes, NTotalSpikes);
  Setlength(GroupedSpikeTimes, NTotalSpikes);
  Setlength(OriginalMembers, NTotalSpikes);
  k:= 0;
  for i:= 0 to NumTemplates - 1 do
    with SpikeTemplates[i] do
      for j:= 0 to n - 1 do
      begin
        OrdinalMembers[k]:= k;
        OriginalMembers[k]:= Members[j];
        Members[j]:= k;
        OrdinalSpikeTimes[k]:= SpikeTimes[j];
        inc(k);
      end{j};
  for i:= 0 to NTotalSpikes - 1 do //order spiketime array by template group/order
    GroupedSpikeTimes[i]:= OrdinalSpikeTimes[OriginalMembers[i]];
  Move(GroupedSpikeTimes[0], SurfBawdForm.SpikeTimes[0], Length(OrdinalSpikeTimes) * 8); //copy to surf
  k:= 0;
  for i:= 0 to NumTemplates - 1 do
    with SpikeTemplates[i] do
      for j:= 0 to n - 1 do
      begin
        SpikeTimes[j]:= GroupedSpikeTimes[k];
        inc(k);
      end{j};

  ShellSort64(GroupedSpikeTimes, ssAscending, OrdinalMembers);
  ShellSort64(OrdinalSpikeTimes, ssAscending);

  { retrieve spike epochs into SpikeSet }
  for i:= 0 to NTotalSpikes - 1 do
  begin
    buffidx:= OrdinalSpikeTimes[i] div 10000;
    if SurfBawdForm.m_CurrentBuffer <> buffidx then
    begin
      SurfBawdForm.m_CurrentBuffer:= buffidx;
      SurfBawdForm.GetFileBuffer;
    end;
    buffoffset:= OrdinalSpikeTimes[i] mod 10000{10200} + 100;
    for k:= 0 to SpikeTemplates[0].NumSites - 1 do  //copy whole-probe interpolated waveforms to sample set array
      Move(SurfBawdForm.m_InterpWaveform[buffoffset + k * 10200], SurfBawdForm.SpikeSet[OrdinalMembers[i], k*100], 200{bytes});
  end{i};
  { transfer template records --> clusters to allow user option to continue building templates }
  with SurfBawdForm do
  begin
    NClust:= NumTemplates;
    NSamples:= NTotalSpikes;
    NClustDims:= 100{samples per channel} * ProbeArray[m_ProbeIndex].numchans {channelsperprobe}; //remove hardcoding
    Setlength(Clusters, NClust);
    for i:= 0 to NClust - 1 do
    with Clusters[i] do
    begin
      n:= SpikeTemplates[i].n;
      lastn:= n;
      lastt:= 0;
      Setlength(Members, n);
      Move(SpikeTemplates[i].Members[0], Members[0], n * SizeOf(Members[0]));
      Setlength(Centroid, NClustDims);
      ComputeCentroid(Clusters[i]);
      ComputeDistortion(Clusters[i]);
      Full:= SpikeTemplates[i].Locked;
    end;
  end{surfbawdform};
  tbBuildTemplates.Enabled:= True;
(*  tbDel.Enabled:= True;
  k:= 0;
    for i:= 0 to NumTemplates - 1 do
      with SpikeTemplates[i] do
        for j:= 0 to n - 1 do
        begin
          SurfBawdForm.ClusterLog.Lines.Append(inttostr(i) + ',' +
          inttostr(members[j]) + ',' + inttostr(SurfBawdForm.clusters[i].members[j]) + ',' + inttostr(SpikeTimes[j])
          + ',' + inttostr(SurfBawdForm.SpikeTimes[k]));
          inc(k);
        end;*)
end;

{---------------------------------------------------------------------------}
procedure CTemplateWin.DeleteTemplate(TemplateIndex : integer);
var i, j, k, temp : integer; //only over-ridden class 'cause SpikeSet owned by Surfbawd...
begin                  //should move SpikeSet to template win to avoid this...
  if TemplateIndex >= NumTemplates then Exit;
  if SurfBawdForm.NClust > 0 then //delete corresponding cluster
    SurfBawdForm.DeleteCluster(TemplateIndex);
  with SpikeTemplates[TemplateIndex] do
  begin
    dec(NTotalSpikes, n);
    { adjust remaining member indexes minus members from deleted template }
    for j:= 0 to NumTemplates - 1 do
      if j = TemplateIndex then Continue else
        for k:= 0 to SpikeTemplates[j].n - 1 do
        begin
          temp:= SpikeTemplates[j].Members[k];
          for i:= 0 to n - 1 do
            if temp > Members[i] then
              dec(SpikeTemplates[j].Members[k]);
        end{k};
    { shrink/concatenate global arrays }
    for i:= TemplateIndex to NumTemplates - 2 do
    begin
      SpikeTemplates[i]:= SpikeTemplates[i+1];
      if GlobalFitResults <> nil then
        GlobalFitResults[i]:= GlobalFitResults[i+1];
    end{i};
    dec(NumTemplates);
    Setlength(SpikeTemplates, NumTemplates);
    if GlobalFitResults <> nil then
      Setlength(GlobalFitResults, NumTemplates)
    else begin
      SurfBawdForm.ShowTemplateMode:= False;
      cbShowFits.Checked:= False;
      cbShowFits.Enabled:= False;
    end;
  end{with};
end;

{---------------------------------------------------------------------------}
procedure CTemplateWin.CombineTemplates(TemplateIdxA, TemplateIdxB : integer);
var i, nactive : integer;
begin
  { combine TemplateA with TemplateB, deleting TemplateA }
  with SpikeTemplates[TemplateIdxB] do
  begin
    { combine member and spiketime indexes }
    Setlength(Members, n + SpikeTemplates[TemplateIdxA].n);
    Move(SpikeTemplates[TemplateIdxA].Members[0], Members[n], SpikeTemplates[TemplateIdxA].n * SizeOf(Members[0]));
    Setlength(SpikeTimes, n + SpikeTemplates[TemplateIdxA].n);
    Move(SpikeTemplates[TemplateIdxA].SpikeTimes[0], SpikeTimes[n], SpikeTemplates[TemplateIdxA].n * SizeOf(SpikeTimes[0]));
    inc(n, SpikeTemplates[TemplateIdxA].n);
    { combine waveform statistics }
    for i:= 0 to High(AvgWaveform) do
    begin
      SumWaveform[i]:= SumWaveform[i] + SpikeTemplates[TemplateIdxA].SumWaveform[i]; //optimize with delphi math unit
      AvgWaveform[i]:= Round(SumWaveform[i] / n);    //routines written in asm???!!!
      SSqWaveform[i]:= SSqWaveform[i] + SpikeTemplates[TemplateIdxA].SumWaveform[i] * SpikeTemplates[TemplateIdxA].SumWaveform[i];
      StdWaveform[i]:= sqrt((SSqWaveform[i] - (SumWaveForm[i] * SumWaveForm[i])/n) / (n - 1));
    end;
    { combine active channels, re-compute template indicies }
    Sites:= Sites + SpikeTemplates[TemplateIdxA].Sites;
    nactive:= 0;
    for i:= 0 to NumSites -1 do
      if i in Sites then inc(nactive);
    FitThreshold:= Round(Sqr(DEFAULT_FIT_THRESHOLD / AD2uV) * nactive * PtsPerChan);
    ComputeTemplateMaxChan(TemplateIdxB);
    Enabled:= True;
    if n > MAX_N_PER_TEMPLATE then Locked:= True
      else Locked:= False;
  end{templateidxB};

  { now update clusters, if required }
  with SurfBawdForm do
  if NClust > 0 then
  begin
    { combine member and spiketime indexes }
    Setlength(Clusters[TemplateIdxB].Members, Clusters[TemplateIdxB].n + SpikeTemplates[TemplateIdxA].n);
    Move(SpikeTemplates[TemplateIdxA].Members[0], Clusters[TemplateIdxB].Members[Clusters[TemplateIdxB].n], Clusters[TemplateIdxA].n * SizeOf(Clusters[TemplateIdxB].Members[0]));
    inc(Clusters[TemplateIdxB].n, SpikeTemplates[TemplateIdxA].n);
    { recompute cluster statistics }
    ComputeCentroid(Clusters[TemplateIdxB]);
    ComputeDistortion(Clusters[TemplateIdxB]);
    Clusters[TemplateIdxB].Full:= SpikeTemplates[TemplateIdxB].Locked;
    //reset lastn/lastt?
    { shrink/concatenate cluster array }
    for i:= TemplateIdxA to NClust - 2 do
      Clusters[i]:= Clusters[i+1];
    dec(NClust);
    Setlength(Clusters, NClust);
  end;

  for i:= TemplateIdxA to NumTemplates - 2 do
    SpikeTemplates[i]:= SpikeTemplates[i+1];
  dec(NumTemplates);
  Setlength(SpikeTemplates, NumTemplates);
  Setlength(GlobalFitResults, NumTemplates);
  DeleteTab(TabSelectA);
end;

{---------------------------------------------------------------------------}
procedure CTemplateWin.SplitTemplate(TemplateIndex : integer);
var i : integer;
begin
  with SurfBawdForm do
  begin
    if (SpikeTemplates[TemplateIndex].n = 1) or
       (NClust = 0){waveforms not loaded}  or
       (MessageDlg('Split this template?', mtConfirmation, [mbYes, mbNo], 0) = mrNo) then Exit;
    IsoClus(TemplateIndex);
    Clusters2Templates;
    for i:= 0 to NumTemplates - 1 do ShrinkTemplate(i);
    tbDel.Enabled:= False;
    SortTemplates(TSortCriteria(cbViewOrder.ItemIndex)); //re-sort per user selection
    ChangeTab(-1); //switch to ALL template tab to reflect changes
  end;
end;


{---------------------------------------------------------------------------}
procedure CTemplateWin.ToggleClusterLock(TemplateIndex : integer);
begin
  with SurfBawdForm do
    if NClust > 0 then
      Clusters[TemplateIndex].Full:= cbLocked.Checked;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.ShowHideTemplates;
var t, i, rawidx : integer;
  Colour : TColor;{bf, BestFit}
begin
  with TemplWin do
  begin
    for t:= 0 to NumTemplates - 1 do
    with GlobalFitResults[t] do
    begin
      case TemplWin.rgChartDisp.ItemIndex of
        1 : Colour:= ColorTable[t mod (high(colortable) + 1)]; // highlight spike in colour
        2 : Colour:= $40404000 //faint outline of spike
        else Exit; // don't plot anything
      end{case};
      if SpikeTemplates[t].Enabled and (n > 0) then
      begin
        for i:= SpikeTemplates[t].ChartIndexArray[m_CurrentBuffer] to
            SpikeTemplates[t].ChartIndexArray[m_CurrentBuffer+1] - 1 do
            begin
              if Residuals[i] < (SpikeTemplates[t].FitThreshold) then //if fit is good enough...
              begin //...then plot it!
                rawidx:= (TimeStamps[i] mod 10000) div 4; //REMOVE HARDCODING...
                ChartWin.PlotSpikeEpoch(SpikeTemplates[t], CR.Waveform, rawidx, Colour);
              end;
            end{i};
      end{enabled};
    end{globalfitresults};
  end{templwin};
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.RipFileWithTemplates;
var t, b, w, sigThreshPos, sigThreshNeg : integer; TemEnabled : boolean;
begin
  TemEnabled:= False;
  with TemplWin do //check if any templates enabled, before proceeding
    for t:= 0 to NumTemplates -1 do
      if SpikeTemplates[t].Enabled then
      begin
        TemEnabled:= True;
        Break;
      end;
  if not TemEnabled then Exit;
  Screen.Cursor:= crHourGlass;
  { set search range }
  m_SelStartCRIdx:= TrackBar.SelStart div 100;
  m_SelEndCRIdx:= TrackBar.SelEnd div 100;
  if m_SelStartCRIdx = m_SelEndCRIdx{no selection} then
  begin //fit whole (of loaded) file...
    m_SelStartCRIdx:= 0;
    m_SelEndCRIdx:= m_NEpochs[m_ProbeIndex] - 1;
  end;
  { progresswin }
  FileProgressWin:= TFileProgressWin.CreateParented(SurfBawdForm.Handle);
  with FileProgressWin do
  begin
    FileProgress.MaxValue:= m_SelEndCRIdx;
    FileProgress.MinValue:= m_SelStartCRIdx;
    FileProgressWin.Caption:= 'Fitting template(s) to file...';
    Show;
    BringToFront;
  end;
  TrackBar.SetFocus; //needed to catch ESC key presses
  { allocate arrays for storing residuals, one global + one for each template }
  SetLength(m_residual, 10000{Length(m_InterpWaveform)}); //remove HARDCODING!
  with TemplWin do
  begin
    GlobalFitResults:= nil; //clear existing fits
    SetLength(GlobalFitResults, NumTemplates);
    sigThreshPos:= Round(MIN_ACCEPTABLE_FIT_AMPLITUDE / AD2uV + 2048);
    sigThreshNeg:= Round(-MIN_ACCEPTABLE_FIT_AMPLITUDE / AD2uV + 2048);
    for t:= 0 to TemplWin.NumTemplates - 1 do
    with SpikeTemplates[t] do
    if Enabled then
    begin                            //for improving fitting...
      Setlength(WavStart, NumSites); //define start of spike template waveform per channel
      Setlength(WavEnd, NumSites);   //define end of spike template waveform per channel
      Setlength(ChartIndexArray, m_NEpochs[m_ProbeIndex] + 1);  //residual buffer offsets for overplotting templates on chartwins
      ChartIndexArray[m_SelStartCRIdx]:= 0;
      { compute start and end of waveform signal within each channel epoch }
      for b:= 0 to NumChans - 1 do
      begin
        if not (b in Sites) then Continue;
        {if FitOnlySignal then}
        WavStart[b]:= 0;
        for w:= 0 to PtsPerChan - 1 do
          if (AvgWaveform[b*PtsPerChan + w] > sigThreshPos) or
             (AvgWaveform[b*PtsPerChan + w] < sigThreshNeg) then
             begin
               WavStart[b]:= w;
               Break;
             end;
        WavEnd[b]:= PtsPerChan - 1;
        for w:= WavEnd[b] downto WavStart[b] do
          if (AvgWaveform[b*PtsPerChan + w] > sigThreshPos) or
             (AvgWaveform[b*PtsPerChan + w] < sigThreshNeg) then
             begin
               WavEnd[b]:= w;
               Break;
             end;
      end{b};
    end{t};
  end;

  { fit all templates to selection or entire file }
  for b:= m_SelStartCRIdx to m_SelEndCRIdx{m_NEpochs[m_ProbeIndex]} do
  begin
    if m_CurrentBuffer <> b then
    begin
      m_CurrentBuffer:= b;
      GetFileBuffer;
    end;
    for t:= 0 to TemplWin.NumTemplates - 1 do
    begin
      with TemplWin, GlobalFitResults[t] do
      if SpikeTemplates[t].Enabled then
      begin
        FitTemplate(t);
        if (n + 10000) > Length(TimeStamps) then
        begin
          Setlength(TimeStamps, n + 10000);
          Setlength(Residuals, n + 10000);
        end;
        { now find and save all 'reasonable' local minima residuals }
        w:= 1; //!!!!!!!!!!!!!!!misses first and last minima in every buffer!!!!!!!!!!!!!!!!!!!!!!!!!!
        while w < 9999 do
        begin
          if (m_residual[w] < SpikeTemplates[t].FitThreshold{absolute FitThreshold}) and //possibly a fit...
             (m_residual[w] < m_residual[w-1]) and (m_residual[w] < m_residual[w+1]) then //...local minima...
             begin //...so save timestamp of possible spike...
               TimeStamps[n]:= int64(m_CurrentBuffer) * 10000 + int64(w);
               Residuals[n]:= m_residual[w];
               inc(n);
               inc(w); //skip next point, cannot be minima
             end;
          inc(w);
        end{w};
        SpikeTemplates[t].ChartIndexArray[b+1]:= n;
      end{with};
    end{t};
    FileProgressWin.FileProgress.Progress:= b;
    Application.ProcessMessages;
    if ESCpressed then
    begin
      Screen.Cursor:= crDefault;
      if MessageDlg('Stop searching for template fits?', mtConfirmation, mbOKCancel, 0) = mrOK then
        Break
      else
      begin
        ESCpressed:= False; //shouldn't be necessary
        Screen.Cursor:= crHourGlass;
      end;
    end{ESCpressed}
  end{b};
  for t:= 0 to TemplWin.NumTemplates - 1 do
    with TemplWin.GlobalFitResults[t] do
    begin
      Setlength(TimeStamps, n); //shrink back
      Setlength(Residuals, n); //shrink back
    end;
  ShowTemplateMode:= True;
  TemplWin.cbShowFits.Enabled:= True;
  TemplWin.ChangeTab(m_TemplateIndex); //refresh open windows to reflect new fits
  ESCpressed:= False; //shouldn't be necessary
  FileProgressWin.Release;
  Screen.Cursor:= crDefault;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.FitTemplate(TemplateIndex : integer);
var i, c, w : integer; dist : integer;{int64;}
begin
  with TemplWin, SpikeTemplates[TemplateIndex] do
  for i:= 0 to 9999 do
  begin
    m_residual[i]:= 0;
    for c:= 0 to NumSites - 1 do
    begin
      if c in Sites then
      begin                //note to self: wavstart/end will introduce (consistent) spiketime offset errors!
        for w:= WavStart[c] to WavEnd[c] do
        begin
          dist:= AvgWaveform[{tindex}c * PtsPerChan + w] - m_InterpWaveform[100 + c*10200 + i + w];
          dist:= dist * dist;                                             //REMOVE HARDCODING!!!
          inc(m_residual[i], dist); //compute least squares
        end{w};
      end;
    end{c};
  end{i};
end;

{---------------------------------------------------------------------------}
procedure CHistogramWin.MoveGUIMarker(MouseX : Single);
var ft : integer;
begin
  with SurfBawdForm do
  begin
    if TemplateWinCreated then
    with TemplWin do
    begin
      if not (ShowTemplateMode and FitHistoWinCreated and (m_TemplateIndex > -1))
         or  (GlobalFitResults[TabTemplateOrder[m_TemplateIndex]].Residuals = nil) then Exit;
      ft:= Round(MouseX * FitHistoWin.Max);
      seFitThresh.Value:= ft;
      //SpikeTemplates[TabTemplateOrder[m_TemplateIndex]].FitThreshold:= ft; //implicitly set by onchange event of seFitTthresh
      PlotGUIMarker(Round(MouseX * FitHistoWin.ClientWidth));
      if not (FitHistoWin.LeftButtonDown) then
        if cbShowFits.Checked then OverplotTemplateFits;
        if ChartWinCreated then
        begin
          if TrackBar.Position div 100 {WaveEpochs per Buffer} <> m_CurrentBuffer then //remove Hardcoding
          begin
            m_CurrentBuffer:= TrackBar.Position div 100;
            GetFileBuffer;
          end;
          ChartWin.RefreshChartPlot;
          ShowHideTemplates;
        end;
    end{with};
  end{with};
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.OverplotTemplateFits;
const n2plot = 20;
var i, j, idx, bcount, buffidx, buffoffset : integer;
    ftHi, ftLow : cardinal;
begin
  with TemplWin do
  begin
    ftHi:= SpikeTemplates[TabTemplateOrder[m_TemplateIndex]].FitThreshold;
    with FitHistoWin do idx:= Round(ftHi / (Max - Min) * NumBins);  //get ft ----> histogram bindex
    bcount:= 0; //use histogram to determine how far to go back to get 20 events just under ft
    while (bcount < n2plot) and (idx > 0) do
    begin
      inc(bcount, FitHistoWin.BinCount[idx]);
      dec(idx);
    end;
    with FitHistoWin do ftLow:= Round(idx / NumBins * Max);
    BlankTabCanvas;
    i:= 0;
    j:= 0;
    with GlobalFitResults[TabTemplateOrder[m_TemplateIndex]] do
    begin
      while (i < high(Residuals)) and (j < n2plot) do
      begin
        if (Residuals[i] > ftLow) and (Residuals[i] < ftHi) then
        begin
          buffidx:= TimeStamps[i] div 10000;
          if m_CurrentBuffer <> buffidx then
          begin
            m_CurrentBuffer:= buffidx;
            GetFileBuffer;
          end;
          buffoffset:= TimeStamps[i] mod 10000{10200?} + 100{m_interp pre-pad}; //REMOVE HARDCODING!
          PlotTemplateEpoch(m_InterpWaveform, buffoffset, 10200); //REMOVE HARDCODING!
          inc(j);
        end;
        inc(i);
      end{while};
    end{with};
    PlotTemplate(TabTemplateOrder[m_TemplateIndex], clWhite, True); //now plot mean template
  end{templwin};
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.tbToglHistWinClick(Sender: TObject);
begin
  AddRemoveFitHistogram;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.AddRemoveFitHistogram;
begin
  if tbToglHistWin.Down = False then
  begin
    if FitHistoWinCreated then FitHistoWin.Close;
    FitHistoWinCreated:= False;
    Exit;
  end else
  try
    if FitHistoWinCreated then HistogramWin.Close; //remove any existing histogram wins...
    FitHistoWin:= CHistogramWin.CreateParented(SurfBawdForm.Handle);//..and create a new one
    with FitHistoWin do
    begin
      Left:= 292;
      Top:= 130;
      Width:= 500;
      Height:= 200;
      Caption:= 'Fit Histogram';
      Button1.Free;
      Show;
      BringToFront;
      FitHistoWinCreated:= True;
    end;
  except
    FitHistoWinCreated:= False;
    Exit;
  end;
  if ShowTemplateMode and (m_TemplateIndex > -1) then
    TemplWin.ChangeTab(m_TemplateIndex); //if fits complete and template tab, update histogram...
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.btResetClick(Sender: TObject);
begin
  m_SpikeCount:= 0;
  lbSpikeCount.Caption:= inttostr(m_SpikeCount);
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.MsgMemoKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if key = VK_SHIFT then //reset, begin new selection start
  begin
    ShiftKeyDown:= True;
//    TrackBar.SelStart:= TrackBar.Position;
//    TrackBar.SelEnd:= TrackBar.Position;
    StatusBar.Panels[2].Text:= 'No range selected';
    UpdateStatusBar;
  end;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.MsgMemoMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var i : integer;
begin
  if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream'  then Exit;
  if SurfFile.GetSurfMsg(MsgMemo.CaretPos.y - 1, SurfMsg) then
  begin
    i:= FindEvent(SurfMsg.TimeStamp, Before, SURF_PT_REC_UFFTYPE, SPIKESTREAM{SURF_SV_REC_UFFTYPE, SURF_DIGITAL});
    if i < 0 then Exit else
    begin
      if ShiftKeyDown then
      begin //define selection for export...
        m_SelStartCRIdx:= m_EventArray[i].Index;
        i:= FindEvent(SurfMsg.TimeStamp, After, SURF_SV_REC_UFFTYPE, SURF_DIGITAL);
        if i < 0 then m_SelStartSValIdx:= -1 //no SVAL exists...
          else m_SelStartSValIdx:= m_EventArray[i].Index; //...first SVAL for this selection
        i:= FindEvent(SurfMsg.TimeStamp, Exact, SURF_DSP_REC_UFFTYPE);
        if i < 0 then m_SelStimHdrIdx:= -1 //no stimulus header exists...
          else m_SelStimHdrIdx:= m_EventArray[i].Index; //...stimulus header for this selection
        TrackBar.SelStart:= m_SelStartCRIdx * 100; //remove hardcoding!!!
        StatusBar.Panels[2].Text:= 'Selection: Epoch# ' + inttostr(TrackBar.SelStart) + '...';
      end else
      begin //jump to message event (nearest spikestream record at or before message)...
        Showmessage(inttostr(m_EventArray[i].Index));
        TrackBar.Position:= m_EventArray[i].Index * 100; //remove hardcoding!!!
      end;
    end;
  end;
  TrackBar.SetFocus;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.MsgMemoMouseUp(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var i : integer; SelLength : int64;
begin
  if not ShiftKeyDown then Exit;
  if SurfFile.GetSurfMsg(MsgMemo.CaretPos.y - 1, SurfMsg) then
  begin
    i:= FindEvent(SurfMsg.TimeStamp, Before{?}, SURF_PT_REC_UFFTYPE, SPIKESTREAM{SURF_SV_REC_UFFTYPE, SURF_DIGITAL});
    if i < 0 then Exit else
    with TrackBar do
    begin
      m_SelEndCRIdx:= m_EventArray[i].Index;
      i:= FindEvent(SurfMsg.TimeStamp, Before, SURF_SV_REC_UFFTYPE, SURF_DIGITAL);
      if i < 0 then m_SelEndSValIdx:= -1 //no SVAL exists...
        else m_SelEndSValIdx:= m_EventArray[i].Index; //last SVAL for this selection...
      SelEnd:= m_SelEndCRIdx * 100; //remove hardcoding!!!
      SelLength:= SelEnd - SelStart;
      if SelLength < 0 then SelLength:= 0;
      StatusBar.Panels[2].Text:= 'Selection: Epoch# ' + inttostr(SelStart) +
                                 '-' + inttostr(SelEnd) + '(' + inttostr(SelLength) +
                                 '); Duration: ' + TimeStamp2Str(SelLength * 1000){remove HARDCODING};
    end;
  end;
  TrackBar.SetFocus;
end;
{---------------------------------------------------------------------------}
function TSurfBawdForm.FindEvent(Time : int64; BeforeExactAfter : TEventTime;
                                 EventType : Char; EventSubtype : Char): integer;
begin // returns the event index # for the specified event time, relation, and type...
  case BeforeExactAfter of
    Before :
    for Result:= m_NEvents - 1 downto 0 do //before or equal to the specified time
      if m_EventArray[Result].Time_Stamp <= Time then
        if EventType = m_EventArray[Result].EventType then
          if (EventSubType = ' '{unspecified}) or
             (EventSubType = m_EventArray[Result].SubType) then Exit;
    Exact :
    for Result:= 0 to m_NEvents - 1 do //matches the specified time
      if m_EventArray[Result].Time_Stamp = Time then
        if EventType = m_EventArray[Result].EventType then
          if (EventSubType = ' '{unspecified}) or
             (EventSubType = m_EventArray[Result].SubType) then Exit;
    After :
    for Result:= 0 to m_NEvents - 1 do //after or equal to the specified time
      if m_EventArray[Result].Time_Stamp >= Time then
        if EventType = m_EventArray[Result].EventType then
          if (EventSubType = ' '{unspecified}) or
             (EventSubType = m_EventArray[Result].SubType) then Exit;
  end{case};
  Result:= -1; //failed
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.Button2Click(Sender: TObject);
type TShrtFile = file of Short;
var dp : short; c, w : integer; TemFile : TShrtFile; InfFile : Textfile; OutFileName : string;
   t : integer;// AD2uV, AD2usec : Single;
begin
  {export template waveform to file}
  for t:= 0 to TemplWin.NumTemplates -1 do
  with TemplWin, SpikeTemplates[t] do
  if Enabled then
  begin
    OutFileName:= 'C:\Desktop\PhD Thesis\Neuron locn ensemble\' + Copy(FileNameOnly, 0, Length(FileNameOnly) - 4) + '_t-' + inttostr(t);
    AssignFile(TemFile, (OutFileName + '.tem'));
    Rewrite(TemFile); //overwrites any existing file of the same name
    for w:= 0 to Length(AvgWaveform) -1 do
    begin
      dp:= Round(AvgWaveform[w]);
      Write(TemFile, dp);
    end;
    CloseFile(TemFile);

    {now write associated .inf file}
    AD2uV:= ((20/ProbeArray[m_ProbeIndex].IntGain) / 4096)
           / ProbeArray[m_ProbeIndex].ExtGain[m_CurrentChan] * V2uV;
    AD2usec:= 1/(ProbeArray[m_ProbeIndex].SampFreqPerChan
              * (m_UpSampleFactor) / V2uV);
    if cbHeader.Checked then
    begin
      AssignFile(InfFile, (OutFileName + '.inf'));
      Rewrite(InfFile); //overwrites any existing file of the same name
      //Writeln(InfFile, '%Information for SURF template file: ''' + FileNameOnly + '.tem''');
      Writeln(InfFile, floattostr(AD2uV) {+ ' %uV per ADC value'}); //conversion factor(V)
      Writeln(InfFile, floattostr(AD2usec){ + ' %sample period usec (interpolated)'});
      Writeln(InfFile, inttostr(m_UpSampleFactor -1){ + ' %number of interpolated points'}); //#interpolated pts
      Writeln(InfFile, inttostr(NumSites) {+ ' %number of channels'}); //#channels
      //Writeln(InfFile, '%channel list follows (ch, x coords, y coords)...');
      for c:= 0 to NumChans - 1 do
        //if c in Sites then
          Writeln(InfFile, inttostr(c) + ', ' + inttostr(Electrode.SiteLoc[c].x) + ', '
                                              + inttostr(Electrode.SiteLoc[c].y));
      CloseFile(InfFile);
    end;
    Screen.Cursor:= crDefault;
    Showmessage('Template (+/- .inf) exported to file.');
  end{with};

end;

//--------------------------------------------------------------------------//
procedure CTemplateWin.BuildTemplates;
begin
  with SurfBawdForm do
  begin
    { get search range }
    m_SelStartCRIdx:= TrackBar.SelStart div 100;
    m_SelEndCRIdx:= TrackBar.SelEnd div 100 + 1;
    if m_SelStartCRIdx = (m_SelEndCRIdx - 1) then //if no selection then...
    begin //...search through entire (loaded) file, from t=0
      m_SelStartCRIdx:= 0;//TrackBar.Position div 100;
      m_SelEndCRIdx:= m_NEpochs[m_ProbeIndex];
    end;
    if NumTemplates = 0 then
    begin //build from scratch...
      InitialiseClusters;
      Setlength(SourceBuffers, m_NEpochs[m_ProbeIndex]); //used to keep tabs on which buffers...
      Randomize;                                         //...the templates derived their spikes
      if not BuildSpikeSampleSet then
      begin
        MessageDlg('No new spikes found.', mtInformation, [mbOK], 0);
        Exit;
      end else ClusterLog.Lines.Append(inttostr(NSamples) + ' sample spikes found, grouped into one supercluster.');
      ClusterLog.Lines.Append('Running binary split algorithm...');
      BinarySplitClusters; //consider running binary split (or k-means) initially on pkpk amplitudes only
      ClusterLog.Lines.Append('Finished binary split, NClusters = ' + inttostr(NClust)
                          + '; samples (outliers) deleted = ' + inttostr(OutliersDeleted));
      ClusterLog.Lines.Append('Running k-Means on initial clusters...');
      kMeans(Clusters, NClust);
      ClusterLog.Lines.Append('Converting clusters into templates. Done!');
      if not TemplateWinCreated then CreateTemplateWin;
      Clusters2Templates;
      SortTemplates({scSimilarity}scDecreasingN); //default order by largest class down
      Exit;
    end;
    { otherwise append to existing templates }
    if not BuildSpikeSampleSet then
    begin
      MessageDlg('No new spikes found.', mtInformation, [mbOK], 0);
      Exit;
    end else ClusterLog.Lines.Append(inttostr(NSamples) + ' sample spikes found, grouped into one supercluster.');
    ClusterLog.Lines.Append('Running ISOCLUS(extended k-Means)...');
    IsoClus; //now refine with extended kmeans algorithm
    ClusterLog.Lines.Append('Finished ISOCLUS, final NClusters = ' + inttostr(NClust)
                          + '; samples (outliers) deleted = ' + inttostr(OutliersDeleted));
    ClusterLog.Lines.Append('Converting clusters into templates. Done.');
    if not TemplateWinCreated then CreateTemplateWin;
    //StupidHack2EnableDel; //might 'patch' user template dels, but what about clustering related deletes?
    Clusters2Templates;
    TemplWin.tbDel.Enabled:= False;
    SortTemplates({scSimilarity}scDecreasingN); //default order by largest class down
  end{surfbawdform};
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.StupidHack2EnableDel;
var b, c, j : integer;
  OrdinalSpikeTimes : array of int64;
  OriginalMembers, OrdinalMembers : TIntArray;
  GroupedSpikeTimes : array of int64;
begin
  { group/order spike times & member indexes }
  Setlength(OrdinalMembers, NSamples);
  Setlength(GroupedSpikeTimes, NSamples);
  Setlength(OriginalMembers, NSamples);
  Setlength(OrdinalSpikeTimes, NSamples);
  b:= 0;
  for c:= 0 to NClust - 1 do
    with Clusters[c] do
      for j:= 0 to n - 1 do
      begin
        OrdinalMembers[b]:= c;
        OriginalMembers[b]:= Members[j];
        Members[j]:= b;
        OrdinalSpikeTimes[b]:= SpikeTimes[j];
        inc(b);
      end{j};
  for j:= 0 to NSamples - 1 do //order spiketime array by template group/order
    GroupedSpikeTimes[j]:= OrdinalSpikeTimes[OriginalMembers[j]];
  Move(GroupedSpikeTimes[0], SpikeTimes[0], Length(GroupedSpikeTimes) * 8); //copy to surf
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.Clusters2Templates;
var i, j, k, s, index: integer; AvgWaveform : TWaveform; AllSites : TSites;
begin
  with TemplWin do
  begin
    { BRUTE FORCE APPROACH -- DELETE ALL EXISTING TEMPLATES, CREAT 'EM ALL AGAIN... }
    if NumTemplates > 0 then
    begin
      while TabControl.Tabs.Count > 1 do
        TabControl.Tabs.Delete(1);
      SpikeTemplates:= nil;
      TabTemplateOrder:= nil;
      GlobalFitResults:= nil;
      NumTemplates:= 0;
      NTotalSpikes:= 0;
    end{erasetemplates};
    AllSites:= []; //clear 54..63 (out of range sites for 54 chan polytrodes - remove HARDCODING)
    for i:= 0 to 53 do include(AllSites, i);
    Setlength(AvgWaveform, NClustDims); //should be unnecessary if Add2Template copes with open array pars...
    k:= 0;
    for i:= 0 to NClust - 1 do
    with Clusters[i] do
    begin
      CreateNewTemplate(AllSites, 54, 5400);
      Setlength(SpikeTemplates[i].Members, n);
      Setlength(SpikeTemplates[i].SpikeTimes, n);
      for j:= 0 to n - 1 do
      begin
        index:= 0;
        //copy all channels of each spike into local template
        for s:= 0 to 53 do
        begin //...should be unnecessary if Add2Template copes with open array
          Move(SpikeSet[Members[j], s*100], AvgWaveform[index], 200);
          inc(index, 100);
        end;
        Add2Template(i, AvgWaveform);
        SpikeTemplates[i].Members[j]:= Members[j];
        SpikeTemplates[i].SpikeTimes[j]:= SpikeTimes[k];//SpikeTimes[Members[j]];
        inc(k);
      end{j};
      inc(NTotalSpikes, n);
      ComputeTemplateMaxChan(i);
      SpikeTemplates[i].Locked:= Full;
    end{Clust[i]};
    lbNumSamples.Caption:= inttostr(NTotalSpikes);
  end{templwin};
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.InitialiseClusters;
var i : integer;
begin
  NDesiredClust:= TemplWin.seMaxClust.Value;
  for i:= 0 to High(Clusters) do
  with Clusters[i] do
  begin
    n:= 0;
    Good:= False;
    Members:= nil;
  end;
  NClust:= 0;
  NSamples:= 0;
end;

//--------------------------------------------------------------------------//
function TSurfBawdForm.BuildSpikeSampleSet : boolean;
var b, c, SpikesFound, wavoffset, {s, sample, min, max,} sRange : integer;
begin
  Result:= False;
  Screen.Cursor:= crHourGlass;
  NClustDims:= 100{samples per channel} * ProbeArray[m_ProbeIndex].numchans {channelsperprobe}; //remove hardcoding
  //NClustDims:= ProbeArray[m_ProbeIndex].numchans {channelsperprobe}; //for binary split, just use pp on all chans
  Setlength(SpikeSet, Length(SpikeSet) + TemplWin.seNSamp.Value, NClustDims);
  //Setlength(AmpSet, Length(AmpSet) + SpinEdit2.Value, ProbeArray[m_ProbeIndex].numchans);
  Setlength(SpikeTimes, Length(SpikeTimes) + TemplWin.seNSamp.Value);

  { start/continue searching buffers for spikes, sequentially or randomly }
  sRange:= m_SelEndCRIdx - m_SelStartCRIdx;
  tbPause.Down:= True;
  SpikesFound:= 0;
  b:= m_SelStartCRIdx;

  with TemplWin do
  while (TemplWin.seNSamp.Value > SpikesFound) and (sRange > NBuffersSearched) do
  begin
    //TEMPORARY, FOR SIMULATED FILE FITS:
    SourceBuffers[0]:= Searched; //ignore first buffer for sims...
    SourceBuffers[high(SourceBuffers)]:= Searched; //ignore last buffer for sims...
    { cue next buffer }
    if cbRandomSample.Checked then
    begin
      b:= Random(sRange);
      while SourceBuffers[b + m_SelStartCRIdx] = Searched do //if already searched...
        b:= (b + 1) mod (sRange); //...find the next un-searched buffer...
      inc(b, m_SelStartCRIdx);
    end else
      while SourceBuffers[b] = Searched do inc(b); //find the next un-searched buffer
    if m_CurrentBuffer <> b then
    begin
      m_CurrentBuffer:= b;
      GetFileBuffer;
      m_TrigIndex:= TransBufferSamples * m_UpsampleFactor; //skip over leading transbuffer samples
      { update displays }
      if ChartWinCreated then
        if BinFileLoaded then
          ChartWin.PlotChart(@CR.Waveform[0], 10000) //update chartwin
        else ChartWin.PlotChart(@CR.Waveform[0], 2500); //update chartwin
      TrackBar.Position:= m_CurrentBuffer * 100; //update trackbar position/statusbar time
      if DisplayStatusBar then UpdateStatusBar;
      Application.ProcessMessages;
    end;
    while (TemplWin.seNSamp.Value > SpikesFound) and FindNextThresholdX(m_InterpWaveform) do
    begin //found spike event, copy to SpikeSet...
      wavoffset:= (m_TrigIndex - 10{400µsec} * m_UpSampleFactor) mod 10200; // REMOVE HARDCODING!!!
      SpikeTimes[NSamples + SpikesFound]:= int64(m_CurrentBuffer) * 10000 + int64(wavoffset) - 100;
      for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
        Move(m_InterpWaveform[wavoffset + c * 10200], SpikeSet[NSamples + SpikesFound, c*100], 200{bytes}); //copy whole-probe interpolated waveforms to sample set array
      { update probewin/chartwin displays }
      wavoffset{for plotting}:= ((m_TrigIndex - 100) mod 10200) div 4;
      if (((wavoffset mod 2500) - ProbeArray[m_ProbeIndex].Trigpt) > 0) and //don't bother with display of...
         (((wavoffset mod 2500) + 25 - ProbeArray[m_ProbeIndex].Trigpt) < 2500) then //...transbuffer spikes
         ProbeWin[m_ProbeIndex].win.PlotWaveForm(@CR.Waveform[wavoffset mod 2500 {+ m_PosLockedSamp[m_TrigIndex div 2500]}
                - ProbeArray[m_ProbeIndex].Trigpt], 2500{SampPerChanPerBuff},{ 1,} 0{white});
      if ChartWinCreated then ChartWin.PlotSpikeMarker(wavoffset mod 2500, m_SpikeMaxMinChan);
      inc(SpikesFound);
      lbNumSamples.Caption:= inttostr(NSamples + SpikesFound);
      if DisplayStatusBar then UpdateStatusBar;
      Application.ProcessMessages;
      m_TrigIndex:= (m_TrigIndex + 10200) mod (550800 -1); //mod minus 1 advances to next sample! REMOVE HARDCODING!
    end;
    inc(NBuffersSearched);
    lbSampleTime.Caption:= FormatFloat('0.0', NBuffersSearched / 10{buffspersecond});
    SourceBuffers[b]:= Searched;
  end{while};

  { shrink arrays to match # spikes found }
  inc(NSamples, SpikesFound);
  Setlength(SpikeSet, NSamples{, NClustDims});
  Setlength(SpikeTimes, NSamples);
  //Setlength(AmpSet, NSamples);
  { quantify peak-peak amplitude on all channels for binary split }
  (*for i:= 0 to NSamples - 1 do
  begin
    for c:= 0 to ProbeArray[m_ProbeIndex].numchans - 1 do
    begin
      min:= SpikeSet[i, c*100];
      max:= SpikeSet[i, c*100];
      for s:= 1 to 99 do
      begin
        sample:= SpikeSet[i, c*100 + s];
        if sample > max then max:= sample else
          if sample < min then min:= sample;
      end{s};
      AmpSet[i, c]:= max - min;
    end{c};
  end{i};*)
  tbPause.Down:= False;
  Screen.Cursor:= crDefault;
  if SpikesFound = 0 then Exit;

  { make a new cluster for the new samples }
  inc(NClust);
  Setlength(Clusters, NClust);
  with Clusters[high(Clusters)] do
  begin
    Setlength(Members, SpikesFound);
    for b:= 0 to High(Members) do
    begin
      Members[b]:= (NSamples - SpikesFound) + b;
      n:= SpikesFound;
    end;
    Setlength(Centroid, {54}NClustDims);
    ComputeCentroid(Clusters[High(Clusters)]); //needed?
    ComputeDistortion(Clusters[High(Clusters)]); //needed?
    Full:= False;
    //Good:= True;
  end;
  Result:= True; //found new spikes
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.BinarySplitClusters;                           
var i, j, k, cluster2split, pc1 : integer;
  maxdistortion, maxvariance, mindist, lastmindist, sqrs, sumsqrs : single;
  //kMinus1Clusters : TClusterArray;
  Outlier : boolean;
begin

  // WARNING -- KMINUS1CLUSTER BULLSHIT COPY METHODS DON'T WORK PROPERLY,
  // PRESUMABLY ALSO NOT FUNCTIONAL IN ISOCLUS!!!!!!!!!!!!!!!
  // ALSO CANNOT USE K-1CLUSTERS UNTIL DEALT WITH THE ISSUE OF SAVING THE k-1 SPIKETIMES AND SPIKESET!!!

  ComputeCentroid(Clusters[0]);
  ComputeDistortion(Clusters[0]);

  MinDist:= 1;
  OutliersDeleted:= 0;
  repeat
    //Setlength(kMinus1Clusters, Length(Clusters)); //necessary?
    //kMinus1Clusters:= Copy(Clusters);//Copy(Clusters, 0, NClust); //save current classes
    Outlier:= False;
    cluster2split:= 0;
    maxdistortion:= -1;
    ClustError:= 0;
    for i:= 0 to NClust - 1 do //find existing cluster with largest distortion...
    with Clusters[i] do
    begin
      inc(ClustError, Round(Distortion * Distortion * n / 10000));
      if (Distortion > maxdistortion) and (n > 1) then
      begin
        maxdistortion:= distortion;
        cluster2split:= i;
      end;
    end;
    ClustError:= ClustError div NSamples;// NClust;
    inc(NClust);
    SetLength(Clusters, NClust); //increase size to accomodate new cluster
    with Clusters[cluster2split] do
    begin
      pc1:= 0;
      maxvariance:= 0;
      for i:= 0 to {53}NClustDims - 1 do //...and find dimension of maximal variance...
      begin
        sumsqrs:= 0;                 //inefficient, as aleady calculated this to find distortion!
        for j:= 0 to n - 1 do
        begin
          sqrs:= sqr(Centroid[i] - SpikeSet[Members[j], i]);
          //sqrs:= (Centroid[i] - AmpSet[Members[j], i]) * (Centroid[i] - AmpSet[Members[j], i]);
          sumsqrs:= sumsqrs + sqrs;
        end;
        if sumsqrs > maxvariance then
        begin
          maxvariance:= sumsqrs {/(n-1)}; //no need to divide, n is constant across all dimensions
          pc1:= i;
        end;
      end;
      j:= 0;      //...and split along this dimension to form two subclusters
      while j < n do
      begin
        if (SpikeSet[Members[j], pc1] - Centroid[pc1]) < 0 then
        //if (AmpSet[Members[j], pc1] - Centroid[pc1]) < 0 then
        begin //keep sample in existing cluster...
          inc(j);
          Continue;
        end;
        with Clusters[High(Clusters)] do
        begin //move sample to new cluster...
          inc(n);
          Setlength(Members, n); //overallocate outside then shrink?
          Members[n-1]:= Clusters[cluster2split].Members[j];
        end;
        //...and remove it from the existing cluster
        for k:= j to High(Members) - 1 do//:= Copy(Members, j+1, Length(Members)); //inefficient?
          Members[k]:= Members[k+1];
        dec(n);
        Setlength(Members, n); //shrink later?
        if n = 1 then
        begin
          //ClusterLog.Lines.Append('n=1 binary split deletion... '+ inttostr(cluster2split));
          //DeleteCluster(Cluster2Split); //delete outliers
          //Outlier:= True;
          inc(OutliersDeleted);
          Break;
        end;
      end{j};
    end{with};  //kMeans(Clusters[cluster2split], 2); //...and split it into two subclusters using k-means
    if not Outlier then
    begin
      ComputeCentroid(Clusters[cluster2split]);  //recompute centroids...
      ComputeDistortion(Clusters[cluster2split]);//..and distortion
    end;
    if Clusters[High(Clusters)].n > 0{1} then
    begin
      Setlength(Clusters[High(Clusters)].Centroid, {54}NClustDims);
      ComputeCentroid(Clusters[High(Clusters)]); //compute centroid of new cluster...
      ComputeDistortion(Clusters[High(Clusters)]); //... and distortion
    end else
    begin
      ClusterLog.Lines.Append('n=1 binary split deletion... '+ inttostr(cluster2split));
      DeleteCluster(high(Clusters)); //also delete new cluster if an 'outlier'
      inc(OutliersDeleted);
    end;
    LastMinDist:= MinDist;
    MinDist:= MinCentroidDist(Clusters);
    ClusterLog.Lines.Append(inttostr(NClust) + '; ' + inttostr(ClustError) + '; ' + inttostr(Round(MinDist)) + '; ' + inttostr(Round(MaxDistortion)));
  until //{(((MinDist/Lastmindist) < 0.8{oversplit-threshold}) and (ClustError < 5000))}
     //or (MinDist < 1500)
     {or }(NClust > NDesiredClust);
  //dec(NClust);
  //Setlength(Clusters, NClust); //shrink array to nclusters
  //Clusters:= Copy(kminus1Clusters);//Copy(kMinus1Clusters, 0, NClust); //restore previous k-1 classes
  //CANNOT USE K-1CLUSTERS UNTIL I DEAL WITH THE ISSUE OF SAVING THE k-1 SPIKETIMES AND SPIKESET!!!

  for i:= 0 to NClust - 1 do
    for j:= 0 to Clusters[i].n - 1 do
      ClusterLog.Lines.Append(inttostr(Clusters[i].Members[j]));// + ' ' + inttostr(kminus1Clusters[i].Members[j]));
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.ComputeCentroid; //for some weird reason, specifying the parameters causes a compile-time error
var i, j, sum : integer;
begin
  with Cluster do  //replace with faster asm methods in Delphi Math unit?
  begin
    if n = 0 then Exit;
    for i:= 0 to {53}NClustDims - 1 do //for all dimensions...
    begin
      sum:= SpikeSet[Members[0], i];
      //sum:= AmpSet[Members[0], i];
      for j:= 1 to n - 1 do //...sum across all members...
        //inc(sum, AmpSet[Members[j], i]);
        inc(sum, SpikeSet[Members[j], i]);
      Centroid[i]:= sum / n; //...to get average of dimj
    end{i};
  end{with};
end;

//--------------------------------------------------------------------------//
function TSurfBawdForm.MinCentroidDist;
{ compute cluster pair with minimum centroid distance }
{ only consider 'active' cluster pairs that are unlocked }
var i, j, k : integer; sqrs, sumsqrs, dist : single;
begin
  Result:= High(integer);
  for i:= 0 to NClust - 1 do
  if Clusters[i].Full then Continue else
  begin
    for j:= (i + 1) to NClust - 1 do
    if Clusters[j].Full then Continue else
    begin
      sumsqrs:= 0;
      for k:= 0 to NClustDims - 1 do
      begin
        dist:= Clusters[i].Centroid[k] - Clusters[j].Centroid[k];
        sqrs:= dist * dist;
        sumsqrs:= sumsqrs + sqrs;
      end{k};
      dist:= sqrt(sumsqrs);
      if dist < Result then Result:= dist;
    end{j};
  end{i};
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.ComputeDistortion;
var i, j : integer; sqrs, sumsqrs, dist : single;
begin
  with Cluster do  //replace with faster asm methods in Delphi Math unit?
  begin
    distortion:= 0;
    if n < 2 then Exit;
    for i:= 0 to n - 1 do //compute across all members...
    begin
      sumsqrs:= 0;
      for j:= 0 to {53}NClustDims - 1 do //...for all dimensions...
      begin
{-or+?} sqrs:= sqr(Centroid[j] - SpikeSet[Members[i], j]);
{-or+?} //sqrs:= {sqr}(Centroid[j] - AmpSet[Members[i], j]) * (Centroid[j] - AmpSet[Members[i], j]);
        sumsqrs:= sumsqrs + sqrs;
      end{j};
      dist:= sqrt(sumsqrs);
      distortion:= distortion + dist;
    end{i};
    Distortion:= Distortion / n{sqrt(n) or (n-1)???}; //avg distance of samples to centroid
  end{with};
end;

//--------------------------------------------------------------------------//
{procedure TSurfBawdForm.SplitCluster;
begin
  kMeans(Clusters, 2); //use k-means to split cluster into two
end; }

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.IsoClus;
var //kMinus1Clusters : TClusterArray;
  dist, mindist, sqrs, sumsqrs, maxvariance, maxdistortion, lastmindist : single;
  i, j, k, pc1, cluster2split, NActiveClusts : integer;
  NoMoreClusters, SourceClusterDeleted, Improving : boolean;
begin
  Improving:= True;
  NoMoreClusters:= False;
  LastMinDist:= 1;
  while (not NoMoreClusters) and Improving do
  begin
    if SplitClusterIndex = -1 then
    begin //find cluster with max distortion for splitting...
      kMeans(Clusters, NClust); //assign all samples to nearest k-representative classes
      //kMinus1Clusters:= Copy(Clusters, 0, NClust); //save current classes
      {find cluster with max distortion for splitting}
      cluster2split:= 0;
      maxdistortion:= -1;
      ClustError:= 0;
      NActiveClusts:= 0;
      for i:= 0 to NClust - 1 do
      with Clusters[i] do
      begin
        if Full then Continue;
        ComputeDistortion(Clusters[i]);
        inc(ClustError, Round(Distortion));
        inc(NActiveClusts);
        if (Distortion > maxdistortion) and (n > 1) then
        begin
          maxdistortion:= distortion;
          cluster2split:= i;
        end;
      end{i};
      ClustError:= ClustError div NActiveClusts; //tally average cluster error for active classes
    end else
    begin //specified cluster 2 split...
      cluster2split:= SplitClusterIndex;
      maxdistortion:= 0;
    end;
    if maxdistortion = -1 then NoMoreClusters:= True else{no more clusters with n > 1, so exit}
    begin  //split cluster with max distortion...
      inc(NClust);
      SetLength(Clusters, NClust);
      Clusters[high(Clusters)].n:= 0; //necessary to initialise?
      with Clusters[cluster2split] do
      begin
        pc1:= 0;
        maxvariance:= 0;
        for i:= 0 to {53}NClustDims - 1 do //for existing cluster find dimension of maximal variance...
        begin
          sumsqrs:= 0;                 //inefficient, as aleady calculated this to find distortion!
          for j:= 0 to n - 1 do
          begin
            sqrs:= sqr(Centroid[i] - SpikeSet[Members[j], i]);
            //sqrs:= (Centroid[i] - AmpSet[Members[j], i]) * (Centroid[i] - AmpSet[Members[j], i]);
            sumsqrs:= sumsqrs + sqrs;
          end; //StdDev(Clusters[maxDistortion].)
          if sumsqrs > maxvariance then
          begin
            maxvariance:= sumsqrs;
            pc1:= i;
          end;
        end;
        //...and split along this dimension to form two sub-clusters, one new
        j:= 0;
        SourceClusterDeleted:= False;
        while j < n do
        begin
          //if (AmpSet[Members[j], pc1] - Centroid[pc1]) < 0 then
          if (SpikeSet[Members[j], pc1] - Centroid[pc1]) < 0 then
          begin
            inc(j);
            Continue; //keep sample in existing cluster
          end;
          with Clusters[high(Clusters)] do
          begin //move sample to new cluster...
            inc(n);
            Setlength(Members, n); //overallocate outside then shrink?
            Members[n-1]:= Clusters[cluster2split].Members[j];
            //ClusterLog.Lines.Append('ISOCLUS adding to new cluster...');
          end;
          //...and remove it from the existing cluster
          dec(n);
          if n = {1}0 then //don't split anymore
          begin
            DeleteCluster(Cluster2Split);
            //inc(OutliersDeleted);
            SourceClusterDeleted:= True;
            Break;
          end;
          for k:= j to high(Members) - 1 do//:= Copy(Members, j+1, Length(Members)); //inefficient?
            Members[k]:= Members[k+1]; //concatenat remaining members of clusters[cluster2split]
          Setlength(Members, n); //shrink later?
        end{while};
      end{with};
      if not SourceClusterDeleted then ComputeCentroid(Clusters[Cluster2Split]);  //update centroid...

      if Clusters[High(Clusters)].n > 0 {> 1} then
      begin
        Setlength(Clusters[High(Clusters)].Centroid, {54}NClustDims);
        ComputeCentroid(Clusters[High(Clusters)]); //compute centroid of new cluster...
        //ComputeDistortion(Clusters[High(Clusters)]); //... and distortion FOR SURE NOT NEEDED?
      end else
      begin
        DeleteCluster(High(Clusters));
        //inc(OutliersDeleted);
      end;
    end{split};
    { 'lock out' (or release) clusters with more (less) than 100 samples }
    for k:= 0 to NClust - 1 do
      with Clusters[k] do
        if n > MAX_N_PER_TEMPLATE
          then Full:= True
          else Full:= False;
    { compute cluster pair with minimum centroid distance }
    MinDist:= MinCentroidDist(Clusters);
//    ClusterLog.Lines.Append(inttostr(NClust) + '; ' + inttostr(ClustError) + '; ' + inttostr(Round(MinDist)) + '; ' + inttostr(Round(MaxDistortion)));
    Application.ProcessMessages;
    if (((MinDist/Lastmindist) < 0.85{oversplit-threshold}) and (ClustError < 5000))
      or (NClust > TemplWin.seMaxClust.Value{maxclasses})
      or (MinDist < 2000{?})
      or (SplitClusterIndex > -1) then Improving:= False
    else lastmindist:= mindist;
  end{while};
  //Clusters:= Copy(kMinus1Clusters, 0, NClust); //restore previous k-1 classes
  //dec(NClust);
  //CANNOT USE K-1CLUSTERS UNTIL WE DEAL WITH THE ISSUE OF SAVING THE k-1 SPIKETIMES AND SPIKESET!!!

  for j:= 0 to NClust - 1 do // IS THIS NEEDED OR NOT???
  begin
    ComputeCentroid(Clusters[j]);   //recompute final centroids and...
    ComputeDistortion(Clusters[j]); //...distortion of all classes before exiting
  end;

  { give chance for user-split cluster to assign samples existing classes }
  //if SplitClusterIndex > -1 then kMeans(Clusters, NClust);

(*
    k:= 0;
    while k < NClust do //delete clusters with small 'n' that hasn't changed since last call to k-means
    begin
      with Clusters[k] do
        if (not Full) and (n = lastn) and (n < 4) then
        begin
          DeleteCluster(k);
          inc(OutliersDeleted);
        end else
        begin
          lastn:= n;
          inc(k);
        end;
    end{k};     *)

(*  k:= 0;
  while k < NClust do //delete clusters with only 1 member
  begin
    if Clusters[k].n = 1 then
    begin
      DeleteCluster(k);
      inc(OutliersDeleted);
    end else
      inc(k);
  end{k};*)

end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.kMeans; //for some weird reason, specifying the parameters causes a compile-time error
var i, j, s, minclass : integer;
  dist, sqrs, sumsqrs, mindist : single;
  AlreadyMoved : array of boolean;
  TotalSamplesMoved : integer;
begin
  NkIterations:= 0;
  NSamplesMoved:= 0;
  TotalSamplesMoved:= 0;
  with binCluster do //initialize binclass
  begin
    n:= 0;
    Members:= nil;
  end;
  Setlength(AlreadyMoved, NSamples);
  repeat
    inc(NkIterations);
    NSamplesMoved:= 0;// exit criteria
    for i:= 0 to NSamples - 1 do AlreadyMoved[i]:= False; //<-- only check each sample once
    k:= 0;
    while k < NClust do
    begin
      with Clusters[k] do
      begin
        if Full then
        begin
          inc(k);
          Continue;
        end;
        i:= 0;
        while i < n do
        begin
          if AlreadyMoved[Members[i]] then
          begin
            inc(i);
            Continue;
          end;
          mindist:= high(integer);
          minclass:= k;
          for j:= 0 to NClust - 1 do
          begin
            sumsqrs:= 0;
            for s:= 0 to NClustDims - 1 do
            begin
              sqrs:= sqr(Clusters[j].Centroid[s] - SpikeSet[Members[i], s]); //sqrs:= {sqr}(Centroid[j] - AmpSet[Members[i], j]) * (Centroid[j] - AmpSet[Members[i], j]);
              sumsqrs:= sumsqrs + sqrs;
            end{s};
            dist:= sqrt(sumsqrs);
            if dist < mindist then
            begin
              mindist:= dist;
              minclass:= j;
            end;
          end{j};
          if minclass = k then inc(i) {don't move} else
          begin //move it to...
            with Clusters[minclass] do
            if not Full then
            begin
              inc(n);
              Setlength(Members, n);
              Members[n-1]:= Clusters[k].Members[i];
              AlreadyMoved[Members[n-1]]:= True;
            end else
            with binCluster do //throw away if best fit is a 'full' cluster...
            begin //note to self: once most clusters are full, won't this simply result
              inc(n);         // in deletion of ALL NEW SPIKES?  WAIT AND SEE....
              Setlength(Members, n); //over-allocate outside?
              Members[n-1]:= Clusters[k].Members[i];
            end;
            //...either way, remove it from the existing cluster
            for s:= i to High(Members) - 1 do//:= Copy(Members, j+1, Length(Members)); //inefficient?
              Members[s]:= Members[s+1];
            dec(n);
            Setlength(Members, n);
            inc(NSamplesMoved); //does sending an epoch to a binCluster constitute moving samples???
            if n = 0 then Break;
          end{move};
        end{i};
        if n = 0 then
          DeleteCluster(k)
          //inc(OutliersDeleted);
        else
          inc(k);
      end{with};
      inc(TotalSamplesMoved, NSamplesMoved);
    end{k};
    for k:= 0 to NClust - 1 do //add flag 'cluster modified' y/n so only compute centroids of those modified
      ComputeCentroid(Clusters[k]); //recompute centroids
    Application.ProcessMessages;
  until (NSamplesMoved = 0){ and (NSamplesMoved <= LastSamplesMoved)};//extra criteria needed?
  NSamplesMoved:= TotalSamplesMoved;
  if binCluster.n > 0 then
    DeleteCluster(-1{bincluster}); //dispose of bin cluster...
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.DeleteCluster(ClusterIndex : integer);
var i, j, k, temp : integer;
begin
  if ClusterIndex = - 1 then //special case, dispose of k-means binCluster...
  with binCluster do
  begin
    if n > 1 then ShellSort(Members, ssDescending);
    for i:= 0 to n - 1 do
    begin
      for j:= Members[i] to NSamples - 2 do //if Members[i] = (NSamples - 1) then Continue;
      begin
        Move(SpikeSet[j+1, 0], SpikeSet[j, 0], 5400*2{SizeOf(SpikeSet[0])});
        //SpikeTimes[j]:= SpikeTimes[j+1];
      end{j};
      if Members[i] <> high(SpikeTimes) then
        Move(SpikeTimes[Members[i]+1], SpikeTimes[Members[i]], (High(SpikeTimes) - Members[i]) * 8);
      dec(NSamples);
      SpikeSet[NSamples]:= nil; //necessary?
    end{i};
    Setlength(SpikeSet, NSamples{, NClustDims}); //shrink SpikeSet
    Setlength(SpikeTimes, NSamples); //               "       "
    { adjust remaining member indexes minus members from deleted cluster }
    for j:= 0 to NClust - 1 do
      if j = ClusterIndex then Continue else
        for k:= 0 to Clusters[j].n - 1 do
        begin
          temp:= Clusters[j].Members[k];
          for i:= 0 to n - 1 do
            if temp > Members[i] then
              dec(Clusters[j].Members[k]);
        end{k};
    ClusterLog.Lines.Append('binCluster disposed n '+ inttostr(n) + ' t ' + FormatFloat('0.0', TemplWin.NBuffersSearched / 10{buffspersecond}));
    Exit;
  end{binCluster};

  { delete 'ordinary' cluster }
  if ClusterIndex >= NClust then Exit;
  { rank members of deleted cluster in descending order to minimize }
  { the number of SpikeSet/SpikeTime/Member bytes that need to be moved/changed }
  with Clusters[ClusterIndex] do
  begin
//    ClusterLog.Lines.Append('Disposed of Cluster with n='+ inttostr(n));
    if n > 1 then ShellSort(Members, ssDescending);
    for i:= 0 to n - 1 do
    begin
      for j:= Members[i] to NSamples - 2 do //if Members[i] = (NSamples - 1) then Continue;
      begin
        Move(SpikeSet[j+1, 0], SpikeSet[j, 0], 5400*2{SizeOf(SpikeSet[0])});
        //SpikeTimes[j]:= SpikeTimes[j+1];
      end{j};
      if Members[i] <> high(SpikeTimes) then
        Move(SpikeTimes[Members[i]+1], SpikeTimes[Members[i]], (High(SpikeTimes) - Members[i]) * 8);
        //this method of concatenating spiketimes doesn't assume contiguous spktimes, grouped by cluster
      dec(NSamples);
      SpikeSet[NSamples]:= nil; //necessary?
    end{i};
    Setlength(SpikeSet, NSamples{, NClustDims}); //shrink SpikeSet
    Setlength(SpikeTimes, NSamples); //shrink spiketimes
    { adjust remaining member indexes minus members from deleted cluster }
    for j:= 0 to NClust - 1 do
      if j = ClusterIndex then Continue else
        for k:= 0 to Clusters[j].n - 1 do
        begin
          temp:= Clusters[j].Members[k];
          for i:= 0 to n - 1 do
            if temp > Members[i] then
              dec(Clusters[j].Members[k]);
        end{k};
  end{with};
  { finally shrink/concatenate cluster array }
  for i:= ClusterIndex to NClust - 2 do
    Clusters[i]:= Clusters[i+1];
  dec(NClust);
  Setlength(Clusters, NClust);
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.tbExport2FileClick(Sender: TObject);
begin
  if CElectrode.Items[CElectrode.ItemIndex] = 'SVal Stream' then
  begin
    if ExportDataDialog.Execute then
      ExportEventData(ExportDataDialog.FileName)
    else MessageDlg('SVals not exported.', mtInformation, [mbOK], 0)
  end else
  case ProbeArray[m_ProbeIndex].ProbeSubType of
{    SURF_DIGITAL : begin
                   end;}
     SPIKESTREAM : begin
                     if (TemplWin.NumTemplates = 0) {or etc} then Exit;
                     if ExportDataDialog.Execute then
                       SendClusterResultsToFile(ExportDataDialog.FileName)
                     else MessageDlg('Spikes not exported.', mtInformation, [mbOK], 0)
                   end;
  end{case};
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.SendClusterResultsToFile(SaveFilename : string);
var t, i, c, buffidx : integer; ct, tOffset : int64; spktime : string;
  ExportStream : TFileStream; nullsites : TSites; delimiter : char;
  nspikes : integer;
const //VTetSites = {t18 18chan}[12, 26, 41, 11, 30, 42, 10, 23, 43, 9, 31, 44, 8, 22, 45, 7, 32, 46];//{t18}[9, 10, 23, 31];//{t27}[29, 24, 36, 40];//{t23}[30, 26, 41, 42];
      VTetSites = {t3 12chan}[0, 1, 2, 3, 18, 19, 34, 35, 50, 51, 52, 53];
begin
  Screen.Cursor:= crHourGlass;
  { set search range }
  m_SelEndCRIdx:= TrackBar.SelEnd div 100;    // buffer index range
  m_SelStartCRIdx:= TrackBar.SelStart div 100;//   "       "    "
  if m_SelStartCRIdx = m_SelEndCRIdx{no selection} then
  begin //export all (fitted) spike-times to file...
    m_SelStartCRIdx:= 0;
    m_SelEndCRIdx:= m_NEpochs[m_ProbeIndex] - 1;
  end;
  { progresswin }
  FileProgressWin:= TFileProgressWin.CreateParented(SurfBawdForm.Handle);
  SaveFilename:= ExtractFileName(SaveFilename);
  with FileProgressWin do
  begin
    FileProgress.MinValue:= 0;//m_SelStartSValIdx;
    //FileProgress.MaxValue:= TemplWin.NumTemplates - 1;//m_SelEndSValIdx;
    FileProgressWin.Caption:= 'Exporting clustered spiketimes to ' + SaveFileName + '...';
    Show;
    BringToFront;
  end;
  { determine timestamp offset wrt SURF's global 0.1s timer }
  if m_CurrentBuffer <> m_SelStartCRIdx then
  begin
    m_CurrentBuffer:= m_SelStartCRIdx;
    GetFileBuffer;
  end;
  tOffset:= CR.time_stamp - m_CurrentBuffer * 100000; //accommodate multiple acq start/stops within file
  { export spiketimes to file }
  delimiter:= chr(13);
  with TemplWin do
  for t:= 0 to NumTemplates - 1 do
  with GlobalFitResults[t] do
  begin
    nspikes:=0;
    i:= 0;
    while i < length(Residuals) do
    begin
      if (Residuals[i] < (SpikeTemplates[t].FitThreshold)) then //if spike fit is good enough...
        inc(nspikes); //...count it!
      inc(i);
    end;//while
    ClusterLog.Lines.Add(inttostr(t)+ ' ' + inttostr(nspikes));
  end;//t
  Exit;
  with TemplWin do
  for t:= 0 to NumTemplates - 1 do
  begin
    nullSites:= []; //empty set
    if SpikeTemplates[t].Enabled and ((VTetSites * SpikeTemplates[t].Sites) <> nullSites) {and (n > 0)}
       and (GlobalFitResults[t].Residuals <> nil) then
    {try}begin
      ExportStream:= TFileStream{64}.Create(SaveFileName + '_t' + inttostr(t) + {'.bin'}'.spk', fmCreate);
      ExportStream.Seek{64}(0, soFromBeginning); //overwrite any existing file
      with GlobalFitResults[t] do
      begin
        FileProgressWin.FileProgress.Progress:= 0;
        FileProgressWin.FileProgress.MaxValue:= high(Residuals);
        { write this template's waveform/n/indicies etc. }
        i:= 0;
        buffidx:= TimeStamps[i] div 10000;

        FileProgressWin.FileProgress.MaxValue:= high(Residuals);//m_SelEndSValIdx;
        while i < length(Residuals) do
        begin
          if (Residuals[i] < (SpikeTemplates[t].FitThreshold)) and //if spike fit is good enough...
             (buffidx >= m_SelStartCRIdx) and //...& within export range...
             (buffidx <= m_SelEndCRIdx) then
             begin //...then export it!
               if m_CurrentBuffer <> buffidx then
               begin
                 //Showmessage('Get next buffer...');
                 m_CurrentBuffer:= buffidx;
                 GetFileBuffer;
               end;
               for c:= 0 to SpikeTemplates[t].NumSites - 1 do
                 if c in VTetSites then
                 begin
                   ct:= TimeStamps[i] mod 10000 + c * 10200 + 100;
                   ExportStream.WriteBuffer(m_InterpWaveform[ct], 200);
                 end;
             end{export};
          inc(i);
          buffidx:= TimeStamps[i] div 10000;
        end{i};
        { export spiketimes for this file }
        (*i:= 0;
        while i < length(Residuals) do
        begin
          buffidx:= TimeStamps[i] div 10000;
          if (Residuals[i] < (SpikeTemplates[t].FitThreshold)) and //if spike fit is good enough...
             (buffidx >= m_SelStartCRIdx) and //...& within export range...
             (buffidx <= m_SelEndCRIdx) then
             begin //...then export it!
               ct:= TimeStamps[i] {* 10 restore when done with bin files!} + tOffset;
               //ExportStream.WriteBuffer(ct, SizeOf(ct));
               spktime:= inttostr(ct);
               ExportStream.WriteBuffer(spktime[1], Length(spktime));
               ExportStream.WriteBuffer(delimiter, SizeOf(delimiter)); //delimiter
               FileProgressWin.FileProgress.Progress:= i;
               FileProgressWin.Refresh;
             end;
          inc(i);
        end{i};*)
      end{with};
    ///except
    //  MessageDlg('Error exporting spike templates/times.', mtError, [mbOK], 0);
    //  Break;
    //  ExportStream.Free;
    end;
  end{t};
  FileProgressWin.Release;
  Screen.Cursor:= crDefault;
  ExportStream.Free;
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if (Key=VK_ESCAPE) then ESCPressed:= True;
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.FormKeyUp(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if (Key=VK_ESCAPE) then ESCPressed:= False;
end;

//--------------------------------------------------------------------------//
// temporary procedure follow, for debugging purposes only...........
//--------------------------------------------------------------------------//
procedure TSurfBawdForm.Button3Click(Sender: TObject);
var i, j, k : integer;
begin
  k:= 0;
  with TemplWin do
    for i:= 0 to NumTemplates - 1 do
      with SpikeTemplates[i] do
        for j:= 0 to n - 1 do
        begin
          ClusterLog.Lines.Append(inttostr(i) + ',' +
          inttostr(members[j]) + ',' + inttostr(clusters[i].members[j]) + ',' + inttostr(SpikeTimes[j])
          + ',' + inttostr(SurfBawdForm.SpikeTimes[k]));
          inc(k);
        end;
end;

//--------------------------------------------------------------------------//
procedure TSurfBawdForm.tbToglISIHistClick(Sender: TObject);
begin
  AddRemoveISIHistogram;
end;

{---------------------------------------------------------------------------}
procedure TSurfBawdForm.AddRemoveISIHistogram;
begin
  if tbToglISIHist.Down = False then
  begin
    if ISIHistoWinCreated then ISIHistoWin.Close;
    ISIHistoWinCreated:= False;
    Exit;
  end else
  try
    if ISIHistoWinCreated then ISIHistoWin.Close; //remove any existing histogram wins...
    ISIHistoWin:= CISIHistoWin.CreateParented(SurfBawdForm.Handle);//..and create a new one
    with ISIHistoWin do
    begin
      Left:= 292;
      Top:= 130;
      Width:= 200;
      Height:= 100;
      Caption:= 'ISI Histogram';
      Show;
      BringToFront;
      ISIHistoWinCreated:= True;
    end;
  except
    ISIHistoWinCreated:= False;
    Exit;
  end;
  if ShowTemplateMode and (m_TemplateIndex > -1) then
    TemplWin.ChangeTab(m_TemplateIndex); //if fits complete and template tab, update histogram...
end;

{---------------------------------------------------------------------------}
procedure CISIHistoWin.UpdateISIHistogram;
var ISICount : array[0..99] of cardinal;
  st : int64; i, j : integer; ft : cardinal;
begin
  Reset;
  for i:= 0 to high(ISICount) do ISICount[i]:= 0; //zero initialize ISICount
  with SurfBawdForm.TemplWin.GlobalFitResults[SurfBawdForm.TemplWin.TabTemplateOrder[SurfBawdForm.m_TemplateIndex]]{.SpikeTemplates[SurfBawdForm.m_TemplateIndex]} do
  begin
    ft:= SurfBawdForm.TemplWin.SpikeTemplates[SurfBawdForm.TemplWin.TabTemplateOrder[SurfBawdForm.m_TemplateIndex]].FitThreshold;
    for i:= 0 to high(Residuals) do
    begin
      if Residuals[i] > ft then Continue;
      st:= TimeStamps[i] + 100*1000{100ms};
      if st > TimeStamps[high(TimeStamps)] then Break; //necessary, with while below?
      j:= i + 1; //index next spike...
      while TimeStamps[j] < st do // < (j mod Length(ISICount))
      begin
        //Showmessage('Bin='+inttostr((TimeStamps[j] - TimeStamps[i]) div 1000));
        //timestamps ok? buffer boundary effects?
        if Residuals[j] < ft then inc(ISICount[(TimeStamps[j] - TimeStamps[i]) div 1000]);
        inc(j);
      end{j};
    end{i};
  end{spiketemplates};
  FillHistogram(ISICount, 100);
  PlotHistogram;
end;

{-------------------------------------------------------------------------------------}
procedure TSurfBawdForm.Button4Click(Sender: TObject);
//count number of neurons per virtual tetrode
var t, s, minN, maxN, count, Ntetrodes, NnonZeroTetrodes : integer;
  avgN, nonZeroAvgN : single;
  NPerTetrode : array [0..53] of integer; //remove HARDCODING!
  nullSites   : TSites;
  NearSites   : TSiteArray;
  OutFile : textfile; OutFileName : string;
begin
  //BuildSiteProximityArray(NearSites, seLockRadius.Value, True); //construct virtual tetrodes
  //for s:= 54 to 63 do NearSites[s]:= [];

  { user defined 'tetrodes' for 54umap1b design }
  for s:= 0 to 63 do NearSites[s]:= [];
  Nearsites[0]:= [0, 1, 35, 18];
  Nearsites[1]:= [18, 52, 51, 35];
  Nearsites[2]:= [1, 2, 19, 35];
  Nearsites[3]:= [35, 19, 53, 51];
  Nearsites[4]:= [2, 3, 34, 19];
  Nearsites[5]:= [19, 34, 50, 53];
  Nearsites[6]:= [3, 4, 20, 34];
  Nearsites[7]:= [34, 20, 49, 50];
  Nearsites[8]:= [4, 5, 33, 20];
  Nearsites[9]:= [20, 33, {48,} 49]; //faulty channel, exclude
  Nearsites[10]:= [5, 6, 21, 33];
  Nearsites[11]:= [33, 21, 47{, 48}];//faulty channel, exclude
  Nearsites[12]:= [6, 7, 32, 21];
  Nearsites[13]:= [21, 32, 46, 47];
  Nearsites[14]:= [7, 8, 22, 32];
  Nearsites[15]:= [32, 22, 45, 46];
  Nearsites[16]:= [8, 9, 31, 22];
  Nearsites[17]:= [22, 31, {44,} 45];//faulty channel, exclude
  Nearsites[18]:= [9, 10, 23, 31];
  Nearsites[19]:= [31, 23, 43{, 44}];//faulty channel, exclude
  Nearsites[20]:= [10, 11, 30, 23];
  Nearsites[21]:= [23, 30, 42, 43];
  Nearsites[22]:= [11, {12,} 26, 30];//faulty channel, exclude
  Nearsites[23]:= [30, 26, 41, 42];
  Nearsites[24]:= [{12,} 13, 29, 26];//faulty channel, exclude
  Nearsites[25]:= [26, 29, 40, 41];
  Nearsites[26]:= [13, 14, 24, 29];
  Nearsites[27]:= [29, 24, 36, 40];
  Nearsites[28]:= [14, 15, 28, 24];
  Nearsites[29]:= [24, 28, 39, 36];
  Nearsites[30]:= [15, 17, 25, 28];
  Nearsites[31]:= [28, 25, {37,} 39];//faulty channel, exclude
  Nearsites[32]:= [17, 16, 27, 25];
  Nearsites[33]:= [25, 27, 38{, 37}];//faulty channel, exclude

  Ntetrodes:= 0; //# virtual tetrodes on this probe
  for t:= 0 to 53 do
  begin //exclude sites with less than 3 adjacent sites (ie. a tetrode)
    count:= 0;
    for s:= 0 to 53 do
      if s in NearSites[t] then inc(count);
    if count <> 4{tetrode} then NearSites[t]:= [] else
      inc(Ntetrodes);
  end;
  if NTetrodes = 0 then Exit;

  count:= 0;
  nullSites:= [];
  for s:= 0 to high(NPerTetrode) do NPerTetrode[s]:= 0;//initialise...
  for t:= 0 to TemplWin.NumTemplates - 1 do
  with TemplWin.SpikeTemplates[t] do
  begin
    if not Enabled then Continue;
    for s:= 0 to {Ntetrodes - 1} high(Nearsites) do
    begin
    if NearSites[s] = [] then Continue;
      if (Sites * NearSites[s]) <> NullSites then
      begin
        inc(NPerTetrode[s]);
        inc(count);
      end;
    end{s};
  end{t};
  minN:= 1000;//NperTetrode[0];
  maxN:= 0;//NperTetrode[0];
  for s:= 0 to high(NperTetrode) do
  begin //get range, excluding tetrodes with n=0
    if (NperTetrode[s] < minN) and
       (NperTetrode[s] > 0) then
        minN:= NperTetrode[s];
    if NperTetrode[s] > maxN then
      maxN:= NperTetrode[s];
  end;
  NnonZeroTetrodes:= 0;
  for t:= 0 to high(NPerTetrode) do
    if NPerTetrode[t] > 0 then inc(NnonZeroTetrodes);
  avgN:= count / NTetrodes;
  nonZeroAvgN:= count / NnonZeroTetrodes;

  Showmessage('Virtual tetrodes without faulty channels and'+ chr(13)+ 'at least 1 neuron = '+ inttostr(NnonZeroTetrodes));
  Showmessage(inttostr(minN) + ' - ' + inttostr(maxN) + ' neurons per virtual tetrode, average '
            + floattostrf(avgN, ffGeneral, 3, 4) + ' (non-zero average = '+ floattostrF(nonZeroAvgN, ffGeneral, 3, 4)+ ').');

  {export textfile with results}
  OutFileName := TemplWin.Caption + '.csv';
  AssignFile(OutFile, OutFileName);
  Rewrite(OutFile); //overwrites any existing file of the same name
  Writeln(OutFile, 'tetrode, #neurons');
  for t:= 0 to high(NperTetrode){tetrodes - 1} do
    Writeln(OutFile, inttostr(t) + ',' + inttostr(NperTetrode[t]));
  CloseFile(OutFile);
end;

{-------------------------------------------------------------------------------------}
procedure TSurfBawdForm.Button5Click(Sender: TObject);
//show approximate neuron positions on polytrode GUI
var c, j, t, p, min, max, totAmp : integer;
  chWeights : array [0..53] of single;
  chPkPkAmp : array [0..53] of integer;
  posn2D    : TPoint;
begin
  //Setlength(posn2D, TemplWin.NumTemplates);
  if not GUICreated then
  begin
    tbToglProbeGUI.Down:= True;
    AddRemoveProbeGUI;
  end;
  for t:= 0 to TemplWin.NumTemplates - 1 do
  with TemplWin.SpikeTemplates[t] do
  if Enabled then
  begin
    j:= 0;
    totAmp:= 0;
    for c:= 0 to high(chWeights) do
    begin
      { calculate peak-peak amplitudes for all channels }
      min:= AvgWaveform[j];
      max:= AvgWaveform[j];
      inc(j);
      for p:= 1 to PtsPerChan - 1 do
      begin
        if AvgWaveform[j] < min then
          min:= AvgWaveform[j] else
        if AvgWaveform[j] > max then
          max:= AvgWaveform[j];
        inc(j);
      end;
      chPkPkAmp[c]:= max - min;
      if c in Sites then inc(totAmp, chPkPkAmp[c]); //for calc. ch weights
    end{c};
    { estimate 2d posn from amplitude-weighted siteloc's }
    posn2D.x:= 0;
    posn2D.y:= 0;
    with ProbeWin[m_ProbeIndex].electrode do
    begin
      for c:= 0 to high(chWeights) do
      begin
        if c in Sites then chWeights[c]:= chPkPkAmp[c] / totAmp
          else chWeights[c]:= 0;
        posn2D.x:= Round(posn2D.x + (chWeights[c] * SiteLoc[c].x));
        posn2D.y:= Round(posn2D.y + (chWeights[c] * SiteLoc[c].y));
      end;
      GUIForm.PlotNeuronPosn(Windows.TPoint(posn2D));// plot neuron icon on polytrode GUI
    end{electrode};
  end{enabled spktem[t]};
  GUIForm.Refresh;
end;

{-------------------------------------------------------------------------------------}
procedure TSurfBawdForm.Button9Click(Sender: TObject);
//show approximate field size for each neuron on polytrode GUI
const VTetSites = {[26, 29, 40, 41];} {[8, 9, 31, 22]} [19, 35, 51, 53];
var c, j, t, p, min, max, totAmp, w, h : integer;
  chWeights : array [0..53] of single;
  chPkPkAmp : array [0..53] of integer;
  posn2D    : TPoint;
  fieldBox  : TRect; boxCol : TColor; boxpm : TPenMode; penw : integer;
begin
  //Setlength(posn2D, TemplWin.NumTemplates);
  if not GUICreated then
  begin
    tbToglProbeGUI.Down:= True;
    AddRemoveProbeGUI;
  end;
  for t:= 0 to TemplWin.NumTemplates - 1 do
  with TemplWin.SpikeTemplates[t] do
  if Enabled then
  begin
    { calculate size of field box }
    with ProbeWin[m_ProbeIndex].electrode, fieldBox do
    begin
      Left:= 10000;
      Right:= -10000;
      Top:= 10000;
      Bottom:= -10000;
      for c:= 0 to NumSites - 1 do
      if c in Sites then
      begin
        if SiteLoc[c].x < Left then
          Left:= SiteLoc[c].x;// else
        if SiteLoc[c].x > Right then
          Right:= SiteLoc[c].x;
        if SiteLoc[c].y < Top then
          Top:= SiteLoc[c].y;
        if SiteLoc[c].y > Bottom then
          Bottom:= SiteLoc[c].y;// else
      end{c};
      w:= Right - Left;
      h:= Bottom - Top;
      if w = 0 then
      begin
        Left:= -30;
        Right:= 30;
      end else
      begin
        Left:= -w div 2 - 30;
        Right:= w div 2 + 30;
      end;
      if h = 0 then
      begin
        Top:= -30;
        Bottom:= 30;
      end else
      begin
        Top:= -h div 2 - 30;
        Bottom:= h div 2 + 30;
      end;
    end;
    //Showmessage(inttostr(fieldBox.Left)+' '+inttostr(fieldBox.Right)+' '+inttostr(fieldBox.Top)+' '+inttostr(fieldBox.Bottom));
    j:= 0;
    totAmp:= 0;
    for c:= 0 to high(chWeights) do
    begin
      { calculate peak-peak amplitudes for all channels }
      min:= AvgWaveform[j];
      max:= AvgWaveform[j];
      inc(j);
      for p:= 1 to PtsPerChan - 1 do
      begin
        if AvgWaveform[j] < min then
          min:= AvgWaveform[j] else
        if AvgWaveform[j] > max then
          max:= AvgWaveform[j];
        inc(j);
      end;
      chPkPkAmp[c]:= max - min;
      if c in Sites then inc(totAmp, chPkPkAmp[c]); //for calc. ch weights
    end{c};
    { estimate 2d posn from amplitude-weighted siteloc's }
    posn2D.x:= 0;
    posn2D.y:= 0;
    with ProbeWin[m_ProbeIndex].electrode do
    begin
      for c:= 0 to high(chWeights) do
      begin
        if c in Sites then chWeights[c]:= chPkPkAmp[c] / totAmp
          else chWeights[c]:= 0;
        posn2D.x:= Round(posn2D.x + (chWeights[c] * SiteLoc[c].x));
        posn2D.y:= Round(posn2D.y + (chWeights[c] * SiteLoc[c].y));
      end;
      {if (Sites * VTetSites) <> [] then}
      begin
        boxCol:= ColorTable[t mod (High(ColorTable)+1)];
        boxpm := {pmCopy;//}pmMerge;
        penw  := 1;
     { end else
      begin
        boxCol:= clDkGray;
        boxpm := pmCopy;
        penw  := 1;}
      end;
      GUIForm.PlotNeuronField(fieldBox, Windows.TPoint(posn2D), boxCol, boxpm, penw);// plot neuron field on polyGUI
    end{electrode};
  end{enabled spktem[t]};
  GUIForm.Refresh;
end;

{-------------------------------------------------------------------------------------}
procedure TSurfBawdForm.Button6Click(Sender: TObject);
const VTetSites = {t18} [9, 10, 23, 31]; //{t23}[30, 26, 41, 42];//{t27}[29, 24, 36, 40];
var t, c, e : integer;
  nullSites: TSites;
  OutFileName : string;
begin
  nullSites:= [];
  with TemplWin do
  begin
    for t:= 0 to NumTemplates - 1 do
    with SpikeTemplates[t] do
    if Enabled and ((VtetSites * Sites) <> nullSites) then
    begin
      OutFileName:= 'C:\Desktop\Tets\t'+ inttostr(t) + '.bin';
      m_ExportStream:= TFileStream{64}.Create(OutFileName, fmCreate);
      m_ExportStream.Seek{64}(0, soFromBeginning); //overwrite any existing file
      for e:= 0 to n - 1 do
        for c:= 0 to NumChans - 1 do
          if c in VTetSites then
             m_ExportStream.WriteBuffer(SpikeSet[Members[e], c*100], 200);
      m_ExportStream.Free;
    end{t};
  end{templwin};
end;

{-------------------------------------------------------------------------------------}
procedure TSurfBawdForm.Button7Click(Sender: TObject);
var
  MyFormat : Word;
  AData    : THandle;
  APalette : hPalette;  //wrong in D3-D7 online example!
begin
  try
  with ChartWin.WaveformBM do
    SaveToClipBoardFormat(MyFormat, AData, APalette);
    ClipBoard.SetAsHandle(MyFormat, AData);
  except
  end;
end;

(*
var
jp : TJpegImage;
begin
  jp:= TJpegImage.Create;
  try
    with jp do
    begin
      Assign(ChartWin.WaveformBM);
      SaveToFile('c:\desktop\mytraces.jpg')
    end;
  finally
    jp.Free;
  end;
end; *)
{-------------------------------------------------------------------------------------}
procedure TSurfBawdForm.Button8Click(Sender: TObject);
var t, s, c : integer;
  OutFile : textfile; OutFileName : string;
begin
  OutFileName:= 'c:\desktop\' + TemplWin.Caption + '_chansperneuron.csv';
  AssignFile(OutFile, OutFileName);
  Rewrite(OutFile); //overwrites any existing file of the same name
  Writeln(OutFile, 'template, #channels');
  for t:= 0 to TemplWin.NumTemplates - 1 do
  with TemplWin.SpikeTemplates[t] do
  begin
    if not Enabled then Continue;
    c:= 0;
    for s:= 0 to NumSites - 1 do
      if s in Sites then inc(c);
    Writeln(OutFile, inttostr(t) + ',' + inttostr(c));
  end;
  CloseFile(OutFile);
end;

end.

