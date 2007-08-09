unit SurfBawdMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  SurfFileAccess, ElectrodeTypes, SurfPublicTypes, WaveFormPlotUnit, SurfMathLibrary,
  RasterPlotUnit, InfoWinUnit, ExtCtrls, StdCtrls, ComCtrls, ToolWin, ImgList, Math;

const
  ParamNames : array[0..7] of string[20] = ('Channel',  'Amplitude',
                'Peak', 'Valley', 'TimeAtPeak', 'Width', 'Area', 'Polarity');
  V2uV = 1000000;
  DefaultSampFreq = 32000;

type

  CProbeWin = class(TWaveFormPlotForm)
    private
      procedure NotAcqOnMouseMove(ChanNum : byte); override;
    end;

  TProbeWin = record
    exists : boolean;
    win : CProbeWin;
  end;

  TSurfBawdForm = class(TForm)
    FileStatsPanel: TPanel;
    Label1: TLabel;
    Label2: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    DataFileSize: TLabel;
    FileNameLabel: TLabel;
    NProbes: TLabel;
    NCRs: TLabel;
    NSpikes: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    NMsgs: TLabel;
    NSVals: TLabel;
    SurfFileAccess: TSurfFileAccess;
    CElectrode: TComboBox;
    Waveforms: TPanel;
    StatusBar: TStatusBar;
    TrackBar: TTrackBar;

    ToolBar1: TToolBar;
    SmlButtonImages: TImageList;

    ToolBar2: TToolBar;
    LrgButtonImages: TImageList;
    tbRethreshold: TToolButton;
    tbStepBack: TToolButton;
    tbReversePlay: TToolButton;
    tbStop: TToolButton;
    tbPlay: TToolButton;
    tbStepFoward: TToolButton;
    tbPause: TToolButton;
    tbWaveformStats: TToolButton;
    ToolButton1: TToolButton;
    tbLocNSort: TToolButton;
    ToolButton3: TToolButton;
    ToolButton5: TToolButton;
    ToolBar3: TToolBar;
    tbToglWform: TToolButton;
    tbToglStatsWin: TToolButton;
    tbRasterPlot: TToolButton;
    ToolButton9: TToolButton;
    ToolButton10: TToolButton;
    ToolButton11: TToolButton;
    ToolButton12: TToolButton;
    ToolButton13: TToolButton;
    DisplayToggleImages: TImageList;
    tbExportData: TToolButton;

    procedure SurfFileAccessNewFile(acFileName: WideString);
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
    procedure TrackBarKeyPress(Sender: TObject; var Key: Char);
    procedure StatusBarDblClick(Sender: TObject);
    procedure tbRethresholdClick(Sender: TObject);
    procedure tbToglStatsWinClick(Sender: TObject);
    procedure tbToglWformClick(Sender: TObject);
    procedure tbLocNSortClick(Sender: TObject);
    procedure tbRasterPlotClick(Sender: TObject);
    procedure tbExportDataClick(Sender: TObject);

  private
    { Private declarations }
    ChanWithMaxAmp: Byte;
    splinex, spliney, spline2ndderiv : SplineArrayNP;
    WaveformStatsWin : TInfoWin;
    RasterWin : TRasterForm;
    tbShiftKeyDown, DisplayStatusBar: boolean ;
    procedure ComputeWaveformParams(var Spike : TSpike; Precision : boolean = true;
                                                OneChanOnly : shortint = -1);
    procedure WaveFormStatsUpdate;
    procedure PredNSort;
    function ADValue2uV(ADValue : SmallInt): SmallInt;
  public
    { Public declarations }
    m_NProbes, m_SumSpikes, m_SumCRs, m_NSVals, m_NMsgs : integer; //...in file
    m_NSpikes, m_NCRs : array of integer; //...in probe[n]
    m_TStep : array of single;
    m_CurrentProbe : byte;
    m_CurrentChan : byte;

    m_SpikeTimeStamp : array of lng;
    m_SpikeClusterID : array of smallint;

    ProbeWin : array[0..SURF_MAX_PROBES-1] of TProbeWin;  //need to shrink after file loaded

    Spike : TSpike; //current spike record (see SurfPublicTypes)
    ProbeArray : TProbeArray;
    Cr : TCr;
    Sval : TSVal;
    SurfMsg : TSurfMsg;
    FileNameOnly : shortstring;
    //put some of these in Private or even local?
  end;

var
  SurfBawdForm: TSurfBawdForm;

implementation

uses SurfLocateAndSort;

{$R *.DFM}

procedure TSurfBawdForm.SurfFileAccessNewFile(acFileName: WideString);
var
  FileNameOnly : shortstring;
  DFileSize : word;
  p, e  : byte;
  Electrode : TElectrode;

begin
  SurfBawdForm.BringToFront;
  StatusBar.Visible:=true;
  TBShiftKeyDown := false;
  WaveForms.Visible:= true;
  //initialise and display global file information
  FileNameOnly := ExtractFileName(acFileName);
  FileNameLabel.Caption := FileNameOnly;
  FileNameLabel.Hint := acFileName;
  DFileSize:= SurfFileAccess.GetDFileSize div 1000000;
  DataFileSize.Caption := inttostr(DFileSize)+'Mb';

  m_NProbes := SurfFileAccess.GetNumProbes;
  NProbes.Caption := inttostr(m_NProbes);
  m_NSVals := SurfFileAccess.GetNumSVals;
  NSVals.Caption := inttostr(m_NSVals);
  m_NMsgs := SurfFileAccess.GetNumSurfMsgs;
  NMsgs.Caption := inttostr(m_NMsgs);

  //Get the array of probe-layout records, individual probe and file spikes/CRs
  m_SumSpikes :=0;
  m_SumCRs :=0;
  SetLength(ProbeArray, m_NProbes);
  SetLength(m_NSpikes, m_NProbes);
  SetLength(m_NCrs, m_NProbes);
  SetLength(m_TStep, m_NProbes);

  for p := 0 to m_NProbes-1 do
  begin
    SurfFileAccess.GetProbeRecord(p,ProbeArray[p]);
    m_NSpikes[p] := SurfFileAccess.GetNumSpikes(p);
    m_NCRs[p] := SurfFileAccess.GetNumCRs(p);
    inc(m_SumSpikes, m_NSpikes[p]);   //total ALL spikes...
    inc(m_SumCRs, m_NCRs[p]);         //and CRs from ALL probes
    if probearray[p].sampfreqperchan = 0 then          //fix for UFF files without
      probearray[p].sampfreqperchan:= DefaultSampFreq; //sampfreq saved
    m_TStep[p]:= 1/ProbeArray[p].sampfreqperchan*1000000;
    ProbeWin[p].exists := FALSE;
  end;

  NSpikes.Caption := inttostr(m_SumSpikes);
  NCRs.Caption := inttostr(m_SumCRs);

  //Generate electrode list from those in file, CURRENTLY UNABLE TO HANDLE OLD UFF FILES
  CElectrode.Items.Clear;
  For p:= 0 to m_NProbes-1 do
    begin
    For e := 0 to KNOWNELECTRODES-1 {from ElectrodeTypes} do
      if ProbeArray[p].Electrode_name = KnownElectrode[e].Name then
      begin
        CElectrode.Items.Add(KnownElectrode[e].Name);
        Break; // found electrode type, continue with next probe
      end;

    CElectrode.Items.Add('GLOBAL');
    CElectrode.ItemIndex := m_NProbes;//select global file stats from list

    if not GetElectrode(Electrode,ProbeArray[p].electrode_name) then
    begin
      ShowMessage(ProbeArray[p].electrode_name+' is an invalid electrode name');
      exit;
    end;

    //now create the probe window associated with this electrode
    ProbeWin[p].win := CProbeWin.CreateParented(Waveforms.Handle);
    ProbeWin[p].win.show;
    ProbeWin[p].exists := TRUE;
    ProbeWin[p].win.InitPlotWin(Electrode,
                              {npts}ProbeArray[p].pts_per_chan,
                              {left}ProbeArray[p].ProbeWinLayout.Left,
                              {top}ProbeArray[p].ProbeWinLayout.Top+ToolBar1.Height+20,
                              {thresh}ProbeArray[p].Threshold,
                              {trigpt}ProbeArray[p].TrigPt,
                              {probeid}p,
                              {probetype}ProbeArray[p].ProbeSubType,
                              {title}ProbeArray[p].probe_descrip,
                              {acquisitionmode}{TRUE}FALSE);

    if SurfBawdForm.ClientWidth < ProbeWin[p].win.Width then SurfBawdForm.ClientWidth := ProbeWin[p].win.Width;
    if SurfBawdForm.ClientHeight  < StatusBar.height + ToolBar1.Height + ProbeWin[p].win.Height + 10
      then SurfBawdForm.ClientHeight  := StatusBar.height + ToolBar1.Height + WaveForms.Top + ProbeWin[p].win.Height + 20;
    ProbeWin[p].win.Visible := TRUE;
  end;

  m_CurrentProbe:= 0;
  m_CurrentChan:= 0;
  CElectrode.ItemIndex := m_CurrentProbe;
  TrackBar.Min := 0;
  TrackBar.Max := m_NSpikes[m_CurrentProbe]-1;
  TrackBar.SetFocus;
  DisplayStatusBar:= true;
  SurfFileAccess.GetSpike(m_CurrentProbe, 0, Spike);
  SetLength(Spike.param, ProbeArray[m_CurrentProbe].numchans, 7); // ideally, not hard coded
  UpdateStatusBar;
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
    pt1 := ProbeArray[m_CurrentProbe].trigpt;
    pt2 := ProbeArray[m_CurrentProbe].trigpt;
    with Spike do
    begin
      peak := waveform[c, ProbeArray[m_CurrentProbe].trigpt{0}]; //assume peak at or beyond trigpt
      negpeak := waveform[c, ProbeArray[m_CurrentProbe].trigpt]; //assume valley at or beyond trigpt
      for w:={1}ProbeArray[m_CurrentProbe].trigpt+1 to ProbeArray[m_CurrentProbe].trigpt+16 do //assume waveform width <16pts from trigpt
      begin
        area:= area + abs(waveform[c,w]-2048);
        if peak < waveform[c,w] then
        begin
          peak:= waveform[c,w];
          pt1:= w;
        end else
        if negpeak > waveform[c,w] then
        begin
          negpeak:= waveform[c,w];
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
    Spline(splinex,spliney,npts,spline2ndderiv);
    for w := 1 to npts * SURF_SPLINE_RES do
    begin
      rx := w/SURF_SPLINE_RES;
      Splint(splinex,spliney,spline2ndderiv,npts,rx,ry);
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
{!}SetLength(SplintArray, ProbeArray[m_CurrentProbe].numchans, 20 * SURF_SPLINE_RES);
  if OneChanOnly > -1 then
  begin
    c:= OneChanOnly;
    RoughParams;
    if Precision then
    begin
      PreciseParams;
      pt1:= round(precise_t1 * m_TStep[m_CurrentProbe]);
      pt2:= round(precise_t2 * m_TStep[m_CurrentProbe]);
    end else
    begin
      pt1:= round(pt1 * m_TStep[m_CurrentProbe]);
      pt2:= round(pt2 * m_TStep[m_CurrentProbe]);
    end;
    AllocateParams;
  end else
  begin
    ChanWithMaxAmp:= 0;
    MaxAmp:= 0;
    for c:= 0 to ProbeArray[m_CurrentProbe].numchans-1 do //find channel with largest waveform
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
    for c:= 0 to ProbeArray[m_CurrentProbe].numchans-1 do //determine voltages at peak times on other chans
    begin
      //if c:= ChanWithMaxAmp then continue;
      if precision then
      begin
        pt1:= round(ProbeArray[m_CurrentProbe].trigpt + precise_t1);
        pt2:= round(ProbeArray[m_CurrentProbe].trigpt + precise_t2);
        Showmessage('pt1='+inttostr(pt1)+'; precise_t1='+floattostr(precise_t1));
        Peak:= SplintArray[c, pt1];
        NegPeak:= SplintArray[c, pt2];
        //pt1:= round(precise_t1 * m_TStep[m_CurrentProbe]);
        //pt2:= round(precise_t2 * m_TStep[m_CurrentProbe]);
      end else
      begin
        Peak:= Spike.waveform[c, pt1];
        NegPeak:= Spike.waveform[c, pt2];
        //pt1:= round(pt1 * m_TStep[m_CurrentProbe]);
        //pt2:= round(pt2 * m_TStep[m_CurrentProbe]);
      end;
      AllocateParams;
      Showmessage('Chan:'+inttostr(c)+'; Peak: '+inttostr(Spike.Param[c,1]));
    end;
  end;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.CElectrodeChange(Sender: TObject);
begin
  if CElectrode.Text = 'GLOBAL' then
  begin
    NSpikes.Caption := inttostr(m_SumSpikes);
    NCRs.Caption := inttostr(m_SumCRs);
  end else
  begin
    NSpikes.Caption := inttostr(m_NSpikes[CElectrode.ItemIndex]);
    NCRs.Caption := inttostr(m_NCRs[CElectrode.ItemIndex]);
  end;
  m_CurrentProbe:= CElectrode.ItemIndex;
  TrackBar.SetFocus; //catch trackbar key events
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.TrackBarChange(Sender: TObject);
begin
 { if TBShiftKeyDown then
    if TrackBar.SelStart > TrackBar.SelEnd then TrackBar.SelEnd:= TrackBar.Position
      else TrackBar.SelStart:= TrackBar.Position;  //range selection code needs debugging!
  }
  SurfFileAccess.GetSpike(m_CurrentProbe,TrackBar.Position,Spike);
  ProbeWin[m_CurrentProbe].win.PlotSpike(Spike);
  if DisplayStatusBar then UpdateStatusBar;
  if tbRasterPlot.Down then RasterForm.UpdateRasterPlot;
  WaveformStatsUpdate;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.UpdateStatusBar;
begin
  StatusBar.Panels[0].Text := 'Time: ' + inttostr(Spike.time_stamp div 600000) //minutes
    +':'+inttostr(Spike.time_stamp div 10000 mod 60) //seconds
    +'.'+inttostr(Spike.time_stamp mod 10000); //milliseconds
  StatusBar.Panels[1].Text := 'Spike: ' + inttostr(TrackBar.Position+1) + '/'
                                        + inttostr(m_NSpikes[m_CurrentProbe]);
  {ComputeWaveformParams(Spike);
  StatusBar.Panels[2].Text := 'Chan with max peak = '+ inttostr(ChanWithMaxPeak)+' (peak = '
    +inttostr(Spike.Param[chanwithmaxpeak])+')';}
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.TrackBarKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  if key = VK_SHIFT then TBShiftKeyDown:= true;
end;

procedure TSurfBawdForm.TrackBarKeyUp(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  if key = VK_SHIFT then TBShiftKeyDown:= false;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.FormClose(Sender: TObject; var Action: TCloseAction); // free memory
var p : byte;
begin
  for p := 0 to SurfFileAccess.GetNumProbes-1 do
  if ProbeWin[p].exists then
  begin
     ProbeWin[p].win.free; //does NOT release waveformstatswin - currently causes runtime error
     ProbeWin[p].exists := false;
  end;

  ProbeArray := nil;
  Spike.waveform := nil;
  Cr.waveform := nil;

  {notify Windows that we no longer want drop-file notification messages}
  SurfFileAccess.Close;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbPlayClick(Sender: TObject);
var spk : integer;
begin
  tbPause.Down:= false;
  spk:= TrackBar.Position;
  while (spk <= m_NSpikes[m_CurrentProbe]) and (spk >= 0) and (tbPlay.Down or tbReversePlay.Down) do
  begin
    if tbPause.Down then exit;
    if tbLocNSort.Down then PredNSort;
    if tbPlay.Down then inc(spk)
      else dec(spk);
    TrackBar.Position:= spk; //implicitly gets and displays spike via TBar.OnChange event handle
  end;
  tbPlay.Down := false;
  tbReversePlay.Down := false;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbStopClick(Sender: TObject);
begin
  TrackBar.Position:= 0;
  tbPlay.Down := false;
  tbReversePlay.Down := false;
  tbPause.Down:= false;
end;

procedure TSurfBawdForm.tbStepFowardClick(Sender: TObject);
begin
  tbPause.Down:= false;
  TrackBar.Position:= TrackBar.Position +1;
  if tbLocNSort.Down then PredNSort;
end;

procedure TSurfBawdForm.tbStepBackClick(Sender: TObject);
begin
  tbPause.Down:= false;
  Trackbar.Position:= TrackBar.Position -1;
  if tbLocNSort.Down then PredNSort;
end;

procedure TSurfBawdForm.TrackBarKeyPress(Sender: TObject; var Key: Char);
begin
   if not((key = 'm') or (key = 'M')) then exit;
   {if TrackBar.SelStart > TrackBar.Position then TrackBar.SelStart:= TrackBar.Position
    else begin
      TrackBar.SelEnd:= TrackBar.SelStart;
      TrackBar.SelStart:= TrackBar.Position;
    end;}
   //again, range marking needs debugging!!!
end;

procedure TSurfBawdForm.StatusBarDblClick(Sender: TObject);
begin
  DisplayStatusBar:= not (DisplayStatusBar);
  if DisplayStatusBar then UpdateStatusBar else
  begin
    StatusBar.Panels[0].Text:= '';
    StatusBar.Panels[1].Text:= '';
  end
end;

procedure TSurfBawdForm.tbRethresholdClick(Sender: TObject);
var spk, w, threshold : integer;
  c : byte;
  keepit : boolean;
begin
  ToolBar1.Enabled:= false;
  ToolBar2.Enabled:= false;
  TrackBar.Enabled:= false; //why don't they appear dimmed?

  Threshold:= 2048 + ProbeWin[m_CurrentProbe].win.SThreshold.value;

  for spk:= 0 to m_NSpikes[m_CurrentProbe] do //!!!add code to exclude spikes outside of trackbar range
  begin
    SurfFileAccess.GetSpike(m_CurrentProbe,Spk,Spike);
    Keepit:= false;
    for c:= 0 to ProbeArray[m_CurrentProbe].numchans-1 do
    begin
      for w:= ProbeArray[m_CurrentProbe].trigpt to ProbeArray[m_CurrentProbe].pts_per_chan-1 do
      begin
        if Spike.waveform[c,w] > Threshold then
        begin
          Keepit:= true;
          break;
        end;
      end;
      if Keepit then break;
    end;
    if not(keepit) then SurfFileAccess.SetSpikeClusterID(m_CurrentProbe,Spk,-1) //exclude
      else SurfFileAccess.SetSpikeClusterID(m_CurrentProbe,Spk,0); //include (but will overwrite ID)
    TrackBar.Position:= spk;  //implicit way to display waveforms and progress thru file
  end;
  ToolBar1.Enabled:= true;
  ToolBar2.Enabled:= true;
  TrackBar.Enabled:= true;
  tbRethreshold.Down:= false;
end;

procedure TSurfBawdForm.tbToglStatsWinClick(Sender: TObject);
var i : byte;
begin
  if tbToglStatsWin.down = true then
  begin
    WaveformStatsWin := TInfoWin.CreateParented(Waveforms.Handle);
    with waveformstatswin do
    begin
      for i := 0 to High(ParamNames) do
        AddInfoLabel(ParamNames[i]);
      Constraints.MinWidth:= 200;
      Top := Toolbar2.Height + 10;
      Left := ProbeWin[m_CurrentProbe].Win.Width + 10;
      Caption := 'Waveform Parameters';
      BringToFront;
    end;
    WaveFormStatsUpdate;
  end else
  WaveFormStatsWin.Release;
end;

procedure TSurfBawdForm.WaveFormStatsUpdate;
var rx, ry : single;
  i : smallint;
begin
  if not tbToglStatsWin.Down then exit;
  ComputeWaveformParams(Spike, True, m_CurrentChan);
  with WaveformStatsWin do
  begin
    ChangeInfoData('Channel',inttostr(m_CurrentChan));
    ChangeInfoData('Peak', inttostr(Spike.param[m_CurrentChan,1])+' µV');
    ChangeInfoData('Valley', inttostr(Spike.param[m_CurrentChan,2])+' µV');
    ChangeInfoData('Amplitude', inttostr(Spike.param[m_CurrentChan,3])+' µV');
    ChangeInfoData('TimeAtPeak', inttostr(Spike.param[m_CurrentChan,4]-ProbeArray[m_CurrentProbe].trigpt)+' µs');
    ChangeInfoData('Width', inttostr(Spike.param[m_CurrentChan,5]-Spike.param[m_CurrentChan,4])+' µs');
    ChangeInfoData('Area', inttostr(Spike.param[m_CurrentChan,6])+' µV.ms');
    if Spike.param[m_CurrentChan,0]=-1 then ChangeInfoData('Polarity', 'Inverted')
      else ChangeInfoData('Polarity', 'Positive');

    for i := 1 to 32 do  // display zoomed waveform...
    begin
      spliney[i] := Spike.Waveform[m_CurrentChan,i-1];
      splinex[i] := i;
      spline2ndderiv[i] := 0;
    end;
    Spline(splinex, spliney,32,spline2ndderiv);    // cubic spline the wave

    Canvas.Brush.Color:= clBtnFace;
    Canvas.FillRect(Rect(97,0,ClientWidth,ClientHeight)); // blank plot space
    for i := 1 to 32 * SURF_SPLINE_RES do
    begin
      rx := i/SURF_SPLINE_RES;
      Splint(splinex,spliney,spline2ndderiv,32,rx,ry);
      Canvas.Pixels[round((rx-1)*3)+100, 145-round(ry/25)]:= clGray; // plot spline points
    end;

    for i := 0 to 31 do    // now plot the raw waveform data
      Canvas.Pixels[i*3+100, 145-Spike.Waveform[m_CurrentChan, i]div 25]:=clRed;
  end;
  {  for i:= 0 to high (Spike.param) do
  begin
    WaveformStatsWin.ChangeInfoData(paramnames[i+1],inttostr(Spike.param[i]));
  end;}
end;

procedure CProbeWin.NotAcqOnMouseMove (ChanNum : byte);
begin
  SurfBawdForm.m_CurrentChan:= ChanNum;
  SurfBawdForm.WaveFormStatsUpdate; //update waveform stats
end;

function TSurfBawdForm.ADValue2uV(ADValue : SmallInt): SmallInt;
begin
  Result:= Round((ADValue - 2048)*(10 / (2048 * ProbeArray[m_CurrentProbe].IntGain
                 * ProbeArray[m_CurrentProbe].ExtGain[m_CurrentChan]))
                 * V2uV);
end;

procedure TSurfBawdForm.tbToglWformClick(Sender: TObject);
begin
  ProbeWin[m_CurrentProbe].win.FormDblClick(tbToglWform); //blanks waveform window
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbLocNSortClick(Sender: TObject);
begin
  {should make LocSortForm runttime created}
  {should check if not already there...} LocSortForm.Show;
  LocSortForm.BringToFront;
  LocSortForm.CreateElectrode(ProbeArray[m_CurrentProbe].electrode_name,
    ProbeArray[0].IntGain, ProbeArray[0].ExtGain[0]); // what if ext. gains vary from channel to channel?!

{  tbPlay.Down:= true;
  tbPlayClick(nil);}
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.tbExportDataClick(Sender: TObject);
var
  rx, ry : single;
  c,w,spk,maxspikes : integer;
  Output : TextFile;
  OutFileName : shortstring;

begin
  OutFileName := FileNameOnly;
  SetLength(OutFileName,Length(OutFileName)-3);
  OutFileName := 'C:/Documents and Settings/Administrator/Desktop/' + OutFileName + 'txt';
  AssignFile(Output, OutFileName);
  if FileExists(OutFileName)
    then Append(Output)
    else Rewrite(Output);
  spk:= TrackBar.Position;       //export from current spike...
  maxspikes:= spk + 100;        //...to current spike + maxspikes (or eof)

  while (spk <= m_NSpikes[m_CurrentProbe]) and (spk < maxspikes) do
  begin
    SurfFileAccess.GetSpike(m_CurrentProbe,Spk,Spike);
    if Spike.Cluster > -1 then // only export spikes with cluster ID
    begin
      for c := 0 to ProbeArray[m_CurrentProbe].numchans-1 do
      begin
        for w := 1 to ProbeArray[m_CurrentProbe].pts_per_chan-1 do
        begin
          spliney[w] := Spike.Waveform[c,w-1];
          splinex[w] := w;
          spline2ndderiv[w] := 0;
        end;
        Spline(splinex, spliney,ProbeArray[m_CurrentProbe].pts_per_chan-1,spline2ndderiv); // cubic spline the wave

        for w := 1 to {ProbeArray[m_CurrentProbe].pts_per_chan-1} 32 * 39{SURF_SPLINE_RES} do
        begin
          rx := w/39{SURF_SPLINE_RES}; //~0.8us per point - makes for easy ADC S/H correction
          Splint(splinex,spliney,spline2ndderiv,32,rx,ry);
          Write (Output, inttostr(round(ry)-2048)+' '); // export splined, DC corrected waveform
        end;
       {for c := 0 to ProbeArray[m_CurrentProbe].numchans-1 do
           Write(Output, inttostr(Spike.waveform[c,w]-2048),' ');} // for raw data export
      Writeln(Output);//end of line
      end;
    end;
    inc (spk);
    Writeln(Output);//spike delimiter
  end;
  CloseFile(Output);
  tbExportData.Down := false;
end;

{---------------------------------------------------------------------------------------------}
procedure TSurfBawdForm.PredNSort;
var z,vo,m,ro,  chi : single;
begin
  if ProbeArray[m_CurrentProbe].ProbeSubType = SPIKEEPOCH {'S'} then //spike epoch record found
    if Spike.Cluster >= 0 then
    begin
      z := 100;//strtofloat(ez.text);
      vo := 10;//strtofloat(evo.text);
      m := 50;//strtofloat(em.text);
      ro := 10;//strtofloat(eo.text);
 {!!!}ComputeWaveformParams(Spike); // STUPID, if waveformstats unit displayed, computed twice!
      LocSortForm.ComputeLoc(Spike, z, vo, m, ro, true, false, false, false, 0, chi
      {LockZ.Checked,LockVo.Checked,LockM.Checked,LockO.Checked,FunctionNum.ItemIndex,chi});
    end;
end;

procedure TSurfBawdForm.tbRasterPlotClick(Sender: TObject);
var e, index : integer;
  id, current_ori : smallint;
  CR : TCr;
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
  Setlength(m_SpikeTimeStamp, length(SurfFileAccess.GetEventArray)); //allocate space for all units
  Setlength(m_SpikeClusterID, length(SurfFileAccess.GetEventArray));
  index:= 0;
  for e:= 0 to length(SurfFileAccess.GetEventArray)-1 do
  begin
    if SurfFileAccess.GetEventArray[e].SubType = 'S' then
    begin
      id:= SurfFileAccess.GetClusterID(m_CurrentProbe, e);
      if id >= 0 then //currently, shows spike rasters of clustered and unclustered units
      begin
        m_SpikeTimeStamp[index] := SurfFileAccess.GetEventArray[e].Time_Stamp; //need to call every time?
        m_SpikeClusterID[index] := id;
        inc(index);// must be a better way!!!
      end;
    end;
  end;

  //retrive all orientation and phase information
  Setlength (ori, SurfFileAccess.GetNumSVals);
  Setlength (phase, SurfFileAccess.GetNumSVals);
  SetLength (phase_time, SurfFileAccess.GetNumSVals);

  for e:= 0 to SurfFileAccess.GetNumSVals-1 do
  begin
    SurfFileAccess.GetSVal(e, SVal);
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

  {for e:= 0 to SurfFileAccess.GetNumCRs(1)-1 do
  begin
    SurfFileAccess.GetCR(1, e, CR);
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
    showmessage('Orientation: '+inttostr(current_ori)
                +' had '+inttostr(oribin[c])+' spikes.');
  end;
end;

end.

