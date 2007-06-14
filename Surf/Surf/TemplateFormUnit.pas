{ (c) 2003 Tim Blanche, University of British Columbia }
unit TemplateFormUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, DTxPascal, SurfPublicTypes, StdCtrls, ComCtrls, ToolWin, ElectrodeTypes,
  Menus, ImgList, Spin, SurfMathLibrary;

const TOPMARGIN = 50;
     LEFTMARGIN = 8;
     DEFAULT_FIT_THRESHOLD = 20{uV}; // one stdev dev. per template sample pt.
     MAX_N_PER_TEMPLATE = 100;
     MIN_ACCEPTABLE_FIT_AMPLITUDE = 10{uV}; //points in template less than +/- this value are not fit
type

  TSortCriteria = (scMaxChan, scEnabled, scDecreasingN, scIncreasingN, scSimilarity);

  TSourceBufferStatus = (NotSearched, Searched); //flags indicate if templates
                                                 //derived from this buffer index
  TResidualMinima = record
    TimeStamps : array of int64;
    Residuals  : array of cardinal;
    n          : integer;
  end;

  TTemplateWin = class(TForm)
    TabControl: TTabControl;
    TabImage: TImage;
    TabPanel: TPanel;
    Label3: TLabel;
    cbLocked: TCheckBox;
    cbEnabled: TCheckBox;
    AllPanel: TPanel;
    rgFitMethod: TRadioGroup;
    rgErWeight: TRadioGroup;
    Panel1: TPanel;
    tbTemplate: TToolBar;
    tbOpenFile: TToolButton;
    tbSaveFile: TToolButton;
    tbReset: TToolButton;
    tbImages: TImageList;
    lblNTemplates: TLabel;
    lbNumTemplates: TLabel;
    seTempRadius: TSpinEdit;
    Label5: TLabel;
    seMaxSpikes: TSpinEdit;
    Label6: TLabel;
    Label7: TLabel;
    Label9: TLabel;
    lbSampleTime: TLabel;
    Label10: TLabel;
    cbRandomSample: TCheckBox;
    tbDump: TButton;
    tbRawAvg: TButton;
    UpDown1: TUpDown;
    Label4: TLabel;
    cbViewOrder: TComboBox;
    tbDel: TButton;
    SaveTemplates: TSaveDialog;
    OpenTemplates: TOpenDialog;
    tbExtractRaw: TButton;
    tbBuildTemplates: TButton;
    udNoise: TUpDown;
    Label1: TLabel;
    seFitThresh: TSpinEdit;
    Label2: TLabel;
    cbShowFits: TCheckBox;
    rgChartDisp: TRadioGroup;
    tbShrinkAll: TButton;
    seNHistBins: TSpinEdit;
    Label11: TLabel;
    Label12: TLabel;
    lbNumSamples: TLabel;
    seNSamp: TSpinEdit;
    seMaxClust: TSpinEdit;
    Label8: TLabel;
    cbGlobalEnable: TCheckBox;
    ToolButton1: TToolButton;
    procedure FormCreate(Sender: TObject);
    procedure TabControlChange(Sender: TObject);
    procedure cbLockedClick(Sender: TObject);
    procedure seTempRadiusChange(Sender: TObject);
    procedure cbEnabledClick(Sender: TObject);
    procedure tbDumpClick(Sender: TObject);
    procedure tbRawAvgClick(Sender: TObject);
    procedure UpDown1Click(Sender: TObject; Button: TUDBtnType);
    procedure cbViewOrderChange(Sender: TObject);
    procedure tbDelClick(Sender: TObject);
    procedure tbSaveFileClick(Sender: TObject);
    procedure tbOpenFileClick(Sender: TObject);
    procedure tbExtractRawClick(Sender: TObject);
    procedure tbResetClick(Sender: TObject);
    procedure tbBuildTemplatesClick(Sender: TObject);
    procedure udNoiseClick(Sender: TObject; Button: TUDBtnType);
    procedure seFitThreshChange(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure FormKeyUp(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure tbShrinkAllClick(Sender: TObject);
    procedure TabControlMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure cbGlobalEnableClick(Sender: TObject);
  private
    ScreenY : array[0..RESOLUTION_12_BIT] of integer;
    ScaledWaveformHeight : real;
    procedure AD2ScreenXY;
    procedure SaveTemplateFile(SaveFilename : string);
    procedure OpenTemplateFile(OpenFilename : string);
    procedure EraseAllTemplates;
    //procedure InitialiseWaveformBM;
    { Private declarations }
  public
    SourceFile : shortstring;
    SourceBuffers : array of TSourceBufferStatus;
    NBuffersSearched : integer;
    SpikeTemplates   : array of TSpikeTemplate;
    GlobalFitResults : array of TResidualMinima;

    TabTemplateOrder : array of integer;
    m_PlotRaw, FitToFileEnabled, ViewRawPlotsEnabled : boolean;
    AltKeyDown, ControlKeyDown : boolean;
    m_RawIdxStart, m_RawIdxEnd, TabSelectA : integer;
    Electrode : TElectrode;
    NumTemplates, NTotalSpikes : integer;
    NumChans, NumWavPts : integer; //number of sites/points per site to plot
    AD2uV, AD2usec : single; //gain factor, uV/bit and usec/sample pt
    NearSites, AdjacentSites : TSiteArray;
    procedure CreateNewTemplate(const SitesInTemplate : TSites; nsites, tlength : integer);
    procedure SetPlotBoundaries(var SpikeTemplate : TSpikeTemplate);
    procedure Add2Template(TemplateIndex : integer; const Waveform : TWaveform);//(var Template : TSpikeTemplate; const Waveform : TWaveform);
    procedure DropTemplateChan(var SpikeTemplate : TSpikeTemplate; setindex : integer);
    procedure ComputeTemplateMaxChan(TemplateIdx : integer);
    function  MatchTemplate(const NewTemplate, ExistingTemplate : TSpikeTemplate) : integer; //-1 = no match
    function  CompareTemplates(const Template1, Template2 : TSpikeTemplate) : double;

    procedure PlotTemplate(TemplateIndex : integer; Colour : TColor = clWhite;
                           PlotHidden : boolean = False);//(const Template : TSpikeTemplate);
    procedure PlotTemplateEpoch(const WaveformEpoch : TWaveform; Offset : integer;
                                SkipPtsForRawBuffer : integer = 0);
    procedure ShrinkTemplate(TemplateIndex : integer);
    procedure SortTemplates(Order : TSortCriteria);
    procedure BlankTabCanvas;
    procedure ChangeTab(TabIndex : integer); virtual; abstract;
    procedure ReloadSpikeSet; virtual; abstract;
    procedure BuildTemplates; virtual; abstract;
    procedure CombineTemplates(TemplateIdxA, TemplateIdxB : integer); virtual; abstract;
    procedure DeleteTemplate(TemplateIndex : integer); virtual; abstract;
    procedure SplitTemplate(TemplateIndex : integer); virtual; abstract;
    procedure ToggleClusterLock(TemplateIndex : integer); virtual; abstract;
    procedure DeleteTab(delTabIndex : integer);
    destructor Destroy; override;                   //only virtual 'cause SpikeSet owned by Surfbawd
    { Public declarations }
  protected
    procedure WMEraseBkGnd(var Msg: TWMEraseBkGnd);  message WM_ERASEBKGND;
  end;

var
  TemplateWin: TTemplateWin;

implementation

{$R *.DFM}

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.FormCreate(Sender: TObject);
begin
  NumTemplates:= 0;
  ControlStyle:= ControlStyle + [csOpaque]; //reduce flicker
  TabControl.ControlStyle:= TabControl.ControlStyle + [csOpaque]; //reduce flicker
  ScaledWaveformHeight:= Height / 3;
  AD2ScreenXY; //compute xy-axis waveform plotting LUT
 { with TabImage.Canvas.Font do
  begin
    Color:= clYellow;
    Size:= 6;
  end;}  //screws up background painting --- somehow???!!!
end;

{-------------------------------------------------------------------------}
procedure TTemplateWin.AD2ScreenXY;
var i : integer;
begin //LUT optimises waveform plotting...
  for i:= 0 to RESOLUTION_12_BIT do
    ScreenY[i]:= Round((i-2047)/2047 * ScaledWaveformHeight * 1.7{2.7{user VZoom?});
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.CreateNewTemplate(const SitesInTemplate : TSites; nsites, tlength : integer);
begin
  inc(NumTemplates);
  if NumTemplates > Length(SpikeTemplates) then
    SetLength(SpikeTemplates, NumTemplates + 10); //allocate space for more templates
  with SpikeTemplates[NumTemplates - 1] do
  begin
    Sites:= SitesInTemplate;
    NumSites:= nsites;
    PtsPerChan:= tlength div nsites;
    SetLength(SiteOrigin, NumSites);
    SetPlotBoundaries(SpikeTemplates[NumTemplates - 1]);
    n:= 0;
    FitThreshold:= Round(Sqr(DEFAULT_FIT_THRESHOLD / AD2uV) * tlength);
    Setlength(AvgWaveform, tlength);
    Setlength(SumWaveform, tlength);
    Setlength(SSqWaveform, tlength);
    Setlength(StdWaveform, tlength);
    Locked:= False;
    Enabled:= True;
  end;
  if NumTemplates = 1 then TabControl.Tabs.Add('All');
  TabControl.Tabs.Add(inttostr(NumTemplates));
  lbNumTemplates.Caption:= inttostr(NumTemplates);
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.SetPlotBoundaries(var SpikeTemplate : TSpikeTemplate);
var i, s, topmost, leftmost, rightmost, bottommost : integer;
begin
  {get boundaries of site coords for this template}
  topmost:= 10000;
  leftmost:= 10000;
  bottommost:= -10000;
  rightmost:= -10000;
  with SpikeTemplate, Electrode do
  begin
    for s:= 0 to NumSites - 1 do
     begin
      //if not (s in Sites) then Continue;
      if topmost > SiteLoc[s].y
        then topmost:= SiteLoc[s].y;
      if leftmost > SiteLoc[s].x
        then leftmost:= SiteLoc[s].x;
      if rightmost < SiteLoc[s].x
        then rightmost:= SiteLoc[s].x;
      if bottommost < SiteLoc[s].y
        then bottommost:= SiteLoc[s].y;
    end;
    dec(bottommost, topmost); //adjust boundaries for those...
    dec(rightmost, leftmost); //...sites in template
    i:= 0;
    for s:= 0 to NumSites - 1 do
    begin
      //if not (s in Sites) then Continue;
      SiteOrigin[i].x:= Round((Electrode.SiteLoc[s].x - LeftMost + LEFTMARGIN) * 3{5.0* ScaledWaveformHeight});
      SiteOrigin[i].y:= Round((Electrode.SiteLoc[s].y - TopMost + TOPMARGIN) / 1.5){0.3}{* ScaledWaveformHeight){- 350};
      if bottommost < SiteOrigin[i].y
        then bottommost:= SiteOrigin[i].y;
      if rightmost < SiteOrigin[i].x
        then rightmost:= SiteOrigin[i].x;
      inc(i);
    end;
    with PlotBounds do
    begin
      Top:= topmost div 2;
      Left:= leftmost - 100;
      Right:= rightmost + {1}200;
      Bottom:= round(bottommost / 1.6) + 200;
    end;
  end{Electrode};
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.Add2Template(TemplateIndex : integer; const Waveform : TWaveform);//(var Template : TSpikeTemplate; const Waveform : TWaveform);
var i : integer;
begin
  if NumTemplates = 0 then Exit;
  with SpikeTemplates[TemplateIndex] do
  begin
    inc(n);
    for i:= 0 to High(Waveform) do
    begin
      SumWaveform[i]:= SumWaveform[i] + Waveform[i]; //optimize with delphi math unit
      AvgWaveform[i]:= Round(SumWaveform[i] / n);    //routines written in asm???!!!
      SSqWaveform[i]:= SSqWaveform[i] + Waveform[i] * Waveform[i];
      if n = 1 then StdWaveform[i]:= 0
        else StdWaveform[i]:= sqrt((SSqWaveform[i] - (SumWaveForm[i] * SumWaveForm[i])/n) / (n - 1));
    end;
    if n = seMaxSpikes.Value then
    begin
      Locked:= True;
      cbLocked.Checked:= SpikeTemplates[TabControl.TabIndex-2].Locked;
    end;
    {recompute maxchan here!}
    (*pkpk:= 0;
    MaxChan:= 0;
    PtsPerChan:= index div tchans;
    for i:= 0 to tchans - 1 do //find channel with maximum peak-peak amplitude...
    begin
      min:= AvgWaveform[i*PtsPerChan];
      max:= AvgWaveform[i*PtsPerChan];
      for j:= 1 to PtsPerChan - 1 do
      begin
        if max < AvgWaveform[i*PtsPerChan + j] then
          max:= AvgWaveform[i*PtsPerChan + j];
        if min > AvgWaveform[i*PtsPerChan + j] then
          min:= AvgWaveform[i*PtsPerChan + j];
      end{j};
      if (max - min) > pkpk then
      begin
        pkpk:= max - min;
        MaxChan:= i;
      end;
    end{i};
    for i:= 0 to NumChans - 1 do
      if i in Sites then
        if MaxChan = 0 then MaxChan:= i
          else dec(MaxChan);*)
  end;
end;

{-------------------------------------------------------------------------------------}
function TTemplateWin.MatchTemplate(const NewTemplate, ExistingTemplate : TSpikeTemplate) : integer;
var i, j, k, residual, dist, idx, idx2 : integer; CommonSites : TSites;
begin
  Result:= -1;
  if NewTemplate.MaxChan in AdjacentSites[ExistingTemplate.MaxChan] then
    CommonSites:= NearSites[NewTemplate.MaxChan] *{intersection}NearSites[ExistingTemplate.MaxChan]
  else Exit; //sites don't match (or overlap) so exit...
  k:= 0;
  residual:= 0;
  idx:= 0;
  idx2:= 0;
  with ExistingTemplate do
  for i:= 0 to NumChans - 1 do
  begin
    if i in CommonSites then
    begin
      for j:= 0 to PtsPerChan -1 do
      begin
        dist:= AvgWaveform[idx2] - NewTemplate.AvgWaveform[idx];
        inc(residual, dist * dist);
        inc(idx);
        inc(idx2);
      end{j};
      inc(k);
      Continue;
    end
    {else if i in NearSites[NewTemplate.MaxChan]
      then inc(idx, PtsPerChan)}
    else if i in NearSites[ExistingTemplate.MaxChan]
      then inc(idx2, PtsPerChan);
    inc(idx, PtsPerChan);
  end{i};
  if residual < (10000{40000 for +/-25uV at int gain = 8} * k * ExistingTemplate.PtsPerChan) then Result:= Residual;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.PlotTemplate(TemplateIndex : integer; Colour : TColor{ = clWhite};
                                    PlotHidden : boolean{ = False});//(const Template : TSpikeTemplate);
var c, w, y, pixelsperpt : integer;
const vTetSites = [26, 29, 40, 41];//[8, 9, 31, 22];//[9, 10, 23, 31]; //temporary
begin
if TabControl.TabIndex = 0 then Exit;
with TabImage.{Picture.Bitmap.}Canvas do //WaveformBM.Canvas do
  begin
    Font.Color:= clYellow;  // this should be set elsewhere...
    with SpikeTemplates[TemplateIndex] do
    begin
      Pen.Mode:= {pmMerge;}pmCopy;
      //Pen.Width:= 3;
      //Pen.Color:= ColorTable[TemplateIndex];
      pixelsperpt:= 5{6} - PtsPerChan div 25; //REMOVE HARDCODING HACK!
      for c:= 0 to NumSites -1 do
      begin
        if c {in vTetSites then} in Sites then
          Pen.Color:= Colour
        else if PlotHidden then
          Pen.Color:= clDkGray
        else Continue;
          MoveTo(SiteOrigin[c].x, SiteOrigin[c].y - ScreenY[AvgWaveform[c * PtsPerChan] and $FFF]);
          for w:= 1 to PtsPerChan - 1 do
          begin //could be optimised, for example, reduce call overhead with polyline/gon methods...
            y:= ScreenY[AvgWaveform[c * PtsPerChan + w] and $FFF];
            LineTo(SiteOrigin[c].x + w * pixelsperpt, SiteOrigin[c].y - y);
          end{w};
        //end{site};
      end{c};
      //if showchannumbers then...
      (*w:= 0;
      for c:= 0 to NumChans - 1 do
        if c in Sites then
        begin
          TextOut(SiteOrigin[w].x, SiteOrigin[w].y - 15, inttostr(c));
          inc(w);
        end;
      //if showstdev then...
      (*Pen.Mode:= pmMerge;
      Pen.Color:= cldkGray;
      for c:= 0 to NumSites -1 do
        for w:= 0 to PtsPerChan - 1 do
        begin //could be optimised, for example, reduce lineto call overhead with polyline/gon methods...
          MoveTo(SiteOrigin[c].x + w * pixelsperpt, SiteOrigin[c].y - ScreenY[AvgWaveform[c * PtsPerChan + w] - Round(StdWaveform[c * PtsPerChan + w])]);
          LineTo(SiteOrigin[c].x + w * pixelsperpt, SiteOrigin[c].y - ScreenY[AvgWaveform[c * PtsPerChan + w] + Round(StdWaveform[c * PtsPerChan + w])]);
        end{w};*)
      Label3.Caption:= 'n = ' + inttostr(n);
    end;
  end;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.PlotTemplateEpoch(const WaveformEpoch : TWaveform; Offset : integer;
                                         SkipPtsForRawBuffer : integer {default = 0});
var c, {s,} w, y, pixelsperpt : integer;
begin
  if TabControl.TabIndex = 0 then Exit;
  with TabImage.Canvas do //WaveformBM.Canvas do
  begin
    with SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]] do
    begin
      if SkipPtsForRawBuffer = 0 then SkipPtsForRawBuffer:= PtsPerChan;
      Pen.Mode:= pmCopy; //??? make colour additive
      Pen.Color:= ColorTable[(TabControl.TabIndex - 2) mod (High(ColorTable) + 1)];
      pixelsperpt:= 5{6} - PtsPerChan div 25; //REMOVE HARDCODING HACK!
      //s:= 0;
      for c:= 0 to NumChans - 1 do
      begin
        if c in Sites then
        begin
          MoveTo(SiteOrigin[c].x, SiteOrigin[c].y - ScreenY[WaveformEpoch[offset] and $FFF]);
          for w:= 1 to PtsPerChan - 1 do
          begin //could be optimised, for example, reduce call overhead with polyline/gon methods...
            y:= ScreenY[WaveformEpoch[offset + w] and $FFF];
            LineTo(SiteOrigin[c].x + w * pixelsperpt, SiteOrigin[c].y - y);
          end{w};
          //inc(s);
        end;
        inc(offset, SkipPtsForRawBuffer); //silly hack to accomoodate raw/interp buffers
        //inc(offset, PtsPerChan);        //instead of continguous template epochs (eg. spikeset)
      end{c};
    end;
  end;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.tbBuildTemplatesClick(Sender: TObject);
begin
  BuildTemplates;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.ShrinkTemplate(TemplateIndex : integer);
var c, p, i, s, min, max : integer; x, chanstd : single;
begin
  //for t:= 0 to NumTemplates - 1 do
  with SpikeTemplates[TemplateIndex] do
  begin
    i:= 0;
    c:= 0;
    s:= 0;
    while c < NumSites do
    begin
      if c in Sites then inc(s); // tally number of sites before shrink
      {chanstd:= 0; //calculate stdev for this channel
      for p:= 0 to PtsPerChan - 1 do
      begin
        x:= AvgWaveform[i] - 2048;
        chanstd:= chanstd + x*x;
        inc(i);
      end;
      chanstd:= sqrt(chanstd / PtsPerChan);}
      // calculate peak-peak amplitude for this channel
      min:= AvgWaveform[i];
      max:= AvgWaveform[i];
      inc(i);
      for p:= 1 to PtsPerChan - 1 do
      begin
        if AvgWaveform[i] < min then
          min:= AvgWaveform[i] else
        if AvgWaveform[i] > max then
          max:= AvgWaveform[i];
        inc(i);
      end;
      chanstd{peak2peak}:= (max - min) * AD2uV;

      if chanstd <= udNoise.Position then
        Exclude(Sites, c)
        //DropTemplateChan(SpikeTemplates[TemplateIndex], c); DON'T DISCARD, ONLY EXCLUDE CHANNELS
        //dec(i, PtsPerChan);
      else
        Include(Sites, c);
      inc(c);
    end{c};
    { adjust threshold for # active sites }
    c:= 0;
    for i:= 0 to NumSites -1 do //tally number of sites after shrink
      if i in Sites then inc(c);
    if (c = 0) or (s = 0) then FitThreshold:= 0
      else FitThreshold:= Round(FitThreshold / (s / c));
  end{with};
  //if numsites = 0 then DeleteTemplate;
  //end{t};
end;

{-------------------------------------------------------------------------------------}
(*procedure TTemplateWin.DeleteTemplate(TemplateIndex : integer);
var i, j, k : integer;
begin
  if TemplateIndex >= NumTemplates then Exit;
  with SpikeTemplates[TemplateIndex] do
  begin
    if n > 1 then ShellSort(Members, ssDescending); //minimises shuffling of SpikeSet array data
    dec(NTotalSpikes, n);
    { adjust remaining member indexes minus members from deleted cluster }
    for j:= 0 to NumTemplates - 1 do
      if j = TemplateIndex then Continue else
        for k:= 0 to SpikeTemplates[j].n - 1 do
          for i:= 0 to n - 1 do
            if SpikeTemplates[j].Members[k] > Members[i] then
              dec(SpikeTemplates[j].Members[k]);
    //this will only work if also concat. SpikeSet!!!
    for i:= TemplateIndex to NumTemplates - 2 do
    begin
      SpikeTemplates[i]:= SpikeTemplates[i+1];
      //GlobalFitResults need concatenating? others?
    end;
    dec(NumTemplates);
    Setlength(SpikeTemplates, NumTemplates);
    //Setlength(GlobalFitResults, NumTemplates);
    //SpikeSet, Others too?
  end{with};
end;
*)
{-------------------------------------------------------------------------------------}
(*procedure TTemplateWin.CollapseTemplates;
var i, j, k, t1, t2, residual, dist, idx, idx2 : integer; CommonSites : TSites;
begin
  for t1:= 0 to NumTemplates - 1 do
    for t2:= 0 to NumTemplates - 1 do
    begin
      if t1 = t2 then Continue;
      if SpikeTemplates[t1].MaxChan in AdjacentSites[SpikeTemplates[t2].MaxChan] then
        CommonSites:= NearSites[NewTemplate.MaxChan] *{intersection}NearSites[ExistingTemplate.MaxChan]
      else Continue;

    end{t2};

  else Exit;
  k:= 0;
  residual:= 0;
  idx:= 0;
  idx2:= 0;
  with ExistingTemplate do
  for i:= 0 to NumChans - 1 do
  begin
    if i in commonsites then
    begin
      for j:= 0 to PtsPerChan -1 do
      begin
        dist:= {ExistingTemplate.}AvgWaveform[idx2] - NewTemplate.AvgWaveform[idx];
        inc(residual, dist * dist);
        inc(idx);
        inc(idx2);
      end{j};
      inc(k);
    end
    else if i in NearSites[NewTemplate.MaxChan]
      then inc(idx, PtsPerChan)
    else if i in NearSites[ExistingTemplate.MaxChan]
      then inc(idx2, PtsPerChan);
  end{i};
  if residual < (40000{+/-25uV at int gain = 8} * k * ExistingTemplate.PtsPerChan) then Result:= Residual;
end;
*)
{-------------------------------------------------------------------------------------}
procedure TTemplateWin.DropTemplateChan(var SpikeTemplate : TSpikeTemplate; setindex : integer);
var i, l, arrayidx, idx : integer;
begin
  with SpikeTemplate do
  begin
    idx:= setindex;
    arrayidx:= setindex * PtsPerChan;
    for i:= 0 to NumChans - 1 do
      if i in Sites then
        if idx = 0 then
        begin
          Exclude(Sites, i);
          Break;
        end else
          dec(idx);
    {remove data for excluded channel, concatenate remaining sites, and shrink arrays}
    l:= Length(AvgWaveform) - PtsPerChan; //new array size
    if arrayidx < l then //not last site in template so...
    begin //...concatenate arrays
      Move(AvgWaveform[arrayidx + PtsPerChan], AvgWaveform[arrayidx], (l - arrayidx) * 2{short});
      Move(SumWaveform[arrayidx + PtsPerChan], SumWaveform[arrayidx], (l - arrayidx) * 8{double});
      Move(SSqWaveform[arrayidx + PtsPerChan], SSqWaveform[arrayidx], (l - arrayidx) * 8{double});
      Move(StdWaveform[arrayidx + PtsPerChan], StdWaveform[arrayidx], (l - arrayidx) * 8{double});
      Move(SiteOrigin[setindex + 1], SiteOrigin[setindex], (Length(SiteOrigin) - setindex) * SizeOf(TPoint));
      //SetPlotBoundaries(SpikeTemplate);
    end;
    Setlength(AvgWaveform, l); //shrink arrays...
    Setlength(SumWaveform, l);
    Setlength(SSqWaveform, l);
    Setlength(StdWaveform, l);
    dec(NumSites);
{adjust per numsites}//FitThreshold:= 100000; {40000 = default for +/-25uV assuming int gain = 8}
    Setlength(SiteOrigin, NumSites);
  end{with};
end;

{-------------------------------------------------------------------------------------}
function TTemplateWin.CompareTemplates(const Template1, Template2 : TSpikeTemplate) : double;
var paddedwave : TWaveform;
  i, j, t, padlen, dist : integer; residual : double;
begin { this function slides template2 over template1 to see if they are similar
        returns the minimal residual for +/- half-template width }
  padlen:= Template2.PtsPerChan div 2;
  {pad the first template with zeros on either side of each channel}
  with Template1 do
  begin
    Setlength(paddedwave, Length(AvgWaveform) * 2);
    for i:= 0 to high(paddedwave) do paddedwave[i]:= 0;
    for i:= 0 to NumSites - 1 do
      Move(AvgWaveForm[i*ptsperchan], paddedwave[padlen + ptsperchan * 2 * i], ptsperchan * 2{bytes});
  end;

  {slide template2 on template1 and return smallest residual}
  result:= high(integer);
  with Template2 do
  for t:= 0 to PtsPerChan - 1 do
  begin
    residual:= 0;
    for i:= 0 to NumSites -1 do
      for j:= 0 to PtsPerChan -1 do
      begin
        dist:= paddedwave[t + i * ptsperchan + j]- AvgWaveform[i * ptsperchan + j];
        residual:= residual + sqr(dist);
      end{j};
    residual:= sqrt(residual) / n;
    if residual < result then
      result:= residual;
  end{t};
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.ComputeTemplateMaxChan(TemplateIdx : integer);
var {t,} i, j, min, max, pkpk : integer;
begin {compute maxchan for single templates}
  //for t:= 0 to NumTemplates -1 do
  with SpikeTemplates[TemplateIdx] do
  begin
    pkpk:= 0;
    MaxChan:= 0;
    for i:= 0 to NumSites - 1 do //find channel with maximum peak-peak amplitude...
    begin
      min:= AvgWaveform[i*PtsPerChan];
      max:= AvgWaveform[i*PtsPerChan];
      for j:= 1 to PtsPerChan - 1 do
      begin
        if max < AvgWaveform[i*PtsPerChan + j] then
          max:= AvgWaveform[i*PtsPerChan + j];
        if min > AvgWaveform[i*PtsPerChan + j] then
          min:= AvgWaveform[i*PtsPerChan + j];
      end{j};
      if (max - min) > pkpk then
      begin
        pkpk:= max - min;
        MaxChan:= i;
      end;
    end{i};
  end{with};
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.TabControlChange(Sender: TObject);
var t : integer;
begin
  t:= TabControl.TabIndex - 2;
  if t = -2 then
  begin
    TabPanel.Visible:= False;
    AllPanel.Visible:= True;
    Height:= 0;
    Width:= 0;
  end else
  begin
    if t = -1 then  //messy hack...
    with SpikeTemplates[0] do
    begin
      AllPanel.Visible:= False;
      TabPanel.Visible:= True;
      //tbDel.Enabled:= False;
      //cbShowFits.Enabled:= False;
      cbLocked.Enabled:= False;
      cbEnabled.Enabled:= False;
      udNoise.Enabled:= False;
      seFitThresh.Enabled:= False;
      Width:= PlotBounds.Right - PlotBounds.Left div 3;
      Height:= PlotBounds.Bottom;
    end else
    with SpikeTemplates[TabTemplateOrder[t]] do
    begin
      AllPanel.Visible:= False;
      TabPanel.Visible:= True;
      //tbDel.Enabled:= True;
      cbLocked.Enabled:= True;
      cbEnabled.Enabled:= True;
      udNoise.Enabled:= True;
      cbLocked.Checked:= Locked;
      cbEnabled.Checked:= Enabled;
      seFitThresh.Enabled:= True;
      seFitThresh.Value:= SpikeTemplates[TabTemplateOrder[t]].FitThreshold;
      Width:= PlotBounds.Right - PlotBounds.Left div 3;
      Height:= PlotBounds.Bottom;
    end;
  end;
  UpDown1.Position:= 0;
  ChangeTab(t); //signal to surfbawd that tab changed...
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.BlankTabCanvas;
begin
with TabImage.Canvas do
  begin
    Brush.Color:= clBlack;
    FillRect(ClientRect); //blank template plotting area
  end;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.cbLockedClick(Sender: TObject);
begin
  SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]].Locked:= cbLocked.Checked;
  ToggleClusterLock(TabTemplateOrder[TabControl.TabIndex - 2]);
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.cbEnabledClick(Sender: TObject);
begin
  SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]].Enabled:= cbEnabled.Checked;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.seTempRadiusChange(Sender: TObject);
begin
  //BuildSiteProximityArray(NearSites, seTempRadius.Value, True);
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.tbDumpClick(Sender: TObject);
var decpts, pad, w, c : integer; OutFile : Textfile; distance : real;
var OutFileName : string;
const decfactor = 1; fftpts = 128;
begin
  {export template records to file}
  OutFileName:= 'C:\Desktop\Spikes\Dipoles\';
  OutFileName:= OutFileName + inttostr(TabTemplateOrder[TabControl.TabIndex - 2])+'.csv';
  AssignFile(OutFile, OutFileName);
  Rewrite(OutFile); //overwrites any existing file of the same name
  with SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]] do
  begin
    decpts:= PtsPerChan div decfactor;
    pad:= (fftpts - decpts) div 2;
    //Writeln(OutFile, 'SpikeTemplate, n='+ inttostr(n));
    //Writeln(Outfile, 'Average, StdDev');
    for c:= 0 to NumSites do
    begin
      if not (c in Sites) then Continue; // only export sites with signal
      for w:= 0 to pad {-1} do Writeln(OutFile, '0'{floattostr((AvgWaveform[c*PtsPerChan]-2048)*0.1221)}); //pad with zeros
      w:= c*PtsPerChan ;
      while w < (c+1)*PtsPerChan do
      begin
        Writeln(OutFile, floattostr((AvgWaveform[w]-2048)*0.1221));// + ',' + inttostr(Round(StdWaveform[w])));
        inc(w, decfactor);
      end;
      for w:= 0 to pad -1 do Writeln(OutFile, '0'{floattostr((AvgWaveform[c*PtsPerChan+96]-2048)*0.1221)}); //pad with zeros
    end{c};
  end;
  CloseFile(OutFile);
  //now export Euclidian distances wrt max channel for sites exported
  OutFileName:= 'C:\Desktop\Spikes\Dipoles\';
  OutFileName:= OutFileName + inttostr(TabTemplateOrder[TabControl.TabIndex - 2])+'.inf';
  AssignFile(OutFile, OutFileName);
  Rewrite(OutFile); //overwrites any existing file of the same name
  with SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]] do
  begin
    for c:= 0 to NumSites do
    begin
      if not (c in Sites) then Continue; // only export sites with signal
      Distance:= sqrt(sqr(Electrode.Siteloc[MaxChan].x - Electrode.Siteloc[c].x)+
                      sqr(Electrode.Siteloc[MaxChan].y - Electrode.Siteloc[c].y));
      Writeln(OutFile, floattostr(distance));
      //Writeln(OutFile, inttostr(Electrode.Siteloc[c].x), ',', inttostr(Electrode.Siteloc[c].y), ',0');
    end{c};
  end;
  CloseFile(OutFile);

  //MessageDlg('Template dumped to file. File closed.', mtInformation, [mbOK], 0);
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.tbRawAvgClick(Sender: TObject);
begin
  m_PlotRaw:= not(m_PlotRaw); //toggle view with overplots of raw waveforms
  UpDown1.Enabled:= m_PlotRaw;
  if not m_PlotRaw then tbRawAvg.Caption:= 'Avg';
  UpDown1.Position:= 0;
  ChangeTab(TabControl.TabIndex - 2); //replot template with/without overplots
  TabControl.SetFocus;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.UpDown1Click(Sender: TObject; Button: TUDBtnType);
begin
  ChangeTab(TabControl.TabIndex - 2); //replot template with new set of waveform samples
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.cbViewOrderChange(Sender: TObject);
begin
  SortTemplates(TSortCriteria(cbViewOrder.itemindex));
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.SortTemplates(Order : TSortCriteria);
var TemplateN : array of integer;
  sqrs, sumsqrs, covar, mincovar : double;
  i, j, k : integer;

begin
  Setlength(TemplateN, NumTemplates);
  Setlength(TabTemplateOrder, NumTemplates);
  for i:= 0 to NumTemplates -1 do
    TabTemplateOrder[i]:= i;

  case Order of
    scSimilarity :
    begin //order by similarity (decreasing covariance)
      for i:= 0 to NumTemplates - 1 do
      begin
        mincovar:= high(integer);
        for j:= (i + 1) to NumTemplates - 1 do
        begin
          sumsqrs:= 0;
          with SpikeTemplates[i] do
          begin
            for k:= 0 to High(AvgWaveform) do
            begin
              covar:= SpikeTemplates[j].AvgWaveform[k] * AvgWaveform[k];
              sqrs:= sqr(covar);
              sumsqrs:= sumsqrs + sqrs;
            end{k};
            covar:= sumsqrs;// / (n + SpikeTemplates[j].n); //check if n + n denominator correct!
          end{with};
          if covar < mincovar then mincovar:= covar;
        end{j};
        TemplateN[i]:= Round(mincovar);
      end{i};
      ShellSort(TemplateN, ssDescending, TIntArray(TabTemplateOrder)); //from highest to lowest similarity
    end;
    scMaxChan :
    begin //order by maxchan (dec maxchan)
      for i:= 0 to NumTemplates -1 do
        TemplateN[i]:= SpikeTemplates[i].MaxChan;
      ShellSort(TemplateN, ssDescending, TIntArray(TabTemplateOrder));
    end;
    scEnabled :
    begin //group all enabled templates at start
      for i:= 0 to NumTemplates -1 do
        TemplateN[i]:= Integer(SpikeTemplates[i].Enabled);
      ShellSort(TemplateN, ssDescending, TIntArray(TabTemplateOrder));
    end else
    begin //order by n
      for i:= 0 to NumTemplates -1 do
        TemplateN[i]:= SpikeTemplates[i].n;
      if Order = scDecreasingN then ShellSort(TemplateN, ssDescending, TIntArray(TabTemplateOrder))
       else ShellSort(TemplateN, ssAscending, TIntArray(TabTemplateOrder)); //sort by n
    end{n};
  end{case};
end;

{-------------------------------------------------------------------------------------}
(*procedure TTemplateWin.CombineTemplates(TemplateIdxA, TemplateIdxB : integer);
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

  {with}
  if NClust > 0 then
  begin
    { combine member and spiketime indexes }
    Setlength(Clusters[TemplateIdxB].Members, Clusters[TemplateIdxB].n + SpikeTemplates[TemplateIdxA].n);
    Move(SpikeTemplates[TemplateIdxA].Members[0], Clusters[TemplateIdxB].Members[Clusters[TemplateIdxB].n], Clusters[TemplateIdxA].n * SizeOf(Clusters[TemplateIdxB].Members[0]));
    inc(Clusters[TemplateIdxB].n, SpikeTemplates[TemplateIdxA].n);
    { shrink/concatenate cluster array }
    for i:= TemplateIdxA to NClust - 2 do
      Clusters[i]:= Clusters[i+1];
    dec(NClust);
    Setlength(Clusters, NClust);
    { recompute cluster statistics }
    ComputeCentroid(TemplateIdxB);
    ComputeDistortion(TemplateIdxB);
  end;

  for i:= TemplateIdxA to NumTemplates - 2 do
    SpikeTemplates[i]:= SpikeTemplates[i+1];
  dec(NumTemplates);
  Setlength(SpikeTemplates, NumTemplates);
  Setlength(GlobalFitResults, NumTemplates);
  DeleteTab(TabSelectA);
end;
*)
{-------------------------------------------------------------------------------------}
(*procedure TTemplateWin.Button3Click(Sender: TObject);
var i, j : integer;
begin
  for i:= 0 to NumTemplates - 1 do
  begin
    if SpikeTemplates[i].n < 3 then Continue;
    for j:= (i + 1) to NumTemplates - 1 do
    begin
      if SpikeTemplates[j].n < 3 then Continue;
      if CompareTemplates(SpikeTemplates[i], SpikeTemplates[j]) < 4000 then
        Showmessage(inttostr(i) + ', ' + inttostr(j));
    end{j};
  end{i};
end;
*)
{-------------------------------------------------------------------------------------}
procedure TTemplateWin.tbDelClick(Sender: TObject);
begin
  DeleteTemplate(TabTemplateOrder[TabControl.TabIndex - 2]);
  DeleteTab(TabControl.TabIndex);
  TabControlChange(Self); //refresh tab display
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.DeleteTab(delTabIndex : integer);
var i, bakTabIndex : integer;
begin
  with TabControl do
  begin
    SortTemplates(TSortCriteria(cbViewOrder.itemindex)); //implicitly corrects tab-template order array
    bakTabIndex:= TabIndex;
    Tabs.Delete(TabIndex);
    for i:= bakTabIndex to NumTemplates + 1 do Tabs[i]:= inttostr(i - 1); //update tab labels
    lbNumTemplates.Caption:= inttostr(NumTemplates);
    lbNumSamples.Caption:= inttostr(NTotalSpikes);
    if bakTabIndex = Tabs.Count then
      TabIndex:= bakTabIndex - 1
    else TabIndex:= bakTabIndex;
    if NumTemplates = 0 then
    begin //reinitialise...
      EraseAllTemplates;
      TabIndex:= 0;
    end;
  end{tabcontrol};
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.tbSaveFileClick(Sender: TObject);
begin
  if NumTemplates = 0 then Exit;
  if SaveTemplates.Execute then
    SaveTemplateFile(SaveTemplates.FileName)
  else MessageDlg('File not saved.', mtInformation, [mbOK], 0);
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.SaveTemplateFile(SaveFilename : string);
var fs : TFileStream;
 i : integer;
begin
  try
    SaveFilename:= ExtractFileName(SaveFilename);
    fs:= TFileStream.Create(SaveFilename + '.tem', fmCreate);
    fs.Seek{64}(0, soFromBeginning); //overwrite any existing file
    with fs do
    begin
      { write file header information }
      WriteBuffer('STMP', 4); //header id
      WriteBuffer('v1.0', 4); //write version number
      { write settings common to all templates }

//       MODIFY WRITE DEPENDING ON WHETHER SOURCEFILE/ELECTRODE MATCH PREVIOUSLY SAVED ELECTRODE/FILE
//       SO AS NOT TO FUCK UP THE SAVE FILE...

      WriteBuffer(SourceFile, Length(SourceFile)+1); //+1 for string length
      WriteBuffer(Electrode, SizeOf(Electrode));
      WriteBuffer(AD2uV, SizeOf(AD2uV));
      WriteBuffer(AD2usec, SizeOf(AD2usec));
      WriteBuffer(NumTemplates, SizeOf(NumTemplates));
      WriteBuffer(NBuffersSearched, SizeOf(NBuffersSearched));
      i:= Length(SourceBuffers);
      WriteBuffer(i, SizeOf(i)); //write length of SourceBuffer...
      WriteBuffer(SourceBuffers[0], Length(SourceBuffers){1 byte/flag}); //..and SourceBuffer array
      { finally, write templates }
      for i:= 0 to NumTemplates - 1 do
        with SpikeTemplates[i] do
        begin
          WriteBuffer(Sites, SizeOf(Sites));
          WriteBuffer(NumSites, SizeOf(NumSites));
          WriteBuffer(PtsPerChan, SizeOf(PtsPerChan));
          WriteBuffer(SiteOrigin[0], Length(SiteOrigin) * SizeOf(SiteOrigin[0]));
          WriteBuffer(PlotBounds, SizeOf(PlotBounds));
          WriteBuffer(MaxChan, SizeOf(MaxChan));
          WriteBuffer(n, SizeOf(n)); //'n' samples in avg
          WriteBuffer(Locked, SizeOf(Locked));
          WriteBuffer(Enabled, SizeOf(Enabled));
          WriteBuffer(FitThreshold, SizeOf(FitThreshold));
          WriteBuffer(SumWaveform[0], Length(SumWaveform) * SizeOf(SumWaveform[0]));
          WriteBuffer(SSqWaveform[0], Length(SSqWaveform) * SizeOf(SSqWaveform[0]));
          WriteBuffer(Members[0], n * SizeOf(Members[0]));
          WriteBuffer(SpikeTimes[0], n * SizeOf(SpikeTimes[0]));
        end{i};
      Free;
    end{fs};
  except
    MessageDlg('Error. File not saved', mtError, [mbOK], 0);
    fs.Free;
    Exit;
  end;
  Caption:= 'Templates (' + SaveFilename + ')';
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.tbOpenFileClick(Sender: TObject);
begin
  if NumTemplates <> 0 then
    if MessageDlg('Existing templates will be overwritten.  Continue?', mtWarning,
      [mbYes, mbNo], 0) = mrNo then Exit;
  if OpenTemplates.Execute then
    OpenTemplateFile(OpenTemplates.FileName);
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.OpenTemplateFile(OpenFilename : string);
var fs : TFilestream;
  i, s : integer;
  len  : byte;
  id, ver : array [0..3] of char;
  saved_source_file, current_polytrode_name : shortstring;
begin
  try
    fs:= TFileStream.Create(OpenFilename, fmOpenRead);
    fs.Seek{64}(0, soFromBeginning); //start of file
  except
    MessageDlg('Cannot open file.', mtError, [mbOK], 0);
    Exit;
  end;
  EraseAllTemplates;
  with fs do
  begin
    { read file header information }
    ReadBuffer(id, 4);
    ReadBuffer(ver, 4);
    if (id <> 'STMP') or (ver <> 'v1.0') then
    begin
      MessageDlg('Wrong file type or version.', mtError, [mbOK], 0);
      Free;
      Exit;
    end;
    { read settings common to all templates }
    ReadBuffer(len, 1); //read sourcefile string length
    SetLength(saved_source_file, len); //truncate shortstring
    ReadBuffer(saved_source_file[1], len);
    if saved_source_file <> SourceFile then
    begin
      ViewRawPlotsEnabled:= False;
      MessageDlg('Template source file doesn''t match current open file.' +
        chr(13) + 'Plots of raw waveform epochs is disabled.', mtWarning, [mbOK], 0);
    end else
      tbExtractRaw.Enabled:= True;
    current_polytrode_name:= Electrode.Name;
    ReadBuffer(Electrode, SizeOf(Electrode));
    if current_polytrode_name <> Electrode.Name then
    begin
      FitToFileEnabled:= False;
      MessageDlg('Source templates don''t match current active polytrode.'
        + chr(13) + 'Fitting/adding to templates is disabled.', mtWarning, [mbOK], 0);
    end;
    ReadBuffer(AD2uV, SizeOf(AD2uV));
    ReadBuffer(AD2usec, SizeOf(AD2usec));
    ReadBuffer(NumTemplates, SizeOf(NumTemplates));
    ReadBuffer(NBuffersSearched, SizeOf(NBuffersSearched));
    ReadBuffer(i, SizeOf(integer)); //read length of SourceBuffer flags
    if Length(SourceBuffers) < i then
      SetLength(SourceBuffers, i); //allocate memory
    ReadBuffer(SourceBuffers[0], i{1 byte/flag});
    Setlength(SpikeTemplates, NumTemplates);
    { read all the templates in this file }
    NTotalSpikes:= 0;
    TabControl.Tabs.Add('All');
    for i:= 0 to NumTemplates - 1 do
    with SpikeTemplates[i] do
    begin
      ReadBuffer(Sites, SizeOf(Sites));
      ReadBuffer(NumSites, SizeOf(NumSites));
      Sites:= Sites * [0..NumSites - 1]; //exclude any out-of-range sites from set
      ReadBuffer(PtsPerChan, SizeOf(PtsPerChan));
      SetLength(SiteOrigin, NumSites);
      ReadBuffer(SiteOrigin[0], NumSites * SizeOf(SiteOrigin[0]));
      ReadBuffer(PlotBounds, SizeOf(PlotBounds));
      ReadBuffer(MaxChan, SizeOf(MaxChan));
      ReadBuffer(n, SizeOf(n)); //'n' samples in avg
      ReadBuffer(Locked, SizeOf(Locked));
      ReadBuffer(Enabled, SizeOf(Enabled));
      ReadBuffer(FitThreshold, SizeOf(FitThreshold));
      Setlength(SumWaveform, NumSites * PtsPerChan);
      ReadBuffer(SumWaveform[0], Length(SumWaveform) * SizeOf(SumWaveform[0]));
      Setlength(SSqWaveform, Length(SumWaveform));
      ReadBuffer(SSqWaveform[0], Length(SSqWaveform) * SizeOf(SSqWaveform[0]));
      Setlength(Members, n);
      ReadBuffer(Members[0], n * SizeOf(Members[0]));
      Setlength(SpikeTimes, n); //read spike epoch times for all templates
      ReadBuffer(SpikeTimes[0], n * SizeOf(SpikeTimes[0]));
      { recompute stats for this template }
      Setlength(AvgWaveform, Length(SumWaveform));
      Setlength(StdWaveform, Length(SumWaveform));
      for s:= 0 to High(AvgWaveform) do
      begin
       AvgWaveform[s]:= Round(SumWaveform[s] / n);
       if n = 1 then StdWaveform[s]:= 0 //compute stdev of each point in the template
         else StdWaveform[s]:= sqrt((SSqWaveform[s] - (SumWaveForm[s] * SumWaveForm[s])/n) / (n - 1));
      end;
      TabControl.Tabs.Add(inttostr(i+1)); //add a tab for this template
      inc(NTotalSpikes, n); //keep tally of number of spike epochs comprising all templates
      SetPlotBoundaries(SpikeTemplates[i]); //temporary, saved values should be o.k.
    end{i};
    NumChans:= SpikeTemplates[0].NumSites;
    NumWavPts:= SpikeTemplates[0].PtsPerChan;
    //??? add initialisations to allow appending to existing templates HERE

    { initialise template window/settings }
    lbNumTemplates.Caption:= inttostr(NumTemplates);
    lbNumSamples.Caption:= inttostr(NTotalSpikes);
    lbSampleTime.Caption:= FormatFloat('0.0', NBuffersSearched / 10{buffspersecond});
    if seMaxClust.Value < NumTemplates then seMaxClust.Value:= NumTemplates;;
    cbViewOrder.ItemIndex:= 0; //order templates by max chan...
    cbViewOrder.OnChange(Self);
    //UpDown1.Enabled:= False;
    tbRawAvg.Caption:= 'Avg';
    tbRawAvg.Enabled:= False;
    Free;
  end{fs};
  Caption:= 'Templates (' + ExtractFileName(OpenFilename) + ')';
  tbBuildTemplates.Enabled:= False;
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.tbResetClick(Sender: TObject);
begin
  if NumTemplates > 0 then
    if MessageDlg('Delete all templates?', mtWarning,
      [mbYes, mbNo], 0) = mrYes then EraseAllTemplates;
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.udNoiseClick(Sender: TObject; Button: TUDBtnType);
begin
  Label1.Caption:= inttostr(udNoise.Position) + 'uV';
  ShrinkTemplate(TabTemplateOrder[TabControl.TabIndex - 2]);
  seFitThresh.Value:= SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]].FitThreshold;
  ChangeTab(TabControl.TabIndex - 2); //replot 'shrunken' template
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.tbShrinkAllClick(Sender: TObject);
const DefaultShrink = 95;
var t : integer;
begin
  udNoise.Position:= DefaultShrink;
  Label1.Caption:= inttostr(udNoise.Position) + 'uV';
  for t:= 0 to NumTemplates - 1 do
    ShrinkTemplate(t);
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.EraseAllTemplates;
var i : integer;
begin
  { clear templates, reinitialise templwin }
  while TabControl.Tabs.Count > 1 do
    TabControl.Tabs.Delete(1);
  SpikeTemplates:= nil;
  TabTemplateOrder:= nil;
  GlobalFitResults:= nil;
  ViewRawPlotsEnabled:= True;
  FitToFileEnabled:= True;
  //NumChans:= 0;
  //NumWavPts:= 0;
  NumTemplates:= 0;
  NTotalSpikes:= 0;
  NBuffersSearched:= 0;
  lbNumSamples.Caption:= '0';
  lbNumTemplates.Caption:= '0';
  lbSampleTime.Caption:= '0.0';
  Caption:= 'Templates';
  for i:= 0 to high(SourceBuffers) do
    SourceBuffers[i]:= NotSearched; //clear flags
  if m_PlotRaw then tbRawAvg.OnClick(Self);
  tbBuildTemplates.Enabled:= True;
  tbDel.Enabled:= True;
  tbExtractRaw.Enabled:= False;
  cbShowFits.Enabled:= False;
  cbShowFits.Checked:= False;
  UpDown1.Enabled:= True;
  tbRawAvg.Enabled:= True;
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.tbExtractRawClick(Sender: TObject);
begin
  if ViewRawPlotsEnabled then ReloadSpikeSet;
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.seFitThreshChange(Sender: TObject);
begin
  SpikeTemplates[TabTemplateOrder[TabControl.TabIndex - 2]].FitThreshold:= seFitThresh.Value;
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if Key = VK_CONTROL then ControlKeyDown:= True else
    if Key = VK_MENU then AltKeyDown:= True;
  TabSelectA:= TabControl.TabIndex;
end;

{------------------------------------------------------------------------------}
procedure TTemplateWin.FormKeyUp(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if Key = VK_CONTROL then ControlKeyDown:= False else
    if Key = VK_MENU then AltKeyDown:= False;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.WMEraseBkgnd(var msg: TWMEraseBkGnd);
begin
  msg.Result:=-1;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.TabControlMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  if (Button = mbLeft) and AltKeyDown
    and (TabControl.TabIndex > 1) then
    begin
      AltKeyDown:= False;
      SplitTemplate(TabTemplateOrder[TabControl.TabIndex - 2]);
    end;
end;

{-------------------------------------------------------------------------------------}
procedure TTemplateWin.cbGlobalEnableClick(Sender: TObject);
var i : integer;
begin
  for i:= 0 to NumTemplates - 1 do
    SpikeTemplates[i].Enabled:= cbGlobalEnable.Checked;
end;

{-------------------------------------------------------------------------------------}
destructor TTemplateWin.Destroy;
begin
  //FreeAndNil(WaveformBM);
  //dynamic arrays too?
  inherited Destroy;
end;

end.
