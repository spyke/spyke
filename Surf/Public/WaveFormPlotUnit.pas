{ (c) 1994-1999 Phil Hetherington, P&M Research Technologies, Inc.}
{ (c) 2000-2003 Tim Blanche, University of British Columbia }
unit WaveFormPlotUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ComCtrls, ExtCtrls, SurfPublicTypes, ElectrodeTypes, Spin, PolytrodeGUI,
  Buttons, ToolWin, ImgList, DTxPascal{!this is only here for LPSHRT declaration!}, Menus;

const XSCALE = 3; //#pixels per waveform sample
      DEFAULTYZOOM = 3; //100%
      V2uV = 1000000;
type
  TWaveFormPlotForm = class(TForm)
    tbImages: TImageList;
    tbControl: TToolBar;
    tbOverlay: TToolButton;
    tbTrigger: TToolButton;
    tbZeroLine: TToolButton;
    spacer: TToolButton;
    seThreshold: TSpinEdit;
    CBZoom: TComboBox;
    Menu: TPopupMenu;
    muFreeze: TMenuItem;
    muProperties: TMenuItem;
    N1: TMenuItem;
    muOverlay: TMenuItem;
    muContDisp: TMenuItem;
    N2: TMenuItem;
    muPolytrodeGUI: TMenuItem;
    muBipolarTrig: TMenuItem;
    muAutoMUX: TMenuItem;
    procedure FormPaint(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormHide(Sender: TObject);
    procedure FormMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure FormMouseDown(Sender: TObject; Button: TMouseButton;
                            Shift: TShiftState; X, Y: Integer); virtual;
    procedure FormMouseUp(Sender: TObject; Button: TMouseButton;
                          Shift: TShiftState; X, Y: Integer);
    procedure FormDblClick(Sender: TObject);
    procedure seThresholdChange(Sender: TObject);
    procedure CBZoomChange(Sender: TObject);
    procedure muOverlayClick(Sender: TObject);
    procedure muContDispClick(Sender: TObject);
    procedure muPolytrodeGUIClick(Sender: TObject);
    procedure ReduceFlicker(var message:TWMEraseBkgnd); message WM_ERASEBKGND;
    procedure tbOverlayClick(Sender: TObject);
    procedure muBipolarTrigClick(Sender: TObject);
    procedure muAutoMUXClick(Sender: TObject);
  private
    { Private declarations }
    WaveformBM : TBitmap;

    Acquisition, DispIsFrozen, LeftButtonDown : boolean;

    NumWavPts, NumSites, Threshold, TraceIndex, LastOldBufferSample, //add decimn factor as global var?
      ProbeId, WaveFormHeight, ScaledWaveFormHeight, OverThreshold : integer;
    ProbeType : char;
    ProbeName : string;
    SiteOrigin : array of TPoint;
    ScreenY : array[0..RESOLUTION_12_BIT] of integer;
    Pts2Plot : integer;
    FS, uVTrig : integer;
    AD2uV, TimeBase : single;
    TimeBaseLbl, FSlbl : string[2]; //'ms' or 's ', 'uV' or 'mV'

    procedure PlotThresholdLines(Erase : boolean = False);
    procedure PlotTriggerLines;
    procedure PlotZeroLines;
    procedure PlotProbeInfo;
    procedure AD2ScreenY;
    //procedure BlankPlotWin;
    //procedure AddRemovePolytrodeGUI; //overridden by method in SurfContAcq
  public
    { Public declarations }
    GUIForm: TPolytrodeGUIForm;
    DrawWaveForms, GUICreated : boolean;
    PolarityLbl : char;
    TriggerPt : integer;

    procedure InitPlotWin(const Electrode : TElectrode; npts,winleft,wintop,thresh,trigpt,pid : integer;
                          prbtype : char; title : shortstring; acquisitionmode : boolean;
                          const intgain : integer = 8; const extgain : integer = 5000;
                          const sampfreq : integer = 25000; const DispDecFactor : integer = 1);
    procedure PlotSpike(const Spike : TSpike);
    procedure PlotWaveform({const}PDeMUXedBuffer : LPUSHRT; const SampPerChanPerBuff : integer;
                           const DispDecFactor : integer = 1{no disp decimn}; Cluster : SHRT = 2{clLime}); overload;
    procedure PlotWaveform(const WaveForm: array of SmallInt; Cluster : SHRT); overload;{<---- TEMP!}
    {procedure PlotDTBufferWaveforms(BufferPtr: LPSHRT;
                              const FirstChan2Plot, NumChans2Plot, ChansPerBuff : integer);}
    procedure ThreshChange(pid, Threshold : integer); virtual; abstract;
    procedure NotAcqOnMouseMove(ChanNum : byte); virtual; abstract;
    procedure ClickProbeChan(pid, ChanNum : byte); virtual; abstract;
    procedure CreatePolytrodeGUI(pid : integer);virtual; abstract;
    procedure FreezeThawDisplay;
  end;

implementation

{$R *.DFM}
{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.InitPlotWin(const Electrode : TElectrode; npts,winleft,wintop,thresh,trigpt,pid : integer;
            prbtype : char; title : shortstring;{ view,}acquisitionmode : boolean;
            const intgain{8} : integer; const extgain{5000} : integer;
            const sampfreq{25000} : integer; const DispDecFactor{1} : integer);
var i, j, leftmost, rightmost, topmost, bottommost, UnscaledWaveFormHeight : integer;
  XOriginScale, YOriginScale : Single;
begin
  {copy procedure parameters into this probe windows' global variables}
  {precompute commonly used indicies and labels}
  Left:= winleft;
  Top:= wintop;
  ProbeId:= pid; //must preceed seThreshold.Value or erroneously overwrites Probe[0]'s value!
  NumWavPts:= npts;
  Pts2Plot:= NumWavPts div DispDecFactor;
  TimeBase:= NumWavPts / SampFreq * 1000 * DispDecFactor; {in ms}
  if prbtype = CONTINUOUS then TimeBase:= TimeBase * XSCALE; //?!
  if TimeBase >= 100 then
  begin
    TimeBase:= TimeBase / 1000; {in sec}
    TimeBaseLbl:= 's ';
  end else
    TimeBaseLbl:= 'ms';
  TriggerPt:= trigpt;
  Threshold:= thresh; //signed value from -2048 to 2047
  AD2uV:= (20 / intgain / RESOLUTION_12_BIT) / extgain * V2uV; //assumes same gain for all channels within a probe
  uVTrig:= Abs(Round(Threshold * AD2uV));
  FS:= Round(2048 * AD2uV);
  if FS >= 1000 then
  begin
    FS:= FS div 1000; //convert to mV
    FSlbl:= 'mV';
  end else
    FSlbl:= 'µV';
  seThreshold.Value:= threshold;
  ProbeType:= prbtype;
  Acquisition:= acquisitionmode;
  Caption:= title;
  NumSites:= Electrode.NumSites;
  ProbeName:= Electrode.Name;

  {get boundaries of electrode site locations}
  leftmost:= 1000;
  rightmost:= -1000;
  topmost:= 1000;
  bottommost:= -1000;
  for i := 0 to NumSites-1 do
  begin
    if leftmost > Electrode.SiteLoc[i].x
      then leftmost := Electrode.SiteLoc[i].x;
    if rightmost < Electrode.SiteLoc[i].x
      then rightmost := Electrode.SiteLoc[i].x;
    if topmost > Electrode.SiteLoc[i].y
      then topmost := Electrode.SiteLoc[i].y;
    if bottommost < Electrode.SiteLoc[i].y
      then bottommost := Electrode.SiteLoc[i].y;
  end;

  { finds the closest of any two electrode sites along the same vertical and
    horizontal axis, and scales waveform height and/or minimum horizontal site
    spacing to fit all sites on screen (vertically), and avoid waveform overlaps (horizontally) }
  WaveFormHeight:= 1000;
  XOriginScale:= 1.0; //site x-spacing will remain 1 pixel/micron, unless there is horizontal waveform overlap
  for i := 0 to NumSites - 2 do
    for j := i+1 to NumSites - 1 do
    begin
       if Electrode.SiteLoc[j].x = Electrode.SiteLoc[i].x then {same column}
         if WaveFormHeight > abs(Electrode.SiteLoc[j].y - Electrode.SiteLoc[i].y) then
           WaveFormHeight := abs(Electrode.SiteLoc[j].y - Electrode.SiteLoc[i].y);
       if Electrode.SiteLoc[j].y = Electrode.SiteLoc[i].y then {same row}
         if XSCALE > abs(Electrode.SiteLoc[j].x - Electrode.SiteLoc[i].x) / {NumWavPts}Pts2Plot then
           XOriginScale := Abs(Electrode.SiteLoc[j].x - Electrode.SiteLoc[i].x) / {NumWavPts}Pts2Plot + 0.2;
    end;
  if WaveFormHeight = 1000 then WaveFormHeight := 50;
  YOriginScale:= 1.0; //waveform height will equal site y-spacing, unless all sites cannot fit on form vertically
  if bottommost - topmost + (WaveFormHeight *  2) > Screen.Height - 150 then
  begin
    UnscaledWaveFormHeight:= WaveFormHeight;
    WaveFormHeight := Round((Screen.Height - 150) / (WaveFormHeight * 2 + bottommost - topmost) * WaveFormHeight);
    YOriginScale:= WaveFormHeight / UnscaledWaveFormHeight;
  end;

  ScaledWaveformHeight:= WaveFormHeight;
  AD2ScreenY; //generate LUT to speed waveform plotting

  SetLength(SiteOrigin, NumSites); //siteorigin[n] is the scaled representation of SiteLoc[n]
  for i := 0 to NumSites - 1 do    //...and is used for all plotting in WaveFormPlotUnit
  begin
    SiteOrigin[i].x:= Round((Electrode.SiteLoc[i].x + RightMost)* XOriginScale) + 15;
    SiteOrigin[i].y:= Round((Electrode.SiteLoc[i].y - TopMost + WaveFormHeight) * YOriginScale);
  end;

  { modify toolbar and pop-up menu depending on probetype or on/offline }
  if ProbeType = CONTINUOUS then
  begin
    FreeAndNil(tbTrigger);
    FreeAndNil(seThreshold);
    FreeAndNil(muContDisp);
    FreeAndNil(muPolytrodeGUI);
    FreeAndNil(muBipolarTrig);
    FreeAndNil(muAutoMUX);
  end;
  if not Acquisition then
  begin
    FreeAndNil(muContDisp);
    FreeAndNil(muPolytrodeGUI);
    FreeAndNil(muAutoMUX);
  end;

  { implicitly sets form width and height, as autosized to toolbar width and height }
  Constraints.MinWidth:= (rightmost - leftmost) + Round((NumWavPts{Pts2Plot} * XSCALE + 30)* XOriginScale);
  if tbControl.Width < ClientWidth then tbControl.Align:= alTop; //extend toolbar width, if necessary
  if (tbControl.Height + (bottommost - topmost) + WaveFormHeight*2) > Screen.Height - 120
    then Constraints.MinHeight:= Screen.Height - 120
    else Constraints.MinHeight:= tbControl.Height + bottommost - topmost + Round(WaveFormHeight * 2.5);

  { generate bitmap for plotting waveforms offscreen }
  WaveformBM:= TBitmap.Create;
  {WaveformBM.Width:= Canvas.ClipRect.Right;
  WaveformBM.Height:= Canvas.ClipRect.Bottom - Canvas.ClipRect.Top;}
  with WaveformBM do
  begin
    PixelFormat:= pf32bit; //format, DIB? vs speed????
    {HandleType:= bmDIB;}
    Width:= 2000;//ClientRect.Right;
    Height:= ClientHeight - tbControl.Height;
    Canvas.Brush.Color:= $00000000; //black
    Canvas.FillRect(Canvas.ClipRect); //erase background
    //Canvas.Pen.Mode:= pmMerge; //ensures trigger/zero line don't obscure waveforms
    Canvas.Font.Name:= 'Small Fonts';
    Canvas.TextFlags:= ETO_OPAQUE;
    Canvas.Font.Color:= clYellow;
    Canvas.Font.Size:= 6;
  end;

  {if ProbeType <> CONTINUOUS then
  begin
    if tbTrigger.Down then PlotTriggerLines;
    if seThreshold.Enabled then PlotThresholdLines;
  end;}
  if tbZeroLine.Down then PlotZeroLines;
  PlotProbeInfo;

  LeftButtonDown:= False;
  OverThreshold:= -1;
  CBZoom.ItemIndex:= DEFAULTYZOOM; //100%
  Show;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotSpike(const Spike : TSpike);
var i,w,y : integer;
begin
  if not DrawWaveForms then Exit;
  if tbOverlay.Down = False then Paint;
  Canvas.Pen.Color := clWhite;//COLORTABLE[Spike.Cluster];
  for i := 0 to NumSites -1 do
  begin
    y := ScreenY[Spike.Waveform[i,0]];
    Canvas.MoveTo(SiteOrigin[i].x, SiteOrigin[i].y - y);
    for w := 1 to {NumWavPts}Pts2Plot -1 do //potentially replace multiple lineto's with polyline?
    begin
      y := ScreenY[Spike.Waveform[i,w]];
      Canvas.LineTo(SiteOrigin[i].x + w * XSCALE, SiteOrigin[i].y - y);
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotWaveform(PDeMUXedBuffer : LPUSHRT; const SampPerChanPerBuff : integer;
                                         const DispDecFactor : integer; Cluster : SHRT);
var c, w, y : integer; {not yet optimised... amenable to asm scanline}
  displayptrbak : LPUSHRT;
begin
  if not DrawWaveForms then
  begin
    if ProbeType = CONTINUOUS then
    begin
      TraceIndex:= (TraceIndex + Pts2Plot) mod (NumWavPts * XSCALE);
      inc(PDeMUXedBuffer, NumWavPts - 1);
      LastOldBufferSample:= ScreenY[PDeMUXedBuffer^];
    end;
    Exit;
  end;

  with WaveformBM.Canvas do
  begin
    if ProbeType = CONTINUOUS then
    begin //plot waveform for Continuous (EEG) probe...
      if tbOverlay.Down = False then
      begin //this is a weird patch to remove residual pixels at start of trace...
        if TraceIndex = 0 then FillRect(Rect(SiteOrigin[0].x + TraceIndex, 0,
                    {erase vertical strip} SiteOrigin[0].x + TraceIndex + 11, Height))
                 else FillRect(Rect(SiteOrigin[0].x + 1 + TraceIndex, 0,
                    {erase vertical strip} SiteOrigin[0].x + TraceIndex + 11, Height));
      end;
      Pen.Mode:= pmCopy;
      Pen.Color:= clWhite;//COLORTABLE[Cluster];
      y:= LastOldBufferSample;
      MoveTo(SiteOrigin[0].x + TraceIndex, SiteOrigin[0].y - y);
      for w:= 1 to {NumWavPts}Pts2Plot do //replace move/lineto's with scanline methods!
      begin
        y:= ScreenY[PDeMUXedBuffer^];
        LineTo(SiteOrigin[0].x + w + TraceIndex, SiteOrigin[0].y - y);
        inc(PDeMUXedBuffer, Pts2Plot);
      end;
      LastOldBufferSample:= y;
      TraceIndex:= (TraceIndex + Pts2Plot) mod (NumWavPts * XSCALE);
    end else
    begin //plot for SpikeStream or SpikeEpoch probe...
      if tbOverlay.Down = False then FillRect(ClipRect); //erase background
      if tbTrigger.Down then PlotTriggerLines;
      if seThreshold.Enabled then PlotThresholdLines;
      Pen.Mode:= pmCopy;
      Pen.Color:= clLime;//COLORTABLE[Cluster];
      displayptrbak:= PDeMUXedBuffer;
      for c:= 0 to NumSites -1 do
      begin
        PDeMUXedBuffer:= displayptrbak;
        inc(PDeMUXedBuffer, c * SampPerChanPerBuff);
        y:= ScreenY[PDeMUXedBuffer^];
        MoveTo(SiteOrigin[c].x, SiteOrigin[c].y - y);
        for w:= 1 to {NumWavPts}Pts2Plot -1 do //replace move/lineto's with scanline methods!
        begin
          inc(PDeMUXedBuffer, DispDecFactor);
          y:= ScreenY[PDeMUXedBuffer^];
          LineTo(SiteOrigin[c].x + w * XSCALE, SiteOrigin[c].y - y);
        end;
      end{s};
    end;
    if tbZeroLine.Down then PlotZeroLines;
    PlotProbeInfo;
  end;
  Paint; //blit bm to ProbeWin's canvas
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotWaveform(const WaveForm: array of SmallInt; Cluster : SHRT);
var i,w,y : integer;
begin
  if not DrawWaveForms then Exit;
  //if tbOverlay.Down = False then Paint;
  Canvas.Pen.Color := clWhite;//COLORTABLE[Cluster];
  for i := 0 to NumSites -1 do
  begin
    y := ScreenY[WaveForm[i{*NumWavPts}]];
    Canvas.MoveTo(SiteOrigin[i].x, SiteOrigin[i].y - y);
    for w := 1 to NumWavPts - 1 do //potentially replace multiple lineto's with polyline?
    begin
      y := ScreenY[WaveForm[w * NumSites + i]];
      Canvas.LineTo(SiteOrigin[i].x + w * XSCALE, SiteOrigin[i].y - y);
    end;
  end;
end;

{-------------------------------------------------------------------------}
{procedure TWaveFormPlotForm.PlotDTBufferWaveforms(BufferPtr: LPSHRT;
                const FirstChan2Plot, NumChans2Plot, ChansPerBuff : integer);

var i, w, y : integer;
begin
  if not DrawWaveForms then Exit;
  if tbOverlay.Down = False then      //only clear canvas if not in overlay mode...
    if FirstChan2Plot = 0 then Paint; //...nor mid-probe display
  Canvas.Pen.Color := clWhite;
  for i := 0 to NumChans2Plot - 1 do
  begin
    //inc(BufferPtr, i);
    y := ScreenY[BufferPtr^];
    Canvas.MoveTo(SiteOrigin[i+FirstChan2Plot].x, SiteOrigin[i+FirstChan2Plot].y - y);
    for w := 1 to NumWavPts - 1 do //potentially replace multiple lineto's with polyline or scanline (see EFG)?
    begin
      inc(BufferPtr, ChansPerBuff);
      y := ScreenY[BufferPtr^];
      Canvas.LineTo(SiteOrigin[i+FirstChan2Plot].x + w * XSCALE, SiteOrigin[i+FirstChan2Plot].y - y);
    end;
  end;
end;
}
{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormPaint(Sender: TObject);
begin
  Canvas.Draw(0, tbControl.Height, WaveformBM);
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotThresholdLines(Erase : boolean);
var i,y : integer;
begin
  with WaveformBM do
  begin
    Canvas.Pen.Color:= clDkGray;
    if Erase then Canvas.Pen.Mode:= pmMaskNotPen //erase old threshold lines...
      else Canvas.Pen.Mode:= pmMerge; //...or draw new threshold lines
    y:= ScreenY[Threshold + 2048];
    for i:= 0 to NumSites - 1 do
    begin
      Canvas.MoveTo(SiteOrigin[i].x + TriggerPt * XSCALE - 10, SiteOrigin[i].y - y);
      Canvas.LineTo(SiteOrigin[i].x + TriggerPt * XSCALE + 10, SiteOrigin[i].y - y);
      if muBipolarTrig.Checked then //plot bipolar threshold lines
      begin
        Canvas.MoveTo(SiteOrigin[i].x + TriggerPt * XSCALE - 10, SiteOrigin[i].y + y);
        Canvas.LineTo(SiteOrigin[i].x + TriggerPt * XSCALE + 10, SiteOrigin[i].y + y);
      end;
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotProbeInfo;
var TextLeft : integer;
begin
  with WaveformBM.Canvas do
  begin
    TextLeft:= ClientWidth - 53;
    TextOut(TextLeft, 5, 'FS' + chr($B1){+/-} + inttostr(FS) + FSlbl);
    TextOut(TextLeft, 13, 'epoch: ' + floattostrF(TimeBase, ffFixed, 3, 1) + TimeBaseLbl);
    if ProbeType = CONTINUOUS then TextOut(TextLeft, 21, 'Chart mode') else
      if seThreshold.Enabled then TextOut(TextLeft, 21,'trig: ' + PolarityLbl + inttostr(uVTrig) + 'µV     ')
        else TextOut(TextLeft, 21, 'continuous');
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotTriggerLines;
var i : integer;
begin
  with WaveformBM do
  begin
    Canvas.Pen.Mode:= pmMerge;
    Canvas.Pen.Color:= clDkGray;
    for i:= 0 to NumSites -1 do
    begin
      Canvas.MoveTo(SiteOrigin[i].x + TriggerPt*XSCALE, SiteOrigin[i].y + ScaledWaveFormHeight);
      Canvas.LineTo(SiteOrigin[i].x + TriggerPt*XSCALE, SiteOrigin[i].y - ScaledWaveFormHeight);
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.PlotZeroLines;
var i : integer;
begin
  with WaveformBM do
  begin
    Canvas.Pen.Mode:= pmMerge;
    Canvas.Pen.Color:= clDkGray;
    for i := 0 to NumSites -1 do
    begin
      Canvas.MoveTo(SiteOrigin[i].x,SiteOrigin[i].y);
      Canvas.LineTo(SiteOrigin[i].x + NumWavPts{Pts2Plot}*XSCALE, SiteOrigin[i].y);
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormCreate(Sender: TObject);
begin
  NumSites:= 0;
  DrawWaveForms:= True;
  {the following should help reduce flicker}
  ControlStyle := ControlStyle + [csOpaque];
  DoubleBuffered:= False; //not needed
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormHide(Sender: TObject);
begin
  SiteOrigin:= nil;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormMouseMove(Sender: TObject; Shift: TShiftState;
                                            X, Y: Integer);
var i : integer;
begin
  if (ProbeType = CONTINUOUS) or (seThreshold.Enabled = False) then Exit;
  dec(Y, tbControl.Height); //align bitmap Y-coords with form coords
  if not LeftButtonDown then
  begin
    Screen.Cursor:= crDefault;
    for i:= 0 to NumSites -1 do
    begin
      if  (X > SiteOrigin[i].x + TriggerPt * XSCALE - 10)
      and (X < SiteOrigin[i].x + TriggerPt * XSCALE + 10)
      and (Abs(Y - (SiteOrigin[i].y-Screeny[Threshold + 2048])) < 2) then
      begin
        if not Acquisition then NotAcqOnMouseMove(i);
        Screen.Cursor:= crVSplit;
        OverThreshold:= i;
        Break;
      end;
    end;
  end else //left button down...
  begin
    if Screen.Cursor = crDefault then Exit;
    PlotThresholdLines(True); //erase old threshold lines
    Threshold:= Round((SiteOrigin[OverThreshold].y-Y) / ScaledWaveFormHeight * 2047);
    if Threshold > 2047 then Threshold:= 2047 else
      if Threshold < -2048 then Threshold:= -2048;
    seThreshold.Value:= Threshold;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormMouseDown(Sender: TObject;
                            Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var i : integer;
begin
  if Button = mbLeft then
  begin
    LeftButtonDown:= True;
    if Acquisition and (Screen.Cursor = crDefault) then //if acquiring select MUX and/or EEG channel
    begin
      dec(Y, tbControl.Height); //align bitmap Y-coords with form coords
      for i:= 0 to NumSites -1 do //get channel wrt mouse position
      begin
        if  (X > SiteOrigin[i].x - 2)
        and (X < SiteOrigin[i].x + {NumWavPts}Pts2Plot * XSCALE)
        and (Abs(Y - SiteOrigin[i].y) < (WaveFormHeight div 2)) then
        begin
          ClickProbeChan(ProbeID, i);
          Break;
        end;
      end;
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormMouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  LeftButtonDown:= False;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.AD2ScreenY;
var i : integer;
begin      //4097 entries?
  for i:= 0 to RESOLUTION_12_BIT do //this LUT optimises waveform plotting
    ScreenY[i] := Round((i-2047)/2047 * ScaledWaveformHeight);
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.seThresholdChange(Sender: TObject);
var tmp : integer;
begin
  tmp := Threshold;
  try
    PlotThresholdLines(True); //erase old lines
    Threshold:= seThreshold.Value;
    ThreshChange(ProbeId, Threshold);
    PlotThresholdLines;
    if tbTrigger.Down then PlotTriggerLines;
    if tbZeroLine.Down then PlotZeroLines;
    PlotProbeInfo;
    Paint;
  except
    Threshold:= tmp;
  end;
  uVTrig:= Abs(Round(Threshold * AD2uV));
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.tbOverlayClick(Sender: TObject);
begin
  muOverlay.Checked:= tbOverlay.Down;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FormDblClick(Sender: TObject);
begin
  FreezeThawDisplay;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.muOverlayClick(Sender: TObject);
begin
  tbOverlay.Down:= not tbOverlay.Down;
  muOverlay.Checked:= tbOverlay.Down;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.muContDispClick(Sender: TObject);
begin
  seThreshold.Enabled:= muContDisp.Checked;
  muContDisp.Checked:= not muContDisp.Checked;
  if seThreshold.Enabled then
  begin
    muBipolarTrig.Enabled:= True;
    seThresholdChange(Self);
  end else
  begin
    Threshold:= 0;
    ThreshChange(ProbeId, Threshold);
    muBipolarTrig.Enabled:= False;
  end;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.muBipolarTrigClick(Sender: TObject);
begin
  PlotThresholdLines(True); //erase old lines
  muBipolarTrig.Checked:= not (muBipolarTrig.Checked);
  seThresholdChange(Self);
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.FreezeThawDisplay;
begin
  DrawWaveForms:= DispIsFrozen;
  DispIsFrozen:= not DispIsFrozen; //toggle
  muFreeze.Checked:= DispIsFrozen;
  if DispIsFrozen then WaveformBM.Canvas.Font.Color:= clYellow
    else WaveformBM.Canvas.Font.Color:= clBlack; //erase text
  WaveformBM.Canvas.TextOut(0, 0, 'Display Frozen');
  WaveformBM.Canvas.Font.Color:= clYellow; //restore
  Paint;
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.CBZoomChange(Sender: TObject);
begin
  if (ProbeType <> CONTINUOUS) and seThreshold.Enabled then
    PlotThresholdLines(True); //erase old threshold lines
  with CBZoom do
  begin
    Tag:= ScaledWaveFormHeight; //store previous zoom setting
    ScaledWaveformHeight:= Round(WaveFormHeight * StrToInt(Copy(Text, 0,
                                 Pos('%', Items[ItemIndex])-1)) / 100);
    if ScaledWaveformHeight = Tag then Exit;
  end;
  AD2ScreenY; //rescale waveform plotting LUT
  if (ProbeType <> CONTINUOUS) then
  begin
    if seThreshold.Enabled then PlotThresholdLines;
    PlotTriggerLines; //fix gap in trigger line
  end;
  Paint;
end;

{-------------------------------------------------------------------------}
(*procedure TWaveFormPlotForm.BlankPlotWin;
begin
  Canvas.Font.color := clYellow;
  if DrawWaveForms then
  begin
    Canvas.Brush.Color := clBlack;
    Canvas.FillRect(Canvas.ClipRect);
    Paint;
  end else
  begin
    Canvas.Brush.Color := clNavy;
    Canvas.FillRect(Canvas.ClipRect);
    Canvas.TextOut(8, tbControl.Height + 5, 'Display Off');
  end;
  Update;
end; *)

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.muPolytrodeGUIClick(Sender: TObject);
begin
  muPolytrodeGUI.Checked:= not(muPolytrodeGUI.Checked);
  CreatePolytrodeGUI(ProbeID); //abstract method in SurfContAcq class
end;

{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.muAutoMUXClick(Sender: TObject);
begin
  muAutoMUX.Checked:= not(muAutoMUX.Checked);
end;

{-------------------------------------------------------------------------}
(*procedure TWaveFormPlotForm.AddRemovePolytrodeGUI;
begin
  if NumSites < 2 then muPolytrodeGUI.Checked:= False
    else muPolytrodeGUI.Checked:= not muPolytrodeGUI.Checked;

  if muPolytrodeGUI.Checked = False then
  begin
    if GUICreated then GUIForm.Close;
    GUICreated:= False;
    Exit;
  end else
  try
    if GUICreated then GUIForm.Close; //remove any existing GUIForms...
    GUIForm:= TPolytrodeGUIForm.Create(Self);//..and create a new one
    with GUIForm do
    begin
      Left:= Left;
      Top:= Top;
      Show;
      BringToFront;
      GUICreated:= True;
    end;
  except
    GUICreated:= False;
    Exit;
  end;

  if not GUIForm.CreateElectrode(ProbeName, True) then
  begin
    GUIForm.Free;
    GUICreated:= False;
    Exit;
  end;
  GUIForm.Caption:= ProbeName;
end;
*)
{-------------------------------------------------------------------------}
procedure TWaveFormPlotForm.ReduceFlicker(var message:TWMEraseBkgnd);
begin
  message.result:= LRESULT(False); //stops WM backgnd repaints -- reduce flicker
end;

end.
