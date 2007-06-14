unit WaveFormUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls, ComCtrls, DTxPascal, SurfPublicTypes;

type
  TWaveFormWin = class(TForm)
    plot: TImage;
    SlideBar: TTrackBar;
    Threshold: TPanel;
    HiVolt: TLabel;
    LoVolt: TLabel;
    MarkerH: TShape;
    MarkerV: TShape;
    procedure FormClick(Sender: TObject);
    procedure SlideBarChange(Sender: TObject);
    procedure SlideBarKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure SlideBarKeyUp(Sender: TObject; var Key: Word;
      Shift: TShiftState);
    procedure FormDestroy(Sender: TObject);
    procedure plotDblClick(Sender: TObject);
  private
    { Private declarations }
    BlankWaveform : TWaveForm;
    backbm : TBitmap;
    ShiftDown,CtrlDown : boolean;
    //VLine,HLIne : Integer;
    RGBClusterTable : array[-2..MAXCLUSTERS] of TRGBTriple;
    procedure BlankPlotWin;
  public
    { Public declarations }
    BlankBuf : HBUFTYPE;
    ProbeId,WinId,xlength,ymax,ymin : integer;
    MarkerHOn,MarkerVOn : boolean;
    Screeny : array[0..4095] of integer;
    DrawWaveForms : boolean;
    Procedure InitPlotWin(npts,winleft,wintop,bmheight,intgain,thresh,trigpt,pid,wid : integer;
                                   probetype : char; title : shortstring; view,acquisitionmode : boolean);
    Procedure PlotWaveform(var wvfrm: TWaveForm; Cluster : integer; Overlay : boolean);

    Procedure ThreshChange(pid,winid : integer; ShiftDown,CtrlDown : boolean); virtual; abstract;
  end;

implementation

{$R *.DFM}

{====================================================================}
procedure TWaveFormWin.FormClick(Sender: TObject);
begin
  ShowMessage(inttostr(height)+','+inttostr(Width));
end;

{====================================================================}
procedure TWaveFormWin.SlideBarChange(Sender: TObject);
begin
  ThreshChange(ProbeId,WinId,ShiftDown,CtrlDown);
end;

{====================================================================}
procedure TWaveFormWin.SlideBarKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
   if ssShift in Shift
     then ShiftDown := TRUE;
   if ssCtrl in Shift
     then CtrlDown := TRUE;
end;

{====================================================================}
procedure TWaveFormWin.SlideBarKeyUp(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  ShiftDown := FALSE;
  CtrlDown := FALSE;
end;

{====================================================================}
procedure TWaveFormWin.FormDestroy(Sender: TObject);
begin
  //bm.Free;
  backbm.Free;
  if BlankBuf <> NULL then olDmFreeBuffer(BlankBuf);
end;

{====================================================================}
Procedure TWaveFormWin.InitPlotWin(npts,winleft,wintop,bmheight,intgain,thresh,trigpt,pid,wid : integer;
                                   probetype : char; title : shortstring; view,acquisitionmode : boolean);
var  i : integer;
     yscale : single;
begin
  ProbeId := pid;
  WinId := wid;
  SetLength(BlankWaveform,npts*2);
  For i := 0 to npts-1 do BlankWaveform[i] := 0;
  Left := winleft;
  top := wintop;

  backbm := TBitmap.Create;
  backbm.height := bmheight;
  backbm.width := npts*2;
  backbm.Canvas.Pen.Color := clLtGray;
  backbm.Canvas.Brush.Color := clBlack;
  backbm.Canvas.FillRect(backbm.Canvas.ClipRect);
  backbm.canvas.CopyMode := cmSrcCopy;
  backbm.PixelFormat := pf24bit;

  Plot.Picture.Bitmap.Assign(backbm);
  Plot.Picture.Bitmap.Height := bmheight;

  yscale := bmheight/4096;
  for i := 0 to 4095 do
  begin
    Screeny[i] := bmheight - trunc(i * yscale)-1;
    //if Screeny[i] > bmheight-1 then Screeny[i] := bmheight-1;
    //if Screeny[i] < 0 then Screeny[i] := 0;
  end;

  For i := -2 to MAXCLUSTERS do
  begin
    if i<0 then
    begin
      RGBClusterTable[i].rgbtred := 255;
      RGBClusterTable[i].rgbtblue := 0;
      RGBClusterTable[i].rgbtgreen := 0;
    end else
    begin
      RGBClusterTable[i].rgbtred := GetRValue(COLORTABLE[i]);
      RGBClusterTable[i].rgbtblue := GetBValue(COLORTABLE[i]);
      RGBClusterTable[i].rgbtgreen := GetGValue(COLORTABLE[i]);
    end;
  end;

  caption := title;
  HiVolt.Caption := '+'+floattostrf(10/intgain,fffixed,3,2)+' V';
  LoVolt.Caption := '-'+floattostrf(10/intgain,fffixed,3,2)+' V';
  if acquisitionmode then
  begin
    case probetype of
      CONTINUOUSTYPE : begin
                     MarkerH.Hide;
                     MarkerV.Hide;
                     SlideBar.Hide;
                     Threshold.Caption := '';
                   end;
      SPIKEEPOCH :  begin
                     MarkerH.Show;
                     MarkerV.Show;
                     SlideBar.Show;
                     Threshold.Caption := inttostr(thresh);
                   end;
    end {case};
  end else
  begin
    Threshold.Caption := '';
    SlideBar.Hide;
  end;

  ClientWidth := backbm.width + plot.left;
  ClientHeight := backbm.height;

  MarkerH.Top := round((2047-thresh) * yscale);
  SlideBar.Position := 2047-thresh;
  MarkerH.Left := plot.left;
  MarkerH.Width := plot.width;
  MarkerV.Left := plot.left + trigpt*2;
  MarkerV.Height := plot.height;
  DrawWaveForms := View;
  BlankPlotWin;
  Show;
end;

{====================================================================}
Procedure TWaveFormWin.PlotWaveform(var wvfrm : TWaveForm; cluster : integer; Overlay{,FastDraw} : boolean);
var npts,p,v,lastv : integer;
    pb : PByteArray;
    rgbt : TRGBTriple;
    bm : TBitmap;
begin
  if not DrawWaveForms then exit;
  npts := length(wvfrm);
  bm := plot.picture.bitmap;
  if not overlay then bm.Canvas.Draw(0,0,backbm);

  rgbt := RGBClusterTable[cluster];

  v := Screeny[wvfrm[0]];
  lastv := v;
  pb := bm.scanline[v];
  Move(rgbt,pb[0],3);
  for p := 1 to npts-1 do
  begin
    v := Screeny[wvfrm[p]];
    pb := bm.scanline[v];
    Move(rgbt,pb[p*3*2],3);
    pb := bm.scanline[(v+lastv) shr 1{div 2}];
    Move(rgbt,pb[p*3*2-3],3);
    lastv := v;
  end;
  if overlay then plot.refresh;
end;

{====================================================================}
procedure TWaveFormWin.BlankPlotWin;
begin
  Plot.Picture.Bitmap.canvas.Font.Name := 'Arial';
  Plot.Picture.Bitmap.canvas.Font.Size := 8;
  Plot.Picture.Bitmap.canvas.Font.color := clYellow;
  if DrawWaveForms then
  begin
    Plot.Picture.Bitmap.Canvas.Brush.Color := clBlack;
    Plot.Picture.Bitmap.Canvas.FillRect(Plot.Picture.Bitmap.canvas.ClipRect);
    Plot.Picture.Bitmap.Canvas.TextOut(0,0,'Ready');
  end else
  begin
    Plot.Picture.Bitmap.Canvas.Brush.Color := clNavy;
    Plot.Picture.Bitmap.Canvas.FillRect(Plot.Picture.Bitmap.canvas.ClipRect);
    Plot.Picture.Bitmap.Canvas.TextOut(0,0,'Display Off');
  end;
  plot.update;
end;

procedure TWaveFormWin.plotDblClick(Sender: TObject);
begin
  DrawWaveForms := not DrawWaveForms;
  BlankPlotWin;
end;

end.
