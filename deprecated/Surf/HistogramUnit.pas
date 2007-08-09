unit HistogramUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs, Math,
  StdCtrls, SurfMathLibrary;

type
  THistogramWin = class(TForm)
    Button1: TButton;
    procedure FormCreate(Sender: TObject);
    procedure FormPaint(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure FormMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure FormMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure FormMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure Button1Click(Sender: TObject);
  private
    GraphBM : TBitmap;
    LastXPos : integer;
    procedure InitialiseGraphBM;
    { Private declarations }
  public
    LeftButtonDown : boolean;
    NumBins  : integer;
    Min, Max : cardinal;
    BinCount : array of integer;
    procedure FillHistogram(const DataArray : array of cardinal; nbins : integer);
    procedure PlotHistogram;
    procedure MoveGUIMarker(MouseX : Single); virtual; abstract;
    procedure UpdateISIHistogram; virtual; abstract;
    procedure PlotGUIMarker(XPos : integer);
    procedure Reset;
    destructor Destroy; override;
    { Public declarations }
  end;

var
  HistogramWin: THistogramWin;

implementation

{$R *.DFM}

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FormCreate(Sender: TObject);
begin
  try
    GraphBM := TBitmap.Create;
  except
    Close;
  end;
  ControlStyle:= ControlStyle + [csOpaque]; //reduce flicker
  InitialiseGraphBM;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.InitialiseGraphBM;
begin
  with GraphBM do
  begin
    PixelFormat:= pf24bit;
    HandleType:= bmDIB;
    Width:= ClientWidth;
    Height:= ClientHeight;// - Panel.Height;
    Canvas.Brush.Color:= clBlack;
    Canvas.Pen.Color:= clLime;
    Canvas.Font.Color:= clYellow;
    Canvas.Font.Name:= 'Small Fonts';
    Canvas.Font.Size:= 6;
  end;
  Paint;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FillHistogram(const DataArray : array of cardinal; nbins : integer);
var i, binidx : integer;
begin { build histogram }
  Min:= 0;//if Min = 0 then Min:= MinIntValue(DataArray); //...then set to range of current DataArray
  Max:= MaxCardValue(DataArray); //if range is undefined... //Max:= 15000000;
  if Max = 0 then Max:= 1; //avoid rtl error div/0
  NumBins:= nbins;
  if Length(BinCount) <> NumBins then SetLength(BinCount, NumBins); //allocate memory, if required
  for i:= Low(DataArray) to High(DataArray) do
  begin
    //if binidx < High(BinCount) then //<--- only check if max can be changed
    binidx:= Round(DataArray[i] / Max * (NumBins - 1));
    inc(BinCount[binidx]);
  end;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.PlotHistogram;
var i, barwidth : integer;
  baroffset : single;
begin { plot histogram }
  with GraphBM.Canvas do
  begin
    Brush.Color:= clBlack;
    FillRect(ClientRect); //clear canvas
    Brush.Color:= clRed;
    BarWidth:= Width div NumBins;
    BarOffset:= Width / NumBins;
    if BarWidth = 0 then BarWidth:= 1;
    for i:= 0 to NumBins - 1 do
      FillRect(Rect(Trunc(BarOffset*i), GraphBM.Height - BinCount[i] * 3,
               Trunc(BarOffset*i) + BarWidth, GraphBM.Height));
  end{canvas};
  Paint;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.Reset;
var i : integer;
begin
  for i:= Low(BinCount) to High(BinCount) do BinCount[i]:= 0; //clear histogram
  Min:= 0;
  Max:= 0;
  LastXpos:= - 1;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.PlotGUIMarker(XPos : integer);
begin
  with GraphBM.Canvas do
  begin
    Pen.Color:= clWhite;
    if LastXPos >= 0 then //erase last time marker...
    begin
      Pen.Mode:= pmNotMask;
      MoveTo(LastXPos, 0);
      LineTo(LastXPos, Height);
    end;
    Pen.Mode:= pmNot;
    MoveTo(XPos, 0); //draw new time marker...
    LineTo(XPos, Height);
    Pen.Mode:= pmCopy;
  end;
  LastXPos:= XPos;
  Paint;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FormMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  if Button = mbLeft then
  begin
    LeftButtonDown:= True;
    MoveGUIMarker(X / ClientWidth);
  end;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FormMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
begin
  if LastXPos < 0 then Exit;
  if LeftButtonDown then
    MoveGUIMarker(X / ClientWidth);
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FormMouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  LeftButtonDown:= False;
  MoveGUIMarker(X / ClientWidth);
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FormResize(Sender: TObject);
begin
  with GraphBM do
  begin
    Width:= ClientWidth;
    Height:= ClientHeight;
    if BinCount <> nil then PlotHistoGram
    else begin
      Canvas.Brush.Color:= clBlack;
      Canvas.FillRect(ClientRect);
      Paint;
    end;
  end;
  LastXPos:= 0;
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.FormPaint(Sender: TObject);
begin
  Canvas.Draw(0, 0, GraphBM);
end;

{-------------------------------------------------------------------------------------}
procedure THistogramWin.Button1Click(Sender: TObject);
begin
  UpdateISIHistogram;
end;

{-------------------------------------------------------------------------------------}
destructor THistogramWin.Destroy;
begin
  FreeAndNil(GraphBM);
  //dynamic arrays too?
  inherited Destroy;
end;

end.
