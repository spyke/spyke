unit FileViewerMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, AfViewers, StdCtrls, ComCtrls, ImgList;

type
  TForm1 = class(TForm)
    Panel1: TPanel;
    OpenBtn: TButton;
    AfFileViewer1: TAfFileViewer;
    OpenDialog1: TOpenDialog;
    CopyBtn: TButton;
    FontBtn: TButton;
    FontDialog1: TFontDialog;
    StatusBar1: TStatusBar;
    ImageList1: TImageList;
    MarkBtn: TButton;
    GotoMarkBtn: TButton;
    procedure OpenBtnClick(Sender: TObject);
    procedure CopyBtnClick(Sender: TObject);
    procedure FontBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure AfFileViewer1CursorChange(Sender: TObject;
      CursorPos: TPoint);
    procedure StatusBar1DrawPanel(StatusBar: TStatusBar;
      Panel: TStatusPanel; const Rect: TRect);
    procedure AfFileViewer1ScanBlock(Sender: TObject);
    procedure AfFileViewer1DrawLeftSpace(Sender: TObject; const Line,
      LeftCharPos: Integer; Rect: TRect; State: TAfCLVLineState);
    procedure MarkBtnClick(Sender: TObject);
    procedure GotoMarkBtnClick(Sender: TObject);
    procedure AfFileViewer1LeftSpaceMouseDown(Sender: TObject;
      Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
  private
    FMarkedLineNumber: Integer;
  public
    procedure SetMarkedLineNumber(L: Integer);
  end;

var
  Form1: TForm1;

implementation

{$R *.DFM}

function IntToExtended(I: Integer): Extended;
begin
  Result := I;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  AfFileViewer1.Font.Assign(FontDialog1.Font);
  StatusBar1.DoubleBuffered := True;
  FMarkedLineNumber := -1;
end;

procedure TForm1.OpenBtnClick(Sender: TObject);
begin
  with OpenDialog1 do
  begin
    if Execute then
    begin
      FMarkedLineNumber := -1;
      AfFileViewer1.FileName := FileName;
      AfFileViewer1.OpenFile;
      AfFileViewer1.SetFocus;
      StatusBar1.Panels[1].Text := Format('%8.0n bytes ', [IntToExtended(AfFileViewer1.FileSize)]);
    end;
  end;
end;

procedure TForm1.CopyBtnClick(Sender: TObject);
begin
  AfFileViewer1.CopyToClipboard;
end;

procedure TForm1.FontBtnClick(Sender: TObject);
begin
  with FontDialog1 do
    if Execute then
    begin
      AfFileViewer1.Font.Assign(Font);
      AfFileViewer1.SetFocus;
    end;
end;

procedure TForm1.MarkBtnClick(Sender: TObject);
begin
  SetMarkedLineNumber(AfFileViewer1.FocusedPoint.Y);
end;

procedure TForm1.GotoMarkBtnClick(Sender: TObject);
begin
  if FMarkedLineNumber <> -1 then
  begin
    AfFileViewer1.FocusedPoint := Point(0, FMarkedLineNumber);
    AfFileViewer1.ScrollIntoView;
    AfFileViewer1.SetFocus;
  end;
end;

procedure TForm1.AfFileViewer1CursorChange(Sender: TObject;
  CursorPos: TPoint);
begin
  with CursorPos do
    StatusBar1.Panels[0].Text := Format('%d:  %d  ', [Y + 1, X + 1]);
end;

procedure TForm1.StatusBar1DrawPanel(StatusBar: TStatusBar;
  Panel: TStatusPanel; const Rect: TRect);
var
  BarWidth: Integer;
  R: TRect;
begin
  if (Panel.Index = 3) and (AfFileViewer1.FileSize > 0) then
  begin
    R := Rect;
    InflateRect(R, -10, -2);
    Dec(R.Right, 20);
    BarWidth := R.Right - R.Left;
    R.Right := R.Left + Round(BarWidth * (AfFileViewer1.ScanPosition / AfFileViewer1.FileSize));
    StatusBar.Canvas.Brush.Color := clHighlight;
    StatusBar.Canvas.FillRect(R);
  end;
end;

procedure TForm1.AfFileViewer1ScanBlock(Sender: TObject);
begin
  StatusBar1.Panels[2].Text := Format('Total lines: %d', [AfFileViewer1.LineCount]);
  StatusBar1.Repaint;
end;

procedure TForm1.AfFileViewer1DrawLeftSpace(Sender: TObject; const Line,
  LeftCharPos: Integer; Rect: TRect; State: TAfCLVLineState);
var
  R: TRect;
begin
  with TAfFileViewer(Sender) do
  begin
    R := Rect;
    Canvas.Brush.Color := clBtnFace;
    Dec(R.Right, 3);
    Canvas.FillRect(R);
    Canvas.Pen.Color := clWindowText;
    Canvas.MoveTo(R.Right, R.Top);
    Canvas.LineTo(R.Right, R.Bottom);
    R.Left := R.Right + 1;
    R.Right := Rect.Right;
    Canvas.Brush.Color := Color;
    Canvas.FillRect(R);
    if (Line <> - 1) and (Line = FMarkedLineNumber) then
    begin
      R := Rect;
      Inc(R.Top, CharHeight div 2 - ImageList1.Height div 2);
      ImageList1.Draw(Canvas, R.Left, R.Top, 0);
    end;
  end;
end;

procedure TForm1.AfFileViewer1LeftSpaceMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var
  P: TPoint;
begin
  with TAfFileViewer(Sender) do
  begin
    P := MouseToPoint(X, Y);
    SetMarkedLineNumber(P.Y);
  end;
end;

procedure TForm1.SetMarkedLineNumber(L: Integer);
begin
  AfFileViewer1.InvalidateLeftSpace(FMarkedLineNumber, FMarkedLineNumber);
  if FMarkedLineNumber = L then
    FMarkedLineNumber := -1
  else
    FMarkedLineNumber := L;
  AfFileViewer1.InvalidateLeftSpace(FMarkedLineNumber, FMarkedLineNumber);
  AfFileViewer1.SetFocus;
end;

end.
