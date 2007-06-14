unit TerminalPortMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, AfViewers, AfDataTerminal, StdCtrls, AfPortControls,
  AfDataDispatcher, AfComPort, ImgList, ComCtrls;

type
  TMainForm = class(TForm)
    Panel1: TPanel;
    AfDataTerminal: TAfDataTerminal;
    AfComPort1: TAfComPort;
    AfDataDispatcher: TAfDataDispatcher;
    AfPortComboBox1: TAfPortComboBox;
    ConfigBtn: TButton;
    ImageList: TImageList;
    ClearBtn: TButton;
    SendBtn: TButton;
    OpenDialog: TOpenDialog;
    StatusBar: TStatusBar;
    AbortBtn: TButton;
    procedure FormCreate(Sender: TObject);
    procedure ConfigBtnClick(Sender: TObject);
    procedure AfPortComboBox1Change(Sender: TObject);
    procedure AfDataTerminalDrawLeftSpace(Sender: TObject; const Line,
      LeftCharPos: Integer; Rect: TRect; State: TAfCLVLineState);
    procedure AfDataTerminalGetColors(Sender: TObject; Line: Integer;
      var Colors: TAfTRMCharAttrs);
    procedure ClearBtnClick(Sender: TObject);
    procedure SendBtnClick(Sender: TObject);
    procedure AfDataDispatcherWriteStreamDone(Sender: TObject);
    procedure AfDataTerminalScrBckModeChange(Sender: TObject);
    procedure AfComPort1PortOpen(Sender: TObject);
    procedure AfDataDispatcherWriteStreamBlock(Sender: TObject;
      const Position, Size: Integer);
    procedure AbortBtnClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  MainForm: TMainForm;

implementation

{$R *.DFM}

procedure TMainForm.FormCreate(Sender: TObject);
begin
  AfDataTerminalScrBckModeChange(nil);
end;

procedure TMainForm.ConfigBtnClick(Sender: TObject);
begin
  AfComPort1.ExecuteConfigDialog;
  AfComPort1PortOpen(nil);
  AfDataTerminal.SetFocus;
end;

procedure TMainForm.AfPortComboBox1Change(Sender: TObject);
begin
  AfDataTerminal.SetFocus;
end;

procedure TMainForm.AfDataTerminalGetColors(Sender: TObject;
  Line: Integer; var Colors: TAfTRMCharAttrs);
var
  S: String;
  ImageIndex: Byte;
begin
  with TAfDataTerminal(Sender) do
  begin
    S := UpperCase(BufferLine[Line]);
    ImageIndex := 0;
    if Pos('AT', S) = 1 then
    begin
      Colors[0].FColor := TermColor[clGreen];
      ImageIndex := 1;
    end else
    if Pos('OK', S) = 1 then
    begin
      Colors[0].FColor := TermColor[clRed];
      ImageIndex := 2;
    end else
      Colors[0].FColor := DefaultTermColor.FColor;
    UserData[Line] := @ImageIndex;
  end;
end;

procedure TMainForm.AfDataTerminalDrawLeftSpace(Sender: TObject;
  const Line, LeftCharPos: Integer; Rect: TRect; State: TAfCLVLineState);
var
  ImageNum: Byte;
  R: TRect;
begin
  with TAfDataTerminal(Sender) do
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
    if Line <> -1 then
    begin
      ImageNum := PByte(UserData[Line])^;
      if ImageNum > 0 then
      begin
        Inc(Rect.Top, CharHeight div 2 - ImageList.Height div 2);
        ImageList.Draw(Canvas, Rect.Left, Rect.Top, ImageNum - 1);
      end;
    end;
  end;
end;

procedure TMainForm.ClearBtnClick(Sender: TObject);
begin
  AfDataDispatcher.Clear;
  AfDataTerminal.SetFocus;
end;

procedure TMainForm.SendBtnClick(Sender: TObject);
var
  FileStream: TFileStream;
begin
  with OpenDialog do
  begin
    if Execute then
    begin
      SendBtn.Enabled := False;
      AbortBtn.Enabled := True;
      FileStream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
      AfDataDispatcher.WriteStream(FileStream, True);
    end;
  end;
  AfDataTerminal.SetFocus;
end;

procedure TMainForm.AfDataDispatcherWriteStreamDone(Sender: TObject);
begin
  SendBtn.Enabled := True;
  AbortBtn.Enabled := False;
end;

procedure TMainForm.AfDataTerminalScrBckModeChange(Sender: TObject);
const
  ScrBckText: array[Boolean] of String = ('Terminal mode', 'Scrollback mode');
begin
  StatusBar.Panels[0].Text := ScrBckText[AfDataTerminal.ScrollBackMode];
end;

procedure TMainForm.AfComPort1PortOpen(Sender: TObject);
begin
  StatusBar.Panels[1].Text := AfComPort1.SettingsStr;
end;

procedure TMainForm.AfDataDispatcherWriteStreamBlock(Sender: TObject;
  const Position, Size: Integer);
begin
  StatusBar.Panels[2].Text := Format('%d%% send', [Trunc(Position / Size * 100)]); 
end;

procedure TMainForm.AbortBtnClick(Sender: TObject);
begin
  AfDataDispatcher.AbortWriteStream;
end;

end.
