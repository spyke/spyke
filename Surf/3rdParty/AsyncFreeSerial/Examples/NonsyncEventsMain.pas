unit NonsyncEventsMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, AfViewers, AfDataDispatcher, AfComPort, AfComPortCore, StdCtrls,
  ComCtrls, Spin;

const
  MeasureBlockCount = 20;

  UM_SYNCNOTIFYMESSAGE = WM_USER + $100;
  SNC_SHOWSTATSTICS = 1;
  SNC_UPDATEBUTTONS = 2;

type
  TForm1 = class(TForm)
    Panel1: TPanel;
    AfComPort1: TAfComPort;
    MessageTerminal: TAfTerminal;
    StartBtn: TButton;
    AfTerminal1: TAfTerminal;
    Splitter1: TSplitter;
    StopBtn: TButton;
    SyncCheckBox: TCheckBox;
    StatusBar1: TStatusBar;
    SyncTimeoutSpinEdit: TSpinEdit;
    Label1: TLabel;
    procedure AfComPort1NonSyncEvent(Sender: TObject;
      EventKind: TAfCoreEvent; Data: Cardinal);
    procedure AfComPort1LineError(Sender: TObject; Errors: Cardinal);
    procedure StartBtnClick(Sender: TObject);
    procedure AfComPort1DataRecived(Sender: TObject; Count: Integer);
    procedure FormCreate(Sender: TObject);
    procedure StopBtnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
  private
    BlockCount, CharSentCount, SyncMessageTimeout: Integer;
    RespondTimes: array[1..MeasureBlockCount] of Extended;
    Running, StopRequest: Boolean;
    TotalMinTime, TotalAvgTime, TotalMaxTime: Extended;
    UseSynchronizedEvents: Boolean;
    procedure UMSyncNotifyMessage(var Message: TMessage); message UM_SYNCNOTIFYMESSAGE;
  public
    procedure CalculateStatistics;
    procedure ClearStatistics;
    function CharReceivedBack: Boolean;
    procedure UpdateButtons;
    procedure UpdateTotalStatistics;
    procedure WriteOneChar;
  end;

var
  Form1: TForm1;

implementation

{$R *.DFM}

uses
  PvStopper;

procedure TForm1.CalculateStatistics;
var
  AvgTime: Extended;
  I, MinLimitIndex, MaxLimitIndex: Integer;
begin
  SyncMessageTimeout := SyncTimeoutSpinEdit.Value;
  AvgTime := 0;
  MinLimitIndex := 0;
  MaxLimitIndex := 0;
  for I := Low(RespondTimes) to High(RespondTimes) do
  begin
    if (MinLimitIndex = 0) or (RespondTimes[I] < RespondTimes[MinLimitIndex]) then
      MinLimitIndex := I;
    if (MaxLimitIndex = 0) or (RespondTimes[I] > RespondTimes[MaxLimitIndex]) then
      MaxLimitIndex := I;
  end;
  for I := Low(RespondTimes) to High(RespondTimes) do
//    if (I <> MaxLimitIndex) and (I <> MaxLimitIndex) then
      AvgTime := AvgTime + RespondTimes[I];
  AvgTime := AvgTime / (High(RespondTimes) {- 2});
  MessageTerminal.WriteString(Format('Min: %5.2f , Avg: %5.2f , Max: %5.2f'#13#10, [
    RespondTimes[MinLimitIndex] * 1E3, AvgTime * 1E3, RespondTimes[MaxLimitIndex] * 1E3]));

  Inc(BlockCount);
  if (TotalMinTime = 0) or (RespondTimes[MinLimitIndex] < TotalMinTime) then
    TotalMinTime := RespondTimes[MinLimitIndex];
  if RespondTimes[MaxLimitIndex] > TotalMaxTime then
    TotalMaxTime := RespondTimes[MaxLimitIndex];
  TotalAvgTime := (TotalAvgTime * (BlockCount - 1) + AvgTime) / BlockCount;
  UpdateTotalStatistics;
end;

function TForm1.CharReceivedBack: Boolean;
begin
  SP_Stop(1);
  RespondTimes[CharSentCount] := SP_Time(1);
  if CharSentCount = MeasureBlockCount then
  begin
    CharSentCount := 0;
    Result := True;
  end else Result := False;
end;

procedure TForm1.ClearStatistics;
begin
  BlockCount := 0;
  TotalMinTime := 0;
  TotalMaxTime := 0;
  TotalAvgTime := 0;
  UpdateTotalStatistics;
end;

procedure TForm1.UpdateButtons;
begin
  StartBtn.Enabled := not Running;
  StopBtn.Enabled := Running;
  SyncTimeoutSpinEdit.Enabled := not SyncCheckBox.Checked;
end;

procedure TForm1.UpdateTotalStatistics;
begin
  with StatusBar1 do
  begin
    Panels[0].Text := Format('Blocks: %d', [BlockCount]);
    Panels[1].Text := Format('Min: %5.2f ms', [TotalMinTime * 1E3]);
    Panels[2].Text := Format('Avg: %5.2f ms', [TotalAvgTime * 1E3]);
    Panels[3].Text := Format('Max: %5.2f ms', [TotalMaxTime * 1E3]);
  end;
end;

procedure TForm1.WriteOneChar;
begin
  if StopRequest then
  begin // if Stop was pressed, don't send another character and notify button state
    StopRequest := False;
    Running := False;
    SendMessage(Handle, UM_SYNCNOTIFYMESSAGE, SNC_UPDATEBUTTONS, 0);
  end else
  begin
    Inc(CharSentCount);
    SP_Start(1);
    AfComPort1.WriteChar(Chr(CharSentCount + 64));
  end;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  AfTerminal1.MaxLineLength := MeasureBlockCount;
  ClearStatistics;
  Running := False;
  StopRequest := False;
  UpdateButtons;
end;

procedure TForm1.AfComPort1NonSyncEvent(Sender: TObject;
  EventKind: TAfCoreEvent; Data: Cardinal);
begin // This event is NOT called from the main VCL thread,
      // so you CAN'T call any VCL's component methods here !
  with TAfComPort(Sender) do
    case EventKind of
      ceLineEvent:
        begin
          if Data and EV_ERR <> 0 then // When line error occurs, handle it in OnXXX event
            SynchronizeEvent(EventKind, Data, 2000);
          if Data and EV_RXCHAR <> 0 then
          begin
            if UseSynchronizedEvents then // When synchronized events are choosen, handle it in OnXXX events
              SynchronizeEvent(EventKind, Data, 1000) // normal synchronized processing
            else
            if Running then
            begin
              if CharReceivedBack then // stop timer
              begin // if test block is done, show statistics and call OnXXX method to show data
                SendMessage(Self.Handle, UM_SYNCNOTIFYMESSAGE, SNC_SHOWSTATSTICS, 0);
              end;
              WriteOneChar; // send next char and start timer
              if SyncMessageTimeout > 0 then
                SynchronizeEvent(EventKind, Data, SyncMessageTimeout);
            end;
          end;
        end;
    end;
end;

procedure TForm1.AfComPort1LineError(Sender: TObject; Errors: Cardinal);
begin
  with MessageTerminal do
    WriteColorStringAndData(Format('Line Errors: %.8x'#13#10, [Errors]),
    TermColor[clWhite], TermColor[clRed], nil);
end;

procedure TForm1.AfComPort1DataRecived(Sender: TObject; Count: Integer);
begin
  if UseSynchronizedEvents then
  begin 
    if CharReceivedBack then CalculateStatistics;
    WriteOneChar;
  end;
  AfTerminal1.WriteString(AfComPort1.ReadString);
end;

procedure TForm1.UMSyncNotifyMessage(var Message: TMessage);
begin
  case Message.WParam of
    SNC_SHOWSTATSTICS:
      CalculateStatistics;
    SNC_UPDATEBUTTONS:
      UpdateButtons;
  end;
  Message.Result := 0;
end;

procedure TForm1.StartBtnClick(Sender: TObject);
begin
  CharSentCount := 0;
  ClearStatistics;
  AfTerminal1.ClearBuffer;
  SyncMessageTimeout := SyncTimeoutSpinEdit.Value;
  UseSynchronizedEvents := SyncCheckBox.Checked;
  Running := True;
  StopRequest := False;
  UpdateButtons;
  WriteOneChar;
end;

procedure TForm1.StopBtnClick(Sender: TObject);
begin
  StopRequest := True;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  StopRequest := True;
  AfComPort1.Close;
end;

end.
