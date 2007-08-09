unit VitalSigns;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  OleCtrls, DTPlot32Lib_TLB, ImgList, ComCtrls, ToolWin, AfComPort,
  StdCtrls, Buttons, AfDataDispatcher;


type
  TVitalSignsForm = class(TForm)
    ComPort: TAfComPort;
    DTPlot: TDTPlot32;
    StatusBar: TStatusBar;
    Button1: TButton;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    UpDown: TUpDown;
    ToolBar1: TToolBar;
    ToolButton1: TToolButton;
    ToolButton2: TToolButton;
    tb_Vital_Sign_images: TImageList;
    ToolButton3: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    ToolButton6: TToolButton;
    ToolButton7: TToolButton;
    ToolButton8: TToolButton;
    procedure ComPortDataRecived(Sender: TObject; Count: Integer);
    procedure Button1Click(Sender: TObject);
    procedure UpDownClick(Sender: TObject; Button: TUDBtnType);
  private
    i: integer;
    { Private declarations }
  public
    { Public declarations }
  end;

var
  VitalSignsForm: TVitalSignsForm;

implementation

{$R *.DFM}

procedure TVitalSignsForm.ComPortDataRecived(Sender: TObject;
  Count: Integer);
var ComBuffer : string;
  SpO2, BPM : integer;
begin
  inc (i);
  Sleep (50); //wait 50ms for whole serial stream before reading buffer
  ComBuffer:= TrimLeft(ComPort.ReadString); //remove any leading spaces
  if (ComBuffer = '') or (Length(ComBuffer) < 10) then exit; //ie. ignore spaces
  StatusBar.Panels[0].Text:= ComBuffer;
  StatusBar.Panels[1].Text:= inttostr(i) +' events';
  StatusBar.Panels[2].Text:= inttostr(Count) + ' bytes';
  BPM:= StrToInt(Copy(ComBuffer, Pos('bpm', ComBuffer)-2, 2)); //extract BPM
  SpO2:= StrToInt(Copy(ComBuffer, Pos('%', ComBuffer)-2, 2)); //extract Sp02
  DTPlot.SinglePoint:= BPM;
  DTPlot.SinglePoint:= SpO2 * 2; //*2 to equalize y axis
end;

procedure TVitalSignsForm.Button1Click(Sender: TObject);
begin
  ComPort.ExecuteConfigDialog;
  ComPort.Open;
  ComPort.PurgeRx;
end;

procedure TVitalSignsForm.UpDownClick(Sender: TObject;
  Button: TUDBtnType);
begin
  DtPlot.XStart:= UpDown.Position;
end;

end.
