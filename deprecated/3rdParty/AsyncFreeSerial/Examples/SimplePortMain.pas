unit SimplePortMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, AfViewers, StdCtrls, AfPortControls, AfDataDispatcher,
  AfComPort;

type
  TForm1 = class(TForm)
    Panel1: TPanel;
    AfTerminal1: TAfTerminal;
    AfComPort1: TAfComPort;
    Button1: TButton;
    AfPortRadioGroup1: TAfPortRadioGroup;
    procedure AfTerminal1SendChar(Sender: TObject; var Key: Char);
    procedure AfComPort1DataRecived(Sender: TObject; Count: Integer);
    procedure Button1Click(Sender: TObject);
    procedure AfPortRadioGroup1Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.DFM}

procedure TForm1.AfTerminal1SendChar(Sender: TObject; var Key: Char);
begin
  AfComPort1.WriteChar(Key);
end;

procedure TForm1.AfComPort1DataRecived(Sender: TObject; Count: Integer);
begin
  AfTerminal1.WriteString(AfComPort1.ReadString);
end;

procedure TForm1.Button1Click(Sender: TObject);
begin
  AfComPort1.ExecuteConfigDialog;
end;

procedure TForm1.AfPortRadioGroup1Click(Sender: TObject);
begin
  AfTerminal1.SetFocus;
end;

end.
