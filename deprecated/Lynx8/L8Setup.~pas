unit L8Setup;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Spin, OleCtrls, vcf1, ExtCtrls;

const MAXAMPS = 8; {determined right now by number of pins on output port}

type
  TSetupL8 = class(TForm)
    StaticText6: TStaticText;
    StaticText7: TStaticText;
    NAmps: TSpinEdit;
    InputGain: TComboBox;
    OutPutPort: TRadioGroup;
    OkButton: TButton;
    CancelBtn: TButton;
    procedure FormActivate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure OkButtonClick(Sender: TObject);
    procedure CancelBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
    NampBak,GainIndexBak,PortAddBak : Integer;
    ok : boolean;
  public
    { Public declarations }
  end;

var
  SetupL8: TSetupL8;

implementation

{$R *.DFM}

procedure TSetupL8.FormActivate(Sender: TObject);
begin
  NampBak := NAmps.Value;
  GainIndexBak := InputGain.ItemIndex;
  PortAddBak := OutPutPort.ItemIndex;
  Ok := FALSE;
end;

procedure TSetupL8.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  if not ok then
  begin
    NAmps.Value := NampBak;
    InputGain.ItemIndex := GainIndexBak;
    OutPutPort.ItemIndex := PortAddBak;
  end;
end;

procedure TSetupL8.OkButtonClick(Sender: TObject);
begin
   ok := TRUE;
   Close;
end;

procedure TSetupL8.CancelBtnClick(Sender: TObject);
begin
  Close;
end;

procedure TSetupL8.FormCreate(Sender: TObject);
begin
  InputGain.ItemIndex := 2;
end;

end.
