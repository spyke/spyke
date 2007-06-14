unit x2000About;

interface

uses
  Windows, Forms, Buttons, ImgList, ShellApi, ExtCtrls, Classes, Controls,
  Graphics, StdCtrls, SysUtils;

type
  TAboutForm = class(TForm)
    Shape1: TShape;
    Shape2: TShape;
    Shape3: TShape;
    Image1: TImage;
    Label1: TLabel;
    Label2: TLabel;
    SpeedButton1: TSpeedButton;
    Image2: TImage;
    Image3: TImage;
    Label13: TLabel;
    Label16: TLabel;
    Label17: TLabel;
    Label18: TLabel;
    Label19: TLabel;
    Label20: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label7: TLabel;
    Label9: TLabel;
    Label10: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    Memo1: TMemo;
    procedure SpeedButton1Click(Sender: TObject);
    procedure Label16Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    { Private-Deklarationen }
  public
    { Public-Deklarationen }
  end;

var
  AboutForm: TAboutForm;

implementation

{$R *.DFM}
{$I X2000.INC}

procedure TAboutForm.SpeedButton1Click(Sender: TObject);
begin
  close;
end;

procedure TAboutForm.Label16Click(Sender: TObject);
begin
 ShellExecute(Handle, 'open', PChar('mailto:Baldemaier.Florian@gmx.net'), nil, nil, SW_SHOW);
end;

procedure TAboutForm.FormCreate(Sender: TObject);
var i:integer;
begin
  Memo1.clear;
  Label2.caption:='Version '+ProgrammVersion+' (Build '+ProgrammBuild+')';
  For i:=1 to MaxPrograms do
    Memo1.lines.add(XPrograms[i]);
end; 

end.
