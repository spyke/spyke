unit InfoWinUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls;

type
  TInfoWin = class(TForm)
  private
    InfoLabels: array of TLabel;
    { Private declarations }
  public
    procedure AddInfoLabel (InfoText : shortstring);
    procedure ChangeInfoData (InfoLabel : shortstring; InfoData : shortstring);
    { Public declarations }
  end;

var
  InfoWin: TInfoWin;

implementation

{$R *.DFM}

procedure TInfoWin.AddInfoLabel (InfoText : shortstring);
var index : byte;
begin
  index := Length(InfoLabels);
  Setlength(InfoLabels, index+1);
  InfoLabels[index] := TLabel.Create(Self);
  InfoLabels[index].Parent := Self;
  InfoLabels[index].Top:= index*Infolabels[0].Height; //space labels depending on size
  InfoLabels[index].Name:= InfoText;
  InfoLabels[index].Caption:= InfoText + ': ';
end;

procedure TInfoWin.ChangeInfoData (InfoLabel : shortstring; InfoData : shortstring); // create overloaded version for ints, floats, string etc..
var i : byte;
begin
  for i := 0 to Length(InfoLabels)-1 do
  begin
    if InfoLabel = InfoLabels[i].Name then
    begin
      InfoLabels[i].Caption:= InfoLabels[i].Name + ': ' + InfoData;
      break;
    end;
  end;
end;

end.
