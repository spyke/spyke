unit FileProgressUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ComCtrls, Gauges;
type
  TFileProgressWin = class(TForm)
    FileProgress: TGauge;
    procedure FormKeyDown(Sender: TObject; var Key: Word;
      Shift: TShiftState);
  private
    { Private declarations }
  public
    ESCPressed : boolean;
    { Public declarations }
  end;

var
  FileProgressWin: TFileProgressWin;

implementation

{$R *.DFM}

procedure TFileProgressWin.FormKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  if (Key=VK_ESCAPE) then ESCPressed:= True;
end;

end.
