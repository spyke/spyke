unit Display2SURF;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  OleCtrls, DTAcq32Lib_TLB;

type
  TDisplay = class(TForm)
    DTAcq321: TDTAcq32;
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Display: TDisplay;

implementation

{$R *.DFM}

end.
