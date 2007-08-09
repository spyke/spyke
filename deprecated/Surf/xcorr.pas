unit xcorr;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Spin, ExtCtrls, TeeProcs, TeEngine, Chart, Series;

type
  TForm5 = class(TForm)
    Button1: TButton;
    Delay: TSpinEdit;
    Chart1: TChart;
    Series1: TLineSeries;
    Series2: TLineSeries;
    Series3: TLineSeries;
    procedure Button1Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form5: TForm5;

implementation

uses SurfMathLibrary;

{$R *.DFM}

procedure TForm5.Button1Click(Sender: TObject);
const x : array [0..15] of Word = (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
      y : array [0..15] of Word = (0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
var r : array [0..16] of TReal32;
  i : integer;
begin
  XCorr(x, y, r);
  for i:= 0 to 15 do
  begin
    Series1.AddY(x[i]);
    Series2.AddY(y[i]);
    Series3.AddY(r[i]);
  end;
end;

end.
