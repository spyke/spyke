unit SpkAvgUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, TeeProcs, TeEngine, Chart;

type
  TWaveAverager = class(TForm)
    Chart1: TChart;
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  WaveAverager: TWaveAverager;

implementation

{$R *.DFM}

end.
