unit InfoWinUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Gauges;

type
  TInfoWin = class(TForm)
    Label1: TLabel;
    TimeMin: TLabel;
    TimeSec: TLabel;
    Label3: TLabel;
    SpikesAcquired: TLabel;
    label9: TLabel;
    Label6: TLabel;
    Label8: TLabel;
    SpikesDisplayed: TLabel;
    SpikesSaved: TLabel;
    Label2: TLabel;
    CRBuffersAcquired: TLabel;
    CRBuffersSaved: TLabel;
    CRBuffersDisplayed: TLabel;
    Label10: TLabel;
    DinBuffersSaved: TLabel;
    DinBuffersAcquired: TLabel;
    Update: TCheckBox;
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  InfoWin: TInfoWin;

implementation

{$R *.DFM}


end.
