unit Hardwareconfig;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ComCtrls;

type
  THardwareConfigWin = class(TForm)
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  HardwareConfigWin: THardwareConfigWin;

implementation

{$R *.DFM}

procedure THardwareConfigWin.FormCreate(Sender: TObject);
// use TTabControl instead?
const
  TabTitles: array[0..3] of ShortString = ('Customer', 'Orders', 'Items', 'Parts' );
var
  i: Integer;
  PageControl1: TPageControl;
begin
  PageControl1 := TPageControl.Create(Self);
  PageControl1.Parent := Self;
  PageControl1.Align := alClient;
  for i := Low(TabTitles) to High(TabTitles) do
    with TTabSheet.Create(PageControl1) do
    begin
      PageControl := PageControl1;

      Name := 'ts' + TabTitles[i];
      Caption := TabTitles[i];
   end;
end;

end.
