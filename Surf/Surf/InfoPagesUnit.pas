unit InfoPagesUnit;

interface

uses Windows, SysUtils, Classes, Graphics, Forms, Controls, StdCtrls,
  Buttons, ComCtrls, ExtCtrls;

type
  TInfoPages = class(TForm)
    Panel1: TPanel;
    InfoWin: TPageControl;
    TabSheet1: TTabSheet;
    TabSheet2: TTabSheet;
    TabSheet3: TTabSheet;
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  InfoPages: TInfoPages;

implementation

{$R *.DFM}

end.

