unit L8ABOUT;

interface

uses Windows, Classes, Graphics, Forms, Controls, StdCtrls,
  Buttons, ExtCtrls{ Animate, GIFCtrl};

type
  TL8AboutBox = class(TForm)
    OKButton: TButton;
    Label1: TLabel;
    Panel1: TPanel;
//    L8Spin: TRxGIFAnimator;
    ProductName: TLabel;
    Version: TLabel;
    Copyright: TLabel;
    procedure FormShow(Sender: TObject);
    procedure FormHide(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  L8AboutBox: TL8AboutBox;

implementation

{$R *.DFM}

procedure TL8AboutBox.FormShow(Sender: TObject);
begin
//  L8Spin.Animate := TRUE;
end;

procedure TL8AboutBox.FormHide(Sender: TObject);
begin
//  L8Spin.Animate := FALSE;
end;

end.

