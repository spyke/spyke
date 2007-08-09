unit Main;

interface

uses
  {$IFDEF WIN32} Windows, {$ELSE} WinTypes, WinProcs, {$ENDIF}
  Graphics, Classes, Controls, Forms, Dialogs, StdCtrls, ExtCtrls;

type
  TfrmMain = class(TForm)
    Panel1: TPanel;
    Panel2: TPanel;
    Panel3: TPanel;
    btnOK: TButton;
    procedure btnOKClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  frmMain: TfrmMain;

implementation

uses
  GIFImage;

{$R *.DFM}

{$IFDEF WIN32}
  {$R GIFRes32.res}
{$ELSE}
  {$R GIFRes16.res}
{$ENDIF}

procedure TfrmMain.FormCreate(Sender: TObject);
var
  GIF: TGIFImage;
begin
  GIF := TGIFImage.Create(Self);
  GIF.Parent := Panel1;
  GIF.Align  := alClient;
  GIF.Center := True;
  GIF.LoadFromResourceName(HINSTANCE, 'Smile');

  GIF := TGIFImage.Create(Self);
  GIF.Parent := Panel2;
  GIF.Align  := alClient;
  GIF.Center := True;
  GIF.LoadFromResourceName(HINSTANCE, 'RedButton');

  GIF := TGIFImage.Create(Self);
  GIF.Parent := Panel3;
  GIF.Align  := alClient;
  GIF.Center := True;
  GIF.LoadFromResourceName(HINSTANCE, 'GreenButton');
end;

procedure TfrmMain.btnOKClick(Sender: TObject);
begin
  Close;
end;

end.

