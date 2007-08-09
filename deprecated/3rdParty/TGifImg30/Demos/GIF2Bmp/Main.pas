unit Main;

interface

uses
  {$IFDEF WIN32} Windows, {$ELSE} WinTypes, WinProcs, {$ENDIF}
  SysUtils, Graphics, Classes, Controls, Forms, Dialogs, StdCtrls, ExtCtrls;

type
  TfrmMain = class(TForm)
    lblInput: TLabel;
    editInput: TEdit;
    btnBrowseInput: TButton;
    lblOutput: TLabel;
    editOutput: TEdit;
    btnBrowseOutput: TButton;
    bvlSpacer: TBevel;
    btnGo: TButton;
    btnExit: TButton;
    dlgOpen: TOpenDialog;
    dlgSave: TSaveDialog;
    procedure btnBrowseInputClick(Sender: TObject);
    procedure btnBrowseOutputClick(Sender: TObject);
    procedure btnGoClick(Sender: TObject);
    procedure btnExitClick(Sender: TObject);
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

procedure TfrmMain.btnBrowseInputClick(Sender: TObject);
begin
  if dlgOpen.Execute then begin
    editInput.Text := dlgOpen.FileName;
  end;
end;

procedure TfrmMain.btnBrowseOutputClick(Sender: TObject);
begin
  if dlgSave.Execute then begin
    editOutput.Text := dlgSave.FileName;
  end;
end;

procedure TfrmMain.btnGoClick(Sender: TObject);
const
  ErrorDescription: array[TGIFError] of String = ('None',
                                                  'Operation Cancelled',
                                                  'Internal Error',
                                                  'Invalid File Format',
                                                  'Image contains too many Colors (>256)',
                                                  'Index out of Bounds',
                                                  'Windows GDI Error',
                                                  'File not Found',
                                                  'Resource not Found');
var
  GIF: TGIFImage;
  Success: Boolean;
  Bitmap: TBitmap;
begin
  GIF := TGIFImage.Create(nil);
  try
    Success := GIF.LoadFromFile(editInput.Text);

    if Success then begin
      { There are 2 ways to do this: }

      { 1st Method:
        GIF.Bitmap.SaveToFile(editOutput.Text);
      }

      { 2nd Method: }
      Bitmap := TBitmap.Create;
      try
        Bitmap.Assign(GIF);  { or Bitmap.Assign(GIF.Bitmap); }
        { Now you're free to manipulate Bitmap before saving it... }
        Bitmap.SaveToFile(editOutput.Text);
      finally
        Bitmap.Free;
      end;
      ShowMessage('GIF Converted. Success!');
    end
    else begin
      ShowMessage(Format('Load Failed. Error Code = %d [%s]',
                         [Ord(GIF.LastError), ErrorDescription[GIF.LastError]]));
    end;
  finally
    GIF.Free;
  end;
end;

procedure TfrmMain.btnExitClick(Sender: TObject);
begin
  Close;
end;

end.

