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
  FileCtrl, GIFImage;

{$R *.DFM}

procedure TfrmMain.btnBrowseInputClick(Sender: TObject);
begin
  if dlgOpen.Execute then begin
    editInput.Text := dlgOpen.FileName;
  end;
end;

procedure TfrmMain.btnBrowseOutputClick(Sender: TObject);
var
  Folder: String;
begin
  Folder := '';
{$IFDEF VER120}
  if SelectDirectory('Select Output Folder', '', Folder) then begin
{$ELSE}
  if SelectDirectory(Folder, [], 0) then begin
{$ENDIF}
    editOutput.Text := Folder;
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
  Lp: Integer;
  FrameInfo: TFrameInfo;
begin
  GIF := TGIFImage.Create(nil);
  try
    Success := GIF.LoadFromFile(editInput.Text);

    if Success then begin
      for Lp := 1 to GIF.NumFrames do begin
        FrameInfo := GIF.GetFrameInfo(Lp);
        FrameInfo.iiImage.SaveToFile(Format('%s\Frame%0.3d.bmp', [editOutput.Text, Lp]));
      end;
      ShowMessage('Animation Saved. Success!');
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

