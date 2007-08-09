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
    cbInterlaced: TCheckBox;
    cbTransparent: TCheckBox;
    pnlColor: TPanel;
    bvlSpacer: TBevel;
    btnGo: TButton;
    btnExit: TButton;
    dlgOpen: TOpenDialog;
    dlgSave: TSaveDialog;
    dlgColor: TColorDialog;
    lblProgress: TLabel;
    procedure btnBrowseInputClick(Sender: TObject);
    procedure btnBrowseOutputClick(Sender: TObject);
    procedure pnlColorClick(Sender: TObject);
    procedure btnGoClick(Sender: TObject);
    procedure btnExitClick(Sender: TObject);
    procedure GIFImageProgress(Sender: TObject; const BytesProcessed,
      BytesToProcess: LongInt; PercentageProcessed: Integer;
      var KeepOnProcessing: Boolean);
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
var
  Bitmap: TBitmap;
begin
  if dlgOpen.Execute then begin
    editInput.Text := dlgOpen.FileName;
    Bitmap := TBitmap.Create;
    try
      Bitmap.LoadFromFile(editInput.Text);
      pnlColor.Color := Bitmap.TransparentColor;
    finally
      Bitmap.Free;
    end;
  end;
end;

procedure TfrmMain.btnBrowseOutputClick(Sender: TObject);
begin
  if dlgSave.Execute then begin
    editOutput.Text := dlgSave.FileName;
  end;
end;

procedure TfrmMain.pnlColorClick(Sender: TObject);
begin
  dlgColor.Color := pnlColor.Color;
  if dlgColor.Execute then begin
    pnlColor.Color := dlgColor.Color;
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
  Bitmap: TBitmap;
  Success: Boolean;
begin
  OnWriteProgress := GIFImageProgress;

  { Note that Delphi1, Delphi2 and C++Builder1 keep their TBitmap as a         }
  { device-dependant bitmap (DDB). This means that you'll lose some color      }
  { information if your screen is running at less than 24 bits per pixel       }
  { (16 Million Colors). This also happens in Delphi3 or later and C++Builder3 }
  { or later if you set (or cause to be set) TBitmap.HandleType = bmDDB.       }
  { The usual culprit is manipulation of the bitmap's Canvas.                  }

  Bitmap := TBitmap.Create;
  try
    Bitmap.LoadFromFile(editInput.Text);

    { Now you can manipulate Bitmap as you wish, eg.: }
    {                                                 }
    { Bitmap.Canvas.TextOut(0, 0, 'Text!');           }
    { etc...                                          }

    Success := SaveToFileSingle(editOutput.Text,        { Output FileName                                    }
                                Bitmap,                 { Bitmap to be converted to GIF                      }
                                cbInterlaced.Checked,   { Should the GIF be interlaced?                      }
                                cbTransparent.Checked,  { Does the image contain transparent areas?          }
                                pnlColor.Color);        { If so, which color should be taken as transparent? }

    { Note that you don't need to create an instance of TGIFImage fo output... }
    { There is also a SaveToStreamSingle procedure for saving to a TStream.... }
  finally
    Bitmap.Free;
  end;

  if Success then begin
    ShowMessage('GIF Saved. Success!');
  end
  else begin
    ShowMessage(Format('Save Failed. Error Code = %d [%s]',
                       [Ord(LastWriteError), ErrorDescription[LastWriteError]]));
  end;
end;

procedure TfrmMain.btnExitClick(Sender: TObject);
begin
  Close;
end;

procedure TfrmMain.GIFImageProgress(Sender: TObject; const BytesProcessed,
  BytesToProcess: LongInt; PercentageProcessed: Integer;
  var KeepOnProcessing: Boolean);
begin
  if (PercentageProcessed = 100) then begin
    lblProgress.Visible := False;
  end
  else begin
    lblProgress.Caption := Format('%d%% Complete', [PercentageProcessed]);
    lblProgress.Visible := True;
  end;
end;

end.

