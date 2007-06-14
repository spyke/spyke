unit Main;

interface

uses
  {$IFDEF WIN32} Windows, {$ELSE} WinTypes, WinProcs, {$ENDIF}
  SysUtils, Graphics, Classes, Controls, Forms, Dialogs, StdCtrls, ExtCtrls;

type
  TfrmMain = class(TForm)
    lblInput: TLabel;
    btnBrowseInput: TButton;
    lblOutput: TLabel;
    editOutput: TEdit;
    btnBrowseOutput: TButton;
    bvlSpacer: TBevel;
    btnGo: TButton;
    btnExit: TButton;
    dlgOpen: TOpenDialog;
    dlgSave: TSaveDialog;
    lbInput: TListBox;
    btnRemove: TButton;
    btnClear: TButton;
    lblProgress: TLabel;
    procedure btnBrowseInputClick(Sender: TObject);
    procedure btnBrowseOutputClick(Sender: TObject);
    procedure btnGoClick(Sender: TObject);
    procedure btnExitClick(Sender: TObject);
    procedure btnRemoveClick(Sender: TObject);
    procedure btnClearClick(Sender: TObject);
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
begin
  if dlgOpen.Execute then begin
    lbInput.Items.AddStrings(dlgOpen.Files);
  end;
end;

procedure TfrmMain.btnRemoveClick(Sender: TObject);
begin
  if (lbInput.ItemIndex >= 0) then lbInput.Items.Delete(lbInput.ItemIndex);
end;

procedure TfrmMain.btnClearClick(Sender: TObject);
begin
  lbInput.Items.Clear;
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
  AnimationInfo: TSaveInfo;
  Lp: Integer;
  Success: Boolean;
begin
  OnWriteProgress := GIFImageProgress;

  { Note that Delphi1, Delphi2 and C++Builder1 keep their TBitmap as a         }
  { device-dependant bitmap (DDB). This means that you'll lose some color      }
  { information if your screen is running at less than 24 bits per pixel       }
  { (16 Million Colors). This also happens in Delphi3 or later and C++Builder3 }
  { or later if you set (or cause to be set) TBitmap.HandleType = bmDDB.       }
  { The usual culprit is manipulation of the bitmap's Canvas.                  }

  { Make sure all the record entries are initialized to 0                      }
  FillChar(AnimationInfo, SizeOf(AnimationInfo), 0);

  { How many frames should the Animation contain? You need to set this to the  }
  { correct value (at least 1):                                                }
  AnimationInfo.siNumFrames := lbInput.Items.Count;

  { It's possible to specify the number of times the animation should repeat   }
  { itself. If you want it to go on forever, just set it to 0:                 }
  AnimationInfo.siNumLoops := 0;

  { Each frame can have it's own table of colors, or it can share a Global one.}
  { Each color table can contain a maximum of 256 colors. If you set           }
  { siUseGlobalColorTable to True, TGIFImage will try to let as many frames as }
  { possible share in the Global table, otherwise each frame will get its own. }
  { Of course this results in a smaller output file, but is only possible if   }
  { the frames don't have too many colors. Upon return from the output         }
  { function, siUseGlobalColorTable will be set to reflect the outcome:        }
  { If a Global color table was possible it will be set to True, and vice versa}
  AnimationInfo.siUseGlobalColorTable := True;

  try
    for Lp := 1 to lbInput.Items.Count do begin
      { Create a new Frame record instance. You need to dispose of it later... }
      New(AnimationInfo.siFrames[Lp]);

      { Make sure all the record entries are initialized to 0                  }
      FillChar(AnimationInfo.siFrames[Lp]^, SizeOf(TFrameInfo), 0);

      with AnimationInfo.siFrames[Lp]^ do begin
        { iiImage is of type TBitmap and stores the actual image. You need to  }
        { create it manually or set it equal to some TBitmap variable. Either  }
        { way, you're responsible for freeing it later.                        }
        iiImage := TBitmap.Create;
        iiImage.LoadFromFile(lbInput.Items[Lp - 1]);
        { Now you can manipulate iiImage as you wish, eg.:                     }
        { iiImage.Canvas.TextOut(0, 0, 'Text!');                               }
        { etc...                                                               }

        { Each frame can cover only as big a section of the total image as it  }
        { wants to. For instance, you can have an image of 100x50, while a     }
        { certain frame just covers a rectangle with a top-left position of    }
        { (20, 10). For this frame you would then set the following:           }
        { iiLeft := 20;                                                        }
        { iiTop  := 10;                                                        }

        { The following 2 entries are ignored by the output routines and only  }
        { gets used by TGIFImage.GetFrameInfo.                                 }
        { iiWidth : Integer;                                                   }
        { iiHeight: Integer;                                                   }

        { iiDelay is the time the frame is displayed before being removed.     }
        { It's specified in milliseconds, but gets rounded to the next lower   }
        { multiple of 10 (ie. a value of 397 will be taken as 390, or 0.39sec )}
        iiDelay := 400;

        { If you want to save a frame interlaced, set iiInterlaced to True.    }
        { iiInterlaced: Boolean;                                               }

        { If the frame contains transparent areas, set iiTransparent to True.  }
        { iiTransparent: Boolean;                                              }
        {                                                                      }
        { If so, then all pixels in iiImage with the same color as:            }
        { iiTransparentColor: TColor;                                          }
        { will be transparent and underlying pixels will shine through.        }

        { iiDisposalMethod specifies how the frame should be removed after its }
        { display period expires. There are 4 possible values:                 }
        { dtUndefined   : The next frame just overwrites this one.             }
        { dtDoNothing   : Same as dtUndefined.                                 }
        { dtToBackground: Frame's area gets filled with the background.        }
        { dtToPrevious  : Frame's area gets restored to what was there before. }
        iiDisposalMethod := dtToBackground;

        { Each frame can have a comment attached to it. It's not visible,      }
        { but you'll see it in a GIF Animation editor. Usually left blank.     }
        iiComment := 'Frame #' + IntToStr(Lp);
      end;
    end;

    { Now compress it and write it to the output file.                         }
    Success := SaveToFile(editOutput.Text, AnimationInfo);

    { Note that you don't need to create an instance of TGIFImage for output.  }
    { There is also a SaveToStream procedure for saving to a TStream...        }
  finally
    { Clean up...                                                              }
    for Lp := 1 to lbInput.Items.Count do begin
      if (AnimationInfo.siFrames[Lp] <> nil) then begin
        AnimationInfo.siFrames[Lp]^.iiImage.Free;
        Dispose(AnimationInfo.siFrames[Lp]);
      end;
    end;
  end;

  if Success then begin
    if AnimationInfo.siUseGlobalColorTable then begin
      ShowMessage('Animation saved using a global palette. Success!');
    end
    else begin
      ShowMessage('Animation saved using multiple local palettes. Success!');
    end;
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

