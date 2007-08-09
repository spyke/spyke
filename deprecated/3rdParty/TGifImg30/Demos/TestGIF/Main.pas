unit Main;

interface

uses
  {$IFDEF WIN32} Windows, {$ELSE} WinTypes, WinProcs, {$ENDIF}
  Messages, SysUtils, Graphics, Classes, Controls, Forms, Dialogs, Menus,
  StdCtrls, ExtCtrls, Spin, Gauges, GIFImage;

type
  TMainForm = class(TForm)
    MainMenu: TMainMenu;
    FileMenuItem: TMenuItem;
    OpenMenuItem: TMenuItem;
    Line1: TMenuItem;
    ChooseBackgroundMenuItem: TMenuItem;
    SaveMenuItem: TMenuItem;
    Line2: TMenuItem;
    ExitMenuItem: TMenuItem;
    PropertiesMenuItem: TMenuItem;
    BackgroundMenuItem: TMenuItem;
    Line3: TMenuItem;
    AnimateMenuItem: TMenuItem;
    CenterMenuItem: TMenuItem;
    LoopMenuItem: TMenuItem;
    StretchMenuItem: TMenuItem;
    TileMenuItem: TMenuItem;
    VisibleMenuItem: TMenuItem;
    GIFPopupMenu: TPopupMenu;
    OpenPopupMenuItem: TMenuItem;
    PopupLine1: TMenuItem;
    ExitPopupMenuItem: TMenuItem;
    OpenDialog: TOpenDialog;
    SaveDialog: TSaveDialog;
    StretchRatioMenuItem: TMenuItem;
    ThreadedMenuItem: TMenuItem;
    ThreadPriorityMenuItem: TMenuItem;
    IdleMenuItem: TMenuItem;
    LowestMenuItem: TMenuItem;
    LowerMenuItem: TMenuItem;
    NormalMenuItem: TMenuItem;
    HigherMenuItem: TMenuItem;
    TimeCriticalMenuItem: TMenuItem;
    HighestMenuItem: TMenuItem;
    SaveBitmapAsMenuItem: TMenuItem;
    InfoPanel: TPanel;
    ProgressBar: TGauge;
    Bevel: TBevel;
    GroupBox1: TGroupBox;
    AnimatedCheckBox: TCheckBox;
    Label2: TLabel;
    ImageTransparentCheckBox: TCheckBox;
    ImageInterlacedCheckBox: TCheckBox;
    Label5: TLabel;
    Label6: TLabel;
    FrameGroupBox: TGroupBox;
    Label11: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    FrameTransparentCheckBox: TCheckBox;
    FrameInterlacedCheckBox: TCheckBox;
    FrameSpinEdit: TSpinEdit;
    Label1: TLabel;
    FrameWidthLabel: TLabel;
    Label4: TLabel;
    FrameHeightLabel: TLabel;
    Label10: TLabel;
    DisposalMethodLabel: TLabel;
    Label15: TLabel;
    ImageWidthLabel: TLabel;
    Label17: TLabel;
    ImageHeightLabel: TLabel;
    Label19: TLabel;
    ImageColorsLabel: TLabel;
    AnimationPanel: TPanel;
    SpeedLabel: TLabel;
    Label7: TLabel;
    FrameCountLabel: TLabel;
    Label21: TLabel;
    LoopCountLabel: TLabel;
    SpeedSpinEdit: TSpinEdit;
    DoubleBufferedMenuItem: TMenuItem;
    OpaqueMenuItem: TMenuItem;
    ColorMenuItem: TMenuItem;
    ColorDialog: TColorDialog;
    StretchBigOnlyMenuItem: TMenuItem;
    procedure FormCreate(Sender: TObject);
    procedure FormActivate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure OpenMenuItemClick(Sender: TObject);
    procedure SaveMenuItemClick(Sender: TObject);
    procedure ChooseBackgroundMenuItemClick(Sender: TObject);
    procedure ExitMenuItemClick(Sender: TObject);
    procedure BackgroundMenuItemClick(Sender: TObject);
    procedure AnimateMenuItemClick(Sender: TObject);
    procedure CenterMenuItemClick(Sender: TObject);
    procedure LoopMenuItemClick(Sender: TObject);
    procedure StretchMenuItemClick(Sender: TObject);
    procedure TileMenuItemClick(Sender: TObject);
    procedure VisibleMenuItemClick(Sender: TObject);
    procedure FrameSpinEditChange(Sender: TObject);
    procedure SpeedSpinEditChange(Sender: TObject);
    procedure StretchRatioMenuItemClick(Sender: TObject);
    procedure ThreadedMenuItemClick(Sender: TObject);
    procedure PriorityMenuItemClick(Sender: TObject);
    procedure SaveBitmapAsMenuItemClick(Sender: TObject);
    procedure AnimatedCheckBoxClick(Sender: TObject);
    procedure ColorMenuItemClick(Sender: TObject);
    procedure DoubleBufferedMenuItemClick(Sender: TObject);
    procedure OpaqueMenuItemClick(Sender: TObject);
    procedure StretchBigOnlyMenuItemClick(Sender: TObject);
  private
    procedure WMEraseBkgnd(var Message: TWmEraseBkgnd); message WM_ERASEBKGND;
  public
    GIFBackground: TGIFImage;         { Displays the textured background       }
    GIFImage: TGIFImage;              { Displays the images loaded at run-time }

    procedure UpdateCaption(aDescription: String);
    procedure GIFChange(Sender: TObject);
    procedure GIFProgress(Sender: TObject; const BytesProcessed,
      BytesToProcess: LongInt; PercentageProcessed: Integer;
      var KeepOnProcessing: Boolean);
  end;

var
  MainForm: TMainForm;

implementation

{$R *.DFM}

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.FormCreate(Sender: TObject);
begin
{$IFNDEF WIN32}
  ThreadedMenuItem.Visible := False;
  ThreadPriorityMenuItem.Visible := False;
{$ENDIF}

  GIFBackground := TGIFImage.Create(Self);
  with GIFBackground do begin
    Parent  := Self;
    Align   := alClient;
    Tile    := True;
    Visible := BackgroundMenuItem.Checked;
{$IFDEF WIN32}
    Threaded := False;
{$ENDIF}
    if not LoadFromFile(ExtractFilePath(ParamStr(0)) + 'TGIFImgB.gif') then begin
      Visible := False;
      BackgroundMenuItem.Checked := False;
    end;
  end;

  GIFImage := TGIFImage.Create(Self);
  with GIFImage do begin
    Parent     := Self;
    Align      := alClient;
    OnChange   := GIFChange;
    OnProgress := GIFProgress;

    Animate    := AnimateMenuItem.Checked;
    Center     := CenterMenuItem.Checked;
    Color      := clOlive;
    Loop       := LoopMenuItem.Checked;
  end;

  OnWriteProgress := GIFProgress;

  if (ParamCount > 0) then begin
    if GIFImage.LoadFromFile(ParamStr(1)) then begin
      SaveMenuItem.Enabled := True;
      OpenDialog.InitialDir := ExtractFilePath(ParamStr(1));
      UpdateCaption(ExtractFileName(ParamStr(1)));
    end;
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.FormActivate(Sender: TObject);
begin
  if GIFImage.Empty then UpdateCaption('No Image Loaded');
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.FormDestroy(Sender: TObject);
begin
  GIFBackground.Free;
  GIFImage.Free;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.WMEraseBkgnd(var Message: TWMEraseBkgnd);
begin
  if GIFBackground.Visible then begin
    Message.Result := 1;
  end
  else begin
    inherited;
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.OpenMenuItemClick(Sender: TObject);
begin
  with OpenDialog do begin
    FilterIndex := 1;
    if (FileName = '') then InitialDir := 'C:\WINNT\Profiles\Administrator\Desktop';;//E:\Source\Old Projects\Components\TGIFImage\GIF Samples';
    if not Execute then Exit;
    InitialDir := '';
  end;

  if GIFImage.LoadFromFile(OpenDialog.FileName) then begin
    UpdateCaption(ExtractFileName(OpenDialog.FileName));
    SaveMenuItem.Enabled := True;
  end
  else begin
    Caption := Format('GIFImage Viewer - Error Loading Image [%d]', [Ord(GIFImage.LastError)]);
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.SaveBitmapAsMenuItemClick(Sender: TObject);
var
  aBitmap: TBitmap;
  aInterlaced, aTransparent: Boolean;
begin
  with OpenDialog do begin
    FilterIndex := 2;
    if not Execute then Exit;
    SaveDialog.FileName := ChangeFileExt(FileName, '.gif');
  end;

  with SaveDialog do begin
    FilterIndex := 1;
    if not Execute then Exit;
  end;

  aInterlaced  := MessageDlg('Do you wish to save the image lines interlaced?',
                             mtConfirmation, [mbYes, mbNo], 0) = mrYes;

  aTransparent := MessageDlg('Does the Bitmap have transparent areas?',
                             mtConfirmation, [mbYes, mbNo], 0) = mrYes;

  aBitmap := TBitmap.Create;
  try
    aBitmap.LoadFromFile(OpenDialog.FileName);

    SaveToFileSingle(SaveDialog.FileName,
                     aBitmap,
                     aInterlaced,
                     aTransparent,
                     aBitmap.TransparentColor);
  finally
    aBitmap.Free;
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.SaveMenuItemClick(Sender: TObject);
var
  Lp: Integer;
  Frames: TSaveInfo;
begin
  if not SaveDialog.Execute then Exit;

  FillChar(Frames, SizeOf(Frames), 0);
  with Frames do begin
    siNumFrames           := GIFImage.NumFrames;
    siNumLoops            := 0;
    siUseGlobalColorTable := True;
  end;

  for Lp := 1 to GIFImage.NumFrames do begin
    with Frames do begin
      New(siFrames[Lp]);
      siFrames[Lp]^              := GIFImage.GetFrameInfo(Lp);
      siFrames[Lp]^.iiInterlaced := MessageDlg('Do you wish to save the image lines interlaced?',
                                               mtConfirmation, [mbYes,mbNo], 0) = mrYes;
      siFrames[Lp]^.iiComment    := '';
    end;
  end;

  if SaveToFile(SaveDialog.FileName, Frames) then begin
    if (GIFImage.NumFrames > 1) then begin
      if Frames.siUseGlobalColorTable then
        ShowMessage('Saved using a global palette')
      else
        ShowMessage('Saved using multiple local palettes');
    end;
  end
  else begin
    ShowMessage(Format('Save Failed. Error %d', [Ord(LastWriteError)]));
  end;

  for Lp := 1 to GIFImage.NumFrames do begin
    Frames.siFrames[Lp]^.iiImage.Free;
    Dispose(Frames.siFrames[Lp]);
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.ChooseBackgroundMenuItemClick(Sender: TObject);
begin
  if OpenDialog.Execute then GIFBackground.LoadFromFile(OpenDialog.FileName);
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.ExitMenuItemClick(Sender: TObject);
begin
  Close;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.BackgroundMenuItemClick(Sender: TObject);
begin
  BackgroundMenuItem.Checked := not BackgroundMenuItem.Checked;
  GIFBackground.Visible := BackgroundMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.AnimateMenuItemClick(Sender: TObject);
begin
  AnimateMenuItem.Checked := not AnimateMenuItem.Checked;
  GIFImage.Animate := AnimateMenuItem.Checked;

  if GIFImage.Animate then begin
    FrameWidthLabel.Caption          := '0';
    FrameHeightLabel.Caption         := '0';
    DisposalMethodLabel.Caption      := '';
    FrameTransparentCheckBox.Checked := False;
    FrameInterlacedCheckBox.Checked  := False;
  end
  else begin
    GIFChange(GIFImage);
  end;

  with FrameSpinEdit do begin
    Enabled := GIFImage.IsAnimated and (not GIFImage.Animate);
    Value := 1;
  end;
  SpeedSpinEdit.Enabled := GIFImage.IsAnimated and GIFImage.Animate;
  SelectNext(nil, True, True);
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.CenterMenuItemClick(Sender: TObject);
begin
  CenterMenuItem.Checked := not CenterMenuItem.Checked;
  GIFImage.Center := CenterMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.ColorMenuItemClick(Sender: TObject);
begin
  with ColorDialog do begin
    Color := GIFImage.Color;
    if Execute then begin
      GIFImage.Color := Color;
    end;
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.DoubleBufferedMenuItemClick(Sender: TObject);
begin
  DoubleBufferedMenuItem.Checked := not DoubleBufferedMenuItem.Checked;
  GIFImage.DoubleBuffered := DoubleBufferedMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.LoopMenuItemClick(Sender: TObject);
begin
  LoopMenuItem.Checked := not LoopMenuItem.Checked;
  GIFImage.Loop := LoopMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.OpaqueMenuItemClick(Sender: TObject);
begin
  OpaqueMenuItem.Checked := not OpaqueMenuItem.Checked;
  GIFImage.Opaque := OpaqueMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.StretchMenuItemClick(Sender: TObject);
begin
  StretchMenuItem.Checked := not StretchMenuItem.Checked;
  GIFImage.Stretch := StretchMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.StretchRatioMenuItemClick(Sender: TObject);
begin
  StretchRatioMenuItem.Checked := not StretchRatioMenuItem.Checked;
  GIFImage.StretchRatio := StretchRatioMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.StretchBigOnlyMenuItemClick(Sender: TObject);
begin
  StretchBigOnlyMenuItem.Checked := not StretchBigOnlyMenuItem.Checked;
  GIFImage.StretchBigOnly := StretchBigOnlyMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.ThreadedMenuItemClick(Sender: TObject);
begin
{$IFDEF WIN32}
  ThreadedMenuItem.Checked := not ThreadedMenuItem.Checked;
  GIFImage.Threaded := ThreadedMenuItem.Checked;
{$ENDIF}
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.PriorityMenuItemClick(Sender: TObject);
{$IFDEF WIN32}
var
  Lp: Integer;
begin
  with (Sender as TMenuItem) do begin
    for Lp := (Parent.Count - 1) downto 0 do begin
      { This loop is necessary because early versions of Delphi don't have a   }
      { TMenuItem.RadioItem property.                                          }
      Parent.Items[Lp].Checked := False;
    end;
    Checked := True;
    GIFImage.ThreadPriority := TThreadPriority(Tag);
  end;
{$ELSE}
begin
{$ENDIF}
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.TileMenuItemClick(Sender: TObject);
begin
  TileMenuItem.Checked := not TileMenuItem.Checked;
  GIFImage.Tile := TileMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.VisibleMenuItemClick(Sender: TObject);
begin
  VisibleMenuItem.Checked := not VisibleMenuItem.Checked;
  GIFImage.Visible := VisibleMenuItem.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.FrameSpinEditChange(Sender: TObject);
begin
  GIFImage.CurrentFrame := FrameSpinEdit.Value;
  FrameSpinEdit.Value := GIFImage.CurrentFrame;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.SpeedSpinEditChange(Sender: TObject);
begin
  GIFImage.Speed := SpeedSpinEdit.Value;
  SpeedSpinEdit.Value := GIFImage.Speed;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.AnimatedCheckBoxClick(Sender: TObject);
begin
  FrameGroupBox.Visible := AnimatedCheckBox.Checked;
  AnimationPanel.Visible := AnimatedCheckBox.Checked;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.UpdateCaption(aDescription: String);
begin
  ProgressBar.Progress := 0;
  with GIFImage do begin
    Caption := 'GIFImage Viewer - ' + aDescription;

    ImageWidthLabel.Caption  := IntToStr(ImageWidth);
    ImageHeightLabel.Caption := IntToStr(ImageHeight);
    ImageColorsLabel.Caption := IntToStr(1 shl BitsPerPixel);

    ImageTransparentCheckBox.Checked := IsTransparent;
    ImageInterlacedCheckBox.Checked  := IsInterlaced;
    AnimatedCheckBox.Checked         := IsAnimated;
    FrameCountLabel.Caption          := IntToStr(NumFrames);
    LoopCountLabel.Caption           := IntToStr(NumIterations);

    with FrameSpinEdit do begin
      Enabled  := IsAnimated and (not Animate);
      MaxValue := NumFrames;
      Value    := 1;
    end;
    SpeedSpinEdit.Enabled := IsAnimated and Animate;
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.GIFChange(Sender: TObject);
const
  Disposals: array[TDisposalType] of String = ('Undefined',
                                               'Nothing',
                                               'Background',
                                               'Previous');
var
  FrameInfo: TFrameInfo;
begin
  if (not GIFImage.Animate) then begin
    FrameInfo := GIFImage.GetFrameInfo(GIFImage.CurrentFrame);
    try
      with FrameInfo do begin
        FrameWidthLabel.Caption          := IntToStr(iiWidth);
        FrameHeightLabel.Caption         := IntToStr(iiHeight);
        DisposalMethodLabel.Caption      := Disposals[iiDisposalMethod];
        FrameTransparentCheckBox.Checked := iiTransparent;
        FrameInterlacedCheckBox.Checked  := iiInterlaced;
      end;
    finally
      FrameInfo.iiImage.Free;
    end;
  end;
end;

{MWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMWMW}
procedure TMainForm.GIFProgress(Sender: TObject; const BytesProcessed,
  BytesToProcess: LongInt; PercentageProcessed: Integer;
  var KeepOnProcessing: Boolean);
begin
  if (PercentageProcessed = 100) then begin
    ProgressBar.Visible := False;
  end
  else begin
    ProgressBar.Visible  := True;
    ProgressBar.Progress := PercentageProcessed;
    KeepOnProcessing     := True;
  end;
end;

end.
