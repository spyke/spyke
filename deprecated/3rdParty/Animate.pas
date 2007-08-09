{*************************************************}
{                                                 }
{ Delphi VCL Extensions (RX)                      }
{                                                 }
{ Copyright (c) 1995, 1996 AO ROSNO               }
{ Copyright (c) 1997 Master-Bank                  }
{                                                 }
{*************************************************}
{                                                 }
{ Improved (C)1997 by Marcin Qfel Zaleski         }
{                                                 }
{ 1. Changed private procedure SetGlyphNum        }
{    (Now setting Glyph>=NumGlyphs cause Glyph:=0 }
{     setting Glyph<0 cause Glyph:=NumGlyphs-1    }
{                                                 }
{ 2. Added OnGlyphNumChange event                 }
{                                                 }
{*************************************************}
unit Animate;

interface

{$I RX.INC}

uses Messages, {$IFDEF WIN32} Windows, {$ELSE} WinTypes, WinProcs, {$ENDIF}
  SysUtils, Classes, Graphics, Controls, Forms, StdCtrls, Menus, RxTimer;

type

  TGlyphOrientation = (goHorizontal, goVertical);

  TGlyphNumChangeEvent = procedure(Sender:TObject;const Frame:Integer) of object;

{ TAnimatedImage }

  TAnimatedImage = class(TGraphicControl)
  private
    { Private declarations }
    FActive: Boolean;
    FAutoSize: Boolean;
    FGlyph: TBitmap;
    FImageWidth: Integer;
    FImageHeight: Integer;
    FInactiveGlyph: Integer;
    FOrientation: TGlyphOrientation;
    FTimer: TRxTimer;
    FNumGlyphs: Integer;
    FGlyphNum: Integer;
    FStretch: Boolean;
    FTransparentColor: TColor;
    FOpaque: Boolean;
    FTimerReapint: Boolean;
    FOnGlyphNumChange : TGlyphNumChangeEvent;
    FOnStart: TNotifyEvent;
    FOnStop: TNotifyEvent;
    procedure DefineBitmapSize;
    procedure ResetImageBounds;
    procedure AdjustBounds;
    function GetInterval: Cardinal;
    procedure SetAutoSize(Value: Boolean);
    procedure SetInterval(Value: Cardinal);
    procedure SetActive(Value: Boolean);
    procedure SetOrientation(Value: TGlyphOrientation);
    procedure SetGlyph(Value: TBitmap);
    procedure SetGlyphNum(Value: Integer);
    procedure SetInactiveGlyph(Value: Integer);
    procedure SetNumGlyphs(Value: Integer);
    procedure SetStretch(Value: Boolean);
    procedure SetTransparentColor(Value: TColor);
    procedure SetOpaque(Value: Boolean);
    procedure ImageChanged(Sender: TObject);
    procedure UpdateInactive;
    procedure TimerExpired(Sender: TObject);
    procedure DrawGlyph;
    procedure WMSize(var Message: TWMSize); message WM_SIZE;
  protected
    { Protected declarations }
    procedure Loaded; override;
    procedure Paint; override;
    procedure Start; dynamic;
    procedure Stop; dynamic;
  public
    { Public declarations }
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
  published
    { Published declarations }
    property Active: Boolean read FActive write SetActive;
    property Align;
    property AutoSize: Boolean read FAutoSize write SetAutoSize default True;
    property Orientation: TGlyphOrientation read FOrientation write SetOrientation
      default goHorizontal;
    property Glyph: TBitmap read FGlyph write SetGlyph;
    property GlyphNum: Integer read FGlyphNum write SetGlyphNum default 0;
    property Interval: Cardinal read GetInterval write SetInterval default 100;
    property NumGlyphs: Integer read FNumGlyphs write SetNumGlyphs default 1;
    property InactiveGlyph: Integer read FInactiveGlyph write SetInactiveGlyph default -1;
    property TransparentColor: TColor read FTransparentColor write SetTransparentColor
      default clOlive;
    property Opaque: Boolean read FOpaque write SetOpaque default False;
    property Color;
    property Cursor;
    property DragCursor;
    property DragMode;
    property ParentColor default True;
    property ParentShowHint;
    property PopupMenu;
    property ShowHint;
    property Stretch: Boolean read FStretch write SetStretch default True;
    property Visible;
    property OnClick;
    property OnDblClick;
    property OnMouseMove;
    property OnMouseDown;
    property OnMouseUp;
    property OnDragOver;
    property OnDragDrop;
    property OnEndDrag;
{$IFDEF WIN32}
    property OnStartDrag;
{$ENDIF}
    property OnGlyphNumChange:TGlyphNumChangeEvent read FOnGlyphNumChange write FOnGlyphNumChange;
    property OnStart: TNotifyEvent read FOnStart write FOnStart;
    property OnStop: TNotifyEvent read FOnStop write FOnStop;
  end;

implementation

uses RxConst, VCLUtils;

{ TAnimatedImage }

constructor TAnimatedImage.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  ControlStyle := [csClickEvents, csCaptureMouse, csOpaque, csDoubleClicks];
  FTimer := TRxTimer.Create(Self);
  Interval := 100;
  FGlyph := TBitmap.Create;
  FGlyph.OnChange := ImageChanged;
  FGlyphNum := 0;
  FNumGlyphs := 1;
  FInactiveGlyph := -1;
  FTransparentColor := clOlive;
  FOrientation := goHorizontal;
  FAutoSize := True;
  FStretch := True;
  Width := 32;
  Height := 32;
  ParentColor := True;
end;

destructor TAnimatedImage.Destroy;
begin
  Active := False;
  FGlyph.OnChange := nil;
  FGlyph.Free;
  inherited Destroy;
end;

procedure TAnimatedImage.Loaded;
begin
  inherited Loaded;
  ResetImageBounds;
  UpdateInactive;
end;

procedure TAnimatedImage.ImageChanged(Sender: TObject);
{$IFDEF RX_D3}
var
  ParentForm: TCustomForm;
{$ENDIF}
begin
  FTransparentColor := FGlyph.TransparentColor and not TransparentMask;
  DefineBitmapSize;
  AdjustBounds;
{$IFDEF RX_D3}
  if Visible and (not (csLoading in ComponentState)) and FGlyph.PaletteModified then
  begin
    ParentForm := GetParentForm(Self);
    if Assigned(ParentForm) and ParentForm.Active and Parentform.HandleAllocated then
      PostMessage(ParentForm.Handle, WM_QUERYNEWPALETTE, 0, 0);
    FGlyph.PaletteModified := False;
  end;
{$ENDIF}
  Invalidate;
end;

procedure TAnimatedImage.UpdateInactive;
begin
  if (not Active) and (FInactiveGlyph >= 0) and
    (FInactiveGlyph < FNumGlyphs) then FGlyphNum := FInactiveGlyph;
end;

procedure TAnimatedImage.SetOpaque(Value: Boolean);
begin
  if Value <> FOpaque then begin
    FOpaque := Value;
    Invalidate;
  end;
end;

procedure TAnimatedImage.SetTransparentColor(Value: TColor);
begin
  if Value <> TransparentColor then begin
    FTransparentColor := Value;
    Invalidate;
  end;
end;

procedure TAnimatedImage.SetOrientation(Value: TGlyphOrientation);
begin
  if FOrientation <> Value then begin
    FOrientation := Value;
    DefineBitmapSize;
    AdjustBounds;
    Invalidate;
  end;
end;

procedure TAnimatedImage.SetGlyph(Value: TBitmap);
begin
  FGlyph.Assign(Value);
end;

procedure TAnimatedImage.SetStretch(Value: Boolean);
begin
  if Value <> FStretch then begin
    FStretch := Value;
    if Active then Repaint
    else Invalidate;
  end;
end;

procedure TAnimatedImage.SetGlyphNum(Value: Integer);
begin
  if Value<>FGlyphNum then
    begin
      if Value in [0..FNumGlyphs-1] then
        FGlyphNum := Value
      else if Value<0 then
        FGlyphNum := FNumGlyphs-1
      else //must be >=FNumGlyphs
        FGlyphNum := 0;
      if Assigned(FOnGlyphNumChange) then
        FOnGlyphNumChange(Self,FGlyphNum);
      UpdateInactive;
      Invalidate;
    end;
end;

procedure TAnimatedImage.SetInactiveGlyph(Value: Integer);
begin
  if Value < 0 then Value := -1;
  if Value <> FInactiveGlyph then begin
    if (Value < FNumGlyphs) or (csLoading in ComponentState) then begin
      FInactiveGlyph := Value;
      UpdateInactive;
      Invalidate;
    end;
  end;
end;

procedure TAnimatedImage.SetNumGlyphs(Value: Integer);
begin
  FNumGlyphs := Value;
  if FInactiveGlyph >= FNumGlyphs then begin
    FInactiveGlyph := -1;
    FGlyphNum := 0;
  end else UpdateInactive;
  ResetImageBounds;
  AdjustBounds;
  Invalidate;
end;

procedure TAnimatedImage.DefineBitmapSize;
begin
  FNumGlyphs := 1;
  FGlyphNum := 0;
  FImageWidth := 0;
  FImageHeight := 0;
  if (FOrientation = goHorizontal) and (FGlyph.Height > 0) and
    (FGlyph.Width mod FGlyph.Height = 0) then
      FNumGlyphs := FGlyph.Width div FGlyph.Height
  else if (FOrientation = goVertical) and (FGlyph.Width > 0) and
    (FGlyph.Height mod FGlyph.Width = 0) then
      FNumGlyphs := FGlyph.Height div FGlyph.Width;
  ResetImageBounds;
end;

procedure TAnimatedImage.ResetImageBounds;
begin
  if FNumGlyphs < 1 then FNumGlyphs := 1;
  if FOrientation = goHorizontal then begin
    FImageHeight := FGlyph.Height;
    FImageWidth := FGlyph.Width div FNumGlyphs;
  end
  else {if Orientation = goVertical then} begin
    FImageWidth := FGlyph.Width;
    FImageHeight := FGlyph.Height div FNumGlyphs;
  end;
end;

procedure TAnimatedImage.AdjustBounds;
begin
  if not (csReading in ComponentState) then begin
    if FAutoSize and (FImageWidth > 0) and (FImageHeight > 0) then
      SetBounds(Left, Top, FImageWidth, FImageHeight);
  end;
end;

type
  TParentControl = class(TWinControl);

procedure TAnimatedImage.DrawGlyph;
var
  TmpImage: TBitmap;
  BmpIndex: Integer;
  SrcRect: TRect;
begin
  if (not Active) and (FInactiveGlyph >= 0) and
    (FInactiveGlyph < FNumGlyphs) then BmpIndex := FInactiveGlyph
  else BmpIndex := FGlyphNum;
  TmpImage := TBitmap.Create;
  try
    with TmpImage do begin
      Width := ClientWidth;
      Height := ClientHeight;
      if (not FOpaque) and (Self.Parent <> nil) then
        Canvas.Brush.Color := TParentControl(Self.Parent).Color
      else Canvas.Brush.Color := Self.Color;
      Canvas.FillRect(Bounds(0, 0, Width, Height));
      { copy image from parent and back-level controls }
      if not FOpaque then CopyParentImage(Self, Canvas);
      if (FImageWidth > 0) and (FImageHeight> 0) then begin
        if Orientation = goHorizontal then
          SrcRect := Bounds(BmpIndex * FImageWidth, 0, FImageWidth, FImageHeight)
        else {if Orientation = goVertical then}
          SrcRect := Bounds(0, BmpIndex * FImageHeight, FImageWidth, FImageHeight);
        if FStretch then
          StretchBitmapRectTransparent(Canvas, 0, 0, Width, Height, SrcRect,
            FGlyph, FTransparentColor)
        else
          DrawBitmapRectTransparent(Canvas, 0, 0, SrcRect, FGlyph,
            FTransparentColor);
      end;
    end;
    Canvas.Draw(ClientRect.Left, ClientRect.Top, TmpImage);
  finally
    TmpImage.Free;
  end;
end;

procedure TAnimatedImage.Paint;
begin
  DrawGlyph;
  if (csDesigning in ComponentState) then
    with Canvas do begin
      Pen.Style := psDash;
      Brush.Style := bsClear;
      Rectangle(0, 0, Width, Height);
    end;
end;

procedure TAnimatedImage.TimerExpired(Sender: TObject);
begin
  if Visible then begin
    if FGlyphNum < (FNumGlyphs - 1) then Inc(FGlyphNum)
    else FGlyphNum := 0;
    if (FGlyphNum = FInactiveGlyph) and (FNumGlyphs > 1) then begin
      if FGlyphNum < (FNumGlyphs - 1) then Inc(FGlyphNum)
      else FGlyphNum := 0;
    end;
    FTimerReapint := True;
    if Assigned(FOnGlyphNumChange) then
      FOnGlyphNumChange(Self,FGlyphNum);
    try
      Repaint;
    finally
      FTimerReapint := False;
    end;
  end;
end;

procedure TAnimatedImage.Stop;
begin
  if not ((csDestroying in ComponentState) or (csReading in ComponentState)) then
    if Assigned(FOnStop) then FOnStop(Self);
end;

procedure TAnimatedImage.Start;
begin
  if not ((csDestroying in ComponentState) or (csReading in ComponentState)) then
    if Assigned(FOnStart) then FOnStart(Self);
end;

procedure TAnimatedImage.SetAutoSize(Value: Boolean);
begin
  if Value <> FAutoSize then begin
    FAutoSize := Value;
    AdjustBounds;
    Invalidate;
  end;
end;

procedure TAnimatedImage.SetInterval(Value: Cardinal);
begin
  FTimer.Interval := Value;
end;

function TAnimatedImage.GetInterval: Cardinal;
begin
  Result := FTimer.Interval;
end;

procedure TAnimatedImage.SetActive(Value: Boolean);
begin
  if FActive<>Value then
    begin
      if Value then
        begin
          FTimer.OnTimer := TimerExpired;
          FTimer.Enabled := True;
          FActive := FTimer.Enabled;
          Start;
        end
      else
        begin
          FTimer.Enabled := False;
          FTimer.OnTimer := nil;
          FActive := False;
          UpdateInactive;
          Stop;
          Invalidate;
        end;
    end;
end;

procedure TAnimatedImage.WMSize(var Message: TWMSize);
begin
  inherited;
  AdjustBounds;
end;

end.
