{ ****************************************************************
  Info               :  TLabel2000X
                        Freeware

  Source File Name   :  X2000LA.PAS
  Autho / Modified   :  Baldemaier Florian(Baldemaier.Florian@gmx.net)
  Original made by   :  Jörg Lingner (jlingner@t-online.de) 
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Decription         :  Rotatet Label with Raised function.

  Testet on          :  Intel Pentium II, 300 Mhz and 450 Mhz, Windows 98
                        Intel Pentium III, 500 Mhz, Windows 98
                        Intel Pentium 200 Mhz, Windows98
                        D4 Service Pack 3
                        Mircosoft Windows 98
                        Mircosoft Windows 98 SE
****************************************************************   }
unit x2000la;
interface

uses

  WinProcs, Wintypes, Messages, SysUtils, Classes, Graphics, Controls, Forms,
  Dialogs, StdCtrls, Menus, ShellApi,DsgnIntf;

type
  TTextStyle = (tsNone,tsRaised,tsRecessed);

  TAbout2000X=class(TPropertyEditor)
  public
    procedure Edit; override;
    function GetAttributes: TPropertyAttributes; override;
    function GetValue: string; override;
  end;

  TLabel2000X = class(TCustomLabel)
   private
    fAbout      : TAbout2000X;
    fEscapement : Integer;
    fTextStyle  : TTextStyle;
    FDate       : boolean;
    FDateTime   : boolean;
    FUrl        : boolean;
    FEmail      : boolean;
    procedure   SetEscapement(aVal:Integer);
    procedure   SetTextStyle (aVal:TTextStyle);
    procedure   CalcTextPos(var aRect:TRect;aAngle:Integer;aTxt:String);
    procedure   DrawAngleText(aCanvas:TCanvas;aRect:TRect;aAngle:Integer;aTxt:String);
   protected
    Procedure SetIt1(value: boolean);
    Procedure SetIt2(value: boolean);
    Procedure SetIt3(value: boolean);
    Procedure SetIt4(value: boolean);
    procedure MouseDown(Button : TMouseButton; Shift : TShiftState; X, Y : Integer); OVERRIDE;
    procedure Paint; override;
    procedure DoDrawText(var Rect:TRect;Flags:Word);
   public
    constructor Create(AOwner: TComponent); override;
    procedure Refresh;
   published
    property About             : TAbout2000X read FAbout             write FAbout;

    property ShowAsDate     : boolean read FDate     write SetIt1   default false;
    property ShowAsDateTime : boolean read FDateTime write SetIt2   default false;
    property ShowAsUrl      : boolean read FUrl      write SetIt3   default false;
    property ShowAsEmailUrl : boolean read FEmail    write SetIt4   default false;

    property Angle: Integer    read fEscapement write SetEscapement;
    property TextStyle : TTextStyle read fTextStyle  write SetTextStyle;
    property Align;
    property Alignment;
    property AutoSize;
    property Caption;
    property Color;
    property DragCursor;
    property DragMode;
    property Enabled;
    property FocusControl;
    property Font;
    property ParentColor;
    property ParentFont;
    property ParentShowHint;
    property PopupMenu;
    property ShowAccelChar;
    property ShowHint;
    property Transparent;
    property Visible;
    property WordWrap;
    property OnClick;
    property OnDblClick;
    property OnDragDrop;
    property OnDragOver;
    property OnEndDrag;
    property OnMouseDown;
    property OnMouseMove;
    property OnMouseUp;
    {$IFDEF WIN32}
    property OnStartDrag;
    {$ENDIF}
  end;

procedure Register;

implementation

uses x2000about;

var
   TempDate, TempDateTime: string;
   TempColor: TColor;
   TempStyle: TFontStyles;

procedure Register;
begin
  RegisterComponents('X2000', [TLabel2000X]);
end;

constructor TLabel2000X.Create(aOwner:TComponent);
begin
  inherited Create(aOwner);

  fEscapement:= 0;
  fTextStyle := tsRaised;
  Font.Name := 'Arial';

end;

procedure TLabel2000X.MouseDown(Button : TMouseButton; Shift : TShiftState;
      X, Y : Integer);
begin
 INHERITED MouseDown(Button, Shift, X, Y);
 if FUrl then begin
   ShellExecute(ValidParentForm(Self).Handle,'open',PChar(caption),
               NIL,NIL,SW_SHOWNORMAL);
 end;
 if FEmail then begin
   ShellExecute(ValidParentForm(Self).Handle,'open',PChar('mailto:'+caption),
               NIL,NIL,SW_SHOWNORMAL);
 end;
end;

procedure TLabel2000X.Refresh;
var h: TDateTime;
begin
  h:=now;
  if FDateTime then caption:=datetimetostr(h);
  if FDate     then caption:=datetostr(h);
end;

Procedure TLabel2000X.SetIt4(value: boolean);
begin
  if FEmail or FUrl then begin
     Font.Color:=TempColor;
     Font.Style:=TempStyle;
     Cursor:=crdefault;
  end;

  FDateTime:=false;
  FDate    :=false;
  FUrl     :=false;

  if FEmail then begin
     FEmail:=false;
     Font.Color:=TempColor;
     Font.Style:=TempStyle;
     Cursor:=crdefault;
     exit;
  end;
  if not FEmail then begin
     FEmail:=true;
     TempColor:=Font.Color;
     TempStyle:=Font.Style;
     Font.Color := clBlue;
     Font.Style := [fsUnderline];
     Cursor:=crhandpoint;
     exit;
  end;
end;

Procedure TLabel2000X.SetIt3(value: boolean);
begin
  if FEmail or FUrl then begin
     Font.Color:=TempColor;
     Font.Style:=TempStyle;
     Cursor:=crdefault;
  end;

  FDateTime:=false;
  FDate    :=false;
  FEmail   :=false;

  if FUrl then begin
     FUrl:=false;
     Font.Color:=TempColor;
     Font.Style:=TempStyle;
     Cursor:=crdefault;
     exit;
  end;
  if not FUrl then begin
     FUrl:=true;
     TempColor:=Font.Color;
     TempStyle:=Font.Style;
     Font.Color := clBlue;
     Font.Style := [fsUnderline];
     Cursor:=crhandpoint;
     exit;
  end;
end;

Procedure TLabel2000X.SetIt1(value: boolean);
begin
  if FEmail or FUrl then begin
     Font.Color:=TempColor;
     Font.Style:=TempStyle;
     Cursor:=crdefault;
  end;

  FDateTime:=false;
  FUrl     :=false;
  FEmail   :=false;

  if FDate then begin
     FDate:=false;
     caption:=tempdate;
     exit;
  end;
  if not FDate then begin
     FDate:=true;
     TempDate:=caption;
     caption:=datetostr(now);
     exit;
  end;
end;

Procedure TLabel2000X.SetIt2(value: boolean);
begin
  if FEmail or FUrl then begin
     Font.Color:=TempColor;
     Font.Style:=TempStyle;
     Cursor:=crdefault;
  end;

  FDate:=false;
  FUrl :=false;
  FEmail   :=false;

  if FDatetime then begin
     FDatetime:=false;
     caption:=tempdatetime;
     exit;
  end;
  if not FDatetime then begin
     FDateTime:=True;
     TempDateTime:=caption;
     caption:=datetimetostr(now);
     exit;
  end;
end;

procedure TLabel2000X.SetEscapement(aVal:Integer);
begin
  if fEscapement <> aVal then begin
     if aVal < 0 then begin
        while aVal < -360 do aVal := aVal + 360;
        aVal := 360 + aVal;
     end;
     while aVal > 360 do aVal := aVal - 360;
     fEscapement := aVal;
     Invalidate;
  end;
end;

procedure TLabel2000X.SetTextStyle(aVal:TTextStyle);
begin
  if fTextStyle <> aVal then begin
     fTextStyle := aVal;
     Invalidate;
  end;
end;

procedure TLabel2000X.Paint;
const
  Alignments: array[TAlignment] of Word = (DT_LEFT,DT_RIGHT,DT_CENTER);
  WordWraps : array[Boolean] of Word = (0,DT_WORDBREAK);
var
  Rect: TRect;
begin
  with Canvas do begin
    if not Transparent then begin
      Brush.Color := Self.Color;
      Brush.Style := bsSolid;
      FillRect(ClientRect);
    end;
    Brush.Style := bsClear;
    Rect := ClientRect;
    DoDrawText(Rect,DT_EXPANDTABS or WordWraps[WordWrap] or Alignments[Alignment]);
  end;
end;

procedure TLabel2000X.CalcTextPos(var aRect:TRect;aAngle:Integer;aTxt:String);
var DC      : HDC;
    hSavFont: HFont;
    Size    : TSize;
    x,y     : Integer;
    cStr    : array[0..255] of Char;

begin
  StrPCopy(cStr,aTxt);
  DC := GetDC(0);
  hSavFont := SelectObject(DC,Font.Handle);
  {$IFDEF WIN32}
  GetTextExtentPoint32(DC,cStr,Length(aTxt),Size);
  {$ELSE}
  GetTextExtentPoint(DC,cStr,Length(aTxt),Size);
  {$ENDIF}
  SelectObject  (DC,hSavFont);
  ReleaseDC(0,DC);

  if          aAngle<=90  then begin             { 1.Quadrant }
     x := 0;
     y := Trunc(Size.cx * sin(aAngle*Pi/180));
  end else if aAngle<=180 then begin             { 2.Quadrant }
     x := Trunc(Size.cx * -cos(aAngle*Pi/180));
     y := Trunc(Size.cx *  sin(aAngle*Pi/180) + Size.cy * cos((180-aAngle)*Pi/180));
  end else if aAngle<=270 then begin             { 3.Quadrant }
     x := Trunc(Size.cx * -cos(aAngle*Pi/180) + Size.cy * sin((aAngle-180)*Pi/180));
     y := Trunc(Size.cy * sin((270-aAngle)*Pi/180));
  end else if aAngle<=360 then begin             { 4.Quadrant }
     x := Trunc(Size.cy * sin((360-aAngle)*Pi/180));
     y := 0;
  end;
  aRect.Top := aRect.Top +y;
  aRect.Left:= aRect.Left+x;

  x := Abs(Trunc(Size.cx * cos(aAngle*Pi/180))) + Abs(Trunc(Size.cy * sin(aAngle*Pi/180)));
  y := Abs(Trunc(Size.cx * sin(aAngle*Pi/180))) + Abs(Trunc(Size.cy * cos(aAngle*Pi/180)));

  if Autosize then begin
     Width  := x;
     Height := y;
  end else if Alignment = taCenter then begin
     aRect.Left:= aRect.Left + ((Width-x) div 2);
  end else if Alignment = taRightJustify then begin
     aRect.Left:= aRect.Left + Width - x;
  end;
end;

procedure TLabel2000X.DrawAngleText(aCanvas:TCanvas;aRect:tRect;aAngle:Integer;aTxt:String);
var LFont             : TLogFont;
    hOldFont, hNewFont: HFont;
begin
  CalcTextPos(aRect,aAngle,aTxt);

  GetObject(aCanvas.Font.Handle,SizeOf(LFont),Addr(LFont));
  LFont.lfEscapement := aAngle*10;
  hNewFont := CreateFontIndirect(LFont);
  hOldFont := SelectObject(aCanvas.Handle,hNewFont);

  aCanvas.TextOut(aRect.Left,aRect.Top,aTxt);

  hNewFont := SelectObject(aCanvas.Handle,hOldFont);
  DeleteObject(hNewFont);
end;

procedure TLabel2000X.DoDrawText(var Rect:TRect;Flags:Word);
var Text        : String;
    TmpRect     : TRect;
    UpperColor  : TColor;
    LowerColor  : TColor;
    {$IFDEF WINDOWS}
    cStr        : array[0..255] of Char;
    {$ENDIF}
  begin
    Text := Caption;
    {$IFDEF WINDOWS}
    StrPCopy(cStr,Text);
    {$ENDIF}

    if (Flags and DT_CALCRECT <> 0) and ((Text = '') or ShowAccelChar and
    (Text[1] = '&') and (Text[2] = #0)) then Text := Text + ' ';

    if not ShowAccelChar then Flags := Flags or DT_NOPREFIX;
    Canvas.Font := Font;

    UpperColor := clBtnHighlight;
    LowerColor := clBtnShadow;

    if FTextStyle = tsRecessed then begin
      UpperColor := clBtnShadow;
      LowerColor := clBtnHighlight;
    end;

    if FTextStyle in [tsRecessed,tsRaised] then begin
      TmpRect := Rect;
      OffsetRect(TmpRect,1,1);
      Canvas.Font.Color := LowerColor;
      if fEscapement <> 0 then DrawAngleText(Canvas,TmpRect,fEscapement,Text)
      {$IFDEF WIN32}
      else DrawText(Canvas.Handle,pChar(Text),Length(Text),TmpRect,Flags);
      {$ELSE}
      else DrawText(Canvas.Handle,cStr,Length(Text),TmpRect,Flags);
      {$ENDIF}

      TmpRect := Rect;
      OffsetRect(TmpRect,-1,-1);
      Canvas.Font.Color := UpperColor;
      if fEscapement <> 0 then DrawAngleText(Canvas,TmpRect,fEscapement,Text)
      {$IFDEF WIN32}
      else DrawText(Canvas.Handle,pChar(Text),Length(Text),TmpRect,Flags);
      {$ELSE}
      else DrawText(Canvas.Handle,cStr,Length(Text),TmpRect,Flags);
      {$ENDIF}
    end;

    Canvas.Font.Color := Font.Color;

    if not Enabled then Canvas.Font.Color := clGrayText;

    if fEscapement <> 0 then DrawAngleText(Canvas,Rect,fEscapement,Text)
    {$IFDEF WIN32}
    else DrawText(Canvas.Handle,pChar(Text),Length(Text),Rect,Flags);
    {$ELSE}
    else DrawText(Canvas.Handle,cStr,Length(Text),Rect,Flags);
    {$ENDIF}
end;

procedure TAbout2000X.Edit;
begin
 with TAboutForm.Create(Application) do begin
  try
    ShowModal;
  finally
    Free;
  end;
 end;
end;

function TAbout2000X.GetAttributes: TPropertyAttributes;
begin
    Result := [paMultiSelect, paDialog, paReadOnly];
end;

function TAbout2000X.GetValue: string;
begin
    Result := '(X2000)';
end;

end.
