unit PolytrodeGUI;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ElectrodeTypes;

type
  TPolytrodeGUIForm = class(TForm)
    procedure FormMouseMove(Sender: TObject; Shift: TShiftState; X,
                            Y: Integer);
    procedure FormMouseDown(Sender: TObject; Button: TMouseButton;
                            Shift: TShiftState; X, Y: Integer);
    procedure FormMouseUp(Sender: TObject; Button: TMouseButton;
                          Shift: TShiftState; X, Y: Integer);
    procedure FormPaint(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
  private
    ebm : TBitmap;
    Electrode : TElectrode;
    xoff, yoff : integer;
    StartX, StartY: integer;
    procedure DrawElectrode;
    function MouseXY2Site(const X, Y : integer;
                          var Site : ShortInt) : boolean;
    { Private declarations }
  public
    LastChanSelect : shortint;
    function ChangeSiteColor(Site : ShortInt; Color : TColor) : boolean;
    function CreateElectrode(ElectrodeName : ShortString) : boolean;
    { Public declarations }
  end;

var
  PolytrodeGUIForm: TPolytrodeGUIForm;

const ELECTRODEBORDER = 100;

implementation

uses ElectroPlateMain; //damn! this is only here to pass mouse events site #

{$R *.DFM}

{-------------------------------------------------------------------------------}
procedure TPolytrodeGUIForm.FormCreate(Sender: TObject);
begin
  ebm := TBitmap.Create;
  ebm.PixelFormat := pf24bit;
end;

{-------------------------------------------------------------------------------}
function TPolytrodeGUIForm.CreateElectrode(ElectrodeName : ShortString) : boolean;
var maxx,minx,maxy,miny,i : integer;
begin
  if not GetElectrode(Electrode,ElectrodeName) then
  begin
    CreateElectrode := FALSE;
    ShowMessage(ElectrodeName+' is an unknown electrode');
    exit;
  end else
  With Electrode do
  begin
    if NumPoints > MAXELECTRODEPOINTS then NumPoints := MAXELECTRODEPOINTS;
    minx := 10000;
    miny := 10000;
    maxx := -10000;
    maxy := -10000;
    For i := 0 to NumSites-1 do
    begin
      if minx > SiteLoc[i].x then minx := SiteLoc[i].x;
      if miny > SiteLoc[i].y then miny := SiteLoc[i].y;
      if maxx < SiteLoc[i].x then maxx := SiteLoc[i].x;
      if maxy < SiteLoc[i].y then maxy := SiteLoc[i].y;
    end;
    TopLeftSite.x := minx;
    TopLeftSite.y := miny;
    BotRightSite.x := maxx;
    BotRightSite.y := maxy;

    minx := minx - ELECTRODEBORDER;
    miny := miny - ELECTRODEBORDER;
    maxx := maxx + ELECTRODEBORDER;
    maxy := maxy + ELECTRODEBORDER;
    For i := 0 to NumPoints-1 do
    begin
      if minx > Outline[i].x then minx := Outline[i].x;
      if miny > Outline[i].y then miny := Outline[i].y;
      if maxx < Outline[i].x then maxx := Outline[i].x;
      if maxy < Outline[i].y then maxy := Outline[i].y;
    end;
    Created := TRUE;
  end;

  ebm.Width := maxx-minx;
  ebm.Height :=maxy-miny;
  ClientWidth := ebm.Width;
  if ebm.Height>Screen.Height then ClientHeight:= Screen.Height-50
    else ClientHeight := ebm.Height;
  xoff := -minx;
  yoff := -miny-10;

  CreateElectrode := TRUE;
  DrawElectrode;
end;

{--------------------------------------------------------------------}
procedure TPolytrodeGUIForm.DrawElectrode;
var i : integer;
begin
  With Electrode do
  begin
    Ebm.Canvas.Brush.Color := clBlack;
    Ebm.Canvas.Font.Color := clDkGray;
    Ebm.Canvas.FillRect(ClientRect); //clear background
    Ebm.Canvas.Pen.Color := clDkGray;
    Ebm.Canvas.MoveTo(Outline[0].x+xoff, Outline[0].y+xoff);
    Ebm.Canvas.Pen.Color := RGB(48,64,48);
    For i := 0 to NumPoints-1 do //draw electrode outline
      Ebm.Canvas.LineTo(Outline[i].x+xoff, OutLine[i].y+yoff);
    Ebm.Canvas.Brush.Color := RGB(48,64,48); //next line is a patch to fix 'disappearing shank' bug
    if OutLine[0].y+yoff < 2 then Ebm.Canvas.FloodFill(Outline[0].x+xoff+1, 2, RGB(48,64,48),fsborder{fol.fillstyle})
      else Ebm.Canvas.FloodFill(Outline[0].x+xoff+1,Outline[0].y+yoff+1,RGB(48,64,48),fsborder{fol.fillstyle});
    Ebm.Canvas.Pen.Color := $005E879B;
    For i := 0 to NumSites-1 do //draw electrode sites
    begin
      if i = MUX2EIB[ActiveChanNumber]-1 then Ebm.Canvas.Brush.Color:= GUISiteCol //active site
        else Ebm.Canvas.Brush.Color := $005E879B;
      case roundsite of
        TRUE : Ebm.Canvas.ellipse(SiteLoc[i].x-SiteSize.x div 2+xoff,SiteLoc[i].y-SiteSize.y div 2+yoff,
                                  SiteLoc[i].x+SiteSize.x div 2+xoff,SiteLoc[i].y+SiteSize.y div 2+yoff);
        FALSE : Ebm.Canvas.framerect(rect(SiteLoc[i].x-SiteSize.x div 2+xoff,SiteLoc[i].y-SiteSize.y div 2+yoff,
                                  SiteLoc[i].x+SiteSize.x div 2+xoff,SiteLoc[i].y+SiteSize.y div 2+yoff));
      end;
      Ebm.Canvas.Brush.Color := RGB(48,64,48); //draw site numbers
      Ebm.Canvas.TextOut(SiteLoc[i].x+SiteSize.x div 2+xoff+1,SiteLoc[i].y+yoff,inttostr(i));
    end;
  end;
  Canvas.Draw(0,0,ebm); //copy ebm to canvas
end;

{-------------------------------------------------------------------------------}
procedure TPolytrodeGUIForm.FormMouseDown(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  if ssRight in Shift then
  begin
    StartX := X-xoff;
    StartY := Y-yoff;
    Screen.Cursor := crSizeAll;
  end else
  if ssLeft in Shift then  //map mouse cursor location to site #
  begin
    Screen.Cursor := crHelp;
    if not MouseXY2Site(X,Y, LastChanSelect) then Exit;
    EPlateMainForm.ChanSelect.Value:= SITE2MUX[LastChanSelect+1];
  end;
end;

{-------------------------------------------------------------------------------}
procedure TPolytrodeGUIForm.FormMouseUp(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  Screen.Cursor := crArrow;  //restore to normal pointer
end;

{-------------------------------------------------------------------------------}
procedure TPolytrodeGUIForm.FormMouseMove(Sender: TObject;
  Shift: TShiftState; X, Y: Integer);
begin
  if ssRight in Shift then //move electrode about canvas
  begin
    xoff:= X-StartX;
    yoff:= Y-StartY;
    // constrain drag...
    if yoff < -Electrode.BotRightSite.y then yoff:= -Electrode.BotRightSite.y
      else if yoff > ClientHeight - 100 then yoff:= ClientHeight - 100;
    if xoff < ELECTRODEBORDER then xoff:= ELECTRODEBORDER
      else if xoff > ClientWidth-ELECTRODEBORDER then xoff:= ClientWidth-ELECTRODEBORDER;
    DrawElectrode;
  end;
end;

{-------------------------------------------------------------------------------}
function TPolytrodeGUIForm.MouseXY2Site(const X, Y : integer;
                                        var Site : ShortInt) : boolean;
var i : integer;
begin
  With Electrode do
  begin
    for i := 0 to NumSites -1 do
    begin
      if  (X-xoff > (SiteLoc[i].x-Sitesize.x div 2))
      and (X-xoff < (SiteLoc[i].x+Sitesize.x div 2))
      and (Y-yoff > (SiteLoc[i].y-Sitesize.y div 2))
      and (Y-yoff < (SiteLoc[i].y+Sitesize.y div 2)) then
      begin
        Site:= i;
        Result:= true;
        Exit;
      end;
    end;
    Result:= false;
  end;
end;

{-------------------------------------------------------------------------------}
function TPolytrodeGUIForm.ChangeSiteColor(Site : ShortInt; Color : TColor) : boolean;
begin
  dec(Site); //zero-based site numbering at this stage
  if site < 0 then
  begin
    Result:= false;
    Exit;
  end;
  Result:= true;
  Ebm.Canvas.Brush.Color := Color;
  With Electrode do
  begin
    case roundsite of
      TRUE : Ebm.Canvas.ellipse(SiteLoc[site].x-SiteSize.x div 2+xoff,SiteLoc[site].y-SiteSize.y div 2+yoff,
                                SiteLoc[site].x+SiteSize.x div 2+xoff,SiteLoc[site].y+SiteSize.y div 2+yoff);
      FALSE : Ebm.Canvas.framerect(rect(SiteLoc[site].x-SiteSize.x div 2+xoff,SiteLoc[site].y-SiteSize.y div 2+yoff,
                                SiteLoc[site].x+SiteSize.x div 2+xoff,SiteLoc[site].y+SiteSize.y div 2+yoff));
  end;
  Canvas.Draw(0,0,ebm); //repaint electrode
  end;
end;

{-------------------------------------------------------------------------------}
procedure TPolytrodeGUIForm.FormPaint(Sender: TObject);
begin
  Canvas.Draw(0,0,ebm); //repaint electrode onto canvas
end;

{-------------------------------------------------------------------------------}
procedure TPolytrodeGUIForm.FormClose(Sender: TObject;
  var Action: TCloseAction);
begin
  GUICreated:= false;
  Ebm.Free;             //deallocate memory for electrode bitmap...
  SiteProperties:= nil; //...and site properties record(s)
  Action := caFree;
end;

end.
