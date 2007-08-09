unit PolytrodeGUI;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ElectrodeTypes;

type
  TPolyGUIForm = class(TForm)
  procedure FormShow(Sender: TObject);
  private
    Procedure DrawElectrode;
  public
    ebm : TBitmap;
  end;

var
  PolyGUIForm: TPolyGUIForm;
  Electrode : TElectrode;

implementation

uses ElectroPlateMain;  //Again, Phil didn't need this!!!

{--------------------------------------------------------------------}
procedure TPolyGUIForm.FormShow(Sender: TObject);
begin
  Showmessage('Got here');
  ebm := TBitmap.Create;
  ebm.PixelFormat := pf24bit;
  GetElectrode(Electrode, 'My Electrode');
end;

{--------------------------------------------------------------------}
Procedure TPolyGUIForm.DrawElectrode;
const PIby2 = PI/2;
var i,x1,y1,eoff : integer;
    plgpts : array of TPoint;
begin
  with Electrode do
  begin
    ebm.Canvas.FillRect(PolyGUIForm.clientRect);//clear electrode bm
    ebm.canvas.Pen.Color := clDkGray;
//    x1 := round(Rx(Outline[0].x+xoff,Outline[0].y+yoff+eoff,0));
//    y1 := Outline[0].y+yoff+eoff;
//    ebm.canvas.moveto(x1,y1); //draw outline
    SetLength(plgpts,NumPoints+1);
//    ebm.Canvas.MoveTo(round(Rx(Outline[0].x+xoff,Outline[0].y+yoff+eoff,0)),Outline[0].y+yoff+eoff);
    ebm.canvas.Pen.Color := RGB(48,64,48);
    For i := 0 to NumPoints-1 do
    begin
      plgpts[i].x := Outline[i].x;
      plgpts[i].y := Outline[i].y;
      ebm.Canvas.LineTo(plgpts[i].x,plgpts[i].y);
    end;
    plgpts[NumPoints].x := plgpts[0].x;
    plgpts[NumPoints].y := plgpts[0].y;
    ebm.Canvas.LineTo(plgpts[0].x,plgpts[0].y);

    ebm.canvas.Brush.Color := clBlack;
    ebm.canvas.Font.Color := clBlack;

    ebm.canvas.Pen.Color := $005E879B;
    ebm.canvas.Brush.Color := $005E879B;
    For i := 0 to NumSites-1 do  //draw the electrode sites
    begin
      ebm.canvas.Brush.Color := $005E879B;
      case roundsite of
        TRUE : ebm.canvas.ellipse(SiteLoc[i].x-SiteSize.x div 2,SiteLoc[i].y+eoff-SiteSize.y div 2,0,SiteLoc[i].y+eoff-SiteSize.y div 2);
                                  {SiteLoc[i].x+SiteSize.x div 2,SiteLoc[i].y+yoff+SiteSize.y div 2,0,SiteLoc[i].y+eoff+SiteSize.y div 2);}
        FALSE : ebm.canvas.framerect(rect(SiteLoc[i].x-SiteSize.x div 2,SiteLoc[i].y+eoff-SiteSize.y div 2,
                                          SiteLoc[i].x+SiteSize.x div 2,SiteLoc[i].y+eoff+SiteSize.y div 2));
      end;
      ebm.canvas.Brush.Color := RGB(48,64,48);
      //ebm.canvas.TextOut(SiteLoc[i].x+SiteSize.x div 2+1,SiteLoc[i].y+eoff,0,SiteLoc[i].y+yoff+eoff,inttostr(i));
    end;
    //ebm.canvas.TextOut(RX(ebm.width div 2-ebm.canvas.TextWidth(Name) div 2,0,0),0,Name);
  end;
  ebm.canvas.Brush.Color := clBlack;
end;

end.
