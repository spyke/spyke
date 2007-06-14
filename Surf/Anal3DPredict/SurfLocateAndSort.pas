unit SurfLocateAndSort;
{ Currently working on:

if not good, let cluster cluster variance = mindist*2
merge of clusters should depend on cluster variance
adding a spike to a cluster should depend on cluster variance

}
interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, ComCtrls, StdCtrls, SurfPublicTypes, ElectrodeTypes, NumRecipies,
  Spin, Math;

const ELECTRODEBORDER = 150;
      MAXDIST = 200;  //max dist a source be away from the nearest electrode
      MAXRADIUS = 600;//max dist a source be away from the furthest electrode
      MAXVo = 1000000;
      MAXM = 500;
      MAXO = 200;

      V2uV = 1000000;//1 million (uV -> V)
      PIby180 = PI/180;
type
  TVoltage = array of single;

  TSpikeLoc = record
    x,y,z,Vo,m,o : single;
    Voltage,FitVoltage : TVoltage;
    FitError : single;
    FitIterations : integer;
    ClusterId : Smallint;
    done : boolean;
    ErValuesForZ : array[0..10] of single;
  end;

(*  LocEstLevMarqObj = class(LevMarqObj)
    public
      Procedure CheckBounds(var a : DArrayNP); override;
      Procedure func(xx : double; var a : DArrayNP; var yfit : double;
               var dyda : DarrayNP;  ma,mfit : integer; computedyda : boolean); override;
      //Procedure CallOut(var well : wellrec; var iter,ma,mfit : integer; var a : DArrayNP; chisq,chidiff : double; wellhist : wellhisttype; final : boolean); {override;}
  end;
*)
  CLocSimplex = class(CSimplex)
    public
      Function func(var pr : DArrayNP) : double; override;
      Procedure callout(var pr : DArrayNP); override;
  end;

  TStats = record
    mean,vari,sum,sum2 : single;
  end;

  TCluster = record
    n : integer;
    good : boolean;
    x,y,z : TStats;
    Vo,Ro,m : TStats;
    Voltage,FitVoltage : array of TStats;
    FitError,FitIterations : TStats;
    SpikeList : array of integer;
  end;

  TLocSortForm = class(TForm)
    StatusBar: TStatusBar;
    Timer: TTimer;
    Panel1: TPanel;
    BSave: TButton;
    BWag: TButton;
    SpinDegrees: TSpinEdit;
    Label1: TLabel;
    CShowElectrode: TCheckBox;
    SpinMaxAngle: TSpinEdit;
    Label2: TLabel;
    Label3: TLabel;
    ShowFitPlot: TCheckBox;
    CBRender: TCheckBox;
    CShowUnClustered: TCheckBox;
    Label4: TLabel;
    lnumclusters: TLabel;
    CShowClusterStats: TCheckBox;
    EElectrodeOffset: TEdit;
    Label5: TLabel;
    CShowClusterLocation: TCheckBox;
    procedure TimerTimer(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure FormHide(Sender: TObject);
    procedure BSaveClick(Sender: TObject);
    procedure BWagClick(Sender: TObject);
  private
    { Private declarations }
    theta : single;
    thetasign : integer; // +/- 1
    Electrode : TElectrode;
    xoff,yoff,ADGain,ExtGain,LoopCount : integer;
    //LocEstLevMarq : LocEstLevMarqObj;
    LocSimplex :   CLocSimplex;
    Saving : boolean;
    FuncNum : integer;
    MeanZ,MeanVo,MeanM,MeanRo,MeanEr : single;
    NSamplesForStats : integer;
    Cluster : array of TCluster;
    NumGoodClusters : integer;
    Function  Rx(x,y,z : single) : single;//the three d to 2d transform
    Procedure DrawElectrode;
    Procedure ComputeXYZ(index : integer; lz,lvo,lm,lo : boolean; var chi : single);
    Procedure TransForm(x,y,z : single; var tx,ty : integer);//the three d to 2d transform
    Procedure SortSpike(index : integer; ReAssignAll : boolean);
  public
    { Public declarations }
    Wagging : Boolean;
    lsbm,ebm : TBitmap;
    SpikeLoc : array of TSpikeLoc;
    Procedure Render;
    Function CreateElectrode(ElectrodeName : ShortString;
                             a2dgain,exgain : Integer) : boolean;
    Procedure ComputeLoc(var Spike : TSpike; z,vo,m,o : single;
                         lz,lvo,lm,lo : boolean;
                         FunctionNum : integer; var chi : single);
    Procedure Finish;
  end;

var
  LocSortForm: TLocSortForm;
  VInfinite : single;

implementation

{$R *.DFM}

{--------------------------------------------------------------------}
procedure TLocSortForm.TimerTimer(Sender: TObject);
var ThetaStep,MaxTheta : Single;

begin
  if wagging then
  begin
    thetastep := SpinDegrees.Value * PIby180;
    maxtheta := SpinMaxAngle.Value * PIby180;

    theta := theta + thetasign * thetastep;
    if theta > maxtheta then
    begin
      theta := maxtheta;
      thetasign := -thetasign; //reverse
      inc(LoopCount);
    end;
    if theta < -maxtheta then
    begin
      theta := -maxtheta;
      thetasign := -thetasign; //reverse
      inc(LoopCount);
    end;
  end;
  if cbrender.checked then Render;
end;

{--------------------------------------------------------------------}
procedure TLocSortForm.FormShow(Sender: TObject);
begin
  lsbm := TBitmap.Create;
  lsbm.PixelFormat := pf24bit;

  ebm := TBitmap.Create;
  ebm.PixelFormat := pf24bit;

  Wagging := FALSE;
  Saving := FALSE;
  theta := 0;
  LoopCount := 0;
  Timer.Enabled := FALSE;
  thetasign := 1;
  Electrode.Created := FALSE;
  LocSimplex := CLocSimplex.Create;
end;

{--------------------------------------------------------------------}
procedure TLocSortForm.FormHide(Sender: TObject);
var i : integer;
begin
  timer.enabled :=FAlSE;
  wagging := false;
  LocSortForm.lsbm.Free;
  LocSortForm.ebm.Free;
  For i := 0 to length(SpikeLoc)-1 do
  begin
    SpikeLoc[i].Voltage := nil;
    SpikeLoc[i].FitVoltage := nil;
  end; 
  SpikeLoc := nil;
  LocSimplex.Free;
end;

procedure TLocSortForm.BSaveClick(Sender: TObject);
begin
  Saving := TRUE;
  LoopCount := 0;
end;

procedure TLocSortForm.BWagClick(Sender: TObject);
begin
  If Wagging then
  begin
    Wagging := FALSE;
    BWag.Caption := 'Wag';
  end else
  begin
    Wagging := TRUE;
    BWag.Caption := 'Stop';
  end;
end;

{--------------------------------------------------------------------}
Function TLocSortForm.CreateElectrode(ElectrodeName : ShortString;
                                     a2dgain,exgain : Integer) : boolean;
var maxx,minx,maxy,miny,i : integer;
begin
  ADGain := a2dgain;
  ExtGain := ExGain;

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

    ebm.Width := maxx-minx;
    ebm.Height := maxy-miny;
    lsbm.Width := ebm.width;
    lsbm.Height := ebm.height;
    LocSortForm.ClientWidth := ebm.width;
    LocSortForm.ClientHeight := ebm.height;

    xoff := -minx;
    yoff := -miny;

    ebm.canvas.Brush.Color := clBlack;
    lsbm.Canvas.Brush.Color := clBlack;
    ebm.canvas.font.color := clDkGray;

    Created := TRUE;
  end;

  For i := 0 to length(SpikeLoc)-1 do
  begin
    SpikeLoc[i].Voltage := nil;
    SpikeLoc[i].FitVoltage := nil;
  end; 
  SpikeLoc := nil;

  CreateElectrode := TRUE;
  Wagging := TRUE;
  Timer.Enabled := TRUE;
  BWag.Caption := 'Stop';
end;

{--------------------------------------------------------------------}
Procedure TLocSortForm.DrawElectrode;
const PIby2 = PI/2;
var i,x1,y1,eoff : integer;
    plgpts : array of TPoint;
begin
  try
    eoff := StrToInt(EElectrodeOffset.Text);
  except
    eoff := 0;
  end;

  With Electrode do
  begin
    ebm.Canvas.FillRect(LocSortForm.clientRect);//clear electrode bm
    ebm.canvas.Pen.Color := clDkGray;
    x1 := round(Rx(Outline[0].x+xoff,Outline[0].y+yoff+eoff,0));
    y1 := Outline[0].y+yoff+eoff;
    ebm.canvas.moveto(x1,y1); //draw outline
    SetLength(plgpts,NumPoints+1);
    ebm.Canvas.MoveTo(round(Rx(Outline[0].x+xoff,Outline[0].y+yoff+eoff,0)),Outline[0].y+yoff+eoff);
    ebm.canvas.Pen.Color := RGB(48,64,48);
    For i := 0 to NumPoints-1 do
    begin
      plgpts[i].x := round(Rx(Outline[i].x+xoff,Outline[i].y+yoff+eoff,0));
      plgpts[i].y := Outline[i].y+yoff+eoff;
      ebm.Canvas.LineTo(plgpts[i].x,plgpts[i].y);
    end;
    plgpts[NumPoints].x := plgpts[0].x;
    plgpts[NumPoints].y := plgpts[0].y;
    ebm.Canvas.LineTo(plgpts[0].x,plgpts[0].y);

    if abs(theta) < PIby2 - 0.05 then
    begin
      ebm.canvas.Brush.Color := RGB(48,64,48);
      ebm.canvas.FloodFill(plgpts[0].x+1,plgpts[0].y+1,RGB(48,64,48),fsborder{fol.fillstyle});
    end;

    ebm.canvas.Brush.Color := clBlack;
    ebm.canvas.Font.Color := clBlack;

    ebm.canvas.Pen.Color := $005E879B;
    ebm.canvas.Brush.Color := $005E879B;
    For i := 0 to NumSites-1 do  //draw the electrode sites
    begin
      ebm.canvas.Brush.Color := $005E879B;
      case roundsite of
        TRUE : ebm.canvas.ellipse(round(RX(SiteLoc[i].x+xoff-SiteSize.x div 2,SiteLoc[i].y+yoff+eoff-SiteSize.y div 2,0)),SiteLoc[i].y+yoff+eoff-SiteSize.y div 2,
                                  round(RX(SiteLoc[i].x+xoff+SiteSize.x div 2,SiteLoc[i].y+yoff+eoff+SiteSize.y div 2,0)),SiteLoc[i].y+yoff+eoff+SiteSize.y div 2);
        FALSE : ebm.canvas.framerect(rect(SiteLoc[i].x+xoff-SiteSize.x div 2,SiteLoc[i].y+yoff+eoff-SiteSize.y div 2,
                                          SiteLoc[i].x+xoff+SiteSize.x div 2,SiteLoc[i].y+yoff+eoff+SiteSize.y div 2));
      end;
      if not wagging then
      begin
        ebm.canvas.Brush.Color := RGB(48,64,48);
        ebm.canvas.TextOut(round(RX(SiteLoc[i].x+xoff+SiteSize.x div 2+1,SiteLoc[i].y+yoff+eoff,0)),SiteLoc[i].y+yoff+eoff,inttostr(i));
      end;
    end;
    //ebm.canvas.TextOut(RX(ebm.width div 2-ebm.canvas.TextWidth(Name) div 2,0,0),0,Name);
  end;
  ebm.canvas.Brush.Color := clBlack;
end;

{--------------------------------------------------------------------}
Function TLocSortForm.Rx(x,y,z : single) : single;//the three d to 2d transform
begin
  Rx := {round}((x-xoff) * cos(theta) + z * sin(theta)) + xoff;
end;

{--------------------------------------------------------------------}
procedure TLocSortForm.TransForm(x,y,z : single; var tx,ty : integer);//the three d to 2d transform
begin
  tx := round(Rx(x,y,z));
  ty := round(y);
end;

{--------------------------------------------------------------------}
procedure TLocSortForm.Render;
var p,tx,ty,ang,s,xo,yo,yp,c,eoff : integer;
    pb : PByteArray;
    col : TColor;
function outofbounds(x,y : integer) : boolean;
begin
  if (x < 0) or (x >= lsbm.Width-1) or (y < 0) or (y >= lsbm.Height-1)
    then OutOfBounds := TRUE
    else OutOfBounds := FALSE;
end;
begin
  DrawElectrode;

  try
    eoff := StrToInt(EElectrodeOffset.Text);
  except
    eoff := 0;
  end;

  lsbm.Canvas.Brush.Color := clBlack;
  if CShowElectrode.Checked
    then lsbm.canvas.Draw(0,0,ebm)//draw the electrode bm
    else  lsbm.Canvas.FillRect(LocSortForm.clientRect);

  For p := 0 to Length(SpikeLoc)-1 do
  begin
    With SpikeLoc[p] do
    if done then
    begin
      TransForm(x+xoff,y+yoff+eoff,z,tx,ty);//The 3D to 2D Transform
      if not outofbounds(tx,ty) then
      begin
        if p = Length(SpikeLoc)-1 then
        begin
          lsbm.Canvas.Pen.Color := clRed;
          lsbm.Canvas.Brush.Color := clRed;
          lsbm.Canvas.Rectangle(tx-2,ty-2,tx+2,ty+2);
        end else
        begin
          pb := lsbm.ScanLine[ty];
          col := COLORTABLE[0];
          if ClusterId>-1 then
            if Cluster[ClusterId].good and (ClusterId < MAXCLUSTERS-2)
              then col := COLORTABLE[ClusterId+1];
          if (CShowUnClustered.Checked or (col <> COLORTABLE[0])) then
          begin
            pb[tx*3]   := GetBValue(col);
            pb[tx*3+1] := GetGValue(col);
            pb[tx*3+2] := GetRValue(col);
          end;
        end;
      end;
      (*
      //plot peak
      TransForm(-100+random(6)-3+xoff,400-ppk div 5+yoff,0,tx,ty);//The 3D to 2D Transform
      if not outofbounds(tx,ty) then
      begin
        pb := lsbm.ScanLine[ty];
        col := clAqua;//{clYellow}COLORTABLE[c];
        pb[tx*3]   := GetBValue(col);
        pb[tx*3+1] := GetGValue(col);
        pb[tx*3+2] := GetRValue(col);
      end;
      //plot slope
      TransForm(-80+random(6)-3+xoff,600-pslp*2+yoff,0,tx,ty);//The 3D to 2D Transform
      if not outofbounds(tx,ty) then
      begin
        pb := lsbm.ScanLine[ty];
        col := clYellow;//{clYellow}COLORTABLE[c];
        pb[tx*3]   := GetBValue(col);
        pb[tx*3+1] := GetGValue(col);
        pb[tx*3+2] := GetRValue(col);
      end;
      *)

      //plot the voltages and fit voltages
      if ShowFitPlot.Checked then
      if p = Length(SpikeLoc)-1 then
      begin
        //clear space
        lsbm.Canvas.Pen.Color := clBlack;
        lsbm.Canvas.Brush.Color := clBlack;
        lsbm.Canvas.Rectangle(0,lsbm.height-150,100,lsbm.height);

        //draw orig voltages
        lsbm.Canvas.Pen.Color := clLime;
        For s := 0 to Electrode.Numsites-1 do
        begin
          xo := Electrode.SiteLoc[s].y div 4;
          yo := lsbm.height - 100 - Electrode.SiteLoc[s].x div 3;
          yp := yo - round(SpikeLoc[p].Voltage[s]/3);
          lsbm.Canvas.MoveTo(xo,yo);
          lsbm.Canvas.LineTo(xo,yp);
        end;
        //draw fit voltages
        lsbm.Canvas.Pen.Color := clRed;
        For s := 0 to Electrode.Numsites-1 do
        begin
          xo := Electrode.SiteLoc[s].y div 4 + 3;
          yo := lsbm.height - 100 - Electrode.SiteLoc[s].x div 3;
          yp := yo - round(SpikeLoc[p].FitVoltage[s]/3);
          lsbm.Canvas.MoveTo(xo,yo);
          lsbm.Canvas.LineTo(xo,yp);
        end;

        lsbm.Canvas.Pen.Color := clGray;
        lsbm.Canvas.MoveTo(0,lsbm.height-100);
        lsbm.Canvas.LineTo(160,lsbm.height-100);

        lsbm.canvas.font.color := clwhite;
        lsbm.canvas.TextOut(1,lsbm.height-50,'x='+inttostr(round(SpikeLoc[p].x))
                                         +' , y='+inttostr(round(SpikeLoc[p].y))
                                         +' , z='+inttostr(round(SpikeLoc[p].z))
                                         +' , Vo='+inttostr(round(SpikeLoc[p].vo))
                                         +' , m='+floattostrf(SpikeLoc[p].m,fffixed,3,2)
                                         +' , o=' +floattostrf(SpikeLoc[p].o,fffixed,3,2)
                                         +' , itr=' +inttostr(SpikeLoc[p].FitIterations)
                                         +' , er='+floattostrf(SpikeLoc[p].FitError/Electrode.NumSites,fffixed,3,2));
      end;
    end;
  end;

  //plot cluster information
  if CShowClusterLocation.Checked or CShowClusterStats.Checked then
  For c := 0 to NumGoodClusters-1 do
  With Cluster[c] do
  if good then
  begin
    TransForm(x.mean+xoff,y.mean+yoff+eoff,z.mean,tx,ty);//The 3D to 2D Transform
    if not outofbounds(tx,ty) then
    begin
      lsbm.Canvas.Pen.Color := COLORTABLE[c+1];
      lsbm.Canvas.Brush.Color := COLORTABLE[c+1];
      lsbm.Canvas.Rectangle(tx-2,ty-2,tx+2,ty+2);
    end;

    TransForm(x.mean+2*sqrt(x.vari)+xoff,y.mean+yoff+eoff,z.mean,tx,ty);//The 3D to 2D Transform
    if not outofbounds(tx,ty) then
    begin
      lsbm.Canvas.Font.Color := COLORTABLE[c+1];
      lsbm.Canvas.Font.Size := 8;
      lsbm.Canvas.Brush.Style := bsClear;
      if CShowClusterLocation.Checked then
        lsbm.canvas.TextOut(tx,ty-6,'c'+inttostr(c)
                              +' x'+inttostr(round(x.mean))
                              +',y'+inttostr(round(y.mean))
                              +',z'+inttostr(round(z.mean)));
      if CShowClusterStats.Checked then
        lsbm.canvas.TextOut(tx,ty+6,'v'+inttostr(round(vo.mean))
                                  +',m'+floattostrf(m.mean,fffixed,3,2)
                                  +',o'+floattostrf(Ro.mean,fffixed,3,2)
                                  +',E'+floattostrf(FitError.mean,fffixed,3,2));
    end;
  end;
  //BitBlt(LocSortForm.Canvas.Handle,0,0,lsbm.Height,lsbm.Width,lsbm.canvas.handle,0,0,SRCCOPY);
  ang := round(theta/PI*180);
  If Saving and (LoopCount in [1,2])
    then lsbm.SaveToFile('3DPic_'+inttostr(ang)+'_deg.bmp');

  {LocSortForm.}Canvas.Draw(0,0,lsbm);
  {LocSortForm.}Update;

end;

{--------------------------------------------------------------------}
Procedure CLocSimplex.callout(var pr : DArrayNP);
begin
end;
{--------------------------------------------------------------------}
Function CLocSimplex.func(var pr : DArrayNP) : double;
const PIby2 = PI/2;
      Inv4PI = 1/(4*PI);
var site,sx,sy,sz : integer;
    er,Vr,r,r2,rx,ry,rz,rx2,ry2,rz2,Vo,m,ro,t,c,e,f,g,s,w,lamda : double;

function fact(s : single): single;
var j,res : integer;
begin
 res := 1;
 if s > 100 then s := 100;
 For j := 2 to trunc(s) do
   res := res * j;
 Result := res * s;
end;

begin
  With LocSortForm.Electrode do
  begin
    if pr[1] < TopLeftSite.x-MAXDIST then pr[1] := TopLeftSite.x-MAXDIST;
    if pr[1] > BotRightSite.x + MAXDIST then pr[1] := BotRightSite.x + MAXDIST;

    if pr[2] < TopLeftSite.y - MAXDIST then pr[1] := TopLeftSite.y - MAXDIST;
    if pr[2] > BotRightSite.y + MAXDIST then pr[1] := BotRightSite.y + MAXDIST;

    if pr[3] < 0 then pr[3] := abs(pr[3]);
    if pr[3] > MAXDIST then pr[3] := MAXDIST;

    if pr[4]{Vo} < 10 then pr[4] := 10;
    if pr[4]{Vo} > MAXVo then pr[4] := MAXVo;

    case LocSortForm.FuncNum of
     0 : if pr[5]{m} < 0 then pr[5] := 0;
     1 : if pr[5]{m} < 1 then pr[5] := 1;
     2 : if pr[5]{m} < 1 then pr[5] := 1;
    end;
    
    if pr[5]{m} > MAXM then pr[5] := MAXM;

    if pr[6]{offset} < 0 then pr[6] := 0;
    if pr[6]{offset} > MAXO then pr[6] := MAXO;
  end;
  //now sum up the error
  er := 0;
  For site := 0 to LocSortForm.Electrode.NumSites-1 do
  begin
    sx := LocSortForm.Electrode.SiteLoc[site].x;
    sy := LocSortForm.Electrode.SiteLoc[site].y;
    sz := 0;

    rx := pr[1]-sx;
    ry := pr[2]-sy;
    rz := pr[3]-sz;
    rx2 := rx * rx;
    ry2 := ry * ry;
    rz2 := rz * rz;
    r2 := rx2 + ry2 + rz2;
    if r2 < 1 then r2 := 1;
    r := sqrt(r2);
    if r < 1 then r := 1;
    if r > MAXRADIUS then r := MAXRADIUS;

    Vo  := pr[4];
    m   := pr[5];
    ro  := pr[6];

    Case LocSortForm.FuncNum of
     0 : Vr := Vo / (r+ro) {* Inv4PI }- m; //Offset Inv Function
     1 : Vr := Vo  / exp((r+ro)/m); // Offset Exp function
     2 : Vr := Vo * exp(-0.5 * sqr((r+ro)/m)); //Offset Gaussian
    end;

    LocSortForm.SpikeLoc[index].FitVoltage[site] := Vr;
    er := er + sqr(Vr - LocSortForm.SpikeLoc[index].Voltage[site]);
  end;
  Result := er/LocSortForm.Electrode.NumSites;
end;

{--------------------------------------------------------------------}
Procedure TLocSortForm.ComputeXYZ(index : integer; lz,lvo,lm,lo : boolean; var chi : single);
var c,i : integer;
    x : integer;
    s : string;
    d : double;
begin
  LocSimplex.ndim := 6;
  LocSimplex.index := index;
  For i := 1 to LocSimplex.ndim+1 do
  begin
    LocSimplex.verticies[i,1] := SpikeLoc[index].x + random(20)-10;
    LocSimplex.verticies[i,2] := SpikeLoc[index].y + random(20)-10;
    if lz
      then LocSimplex.verticies[i,3] := SpikeLoc[index].z
      else LocSimplex.verticies[i,3] := SpikeLoc[index].z + random(20)-10;
    if lvo
      then LocSimplex.verticies[i,4] := SpikeLoc[index].vo
      else LocSimplex.verticies[i,4] := SpikeLoc[index].vo + random(50)-25;
    if lm
      then LocSimplex.verticies[i,5] := SpikeLoc[index].m
      else LocSimplex.verticies[i,5] := SpikeLoc[index].m + random(5)-2.5;
    if lo
      then LocSimplex.verticies[i,6] := SpikeLoc[index].o
      else LocSimplex.verticies[i,6] := SpikeLoc[index].o + random(6)-3;

    LocSimplex.funcvals[i] := LocSimplex.Func(LocSimplex.verticies[i]);
  end;

  LocSimplex.ftol := 0.00001;
  LocSimplex.Run;

  SpikeLoc[index].x := LocSimplex.verticies[LocSimplex.best,1];
  SpikeLoc[index].y := LocSimplex.verticies[LocSimplex.best,2];
  SpikeLoc[index].z := LocSimplex.verticies[LocSimplex.best,3];
  SpikeLoc[index].vo := LocSimplex.verticies[LocSimplex.best,4];
  SpikeLoc[index].m := LocSimplex.verticies[LocSimplex.best,5];
  SpikeLoc[index].o := LocSimplex.verticies[LocSimplex.best,6];

  //get the fit values (done in function call)
  SpikeLoc[index].FitError := sqrt(LocSimplex.Func(LocSimplex.verticies[LocSimplex.best]));
  SpikeLoc[index].FitIterations := LocSimplex.iter;


end;

{--------------------------------------------------------------------}
procedure TLocSortForm.ComputeLoc(var Spike : TSpike; z,vo,m,o : single;
                         lz,lvo,lm,lo : boolean;
                         FunctionNum : integer; var chi : single);
var  np,w,meanx,meany,c,index : integer;
     amp : single;
     s : String;
  procedure GetCentroid(pass : integer);
  var sumw,sumwx,sumwy : double;
      c,x,y : integer;
  begin
    With Spike do
    begin
      sumwx := 0;
      sumwy := 0;
      sumw := 0;
      for c := 0 to np-1 do
      begin
        x := Electrode.SiteLoc[c{ mod 16}].x;
        y := Electrode.SiteLoc[c{ mod 16}].y;

        w := param[c,3];{SHRT peak for chan c}
        w := w*w;
        sumwx := sumwx + w*x;
        sumwy := sumwy + w*y;
        sumw := sumw + w;
      end;
      if sumw = 0 then sumw := 1;
      meanx := round(sumwx/sumw);
      meany := round(sumwy/sumw);
    end;
  end;

begin
  if not electrode.Created then begin beep; exit; end;
  np := Length(Spike.param);
  if electrode.NumSites{*2} < np then begin beep; exit; end;
  meanx := 0;
  meany := 0;
  GetCentroid(1); //pass1
  GetCentroid(2); //pass2

  index := Length(SpikeLoc);
  SetLength(SpikeLoc,index+1);
  SetLength(SpikeLoc[index].Voltage,Electrode.NumSites{*2});
  SetLength(SpikeLoc[index].FitVoltage,Electrode.NumSites{*2});

  FuncNum := FunctionNum;

  SpikeLoc[index].done := false;
  SpikeLoc[index].x := meanx; //initial guess for levmarq
  SpikeLoc[index].y := meany; //initial guess for levmarq
  SpikeLoc[index].z := z;    //initial guess for levmarq
  SpikeLoc[index].vo  := vo;
  SpikeLoc[index].m   := m;
  SpikeLoc[index].o   := o;

  amp := 10 / (2048 * ExtGain * ADGAin);
  For c := 0 to np-1 do
    SpikeLoc[index].Voltage[c] := Spike.Param[c,3] * amp * V2uV;

  ComputeXYZ(index,lz,lvo,lm,lo,chi);
  SpikeLoc[index].ClusterId := -1;
  SortSpike(index,False{don't assign all});
  Spike.Cluster := SpikeLoc[index].ClusterId+1;
  if Spike.Cluster > 0 then
  begin
    if not Cluster[Spike.Cluster-1].good
      then Spike.Cluster := 0;
  end;
  SpikeLoc[index].done := true;

  if index=0 then NSamplesForStats := 0;
  if (SpikeLoc[index].vo <> MAXVo) and (SpikeLoc[index].m <> MAXM)
  and (SpikeLoc[index].ClusterId >= 0) and Cluster[SpikeLoc[index].ClusterId].good then
  begin
    MeanZ := (NSamplesForStats * MeanZ + SpikeLoc[index].z) / (NSamplesForStats+1);
    MeanVo := (NSamplesForStats * MeanVo + SpikeLoc[index].vo) / (NSamplesForStats+1);
    MeanM  := (NSamplesForStats * MeanM + SpikeLoc[index].m) / (NSamplesForStats+1);
    MeanRo  := (NSamplesForStats * MeanRo + SpikeLoc[index].o) / (NSamplesForStats+1);
    MeanEr  := (NSamplesForStats * MeanEr + SpikeLoc[index].FitError) / (NSamplesForStats+1);
    inc(NSamplesForStats);
  end;

  if cbrender.checked then
  begin
    s := 'mZ= '+floattostrf(MeanZ,fffixed,4,2);
    s := s+' mVo= '+inttostr(round(MeanVo));
    s := s+' mM= '+floattostrf(MeanM,fffixed,4,2);
    s := s+' mRo= '+floattostrf(MeanRo,fffixed,4,2);
    s := s+' mEr= '+floattostrf(MeanEr,fffixed,4,2);
    StatusBar.SimpleText := s;
    StatusBar.refresh;
  end;
  lnumclusters.caption := inttostr(numgoodclusters);
end;

{--------------------------------------------------------------------}
procedure TLocSortForm.SortSpike(index : integer; ReAssignAll : boolean);
const MIN_CLUS_DIST2 = sqr(3);
      AVE_CLUS_DIST2 = sqr(5);
      MAX_CLUS_DIST2 = sqr(20);
      MIN_SPIKES_IN_CLUSTER = 15;
      CLUS_SIZE = 2.8;
      USE_CLVAR = TRUE;

var c,c2,i,i2 : integer;
    ClosestCluster : integer;
    Dist,ClosestDistance : single;
    tmpcluster : TCluster;
    //merged : boolean;
  //------------------------------------------------------
  Function DistSpkToClus(Spk : TSpikeLoc; Clus : TCluster) : single;
  var dx2,dy2,dz2,d1,d2 : single;
  begin
    With Clus do
    begin
      dx2 := sqr(Spk.x-x.mean);
      dy2 := sqr(Spk.y-y.mean);
      dz2 := sqr(Spk.z-z.mean);
      if good and USE_CLVAR
        then Result := sqrt(dx2/x.vari + dy2/y.vari + dz2/z.vari)
        else Result := sqrt((dx2 + dy2 + dz2)/MIN_CLUS_DIST2);
    end;
  end;
  //------------------------------------------------------
  Function DistClustToClus(Clust1,Clust2 : TCluster) : single;
  var dx2,dy2,dz2,d1,d2 : single;
  begin
    dx2 := sqr(Clust1.x.mean - Clust2.x.mean);
    dy2 := sqr(Clust1.y.mean - Clust2.y.mean);
    dz2 := sqr(Clust1.z.mean - Clust2.z.mean);

    With Clust1 do
      if good and USE_CLVAR
        then d1 := {sqrt}(dx2/x.vari + dy2/y.vari + dz2/z.vari)
        else d1 := {sqrt}((dx2 + dy2 + dz2) /AVE_CLUS_DIST2);

    With Clust2 do
      if good and USE_CLVAR
        then d2 := {sqrt}(dx2/x.vari + dy2/y.vari + dz2/z.vari)
        else d2 := {sqrt}((dx2 + dy2 + dz2) /AVE_CLUS_DIST2);

    Result := sqrt((d1{*d1}+d2{*d2})/2);
  end;
  //------------------------------------------------------
  Procedure InitCluster(var clust : TCluster);
  var s : integer;
  begin
    With clust do
    begin
      good := false;
      n := 0;
      x.sum := 0;
      y.sum := 0;
      z.sum := 0;
      x.sum2 := 0;
      y.sum2 := 0;
      z.sum2 := 0;

      Vo.sum := 0;
      Ro.sum := 0;
      m.sum  := 0;
      Vo.sum2 := 0;
      Ro.sum2 := 0;
      m.sum2  := 0;

      FitError.sum := 0;
      FitIterations.sum := 0;
      FitError.sum2 := 0;
      FitIterations.sum2 := 0;

      SetLength(Voltage,Electrode.NumSites);
      SetLength(FitVoltage,Electrode.NumSites);

      For s := 0 to Electrode.NumSites-1 do
      begin
        Voltage[s].sum := 0;
        FitVoltage[s].sum := 0;
        Voltage[s].sum2 := 0;
        FitVoltage[s].sum2 := 0;
      end;
    end;
  end;
  //------------------------------------------------------
  Procedure AddToCluster(spk : TSpikeLoc; var clust : TCluster);
  var s : integer;
  begin
    inc(clust.n);

    clust.x.sum := clust.x.sum + spk.x;
    clust.y.sum := clust.y.sum + spk.y;
    clust.z.sum := clust.z.sum + spk.z;

    clust.x.sum2 := clust.x.sum2 + sqr(spk.x);
    clust.y.sum2 := clust.y.sum2 + sqr(spk.y);
    clust.z.sum2 := clust.z.sum2 + sqr(spk.z);

    clust.x.mean := clust.x.sum/clust.n;
    clust.y.mean := clust.y.sum/clust.n;
    clust.z.mean := clust.z.sum/clust.n;

    clust.x.vari := (clust.x.sum2/clust.n - sqr(clust.x.mean));
    clust.y.vari := (clust.y.sum2/clust.n - sqr(clust.y.mean));
    clust.z.vari := (clust.z.sum2/clust.n - sqr(clust.z.mean));

    if clust.x.vari < MIN_CLUS_DIST2 then clust.x.vari := MIN_CLUS_DIST2;
    if clust.y.vari < MIN_CLUS_DIST2 then clust.y.vari := MIN_CLUS_DIST2;
    if clust.z.vari < MIN_CLUS_DIST2 then clust.z.vari := MIN_CLUS_DIST2;
    if clust.x.vari > MAX_CLUS_DIST2 then clust.x.vari := MAX_CLUS_DIST2;
    if clust.y.vari > MAX_CLUS_DIST2 then clust.y.vari := MAX_CLUS_DIST2;
    if clust.z.vari > MAX_CLUS_DIST2 then clust.z.vari := MAX_CLUS_DIST2;

    clust.Vo.sum := clust.Vo.sum + spk.Vo;
    clust.Ro.sum := clust.Ro.sum + spk.o;
    clust.m.sum := clust.m.sum + spk.m;
    clust.Vo.sum2 := clust.Vo.sum2 + sqr(spk.Vo);
    clust.Ro.sum2 := clust.Ro.sum2 + sqr(spk.o);
    clust.m.sum2 := clust.m.sum2 + sqr(spk.m);
    clust.Vo.mean := clust.Vo.sum/clust.n;
    clust.Ro.mean :=  clust.Ro.sum/clust.n;
    clust.m.mean :=  clust.m.sum/clust.n;
    clust.Vo.vari := (clust.Vo.sum2/clust.n - sqr(clust.Vo.mean));
    clust.Ro.vari := (clust.Ro.sum2/clust.n - sqr(clust.Ro.mean));
    clust.m.vari := (clust.m.sum2/clust.n - sqr(clust.m.mean));

    if clust.Vo.vari < 1 then clust.Vo.vari := 1;
    if clust.Ro.vari < 1 then clust.Ro.vari := 1;
    if clust.m.vari < 1 then clust.m.vari := 1;

    clust.FitError.sum := clust.FitError.sum + spk.FitError;
    clust.FitIterations.sum := clust.FitIterations.sum + spk.FitIterations;
    clust.FitError.sum2 := clust.FitError.sum2 + sqr(spk.FitError);
    clust.FitIterations.sum2 := clust.FitIterations.sum2 + sqr(spk.FitIterations);
    clust.FitError.mean := clust.FitError.sum/clust.n;
    clust.FitIterations.mean :=  clust.FitIterations.sum/clust.n;
    clust.FitError.vari := (clust.FitError.sum2/clust.n - sqr(clust.FitError.mean));
    clust.FitIterations.vari := (clust.FitIterations.sum2/clust.n - sqr(clust.FitIterations.mean));

    if clust.FitError.vari < 1 then clust.FitError.vari := 1;
    if clust.FitIterations.vari < 1 then clust.FitIterations.vari := 1;

    For s := 0 to Electrode.NumSites-1 do
    begin
      clust.Voltage[s].sum := clust.Voltage[s].sum + spk.Voltage[s];
      clust.FitVoltage[s].sum := clust.FitVoltage[s].sum + spk.FitVoltage[s];
      clust.Voltage[s].sum2 := clust.Voltage[s].sum2 + sqr(spk.Voltage[s]);
      clust.FitVoltage[s].sum2 := clust.FitVoltage[s].sum2 + sqr(spk.FitVoltage[s]);
      clust.Voltage[s].mean := clust.Voltage[s].sum/clust.n;
      clust.FitVoltage[s].mean :=  clust.FitVoltage[s].sum/clust.n;
      clust.Voltage[s].vari := (clust.Voltage[s].sum2/clust.n - sqr(clust.Voltage[s].mean));
      clust.FitVoltage[s].vari := (clust.FitVoltage[s].sum2/clust.n - sqr(clust.FitVoltage[s].mean));

      if clust.Voltage[s].vari < 1 then clust.Voltage[s].vari := 1;
      if clust.FitVoltage[s].vari < 1 then clust.FitVoltage[s].vari := 1;
    end;
    //see if there is enough spikes in the cluster to call it 'good'
    if Clust.n > MIN_SPIKES_IN_CLUSTER then Clust.good := TRUE;
  end;

  //------------------------------------------------------
  Procedure Copy(var FromClus,ToClus : TCluster);
  var i : integer;
  begin
    Move(FromClus.x,ToClus.x,sizeof(TStats));
    Move(FromClus.y,ToClus.y,sizeof(TStats));
    Move(FromClus.z,ToClus.z,sizeof(TStats));
    Move(FromClus.Vo,ToClus.Vo,sizeof(TStats));
    Move(FromClus.m,ToClus.m,sizeof(TStats));
    Move(FromClus.Ro,ToClus.Ro,sizeof(TStats));
    Move(FromClus.FitError,ToClus.FitError,sizeof(TStats));
    Move(FromClus.FitIterations,ToClus.FitIterations,sizeof(TStats));
    Move(FromClus.Voltage[0],ToClus.Voltage[0],Length(FromClus.Voltage)*sizeof(TStats));
    Move(FromClus.FitVoltage[0],ToClus.FitVoltage[0],Length(FromClus.Voltage)*sizeof(TStats));
    ToClus.good := FromClus.good;
    ToClus.n := FromClus.n;
    SetLength(ToClus.SpikeList,ToClus.n{Length(FromClus.SpikeList)});
    Move(FromClus.SpikeList[0],ToClus.SpikeList[0],Sizeof(integer)*ToClus.n{Length(ToClus.SpikeList)});
  end;
  //------------------------------------------------------
  Procedure Merge(c1,c2 : integer);
  var i,n : integer;
    procedure MergeStats(var s1,s2 : TStats; n : integer);
    begin
      s1.sum := s1.sum + s2.sum;
      s1.sum2 := s1.sum2 + s2.sum2;
      s1.mean := s1.sum/n;
      s1.vari := s1.sum2/n - sqr(s1.mean);
      if s1.vari < 1 then s1.vari := 1;
    end;
  begin
    Cluster[c1].n := Cluster[c1].n + Cluster[c2].n;
    n := Cluster[c1].n;
    MergeStats(Cluster[c1].x,Cluster[c2].x,n);
    MergeStats(Cluster[c1].y,Cluster[c2].y,n);
    MergeStats(Cluster[c1].z,Cluster[c2].z,n);

    if Cluster[c1].x.vari < MIN_CLUS_DIST2 then Cluster[c1].x.vari := MIN_CLUS_DIST2;
    if Cluster[c1].y.vari < MIN_CLUS_DIST2 then Cluster[c1].y.vari := MIN_CLUS_DIST2;
    if Cluster[c1].z.vari < MIN_CLUS_DIST2 then Cluster[c1].z.vari := MIN_CLUS_DIST2;

    if Cluster[c1].x.vari > MAX_CLUS_DIST2 then Cluster[c1].x.vari := MAX_CLUS_DIST2;
    if Cluster[c1].y.vari > MAX_CLUS_DIST2 then Cluster[c1].y.vari := MAX_CLUS_DIST2;
    if Cluster[c1].z.vari > MAX_CLUS_DIST2 then Cluster[c1].z.vari := MAX_CLUS_DIST2;

    MergeStats(Cluster[c1].Vo,Cluster[c2].Vo,n);
    MergeStats(Cluster[c1].Ro,Cluster[c2].Ro,n);
    MergeStats(Cluster[c1].m,Cluster[c2].m,n);
    MergeStats(Cluster[c1].FitError,Cluster[c2].FitError,n);
    MergeStats(Cluster[c1].FitIterations,Cluster[c2].FitIterations,n);

    for i := 0 to Length(Cluster[c1].Voltage) -1 do
    begin
      MergeStats(Cluster[c1].Voltage[i],Cluster[c2].Voltage[i],n);
      MergeStats(Cluster[c1].FitVoltage[i],Cluster[c2].FitVoltage[i],n);
    end;
    if n > MIN_SPIKES_IN_CLUSTER then Cluster[c1].good := TRUE;

    i := Length(Cluster[c1].SpikeList);

    SetLength(Cluster[c1].SpikeList,n);
    Move(Cluster[c2].SpikeList[0],
         Cluster[c1].SpikeList[i],
         Sizeof(integer)*Length(Cluster[c2].SpikeList));

    //now shift all others back
    n := length(cluster);
    if c2 < n-1 then
      For i := c2+1 to n-1 do
        Copy(Cluster[i],Cluster[i-1]);
    Cluster[n-1].Voltage := nil;
    Cluster[n-1].FitVoltage := nil;
    Cluster[n-1].SpikeList := nil;
    SetLength(Cluster,n-1);

    //set the cluster ids of all the member spikes to the cluster number
    For i := c1 to Length(Cluster)-1 do
      For n := 0 to Length(Cluster[i].SpikeList)-1 do
         SpikeLoc[Cluster[i].SpikeList[n]].ClusterId := i;
  end;

begin//========================================================================
  if index=0 then
  begin
    For c := 0 to Length(Cluster)-1 do
    begin
      Cluster[c].Voltage := nil;
      Cluster[c].FitVoltage := nil;
      Cluster[c].SpikeList := nil;
    end;
    Cluster := nil;
  end;

  //see if it is within MIN_CLUS_DIST of any cluster
  //if it is then assign it to that one
  if not ReAssignAll then
  begin
    ClosestDistance := CLUS_SIZE;
    ClosestCluster := -1;
    For c := 0 to Length(Cluster)-1 do
    begin
      Dist := DistSpkToClus(SpikeLoc[index],Cluster[c]);
      if Dist < ClosestDistance then
      begin
        ClosestCluster := c;
        ClosestDistance := Dist;
      end;
    end;

    if ClosestCluster = -1 {near to none} then
    begin
      c := Length(Cluster);
      SetLength(Cluster,c+1);
      InitCluster(Cluster[c]);
      AddToCluster(SpikeLoc[index],Cluster[c]);
      SetLength(Cluster[c].SpikeList,1);
      Cluster[c].SpikeList[0] := index;
      SpikeLoc[index].ClusterId := c;
    end else
    begin//nearest to this one
      c := ClosestCluster;
      AddToCluster(SpikeLoc[index],Cluster[c]);
      SetLength(Cluster[c].SpikeList,Cluster[c].n);
      Cluster[c].SpikeList[Cluster[c].n-1] := index;
      SpikeLoc[index].ClusterId := c;
      //check for two close together clusters, merge if overlapping
      c := 0;
      //merged := false;
      While c < Length(Cluster)-1 do
      begin
        c2 := c+1;
        While c2 < Length(Cluster) do
           if DistClustToClus(Cluster[c],Cluster[c2]) < 1.4*CLUS_SIZE then
           begin
             Merge(c,c2);
             //merged := true;
           end else c2 := c2 + 1;
        c := c + 1;
      end;
    end;

    //make sure that the 'good' clusters come first
    SetLength(tmpcluster.Voltage,Electrode.NumSites);
    SetLength(tmpcluster.FitVoltage,Electrode.NumSites);
    For c := 0 to Length(cluster)-2 do
      For c2 := c+1 to Length(cluster)-1 do
        if not Cluster[c].good and Cluster[c2].good then
        begin  //swap
          Copy(Cluster[c],tmpcluster);
          Copy(Cluster[c2],Cluster[c]);
          Copy(tmpcluster,Cluster[c2]);
          For i := 0 to Length(Cluster[c].SpikeList)-1 do
            SpikeLoc[Cluster[c].SpikeList[i]].ClusterId := c;
          For i := 0 to Length(Cluster[c2].SpikeList)-1 do
            SpikeLoc[Cluster[c2].SpikeList[i]].ClusterId := c;
        end;

    //count how many good clusters
    NumGoodClusters := 0;
    For c := 0 to Length(cluster)-1 do
      if Cluster[c].good then inc(NumGoodClusters);

    //sort by highest number of spikes
    For c := 0 to NumGoodClusters-1{Length(cluster)-2} do
      For c2 := c+1 to NumGoodClusters-1 do
        if Cluster[c2].y.mean < Cluster[c].y.mean then
        begin  //swap
          Copy(Cluster[c],tmpcluster);
          Copy(Cluster[c2],Cluster[c]);
          Copy(tmpcluster,Cluster[c2]);
          For i := 0 to Length(Cluster[c].SpikeList)-1 do
            SpikeLoc[Cluster[c].SpikeList[i]].ClusterId := c;
          For i := 0 to Length(Cluster[c2].SpikeList)-1 do
            SpikeLoc[Cluster[c2].SpikeList[i]].ClusterId := c;
        end;

    tmpcluster.Voltage := nil;
    tmpcluster.FitVoltage := nil;
    tmpcluster.SpikeList := nil;
  end else//reassign all
  begin
    For i := 0 to index do
    begin
      ClosestDistance := CLUS_SIZE{n stds};
      ClosestCluster := -1;
      For c := 0 to Length(Cluster)-1 do
      begin
        Dist := DistSpkToClus(SpikeLoc[i],Cluster[c]);
        if Dist < ClosestDistance then
        begin
          ClosestCluster := c;
          ClosestDistance := Dist;
        end;
      end;
      SpikeLoc[i].ClusterId := ClosestCluster;
    end;
    //now recompute the cluster stats
    For c := 0 to Length(Cluster)-1 do
      InitCluster(Cluster[c]);
    For i := 0 to index do
      if SpikeLoc[i].ClusterId >= 0 then
      begin
        c := SpikeLoc[i].ClusterId;
        AddToCluster(SpikeLoc[i],Cluster[c]);
        SetLength(Cluster[c].SpikeList,Cluster[c].n);
        Cluster[c].SpikeList[Cluster[c].n-1] := i;
      end;
    //count how many good clusters
    NumGoodClusters := 0;
    For c := 0 to Length(cluster)-1 do
      if Cluster[c].good then inc(NumGoodClusters);
  end;
end;

{--------------------------------------------------------------------}
procedure TLocSortForm.Finish;
const TB = chr(9);
var i,j,k,itmp,c : integer;
    tfo : TextFile;
    SortedSites : array of integer;
    s : string;

  function DistToSite(i,s : integer) : single;
  begin
    Result := sqrt(sqr(SpikeLoc[i{ mod 16}].x-Electrode.SiteLoc[SortedSites[s] {mod 16}].x)
                 + sqr(SpikeLoc[i{ mod 16}].y-Electrode.SiteLoc[SortedSites[s] {mod 16}].y)
                 + sqr(SpikeLoc[i{ mod 16}].z));
  end;

begin //output to disk
  SortSpike(Length(SpikeLoc)-1,TRUE{reassign all});

  s := 'mZ= '+floattostrf(MeanZ,fffixed,4,2);
  s := s+'  mVo= '+inttostr(round(MeanVo));
  s := s+'  mM= '+floattostrf(MeanM,fffixed,4,2);
  s := s+'  mRo= '+floattostrf(MeanRo,fffixed,4,2);
  s := s+'  mEr= '+floattostrf(MeanEr,fffixed,4,2);
  StatusBar.SimpleText := s;

  SetLength(SortedSites,Electrode.NumSites{*2});
  try
    AssignFile(tfo,'Fit.txt');
    Rewrite(tfo);
    For i := 0 to Length(SpikeLoc)-1 do
    With SpikeLoc[i] do
    begin
      c := ClusterId;
      if c > NumGoodClusters-1 then c := -1;
      Write(tfo,c,TB,x:4,TB, y:4,TB,z:4,TB,vo:4:1,TB,m:4:1,TB,o:4:2,TB{,nslp:4,TB,wvln:4,TB});

      //Sort in order of dist away from cell loc
      For j := 0 to Length(SortedSites)-1 do
        SortedSites[j] := j;
      For j := 0 to Length(SortedSites)-2 do
        For k := j+1 to Length(SortedSites)-1 do
           if DistToSite(i,j) > DistToSite(i,k) then
           begin
             itmp := SortedSites[j];
             SortedSites[j] := SortedSites[k];
             SortedSites[k] := itmp;
           end;
      Write(tfo,TB);
      For j := 0 to Length(SortedSites)-1 do
        Write(tfo,DistToSite(i,j):4:1,TB);
      Write(tfo,TB);
      For j := 0 to Length(SortedSites)-1 do
        Write(tfo,Voltage[SortedSites[j]]:4:1,TB);
      Write(tfo,TB);
      For j := 0 to Length(SortedSites)-1 do
        Write(tfo,FitVoltage[SortedSites[j]]:4:1,TB);
      Writeln(tfo,TB,FitIterations,TB,FitError:4:2);
    end;
    CloseFile(tfo);
  except
    ShowMessage('Error opening file for write');
  end;
  SortedSites := nil;
end;

end.


