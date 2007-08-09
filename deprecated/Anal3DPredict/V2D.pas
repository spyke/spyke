unit V2D;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs, Math,
  ComCtrls, SurfPublicTypes, ElectrodeTypes,NumRecipies,SurfLocateAndSort;

const
  //ADPERPIX = 10;
  MAXDIST = 150;
  MAXVOLT = 200;
  NUMCURVESTOUSE = 20;
  //V2uV = 1000000;//1 million (uV -> V)
  //Vinf = 300;{uV}
type
  V2DLevMarqObj = class(LevMarqObj)
    public
      DOff : double;
      Procedure CheckBounds(var a : DArrayNP); override;
      Procedure func(xx : double; var a : DArrayNP; var yfit : double;
               var dyda : DarrayNP;  ma,mfit : integer; computedyda : boolean); override;
      //Procedure CallOut(var well : wellrec; var iter,ma,mfit : integer; var a : DArrayNP; chisq,chidiff : double; wellhist : wellhisttype; final : boolean); {override;}
  end;

  v2dcurverec = record
    mean,std,sum,sumsqr : single;
    val : array[0..NUMCURVESTOUSE-1] of single;
  end;
  Tv2dform = class(TForm)
    StatusBar: TStatusBar;
    procedure FormShow(Sender: TObject);
  private
    { Private declarations }
    v2dcurve : array[0..MAXDIST] of v2dcurverec;
    curv2dcurve : array[0..MAXDIST] of integer;
    npts : integer;
    HalfHei : integer;
  public
    { Public declarations }
    Procedure Plot(var Electrode : ElectrodeRec; var Spike : TSpike; intgain,extgain : integer);
    Procedure ComputeV2D(var m,b,v0 : double);
  end;

var
  v2dform: Tv2dform;

implementation

{$R *.DFM}

{=========================================================================}
procedure Tv2dform.Plot(var Electrode : ElectrodeRec; var Spike : TSpike; intgain,extgain : integer);
var i,j,p,Dist,MaxVIndex,n : integer;
    v,amp,maxV : single;
    s : string;
    numdists,d0,d1,d,diff : integer;
    distances : array[0..20] of integer;
    decreasing : boolean;

procedure DrawCurves;
var i : integer;
begin
    Canvas.Pen.Mode := pmXOR;
    Canvas.Pen.Color := clRed;
    Canvas.MoveTo(60,HalfHei-round(v2dcurve[0].Mean));
    For i := 1 to MAXDIST do
      if curv2dcurve[i]<maxV then Canvas.LineTo(60+i,HalfHei-round(v2dcurve[i].Mean));

    Canvas.Pen.Color := clLime;
    Canvas.MoveTo(60,HalfHei-round((v2dcurve[0].Mean+2*v2dcurve[0].Std)));
    For i := 1 to MAXDIST do
      if curv2dcurve[i]<maxV then Canvas.LineTo(60+i,HalfHei-round((v2dcurve[i].Mean+1*v2dcurve[i].Std){/ADPERPIX}));
    (*
    Canvas.Pen.Color := clYellow;
    Canvas.MoveTo(60,HalfHei-round(curv2dcurve[0]{/ADPERPIX}));
    For i := 1 to MAXDIST do
      if curv2dcurve[i]<maxV then Canvas.LineTo(60+i,HalfHei-round(curv2dcurve[i]{/ADPERPIX}));
    *)
    Canvas.Pen.Mode := pmCopy;
end;
begin
  if length(Spike.param) <> 2*Electrode.NumSites then exit;

  amp := 10 * V2uV / (2048 * extgain * intgain);
  //StatusBar.SimpleText := inttostr(intgain)+','+inttostr(extgain)+','+floattostrf(amp,fffixed,5,2);
  //GetMaxV and MaxVIndex
  MaxV := -1000;
  MaxVIndex := 0;
  for i := 0 to Electrode.NumSites-1 do
    begin
      v := Spike.param[i*2] * amp;
      if v > MaxV then
      begin
        MaxV := v;
        MaxVIndex := i;
      end;
    end;
  StatusBar.SimpleText := (s);
  //if NPts > 0 then DrawCurves;
  //maxV := round(2047 * amp);
  //initialize the stats for the current curve
  For i := 0 to MAXDIST do
    curv2dcurve[i] := MAXVOLT;

  //Create and plot the voltage to distance array for this spike
  for i := 0 to Electrode.NumSites-1 do
    With Electrode do
    begin
      Dist := trunc(sqrt(sqr(SiteLoc[i].x-SiteLoc[MaxVIndex].x) + sqr(SiteLoc[i].y-SiteLoc[MaxVIndex].y))/5)*5;
      if Dist < MAXDIST then
      begin
        p := round(Spike.param[i*2]*amp);
        Canvas.Pixels[60+Dist,round(HalfHei - p)] := clLime;
        if p < curv2dcurve[Dist] then curv2dcurve[Dist] := p;
      end;
    end;

  //count and index the distances
  numdists := 0;
  For i := 0 to MAXDIST-1 do
   if curv2dcurve[i] < MAXVOLT then
   begin
     distances[numdists] := i;
     inc(numdists);
   end;

  //see if the slope always decreases (no inflection)
  d0 := Distances[0];
  d1 := Distances[1];
  decreasing := true;
  j := curv2dcurve[d0];
  For i := 1 to NumDists-1 do
  begin
    if curv2dcurve[Distances[i]] > j then decreasing := false;
    j := curv2dcurve[Distances[i]];
  end;

  //add to array if decreasing and steep slope
  if decreasing then
  begin
    diff := curv2dcurve[d0]-curv2dcurve[d1];
    For j := 0 to NUMCURVESTOUSE-1 do
    begin
      if diff > (v2dcurve[d0].val[j]-v2dcurve[d1].val[j]) then
      begin
        For i := 0 to NumDists-1 do
          v2dcurve[Distances[i]].val[j] := curv2dcurve[Distances[i]];
        break;
      end;
    end;

    //compute stats of this array
    s := '';
    n := 0;
    For i := 0 to NumDists-1 do
    if curv2dcurve[Distances[i]]<maxV then
    begin
      d := Distances[i];
      v2dcurve[d].sum := 0;
      v2dcurve[d].sumsqr := 0;
      For j := 0 to NUMCURVESTOUSE-1 do
      if v2dcurve[d].val[j] <> 9999 then
      begin
        v2dcurve[d].sum :=  v2dcurve[d].sum + v2dcurve[d].val[j];
        v2dcurve[d].sumsqr :=  v2dcurve[d].sumsqr + sqr(v2dcurve[d].val[j]);
        inc(n);
      end;
      v2dcurve[d].mean   := v2dcurve[d].sum/n;
      v2dcurve[d].std    := sqrt(abs(v2dcurve[d].sumsqr/n - sqr(v2dcurve[d].mean)));
      s := s + inttostr(round(v2dcurve[d].mean))+' ';
    end;
  end;
  Inc(NPts);

  {Canvas.Pen.Color := clGray;
  Canvas.MoveTo(0,HalfHei);
  Canvas.LineTo(ClientWidth,HalfHei);
  DrawCurves; }
end;

{=========================================================================}
procedure Tv2dform.ComputeV2D(var m,b,v0 : double);
var V2DLevMarq : V2DLevMarqObj;
    x,y,sig     : DArrayNDATA;
    ndata       : integer;
    a           : DArrayNP;
    ma          : integer;
    lista       : IArrayNP;
    mfit        : integer;

    i : integer;
    firstchisq,lastchisq,chidiff,yfit: double;
    dyda : DarrayNP;
    iter : integer;
    s : string;

Procedure RunLevenberg;
const RUNAVEWIN = 10;
var
    runningchidiff : double;
    covar,alpha : DArrayNPbyNP;
    chisq,alamda: double;
begin
  //Initialize
  iter := 0;
  chisq := 0;
  alamda := -1.0;
  V2DLevMarq.mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,chisq,alamda);
  firstchisq := chisq;
  //StatusBar.SimpleText := floattostr(chisq);
  iter := 1;
  lastchisq := chisq;
  runningchidiff := 1;
  While true do
  begin
    Application.ProcessMessages;
    V2DLevMarq.mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,chisq,alamda);

    inc(iter);
    if iter > 1000 then break;
    chidiff := ((lastchisq-chisq)/(lastchisq))*1000000;
    if chidiff >= 0 then //there has been little or no decrease, but not an increase
    begin
      runningchidiff := runningchidiff * 0.8 + chidiff * 0.2;
      if runningchidiff < 0.000001{%} then break;
    end;
    {if iter=2 then
      StatusBar.SimpleText := floattostr(chisq)+' , '+floattostr(chidiff)+' , '+floattostr(runningchidiff);}
    lastchisq := chisq;
  end;
  alamda := 0.0;
  V2DLevMarq.mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,chisq,alamda);
  StatusBar.SimpleText := inttostr(iter)
                      +','+floattostrf(firstchisq,fffixed,8,2)
                      +' , '+floattostrf(chisq,fffixed,8,2);
end{RunLevenberg};

begin
  V2DLevMarq := V2DLevMarqObj.Create;
  //Setup LevMarq
  ndata := 0;
  s := '';
  for i := 0 to MAXDIST-1 do
    if v2dcurve[i].sumsqr > 0 then
    begin
      inc(ndata);
      x[ndata] := i;
      y[ndata] := v2dcurve[i].mean + 2*v2dcurve[i].std;
      //s := s + inttostr(round(y[ndata])) + ' ';
      sig[ndata] := 1;
    end;
  //Showmessage('npts='+inttostr(npts)+', ndata='+inttostr(ndata)+', pts='+s);
  //inverse square
  ma := 2;
  mfit := 2;
  a[1] := 3.0;{the m in y = mx + b}
  a[2] := 50;{the b in y = mx + b}
  lista[1] := 1;
  lista[2] := 2;
  //logistic
  (*ma := 3;
  mfit := 3;
  a[1] := m;{m}
  a[2] := b;{k or offset}
  a[3] := v0;{V0}
  lista[1] := 1;
  lista[2] := 2;
  lista[3] := 3;
  *)
  //Call it
  //ShowMessage('calling lev');
  V2DLevMarq.DOff := sqrt(V2uV/VINFINITE);
  RunLevenberg;
  (*ShowMessage({'V0='+inttostr(round(a[3]))+','+}
              'm='+inttostr(round(a[1]))+','+
              'b='+inttostr(round(a[2])));
  *)
  //so, ad = 1 / sqr( a[1]*d + a[2]);
  Canvas.Pen.Mode := pmCopy;
  Canvas.Pen.Color := clRed;
  Canvas.MoveTo(60,HalfHei-round(v2dcurve[0].Mean));
  For i := 1 to MAXDIST do
    if v2dcurve[i].SumSqr>0 then Canvas.LineTo(60+i,HalfHei-round(v2dcurve[i].Mean));

  Canvas.Pen.Color := clLime;
  Canvas.MoveTo(60,HalfHei-round((v2dcurve[0].Mean+2*v2dcurve[0].Std)));
  For i := 1 to MAXDIST do
    if v2dcurve[i].SumSqr>0 then Canvas.LineTo(60+i,HalfHei-round((v2dcurve[i].Mean+1*v2dcurve[i].Std)));

  Canvas.Pen.Color := clYellow;
  V2DLevMarq.func(-50,a,yfit,dyda,ma,mfit,FALSE);
  Canvas.MoveTo(60-50,HalfHei-round(yfit));
  For i := -50 to MAXDIST do
  begin
    V2DLevMarq.func(i,a,yfit,dyda,ma,mfit,FALSE);
    Canvas.LineTo(60+i,HalfHei-round(yfit));
  end;
  //v0 := a[3];
  m := a[1];
  b := a[2];
  V2DLevMarq.Free;
end;

{=========================================================================}
Procedure V2DLevMarqObj.CheckBounds(var a : DArrayNP); //parameters
//var k : integer;
begin
end;

{=========================================================================}
Procedure V2DLevMarqObj.func(xx : double;   //xindex
                             var a : DArrayNP; //parameters
                          var yfit : double;   //return
                          var dyda : DarrayNP; //derivatives wr2 params
                           ma,mfit : integer;  //numparams, numfit
                       computedyda : boolean); //compute the derivatives
var m,b,v0,rt,ex,num,den,den2,den3 : double;
begin
  //logistic function
  (*//logistic
  m := a[1];
  b := a[2];
  v0 := a[3];
  rt := sqrt(xx*xx + b*b);
  ex := exp(rt/m);
  yfit := v0/ex;
  if computedyda then
  begin
    dyda[1] := 1/ex;
    dyda[2] := yfit * rt / sqr(m);
    dyda[3] := -yfit * sqr(rt) * b / (m * rt);
  end;
  *)
  //inverse square
  m := a[1];
  b := a[2];
  den := m * xx + b + DOff;
  if den = 0 then den := 0.000001;
  den2 := den * den;
  den3 := den2*den;
  yfit := V2uV/den2;
  if computedyda then
  begin
    dyda[2] := -V2uV/den3;
    dyda[1] := xx*dyda[2];
  end;
end;

{=========================================================================}
procedure Tv2dform.FormShow(Sender: TObject);
var i,j : integer;
begin
  ClientWidth := MAXDIST+80;
  ClientHeight := MAXVOLT*2;// div ADPERPIX;
  Canvas.Brush.Color := clBlack;
  Canvas.FillRect(rect(0,0,ClientWidth,ClientHeight));
  For i := 0 to MAXDIST do
  With v2dcurve[i] do
  begin
    mean := 0;
    sum := 0;
    sumsqr := 0;
    std := 0;
    For j := 0 to NUMCURVESTOUSE-1 do val[j] := 9999;
  end;
  npts := 0;
  HalfHei := MAXVOLT;// div ADPERPIX;
  Canvas.Refresh;
end;

end.
