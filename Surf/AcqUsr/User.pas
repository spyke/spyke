UNIT User;

INTERFACE
USES
  Windows, Messages, SysUtils, Classes, Graphics, Controls,
  Forms, Dialogs,  StdCtrls, ComCtrls, ExtCtrls, Spin,
  //surf units
  SurfPublicTypes, SurfAcq, SuperTimer;

CONST
  SAMPLE = 1;

TYPE
  TUserWin = class(TForm)
    spkcount: TLabel;
    Label1: TLabel;
    Label2: TLabel;
    crcount: TLabel;
    Label5: TLabel;
    dincount: TLabel;
    Label4: TLabel;
    Bits: TLabel;
    dinval: TLabel;
    MSByte: TLabel;
    LSByte: TLabel;
    Interval: TLabel;
    dinfreq: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label10: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    mseq: TLabel;
    mseqlabel: TLabel;
    Label14: TLabel;
    Label16: TLabel;
    Ori: TLabel;
    orilabel: TLabel;
    orideglabel: TLabel;
    phase: TLabel;
    phaselabel: TLabel;
    phasedeglabel: TLabel;
    X: TLabel;
    xylabel: TLabel;
    Y: TLabel;
    con: TLabel;
    consflabel: TLabel;
    sf: TLabel;
    len: TLabel;
    lenwidlabel: TLabel;
    wid: TLabel;
    xydeglabel: TLabel;
    lenwiddeglabel: TLabel;
    Experiment: TComboBox;
    Label17: TLabel;
    Label3: TLabel;
    Label8: TLabel;
    Shape1: TShape;
    Shape2: TShape;
    Shape3: TShape;
    updatebox: TCheckBox;
    SurfAcq: TSurfAcq;
    Button1: TButton;
    lval: TEdit;
    dval: TTrackBar;
    Label9: TLabel;
    freq: TSpinEdit;
    SendDAC: TButton;
    dioval: TSpinEdit;
    procedure FormCreate(Sender: TObject);
   // procedure ExperimentChange(Sender: TObject);
    procedure SurfAcqSpike(Spike: TSpike);
    procedure SurfAcqSurfMsg(Msg: TSurfMsg);
    procedure SurfAcqSV(SV: TSVal);
    procedure SurfAcqCR(Cr: TCr);
    procedure Button1Click(Sender: TObject);
    procedure dvalChange(Sender: TObject);
    procedure OutputDA;
    procedure SendDACClick(Sender: TObject);
    procedure diovalChange(Sender: TObject);
  private
    nspk,ncr,ndin : Longint;

    Function  BitOn(w : WORD; Bit : integer) : boolean;
    procedure SetDinBits(w : word);
  public
    //DO NOT WRITE IN THIS AREA
    {procedure ShowWin (var msg : TMessage); message WM_USERWINSHOW;
    procedure CloseWin(var msg : TMessage); message WM_USERWINCLOSE;
    procedure Ping    (var msg : TMessage); message WM_TIMERPING;
    Procedure NewSpike(var msg : TMessage); message WM_SPKAVAIL;
    Procedure NewCR   (var msg : TMessage); message WM_CRAVAIL;
    Procedure NewDIN  (var msg : TMessage); message WM_DINAVAIL;
    Procedure AcqStart(var msg : TMessage); message WM_ACQSTART;
    Procedure AcqStop (var msg : TMessage); message WM_ACQSTOP;
    Procedure RecStart(var msg : TMessage); message WM_RECSTART;
    Procedure RecStop (var msg : TMessage); message WM_RECSTOP;
    }
  end;

VAR
  UserWin: TUserWin;

implementation

{$R *.DFM}

{--------------------------------------------------------------------------------}
(*procedure TUserWin.Ping;
{Timer pings occur once every ms, unless interfering messages (such as moving and
 resizing windows) occur.}
begin
  //Insert code here to respond to timer pings
end;

{--------------------------------------------------------------------------------}
procedure TUserWin.ShowWin;
begin
  Show;
  Update;
end;
{--------------------------------------------------------------------------------}
procedure TUserWin.CloseWin;
begin
  Hide;
  Update;
end;
*)
(*
{--------------------------------------------------------------------------------}
procedure TUserWin.NewSpike;
begin
  //Insert code here to respond to new spikes
  inc(nspk);
  if updatebox.Checked then spkcount.caption := inttostr(nspk);
end;
*)
(*
{--------------------------------------------------------------------------------}
procedure TUserWin.NewCR;
const XPOSCHANNUM = 26;
      AD2DEG = 81.92;
var crrec : PolyTrodeRecord;
    crindex,npts,channum,val,meanval: Longint;
    i,j,startchan : SHORT;
    sval : string;
begin
  //Insert code here to respond to new crs
  inc(ncr);
  if not updatebox.Checked then exit;
  crcount.caption := inttostr(ncr);

  crindex := SurfAcqForm.GetCRRingWriteIndex;
  crrec   := SurfAcqForm.GetCRFromRing(crindex);
  channum := SurfAcqForm.GetChanStartNumberForProbe(crrec.probenum);
  startchan := XPOSCHANNUM-channum;

  For i := startchan to startchan+5 do
  begin
    crrec := SurfAcqForm.GetCRFromRing(crindex+i);
    if crrec.waveform = NIL then break;
    npts := length(crrec.waveform);
    //val := crrec.waveform[npts-1];
    meanval := 0;
    for j := npts-10 to npts-1 do
    begin
      val := crrec.waveform[j];
      meanval := meanval + val;
    end;
    meanval := round(meanval / 10);
    meanval := val;
    sval := inttostr(meanval);
    case (channum+i) of
      26 :{X} if x.enabled then x.caption := floattostrf((meanval-2047)/AD2DEG,ffFixed,4,2);
      27 :{Y} if y.enabled then y.caption := floattostrf((meanval-2047)/AD2DEG,ffFixed,4,2);
      28 :{C} if con.enabled then con.caption := sval;
      29 :{S} if sf.enabled then sf.caption  := sval;
      30 :{L} if len.enabled then len.caption := floattostrf(meanval/AD2DEG,ffFixed,4,2);
      31 :{W} if wid.enabled then wid.caption := floattostrf(meanval/AD2DEG,ffFixed,4,2);
    end;
  end;
end;
*)
(*
{--------------------------------------------------------------------------------}
procedure TUserWin.NewDIN;
var w,lsb,msb,orival,phaseval,mseqval : WORD;
  lastdin,din : DinRecordType;
  timediff,dindex : longint;
begin
  //Insert code here to respond to new digital inputs
  inc(ndin);

  dindex := SurfAcqForm.GetDINRingWriteIndex;
  din := SurfAcqForm.GetDINFromRing(dindex);
  lastdin := SurfAcqForm.GetDINFromRing(dindex-1);

  timediff := din.time-lastdin.time;

  if timediff < 0 then exit;
  if not updatebox.Checked then exit;

  interval.caption := inttostr(timediff);
  if timediff <> 0 then
    dinfreq.caption := inttostr(round(10000 / timediff));

  w := din.DinVal;
  dinval.caption := inttostr(w);
  dincount.caption := inttostr(ndin);
  //SetDinBits(w);

  msb := w and $00FF; {get the last byte of this word}
  lsb := w shr 8;      {get the first byte of this word}
  MSByte.caption := inttostr(msb);
  LSByte.caption  := inttostr(lsb);

  if ori.Enabled then
  begin
    orival := (msb and $01) shl 8 + lsb; {get the last bit of the msb}
    Ori.caption := inttostr(orival);
  end;

  if phase.enabled then
  begin
    phaseval := msb shr 1;{get the first 7 bits of the msb}
    phase.caption := inttostr(phaseval);
  end;

  if mseq.enabled then
  begin
    mseqval := msb*256+lsb;//(msb and $03) shl 8 + lsb; {get the last 2 bits of the msb}
    mseq.caption := inttostr(mseqval);
  end;
end;
*)
(*
{--------------------------------------------------------------------------------}
procedure TUserWin.AcqStart;
begin
  //Insert code here to respond to start acquisition
  nspk := 0;
  ncr := 0;
  ndin := 0;
  if not updatebox.Checked then exit;
  spkcount.caption  := inttostr(nspk);
  crcount.caption   := inttostr(ncr);
  dincount.caption  := inttostr(ndin);
end;
*)
(*
{--------------------------------------------------------------------------------}
procedure TUserWin.AcqStop;
begin
  //Insert code here to respond to stop acquisition
end;
{--------------------------------------------------------------------------------}
procedure TUserWin.RecStart;
begin
  //Insert code here to respond to start recording
end;

{--------------------------------------------------------------------------------}
procedure TUserWin.RecStop;
begin
  //Insert code here to respond to stop recording
end;
*)
{--------------------------------------------------------------------------------}
procedure TUserWin.FormCreate(Sender: TObject);
begin
  nspk := 0;
  ncr := 0;
  ndin := 0;
  Experiment.ItemIndex := 0;{M-Sequence}
  //Experiment.ItemIndex := 1;{Moving Bar}
  //Experiment.ItemIndex := 2;{Gratings}
end;

{--------------------------------------------------------------------------------}
Function TUserWin.BitOn(w : WORD; Bit : integer) : boolean;
const
  BIT0   =   $0001;      // bit #0 value
  BIT1   =   $0002;      // bit #1 value
  BIT2   =   $0004;      // bit #2 value
  BIT3   =   $0008;      // bit #3 value
  BIT4   =   $0010;      // bit #4 value
  BIT5   =   $0020;      // bit #5 value
  BIT6   =   $0040;      // bit #6 value
  BIT7   =   $0080;      // bit #7 value
  BIT8   =   $0100;      // bit #8 value
  BIT9   =   $0200;      // bit #9 value
  BIT10  =   $0400;      // bit #10 value
  BIT11  =   $0800;      // bit #11 value
  BIT12  =   $1000;      // bit #12 value
  BIT13  =   $2000;      // bit #13 value
  BIT14  =   $4000;      // bit #14 value
  BIT15  =   $8000;      // bit #15 value
begin
  BitOn := FALSE;
  case bit of
    0 : if w AND BIT0 <> 0 then Biton := TRUE;
    1 : if w AND BIT1 <> 0 then Biton := TRUE;
    2 : if w AND BIT2 <> 0 then Biton := TRUE;
    3 : if w AND BIT3 <> 0 then Biton := TRUE;
    4 : if w AND BIT4 <> 0 then Biton := TRUE;
    5 : if w AND BIT5 <> 0 then Biton := TRUE;
    6 : if w AND BIT6 <> 0 then Biton := TRUE;
    7 : if w AND BIT7 <> 0 then Biton := TRUE;
    8 : if w AND BIT8 <> 0 then Biton := TRUE;
    9 : if w AND BIT9 <> 0 then Biton := TRUE;
    10 : if w AND BIT10 <> 0 then Biton := TRUE;
    11 : if w AND BIT11 <> 0 then Biton := TRUE;
    12 : if w AND BIT12 <> 0 then Biton := TRUE;
    13 : if w AND BIT13 <> 0 then Biton := TRUE;
    14 : if w AND BIT14 <> 0 then Biton := TRUE;
    15 : if w AND BIT15 <> 0 then Biton := TRUE;
  end;
end;

{--------------------------------------------------------------------------------}
procedure TUserWin.SetDinBits(w : word);
var s : string;
    i : integer;
begin
  s := '';
  for i := 0 to 15 do
    if BitOn(w,i) then s := '+'+s else s := '-'+s;
  Bits.Caption := s;
end;

(*procedure TUserWin.ExperimentChange(Sender: TObject);
begin
  Case Experiment.ItemIndex of
    0:{mseq} begin
               {digital}
               mseq.enabled := TRUE; mseq.show; mseqlabel.show;
               ori.enabled := FALSE; ori.hide; orilabel.hide; orideglabel.hide;
               phase.enabled := FALSE;phase.hide; phaselabel.hide; phasedeglabel.hide;
               {cr}
               x.enabled := TRUE; x.show; xylabel.show; xydeglabel.show;
               y.enabled := TRUE; y.show;
               con.enabled := TRUE; con.show; consflabel.show;
               sf.enabled := FALSE;  sf.hide;
               len.enabled := TRUE; len.Show; lenwidlabel.Show; lenwiddeglabel.Show;
               wid.enabled := TRUE; wid.Show;
             end;
    1:{bar}  begin
               {digital}
               mseq.enabled := FALSE; mseq.hide; mseqlabel.hide;
               ori.enabled := TRUE;   ori.show; orilabel.show; orideglabel.show;
               phase.enabled := FALSE;phase.hide; phaselabel.hide; phasedeglabel.hide;
               {cr}
               x.enabled := TRUE; x.show; xylabel.show; xydeglabel.show;
               y.enabled := TRUE; y.show;
               con.enabled := TRUE; con.show; consflabel.show;
               sf.enabled := FALSE; sf.hide;
               len.enabled := TRUE; len.Show; lenwidlabel.show; lenwiddeglabel.show;
               wid.enabled := TRUE; wid.show;
             end;
    2:{grat} begin
               {digital}
               mseq.enabled := FALSE; mseq.hide; mseqlabel.hide;
               ori.enabled := TRUE;   ori.show; orilabel.show; orideglabel.show;
               phase.enabled := TRUE; phase.show; phaselabel.show; phasedeglabel.show;
               {cr}
               x.enabled := TRUE; x.show; xylabel.show; xydeglabel.show;
               y.enabled := TRUE; y.show;
               con.enabled := TRUE; con.show; consflabel.show;
               sf.enabled := TRUE;  sf.show;
               len.enabled := TRUE; len.Show; lenwidlabel.show; lenwiddeglabel.show;
               wid.enabled := TRUE; wid.show;
             end;
  end;
end;
*)
procedure TUserWin.SurfAcqSpike(Spike: TSpike);
begin
beep;
end;

procedure TUserWin.SurfAcqSurfMsg(Msg: TSurfMsg);
begin
beep;
end;

procedure TUserWin.SurfAcqSV(SV: TSVal);
var w,lsb,msb,orival,phaseval,mseqval : WORD;
  //lastdin,din : DinRecordType;
  timediff,dindex : longint;
begin
  //Insert code here to respond to new digital inputs
  inc(ndin);

  {dindex := SurfAcqForm.GetDINRingWriteIndex;
  din := SurfAcqForm.GetDINFromRing(dindex);
  lastdin := SurfAcqForm.GetDINFromRing(dindex-1);

  timediff := din.time-lastdin.time;

  if timediff < 0 then exit;
  }
  if not updatebox.Checked then exit;

  {interval.caption := inttostr(timediff);
  if timediff <> 0 then
    dinfreq.caption := inttostr(round(10000 / timediff));
  }
  w := sv.SVal;
  dinval.caption := inttostr(w);
  dincount.caption := inttostr(ndin);
  SetDinBits(w);

  msb := w and $00FF; {get the last byte of this word}
  lsb := w shr 8;      {get the first byte of this word}
  MSByte.caption := inttostr(msb);
  LSByte.caption  := inttostr(lsb);

  if ori.Enabled then
  begin
    orival := (msb and $01) shl 8 + lsb; {get the last bit of the msb}
    Ori.caption := inttostr(orival);
  end;

  if phase.enabled then
  begin
    phaseval := msb shr 1;{get the first 7 bits of the msb}
    phase.caption := inttostr(phaseval);
  end;

  if mseq.enabled then
  begin
    mseqval := msb*256+lsb;//(msb and $03) shl 8 + lsb; {get the last 2 bits of the msb}
    mseq.caption := inttostr(mseqval);
  end;
end;

procedure TUserWin.SurfAcqCR(Cr: TCr);
begin
beep;
end;

procedure TUserWin.dvalChange(Sender: TObject);
const VOLTTOVAL = 2048/10;
begin
  lval.text := floattostrf(dval.position/100,fffixed,4,2);
  if freq.value = 0 then OutPutDA;
end;

procedure TUserWin.OutputDA;
var dac : TDAC;
begin
  dac.channel := 0;//channel 0
  dac.voltage := dval.position/100;
  dac.frequency := freq.value;
  SurfAcq.SendDACToSurf(dac);
end;

procedure TUserWin.SendDACClick(Sender: TObject);
begin
  OutputDA;
end;

procedure TUserWin.Button1Click(Sender: TObject);
var dio : TDIO;
begin
  //dio.mask := $FFFF;
  dio.val := dioval.value;//$FF00;
  SurfAcq.SendDIOToSurf(dio);
end;

procedure TUserWin.diovalChange(Sender: TObject);
var dio : TDIO;
begin
  //dio.mask := $FFFF;
  dio.val := dioval.value;//$FF00;
  SurfAcq.SendDIOToSurf(dio);
end;

end.
