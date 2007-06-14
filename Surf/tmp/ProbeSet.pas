unit ProbeSet;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Spin, Menus, ProbeRowFormUnit, ExtCtrls, SurfTypes, SurfPublicTypes,
  Mask, PahUnit;
const
  MAXSAMPFREQ = 1250000;{Hz}
  MAXENTRIES = 1024;
type
  IndivProbeSetupRec = record
    ChanStart,NChannels,ChanEnd,NPtsPerChan,TrigPt,Lockout,
    SampFreq,Threshold,InternalGain,SkipPts : Integer;
    ProbeType : char;{POLYTRODE or CONTINUOUS}
    Descrip : ShortString;
    view,save,created : boolean;
  end;

  ProbeSetupRec = record
    NSpikeProbes,NCRProbes,NProbes,TotalChannels : Integer;
    Probe : array[0..SURF_MAX_PROBES-1] of IndivProbeSetupRec;
  end;

  TProbeRowFormObj = class(TProbeRowForm)
    public
      Procedure CheckProbeChannels(ProbeNum : integer); override;
  end;

  TProbeWin = class(TForm)
    Label22: TLabel;
    NSpikeProbeSpin: TSpinEdit;
    OkBut: TButton;
    CancelBut: TButton;
    Label11: TLabel;
    CreateProbes: TButton;
    ScrollBox: TScrollBox;
    Panel: TPanel;
    Panel1: TPanel;
    Label15: TLabel;
    Label16: TLabel;
    Label17: TLabel;
    Label18: TLabel;
    Label19: TLabel;
    Label20: TLabel;
    Label23: TLabel;
    Label24: TLabel;
    Label25: TLabel;
    Label26: TLabel;
    Label27: TLabel;
    Label1: TLabel;
    Label2: TLabel;
    NCRProbesSpin: TSpinEdit;
    Label3: TLabel;
    Label4: TLabel;
    TotFreq: TSpinEdit;
    SampFreqPerChan: TSpinEdit;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    DinCheckBox: TCheckBox;
    procedure OkButClick(Sender: TObject);
    procedure CancelButClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure CreateProbesClick(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure TotFreqChange(Sender: TObject);
    procedure SampFreqPerChanChange(Sender: TObject);
    procedure DinCheckBoxClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure FormHide(Sender: TObject);
  private
    { Private declarations }

    Procedure FreeProbeRows;
  public
    { Public declarations }
    Setup : ProbeSetupRec;
    Ok : boolean;
    ProbeRow : array[0..SURF_MAX_PROBES-1] of TProbeRowFormObj;
    Procedure CreateProbeRows;
  end;
var
  ProbeWin: TProbeWin;
  ready : boolean;
implementation

{$R *.DFM}

{=====================================================================}
procedure TProbeWin.OkButClick(Sender: TObject);
var i : integer;
begin
  ok := TRUE;
  Setup.TotalChannels := 0;
  For i := 0 to Setup.NProbes-1 do
  begin
    setup.probe[i].ChanStart := ProbeRow[i].ChanStartSpin.Value;
    setup.probe[i].NChannels := ProbeRow[i].NumChanSpin.Value;
    setup.probe[i].ChanEnd   := ProbeRow[i].ChanEndSpin.Value;
    setup.probe[i].NPtsPerChan:= ProbeRow[i].NPtsSpin.Value;
    setup.probe[i].TrigPt    := ProbeRow[i].TrigPtSpin.Value;
    setup.probe[i].Lockout   := ProbeRow[i].LockoutSpin.Value;
    setup.probe[i].Threshold := ProbeRow[i].ThresholdSpin.Value;
    setup.probe[i].SkipPts   := ProbeRow[i].SkipSpin.Value;
    setup.probe[i].InternalGain := StrToInt(ProbeRow[i].ADGainBox.Text);
    setup.probe[i].ProbeType := ProbeRow[i].Probetype;
    setup.probe[i].Descrip   := ProbeRow[i].ProbeDescription.Text;
    inc(Setup.TotalChannels,setup.probe[i].NChannels);
    //Check for bad settings
    //Basically, the total frequency has to work
    if setup.probe[i].SampFreq > MAXSAMPFREQ then Ok := FALSE;
    setup.probe[i].view      := ProbeRow[i].View.Checked;
    setup.probe[i].save      := ProbeRow[i].Save.Checked;
  end;
  Setup.NSpikeProbes := NSpikeProbeSpin.Value;
  Setup.NCRProbes := NCRProbesSpin.Value;

  if ProbeWin.DinCheckBox.Checked then inc(ProbeWin.Setup.TotalChannels);
  if ok then Close
  else ShowMessage('Setup error: Reduce sampling frequency or number of channels. Maximum frequencay is '+inttostr(MAXSAMPFREQ)+'.');
end;

{=====================================================================}
procedure TProbeWin.CancelButClick(Sender: TObject);
begin
  ok := FALSE;
  Close;
end;

{=====================================================================}
procedure TProbeWin.FormCreate(Sender: TObject);
var i : integer;
begin
  //Initialize a setup for me, deal with custom later
  Setup.NProbes := 0;
  For i := 0 to SURF_MAX_PROBES-1 do
    With setup.probe[i] do
    begin
      ChanStart := 0;
      NChannels := 0;
      ChanEnd := 0;
      NPtsPerChan := 32;
      TrigPt := 7;
      Lockout := 15;
      InternalGain := 1;
      SampFreq := 31980;
      Threshold := 1000;
      SkipPts := 1;
      ProbeType := SPIKETYPE;
      View := TRUE;
      Save := TRUE;
      Created := FALSE;
      Descrip := IntToStr(NChannels) + ' Channel Polytrode';
    end;
  DinCheckBox.Checked := FALSE;
  ok := FALSE;
  ready := TRUE;
end;

{=====================================================================}
procedure TProbeWin.CreateProbeRows;
var i : integer;
begin
  if Setup.NProbes = 0 then exit;
  ready := FALSE;
  For i := 0 to Setup.NProbes-1 do
  begin
    //Setup probe row form:
    ProbeRow[i] := TProbeRowFormObj.CreateParented(Panel.Handle);
    ProbeRow[i].Parent := Panel;
    ProbeRow[i].Tag := i;
    ProbeRow[i].Top := i*ProbeRow[i].Height;
    ProbeRow[i].Left := 5;
    ProbeRow[i].ProbeNum.Caption := IntToStr(i);
    ProbeRow[i].ChanStartSpin.Value := Setup.probe[i].chanstart;
    ProbeRow[i].NumChanSpin.Value := Setup.probe[i].nchannels;
    ProbeRow[i].ChanEndSpin.Value := Setup.probe[i].chanend;
    ProbeRow[i].Visible := TRUE;
    ProbeRow[i].ChannelsCreated := TRUE;
    ProbeRow[i].View.Checked := Setup.probe[i].View;
    ProbeRow[i].Save.Checked := Setup.probe[i].Save;
    ProbeRow[i].ADGainBox.Text := IntToStr(setup.probe[i].InternalGain);

    if Setup.probe[i].created
      then ProbeRow[i].ProbeDescription.Text := Setup.probe[i].Descrip
      else ProbeRow[i].ProbeDescription.Text := '';

    if i < NSpikeProbeSpin.Value then //Spike Probe
    begin
      ProbeRow[i].SkipSpin.Visible := FALSE;
      ProbeRow[i].SkipSpin.Value := 1;
      if Setup.probe[i].created
      then begin
        ProbeRow[i].NPtsSpin.Value := Setup.probe[i].nptsperchan;
        ProbeRow[i].SkipSpin.Value := Setup.probe[i].Skippts;
      end else
      begin
        ProbeRow[i].NPtsSpin.Value := 32;
        ProbeRow[i].SkipSpin.Value := 1;
      end;
      ProbeRow[i].probetype := SPIKETYPE;
    end else   //Continuous Probe
    begin
      ProbeRow[i].LockOutSpin.Visible := FALSE;
      ProbeRow[i].TrigPtSpin.Visible := FALSE;
      ProbeRow[i].ThresholdSpin.Visible := FALSE;
      ProbeRow[i].ChanEndSpin.Visible := FALSE;
      if Setup.probe[i].created
      then begin
        ProbeRow[i].NPtsSpin.Value := Setup.probe[i].nptsperchan;
        ProbeRow[i].SkipSpin.Value := Setup.probe[i].Skippts;
      end else
      begin
        ProbeRow[i].NPtsSpin.Value := 60;
        ProbeRow[i].SkipSpin.Value := 533;
        ProbeRow[i].ProbeDescription.Text := '';
      end;
      ProbeRow[i].SkipSpin.Enabled := TRUE;
      ProbeRow[i].NumChanSpin.Value := Setup.probe[i].nchannels;
      ProbeRow[i].NumChanSpin.MaxValue := 1;
      ProbeRow[i].probetype := CONTINUOUSTYPE;
    end;
    ProbeRow[i].ActualTimeLabel.Caption
      := FloatToStrF((ProbeRow[i].NptsSpin.Value * 1000 * ProbeRow[i].SkipSpin.Value) / SampFreqPerChan.Value, ffFixed, 4, 1);
    Setup.probe[i].Created := TRUE;
  end;


  TotFreq.Enabled := TRUE;
  SampFreqPerChan.Enabled := TRUE;

  Panel.Height :=  ProbeRow[Setup.NProbes-1].Top + ProbeRow[Setup.NProbes-1].Height + 4;
  if  Setup.NProbes <= 8 then
  begin
    Height := Panel.Height + 150;
    Width := 945;
    ScrollBox.Width := Width -5;
  end else
  begin
    Height := ProbeRow[7].Top + ProbeRow[7].Height + 4 + 150;
    Width := 960;
    ScrollBox.Width := Width -5;
    Panel.Height :=  ProbeRow[Setup.NProbes-1].Top + ProbeRow[Setup.NProbes-1].Height + 18;
  end;
  ready := TRUE;
end;

{=====================================================================}
procedure TProbeWin.FreeProbeRows;
var i : integer;
begin
  For i := 0 to Setup.Nprobes-1 do ProbeRow[i].Free;
end;

{=====================================================================}
procedure TProbeWin.FormDestroy(Sender: TObject);
begin
  FreeProbeRows;
end;

{=====================================================================}
procedure TProbeWin.CreateProbesClick(Sender: TObject);
begin
  Screen.Cursor := crHourGlass;
  FreeProbeRows;
  Setup.NProbes := NSpikeProbeSpin.Value + NCRProbesSpin.Value;
  CreateProbeRows;
  Screen.Cursor := crDefault;
end;

{=====================================================================}
procedure TProbeWin.FormResize(Sender: TObject);
begin
  ScrollBox.Height := Height-110;
  ScrollBox.Top := 100;
end;

{=====================================================================}
Procedure TProbeRowFormObj.CheckProbeChannels(ProbeNum : integer);
//Use this to check all other chanlists and compute maxfreq, etc.
var i,thisprobestart,thisprobeend,otherprobestart,otherprobeend : integer;
begin
  //check to make sure no other probes are using these channels
  if ProbeWin = nil then exit;
  if not ready then exit;
  if ProbeNum > ProbeWin.Setup.Nprobes-1 then exit;
  if ProbeWin.ProbeRow[ProbeNum] = nil then exit;

  With ProbeWin do
  if ProbeRow[ProbeNum].NumChanSpin.Value > 0 then
  begin
    For i := 0 to Setup.NProbes-1 do
      if (i <> ProbeNum)
        and (ProbeRow[i].NumChanSpin.Value > 0) then
      begin
        thisprobestart := ProbeRow[ProbeNum].ChanStartSpin.Value;
        otherprobestart := ProbeRow[i].ChanStartSpin.Value;
        thisprobeend := ProbeRow[ProbeNum].ChanEndSpin.Value;
        otherprobeend := ProbeRow[i].ChanEndSpin.Value;

        //if other probe starts within the range, move its start to the end of this range
        if (otherprobestart >= thisprobestart) and (otherprobestart <= thisprobeend)
        then otherprobestart := thisprobeend+1;
        //if other probe ends within the range, move its end to the start of this range
        if (otherprobeend >= thisprobestart) and (otherprobeend <= thisprobeend)
        then otherprobeend := thisprobestart-1;
        //now see if the other probe has a valid range
        if   (otherprobestart > otherprobeend)
          or (otherprobestart > SURF_MAX_CHANNELS-1)
          or (otherprobeend < 0) then
        begin
          ProbeRow[i].ChanStartSpin.Value := 0;
          ProbeRow[i].NumChanSpin.Value := 0;
        end else
        begin
          ProbeRow[i].ChanStartSpin.Value := otherprobestart;
          ProbeRow[i].NumChanSpin.Value := otherprobeend-otherprobestart+1;
        end;
      end;
    end;
  //set the tot freq for this probe
  ProbeWin.Setup.TotalChannels := 0;
  For i := 0 to ProbeWin.Setup.NProbes-1 do
  begin
    With ProbeWin.ProbeRow[i] do
    begin
      //NumChansLabel.caption := IntToStr(NumChanSpin.Value);
      inc(ProbeWin.Setup.TotalChannels,NumChanSpin.Value);
      if (probetype=CONTINUOUSTYPE) {and (NumChanSpin.Value > 0)} then
        {ProbeDescription.Text := 'Continuous Record Probe'}
      else if NumChanSpin.Value = 1 then
        ProbeDescription.Text := 'Single Channel Probe'
      else if NumChanSpin.Value = 2 then
        ProbeDescription.Text := 'Stereotrode'
      else if NumChanSpin.Value = 4 then
        ProbeDescription.Text := 'Tetrode'
      else if NumChanSpin.Value > 0 then
        ProbeDescription.Text := IntToStr(NumChanSpin.Value)+' Channel Polytrode'
      else if NumChanSpin.Value > 0 then
        ProbeDescription.Text := 'Not used'
      //TotFreq := NumChans*SampFreqSpin.Value;
      //TotalFreqLabel.Caption := IntToStr(TotFreq);
    end;
  end;
  if ProbeWin.DinCheckBox.Checked then inc(ProbeWin.Setup.TotalChannels);

  With ProbeWin do
  if Setup.TotalChannels <> 0 then
  begin
    TotFreq.Value := SampFreqPerChan.Value * Setup.TotalChannels;
    For i := 0 to Setup.NProbes-1 do
      if ProbeRow[i].probetype = SPIKETYPE
        then ProbeRow[i].SampFreq.Caption := inttostr(round(TotFreq.Value/Setup.TotalChannels))
        else if ProbeRow[i].SkipSpin.Value > 0 then ProbeRow[i].SampFreq.Caption := inttostr(round(TotFreq.Value/Setup.TotalChannels/ProbeRow[i].SkipSpin.Value));

    if TotFreq.Value > 0 then
      For i := 0 to Setup.NProbes-1 do
        ProbeRow[i].ActualTimeLabel.Caption
          := FloatToStrF((ProbeRow[i].NptsSpin.Value * 1000 * ProbeRow[i].SkipSpin.Value) / SampFreqPerChan.Value, ffFixed, 4, 1);
  end;
end;

{=====================================================================}
procedure TProbeWin.TotFreqChange(Sender: TObject);
var i : integer;
begin
  SampFreqPerChan.Value := round(ProbeWin.TotFreq.Value / ProbeWin.Setup.TotalChannels);
  if Setup.TotalChannels <> 0 then
    For i := 0 to Setup.NProbes-1 do
      ProbeRow[i].CheckProbeChannels(i);
end;

{=====================================================================}
procedure TProbeWin.SampFreqPerChanChange(Sender: TObject);
begin
  TotFreq.Value := SampFreqPerChan.Value * Setup.TotalChannels;
end;

{=====================================================================}
procedure TProbeWin.DinCheckBoxClick(Sender: TObject);
begin
  if DinCheckBox.Checked
    then inc(Setup.TotalChannels)
    else if Setup.TotalChannels > 0 then dec(Setup.TotalChannels);
  if Setup.TotalChannels > 0 then
    TotFreq.Value := SampFreqPerChan.Value * Setup.TotalChannels;
end;

procedure TProbeWin.FormShow(Sender: TObject);
begin
  ok := FALSE;
  //CreateProbesClick(nil);
end;

procedure TProbeWin.FormHide(Sender: TObject);
begin
  //FreeProbeRows;
  //Setup.NProbes := NSpikeProbeSpin.Value + NCRProbesSpin.Value;
end;

Initialization
    ready := FALSE;

end.
