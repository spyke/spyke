{ (c) 1994-1999 Phil Hetherington, P&M Research Technologies, Inc.}
{ (c) 2000-2003 Tim Blanche, University of British Columbia }
unit ProbeSet;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Spin, Menus, ProbeRowFormUnit, ExtCtrls, SurfTypes, SurfPublicTypes,
  ElectrodeTypes;

const
  MAXSAMPFREQPERBOARD = 1000000;

type
  TIndivProbeSetup = record
    ChanStart, NChannels, ChanEnd, NPtsPerChan, TrigPt, Lockout,
    SampFreq, Threshold, InternalGain, SkipPts : integer;
    ProbeType : char;{SPIKESTREAM, SPIKEEPOCH or CONTINUOUS}
    Descrip, ElectrodeName : ShortString;
    View, Save, Created : boolean;
  end;

  TProbeSetup = record
    NSpikeStreamProbes, NSpikeEpochProbes, NCRProbes, NProbes, TotalChannels : Integer;
    Probe : array[0..SURF_MAX_PROBES-1] of TIndivProbeSetup;
  end;

  TProbeRowFormObj = class(TProbeRowForm)
  public
    procedure CheckProbeChannels(ProbeNum : integer); override;
  end;

  TProbeSetupWin = class(TForm)
    Label22: TLabel;
    NSpikeProbeSpin: TSpinEdit;
    OkBut: TButton;
    CancelBut: TButton;
    CreateProbes: TButton;
    ScrollBox: TScrollBox;
    Panel: TPanel;
    ProbeRowTitlePanel: TPanel;
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
    Label2: TLabel;
    NCRProbesSpin: TSpinEdit;
    Label3: TLabel;
    SampFreqPerChan: TSpinEdit;
    Label5: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    DINCheckBox: TCheckBox;
    Label9: TLabel;
    Label10: TLabel;
    NCRSpikeProbeSpin: TSpinEdit;
    gb_HardwareCaps: TGroupBox;
    Label12: TLabel;
    Label13: TLabel;
    Label14: TLabel;
    Label21: TLabel;
    Label28: TLabel;
    lb_ADCTotFreq: TLabel;
    Label29: TLabel;
    lb_ADChans: TLabel;
    lb_NumADBoards: TLabel;
    lb_MUXSS: TLabel;
    lb_TimerSS: TLabel;
    lb_DINSS: TLabel;
    Label1: TLabel;
    Label4: TLabel;
    procedure OKButClick(Sender: TObject);
    procedure CancelButClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure CreateProbesClick(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure NCRSpikeProbeSpinChange(Sender: TObject);
    procedure NCRProbesSpinChange(Sender: TObject);
    procedure NSpikeProbeSpinChange(Sender: TObject);
    procedure SampFreqPerChanChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
  private
    { Private declarations }
    procedure RefreshProbeButton;
    procedure FreeProbeRows;
    procedure ResetProbeSetup;
  public
    { Public declarations }
    MaxADChannels : integer;
    Setup : TProbeSetup;
    OK : boolean;
    ProbeRow : array[0..SURF_MAX_PROBES-1] of TProbeRowFormObj;
    procedure CreateProbeRows;
    function  CalcActualFreqPerChan(DesiredSampFreq : integer) : integer; virtual; abstract;
  end;
var
  ProbeSetupWin: TProbeSetupWin;
  Ready : boolean;
implementation

{$R *.DFM}

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.OKButClick(Sender: TObject);
var i{, MaxRetriggerFreq} : integer;
begin
  {final check for bad/absent settings}
  CalcActualFreqPerChan(SampFreqPerChan.Value);
  OK:= True;
  Setup.TotalChannels := 0;
  for i := 0 to Setup.NProbes - 1 do
    with Setup.probe[i] do
    begin
      ProbeType := ProbeRow[i].Probetype;
      ChanStart := ProbeRow[i].ChanStartSpin.Value;
      NChannels := ProbeRow[i].NumChanSpin.Value;
      ChanEnd   := ProbeRow[i].ChanEndSpin.Value;
      NPtsPerChan:= ProbeRow[i].NPtsSpin.Value;
      TrigPt    := ProbeRow[i].TrigPtSpin.Value;
      Lockout   := ProbeRow[i].LockoutSpin.Value;
      SampFreq  := StrToInt(ProbeRow[i].lblSampFreq.Caption);
      Threshold := ProbeRow[i].ThresholdSpin.Value;
      SkipPts   := ProbeRow[i].SkipSpin.Value;
      InternalGain := StrToInt(ProbeRow[i].ADGainBox.Text);
      ElectrodeName := ProbeRow[i].CElectrode.Text;
      Descrip   := ProbeRow[i].ProbeDescription.Text;
      inc(Setup.TotalChannels, NChannels);
      if ProbeRow[i].CElectrode.Text = 'UnDefined' then
      begin //an electrode must be selected from the list of known types
        Showmessage('Please select an electrode type for probe '+ Inttostr(i));
        ok := False;
        Exit;
      end;
      View := ProbeRow[i].View.Checked;
      Save := ProbeRow[i].Save.Checked;
    end;
  {MaxRetriggerFreq:= Round (1 / (Round(Setup.TotalChannels / StrToInt(lb_NumADBoards.Caption)) /
    (StrToFloat(lb_ADCTotFreq.Caption) / StrToInt(lb_NumADBoards.Caption) * 1000000) + 1.2e-6));
  if SampFreqPerChan.Value > MaxRetriggerFreq then //check if SampFreqPerChan doesn't exceed the
  begin                                            //MaxRetriggerFreq for the total number of channels setup
    ShowMessage('Reduce sampling frequency or number of channels. The maximum recommended'
              + Chr(13) + 'frequency (for ' + Inttostr(Setup.TotalChannels) + ' channels) is '
              + InttoStr(MaxRetriggerFreq) + ' Hz per channel.');
    ok := False;
    exit;
  end;}
  if OK then Close;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.CancelButClick(Sender: TObject);
begin
  OK:= False;
  Close;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.FormCreate(Sender: TObject);
begin
  ResetProbeSetup;
  DINCheckBox.Checked := False;
  OK:= False;
  Ready:= True;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.ResetProbeSetup;
var i : integer;
begin
  Setup.NProbes:= 0;
  for i := 0 to SURF_MAX_PROBES - 1 do //initialise default probe values
    with Setup.probe[i] do
    begin
      ProbeType    := SPIKESTREAM;
      ChanStart    := 0;
      NChannels    := 0;
      ChanEnd      := 0;
      NPtsPerChan  := 25;
      TrigPt       := 7;
      Lockout      := 15;
      InternalGain := 8;
      SampFreq     := 0;
      Threshold    := 500;
      SkipPts      := 1;
      View         := True;
      Save         := True;
      Created      := False;
      ElectrodeName:= 'Undefined';
      Descrip := IntToStr(NChannels) + ' Channel Polytrode';
    end;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.CreateProbeRows;
var i,j : integer;
begin
  if Setup.NProbes = 0 then Exit;
  Ready := False;
  for i := 0 to Setup.NProbes - 1 do
  begin  //Setup probe row forms...
    ProbeRow[i] := TProbeRowFormObj.CreateParented(Panel.Handle);
    with ProbeRow[i] do
    begin
      TotalADChannels:= MaxADChannels; //pass max number of A/D channels to ProbeRowForm
      ChanStartSpin.MaxValue:= MaxADChannels - 1;
      NumChanSpin.MaxValue:= MaxADChannels;
      ChanEndSpin.MaxValue:= MaxADChannels - 1;
      Parent:= Panel;
      Tag:= i;
      Top:= i*ProbeRow[i].Height;
      Left:= 2;
      ProbeNum.Caption:= IntToStr(i);
      ChanStartSpin.Value:= Setup.probe[i].chanstart;
      NumChanSpin.Value:= Setup.probe[i].nchannels;
      ChanEndSpin.Value:= Setup.probe[i].chanend;
      Visible:= True;
      ChannelsCreated:= True;
      View.Checked:= Setup.probe[i].View;
      Save.Checked:= Setup.probe[i].Save;
      ADGainBox.Text:= IntToStr(Setup.probe[i].InternalGain);
      case Setup.probe[i].InternalGain of //assumes DT3010 board
        1 : ADGainBox.ItemIndex:= 0;
        2 : ADGainBox.ItemIndex:= 1;
        4 : ADGainBox.ItemIndex:= 2;
        8 : ADGainBox.ItemIndex:= 3;
      else ADGainBox.ItemIndex:= 0;
      end;
    end;
    if Setup.Probe[i].Created
      then ProbeRow[i].ProbeDescription.Text := Setup.probe[i].Descrip
      else ProbeRow[i].ProbeDescription.Text := '';
    for j := 0 to KNOWNELECTRODES-1 do
      if Setup.Probe[i].ElectrodeName = KnownElectrode[j].Name then
        ProbeRow[i].CElectrode.ItemIndex := j;
    if Setup.Probe[i].Created and (Setup.Probe[i].ProbeType = SPIKESTREAM)
      xor (i < NCRSpikeProbeSpin.Value) then // modify probe row for SpikeStream probe(s)...
    begin
      {ProbeRow[i].NPtsSpin.Visible:= False;
      ProbeRow[i].ThresholdSpin.Visible:= False;
      ProbeRow[i].TrigPtSpin.Visible:= False;}
      ProbeRow[i].LockOutSpin.Visible:= False;
      ProbeRow[i].SkipSpin.Visible:= False;
      if Setup.Probe[i].Created then
      begin
        ProbeRow[i].NptsSpin.Value:= Setup.probe[i].NPtsPerChan;
        ProbeRow[i].ThresholdSpin.Value:= Setup.probe[i].Threshold;
        ProbeRow[i].TrigPtSpin.Value:= Setup.probe[i].TrigPt;
      end else
      begin
        ProbeRow[i].NptsSpin.Value:= SampFreqPerChan.Value div 1000; //default ~1ms
        ProbeRow[i].ThresholdSpin.Value:= 500;
        ProbeRow[i].TrigPtSpin.Value:= 7;
      end;
      ProbeRow[i].LockOutSpin.Value:= 0; //not applicable
      ProbeRow[i].SkipSpin.Value := 1;   //no decimation
      ProbeRow[i].Probetype := SPIKESTREAM;
    end else
    if Setup.Probe[i].Created and (Setup.Probe[i].ProbeType = SPIKEEPOCH)
      xor (i < (NSpikeProbeSpin.Value + NCRSpikeProbeSpin.Value)) then //modify for EpochSpike probe(s)...
    begin
      ProbeRow[i].SkipSpin.Visible := False;
      if Setup.Probe[i].Created
      then begin
        ProbeRow[i].NPtsSpin.Value:= Setup.probe[i].NPtsperchan;
        ProbeRow[i].ThresholdSpin.Value:= Setup.probe[i].Threshold;
        ProbeRow[i].TrigPtSpin.Value:= Setup.probe[i].TrigPt;
        ProbeRow[i].LockOutSpin.Value:= Setup.probe[i].Lockout;
        ProbeRow[i].SkipSpin.Value:= Setup.probe[i].Skippts;
      end else
      begin
        ProbeRow[i].NptsSpin.Value:= SampFreqPerChan.Value div 1000; //default ~1ms
        ProbeRow[i].ThresholdSpin.Value:= 500;
        ProbeRow[i].TrigPtSpin.Value:= 7;
        ProbeRow[i].LockOutSpin.Value:= 15;
        ProbeRow[i].SkipSpin.Value:= 1;
      end;
      ProbeRow[i].Probetype := SPIKEEPOCH;
    end else //...finally, modify probe row for CR probe(s)
    begin
      ProbeRow[i].LockOutSpin.Visible:= False;
      ProbeRow[i].TrigPtSpin.Visible:= False;
      ProbeRow[i].ThresholdSpin.Visible:= False;
      ProbeRow[i].ChanEndSpin.Visible:= False;
      if Setup.probe[i].Created
      then begin
        ProbeRow[i].NPtsSpin.Value:= Setup.probe[i].NPtsPerChan;
        ProbeRow[i].SkipSpin.Value:= Setup.probe[i].Skippts;
      end else
      begin
        ProbeRow[i].SkipSpin.Value:= SampFreqPerChan.Value div 1000; //default 1:1000 decimation
        ProbeRow[i].NptsSpin.Value:= 100; //default ~1/10th sec
        ProbeRow[i].ProbeDescription.Text := '';
      end;
      ProbeRow[i].LockOutSpin.Value:= 0; //not applicable
      ProbeRow[i].SkipSpin.Enabled:= True;
      ProbeRow[i].NumChanSpin.Value:= Setup.Probe[i].NChannels;
      ProbeRow[i].NumChanSpin.MaxValue:= 1;
      ProbeRow[i].Probetype:= CONTINUOUS;
    end;
    if ProbeRow[i].SkipSpin.Value > 0 then ProbeRow[i].lblSampFreq.Caption:=
      Inttostr(Round(SampFreqPerChan.Value/ProbeRow[i].SkipSpin.Value));
    {if ProbeRow[i].Probetype <> SPIKESTREAM then} ProbeRow[i].ActualTimeLabel.Caption:=
      FloatToStrF((ProbeRow[i].NptsSpin.Value * 1000 * ProbeRow[i].SkipSpin.Value) / SampFreqPerChan.Value, ffFixed, 4, 1);
      {else ProbeRow[i].Label1.Caption:= 'Continuous';}
    Setup.Probe[i].Created := True;
  end;
  SampFreqPerChan.Enabled := True; //now user-modifiable in both versions of SURF
  Panel.Height :=  ProbeRow[Setup.NProbes-1].Top + ProbeRow[Setup.NProbes-1].Height + 4;
  if Setup.NProbes <= 8 then
  begin
    Height := Panel.Height + 128;
    Width := 1035;
    ScrollBox.Width := Width - 5;
  end else
  begin
    Height := ProbeRow[7].Top + ProbeRow[7].Height + 4 + 150;
    Width := 1050;
    ScrollBox.Width := Width - 5;
    Panel.Height :=  ProbeRow[Setup.NProbes-1].Top + ProbeRow[Setup.NProbes-1].Height + 25;
  end;
  Ready := True;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.FreeProbeRows;
var p : integer;
begin
  for p:= 0 to Setup.NProbes - 1 do
    ProbeRow[p].Free;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.CreateProbesClick(Sender: TObject);
begin
  Screen.Cursor := crHourGlass;
  FreeProbeRows;
  ResetProbeSetup;
  Setup.NProbes:= NCRSpikeProbeSpin.Value + NSpikeProbeSpin.Value + NCRProbesSpin.Value;
  Setup.NSpikeStreamProbes:= NCRSpikeProbeSpin.Value;
  Setup.NSpikeEpochProbes := NSpikeProbeSpin.Value;
  Setup.NCRProbes         := NCRProbesSpin.Value;
  CreateProbeRows;
  RefreshProbeButton;
  Screen.Cursor := crDefault;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.RefreshProbeButton;
begin
  if Setup.NProbes = 0 then CreateProbes.Caption:= 'Create &Probes'
    else CreateProbes.Caption:= 'Update &Probes';
  if Setup.NProbes = (NCRSpikeProbeSpin.Value + NSpikeProbeSpin.Value + NCRProbesSpin.Value) then
    CreateProbes.Enabled:= False else CreateProbes.Enabled:= True;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.FormResize(Sender: TObject);
begin
  ScrollBox.Height:= Height-100;
  ScrollBox.Top:= 100;
end;

{-------------------------------------------------------------------------}
procedure TProbeRowFormObj.CheckProbeChannels(ProbeNum : integer);
{this is to check all other chanlists in response to user changes to a probe entry.
 procedure could do with some improvements in both clarity and functionality!}
var i, thisprobestart, thisprobeend, otherprobestart, otherprobeend : integer;
begin
  {check to make sure no other probes are using these channels}
  if ProbeSetupWin = nil then Exit;
  if not Ready then Exit;
  if ProbeNum > ProbeSetupWin.Setup.Nprobes-1 then Exit;
  if ProbeSetupWin.ProbeRow[ProbeNum] = nil then Exit;

  with ProbeSetupWin do
  if ProbeRow[ProbeNum].NumChanSpin.Value > 0 then
  begin
    for i := 0 to Setup.NProbes-1 do
      if (i <> ProbeNum)
        and (ProbeRow[i].NumChanSpin.Value > 0) then
      begin
        thisprobestart := ProbeRow[ProbeNum].ChanStartSpin.Value;
        otherprobestart := ProbeRow[i].ChanStartSpin.Value;
        thisprobeend := ProbeRow[ProbeNum].ChanEndSpin.Value;
        otherprobeend := ProbeRow[i].ChanEndSpin.Value;

        //if other probe starts within the range, move its start to the end of this range
        if (otherprobestart >= thisprobestart) and (otherprobestart <= thisprobeend)
        then otherprobestart := thisprobeend + 1;
        //if other probe ends within the range, move its end to the start of this range
        if (otherprobeend >= thisprobestart) and (otherprobeend <= thisprobeend)
        then otherprobeend := thisprobestart - 1;
        //now see if the other probe has a valid range
        if   (otherprobestart > otherprobeend)
          or (otherprobestart > MaxADChannels - 1)
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
  ProbeSetupWin.Setup.TotalChannels := 0;
  for i := 0 to ProbeSetupWin.Setup.NProbes-1 do
  begin
    with ProbeSetupWin.ProbeRow[i] do
    begin
      //NumChansLabel.caption := IntToStr(NumChanSpin.Value);
      inc(ProbeSetupWin.Setup.TotalChannels, NumChanSpin.Value);
      if (probetype = CONTINUOUS) {and (NumChanSpin.Value > 0)} then
        {ProbeDescription.Text := 'Continuous Record Probe'}
      else if NumChanSpin.Value > 0 then
      begin
        if (CElectrode.ItemIndex < KNOWNELECTRODES)
        and (CElectrode.ItemIndex >= 0)
          then ProbeDescription.Text := KnownElectrode[CElectrode.ItemIndex].Description
          else ProbeDescription.Text := 'UnDefined';
      end else
        ProbeDescription.Text := 'Unused';
      {case NumChanSpin.Value of
        1 : //ProbeDescription.Text:= 'Single Channel Probe';
        2 : ProbeDescription.Text:= 'Stereotrode';
        4 : ProbeDescription.Text:= 'Tetrode';
      else if NumChanSpin.Value > 0 then
        ProbeDescription.Text := IntToStr(NumChanSpin.Value)+' Channel Polytrode'
          else ProbeDescription.Text := 'Not used';
      end{case;}
    end;
  end;
  with ProbeSetupWin do
  if Setup.TotalChannels <> 0 then {update sample frequency per channel and waveform duration labels}
  begin
    SampFreqPerChan.Value:= CalcActualFreqPerChan(SampFreqPerChan.Value);
    for i := 0 to Setup.NProbes-1 do
    begin
      if ProbeRow[i].SkipSpin.Value > 0 then ProbeRow[i].lblSampFreq.Caption:=
        inttostr(Round(SampFreqPerChan.Value/ProbeRow[i].SkipSpin.Value))
      else ProbeRow[i].lblSampFreq.Caption := inttostr(SampFreqPerChan.Value);
      {if ProbeRow[i].Probetype <> SPIKESTREAM then }ProbeRow[i].ActualTimeLabel.Caption:=
        FloatToStrF((ProbeRow[i].NptsSpin.Value * 1000 * ProbeRow[i].SkipSpin.Value) / SampFreqPerChan.Value, ffFixed, 4, 1);
      {else ProbeRow[i].Label1.Caption:= 'Continuous';}
    end;
  end;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.FormShow(Sender: TObject);
begin
  OK:= False;
  CreateProbeRows;
  NCRSpikeProbeSpin.Value:= Setup.NSpikeStreamProbes;
  NSpikeProbeSpin.Value:= Setup.NSpikeEpochProbes;
  NCRProbesSpin.Value:= Setup.NCRProbes;
  RefreshProbeButton;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.NCRSpikeProbeSpinChange(Sender: TObject);
begin
  if NCRSpikeProbeSpin.Value + NSpikeProbeSpin.Value + NCRProbesSpin.Value > SURF_MAX_PROBES then
    NCRSpikeProbeSpin.Value:= NCRSpikeProbeSpin.Value - 1;
  RefreshProbeButton;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.NCRProbesSpinChange(Sender: TObject);
begin
  if NCRSpikeProbeSpin.Value + NSpikeProbeSpin.Value + NCRProbesSpin.Value > SURF_MAX_PROBES then
    NCRProbesSpin.Value:= NCRProbesSpin.Value - 1;
  RefreshProbeButton;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.NSpikeProbeSpinChange(Sender: TObject);
begin
  if NCRSpikeProbeSpin.Value + NSpikeProbeSpin.Value + NCRProbesSpin.Value > SURF_MAX_PROBES then
    NSpikeProbeSpin.Value:= NSpikeProbeSpin.Value - 1;
  RefreshProbeButton;
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.SampFreqPerChanChange(Sender: TObject);
var i : integer;
begin
  if Setup.TotalChannels <> 0 then   //update proberow information to reflect
    for i := 0 to Setup.NProbes - 1 do //change in SampFreqPerChan
      ProbeRow[i].CheckProbeChannels(i);
end;

{-------------------------------------------------------------------------}
procedure TProbeSetupWin.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  FreeProbeRows;
  Action:= caHide;
end;

{-------------------------------------------------------------------------}
initialization
  Ready := False;

end.
