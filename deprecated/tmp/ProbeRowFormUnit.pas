unit ProbeRowFormUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Spin, ExtCtrls,SurfTypes,SurfPublicTypes;

type
  TProbeRowForm = class(TForm)
    ThresholdSpin: TSpinEdit;
    ADGainBox: TComboBox;
    TrigPtSpin: TSpinEdit;
    LockOutSpin: TSpinEdit;
    NptsSpin: TSpinEdit;
    ProbeDescription: TEdit;
    ActualTimeLabel: TLabel;
    ProbeNum: TLabel;
    SkipSpin: TSpinEdit;
    NumChanSpin: TSpinEdit;
    ChanStartSpin: TSpinEdit;
    ChanEndSpin: TSpinEdit;
    SampFreq: TLabel;
    Label2: TLabel;
    Label1: TLabel;
    View: TCheckBox;
    Save: TCheckBox;
    procedure FormCreate(Sender: TObject);
    //procedure SampFreqSpinChange(Sender: TObject);
    procedure SampFreqSpinExit(Sender: TObject);
    procedure ChanStartSpinChange(Sender: TObject);
    procedure NumChanSpinChange(Sender: TObject);
    procedure ChanEndSpinChange(Sender: TObject);
    procedure SkipSpinChange(Sender: TObject);
    procedure NptsSpinChange(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
    ChannelsCreated : boolean;
    probetype : char;//POLYTRODE  = 'N',CONTINUOUS = 'R';

    Procedure CheckProbeChannels(ProbeNum : integer); virtual; abstract;
  end;

implementation

{$R *.DFM}

procedure TProbeRowForm.FormCreate(Sender: TObject);
begin
  //showmessage('create probe row form');
  ChannelsCreated := FALSE;
  NumChanSpin.Value := 0;
end;
(*
procedure TProbeRowForm.SampFreqSpinChange(Sender: TObject);
//var sf : integer;
begin
  if not ChannelsCreated then exit;
  //sf := SampFreqSpin.value;
  //TotFreq := sf*NumChanSpin.Value;
end;  *)

procedure TProbeRowForm.SampFreqSpinExit(Sender: TObject);
begin
  if not ChannelsCreated then exit;
  (Sender as TSpinEdit).Update;
end;

procedure TProbeRowForm.ChanStartSpinChange(Sender: TObject);
begin
  if not ChannelsCreated then exit;
  if ChanStartSpin.Value + NumChanSpin.Value-1 >= SURF_MAX_CHANNELS then
    ChanStartSpin.Value := SURF_MAX_CHANNELS - NumChanSpin.Value;
  if (NumChanSpin.value = 0) or (probetype=CONTINUOUSTYPE )
    then ChanEndSpin.Value := ChanStartSpin.Value
    else ChanEndSpin.Value := ChanStartSpin.Value + NumChanSpin.Value-1;
  if (ChanStartSpin.Value >= 0) and (ChanStartSpin.Value < SURF_MAX_CHANNELS)
    then CheckProbeChannels(Tag);
end;

procedure TProbeRowForm.NumChanSpinChange(Sender: TObject);
begin
  if not ChannelsCreated then exit;
  if not ((ChanStartSpin.Value >= 0) and (NumChanSpin.Value <= SURF_MAX_CHANNELS)) then exit;
  if ChanStartSpin.Value + NumChanSpin.Value-1 >= SURF_MAX_CHANNELS then
    NumChanSpin.Value := SURF_MAX_CHANNELS - ChanStartSpin.Value;
  ChanEndSpin.Value := ChanStartSpin.Value + NumChanSpin.Value-1;
  if (NumChanSpin.Value >= 0) and (NumChanSpin.Value < 32)
   then CheckProbeChannels(Tag);
end;

procedure TProbeRowForm.ChanEndSpinChange(Sender: TObject);
begin
  if not ChannelsCreated then exit;
  If ChanEndSpin.Value < ChanStartSpin.Value then ChanEndSpin.Value := ChanStartSpin.Value;
  if (ChanEndSpin.Value = ChanStartSpin.Value)
    then begin if (probetype<>CONTINUOUSTYPE) then NumChanSpin.Value := 0; end
    else NumChanSpin.Value := ChanEndSpin.Value - ChanStartSpin.Value + 1;
  if (ChanEndSpin.Value >= 0) and (ChanEndSpin.Value < SURF_MAX_CHANNELS)
    then CheckProbeChannels(Tag);
end;

procedure TProbeRowForm.SkipSpinChange(Sender: TObject);
var i: integer;
begin
  try
    //if value is not an int this will raise and exception and bounce to 'except'.
    i := SkipSpin.Value;
    if i > SkipSpin.MinValue then CheckProbeChannels(Tag);
  except
  end;
end;

procedure TProbeRowForm.NptsSpinChange(Sender: TObject);
var i: integer;
begin
  try
    //if value is not an int this will raise and exception and bounce to 'except'.
    i := NPtsSpin.Value;
    if i > NPtsSpin.MinValue then CheckProbeChannels(Tag);
  except
  end;
end;

end.
