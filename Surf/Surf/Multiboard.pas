unit Multiboard;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  OleCtrls, DTAcq32Lib_TLB, DTxPascal, StdCtrls, ExtCtrls;

type
  TDTConfig = class(TForm)
    GroupBox1: TGroupBox;
    ListBox1: TListBox;
    RadioGroup1: TRadioGroup;
    RadioGroup2: TRadioGroup;
    GroupBox2: TGroupBox;
    ListBox2: TListBox;
    GroupBox3: TGroupBox;
    CheckBox1: TCheckBox;
    Label1: TLabel;
    Edit1: TEdit;
    RadioGroup4: TRadioGroup;
    Button1: TButton;
    Button2: TButton;
    GroupBox4: TGroupBox;
    Label2: TLabel;
    Edit2: TEdit;
    RadioGroup3: TRadioGroup;
    procedure FormCreate(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure ListBox1Click(Sender: TObject);
    procedure RadioGroup1Click(Sender: TObject);
    procedure RadioGroup2Click(Sender: TObject);
    procedure RadioGroup3Click(Sender: TObject);
    procedure Edit2Change(Sender: TObject);
    procedure RadioGroup4Click(Sender: TObject);
    procedure CheckBox1Click(Sender: TObject);
    procedure Edit1Change(Sender: TObject);
    procedure Button2Click(Sender: TObject);
  private
    DTAcq : array of TDTAcq32;
    procedure UpdateOptions;
    { Private declarations }
  public
    { Public declarations }
  end;

var
  DTConfig: TDTConfig;
implementation

{$R *.DFM}

procedure TDTConfig.FormCreate(Sender: TObject);
var i, MaxFreq : integer;
  NumBoards, NumADChans, NumADGains : byte;
  hbuf : hbuftype;
begin
  //create DTAcq32 object to access Numboards property
  Setlength(DTAcq, 1);
  try
    DTAcq[0]:= TDTAcq32.Create(Self);
  except
    ListBox1.Items.Add('Unable to initialise DTx driver');
    Exit;
  end;
  Numboards:= DTAcq[0].Numboards;
  DTAcq[0].Free; //will be recreated in loop
  Setlength(DTAcq, 2{Numboards}); //should not be hard coded

  //create DTAcq32 objects and add boards to listbox
  if Numboards > 0 then
  begin
    for i:= 0 to 2{Numboards} - 1 do
    begin
      DTAcq[i]:= TDTAcq32.Create(Self);
      DTAcq[i].Board:= DTAcq[i].BoardList[i];
      if DTAcq[i].GetDevCaps(OLSS_AD) <> 0 then
      begin
        DTAcq[i].SubSystem := OLSS_AD;
        if DTAcq[i].GetSSCaps(OLSSC_SUP_CONTINUOUS) <> 0 //check if board can handle continuous operation
          then DTAcq[i].DataFlow := OL_DF_CONTINUOUS //set up for continuous about trig operation
          else Exit;
        ListBox1.Items.Add(DTAcq[i].BoardList[i]);
      end else DTAcq[i].Free; //only create objects for A/D subsystems
    end;
  end else
  begin
    ListBox1.Items.Add('No DT boards available');
    Setlength(DTAcq, 0);
    Exit;
  end;
//  ListBox1.ItemIndex:= 0; //select first DT board as default

  for i:= 0 to 1 do
  begin
    if ErrorMsg(olDmAllocBuffer(0, 1000000, @hbuf)) then
    begin
      Showmessage('Unable to allocate ADC buffers');
      Exit;
    end;
    DTAcq[i].Queue := {U}LNG(hbuf);
    DTAcq[i].WrapMode := OL_WRP_MULTIPLE;
  end;
end;

procedure TDTConfig.UpdateOptions;
begin
end;

procedure TDTConfig.Button1Click(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].Config;
  DTAcq[ListBox1.ItemIndex].Start;
end;

procedure TDTConfig.ListBox1Click(Sender: TObject);
begin
  UpdateOptions; //options should be dimmed/undimmed according to specific
                 //board subsystems capabilities... removes need to check
                 //when actually setting subsystem properties.
end;

procedure TDTConfig.RadioGroup1Click(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].ChannelType := RadioGroup1.ItemIndex;
end;

procedure TDTConfig.RadioGroup2Click(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].Encoding := RadioGroup2.ItemIndex;
end;

procedure TDTConfig.RadioGroup3Click(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].ClockSource := RadioGroup3.ItemIndex;
end;

procedure TDTConfig.Edit2Change(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].Frequency:= StrToInt(Edit2.Text);
  if StrToInt(Edit2.Text) > DTAcq[ListBox1.ItemIndex].Frequency
    then Edit2.Text:= IntTostr(Round(DTAcq[ListBox1.ItemIndex].Frequency))
end;

procedure TDTConfig.RadioGroup4Click(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].ReTriggerMode:= RadioGroup4.ItemIndex;
end;

procedure TDTConfig.CheckBox1Click(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].TriggeredScan:= Ord(CheckBox1.Checked);
  Label1.Enabled:= CheckBox1.Checked;
  Edit1.Enabled:= CheckBox1.Checked;
  DTAcq[ListBox1.ItemIndex].MultiScanCount:= 0;
end;

procedure TDTConfig.Edit1Change(Sender: TObject);
begin
  DTAcq[ListBox1.ItemIndex].RetriggerFreq:= StrToInt(Edit1.Text);
end;

procedure TDTConfig.Button2Click(Sender: TObject);
begin
  //Buffers/arrays/DT objects should be freed here!
  Halt;
end;

end.
