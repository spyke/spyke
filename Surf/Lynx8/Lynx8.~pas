unit Lynx8;

interface

uses Windows, Classes, Graphics, Forms, Controls, Menus,
  Dialogs, StdCtrls, Buttons, ExtCtrls, ComCtrls, Spin,
  Lynx8CControl, SysUtils, L8About, L8Setup, PahUnit, EnterGain, Messages;

const CHANSPERAMP = 8; {for Lynx-8 Amps}
      MAJVERSION = Word(1);
      MINVERSION = Word(0);
      DEFAULTFILE = 'Default.lcf';
      LASTFILE = 'Last.lcf';

      //WM_LYNX8SHOW  = WM_USER+1103; {1024 + 1103}
      //WM_LYNX8CLOSE = WM_USER+1104; {1024 + 1104}
type
  chanrec = record
    use : boolean;
    GainValue,LoIndex,HiIndex : Word;
  end;
  amprec = record
    chan : array[0..CHANSPERAMP] of chanrec;
  end;

  TLynx8Form = class(TForm)
    OpenDialog: TOpenDialog;
    SaveDialog: TSaveDialog;
    Panel1: TPanel;
    StaticText1: TStaticText;
    AmpID: TSpinEdit;
    StaticText3: TStaticText;
    AllChans: TGroupBox;
    Chan1: TGroupBox;
    Chan2: TGroupBox;
    Chan3: TGroupBox;
    Chan4: TGroupBox;
    Chan5: TGroupBox;
    Chan6: TGroupBox;
    Chan7: TGroupBox;
    Chan8: TGroupBox;
    StaticText2: TStaticText;
    StaticText4: TStaticText;
    OpenConfigBtn: TBitBtn;
    SaveConfigBtn: TBitBtn;
    BitBtn1: TBitBtn;
    StaticText5: TStaticText;
    GainAll: TEdit;
    Gain1: TEdit;
    Gain2: TEdit;
    Gain3: TEdit;
    Gain4: TEdit;
    Gain5: TEdit;
    Gain6: TEdit;
    Gain7: TEdit;
    Gain8: TEdit;
    GainUpDownAll: TUpDown;
    GainUpDown1: TUpDown;
    GainUpDown2: TUpDown;
    GainUpDown3: TUpDown;
    GainUpDown4: TUpDown;
    GainUpDown5: TUpDown;
    GainUpDown6: TUpDown;
    GainUpDown7: TUpDown;
    GainUpDown8: TUpDown;
    Use1: TCheckBox;
    Use2: TCheckBox;
    Use3: TCheckBox;
    Use4: TCheckBox;
    Use5: TCheckBox;
    Use6: TCheckBox;
    Use7: TCheckBox;
    Use8: TCheckBox;
    LoCutoffAll: TComboBox;
    LoCutoff1: TComboBox;
    LoCutoff2: TComboBox;
    LoCutoff3: TComboBox;
    LoCutoff4: TComboBox;
    LoCutoff5: TComboBox;
    LoCutoff6: TComboBox;
    LoCutoff7: TComboBox;
    LoCutoff8: TComboBox;
    HiCutoffAll: TComboBox;
    HiCutoff1: TComboBox;
    HiCutoff2: TComboBox;
    HiCutoff3: TComboBox;
    HiCutoff4: TComboBox;
    HiCutoff5: TComboBox;
    HiCutoff6: TComboBox;
    HiCutoff7: TComboBox;
    HiCutoff8: TComboBox;
    AllAmps: TCheckBox;
    SetupBtn: TBitBtn;
    Configuration: TRadioGroup;
    timer: TTimer;
    AudioCh: TSpinEdit;
    StaticText6: TStaticText;
    MonitorNull: TCheckBox;
    procedure ExitItemClick(Sender: TObject);
    procedure OpenItemClick(Sender: TObject);
    procedure SaveItemClick(Sender: TObject);
    procedure About1Click(Sender: TObject);
    procedure LoCutoffAllChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure HiCutoffAllChange(Sender: TObject);
    procedure AmpUpDownClick(Sender: TObject; Button: TUDBtnType);
    procedure UseClick(Sender: TObject);
    procedure AmpUpDownMouse(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure AmpIDChange(Sender: TObject);
    procedure AudioChChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure AllAmpsClick(Sender: TObject);
    procedure GainDblClick(Sender: TObject);
    procedure LoCutoffChange(Sender: TObject);
    procedure SetupBtnClick(Sender: TObject);
    procedure FormActivate(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure timerTimer(Sender: TObject);
    procedure MonitorNullClick(Sender: TObject);
  private
    { Private declarations }
    amp : array[1..MAXAMPS] of amprec;
    count : Word;
    alreadyactivated : boolean;
    autoupdate : boolean;

    function StartUpNTPorts (PortToAccess : word) : boolean;
    procedure SelectAudioChan(audchn : byte);
    procedure SelectAmp(ampnum : byte);
    procedure UpdateRealChans(ampno : byte);
    procedure UpdateAllChansForm(ampno : byte);
    procedure SetGainAllChan;
    procedure OpenFile(Filename : string);
    procedure SaveFile(Filename : string);
    procedure GetSettings;
    procedure SaveSettings;
  public
    { Public declarations }
    //procedure ShowWin ( var msg : TMessage ); message WM_LYNX8SHOW;
    //procedure CloseWin ( var msg : TMessage ); message WM_LYNX8CLOSE;
  end;

var
  Lynx8Form: TLynx8Form;

implementation

{$R *.DFM}

{==============================================================================}
procedure TLynx8Form.FormCreate(Sender: TObject);
var i,j : integer;
  VersionInfo: TOSVersionInfo;
begin
//--------------------------------------------------------------------
//   this code added by TJB on 30.5.01 to enable Lynx8 to run under
//   WIN2000/NT (while maintaining WIN95/98 compatability).
//   It enables direct asm port writes via a kernel-based VxD.
//--------------------------------------------------------------------
  VersionInfo.dwOSVersionInfoSize := Sizeof(TOSVersionInfo);
  GetVersionEx(VersionInfo);
  case VersionInfo.dwPlatformID of
    VER_PLATFORM_WIN32_WINDOWS:
      Lynx8Form.Caption := Lynx8Form.Caption + ' (Win95/98)';
    VER_PLATFORM_WIN32_NT:
      begin
        if not StartUpNTPorts($378) then halt //test access to lpt1
        else Lynx8Form.Caption := Lynx8Form.Caption + ' (Win2000/NT)';
      end;
    end;

//-----------------------------------------------
  autoupdate := true;
  SetupL8 := TSetupL8.Create(self);

  SetupL8.Namps.Value := MAXAMPS;
  SetupL8.InputGain.ItemIndex := 2{100};
  SetupL8.OutPutPort.ItemIndex := 0{LPT 1:};

  For i := 1 to SetupL8.Namps.Value do
    With amp[i] do
    begin
      For j := 0 to CHANSPERAMP do
        With chan[j] do
        begin
          use := TRUE;
          GainValue := 1;
          LoIndex := 5{600};
          HiIndex := 8{3000};
        end;
    end;

  SetGainAllChan;
  SetupL8.Namps.Value := 1;
  AmpID.value := 1;

  GetSettings;
  alreadyactivated := FALSE;
end;

{==============================================================================}
function TLynx8Form.StartUpNTPorts(PortToAccess : word) : boolean;
Var hUserPort : THandle;
Begin
  hUserPort := CreateFile('\\.\UserPort', GENERIC_READ, 0, nil,OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  CloseHandle(hUserPort); // Activate the driver
  Sleep(100); // Wait for process switch

  try
    Lynx8Amp.InPort(PortToAccess);  // Try to access port
    StartUpNTPorts := true;
  except
    MessageBox(0,'Unable to write to port','Is VxD running?',MB_OK);
    StartUpNTPorts := false;
  end;
end;

{==============================================================================}
procedure TLynx8Form.FormActivate(Sender: TObject);
//var b : integer;
begin
(*  if alreadyactivated then exit;
  alreadyactivated := TRUE;
  //Delay(0,100);
  //Lynx8Amp.SurfHandle := FindWindow('TSurfAcqForm','SURF');
  showmessage('about to init amp');
  Lynx8Amp.InitAmp;
  Lynx8Amp.SetEqualizeSwitches(0); {initilize amps}

  autoupdate := false;
  Case Configuration.ItemIndex of
    0 : OpenFile(LASTFILE);
    1 : OpenFile(DEFAULTFILE);
  end;
  UpdateAllChansForm(1);

  Count := 0;
  autoupdate := true;

  AmpID.MaxValue := SetupL8.Namps.Value;

  For b := 1 to SetupL8.Namps.Value do
    UpdateRealChans(b);*)
end;

{==============================================================================}
procedure TLynx8Form.SelectAmp(ampnum : byte);
var portadd : word;
begin
  portadd := PORTADDRESS[SetupL8.OutPutPort.ItemIndex];
  case ampnum of
    0 : Lynx8Amp.OutPort(portadd,$FF{11111111});//None
    1 : Lynx8Amp.OutPort(portadd,$FE{11111110});
    2 : Lynx8Amp.OutPort(portadd,$FD{11111101});
    3 : Lynx8Amp.OutPort(portadd,$FB{11111011});
    4 : Lynx8Amp.OutPort(portadd,$F7{11110111});
    5 : Lynx8Amp.OutPort(portadd,$EF{11101111});
    6 : Lynx8Amp.OutPort(portadd,$DF{11011111});
    7 : Lynx8Amp.OutPort(portadd,$BF{10111111});
    8 : Lynx8Amp.OutPort(portadd,$7F{01111111});
  end;
end;
{==============================================================================}
procedure TLynx8Form.UpdateRealChans(ampno : byte);
var ch : byte;
    nlfilter_setting,nhfilter_setting : Word;
begin
  {Select the amp number}
  Lynx8Amp.SetPortsForOutput;
  SelectAmp(ampno);
  nlfilter_setting := LOWCUT_900HZ;
  nhfilter_setting := HICUT_50HZ;
  For ch := 0 to CHANSPERAMP-1 do
  begin
    {Update low filters}
    case amp[ampno].chan[ch+1].LoIndex of
      0 : nlfilter_setting := LOWCUT_TENTHHZ;
      1 : nlfilter_setting := LOWCUT_1HZ;
      2 : nlfilter_setting := LOWCUT_10HZ;
      3 : nlfilter_setting := LOWCUT_100HZ;
      4 : nlfilter_setting := LOWCUT_300HZ;
      5 : nlfilter_setting := LOWCUT_600HZ;
      6 : nlfilter_setting := LOWCUT_900HZ;
    end;
    {Update hi filters}
    case amp[ampno].chan[ch+1].HiIndex of
      0 : nhfilter_setting := HICUT_50HZ;
      1 : nhfilter_setting := HICUT_125HZ;
      2 : nhfilter_setting := HICUT_200HZ;
      3 : nhfilter_setting := HICUT_250HZ;
      4 : nhfilter_setting := HICUT_275HZ;
      5 : nhfilter_setting := HICUT_325HZ;
      6 : nhfilter_setting := HICUT_400HZ;
      7 : nhfilter_setting := HICUT_475HZ;
      8 : nhfilter_setting := HICUT_3KHZ;
      9 : nhfilter_setting := HICUT_6KHZ;
      10: nhfilter_setting := HICUT_9KHZ;
    end;
    Lynx8Amp.LoadSingleFilterVal(ch, nlfilter_setting OR nhfilter_setting);
    {Update Gain}
    Lynx8Amp.LoadSingleGainVal(ch, amp[ampno].chan[ch+1].GainValue);
  end;
  SelectAmp(0);
  Lynx8Amp.SetPortsForInput;
end;

{==============================================================================}
procedure TLynx8Form.UpdateAllChansForm(ampno : byte);
var scale : single;
    i,loi,hii,ingain : integer;
    s : string;
    tmpuse : TCheckBoxState;
begin
  ingain := StrToInt(SetupL8.InputGain.text);
  scale := (REF_FULL_GAIN/100*InGain) / DTDIO_DAC_MAX_VALUE;

  For i := 0 to CHANSPERAMP do
  begin
    loi    := amp[ampno].chan[i].LoIndex;
    hii    := amp[ampno].chan[i].HiIndex;
    if amp[ampno].chan[i].use then tmpuse := cbChecked else tmpuse := cbUnChecked;
    s := FloatToStrF(amp[ampno].chan[i].GainValue * scale,ffFixed,9,2);
    case i of
      0 : begin GainAll.Text:=s; LoCutoffAll.ItemIndex:=loi; HiCutoffAll.ItemIndex:=hii; end;
      1 : begin Gain1.Text := s; LoCutoff1.ItemIndex := loi; HiCutoff1.ItemIndex := hii; use1.state := tmpuse; end;
      2 : begin Gain2.Text := s; LoCutoff2.ItemIndex := loi; HiCutoff2.ItemIndex := hii; use2.state := tmpuse; end;
      3 : begin Gain3.Text := s; LoCutoff3.ItemIndex := loi; HiCutoff3.ItemIndex := hii; use3.state := tmpuse; end;
      4 : begin Gain4.Text := s; LoCutoff4.ItemIndex := loi; HiCutoff4.ItemIndex := hii; use4.state := tmpuse; end;
      5 : begin Gain5.Text := s; LoCutoff5.ItemIndex := loi; HiCutoff5.ItemIndex := hii; use5.state := tmpuse; end;
      6 : begin Gain6.Text := s; LoCutoff6.ItemIndex := loi; HiCutoff6.ItemIndex := hii; use6.state := tmpuse; end;
      7 : begin Gain7.Text := s; LoCutoff7.ItemIndex := loi; HiCutoff7.ItemIndex := hii; use7.state := tmpuse; end;
      8 : begin Gain8.Text := s; LoCutoff8.ItemIndex := loi; HiCutoff8.ItemIndex := hii; use8.state := tmpuse; end;
    end;
  end;
end;

{==============================================================================}
procedure TLynx8Form.SetGainAllChan;
var ch,ampno : byte;
begin
  For ampno := 1 to SetupL8.Namps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
      For ch := 1 to CHANSPERAMP do
        if amp[ampno].chan[ch].use
        then amp[ampno].chan[ch].GainValue :=  amp[ampno].chan[0].GainValue;
end;

{==============================================================================}
procedure TLynx8Form.ExitItemClick(Sender: TObject);
begin
  Close;
end;

{==============================================================================}
procedure TLynx8Form.GetSettings;
var sF : File;
  wb : WordBool;
begin
  AssignFile(sF,'Lynx8.ini');
  {$I-}Reset(sF,1);{$I+}
  If IOResult<>0 then exit;
  //begin
    //MessageDlg('Error opening Lynx8.ini', mtWarning, [mbOk], 0);
    //Exit;
  //end;
  Configuration.ItemIndex := 0;
  BlockRead(sF,wb,2);  if wb then Configuration.ItemIndex := 0;
  BlockRead(sF,wb,2);  if wb then Configuration.ItemIndex := 1;

  //ShowMessage(lastfilename);

  {if UseLastSaved.Checked
    then ShowMessage('use last') else ShowMessage('not use last');
  if UseDefault.Checked
    then ShowMessage('use default') else ShowMessage('not use default');
  }
  CloseFile(sF);
end;
{==============================================================================}
procedure TLynx8Form.SaveSettings;
var sF : File;
  wb : WordBool;
begin
  AssignFile(sF,'Lynx8.ini');
  {$I-}ReWrite(sF,1);{$I+}
  If IOResult<>0 then exit;
  //begin
    //MessageDlg('Error saving Lynx8.ini', mtWarning, [mbOk], 0);
    //Exit;
  //end;
  if Configuration.ItemIndex = 0 then wb := TRUE else wb := FALSE;
  BlockWrite(sF,wb,2);
  if Configuration.ItemIndex = 1 then wb := TRUE else wb := FALSE;
  BlockWrite(sF,wb,2);
  CloseFile(sF);
end;

{==============================================================================}
procedure TLynx8Form.SaveFile(Filename : string);
var lcF : File;
  na,maj,min,ampno : word;
  wb : WordBool;
begin
  AssignFile(lcF,Filename);
  {$I-}Rewrite(lcF,1);{$I+}
  If IOResult<>0 then exit;
  //begin
    //MessageDlg('Error saving '+ Filename, mtWarning, [mbOk], 0);
    //Exit;
  //end;
  maj := MAJVERSION;
  min := MINVERSION;
  BlockWrite(lcF,maj,2);{write major version}
  BlockWrite(lcF,min,2);{write minor version}

  wb := AllAmps.Checked;      BlockWrite(lcF,wb,2);
  na := SetupL8.Namps.Value;    BlockWrite(lcF,na,2);

  For ampno := 1 to na do
    BlockWrite(lcF,amp[ampno],sizeof(amprec));
  CloseFile(lcF);
end;

{==============================================================================}
procedure TLynx8Form.SaveItemClick(Sender: TObject);
begin
  If SaveDialog.Execute then
    SaveFile(SaveDialog.Filename);
end;

{==============================================================================}
procedure TLynx8Form.OpenFile(Filename : string);
var lcF : File;
  na,maj,min,ampno : word;
  wb : WordBool;
begin
  //ShowMessage('opening >'+Filename+'<');
  //exit;
  AssignFile(lcF,Filename);
  {$I-}Reset(lcF,1);{$I+}
  If IOResult<>0 then exit;
  //begin
    //MessageDlg('Error opening '+ Filename, mtWarning, [mbOk], 0);
    //Exit;
  //end;
  BlockRead(lcF,maj,2);{read major version}
  BlockRead(lcF,min,2);{read minor version}

  BlockRead(lcF,wb,2);  AllAmps.Checked := wb;
  BlockRead(lcF,na,2);  SetupL8.Namps.Value := na;

  For ampno := 1 to SetupL8.Namps.Value do
    BlockRead(lcF,amp[ampno],sizeof(amprec));
  CloseFile(lcF);

  UpdateAllChansForm(AmpID.value);
  For ampno := 1 to SetupL8.NAmps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
      UpdateRealChans(ampno);

end;

{==============================================================================}
procedure TLynx8Form.OpenItemClick(Sender: TObject);
begin
  If OpenDialog.Execute then
    OpenFile(OpenDialog.Filename);
end;

{==============================================================================}
procedure TLynx8Form.About1Click(Sender: TObject);
begin
  L8AboutBox.ShowModal;
end;

{==============================================================================}
procedure TLynx8Form.LoCutoffAllChange(Sender: TObject);
var ch,ampno : byte;
begin
  For ampno := 1 to SetupL8.Namps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
    begin
      For ch := 0 to CHANSPERAMP do
        if amp[ampno].chan[ch].use then
          amp[ampno].chan[ch].LoIndex :=  LoCutoffAll.ItemIndex;
      UpdateRealChans(ampno);
    end;
  UpdateAllChansForm(AmpID.value);
end;

{==============================================================================}
procedure TLynx8Form.HiCutoffAllChange(Sender: TObject);
var ch,ampno : smallint;
begin
  For ampno := 1 to SetupL8.Namps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
    begin
      For ch := 0 to CHANSPERAMP do
        if amp[ampno].chan[ch].use then
          amp[ampno].chan[ch].HiIndex :=  HiCutoffAll.ItemIndex;
      UpdateRealChans(ampno);
    end;
  UpdateAllChansForm(AmpID.value);
end;

{==============================================================================}
procedure TLynx8Form.UseClick(Sender: TObject);
var ch,ampno : smallint;
begin
  if not autoupdate then exit;
  ch := (Sender as TCheckBox).Tag;
  case ch of
    1 : Chan1.Enabled := not Chan1.Enabled;
    2 : Chan2.Enabled := not Chan2.Enabled;
    3 : Chan3.Enabled := not Chan3.Enabled;
    4 : Chan4.Enabled := not Chan4.Enabled;
    5 : Chan5.Enabled := not Chan5.Enabled;
    6 : Chan6.Enabled := not Chan6.Enabled;
    7 : Chan7.Enabled := not Chan7.Enabled;
    8 : Chan8.Enabled := not Chan8.Enabled;
  end;
  For ampno := 1 to SetupL8.Namps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
    With amp[ampno].chan[ch] do
    begin
      use := not amp[AmpID.value].chan[ch].use;
      if use then
      begin
        LoIndex :=  amp[ampno].chan[0].LoIndex;
        HiIndex :=  amp[ampno].chan[0].HiIndex;
        GainValue := amp[ampno].chan[0].GainValue;
      end else
      begin
        LoIndex :=  6;
        HiIndex :=  0;
        GainValue := 0;
      end;
      UpdateRealChans(ampno);
    end;
  UpdateAllChansForm(AmpID.value);
end;

{==============================================================================}
procedure TLynx8Form.AmpUpDownClick(Sender: TObject; Button: TUDBtnType);
var amount,diff,pos,ch,gain,ampno : integer;
begin
  ch := (Sender as TUpDown).Tag;
  pos := (Sender as TUpDown).Position;

  diff := pos - count;
  if diff > 0
    then amount := round(exp((pos - count)/4))
    else amount := round(exp((count - pos)/4));
  if amount < 1 then amount := 1;

  For ampno := 1 to SetupL8.Namps.Value do
  begin
    if (AllAmps.Checked or (ampno = AmpID.value)) then
    begin
      With amp[ampno].chan[ch] do
      if use then
      begin
        Gain := GainValue;
        if diff > 0
          then Gain := Gain + amount
          else Gain := Gain - amount;
        if Gain > 4095 then Gain := 4095;
        if Gain < 0 then Gain := 0;
        GainValue := Gain;
      end;
      UpdateRealChans(ampno);
    end;
  end;
  if ch = 0 then SetGainAllChan;
  UpdateAllChansForm(AmpID.value);
end;

{==============================================================================}
procedure TLynx8Form.AmpUpDownMouse(Sender: TObject;
  Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  (Sender as TUpDown).Position := 0;
  count := 0;
end;

{==============================================================================}
procedure TLynx8Form.AmpIDChange(Sender: TObject);
begin
  if AMPID.Value > SetupL8.NAmps.Value then
    AMPID.Value := SetupL8.NAmps.Value;
  if AMPID.Value = 0 then AMPID.Value := 1;
  autoupdate := false;
  UpdateAllChansForm(AmpID.value);
  autoupdate := true;
end;

{==============================================================================}
procedure TLynx8Form.AudioChChange(Sender: TObject);
begin
  SelectAudioChan(AudioCh.Value);
end;

{==============================================================================}
procedure TLynx8Form.SelectAudioChan(audchn : byte);
var i: byte;
begin
  i := 0;
  Lynx8Amp.OutPortV(PORTADDRESS[SetupL8.OutPutPort.ItemIndex]+2,$0C{00001100},$08);
  Lynx8Amp.OutPortV(PORTADDRESS[SetupL8.OutPutPort.ItemIndex]+2,$08{00001000},$0C);
  while i < audchn do
  begin
    if Lynx8Amp.OutPortV(PORTADDRESS[SetupL8.OutPutPort.ItemIndex]+2,$00{00000000},$08) <> true then dec(i);
    if Lynx8Amp.OutPortV(PORTADDRESS[SetupL8.OutPutPort.ItemIndex]+2,$08{00001000},$00) <> true then dec(i);
    inc (i);
  end;
end;

{==============================================================================}
procedure TLynx8Form.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  SaveSettings;
  SaveFile(LASTFILE);
  Delay(0,100);
  SetupL8.Free;
  //Lynx8Amp.Free;
end;

{==============================================================================}
procedure TLynx8Form.AllAmpsClick(Sender: TObject);
begin
  AmpID.enabled := not AllAmps.Checked;
end;

{==============================================================================}
procedure TLynx8Form.GainDblClick(Sender: TObject);
var scale : single;
    ch,ingain,ampno : integer;
begin
  ch := (Sender as TEdit).Tag;
  EnterGainForm.GainEdit.Text := (Sender as TEdit).Text;//GainAll.Text;
  ingain := StrToInt(SetupL8.InputGain.text);
  scale := (REF_FULL_GAIN/100*InGain) / DTDIO_DAC_MAX_VALUE;
  EnterGainForm.Min := 0;
  EnterGainForm.Max := 4095 * scale;
  EnterGainForm.ShowModal;

  amp[AmpID.Value].chan[ch].GainValue := round(StrToFloat(EnterGainForm.GainEdit.Text)/scale);
  if ch = 0 then SetGainAllChan;{copy this gain to all other chans}
  UpdateAllChansForm(AmpID.value);{update the form}
  For ampno := 1 to SetupL8.NAmps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
      UpdateRealChans(ampno);{update the real amps}
end;

{==============================================================================}
procedure TLynx8Form.LoCutoffChange(Sender: TObject);
var ch,ampno : integer;
begin
  ch := (Sender as TComboBox).Tag;
  For ampno := 1 to SetupL8.Namps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
    begin
      amp[ampno].chan[ch].LoIndex :=  LoCutoffAll.ItemIndex;
      UpdateRealChans(ampno);
    end;
end;

{==============================================================================}
procedure TLynx8Form.SetupBtnClick(Sender: TObject);
var ampno : byte;
begin
  SetupL8.ShowModal;
  {Setup has changed, so update current window}
  if AMPID.Value > SetupL8.NAmps.Value then
    AMPID.Value := SetupL8.NAmps.Value;
  UpdateAllChansForm(AmpID.value);
  For ampno := 1 to SetupL8.NAmps.Value do
    if (AllAmps.Checked or (ampno = AmpID.value)) then
      UpdateRealChans(ampno);
end;

{==============================================================================}
procedure TLynx8Form.FormShow(Sender: TObject);
begin
  //need to call init amps, but only after the form on which it resides is created
  timer.enabled := TRUE;
end;

{==============================================================================}
procedure TLynx8Form.timerTimer(Sender: TObject);
var b : integer;
begin
  timer.enabled := FALSE;
  //Lynx8Amp.SurfHandle := FindWindow('TSurfAcqForm','SURF');
  Lynx8Amp.InitAmp;
  if Lynx8Amp.DIO.numboards = 0 then Close;

  Lynx8Amp.SetEqualizeSwitches(0); {initilize amps}

  autoupdate := false;
  Case Configuration.ItemIndex of
    0 : OpenFile(LASTFILE);
    1 : OpenFile(DEFAULTFILE);
  end;
  UpdateAllChansForm(1);

  Count := 0;
  autoupdate := true;

  AmpID.MaxValue := SetupL8.Namps.Value;

  For b := 1 to SetupL8.Namps.Value do
    UpdateRealChans(b);
end;

{==============================================================================}
procedure TLynx8Form.MonitorNullClick(Sender: TObject);
begin
  AudioCh.Enabled := not(AudioCh.Enabled);
  If MonitorNull.Checked then
    Lynx8Amp.OutPort(PORTADDRESS[SetupL8.OutPutPort.ItemIndex]+2,$0C{00001100})
  else SelectAudioChan(AudioCh.Value);
end;

end.

