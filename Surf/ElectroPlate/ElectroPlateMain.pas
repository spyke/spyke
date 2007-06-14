unit ElectroPlateMain;
{ (c) 2001-2003 Tim Blanche, University of British Columbia }

{ Known bugs and improvements yet to be implemented:
 i) phase measurements flip by 180deg in an unpredictable fashion, producing incorrect impedance measurements.
    The current 'work-around' simply re-samples the current channel if the calculated phase is negative.
 ii) if the user clicks on the PolytrodeGUI, the auto-mode operation will be interrupted.
 iii) code surrounding the Auto impedance mode, particularly that dealing with the SITE2MUX mapping,
      and indexing through TSiteProperties, etc. could be cleaned up and optimised considerably.
 iv) it is not currently possible to change the board used for acquisition, nor use a different board for
     DOUT (for controlling the meter/plater) and the two analogue input channels for impedance measurements.
  v) Electoplate mode is not fully implemented or tested.
 vi) Impedance spectroscopy mode is not fully implemented or tested.
 vii) Control of imp.meter with parallel port currently disabled.
 viii) Users are not able to select probes other than 54EIB probes, since only the EIB54 mapping currently coded.
       Very little additional code (and hardware) would be required to map to older 16 channel probes.
 ix) Changing selected acq. board from drop down list not fully tested/buggy }
 
interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls, Spin, DTAcq32Lib_TLB, DTxPascal, Math, SurfMathLibrary,
  PolytrodeGUI, OleCtrls, ComCtrls;

type
  TSiteProperties = record
    Impedance : single;   //kOhms
    Phase : single;       //degrees
    TimePlated : integer; //seconds
    Faulty : boolean;     //true/false
  end;

  TEPlateMainForm = class(TForm)
    Timer: TTimer;
    StatusBar: TStatusBar;
    Manual: TGroupBox;
    Label2: TLabel;
    GroupBox1: TGroupBox;
    PPlate: TPanel;
    CElectrode: TComboBox;
    Label1: TLabel;
    ModeSelect: TRadioGroup;
    ProgressBar: TProgressBar;
    ChanSelect: TSpinEdit;
    DTDIO: TDTAcq32;
    DTAcq: TDTAcq32;
    Setup: TGroupBox;
    CInputGain: TComboBox;
    CADChan: TComboBox;
    Label3: TLabel;
    RadioGroup1: TRadioGroup;
    Label5: TLabel;
    ManualAcq: TButton;
    Label10: TLabel;
    TargetImp: TSpinEdit;
    PTestImp: TPanel;
    Label6: TLabel;
    DumpCSV: TCheckBox;
    Button1: TButton;
    Label9: TLabel;
    CBoard: TComboBox;
    DTDAC: TDTAcq32;
    Button2: TButton;
    procedure TimerTick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure CElectrodeChange(Sender: TObject);
    procedure ChanSelectChange(Sender: TObject);
    procedure ModeSelectClick(Sender: TObject);
    procedure ManualAcqClick(Sender: TObject);
    procedure CAD_CGChange(Sender: TObject);
    procedure PTestImpClick(Sender: TObject);
    procedure PlateClick(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure DTAcqQueueDone(Sender: TObject);
    procedure CBoardChange(Sender: TObject);
  private
    { Private declarations }
    GUIForm: TPolytrodeGUIForm;
    DOUT : byte;
    hbuf : HBUFTYPE;
    function MUXOut(a : byte) : byte;
    function CalcZ(const VRMSin, VRMSout, Phase : single;
                         Calibrated : boolean = False) : TComplex;
    procedure PauseAuto;
    procedure SaveParametersToFile;
    {procedure Out32(port : word; bval : byte);
    function Inp32 (port : word): smallint;}
  public
    { Public declarations }
  end;

var
  EPlateMainForm: TEPlateMainForm;
  Plating, TestingImp, GUICreated : boolean;
  BasePlatingTime : integer = 500; //ie. 1/2 second plating increments
  TimeOnCurrentSite: byte;
  ActiveChanNumber : byte;
  SkipChannel: byte;
  BufferReady : boolean;
  SiteProperties : array of TSiteProperties;
  GUISiteCol : TColor;

const
  PORTADDRESS : array[0..1] of Word = ($378,$37A); //LPT1 + control port
  SITE2MUX : array[1..54] of Byte = (9,10,11,12,13,1,2,3,4,5,6,7,15,{EIBfront}
                                    25,26,27,16,17,18,19,20,21,22,23,8,14,24,
                           {EIBback}51,50,49,48,47,46,45,44,55,54,53,43,35,34,
                                    33,32,31,30,29,41,40,39,38,37,52{!},36,56);
  MUX2EIB : array[1..56] of Byte = (6,7,8,9,10,11,12,25,1,2,3,4,5,26,{EIBfront}
                                   13,17,18,19,20,21,22,23,24,27,14,15,16,25,
                          {EIBback}46,45,44,43,42,41,40,53,51,50,49,48,47,52,
                                   39,35,34,33,32,31,30,29,28,52{!},38,37,36,54);

  ADCRate = 1000000; //ADC clock (burst mode)
  SampFreqPerChan = 50000;// retrigger frequency
  FFTSize = 65536; //#points in FFT, must be (2^n and <= BuffSize)
  BuffSize = FFTSize {+ 1000}; //ADC subsystem buffer size/channel
  TimerInterval = 3000; //2.0s, interval between MUX channel changes in auto mode

  ZREF = 0.9963; //reference impedance (MOhm)
  VOLTDIVIDER = 99.52; //input voltage divider factor
  OPAMPGAIN = 100.85{bread board};{99.73 - 1012c opamp pair before they were fried}//output is ~100x impedance input signal
  CIRCUITPHI = 0.18675;//0.18151;//-0.09076; //correct for circuit phase shift (radians), old value: -0.0785
  PIBY2 = 2 * pi;

implementation

uses {PolytrodeGUI,} ElectrodeTypes, SurfPublicTypes;

{$R *.DFM}

procedure TEPlateMainForm.FormCreate(Sender: TObject);
var e, MaxFreq : integer;
  NumADChans, NumADGains : byte;
  ArrayPtr : pointer;
  Wave : array of SmallInt;
begin
  //generate DTx board list
  if DTAcq.Numboards > 0 then //add DT3010 boards to drop-down list
  begin
     for e:= 0 to DTAcq.Numboards -1 do
      if Pos('DT3010', DTAcq.BoardList[e]) <> 0 then CBoard.Items.Add(DTAcq.BoardList[e])
  end else
  begin
    ShowMessage('No boards found');
    Exit;
  end;

  CBoard.ItemIndex:= 1;
  DTAcq.Board:= DTAcq.BoardList[CBoard.ItemIndex]; // select 1st board as default

  //initialise DTACQ
  DTAcq.Subsystem := OLSS_AD;
  if DTAcq.GetSSCaps(OLSSC_SUP_CONTINUOUS) <> 0 //check if board can handle continuous operation
    then DTAcq.DataFlow := OL_DF_CONTINUOUS //set up for continuous about trig operation
  else begin
    ShowMessage(DTAcq.Board+ ' does not support continuous mode');
    Exit;
  end;
  if DTAcq.GetSSCaps(OLSSC_SUP_SINGLEENDED) <> 0 then
    DTAcq.ChannelType := OL_CHNT_SINGLEENDED //set up for single ended acquisition
  else begin
    ShowMessage(DTAcq.Board+ ' does not support single ended acquisition');
    Exit;
  end;
  if DTAcq.GetSSCaps(OLSSC_SUP_BINARY) <> 0 then
    DTAcq.Encoding := OL_ENC_BINARY //set up for binary encoding
  else begin
    ShowMessage(DTAcq.Board+ ' does not support binary encoding');
    Exit;
  end;
  if DTAcq.GetSSCaps(OLSSC_SUP_INTCLOCK) <> 0 then
    DTAcq.ClockSource := OL_CLK_INTERNAL //set A/D clock to internal
  else begin
    ShowMessage(DTAcq.Board+ ' does not support internal clock');
    Exit;
  end;
  DTAcq.WrapMode := OL_WRP_NONE; //no buffering - fill buffer(s) then stop
  DTAcq.Frequency := 10000000; //returns ADC clock maximum frequency
  MaxFreq := round(DTAcq.Frequency);
  if MaxFreq > ADCRate then
    DtAcq.Frequency := ADCRate //set ADC clock to specified constant
  else begin
    ShowMessage(DTAcq.Board+ ' does not support fast enough sample rate');
    Setup.Enabled:= False;
    Exit;
  end;
  if DTAcq.GetSSCaps(OLSS_SUP_RETRIGGER_INTERNAL) <> 0 then
  begin
    DTAcq.ReTriggerMode:= OL_RETRIGGER_SCAN_PER_TRIGGER;
    DTAcq.TriggeredScan:= 1;
    DTAcq.MultiScanCount:= 1;
    DTAcq.RetriggerFreq:= SampFreqPerChan;
  end else
  begin
    ShowMessage(DTAcq.Board+ ' does not support internal retrigger mode');
    Exit;
  end;
  Label5.Caption:= FloattoStr((DTAcq.RetriggerFreq / 1000)) + 'kHz ADC';

  NumADChans := DTAcq.GetSSCaps(OLSSC_MAXSECHANS);
  NumADGains := DTAcq.GetSSCaps(OLSSC_NUMGAINS);
  for e := 0 to NumADChans - 2 do
    CADChan.Items.Add(inttostr(e) + ',' + inttostr(e+1));
  CADChan.ItemIndex:= 22; //set default to acquire from channels 0,1
  for e := 0 to NumADGains - 1 do
    CInputGain.Items.Add(floattostr(DTAcq.GainValues[e]));
  CInputGain.ItemIndex:= NumADGains - 2; //set gain of 4 (on DT3010) as default

  DTAcq.ListSize:= 2;// 1 chan for imp. input signal, 1 chan for opamp output
{  for e:= 0 to DTAcq.ListSize -1 do
  begin
    DTAcq.ChannelList[e]:= CADChan.ItemIndex + e;
    DTAcq.GainList[e]:= StrtoFloat(CInputGain.Items[CInputGain.ItemIndex]);
  end;}
    DTAcq.ChannelList[0]:= 7;
    DTAcq.GainList[0]:= 4;
    DTAcq.ChannelList[1]:= 15;
    DTAcq.GainList[1]:= 4;

  //allocate a single buffer and put on DTAcq's Ready Queue
  if ErrorMsg(olDmCallocBuffer(0,0, BuffSize * DTAcq.ListSize, 2, @hbuf)) then
  begin
    Showmessage('Problem allocating ADC buffers');
    Exit;
  end;
  DTAcq.Queue := {U}LNG(hbuf);
  BufferReady:= True;

  DTAcq.Config;	// configure subsystem
  //initialise DTDIO
  DTDIO.Board := DTAcq.BoardList[CBoard.ItemIndex]; //select board--same as acq. board
  DTDIO.SubSysType := OLSS_DOUT;//subtype = output
  DTDIO.SubSysElement := 0;//first element
  DTDIO.Resolution := 8;
  if DTDIO.GetSSCaps(OLSSC_SUP_SINGLEVALUE) <> 0
    then DTDIO.DataFlow := OL_DF_SINGLEVALUE //single value operation
    else begin
      ShowMessage(DTDIO.Board+ ' cannot support single-value output');
      Exit;
    end;
  DTDIO.Config;	// configure subsystem
  DOUT:= 0; //impedance mode, no channel selected
  DTDIO.PutSingleValue(0, 1.0, DOUT);
  //Out32 (PortAddress[0], DOUT);

  //initialise DTDAC
  DTDAC.Board := DTAcq.BoardList[CBoard.ItemIndex]; //select board--same as acq. board
  DTDAC.SubSysType := OLSS_DA;//subtype = analog output
  DTDAC.SubSysElement := 0;//first element
  if DTDAC.GetSSCaps(OLSSC_SUP_CONTINUOUS) <> 0
    then DTDAC.DataFlow := OL_DF_CONTINUOUS //continuous operation
    else begin
      ShowMessage(DTDIO.Board+ ' cannot support continuous output');
      Exit;
    end;
  DTDAC.ListSize := 1;
  DTDAC.ChannelList[0]:= 0;
  if DTDAC.GetSSCaps(OLSSC_SUP_INTCLOCK) <> 0 then
    DTDAC.ClockSource := OL_CLK_INTERNAL //set D/A clock to internal
  else begin
    ShowMessage(DTDAC.Board+ ' does not support internal clock');
    Exit;
  end;
  DTDAC.Frequency := 100; //returns DAC clock maximum frequency
  DTDAC.Trigger:= OL_TRG_SOFT; //start DAC with software handle

  DTDAC.WrapMode := OL_WRP_NONE; //multiple buffering
  //allocate buffers and put on Ready Queue
  if ErrorMsg(olDmAllocBuffer(0, 1000, @hbuf)) then
    begin
      Showmessage('Problem allocating DAC buffers');
      Exit;
    end;
    DTDAC.Queue := {U}LNG(hbuf);
  //generate chirp
  SetLength(wave, 4000);
  for e:= 0 to 3999 do
    wave[e]:= Round(0);
  ArrayPtr := @(Wave[0]);
  olDMCopytoBuffer(hbuf, ArrayPtr, 4000);
  DTDAC.Config;	// configure subsystem

  //add all known 54 channel (EIB54 type) polytrodes in drop-down list
  CElectrode.Items.Clear;
  for e := 0 to KNOWNELECTRODES{from ElectrodeTypes} - 1 do
    CElectrode.Items.Add(KnownElectrode[e].Name);
  CElectrode.ItemIndex:= CElectrode.Items.Count - 1;
  CElectrode.Tag:= CElectrode.ItemIndex; //for storing previous selection

  //initialise dynamic array for storing site properties
  SetLength(SiteProperties, KnownElectrode[CElectrode.ItemIndex].NumSites);

  ProgressBar.Max:= KnownElectrode[CElectrode.ItemIndex].NumSites;
  ActiveChanNumber:= ChanSelect.Value;
  GUISiteCol:= $005E879B; //default colour is dimmed

  //initialise timer for automatic functions
  Timer.Interval := TimerInterval;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.TimerTick(Sender: TObject);
begin
  if not(BufferReady) then Exit;
  if TestingImp then
  begin
    if ActiveChanNumber > KnownElectrode[CElectrode.ItemIndex].NumSites then
      PTestImpClick(Timer) {stop}
    else begin
      DOUT := (DOUT and $C0) + MUXOut(SITE2MUX[ActiveChanNumber]);
      DTDIO.PutSingleValue(0,1,DOUT); //select MUX channel
      ChanSelect.Value:= SITE2MUX[ActiveChanNumber] mod 28; //for display purposes only
      if GUICreated then
      with GUIForm do
        begin
          ChangeSiteColor(ActiveChanNumber,clLime{,0}); //highlight active site
          ChangeSiteColor(ActiveChanNumber-1,$005E879B{,0}); //dim last site
          GUIForm.Refresh;
        end;
      DTAcq.Config; //reset buffers/triggers
      Sleep(100); //wait 100ms for MUX switch glitch to pass before sampling
      DTAcq.Start; //grab an ADC buffer
      BufferReady:= False;
      ProgressBar.Position:= ActiveChanNumber; //update progress bar
      Inc(ActiveChanNumber); // increment MUX channel for next TTimer event
      if ActiveChanNumber = 28 then PauseAuto;
      if PTestImp.color = clBtnFace then PTestImp.color:= clLime //flash button
        else PTestImp.color:= clBtnFace;
    end;
  end{TestImp}
  else if Plating then
  begin
    if ActiveChanNumber > KnownElectrode[CElectrode.ItemIndex].NumSites then
      PlateClick(Timer)  {stop}
    else begin //check impedance
      if BufferReady then
      begin
        DOUT := (DOUT and $C0) + MUXOut(SITE2MUX[ActiveChanNumber]);
        DTDIO.PutSingleValue(0,1,DOUT); //select MUX channel
        DTAcq.Config; //reset buffers/triggers
        DTAcq.Start; //grab an ADC buffer
        BufferReady:= false;
      end;
      if SiteProperties[ActiveChanNumber].Impedance > TargetImp.Value then
      begin //plate some more
        //switch to DC mode
        Sleep(BasePlatingTime); // plate for "BasePlatingTime" seconds
        //dec(SiteProperties[ActiveChanNumber].Impedance, 100);  //SIMULATION--- Remove
        //switch back to impedance mode
      end else if ActiveChanNumber = 27 then
        PauseAuto
      else Inc(ActiveChanNumber);
      if pplate.color = clBtnFace then pplate.color:= clRed //make button flash
        else pplate.color:= clBtnFace;
    end;
  end{plating};
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.CElectrodeChange(Sender: TObject);
begin
  if KnownElectrode[CElectrode.ItemIndex].NumSites <> 54 then
  begin
    CElectrode.ItemIndex:= CElectrode.Tag;
    Exit;
  end;
  CElectrode.Tag:= CElectrode.ItemIndex; //only allows EIB54 probes to be selected

  if GUICreated then GUIForm.Close; //remove any existing GUIForms...
  try
    GUIForm:= TPolytrodeGUIForm.Create(Self);//..and create a new one
    GUIForm.Show;
    GUIForm.BringToFront;
  except
    GUICreated:= False;
    Exit; //whatever the exception
  end;
  GUICreated:= True;

  if not GUIForm.CreateElectrode(KnownElectrode[CElectrode.ItemIndex].Name) then
  begin
    ShowMessage('Electrode not defined');
    GUIForm.Free;
    Exit;
  end;
  GUIForm.Caption:= KnownElectrode[CElectrode.ItemIndex].Name;

  SiteProperties:= nil; //clear any previous records
  SetLength(SiteProperties, KnownElectrode[CElectrode.ItemIndex].NumSites);
  ProgressBar.Max:= KnownElectrode[CElectrode.ItemIndex].NumSites;
  ProgressBar.Position:= 0;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.ChanSelectChange(Sender: TObject);
begin
  if (Plating or TestingImp) then Exit; //this allows spinedit 'value' to be changed during auto modes
  if GUICreated then GUIForm.LastChanSelect:= ActiveChanNumber;
  ActiveChanNumber:= ChanSelect.Value;
  DOUT := (DOUT and $C0) + MUXOut(ActiveChanNumber);
  if ModeSelect.ItemIndex = 0 then DOUT:= (DOUT and $0F);
  DTDIO.PutSingleValue(0,1,DOUT);
  if GUICreated then
  begin
    GUIForm.ChangeSiteColor(MUX2EIB[GUIForm.LastChanSelect] - 1{zero based},$005E879B{,0}); //dim last site
    GUIForm.ChangeSiteColor(MUX2EIB[ActiveChanNumber] - 1{zero based},GUISiteCol{,0}); //highlight active site
    GUIForm.Refresh;
  end;
  //Out32 (PortAddress[0], DOUT);
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.ModeSelectClick(Sender: TObject);
begin
  if GUICreated then GUIForm.LastChanSelect:= ActiveChanNumber;
  case ModeSelect.ItemIndex of
    1: begin
         DOUT := MUXOut(ActiveChanNumber); //impedance mode (bit 7 low)
         GUISiteCol:= clLime;
       end;
    2: begin
         DOUT := MUXOut(ActiveChanNumber) + 64; //plating mode (bit 7 high)
         GUISiteCol:= clRed;
       end else
    begin
      DOUT := 0; //clear all bits (channel, EN1&2, and mode)
      DTDIO.PutSingleValue(0,1,DOUT);
      //Out32 (PortAddress[0], DOUT);
      ChanSelect.Enabled:= false;
      GUISiteCol:= $005E879B;
      if GUICreated then
      begin
        GUIForm.ChangeSiteColor(MUX2EIB[ActiveChanNumber],GUISiteCol{,0}); //dim site
        GUIForm.Refresh;
      end;
      Exit;
    end;
  end{case};
  //Out32 (PortAddress[0], DOUT);
  DTDIO.PutSingleValue(0,1,DOUT);
  ChanSelect.Enabled:= true;
  if GUICreated then
  begin
    GUIForm.ChangeSiteColor(MUX2EIB[ActiveChanNumber],GUISiteCol{,0}); //highlight site
    GUIForm.Refresh;
  end;
end;

{-------------------------------------------------------------------------------}
function TEPlateMainForm.MUXOut(a : byte) : byte; //converts 'a' into 8-bit MUX control
var                                               //nb: assumes 'a' is zero based!
  b: byte;
begin
  dec(a); //as MUX is zero-based ie. channel 1 = xx0000
  a:= a mod 28; //wraps 'a' to actual # of MUX channels
  begin
    b:= 16; //ie. 00010000
    b:= b shl (a div 16); //EN1 or EN2 of MUXes (bits 5 & 6, respectively)
    a:= a and $0F; //mask ms-nibble, keep MUX A0-A3 (bits 1 to 4)
    Result:= a + b;
  end;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.ManualAcqClick(Sender: TObject);
begin
  if BufferReady then
  begin
    BufferReady:= false;
    DTAcq.Config;
    DTAcq.Start;
  end;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.DTAcqQueueDone(Sender: TObject);
var ValidSamples,
  FreqIndex, i, Freq    : integer;
  ADCBuffer             : TWaveform;//array of Shrt;
  MUXedSigs             : array of TReal32;//in volts: chx[1],chx+1[1],chx[2],chx+1[2]...
  FFT1, FFT2            : array of TComplex;
  Impedance             : TComplex;
  ArrayPtr              : pointer;
  MUXedSigPtr           : LPReal32;
  VRMSIn, VRMSOut,
  Phase, ADCPhaseOffset : single;
begin
  BufferReady:= False;
  hbuf:= DTAcq.Queue;
  if hbuf = NULL then Exit;
  ErrorMsg(olDmGetValidSamples(hBuf,ULNG(ValidSamples))); //get number of valid samples from buffer
  if ValidSamples = BuffSize*DtAcq.ListSize then //check for dropped samples
  begin
    SetLength(ADCBuffer, ValidSamples+1);
    SetLength(MUXedSigs, ValidSamples+1);
    Setlength(FFT1, FFTSize+1);
    Setlength(FFT2, FFTSize+1);
    ArrayPtr := @(ADCBuffer[1]); //fft algorithm is 1-based
    MUXedSigPtr:= @(MUXedSigs[0]);
    olDmCopyFromBuffer(hbuf, ArrayPtr, ValidSamples);//copy the buffer into local array
    Input2Volts(ADCBuffer, 12, 1, {no external gain}
                MUXedSigPtr, 20/StrtoInt(CInputGain.Items[CInputGain.ItemIndex]));
    VRMSIn:= 0.0; //calculate VRMS of MUXed signals
    VRMSOut:= 0.0;
    i:= 1;
    while i < ValidSamples + 1 do
    begin
      VRMSIn:= VRMSIn + Sqr(MUXedSigs[i]);
      VRMSOut:= VRMSOut + Sqr(MUXedSigs[i+1]);
      Inc(i,2);
    end;

    VRMSIn:= Sqrt(VRMSIn/(ValidSamples / 2));
    VRMSOut:= Sqrt(VRMSOut/(ValidSamples / 2));
    MuxFFT(MUXedSigs, FFT1, FFT2, FFTSize); //calculate FFT

    FreqIndex:= {MaxMagPosn}MaxCmplxRealPos(FFT1)-1; //-1 as first term is DC
    if FreqIndex = 0 then FreqIndex:= 1; //avoids RTL index error (ie. when h/w off)
    Freq:= Round(FreqIndex*SampFreqPerChan/FFTSize);
    ADCPhaseOffset:= Freq/ADCRate*PIBY2; //correct for ADC S&H delay
    Phase:= CPhi(FFT1[FreqIndex]) - CPhi(FFT2[FreqIndex])
            - ADCPhaseOffset + CIRCUITPHI;
    Impedance:= CalcZ(VRMSIn, VRMSOut, Phase, True);

    //update status bar
    StatusBar.Panels[0].Text:= 'Vin='  + FloattoStrF(VRMSIn, ffGeneral, 2, 3)+'Vrms'
                          + ' | Vout=' + FloattoStrF(VRMSOut, ffGeneral, 2, 3)+'Vrms';
    StatusBar.Panels[1].Text:= 'f='    + Inttostr(Freq)+'Hz';
    StatusBar.Panels[2].Text:= 'phi='  + FloattoStrF(RadToDeg(Phase), ffFixed, 3, 1) + '°';
    StatusBar.Panels[3].Text:= 'z='    + FloattoStrF(Impedance.x, ffFixed, 3, 2) + 'MOhm | '
                                       + FloattoStrF(RadToDeg(Impedance.y), ffGeneral, 3, 3) + '°';

    if (Plating or TestingImp) then
    begin
      DOUT := 0; //this is not an ideal place for this to...
      DTDIO.PutSingleValue(0, 1, DOUT); //...disable both MUXes, when flipping polytrode
      SiteProperties[ActiveChanNumber-2].Impedance:= Impedance.x;
      SiteProperties[ActiveChanNumber-2].Phase:= RadToDeg(Impedance.y);
      if (Impedance.x < 0.3) or (Impedance.x > 5.0) then SiteProperties[ActiveChanNumber-2].Faulty:= true
        else SiteProperties[ActiveChanNumber-2].Faulty:= false;
      if (SiteProperties[ActiveChanNumber-2].Phase < -180)
      or (SiteProperties[ActiveChanNumber-2].Phase > -0) then dec(ActiveChanNumber);
    end else
    begin
      SiteProperties[MUX2EIB[ActiveChanNumber]-1].Impedance:= Impedance.x;
      SiteProperties[MUX2EIB[ActiveChanNumber]-1].Phase:= RadToDeg(Impedance.y);
      if (Impedance.x < 0.3) or (Impedance.x > 5.0) then SiteProperties[MUX2EIB[ActiveChanNumber]-1].Faulty:= true
        else SiteProperties[MUX2EIB[ActiveChanNumber]-1].Faulty:= false;
    end;
  end;
  DTAcq.Queue := {U}LNG(hBuf); //recycle buffer
  BufferReady:= True;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.PauseAuto;
begin
  Timer.Enabled:= false;
  Showmessage ('Flip electrode around. Click OK to continue');
  Timer.Enabled:= true;
end;

{-------------------------------------------------------------------------------}
function TEPlateMainForm.CalcZ(const VRMSin, VRMSout, Phase : single;
                        Calibrated : boolean = false) : TComplex;
begin
  Result:= CSet((VRMSOut/OpAmpGain * ZREF) / Sqrt(Sqr((VRMSIn/VOLTDIVIDER) - (VRMSOut/OpAmpGain) * cos(Phase))
               + Sqr(VRMSOut / OpAmpGain * Sin (Phase))){mag},
               Phase + Arctan2(VRMSOut/OPAmpGain * Sin (Phase), (VRMSIn/VOLTDIVIDER
               - VRMSOut/OPAmpGain * Cos (Phase))){phase}, cfRectangular);
  if Calibrated then
    Result:= CSet(Result.x * 1.0958 - 0.1378, Result.y * 0.9865 - 0.008918);
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.CAD_CGChange(Sender: TObject);
var cg : integer;
begin
  for cg:= 0 to DTAcq.ListSize -1 do
  begin
    DTAcq.ChannelList[cg]:= CADChan.ItemIndex + cg;
    DTAcq.GainList[cg]:= StrtoFloat(CInputGain.Items[CInputGain.ItemIndex]);
  end;
  DTAcq.Config; //reconfigure subsystem with new CG list
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.CBoardChange(Sender: TObject);
begin
  DTAcq.Board:= DTAcq.BoardList[CBoard.ItemIndex];
  DTAcq.Config; //reconfigure subsystem with new board selection
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.PTestImpClick(Sender: TObject);
begin
  if not TestingImp then
  begin
    SiteProperties:= nil; //clear existing records
    SetLength(SiteProperties, KnownElectrode[CElectrode.ItemIndex].NumSites);
    ModeSelect.ItemIndex:= 0;
    Manual.Enabled:= false;
    Setup.Enabled:= false;
    PPlate.Enabled:= false;
    CElectrode.Enabled:= false;
    ProgressBar.Position:= 0;
    PTestImp.Caption:= 'Stop Testing';
    PTestImp.BevelOuter:= bvLowered;
    ActiveChanNumber:= 1; //start testing from site 0
    TestingImp:= true;
    Timer.Enabled:= true;
  end else
  begin
    Timer.Enabled:= false;
    PTestImp.Caption:= 'Test Impedances';
    PTestImp.BevelOuter:= bvRaised;
    Manual.Enabled:= true;
    Setup.Enabled:= true;
    PPlate.Enabled:= true;
    CElectrode.Enabled:= true;
    PTestImp.color:= clBtnFace;
    DOUT := 0;
    DTDIO.PutSingleValue(0,1,DOUT); //disable both MUXes, irrespective of current channel
    if GUICreated then
    begin
      GUIForm.ChangeSiteColor(ActiveChanNumber-1,$005E879B{,0}); //dim last site
      GUIForm.Refresh;
    end;
    if DumpCSV.Checked then SaveParametersToFile;
    TestingImp:= false;
  end;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.PlateClick(Sender: TObject);
begin
  if not plating then
  begin
    PPlate.Caption:= 'Stop Plating';
    PPlate.BevelOuter:= bvLowered;
    Manual.Enabled:= false;
    Setup.Enabled:= false;
    PTestImp.Enabled:= false;
    CElectrode.Enabled:= false;
    Timer.Enabled:= true;
    ActiveChanNumber:= 1; //start plating with MUX chan1
    ModeSelect.ItemIndex:= 0;
    Plating:= true;
  end else
  begin
    PPlate.Caption:= 'Start Plating';
    PPlate.BevelOuter:= bvRaised;
    Manual.Enabled:= true;
    Setup.Enabled:= true;
    PTestImp.Enabled:= true;
    CElectrode.Enabled:= true;
    Timer.Enabled:= false;
    PPlate.color:= clBtnFace;
    DOUT := 0;
    DTDIO.PutSingleValue(0,1,DOUT); //disable both MUXes
    Plating:= false;
  end;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.Button1Click(Sender: TObject);
begin
  SaveParametersToFile;
end;

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.SaveParametersToFile;
var FileName : string;
  s : integer;
begin
  FileName := 'C:\Desktop\' + KnownElectrode[CElectrode.ItemIndex].Name + ' Impedance Results.csv';
  SetLength(FileName,Length(FileName));
  AssignFile(Output, FileName);
  if FileExists(FileName) then
   if MessageDlg('Do you really want to overwrite ' + ExtractFileName(FileName) + '?',
     mtConfirmation, [mbYes, mbNo], 0) = IDNo then
   begin
     Showmessage('Impedance results not saved');
     Exit;
   end;
  Rewrite(Output);
  WriteLn(Output, 'Site#, Impedance(MOhm), Phase(deg), Time Plated(sec), Open/Short?');
  for s := 0 to KnownElectrode[CElectrode.ItemIndex].NumSites - 1 do
  begin
    Write(Output, s, ',');
    Write(Output, FloattoStrF(SiteProperties[s].Impedance, ffFixed, 3, 3), ',');
    Write(Output, FloattoStrF(SiteProperties[s].Phase, ffFixed, 3, 2), ',');
    Write(Output, SiteProperties[s].TimePlated, ',');
    Write(Output, SiteProperties[s].Faulty, ',');
    WriteLn(Output);//new line
  end;
  CloseFile(Output);
  end;

{-------------------------------------------------------------------------------}
{procedure TEPlateMainForm.Out32(port : word; bval : byte);
begin
  asm
    push ax  // back up ax
    push dx  // back up dx

    mov dx,port
    mov al,bval
    out dx,al

    pop dx  // restore dx
    pop ax  // restore ax
  end;
end;

{-------------------------------------------------------------------------------}
{function TEPlateMainForm.Inp32(port : word): smallint;
var
  byteval : byte;
begin
  asm
    push ax  // back up ax
    push dx  // back up dx

    mov dx,port
    in al,dx
    mov byteval,al

    pop dx  // restore dx
    pop ax  // restore ax
  end;
  Inp32:=smallint(byteval) and $00FF;
end;}

{-------------------------------------------------------------------------------}
procedure TEPlateMainForm.FormClose(Sender: TObject;
  var Action: TCloseAction);
begin
  if MessageDlg('Did you remember to save site parameters?', mtConfirmation,
    [mbYes, mbNo], 0) = mrYes then
    begin
      DOUT := 0; //clear all bits (channel, EN1&2, and mode)
      DTDIO.PutSingleValue(0,1,DOUT);
      Action := caFree;
    end else
      Action := caNone;
end;

end.
