unit SurfSortMain;
//A unit for the automatic spike sorting of Surf polytrode data
//Also for location prediction

interface

uses Windows, Classes, Messages, Dialogs, Menus, Controls, ComCtrls, Forms,
     ShellApi, SurfPublicTypes, SysUtils, StdCtrls, Graphics, ExtCtrls,
     Gauges, SurfLocateAndSort, Spin, ElectrodeTypes,
     SurfAnal, WaveFormPlotUnit, About;

type
  {TChanObj = class(TWaveFormWin)
    public
      Procedure ThreshChange(pid,cid : integer; ShiftDown,CtrlDown : boolean); override;
    end;
   }
  (*TMultiChan = class(TMultiChanForm)
    public
      Procedure ThreshChange(pid,threshold : integer); override;
    end; *)

  TSurfSortForm = class(TForm)
    OpenDialog: TOpenDialog;
    SaveDialog: TSaveDialog;
    StatusBar: TStatusBar;
    MainMenu1: TMainMenu;
    File1: TMenuItem;
    FileOpenItem: TMenuItem;
    FileSaveAsItem: TMenuItem;
    N1: TMenuItem;
    FileExitItem: TMenuItem;
    Edit1: TMenuItem;
    CutItem: TMenuItem;
    CopyItem: TMenuItem;
    PasteItem: TMenuItem;
    Help1: TMenuItem;
    HelpAboutItem: TMenuItem;
    Guage: TGauge;
    Label4: TLabel;
    nspikes: TLabel;
    Label7: TLabel;
    time: TLabel;
    StopButton: TButton;
    Pause: TCheckBox;
    Display: TCheckBox;
    WaveForms: TPanel;
    ExtGain: TSpinEdit;
    StaticText3: TStaticText;
    Step: TButton;
    Label2: TStaticText;
    em: TEdit;
    ElectrodePick: TComboBox;
    SurfAnal: TSurfAnal;
    evo: TEdit;
    StaticText1: TStaticText;
    LockVo: TCheckBox;
    LockM: TCheckBox;
    FunctionNum: TRadioGroup;
    LockO: TCheckBox;
    eo: TEdit;
    StaticText2: TStaticText;
    Label1: TLabel;
    LockZ: TCheckBox;
    Ez: TEdit;
    StaticText4: TStaticText;
    BLastSpike: TButton;
    BRepeatFile: TButton;
    CWaveFormNormalize: TCheckBox;
    BDumpWaveforms: TButton;
    CDumpText: TCheckBox;
    procedure FileNew1Execute(Sender: TObject);
    procedure FileOpen1Execute(Sender: TObject);
    procedure FileSave1Execute(Sender: TObject);
    procedure FileExit1Execute(Sender: TObject);
    procedure HelpAbout1Execute(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure StopButtonClick(Sender: TObject);
    procedure StepClick(Sender: TObject);
    procedure SurfAnalSurfFile(SurfFile: TSurfFileInfo);
    procedure FunctionNumClick(Sender: TObject);
    procedure BRepeatFileClick(Sender: TObject);
    procedure BLastSpikeClick(Sender: TObject);
    procedure BDumpWaveformsClick(Sender: TObject);
  private
    { Private declarations }
    HaltRead,RepeatFile,RepeatLastSpike : boolean;
    tdbm : TBitmap;
    StepNext : boolean;
    MaxWavePt,ChanWithMaxPeak : integer;
    DumpWaveForm : boolean;
    Function SpikeIsArtifact(var Spike : TSpike) : boolean;
    Procedure ComputeWaveformParams(var Spike : TSpike);
  public
    { Public declarations }
    procedure AcceptFiles( var msg : TMessage ); message WM_DROPFILES;
  end;

var
  SurfSortForm: TSurfSortForm;

implementation

{$R *.DFM}

{==========================================================}
procedure TSurfSortForm.AcceptFiles( var msg : TMessage );
const
  cnMaxFileNameLen = 255;var  i,  nCount     : integer;
  acFileName : array [0..cnMaxFileNameLen] of char;
begin
  // find out how many files we're accepting
  nCount := DragQueryFile( msg.WParam,$FFFFFFFF,acFileName,cnMaxFileNameLen );
  // query Windows one at a time for the file name
  for i := 0 to nCount-1 do
  begin
    DragQueryFile( msg.WParam, i, acFileName, cnMaxFileNameLen );
    // do your thing with the acFileName
    {MessageBox( Handle, acFileName, '', MB_OK );}
    //ReadDataFile(acFileName);
    SurfAnal.SendFileRequestToSurf(acFileName);
  end;
  // let Windows know that you're done
  DragFinish( msg.WParam );
end;

{==========================================================}
Function TSurfSortForm.SpikeIsArtifact(var Spike : TSpike) : boolean;
var i,j,p,n : integer;
    median,crit : single;
    sortedparams : array of SmallInt;
begin
(*  //Check for all maxed out spikes
  {For i := 0 to Length(Spike.Param) div 2 do
  begin
  end;}
  //Check for little or no variance, which will also catch all maxed out spikes
  //sum := 0;
  //sumsqr := 0;
  n := Length(Spike.Param) div 2;
  SetLength(SortedParams,n);
  For i := 0 to n-1 do
  begin
    p := Spike.Param[i*2];
    //sum := sum + p;
    //sumsqr := sumsqr + p*p;
    SortedParams[i] := p;
  end;
  //mean := sum/n;
  //std := sqrt(sumsqr/n - sqr(mean));
  For i := 0 to n div 2{-2} do
    For j := i+1 to n-1 do
      if SortedParams[i] < SortedParams[j] then
      begin
        p := SortedParams[i];
        SortedParams[i] := SortedParams[j];
        SortedParams[j] := p;
      end;
  Median := (SortedParams[n div 2-1] + SortedParams[n div 2])/2;
  //LTemp.Caption := Inttostr(round(mean))+','+Inttostr(round(median))+','+Inttostr(round(std));
  //Check for a zeroed out spike
  For i := 0 to Length(Spike.Param) div 2 do
  begin
  end;
  try
    crit := strtofloat(lcritmedian.text);
  except
    crit := 225;
    beep;
  end;

  if abs(Median) > crit
    then SpikeIsArtifact := TRUE
    else SpikeIsArtifact := FALSE;

  SortedParams := nil;
*)
end;

{==========================================================}
procedure TSurfSortForm.ComputeWaveformParams(var Spike : TSpike);
var c,c2,pt,nchans,npts,max,maxpt,maxchan : integer;
    mean,median,sum,tmp : integer;
    Sorted : array of integer;
begin
  With Spike do
  begin
    nchans := Length(waveform);
    SetLength(param,nchans{*2});
    npts := length(waveform[0]);
    if CWaveFormNormalize.Checked then
    begin
      SetLength(Sorted,nchans);
      For pt := 0 to npts-1 do
      begin
        For c := 0 to nchans-1 do
          Sorted[c] := waveform[c,pt];
        For c := 0 to nchans-2 do
          For c2 := c+1 to nchans-1 do
            if Sorted[c] < Sorted[c2] then
            begin
              tmp := Sorted[c];
              Sorted[c] := Sorted[c2];
              Sorted[c2] := tmp;
            end;
       median := (Sorted[nchans div 2] + Sorted[nchans div 2 + 1]) div 2;
       For c := 0 to nchans-1 do
         waveform[c,pt] := waveform[c,pt] - median + 2048;
       {sum := 0;
       For c := 0 to nchans-1 do
         sum := sum + waveform[c,pt];
       mean := sum div nchans;
       For c := 0 to nchans-1 do
         waveform[c,pt] := waveform[c,pt] - mean + 2048;
       }

       end;
      Sorted := nil;
    end;
    //Fnd the largest and smallest values
    max := -2048;
    maxpt := 0;
    maxchan := 0;
    For c := 0 to nchans-1 do
      For pt := 0 to npts-1 do
        if max < {abs}(waveform[c,pt]-2048) then
        begin
          max := {abs}(waveform[c,pt]-2048);
          maxpt := pt;
          maxchan := c;
        end;
    //Now assign all the params
    For c := 0 to nchans-1 do
      param[c] := waveform[c,maxpt]-2048;
    MaxWavePt := maxpt;
    {if maxpt > 0 then dec(maxpt) else inc(maxpt);
    For c := 0 to nchans-1 do
      param[c+nchans] := waveform[c,maxpt]-2048;
    }
  end;
  ChanWithMaxPeak := maxchan;
end;

{==========================================================}
procedure TSurfSortForm.SurfAnalSurfFile(SurfFile: TSurfFileInfo);
const TB = chr(9);
{type
  WaveFormRec = record
    exists : boolean;
    Win : TChanObj;
  end; }
var
  spikeprobeindex : array[0..32] of boolean;
  c,np,p,e,i,nsp,OrigHeight,OrigWidth,tmp,pt,s : integer;
  //WaveForm : array[0..32] of WaveFormRec;
  Electrode : TElectrode;
  chi : single;
  RepeatLoop : boolean;

  z,vo,m,ro : single;

  tFo : TextFile;
  writeit : boolean;

  MultiChan : array of TWaveFormPlotForm;

  Output : TextFile;
  OutFileName : string;

begin
  Show;

  if CDumpText.Checked then
  begin
    OutFileName := SurfFile.FileName;
    SetLength(OutFileName,Length(OutFileName)-3);
    OutFileName := OutFileName + '.txt';
    AssignFile(Output, OutFileName);
    if FileExists(OutFileName)
      then Append(Output)
      else Rewrite(Output);
  end;

  LocSortForm.Show;
  SurfSortForm.BringToFront;

  for p := 0 to 32 do
    spikeprobeindex[p] := FALSE;

  HaltRead := FALSE;
  RepeatLastSpike := FALSE;

  nspikes.caption := inttostr(0);

  //For c := 0 to 32 do
    //WaveForm[c].exists := FALSE;

  OrigWidth := SurfSortForm.ClientWidth;
  OrigHeight := SurfSortForm.ClientHeight;

  With SurfFile do
  begin
    extgain.value := ProbeArray[0].extgain[0];

    if not LocSortForm.CreateElectrode(ElectrodePick.Text,ProbeArray[0].intgain,ExtGain.Value)
    then begin
      ShowMessage('Electrode not defined');
      LocSortForm.Hide;
      exit;
    end;
    if ElectrodePick.ItemIndex <0 then ElectrodePick.ItemIndex := 0;
    GetElectrode(Electrode,ElectrodePick.Items[ElectrodePick.ItemIndex]);

    //Compute params
    for e := 0 to NEvents-1 do
      With SurfEventArray[e] do
        If (EventType = SURF_PT_REC_UFFTYPE) and (subtype = SPIKETYPE) then
          ComputeWaveformParams(ProbeArray[probe].spike[Index]);

    //Setup windows
    np := length(ProbeArray);
    //Setup WaveForm Windows
    for p := 0 to np-1 do
      case ProbeArray[p].ProbeSubType of
        SPIKETYPE :
          begin
            spikeprobeindex[p] := TRUE;
            (*if  SwapBanks.Checked then
              for c := 0 to 7 do
              begin
                //swap banks
                tmp := ProbeArray[p].screenlayout[c].x;
                ProbeArray[p].screenlayout[c].x := ProbeArray[p].screenlayout[c+8].x;
                ProbeArray[p].screenlayout[c+8].x := tmp;
                tmp := ProbeArray[p].screenlayout[c].y;
                ProbeArray[p].screenlayout[c].y := ProbeArray[p].screenlayout[c+8].y;
                ProbeArray[p].screenlayout[c+8].y := tmp;
              end;
            if  Swap67.Checked then
            begin
              //swap sites
              tmp := ProbeArray[p].screenlayout[6].x;
              ProbeArray[p].screenlayout[6].x := ProbeArray[p].screenlayout[7].x;
              ProbeArray[p].screenlayout[7].x := tmp;
              tmp := ProbeArray[p].screenlayout[6].y;
              ProbeArray[p].screenlayout[6].y := ProbeArray[p].screenlayout[7].y;
              ProbeArray[p].screenlayout[7].y := tmp;
            end;

            for c := 0 to ProbeArray[p].numchans-1 do
            begin
              WaveForm[ProbeArray[p].chanlist[c]].exists := TRUE;
              WaveForm[ProbeArray[p].chanlist[c]].win := TChanObj.CreateParented(WaveForms.Handle);
              WaveForm[ProbeArray[p].chanlist[c]].win.InitPlotWin({npts}ProbeArray[p].pts_per_chan,
                                                           {left}ProbeArray[p].screenlayout[c].x,
                                                           {top} ProbeArray[p].screenlayout[C].y,
                                                       {bmheight}50,
                                                        {intgain}ProbeArray[p].intgain,
                                                         {thresh}ProbeArray[p].threshold,
                                                         {trigpt}ProbeArray[p].trigpt,
                                                        {probeid}p,
                                                         {chanid}ProbeArray[p].chanlist[c],
                                                      {probetype}ProbeArray[p].ProbeSubType,
                                                          {title}'Chan '+inttostr(ProbeArray[p].chanlist[c]),
                                                           {view}TRUE,
                                                {acquisitionmode}FALSE);
              WaveForm[ProbeArray[p].chanlist[c]].win.MarkerV.Visible := TRUE;
              WaveForm[ProbeArray[p].chanlist[c]].win.MarkerH.Visible := FALSE;
              if SurfSortForm.ClientWidth < ProbeArray[p].screenlayout[c].x + WaveForm[ProbeArray[p].chanlist[c]].win.width
                then SurfSortForm.ClientWidth := ProbeArray[p].screenlayout[c].x + WaveForm[ProbeArray[p].chanlist[c]].win.width;
              if SurfSortForm.ClientHeight      < StatusBar.height + Guage.Height + WaveForms.Top + WaveForm[ProbeArray[p].chanlist[c]].win.top + WaveForm[ProbeArray[p].chanlist[c]].win.height
                then SurfSortForm.ClientHeight := StatusBar.height + Guage.Height + WaveForms.Top + WaveForm[ProbeArray[p].chanlist[c]].win.top + WaveForm[ProbeArray[p].chanlist[c]].win.height;
            end;
            *)
            if Length(MultiChan) < p+1
              then SetLength(MultiChan,p+1);
            MultiChan[p] := TWaveFormPlotForm.CreateParented(WaveForms.Handle);
            MultiChan[p].InitPlotWin(Electrode,
                                  {npts}ProbeArray[p].pts_per_chan,
                                  {left}2,{top}2,
                                  {thresh}ProbeArray[p].threshold,
                                  {probeid}p,
                                  {trigpt}ProbeArray[p].trigpt,
                                  {probetype}ProbeArray[p].ProbeSubType,
                                  {title}ProbeArray[p].probe_descrip,
                                  {acquisitionmode}FALSE);

            if SurfSortForm.ClientWidth < MultiChan[p].Width then SurfSortForm.ClientWidth := MultiChan[p].Width;
            if SurfSortForm.ClientHeight  < StatusBar.height + Guage.Height + WaveForms.Top + MultiChan[p].Height + 10
              then SurfSortForm.ClientHeight  := StatusBar.height + Guage.Height + WaveForms.Top + MultiChan[p].Height + 10;
            MultiChan[p].Visible := TRUE;
          end;
      end{case};

    RepeatFile := TRUE;
    HaltRead := FALSE;
    RepeatLastSpike := FALSE;

    While RepeatFile do //plot
    begin
      RepeatFile := FALSE;
      LocSortForm.CreateElectrode(ElectrodePick.Text,ProbeArray[0].intgain,ExtGain.Value);

      // Now read the data using the event array
      Guage.MinValue := 0;
      Guage.MaxValue := NEvents-1;
      nsp := 0;
      DumpWaveForm := FALSE;
      e := 0;
      if not HaltRead then
      begin
        RepeatFile := FALSE;
        While (e < NEvents) do
        begin
          p := SurfEventArray[e].probe;
          i := SurfEventArray[e].Index;
          if SurfEventArray[e].EventType = SURF_PT_REC_UFFTYPE {'N'} then//handle spikes and continuous records
            if SurfEventArray[e].subtype = SPIKETYPE  {'S'} then
            begin //spike record found
              if nsp mod 10 = 0 then time.caption := inttostr(ProbeArray[p].spike[i].time_stamp);
              inc(nsp);
              //if SpikeIsArtifact(ProbeArray[p].spike[i]) then ProbeArray[p].spike[i].Cluster := -1;

              //need to get max pt
              ComputeWaveformParams(ProbeArray[p].spike[i]);

              if ProbeArray[p].spike[i].Cluster >= 0 then
              begin
                try
                  z := strtofloat(ez.text);
                  vo := strtofloat(evo.text);
                  m := strtofloat(em.text);
                  ro := strtofloat(eo.text);
                  //etim := strtofloat(etim.text);
                except
                end;
                LocSortForm.ComputeLoc(ProbeArray[p].spike[i], z,vo,m,ro,
                  LockZ.Checked,LockVo.Checked,LockM.Checked,LockO.Checked,FunctionNum.ItemIndex,chi);
                if nsp mod 10 = 0 then nspikes.caption := inttostr(nsp);

              end;

              if Display.Checked then
              begin
                if LocSortForm.CShowUnClustered.Checked or (ProbeArray[p].spike[i].Cluster > 0) then
                  MultiChan[p].TriggerPt := MaxWavePt;
                MultiChan[p].PlotSpike(ProbeArray[p].spike[i]);

                (*
                for c := 0 to ProbeArray[p].numchans-1 do
                begin
                  WaveForm[ProbeArray[p].chanlist[c]].win.MarkerV.Left := WaveForm[ProbeArray[p].chanlist[c]].win.plot.left + MaxWavePt*2;
                  {if c=ChanWithMaxPeak
                    then WaveForm[ProbeArray[p].ChanList[c]].win.PlotWaveForm(ProbeArray[p].spike[i].WaveForm[c],1,overlay.checked)
                    else} WaveForm[ProbeArray[p].ChanList[c]].win.PlotWaveForm(ProbeArray[p].spike[i].WaveForm[c],ProbeArray[p].spike[i].Cluster,overlay.checked);
                end;
                *)
              end;

              if HaltRead or RepeatFile then break;

              DumpWaveForm := FALSE;
              While pause.checked do
              begin
                Application.ProcessMessages;

                if StepNext then
                begin
                  StepNext := FALSE;
                  break;
                end;

                if RepeatLastSpike then
                begin
                  RepeatLastSpike := FALSE;
                  dec(e,2);
                  if e < -1 then e := -1;
                end;

                if RepeatFile or HaltRead then
                begin
                  pause.Checked := false;
                  break;
                end;

                if DumpWaveForm then
                begin
                  AssignFile(tFo,ExtractFilePath(paramstr(0))+'WaveForm.txt');
                  writeit := TRUE;
                  try
                    Append(tFo);
                  except
                    try
                      Rewrite(tFo);
                    except
                      writeit := FALSE;
                    end;
                  end;

                  if writeit then
                  With ProbeArray[p] do
                  begin
                    Writeln(tFo,i,TB,spike[i].Cluster);
                    for c := 0 to numchans-1 do
                    begin
                      Write(tFo,c,TB);
                      For pt := 0 to ProbeArray[p].pts_per_chan-1 do
                        Write(tFo,spike[i].WaveForm[c,pt],TB);
                      Writeln(tFo);
                    end;
                    CloseFile(tFo);
                  end;
                  DumpWaveForm := FALSE;
                end;
              end;
            end {if a spike};
          inc(e);
          Guage.Progress := e;
          Application.ProcessMessages;
        end{while event loop};
        //RepeatFile := TRUE;
        LocSortForm.Finish;
      end;
      nspikes.caption := inttostr(nsp);
      if HaltRead  then break;
      if not RepeatFile then
        pause.Checked := TRUE;
      While pause.checked do
      begin
        Application.ProcessMessages;
        if HaltRead or RepeatFile then pause.checked := FALSE;
      end;
    end;
  end;


//**************** TEXT DUMPING OF SPIKE INFO ************
  if CDumpText.Checked then
  begin
    //get loc form to fill in the spike cluster ids in the event array
    LocSortForm.GetSpikeClusterIDs(SurfFile.ProbeArray[0].Spike);

    //Now write them out to a text file
    //Works only with one probe
    For s := 0 to Length(SurfFile.ProbeArray[0].spike)-1 do
    begin
      Write(Output, IntToStr(SurfFile.ProbeArray[0].spike[s].time_stamp),' ');
      Write(Output, 'S ');//that this is a spike
      //Write(Output, IntToStr(SurfEventArray[e].Probe),' ');
      Writeln(Output, IntToStr(SurfFile.ProbeArray[0].spike[s].cluster),' ');
    end;
    CloseFile(Output);
  end;
//**************** END TEXT DUMPING ************


  For p := 0 to Length(MultiChan)-1 do
    MultiChan[p].Free;

  nsp := 0;
  //ncp := 0;
  for p := 0 to 32 do
    if spikeprobeindex[p] then inc(nsp);

  SurfSortForm.ClientWidth := OrigWidth;
  SurfSortForm.ClientHeight := OrigHeight;
  Guage.Progress := 0;

  LocSortForm.Hide;
end;

procedure TSurfSortForm.FileNew1Execute(Sender: TObject);
begin
  { Do nothing }
end;

procedure TSurfSortForm.FileOpen1Execute(Sender: TObject);
begin
  If OpenDialog.Execute then
    SurfAnal.SendFileRequestToSurf(OpenDialog.FileName);
end;

procedure TSurfSortForm.FileSave1Execute(Sender: TObject);
begin
  //SaveDialog.Execute;
end;

procedure TSurfSortForm.FileExit1Execute(Sender: TObject);
begin
  Close;
end;

procedure TSurfSortForm.HelpAbout1Execute(Sender: TObject);
begin
  AboutBox.ShowModal;
end;

procedure TSurfSortForm.FormCreate(Sender: TObject);
var i,e : integer;
    s : single;
begin
  ElectrodePick.Items.Clear;
  For e := 0 to KNOWNELECTRODES-1 {from ElectrodeTypes} do
    ElectrodePick.Items.Add(KnownElectrode[e].Name);
  ElectrodePick.Items.Add('UnDefined');
  ElectrodePick.ItemIndex := KNOWNELECTRODES;
  DragAcceptFiles( Handle, True );
  FunctionNumClick(nil);
end;

procedure TSurfSortForm.StopButtonClick(Sender: TObject);
begin
  HaltRead := TRUE;
end;

{Procedure TChanObj.ThreshChange(pid,cid : integer; ShiftDown,CtrlDown : boolean);
begin
end;  }

procedure TSurfSortForm.StepClick(Sender: TObject);
begin
  StepNext := TRUE;
end;

procedure TSurfSortForm.FunctionNumClick(Sender: TObject);
begin
  case FunctionNum.ItemIndex of
    0 : begin //inv
          evo.Text := inttostr(6000);
          em.Text := inttostr(45);
          //lockm.checked := true;
          eo.Text := inttostr(5);
        end;
    1 : begin //exp
          evo.Text := inttostr(100);
          em.Text := inttostr(50);
          eo.Text := inttostr(10);
        end;
    2 : begin //gaus
          evo.Text := inttostr(125);
          em.Text := inttostr(50);
          eo.Text := inttostr(0);
        end;
  end;
  //if not (FunctionNum.ItemIndex in [1,2]) then FunctionNum.ItemIndex := 1;
end;


procedure TSurfSortForm.BRepeatFileClick(Sender: TObject);
begin
  RepeatFile := TRUE;
end;

procedure TSurfSortForm.BLastSpikeClick(Sender: TObject);
begin
  RepeatLastSpike := TRUE;
end;

procedure TSurfSortForm.BDumpWaveformsClick(Sender: TObject);
begin
  DumpWaveForm := TRUE;
end;

(*Procedure TMultiChan.ThreshChange(pid,threshold : integer);
begin
  //this must be here, but doesn't do anything in acquisition
end;  *)

end.
