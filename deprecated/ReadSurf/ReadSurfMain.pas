unit ReadSurfMain;
interface

uses Windows, Classes, Messages, Dialogs, Menus, Controls, ComCtrls, Forms,
     SysUtils, StdCtrls, Graphics, ExtCtrls, ElectrodeTypes,
     Gauges, WaveFormPlotUnit, SurfPublicTypes, {SurfAnal,} SurfFileAccess;

type
  TReadSurfForm = class(TForm)
    StatusBar: TStatusBar;
    Label1: TLabel;
    nspikeprobes: TLabel;
    Label2: TLabel;
    ncrprobes: TLabel;
    Label4: TLabel;
    nspikes: TLabel;
    Label5: TLabel;
    ncr: TLabel;
    Label6: TLabel;
    ndig: TLabel;
    StopButton: TButton;
    ProbeLabel: TLabel;
    Label3: TLabel;
    DigVal: TLabel;
    CPause: TCheckBox;
    label13: TLabel;
    mseq: TLabel;
    label12: TLabel;
    ori: TLabel;
    label11: TLabel;
    phase: TLabel;
    Label7: TLabel;
    time: TLabel;
    label0: TLabel;
    DataFileName: TLabel;
    WaveForms: TPanel;
    Guage: TGauge;
    Label8: TLabel;
    xpos: TLabel;
    Label10: TLabel;
    ypos: TLabel;
    Label15: TLabel;
    con: TLabel;
    Label17: TLabel;
    sf: TLabel;
    Label19: TLabel;
    wid: TLabel;
    Label21: TLabel;
    len: TLabel;
    MainMenu1: TMainMenu;
    FileMenu: TMenuItem;
    OpenFile: TMenuItem;
    OpenDialog: TOpenDialog;
    CElectrode: TComboBox;
    bStep: TButton;
    CDumpText: TCheckBox;
    CDisplayCrs: TCheckBox;
    SurfFileAccess: TSurfFileAccess;
    procedure StopButtonClick(Sender: TObject);
    procedure ExperimentClick(Sender: TObject);
    procedure OpenFileClick(Sender: TObject);
    //procedure SurfAnalSurfFile(SurfFile: TSurfFileInfo);
    procedure FormCreate(Sender: TObject);
    procedure bStepClick(Sender: TObject);
    procedure SurfFileAccessNewFile(acFileName: WideString);
  private
    { Private declarations }
    HaltRead,Step : boolean;
  public
    { Public declarations }
  end;

var
  ReadSurfForm: TReadSurfForm;

implementation

uses About;

{$R *.DFM}

procedure TReadSurfForm.StopButtonClick(Sender: TObject);
begin
  HaltRead := TRUE;
end;

procedure TReadSurfForm.ExperimentClick(Sender: TObject);
begin
  mseq.caption := inttostr(0);
  ori.caption := inttostr(0);
  phase.caption := inttostr(0);
end;

procedure TReadSurfForm.OpenFileClick(Sender: TObject);
begin
  //If OpenDialog.Execute then
    //SurfAnal.SendFileRequestToSurf(OpenDialog.Filename);
end;
(*
procedure TReadSurfForm.SurfAnalSurfFile(SurfFile: TSurfFileInfo);
type
  TProbeWin = record
    exists : boolean;
    win : TWaveFormPlotForm;
  end;
var
  spikeprobeindex,crprobeindex : array[0..32] of boolean;
  c,np,p,e,i,nsp,ncp,OrigHeight,OrigWidth : integer;
  w : WORD;
  msb,lsb : BYTE;
  ProbeWin : array[0..SURF_MAX_PROBES-1] of TProbeWin;
  Electrode : TElectrode;

  tmpthreshold : integer;
  plotit : boolean;

  Output : TextFile;
  OutFileName : string;
begin
//check integrity of file
  ReadSurfForm.BringToFront;
  DataFileName.Caption := SurfFile.FileName;
  mseq.caption := inttostr(0);
  ori.caption := inttostr(0);
  phase.caption := inttostr(0);
  digval.caption := inttostr(0);

  if CDumpText.Checked then
  begin
    OutFileName := SurfFile.FileName;
    SetLength(OutFileName,Length(OutFileName)-3);
    OutFileName := OutFileName + 'txt';
    AssignFile(Output, OutFileName);
    if FileExists(OutFileName)
      then Append(Output)
      else Rewrite(Output);
  end;

  for p := 0 to length(SurfFile.ProbeArray)-1 do
  begin
    spikeprobeindex[p] := FALSE;
    crprobeindex[p] := FALSE;
  end;

  HaltRead := FALSE;

  nspikeprobes.caption := inttostr(0);
  ncrprobes.caption := inttostr(0);
  nspikes.caption := inttostr(0);
  ncr.caption := inttostr(0);
  ndig.caption := inttostr(0);

  for p := 0 to length(SurfFile.ProbeArray)-1 do
    ProbeWin[p].exists := FALSE;

  OrigWidth := ReadSurfForm.ClientWidth;
  OrigHeight := ReadSurfForm.ClientHeight;

  With SurfFile do
  begin
    np := length(ProbeArray);
    //Setup WaveForm Windows
    for p := 0 to np-1 do
    begin
      if (ProbeArray[p].ProbeSubType = SPIKETYPE)
      or (ProbeArray[p].ProbeSubType = CONTINUOUSTYPE) then
      begin
        //get the electrode name
        if ProbeArray[p].electrode_name = 'UnDefined' then
        begin
          //if 1-4 channels then pick a corresponding "ANY" electrode
          if (ProbeArray[p].numchans < 5) then
            for i := 0 to KNOWNELECTRODES do
              if ProbeArray[p].numchans = KnownElectrode[i].NumSites then
              begin
                ProbeArray[p].electrode_name := KnownElectrode[i].Name;
                //ProbeArray[p].probe_descrip := KnownElectrode[i].Description;
                if (ProbeArray[p].ProbeSubType = SPIKETYPE)
                  then CElectrode.ItemIndex := CElectrode.Items.IndexOf(ProbeArray[p].electrode_name);
                Break;
              end;
          //if still undefined
          if ProbeArray[p].electrode_name = 'UnDefined' then
          begin
            //pick from the list
            if (CElectrode.ItemIndex >= 0) and (CElectrode.ItemIndex < KNOWNELECTRODES) then
              if ProbeArray[p].numchans = KnownElectrode[CElectrode.ItemIndex].NumSites then
              begin
                ProbeArray[p].electrode_name := KnownElectrode[CElectrode.ItemIndex].Name;
                //ProbeArray[p].probe_descrip := KnownElectrode[CElectrode.ItemIndex].Description;
              end;
          end;
          //if still undefined
          if ProbeArray[p].electrode_name = 'UnDefined' then
          begin
            ShowMessage('Please select an electrode from the list');
            Exit;
          end;
        end else
        begin  //the file comes with an electrode name.  Is is valid?  If so then set the list to it
          i := CElectrode.Items.IndexOf(ProbeArray[p].electrode_name);
          if (i >= 0) and (i <= KNOWNELECTRODES) then
          begin
            if (ProbeArray[p].ProbeSubType = SPIKETYPE)
              then CElectrode.ItemIndex := i;
          end else
          begin
            ShowMessage('No electrode known by the name of '+ProbeArray[p].electrode_name);
            Exit;
          end;
        end;

        //now we have an electrode, fill it in.
        if not GetElectrode(Electrode,ProbeArray[p].electrode_name) then
        begin
          ShowMessage(ProbeArray[p].electrode_name+' is an invalid electrode name');
          exit;
        end;
        //now create the probe window associated with this electrode
        ProbeWin[p].win := TWaveFormPlotForm.CreateParented(WaveForms.Handle);
        ProbeWin[p].exists := TRUE;
        ProbeWin[p].win.InitPlotWin(Electrode,
                                  {npts}ProbeArray[p].pts_per_chan,
                                  {left}ProbeArray[p].ProbeWinLayout.Left,
                                   {top}ProbeArray[p].ProbeWinLayout.Top,
                                  {thresh}ProbeArray[p].Threshold,
                                  {trigpt}ProbeArray[p].TrigPt,
                                  {probeid}p,
                                  {probetype}ProbeArray[p].ProbeSubType,
                                  {title}ProbeArray[p].probe_descrip,
                                  {acquisitionmode}TRUE{FALSE});

        if ReadSurfForm.ClientWidth < ProbeWin[p].win.Width then ReadSurfForm.ClientWidth := ProbeWin[p].win.Width;
        if ReadSurfForm.ClientHeight  < StatusBar.height + Guage.Height + ProbeWin[p].win.Height + 10
          then ReadSurfForm.ClientHeight  := StatusBar.height + Guage.Height + WaveForms.Top + ProbeWin[p].win.Height + 10;
        ProbeWin[p].win.Visible := TRUE;
      end;
      //housekeeping
      case ProbeArray[p].ProbeSubType of
        SPIKETYPE: spikeprobeindex[p] := TRUE;
        CONTINUOUSTYPE : crprobeindex[p] := TRUE;
      end{case};
    end;
    nsp := 0;
    ncp := 0;
    for p := 0 to length(ProbeArray)-1 do
    begin
      if spikeprobeindex[p] then inc(nsp);
      if crprobeindex[p] then inc(ncp);
    end;
    nspikeprobes.caption := inttostr(nsp);
    ncrprobes.caption := inttostr(ncp);

    // Now read the data using the event array
    Guage.MinValue := 0;
    Guage.MaxValue := NEvents-1;
    Step := false;

    for e := 0 to NEvents-1 do
    begin
      p := SurfEventArray[e].probe;
      i := SurfEventArray[e].Index;
      case SurfEventArray[e].EventType of
        SURF_PT_REC_UFFTYPE {'N'}: //handle spikes and continuous records
          case SurfEventArray[e].subtype of
            SPIKETYPE  {'S'}:
              begin //spike record found
                time.caption := inttostr(ProbeArray[p].spike[i].time_stamp);

                tmpThreshold := 2048 + ProbeWin[p].win.SThreshold.value;
                //so now tmpthresh goes from 0 to 4096, like waveforms
                plotit := false;
                if (tmpThreshold > 2048) then   //positive trigger
                  for c := 0 to ProbeArray[p].numchans-1 do
                  begin
                    for w := 0 to ProbeArray[p].pts_per_chan-1 do
                      if ProbeArray[p].spike[i].waveform[c,w] > tmpThreshold then
                      begin
                        plotit := true;
                        break;
                      end;
                    if plotit then break;
                  end
                else
                  for c := 0 to ProbeArray[p].numchans-1 do
                  begin
                    for w := 0 to ProbeArray[p].pts_per_chan-1 do
                      if ProbeArray[p].spike[i].waveform[c,w] < tmpThreshold then
                      begin
                        plotit := true;
                        break;
                      end;
                    if plotit then break;
                  end;
                if plotit then
                begin
                  ProbeWin[p].win.PlotSpike(ProbeArray[p].spike[i]);
                  nspikes.caption := inttostr(i+1);
                end;
//**************** TEXT DUMPING OF SPIKE INFO ************
                if CDumpText.Checked then
                begin
                  Write(Output, IntToStr(ProbeArray[p].spike[i].time_stamp),' ');
                  Write(Output, 'S ');//that this is a spike
                  Write(Output, IntToStr(SurfEventArray[e].Probe),' ');
                  for c := 0 to ProbeArray[p].numchans-1 do
                  begin
                    for w := 0 to ProbeArray[p].pts_per_chan-1 do
                      Write(Output, IntToStr(ProbeArray[p].spike[i].waveform[c,w]),' ');
                    Write(Output,' ');
                  end;
                  Writeln(Output);//end of line
                end;
//**************** END TEXT DUMPING ************
              end;
            CONTINUOUSTYPE {'C'}:
              begin //continuous record found
                time.caption := inttostr(ProbeArray[p].cr[i].time_stamp);
                ProbeWin[p].win.PlotWaveform(ProbeArray[p].cr[i].WaveForm,2{ltgreen});
                //WaveFormWin[ProbeArray[p].ChanList[0]].win.PlotWaveForm(ProbeArray[p].cr[i].WaveForm,2{ltgreen},FALSE{overlay});
                ncr.caption := inttostr(i+1);
              end;
          end;
        SURF_SV_REC_UFFTYPE {'V'}: //handle single values (including digital signals)
          case SurfEventArray[e].subtype of
            SURF_DIGITAL {'D'}:
              begin
                time.caption := inttostr(SValArray[i].time_stamp);
                w := SValArray[i].sval;
                digval.caption := inttostr(w);
                msb := w and $00FF; {get the last byte of this word}
                lsb := w shr 8;      {get the first byte of this word}
                mseq.caption := inttostr(msb*256+lsb);
                ori.caption := inttostr((msb and $01) shl 8 + lsb); //get the last bit of the msb
                phase.caption := inttostr(msb shr 1);//get the first 7 bits of the msb
//**************** TEXT DUMPING OF DIGITAL INFO ************
                if CDumpText.Checked then
                begin
                  Write(Output, IntToStr(SValArray[i].time_stamp),' ');
                  Write(Output, 'D ');//that this is a digital value
                  Write(Output, 'Mseq: '+ IntToStr(msb*256+lsb),' ');
                  Write(Output, 'Ori: '+ IntToStr((msb and $01) shl 8 + lsb),' ');
                  Write(Output, 'Phase: '+ IntToStr(msb shr 1),' ');
                  Writeln(Output);//end of line
                end;
//**************** END TEXT DUMPING ************

                ndig.caption := inttostr(i+1);
              end;
          end;
        SURF_MSG_REC_UFFTYPE {'M'}://handle surf messages
          begin
            time.caption := inttostr(SurfMsgArray[i].time_stamp);
            StatusBar.SimpleText := SurfMsgArray[i].Msg;
          end;
      end {case};
      Guage.Progress := e;
      Application.ProcessMessages;
      if HaltRead then break;
      While cpause.checked do
      begin
        Application.ProcessMessages;
        if HaltRead then break;
        if Step then
        begin
          Step := false;
          break;
        end;
        if not (SurfEventArray[e].EventType = SURF_PT_REC_UFFTYPE) then break;
        case SurfEventArray[e].subtype of
          SPIKETYPE,CONTINUOUSTYPE : if not ProbeWin[p].win.DrawWaveForms then break;
        end{case};
      end;
    end{event loop};
  end;

  if CDumpText.Checked then
  begin
    CloseFile(Output);
  end;

  cpause.checked := true;
  While cpause.checked do
  begin
    Application.ProcessMessages;
    if HaltRead then break;
  end;

  for p := 0 to length(SurfFile.ProbeArray)-1 do
    if ProbeWin[p].exists then
    begin
       ProbeWin[p].win.free;
       ProbeWin[p].exists := false;
    end;

  //free memory?
  ReadSurfForm.ClientWidth := OrigWidth;
  ReadSurfForm.ClientHeight := OrigHeight;
  Guage.Progress := 0;
end;
*)
procedure TReadSurfForm.FormCreate(Sender: TObject);
var e : integer;
begin
  CElectrode.Items.Clear;
  For e := 0 to KNOWNELECTRODES-1 {from ElectrodeTypes} do
    CElectrode.Items.Add(KnownElectrode[e].Name);
  CElectrode.Items.Add('UnDefined');
  CElectrode.ItemIndex := KNOWNELECTRODES;
end;

procedure TReadSurfForm.bStepClick(Sender: TObject);
begin
  Step := TRUE;
end;

//================= New Method for reading files with SurfFileAccess ===
procedure TReadSurfForm.SurfFileAccessNewFile(acFileName: WideString);
type
  TProbeWin = record
    exists : boolean;
    win : TWaveFormPlotForm;
  end;
var
  spk,p,i,c,s,e,m : integer;
  OrigHeight,OrigWidth : integer;
  w : WORD;
  ProbeWin : array[0..SURF_MAX_PROBES-1] of TProbeWin;
  Electrode : TElectrode;
  ProbeArray : TProbeArray;
  tmpthreshold : integer;
  plotit : boolean;

  Spike : TSpike;
  Cr : TCr;
  Sval : TSval;
  SurfMsg : TSurfMsg;

begin
  //This function is only called when the file has been read and is valid
  ReadSurfForm.BringToFront;
  DataFileName.Caption := acFileName;
  mseq.caption := inttostr(0);
  ori.caption := inttostr(0);
  phase.caption := inttostr(0);
  digval.caption := inttostr(0);

  nspikeprobes.caption := inttostr(0);
  ncrprobes.caption := inttostr(0);
  nspikes.caption := inttostr(0);
  ncr.caption := inttostr(0);
  ndig.caption := inttostr(0);

  for p := 0 to SurfFileAccess.GetNumProbes-1 do
    ProbeWin[p].exists := FALSE;

  OrigWidth := ReadSurfForm.ClientWidth;
  OrigHeight := ReadSurfForm.ClientHeight;

  //Get the array of probes layout records
  SetLength(ProbeArray,SurfFileAccess.GetNumProbes);
  for p := 0 to SurfFileAccess.GetNumProbes-1 do
  begin
    SurfFileAccess.GetProbeRecord(p,ProbeArray[p]);
  end;

  HaltRead:= false;

  //Setup WaveForm Windows
  for p := 0 to SurfFileAccess.GetNumProbes-1 do
  begin
    if (ProbeArray[p].ProbeSubType = SPIKETYPE)
    or (CDisplayCrs.Checked and (ProbeArray[p].ProbeSubType = CONTINUOUSTYPE)) then
    begin
      //get the electrode name
      if ProbeArray[p].electrode_name = 'UnDefined' then
      begin
        //if 1-4 channels then pick a corresponding "ANY" electrode
        if (ProbeArray[p].numchans < 5) then
          for i := 0 to KNOWNELECTRODES do
            if ProbeArray[p].numchans = KnownElectrode[i].NumSites then
            begin
              ProbeArray[p].electrode_name := KnownElectrode[i].Name;
              if (ProbeArray[p].ProbeSubType = SPIKETYPE)
                then CElectrode.ItemIndex := CElectrode.Items.IndexOf(ProbeArray[p].electrode_name);
              Break;
            end;
        //if still undefined
        if ProbeArray[p].electrode_name = 'UnDefined' then
        begin
          //pick from the list
          if (CElectrode.ItemIndex >= 0) and (CElectrode.ItemIndex < KNOWNELECTRODES) then
            if ProbeArray[p].numchans = KnownElectrode[CElectrode.ItemIndex].NumSites then
            begin
              ProbeArray[p].electrode_name := KnownElectrode[CElectrode.ItemIndex].Name;
            end;
        end;
        //if still undefined
        if ProbeArray[p].electrode_name = 'UnDefined' then
        begin
          ShowMessage('Please select an electrode from the list');
          Exit;
        end;
      end else
      begin  //the file comes with an electrode name.  Is is valid?  If so then set the list to it
        i := CElectrode.Items.IndexOf(ProbeArray[p].electrode_name);
        if (i >= 0) and (i <= KNOWNELECTRODES) then
        begin
          if (ProbeArray[p].ProbeSubType = SPIKETYPE)
            then CElectrode.ItemIndex := i;
        end else
        begin
          ShowMessage('No electrode known by the name of '+ProbeArray[p].electrode_name);
          Exit;
        end;
      end;

      //now we have an electrode, fill it in.
      if not GetElectrode(Electrode,ProbeArray[p].electrode_name) then
      begin
        ShowMessage(ProbeArray[p].electrode_name+' is an invalid electrode name');
        exit;
      end;
      //now create the probe window associated with this electrode
      ProbeWin[p].win := TWaveFormPlotForm.CreateParented(WaveForms.Handle);
      ProbeWin[p].exists := TRUE;
      ProbeWin[p].win.InitPlotWin(Electrode,
                                {npts}ProbeArray[p].pts_per_chan,
                                {left}ProbeArray[p].ProbeWinLayout.Left,
                                 {top}ProbeArray[p].ProbeWinLayout.Top,
                                {thresh}ProbeArray[p].Threshold,
                                {trigpt}ProbeArray[p].TrigPt,
                                {probeid}p,
                                {probetype}ProbeArray[p].ProbeSubType,
                                {title}ProbeArray[p].probe_descrip,
                                {acquisitionmode}TRUE{FALSE});

      if ReadSurfForm.ClientWidth < ProbeWin[p].win.Width then ReadSurfForm.ClientWidth := ProbeWin[p].win.Width;
      if ReadSurfForm.ClientHeight  < StatusBar.height + Guage.Height + ProbeWin[p].win.Height + 10
        then ReadSurfForm.ClientHeight  := StatusBar.height + Guage.Height + WaveForms.Top + ProbeWin[p].win.Height + 10;
      ProbeWin[p].win.Visible := TRUE;
    end;
  end;
  // Now read the data using SurfFileAccess
  p := 0;//assume probe with spikes is on probe 0--I know I know, Tim, bad assumption, so search for them

  Guage.MinValue := 0;
  Guage.MaxValue := SurfFileAccess.GetNumSpikes(p)-1;
  Step := false;
  nspikeprobes.caption := IntToStr(SurfFileAccess.GetNumProbes);
  nspikes.caption := IntToStr(SurfFileAccess.GetNumSpikes(p));
  plotit := false;
  //ncrprobes.caption := inttostr(ncp);

  for spk := 0 to SurfFileAccess.GetNumSpikes(p)-1 do
  begin
    SurfFileAccess.GetSpike(p,spk,Spike);
//    SurfFileAccess.GetSpike(p,spk,Spike);

    time.caption := inttostr(Spike.time_stamp);

    tmpThreshold := 2048 + ProbeWin[p].win.SThreshold.value;
    //so now tmpthresh goes from 0 to 4096, like waveforms
    plotit := false;
    if (tmpThreshold > 2048) then   //positive trigger
      for c := 0 to ProbeArray[p].numchans-1 do
      begin
        for w := 0 to ProbeArray[p].pts_per_chan-1 do
          if Spike.waveform[c,w] > tmpThreshold then
          begin
            plotit := true;
            break;
          end;
        if plotit then break;
      end
    else
      for c := 0 to ProbeArray[p].numchans-1 do
      begin
        for w := 0 to ProbeArray[p].pts_per_chan-1 do
          if Spike.waveform[c,w] < tmpThreshold then
          begin
            plotit := true;
            break;
          end;
        if plotit then break;
      end;
    if plotit then
    begin
      ProbeWin[p].win.PlotSpike(Spike);

      //nspikes.caption := inttostr(i+1);

      //An example of writing the spike (changing the clusterid) to the file
      Spike.cluster := 2;//random(MAXCLUSTERS);
      SurfFileAccess.SetSpikeClusterId(p,spk,Spike.cluster);
    end;

    Guage.Progress := spk;
    Application.ProcessMessages;
    if HaltRead then break;
    While cpause.checked do
    begin
      Application.ProcessMessages;
      if HaltRead then break;
      if Step then
      begin
        Step := false;
        break;
      end;
      if not ProbeWin[p].win.DrawWaveForms then break;
    end;
  end;

  if CDumpText.Checked then
  begin
    CloseFile(Output);
  end;

  cpause.checked := true;
  While cpause.checked do
  begin
    Application.ProcessMessages;
    if HaltRead then break;
  end;

  for p := 0 to SurfFileAccess.GetNumProbes-1 do
    if ProbeWin[p].exists then
    begin
       ProbeWin[p].win.free;
       ProbeWin[p].exists := false;
    end;

  //free memory?
  ReadSurfForm.ClientWidth := OrigWidth;
  ReadSurfForm.ClientHeight := OrigHeight;
  Guage.Progress := 0;

  //Example of how to read the cr records from SurfFileAceess
  (*
  for p := 0 to SurfFileAccess.GetNumProbes-1 do
  begin
    for c := 0 to SurfFileAccess.GetNumCrs(p)-1 do
    begin
      SurfFileAccess.GetCr(p,c,Cr);
    end;
  end;
  *)

  //Example of how to read the digital values from SurfFileAceess
  (*
  for s := 0 to SurfFileAccess.GetNumSVals-1 do
  begin
    SurfFileAccess.GetSVal(s,SVal);
    //do something with the sval record
  end;
  *)

  //Example of how to read the message values from SurfFileAceess
  (*
  for m := 0 to SurfFileAccess.GetNumSurfMsgs-1 do
  begin
    SurfFileAccess.GetSurfMsg(m,SurfMsg);
    //do something with the message record
  end;
  *)

  //Example of how to read the event array (chronological order of all events)
  (*
  for e := 0 to Length(SurfFileAccess.GetEventArray)-1 do
  begin
    s := SurfFileAccess.GetEventArray[e].Time_Stamp;
    p := SurfFileAccess.GetEventArray[e].Probe;
    c := SurfFileAccess.GetEventArray[e].Index;
    //also  SurfFileAccess.GetEventArray[e].EventType
    //also  SurfFileAccess.GetEventArray[e].SubType
    //now, depending on the event array eventtype and subtype, you can recreate the
    //flow of the experiment using the other spike,cr,sval, and msg access functions above
  end;
  *)

  //Example of how to zero out the spike clusters all in one go:
  (*
  for p := 0 to SurfFileAccess.GetNumProbes-1 do
  begin
    for s := 0 to SurfFileAccess.GetNumSpikes(p)-1 do
    begin
      SurfFileAccess.SetSpikeClusterID(p,s,0);
    end;
  end;
  *)

  //free memory
  ProbeArray := nil;
  Spike.waveform := nil;
  Cr.waveform := nil;

  SurfFileAccess.Close;
end;

end.
