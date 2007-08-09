{ (c) 1994-99 Phil Hetherington, P&M Research Technologies, Inc.}
UNIT SurfFile;
INTERFACE
USES WINDOWS,UFFTYPES,DIALOGS,SYSUTILS,CLASSES,SURFTYPES,SURFPUBLICTYPES,GRAPHICS,FORMS,FileProgressUnit;
//CONST
  //NOSTIMULUS = -9999;
TYPE
  SURF_FILE_HEADER          = UFF_FILE_HEADER;
  SURF_DATA_REC_DESC_BLOCK  = UFF_DATA_REC_DESC_BLOCK;

  flagtype = array[0..1] of char;

  TSurfFile = class
    private
      { Private declarations }
      SurfStream : TFileStream;
      FileIsOpen : boolean;
      ProbeWaveFormLength : array[0..255] of integer;
      Function GetSurfFileHeader(var Header : SURF_FILE_HEADER) : boolean;{success/nosuccess}
      Function PutSurfFileHeader(var Header : UFF_FILE_HEADER) : boolean;{success/nosuccess}
      Function GetSURFDRDB(var drdb : SURF_DATA_REC_DESC_BLOCK) : boolean;{success/nosuccess}
      Function PutSURFDRDB(var drdb : UFF_DATA_REC_DESC_BLOCK) : boolean;{success/nosuccess}

      Function OpenSurfFileForRead(Filename : wideString) : boolean;{success/nosuccess}
      Function GetNextFlag(var rt : flagtype) : boolean;{success/nosuccess}
      Function GetPolytrodeRecord(var PTRecord : SURF_PT_REC) : boolean;{success/nosuccess}
      Function GetSingleValueRecord(var SVRecord : SURF_SV_REC) : boolean;{success/nosuccess}
      Function GetSurfRecord(var SurfRecord_v1 : SURF_LAYOUT_REC_V1) : boolean;{success/nosuccess}
      Function GetMessageRecord(var msg : SURF_MSG_REC) : boolean;{success/nosuccess}
    public
      { Public declarations }
      errors : boolean;
      SurfFileName : widestring;
      FileSize,FilePosition : Integer;
      ptdrdb,lrdrdb,svdrdb,msgdrdb : UFF_DATA_REC_DESC_BLOCK;

      SurfRecord_v1 : SURF_LAYOUT_REC_V1;
      Prb : TProbeArray;
      SVal: TSValArray;
      Msg : TSurfMsgArray;
      SurfEvent : TSurfEventArray;
      NEvents : LNG;
      PositionsExist : boolean;

      constructor Create;
      Procedure CleanUp;
      Function OpenSurfFileForWrite(Filename : wideString) : boolean;{success/nosuccess}
      Function PutPolytrodeRecord(var PTRecord : SURF_PT_REC) : boolean;{success/nosuccess}
      Function PutSingleValueRecord(var SVRecord : SURF_SV_REC) : boolean;{success/nosuccess}
      Function PutSurfRecord(var SurfRecord_v1 : SURF_LAYOUT_REC_V1) : boolean;{success/nosuccess}
      Function PutMessageRecord(var msg : SURF_MSG_REC) : boolean;{success/nosuccess}
      Function ReadEntireSurfFile(Filename : wideString; ReadSpikeWaveForms,AverageWaveforms : boolean) : boolean;{success/nosuccess}
      Procedure CloseFile;
    end;

IMPLEMENTATION

{=========================================================================}
Constructor TSurfFile.Create;
begin
  FileIsOpen := FALSE;
end;

{=========================================================================}
Procedure TSurfFile.CleanUp;
var np,p,ns,s,nc,c : integer;
begin
  np := Length(prb);
  For p := 0 to np-1 do
  begin
    ns := Length(prb[p].Spike);
    for s := 0 to ns-1 do
    begin
      nc := Length(prb[p].Spike[s].waveform);
      for c := 0 to nc-1 do prb[p].Spike[s].waveform[c] := nil;
      prb[p].Spike[s].waveform := nil;
    end;
    prb[p].Spike := nil;
    nc := Length(prb[p].Cr);
    for c := 0 to nc-1 do
      prb[p].CR[c].waveform := nil;
    prb[p].CR := nil;
  end;
  prb := nil;
  SVal := nil;
  Msg := nil;
  SurfEvent := nil;
end;


{=========================================================================}
Function TSurfFile.OpenSurfFileForRead(Filename : wideString) : boolean;{success/nosuccess}
var header : UFF_FILE_HEADER;
    drdb : UFF_DATA_REC_DESC_BLOCK;
    ok : boolean;
begin
  //Check to see if file exists
  surfStream := TFileStream.Create(Filename, fmOpenRead);
  SurfFileName := Filename;
  FileSize := SurfStream.Size;
  FileIsOpen := TRUE;
  OpenSurfFileForRead := FileIsOpen;
  //READ THE HEADER
  if not GetSurfFileHeader(Header) then
  begin
    OpenSurfFileForRead := FALSE;
    surfStream.free;
    exit;
  end;

  //READ THE DRDBS
  ok := TRUE;
  While ok do
  begin
    if GetSurfDRDB(drdb) then
    begin
      //ShowMessage('DR_name: '+ drdb.DR_name);
      case drdb.DR_rec_type of
        SURF_PT_REC_UFFTYPE : move(drdb,ptdrdb,sizeof(drdb));
        SURF_SV_REC_UFFTYPE : move(drdb,svdrdb,sizeof(drdb));
        SURF_PL_REC_UFFTYPE : move(drdb,lrdrdb,sizeof(drdb));
        SURF_MSG_REC_UFFTYPE : move(drdb,msgdrdb,sizeof(drdb));
        else                  begin
                                ok := false;
                                OpenSurfFileForRead := FALSE;
                                ShowMessage('Unknown data record found');
                              end;
      end;
    end else begin
      //not a drdb, so backup read position
      surfStream.Seek(-sizeof(drdb),soFromCurrent);
      ok := FALSE;
    end;
  end;
  FilePosition := SurfStream.Position;
end;

{=========================================================================}
Function TSurfFile.OpenSurfFileForWrite(Filename : wideString) : boolean;{success/nosuccess}
var SystemTime : TSystemTime;
    l,i,p : integer;
    ufn,upath : widestring;
    header : UFF_FILE_HEADER;
    ptdrdb,lrdrdb,svdrdb,msgdrdb : UFF_DATA_REC_DESC_BLOCK;
begin
  //Check to see if file exists already
  SurfStream := TFileStream.Create(Filename, fmCreate OR fmShareExclusive);
  SurfFileName := Filename;
  FileIsOpen := TRUE;
  OpenSurfFileForWrite := FileIsOpen;

  GetLocalTime(SystemTime);

  upath := '';
  ufn := '';
  l := length(Filename);

  p := l;
  While (p > 0) and (Filename[p] <> '\') do dec(p);
  if p = 0 then p := 1;

  upath := Copy(Filename,1,p);
  ufn := Copy(Filename,p+1,l-p);

  //CREATE THE HEADER BLOCK
  With header do
  begin
    FH_rec_type     := 1; // 1 must be 1
    FH_rec_type_ext := 0; // 1 must be 0
    UFF_name        := 'UFF'; // 10 must be "UFF" sz
    UFF_major       := 6;  // 1 major UFF ver
    UFF_minor       := 0;  // 1 minor UFF ver
    FH_rec_len      := WORD(sizeof(UFF_FILE_HEADER));  // 2 FH record length in bytes
    DRDB_rec_len    := WORD(sizeof(UFF_DRDB_RSFD));;  // 2 DBRD record length in bytes
    bi_di_seeks     := FALSE; // 2 bi-directional seeks format

    OS_name         := 'WINDOWS 98';  // 12 OS name ie "MS-DOS"
    OS_major        := 4;  // 1 OS major rev
    OS_minor        := 1;  // 1 OS minor rev

    create.Sec      :=  SystemTime.wSecond;
    create.Min      :=  SystemTime.wMinute;
    create.Hour     :=  SystemTime.wHour;
    create.Day      :=  SystemTime.wDay;
    create.Month    :=  SystemTime.wMonth;
    create.Year     :=  SystemTime.wYear;

    append.Sec      :=  SystemTime.wSecond;
    append.Min      :=  SystemTime.wMinute;
    append.Hour     :=  SystemTime.wHour;
    append.Day      :=  SystemTime.wDay;
    append.Month    :=  SystemTime.wMonth;
    append.Year     :=  SystemTime.wYear;

    node            := ''; // 32 system node name - same as BDT        huh?
    device          := ''; // 32 device name - same as BDT             huh?
    l := length(upath);
    if l > UFF_PATH_LEN then l := UFF_PATH_LEN;
    for i := 0 to l-1 do path[i] := char(upath[i+1]); // 160 path name

    l := length(ufn);
    if l > UFF_FILENAME_LEN then l := UFF_FILENAME_LEN;
    for i := 0 to l-1 do filename[i] := char(ufn[i+1]); // 32 original file name at creation
    pad             := '';  // 76 pad area to bring uff area to 512

    app_info        := 'Surf 1.0';  // 32 application task name & version
    user_name       := 'User name'; // 14 user's name as owner of file
    file_desc       := 'Surf data acquisition system';  // 64 description of file/exp
    FillChar(user_area,sizeof(user_area),0);  // 1536 additional user area
    bd_FH_rec_type  := 1; // 1 must be 1 BIDIRECTIONAL SUPPORT
    bd_FH_rec_type_ext := 0; // 1 must be 0 BIDIRECTIONAL SUPPORT
  end;
  //WRITE THE HEADER
  if not PutSURFFileHeader(Header) then OpenSurfFileForWrite := FALSE;
  //CREATE THE DRDBS

  With ptdrdb do
  begin
    DRDB_rec_type     := 2;    // record type; must be 2
    DRDB_rec_type_ext := 0;    // record type extension; must be 0
    DR_rec_type       := SURF_PT_REC_UFFTYPE;  // 'N' Data Record type for DBRD 3-255  233=ACTREC
    DR_rec_type_ext   := 0;    // Data Record type ext; ignored
    DR_size           := -1;   // Data Record size in bytes  SIZES GIVEN IN THE SLRS
    DR_name           :='POLYTRODE RECORD';// Data Record name
    DR_num_fields     := 0;    // number of sub-fields in Data Record
    FillChar(DR_pad,UFF_DRDB_PAD_LEN,0);  // pad bytes for expansion
    FillChar(DR_subfields,UFF_RSFD_PER_DRDB*sizeof(UFF_DRDB_RSFD),0);  // pad bytes for expansion
    bd_DRDB_rec_type  := 2;   // record type; must be 2 BIDIRECTIONAL SUPPORT
    bd_DRDB_rec_type_ext := 0; // record type extension; must be 0 BIDIRECTIONAL SUPPORT
  end;
  //WRITE THIS DRDB
  if not PutSURFDRDB(ptdrdb) then OpenSurfFileForWrite := FALSE;

  With svdrdb do
  begin
    DRDB_rec_type     := 2;    // record type; must be 2
    DRDB_rec_type_ext := 0;    // record type extension; must be 0
    DR_rec_type       := SURF_SV_REC_UFFTYPE;  // 'V'
    DR_rec_type_ext   := 0;    // Data Record type ext; ignored
    DR_size           := sizeof(SURF_SV_REC);   // Data Record size in bytes
    DR_name           :='SINGLE VALUE RECORD';// Data Record name
    DR_num_fields     := 0;    // number of sub-fields in Data Record
    FillChar(DR_pad,UFF_DRDB_PAD_LEN,0);  // pad bytes for expansion
    FillChar(DR_subfields,UFF_RSFD_PER_DRDB*sizeof(UFF_DRDB_RSFD),0);  // pad bytes for expansion
    bd_DRDB_rec_type  := 2;   // record type; must be 2 BIDIRECTIONAL SUPPORT
    bd_DRDB_rec_type_ext := 0; // record type extension; must be 0 BIDIRECTIONAL SUPPORT
  end;
  //WRITE THIS DRDB
  if not PutSURFDRDB(svdrdb) then OpenSurfFileForWrite := FALSE;

  With lrdrdb do
  begin
    DRDB_rec_type     := 2;    // record type; must be 2
    DRDB_rec_type_ext := 0;    // record type extension; must be 0
    DR_rec_type       := SURF_PL_REC_UFFTYPE;  // (234) Data Record type for DBRD 3-255  233=ACTREC
    DR_rec_type_ext   := 0;    // Data Record type ext; ignored
    DR_size           := sizeof(SURF_LAYOUT_REC_V1); // Data Record size in bytes
    DR_name           :='SURF LAYOUT RECORD';// Data Record name
    DR_num_fields     := 0;    // number of sub-fields in Data Record
    FillChar(DR_pad,UFF_DRDB_PAD_LEN,0);  // pad bytes for expansion
    FillChar(DR_subfields,UFF_RSFD_PER_DRDB*sizeof(UFF_DRDB_RSFD),0);  // pad bytes for expansion
    bd_DRDB_rec_type  := 2;   // record type; must be 2 BIDIRECTIONAL SUPPORT
    bd_DRDB_rec_type_ext := 0; // record type extension; must be 0 BIDIRECTIONAL SUPPORT
  end;
  //WRITE THIS DRDB
  if not PutSURFDRDB(lrdrdb) then OpenSurfFileForWrite := FALSE;

  With msgdrdb do
  begin
    DRDB_rec_type     := 2;    // record type; must be 2
    DRDB_rec_type_ext := 0;    // record type extension; must be 0
    DR_rec_type       := SURF_MSG_REC_UFFTYPE;
    DR_rec_type_ext   := 0;    // Data Record type ext; ignored
    DR_size           := sizeof(SURF_MSG_REC_UFFTYPE); // Data Record size in bytes
    DR_name           :='SURF MESSAGE RECORD';// Data Record name
    DR_num_fields     := 0;    // number of sub-fields in Data Record
    FillChar(DR_pad,UFF_DRDB_PAD_LEN,0);  // pad bytes for expansion
    FillChar(DR_subfields,UFF_RSFD_PER_DRDB*sizeof(UFF_DRDB_RSFD),0);  // pad bytes for expansion
    bd_DRDB_rec_type  := 2;   // record type; must be 2 BIDIRECTIONAL SUPPORT
    bd_DRDB_rec_type_ext := 0; // record type extension; must be 0 BIDIRECTIONAL SUPPORT
  end;
  //WRITE THIS DRDB
  if not PutSURFDRDB(msgdrdb) then OpenSurfFileForWrite := FALSE;

end;

{=========================================================================}
Procedure TSurfFile.CloseFile;
begin
  //Check to see if file is even open
  if FileIsOpen then surfStream.Free;
  FileIsOpen := FALSE;
  SurfFileName := '';
end;

{=========================================================================}
Function TSurfFile.GetNextFlag(var rt : flagtype) : boolean;{success/nosuccess}
begin
  try
    surfStream.ReadBuffer(rt, 2);
  except
    {if EReadError}
    GetNextFlag := FALSE;
    exit;
  end;
  GetNextFlag := TRUE;
  Surfstream.Seek(-2,soFromCurrent);
  FilePosition := SurfStream.Position;
end;

{=========================================================================}
Function TSurfFile.PutSurfFileHeader(var Header : UFF_FILE_HEADER) : boolean;{success/nosuccess}
begin
  try
    SurfStream.WriteBuffer(header, sizeof(UFF_FILE_HEADER));
    PutSurfFileHeader := TRUE;
  except
    {if EWriteError}
    PutSurfFileHeader := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.GetSurfFileHeader(var Header : UFF_FILE_HEADER) : boolean;{success/nosuccess}
begin
  try
    surfStream.ReadBuffer(header, sizeof(SURF_FILE_HEADER));
    if header.UFF_name <> 'UFF'
      then GetSurfFileHeader := FALSE
      else GetSurfFileHeader := TRUE;
  except
    {if EReadError}
    GetSurfFileHeader := FALSE;
  end;
end;
{=========================================================================}
Function TSurfFile.PutSURFDRDB(var drdb : UFF_DATA_REC_DESC_BLOCK) : boolean;{success/nosuccess}
begin
  try
    SurfStream.WriteBuffer(drdb, sizeof(UFF_DATA_REC_DESC_BLOCK));
    PutSURFDRDB := TRUE;
  except
    {if EWriteError}
    PutSURFDRDB := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.GetSurfDRDB(var drdb : UFF_DATA_REC_DESC_BLOCK) : boolean;{success/nosuccess}
begin
  try
    surfStream.ReadBuffer(drdb, sizeof(SURF_DATA_REC_DESC_BLOCK));
    if drdb.DRDB_rec_type <> 2
      then GetSurfDRDB := FALSE
      else GetSurfDRDB := TRUE;
  except
    {if EReadError}
    GetSurfDRDB := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.PutPolytrodeRecord(var PTRecord : SURF_PT_REC) : boolean;{success/nosuccess}
begin
  try
    //write the record w/o the waveform
    SurfStream.WriteBuffer(PTRecord, sizeof(SURF_PT_REC)-4{the waveform pointer});
    //now write the waveform
    SurfStream.WriteBuffer(PTRecord.adc_waveform[0],Length(PTRecord.adc_waveform)*2);
    PutPolytrodeRecord := TRUE;
  except
    PutPolytrodeRecord := FALSE;
  end;
end;
{=========================================================================}
Function TSurfFile.GetPolytrodeRecord(var PTRecord : SURF_PT_REC) : boolean;{success/nosuccess}
begin
  try
    //read the record w/o the waveform
    surfStream.ReadBuffer(PTRecord, sizeof(SURF_PT_REC)-4{the waveform pointer});
    SetLength(PTRecord.adc_waveform,ProbeWaveFormLength[PTRecord.probe]);
    //now read the waveform
    surfStream.ReadBuffer(PTRecord.adc_waveform[0],Length(PTRecord.adc_waveform)*2{sizeof(SHRT)});
    GetPolytrodeRecord := TRUE;
  except
    {if EReadError}
    GetPolytrodeRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.PutSingleValueRecord(var SVRecord : SURF_SV_REC) : boolean;{success/nosuccess}
begin
  try
    SurfStream.WriteBuffer(SVRecord, sizeof(SURF_SV_REC));
    PutSingleValueRecord := TRUE;
  except
    PutSingleValueRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.GetSingleValueRecord(var SVRecord : SURF_SV_REC) : boolean;{success/nosuccess}
begin
  try
    surfStream.ReadBuffer(SVRecord, sizeof(SURF_SV_REC));
    GetSingleValueRecord := TRUE;
  except
    {if EReadError}
    GetSingleValueRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.PutSurfRecord(var SurfRecord_V1 : SURF_LAYOUT_REC_V1) : boolean;{success/nosuccess}
begin
  try
    SurfStream.WriteBuffer(SurfRecord_V1, sizeof(SURF_LAYOUT_REC_V1));
    PutSurfRecord := TRUE;
  except
    {if EWriteError}
    PutSurfRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.GetSurfRecord(var SurfRecord_v1 : SURF_LAYOUT_REC_V1) : boolean;{success/nosuccess}
begin
  try
    surfStream.ReadBuffer(SurfRecord_V1, sizeof(SURF_LAYOUT_REC_V1));

    With SurfRecord_V1 do
      ProbeWaveFormLength[probe] := nchans * pts_per_chan;
    GetSurfRecord := TRUE;
  except
    {if EReadError}
    GetSurfRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.PutMessageRecord(var msg : SURF_MSG_REC) : boolean;{success/nosuccess}
begin
  try
    SurfStream.WriteBuffer(msg, sizeof(SURF_MSG_REC));
    PutMessageRecord := TRUE;
  except
    PutMessageRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.GetMessageRecord(var msg : SURF_MSG_REC) : boolean;{success/nosuccess}
begin
  try
    surfStream.ReadBuffer(msg, sizeof(SURF_MSG_REC));
    GetMessageRecord := TRUE;
  except
    {if EReadError}
    GetMessageRecord := FALSE;
  end;
end;

{=========================================================================}
Function TSurfFile.ReadEntireSurfFile(Filename : wideString; ReadSpikeWaveForms,AverageWaveforms : boolean) : boolean;{success/nosuccess}
const NSPKALLOC = 200;//memory is allocated for this many spikes at a time
      NCRALLOC  = 100;//memory is allocated for this many crs at a time
      NEVENTSALLOC = 600;//memory is allocated for this many events at a time
      NVALALLOC = 600;//memory is allocated for this many single values at a time
      NMSGALLOC = 5;//memory is allocated for this many messages at a time
var rt : flagtype;
  ufftype,subtype : char;
  w,i,pr,e,p,s,s2,c,c2,nval,nmsg,begintime,endtime,wavetime,y,pt,curcr,nposprb,tm : integer;
  pk,vl,maxpk,maxpkindex,pkindex,avept : integer;
  HaltRead : boolean;
  PTRecord : SURF_PT_REC;
  SVRecord : SURF_SV_REC;
  MSGRecord : SURF_MSG_REC;
  f1,f2,f3,f4 : double;
  finished : boolean;
  PositionProbes : array[1..12] of integer;

Procedure FillInProbe(var prb : TProbe; sr : SURF_LAYOUT_REC_V1);
var c : integer;
begin
  Prb.ProbeSubType    := sr.ProbeSubType;
  Prb.numchans        := sr.nchans;
  Prb.pts_per_chan    := sr.pts_per_chan;
  Prb.trigpt          := sr.trigpt;
  Prb.lockout         := sr.lockout;
  Prb.intgain         := sr.intgain;
  Prb.threshold       := sr.threshold;
  Prb.skippts         := sr.skippts;
  Prb.sampfreqperchan := sr.sampfreqperchan;
  Prb.probe_descrip   := sr.probe_descrip;
  Prb.numspikes       := 0;
  Prb.numcr           := 0;
  Prb.Spike           := nil;
  Prb.CR              := nil;
  Prb.numparams       := 0;
  Prb.paramname       := nil;
  For c:= 0 to SURF_MAX_CHANNELS-1 do
  begin
    Prb.chanlist[c]     := sr.chanlist[c];
    Prb.screenlayout[c] := sr.screenlayout[c];
    Prb.extgain[c]      := sr.extgain[c];
  end;
end;
begin
  //open the file, read the header and the drdbs
  if not OpenSurfFileForRead(FileName) then //this reads the header and drdbs
  begin
    ReadEntireSurfFile := FALSE;
    Exit;
  end;

  //initialize the probe and single value arrays
  Prb := nil;
  SVal := nil;
  SurfEvent := nil;

  nval := 0;
  nmsg := 0;
  NEvents := 0;

  HaltRead := FALSE;

  FileProgressWin.FileProgress.MinValue := 0;
  FileProgressWin.FileProgress.MaxValue := FileSize-1;
  FileProgressWin.Show;
  FileProgressWin.BringToFront;
  FileProgressWin.StatusBar.SimpleText := 'Reading '+Filename;

  //read the file
  nposprb := 0;
  //tmptime := 0;
  While GetNextFlag(rt) do
  begin
    ufftype := rt[0];
    subtype := rt[1];
    //if Display.checked and visible then StatusBar.SimpleText := ufftype+' '+subtype;
    case ufftype of
      SURF_PL_REC_UFFTYPE {'L'}: //main probe layout record
        if GetSurfRecord(SurfRecord_v1) then with SurfRecord_v1 do
        begin
          case ProbeSubType of
            'S' :  begin //this is the probe layout record for spikes
                     if (probe+1) > Length(Prb) then SetLength(Prb,probe+1);
                     FillInProbe(Prb[probe],SurfRecord_v1);
                     Prb[probe].numparams := nchans*2;{peak and valley for each channel}
                     SetLength(Prb[probe].paramname,Prb[probe].numparams);
                     For c := 0 to nchans-1 do
                     begin
                       Prb[probe].paramname[c*2]   := 'Peak Ch '+Inttostr(c);
                       Prb[probe].paramname[c*2+1] := 'Valley Ch '+Inttostr(c);
                     end;
                   end;
            'C' :  begin //this is the probe layout record for continuous records
                     if (probe+1) > Length(Prb) then SetLength(Prb,probe+1);
                     FillInProbe(Prb[probe],SurfRecord_v1);
                     if Prb[probe].chanlist[0] in[26..31] then
                     begin
                       inc(nposprb);
                       PositionProbes[nposprb] := probe;
                     end;
                   end;
            else ShowMEssage('Unknown probe subtype: '+ProbeSubType);
          end; {case}
        end
        else begin
          ShowMessage('Error reading Spike Layout Record');
          HaltRead := TRUE;
        end;
      SURF_PT_REC_UFFTYPE {'N'}: //handle spikes and continuous records
        case subtype of
          SPIKETYPE  {'S'}: begin //spike record found
                              if not GetPolytrodeRecord(PTRecord) then
                              begin
                                HaltRead := TRUE;
                                break;
                              end;
                              With Prb[PTRecord.probe] do
                              begin
                                s := numspikes;
                                inc(numspikes);
                                if numspikes > Length(spike) then
                                begin
                                  SetLength(spike,s+NSPKALLOC);
                                  if ReadSpikeWaveForms then
                                    For s2 := s to s+NSPKALLOC-1 do
                                    begin
                                      SetLength(spike[s2].param,numparams);
                                      SetLength(spike[s2].waveform,numchans);
                                      for c := 0 to numchans-1 do
                                        SetLength(spike[s2].waveform[c],pts_per_chan);
                                    end;
                                end;
                                spike[s].time_stamp := PTRecord.time_stamp;
                                spike[s].cluster := PTRecord.cluster;
                                spike[s].EventNum := NEvents;

                                //demultiplex the waveform and get peaks and valley params
                                if ReadSpikeWaveForms then
                                begin
                                  //copy waveforms
                                  for c := 0 to numchans-1 do
                                    for p := 0 to pts_per_chan-1 do
                                      spike[s].waveform[c,p] := PTRecord.adc_waveform[p*numchans+c];

                                  if AverageWaveforms then
                                    for p := 0 to pts_per_chan-1 do
                                    begin
                                      avept := 0;
                                      for c := 0 to numchans-1 do
                                        avept := avept + spike[s].waveform[c,p];
                                      avept := round(avept / numchans) - 2047;
                                      for c := 0 to numchans-1 do
                                      begin
                                        spike[s].waveform[c,p] := spike[s].waveform[c,p] - avept;
                                        if spike[s].waveform[c,p] < 0 then spike[s].waveform[c,p] := 0;
                                        if spike[s].waveform[c,p] > 4095 then spike[s].waveform[c,p] := 4095;
                                      end;
                                    end;

                                  maxpk := 0;
                                  maxpkindex := 0;
                                  pkindex := 0;
                                  for c := 0 to numchans-1 do
                                  begin
                                    pk := 0;{peak chan c}
                                    vl := 4096;{valley chan c}
                                    for p := 0 to pts_per_chan-1 do
                                    begin
                                      if pk < spike[s].waveform[c,p] then
                                      begin
                                        pk := spike[s].waveform[c,p];
                                        pkindex := p;
                                      end;
                                      if vl > spike[s].waveform[c,p] then vl := spike[s].waveform[c,p];
                                    end;
                                    spike[s].param[c*2+1] := vl-2047;{valley chan c}
                                    if maxpk < pk then
                                    begin
                                      maxpk := pk;
                                      maxpkindex := pkindex;
                                    end;
                                  end;
                                  for c := 0 to numchans-1 do
                                    spike[s].param[c*2] := spike[s].waveform[c,maxpkindex]-2047;{peak chan c}
                                end;
                              end;
                              if NEvents+1 > Length(SurfEvent) then SetLength(SurfEvent,NEvents+NEVENTSALLOC);
                              SurfEvent[NEvents].Time_Stamp := PTRecord.time_stamp;
                              SurfEvent[NEvents].EventType :=  SURF_PT_REC_UFFTYPE;
                              SurfEvent[NEvents].SubType := SPIKETYPE;
                              SurfEvent[NEvents].Probe := PTRecord.probe;
                              SurfEvent[NEvents].Index := Prb[PTRecord.probe].numspikes-1;
                              inc(NEvents);
                            end;
          CONTINUOUSTYPE {'C'}: begin //continuous record found
                              if not GetPolytrodeRecord(PTRecord) then
                              begin
                                HaltRead := TRUE;
                                break;
                              end;
                              With Prb[PTRecord.probe] do
                              begin
                                c := numcr;
                                inc(numcr);
                                if numcr > Length(cr) then
                                begin
                                  SetLength(cr,c+NCRALLOC);
                                  For c2 := c to c+NCRALLOC-1 do
                                    SetLength(cr[c2].waveform,pts_per_chan);
                                end;
                                cr[c].time_stamp := PTRecord.time_stamp;
                                cr[c].EventNum := NEvents;
                                //might be able to do the following with a move command
                                for p := 0 to pts_per_chan-1 do
                                begin
                                  cr[c].waveform[p] := PTRecord.adc_waveform[p];
                                  if cr[c].waveform[p] < 0 then cr[c].waveform[p] := 0;
                                  if cr[c].waveform[p] > 4095 then cr[c].waveform[p] := 4095;
                                end;
                              end;
                              if NEvents+1 > Length(SurfEvent) then SetLength(SurfEvent,NEvents+NEVENTSALLOC);
                              SurfEvent[NEvents].Time_Stamp := PTRecord.time_stamp;
                              SurfEvent[NEvents].EventType :=  SURF_PT_REC_UFFTYPE;
                              SurfEvent[NEvents].SubType := CONTINUOUSTYPE;
                              SurfEvent[NEvents].Probe := PTRecord.probe;
                              SurfEvent[NEvents].Index := Prb[PTRecord.probe].numcr-1;
                              inc(NEvents);
                            end;
          else break;
        end;
      SURF_SV_REC_UFFTYPE {'V'}: //handle single values (including digital signals)
        case subtype of
          SURF_DIGITAL{'D'}:begin
                              if not GetSingleValueRecord(SVRecord) then
                              begin
                                HaltRead := TRUE;
                                break;
                              end;
                              if nval+1 > Length(SVal) then SetLength(SVal,nval+NVALALLOC);
                              SVal[nval].time_stamp := SVRecord.time_stamp;
                              SVal[nval].subtype    := SVRecord.subtype;
                              SVal[nval].sval       := SVRecord.sval;
                              SVal[nval].EventNum   := NEvents;
                              inc(nval);

                              if NEvents+1 > Length(SurfEvent) then SetLength(SurfEvent,NEvents+NEVENTSALLOC);
                              SurfEvent[NEvents].Time_Stamp := SVRecord.time_stamp;
                              SurfEvent[NEvents].EventType :=  SURF_SV_REC_UFFTYPE;
                              SurfEvent[NEvents].SubType := SURF_DIGITAL;
                              SurfEvent[NEvents].Probe := -1;
                              SurfEvent[NEvents].Index := nval-1;
                              inc(NEvents);
                            end;
          'T' :             begin
                              if not GetSingleValueRecord(SVRecord) then
                              begin
                                HaltRead := TRUE;
                                break;
                              end;
                              //tmptime := SVRecord.time_stamp;
                            end;
          else break;
        end;
       SURF_MSG_REC_UFFTYPE {'M'}:
                            begin//handle surf messages
                              if not GetMessageRecord(MSGRecord) then
                              begin
                                HaltRead := TRUE;
                                break;
                              end;
                              if nmsg+1 > Length(Msg) then SetLength(Msg,nmsg+NMSGALLOC);
                              Msg[nmsg].time_stamp := MSGRecord.time_stamp;
                              Msg[nmsg].msg        := MSGRecord.msg;
                              Msg[nmsg].EventNum   := NEvents;
                              inc(nmsg);

                              if NEvents+1 > Length(SurfEvent) then SetLength(SurfEvent,NEvents+NEVENTSALLOC);
                              SurfEvent[NEvents].Time_Stamp := MSGRecord.time_stamp;
                              SurfEvent[NEvents].EventType :=  SURF_MSG_REC_UFFTYPE;
                              SurfEvent[NEvents].SubType := '-';
                              SurfEvent[NEvents].Probe := -1;
                              SurfEvent[NEvents].Index := nmsg-1;
                              //InitEventPosition(NEvents);
                              inc(NEvents);
                            end;
    end {main case};
    Application.ProcessMessages;
    if HaltRead then break;
    FileProgressWin.FileProgress.Progress := FilePosition;
  end {main loop};

  For p := 0 to Length(prb)-1 do
  with prb[p] do
  begin
    if numspikes < Length(spike) then
    begin
      if ReadSpikeWaveForms then
        For s := Length(spike)-1 downto numspikes do
        begin
          spike[s].param := nil;
          for c := 0 to numchans-1 do
           spike[s].waveform[c] := nil;
          spike[s].waveform := nil;
        end;
      SetLength(spike,numspikes);
    end;
    if numcr < Length(cr) then
    begin
      For c := Length(cr)-1 downto numcr do
        cr[c].waveform := nil;
      SetLength(cr,numcr);
    end;
  end;

  SetLength(SVal,nval);
  SetLength(Msg,nmsg);
  SetLength(SurfEvent,NEvents);

  CloseFile;
  ReadEntireSurfFile := not HaltRead;
  FileProgressWin.Hide;
end;

END.
