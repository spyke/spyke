unit SurfAnal;
{ Handles the execution and communication with user ANALYSIS programs.
  Handles the following calls from Surf:

  CallUserApp(Filename : string);
}
interface

USES  Messages,Sysutils,Controls,Classes,Exec,Surf2SurfBridge,SurfFile,Dialogs,SurfTypes,PahUnit;

TYPE
  //SurfAnal needs a handle for communications, so make it descend from TWinControl
  TSurfAnal = class (TWinControl)
    private
      Procedure ReadDataFile(Filename : String);
      Procedure SaveDataFile;
    public
      //my message handler for messages to SURF
      procedure MesgFromSurfBridge( var msg : TMessage ); message WM_SURF_IN;
      //calls from SurfMain
      Procedure CallUserApp(Filename : string);
  end;

implementation

{------------------------------------------------------------------------------}
procedure TSurfAnal.MesgFromSurfBridge(var msg : TMessage );
var  Spike : TSpike;
     Sv : TSVal;
     D2A : TD2A;
     Probe : TProbe;
     FileName : String;
begin
  Case msg.WParam of
    SURF_IN_HANDLE     : SurfBridgeFormHandle := msg.lparam;
    SURF_IN_SV         : begin
                           GetSvFromSurfBridge(Sv,msg.lparam);
                         end;
    SURF_IN_D2A        : begin
                           GetD2AFromSurfBridge(D2A,msg.lparam);
                         end;
    SURF_IN_SPIKE      : begin
                           GetSpikeFromSurfBridge(Spike,msg.lparam);
                         end;
    SURF_IN_READFILE   : begin  //user wants a file by the name of...
                           GetFileNameFromSurfBridge(FileName,msg.lparam);
                           //do something with filename
                           ReadDataFile(Filename);
                         end;
    SURF_IN_SAVEFILE   : begin
                           SaveDataFile;
                         end;
  end{case};
end;

{------------------------------------------------------------------------------}
Procedure TSurfAnal.CallUserApp(Filename : string);
begin
  SurfBridgeFormHandle := -1;
  NewExec(Filename + ' SURFv1.0 '+inttostr(Handle));
end;

{------------------------------------------------------------------------------}
Procedure TSurfAnal.ReadDataFile(Filename : String);
//begin
var
  ReadSurf : TSurfFile;
  c,np,p,e,i,j : integer;
  w : WORD;
  SurfEventArray : TSurfEventArray;
begin
  ReadSurf := TSurfFile.Create;
  if not ReadSurf.ReadEntireSurfFile(FileName,TRUE{read the spike waveforms},FALSE{average the waveforms}) then //this reads everything
  begin
    ReadSurf.Free;
    ShowMessage('Error Reading '+ FileName);
    Exit;
  end;

  With ReadSurf do
  begin
    //must send it in pieces to surfbridge, who will reassemble it
    //send a message first that the stream will be coming
    StartFileSend;
    SendEventArrayToSurfBridge(SurfEvent);
    Delay(0,50);
    SendProbeArrayToSurfBridge(Prb);
    Delay(0,50);
    if Length(SVal)>0 then SendSValArrayToSurfBridge(SVal);
    Delay(0,50);
    if Length(Msg)>0 then SendMsgArrayToSurfBridge(Msg);
    Delay(0,50);
    EndFileSend;
  end;
  ReadSurf.CleanUp;
  ReadSurf.Free;
end;

{------------------------------------------------------------------------------}
Procedure TSurfAnal.SaveDataFile;
begin
  //don't know how to handle this yet
  //possibly read all into an event rec and dump messages to user?
  //can't dump entire event rec because surfbridge buffer is too small
  //and can't increase it because some files may be over 100MB.
end;

end.
