unit Exec;

interface

{==============================================================================}
function NewExec(CommandLine: String{TCommandLine}): Boolean;
{==============================================================================}

implementation

uses ShellAPI, Windows, Messages, SysUtils, Classes, Graphics, Controls,
  Forms, Dialogs, StdCtrls, Buttons;

const
  EXEExtension = '.EXE';

{------------------------------------------------------------------------------}
function IsExecutableFile(Value: String{TCommandLine}): Boolean;
{ This method returns whether or not the Value represents a valid
  executable file by ensuring that its file extension is 'EXE' }
var
  Ext: String[4];
begin
  Ext := ExtractFileExt(Value);
  Result := (UpperCase(Ext) = EXEExtension);
end;

{------------------------------------------------------------------------------}
function ProcessExecute(CommandLine: String{TCommandLine}; cshow : Word): Integer;
{ This method encapsulates the call to CreateProcess() which creates
  a new process and its primary thread. This is the method used in
  Win32 to execute another application, This method requires the use
  of the TStartInfo and TProcessInformation structures. These structures
  are not documented as part of the Delphi 4 online help but rather
  the Win32 help as STARTUPINFO and PROCESS_INFORMATION.

  The CommandLine paremeter specifies the pathname of the file to
  execute.

  The cShow paremeter specifies one of the SW_XXXX constants which
  specifies how to display the window. This value is assigned to the
  sShowWindow field of the TStartupInfo structure.
SW_HIDE	            Hides the window and activates another window.
SW_MAXIMIZE	    Maximizes the specified window.
SW_MINIMIZE	    Minimizes the specified window and activates the next top-level window in the Z order.
SW_RESTORE	    Activates and displays the window. If the window is minimized or maximized, Windows restores it to its original size and position. An application should specify this flag when restoring a minimized window.
SW_SHOW	            Activates the window and displays it in its current size and position.
SW_SHOWDEFAULT	    Sets the show state based on the SW_ flag specified in the STARTUPINFO structure passed to the CreateProcess function by the program that started the application.
SW_SHOWMAXIMIZED    Activates the window and displays it as a maximized window.
SW_SHOWMINIMIZED    Activates the window and displays it as a minimized window.
SW_SHOWMINNOACTIVE  Displays the window as a minimized window. The active window remains active.
SW_SHOWNA	    Displays the window in its current state. The active window remains active.
SW_SHOWNOACTIVATE   Displays a window in its most recent size and position. The active window remains active.
SW_SHOWNORMAL	    Activates and displays a window. If the window is minimized or maximized, Windows restores it to its original size and position. An application should specify this flag when displaying the window for the first time.
  }
var
  Rslt: LongBool;
  StartUpInfo: TStartUpInfo;  // documented as STARTUPINFO
  ProcessInfo: TProcessInformation; // documented as PROCESS_INFORMATION
  ParamStr : string;
  tmp : string;
  i,j : integer;
  s : string[4];
begin
//strip filename from commandline, which  may contain parameters
//where do I put the parameters???
  //tmps := pchar(CommandLine);
  ParamStr := ExpandFileName(CommandLine);
  tmp := ParamStr;
  CommandLine  := '';

  j := 256;
  for i := 1 to length(tmp) do
    if i < j then
    begin
      if i>length(tmp)-4 then break;
      s := tmp[i]+tmp[i+1]+tmp[i+2]+tmp[i+3];
      if s = '.exe' then j := i+4;
      CommandLine := CommandLine + tmp[i];
    end;

  if j>length(tmp)-4 then
    Raise Exception.Create(CommandLine+' contains no parameters.');

  CommandLine := CommandLine + #0;

  if not IsExecutableFile(CommandLine) then
    Raise Exception.Create(CommandLine+' is not an executable file.');
  if not FileExists(CommandLine) then
    Raise Exception.Create('The file: '+CommandLine+' cannot be found.');

  { Clear the StartupInfo structure }
  FillChar(StartupInfo, SizeOf(TStartupInfo), 0);
  { Initialize the StartupInfo structure with required data.
    Here, we assign the SW_XXXX constant to the wShowWindow field
    of StartupInfo. When specifying a value to this field the
    STARTF_USESSHOWWINDOW flag must be set in the dwFlags field.
    Additional information on the TStartupInfo is provided in the Win32
    online help under STARTUPINFO. }
  //s := 'Hi there Phil!';
  with StartupInfo do
  begin
    cb := SizeOf(TStartupInfo); // Specify size of structure
    dwFlags := STARTF_USESHOWWINDOW or STARTF_FORCEONFEEDBACK;
    wShowWindow := cShow;
  end;

  { Create the process by calling CreateProcess(). This function
    fills the ProcessInfo structure with information about the new
    process and its primary thread. Detailed information is provided
    in the Win32 online help for the TProcessInfo structure under
    PROCESS_INFORMATION. }

  Rslt := CreateProcess(PChar(CommandLine), PChar(ParamStr), nil, nil, FALSE,
    NORMAL_PRIORITY_CLASS, nil, nil, StartupInfo, ProcessInfo);
  { If Rslt is true, then the CreateProcess call was successful.
    Otherwise, GetLastError will return an error code representing the
    error which occurred. }
  if Rslt then
    with ProcessInfo do
    begin
      { Wait until the process is in idle. }
      WaitForInputIdle(hProcess, INFINITE);
      CloseHandle(hThread); // Free the hThread  handle
      CloseHandle(hProcess);// Free the hProcess handle
      Result := 0;          // Set Result to 0, meaning successful
    end
  else Result := GetLastError; // Set result to the error code.
end;


{------------------------------------------------------------------------------}
function NewExec(CommandLine: String): Boolean;
var
  WERetVal: Word;
begin
  WERetVal := ProcessExecute(CommandLine,SW_SHOWNORMAL);
  if WERetVal <> 0 then
  begin
    raise Exception.Create('Error executing program. Error Code:; '+
          IntToStr(WERetVal));
    RESULT := FALSE;
  end else RESULT := TRUE;
end;

end.
